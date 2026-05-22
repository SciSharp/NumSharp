using System;
using System.Buffers;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp.Statistics
{
    /// <summary>
    ///     11 NumPy quantile estimation methods (Hyndman &amp; Fan 1996 + NumPy-only variations).
    ///     Mirrors numpy.lib._function_base_impl._QuantileMethods.
    /// </summary>
    internal enum QuantileMethod
    {
        Linear,                       // R-7, NumPy default — (n-1)*q
        Lower,                        // floor((n-1)*q), discrete
        Higher,                       // ceil((n-1)*q), discrete
        Nearest,                      // round((n-1)*q), discrete
        Midpoint,                     // 0.5*(floor+ceil), gamma=0 at integer indices
        InvertedCdf,                  // H&F R-1, discrete
        AveragedInvertedCdf,          // H&F R-2
        ClosestObservation,           // H&F R-3, discrete
        InterpolatedInvertedCdf,      // H&F R-4, α=0, β=1
        Hazen,                        // H&F R-5, α=β=0.5
        Weibull,                      // H&F R-6, α=β=0
        MedianUnbiased,               // H&F R-8, α=β=1/3
        NormalUnbiased,               // H&F R-9, α=β=3/8
    }

    /// <summary>
    ///     Sort-and-interpolate engine that backs <c>np.median</c>, <c>np.percentile</c> and <c>np.quantile</c>.
    ///     The public APIs route through <see cref="Compute"/> after validating their respective q-range.
    /// </summary>
    internal static class QuantileEngine
    {
        /// <summary>Discrete methods skip linear interpolation; integer inputs keep their dtype.</summary>
        internal static bool IsDiscrete(QuantileMethod m) =>
            m == QuantileMethod.Lower ||
            m == QuantileMethod.Higher ||
            m == QuantileMethod.Nearest ||
            m == QuantileMethod.InvertedCdf ||
            m == QuantileMethod.ClosestObservation;

        internal static QuantileMethod ParseMethod(string method)
        {
            if (method == null) throw new ArgumentNullException(nameof(method));
            switch (method)
            {
                case "linear":                    return QuantileMethod.Linear;
                case "lower":                     return QuantileMethod.Lower;
                case "higher":                    return QuantileMethod.Higher;
                case "nearest":                   return QuantileMethod.Nearest;
                case "midpoint":                  return QuantileMethod.Midpoint;
                case "inverted_cdf":              return QuantileMethod.InvertedCdf;
                case "averaged_inverted_cdf":     return QuantileMethod.AveragedInvertedCdf;
                case "closest_observation":      return QuantileMethod.ClosestObservation;
                case "interpolated_inverted_cdf": return QuantileMethod.InterpolatedInvertedCdf;
                case "hazen":                     return QuantileMethod.Hazen;
                case "weibull":                   return QuantileMethod.Weibull;
                case "median_unbiased":           return QuantileMethod.MedianUnbiased;
                case "normal_unbiased":           return QuantileMethod.NormalUnbiased;
                default:
                    throw new ArgumentException(
                        $"'{method}' is not a valid method. Use one of: linear, lower, higher, nearest, " +
                        "midpoint, inverted_cdf, averaged_inverted_cdf, closest_observation, " +
                        "interpolated_inverted_cdf, hazen, weibull, median_unbiased, normal_unbiased");
            }
        }

        /// <summary>
        ///     Compute quantiles of <paramref name="a"/> along the requested axis (or flattened when <paramref name="axisArr"/> is null).
        ///     <paramref name="q"/> is the array of quantile fractions in [0,1] (percentile callers pass q/100).
        /// </summary>
        public static NDArray Compute(
            NDArray a, double[] q, int[] axisArr, NDArray @out,
            bool overwrite_input, QuantileMethod method, bool keepdims, bool qIsScalar)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            if (q is null) throw new ArgumentNullException(nameof(q));
            if (a.typecode == NPTypeCode.Complex)
                throw new ArgumentException("a must be an array of real numbers");

            // Resolve output dtype (NEP50-aligned):
            //   discrete methods on integer dtypes preserve the dtype;
            //   anything continuous on integer/bool dtypes promotes to float64;
            //   floating dtypes preserve themselves.
            NPTypeCode outTypeCode = ResolveOutputDtype(a.typecode, method);

            // Goal: end up with `staged`, a C-contiguous read-only view where the reduction
            // axis is innermost (stride-1 rows of length n). The inner loop later copies
            // each row into a managed scratch buffer and partitions in place there — we
            // never mutate `a`, so the only required materialization is the moveaxis +
            // C-contig conversion (when needed). The pre-2025 version of this code did an
            // extra full `a.copy()` per call, which doubled allocations.
            NDArray staged;
            int n;
            long outerSize;
            int axis;

            if (axisArr == null || axisArr.Length == 0)
            {
                // Flatten: 1-D contiguous input is already in its final shape; otherwise
                // build a C-contig copy. Either way we never call `a.flatten()` (which
                // would always allocate).
                if (a.ndim == 1 && a.Shape.IsContiguous)
                    staged = a;
                else
                    staged = a.Shape.IsContiguous
                        ? a.reshape((long)a.size)
                        : a.flatten();
                n = (int)staged.size;
                outerSize = 1;
                axis = 0;
            }
            else
            {
                int[] normalized = NormalizeAxes(axisArr, a.ndim);

                if (normalized.Length == 1)
                {
                    axis = normalized[0];
                    int nd = a.ndim;
                    if (nd == 1)
                    {
                        staged = a.Shape.IsContiguous ? a : a.copy();
                    }
                    else if (axis == nd - 1)
                    {
                        // Reduction axis already at innermost position.
                        staged = a.Shape.IsContiguous ? a : a.copy();
                    }
                    else
                    {
                        // Need to permute the axis to innermost. copy() materializes into
                        // a fresh C-contig buffer.
                        staged = np.moveaxis(a, axis, nd - 1).copy();
                    }
                    n = (int)staged.shape[staged.ndim - 1];
                    outerSize = 1;
                    for (int i = 0; i < staged.ndim - 1; i++) outerSize *= staged.shape[i];
                }
                else
                {
                    // Multi-axis: merge the reduction axes into one by moving them to the
                    // back and reshaping. Mirrors numpy._ureduce's reshape_arr.
                    int[] sortedAxis = (int[])normalized.Clone();
                    Array.Sort(sortedAxis);
                    int[] keep = Enumerable.Range(0, a.ndim).Where(i => Array.BinarySearch(sortedAxis, i) < 0).ToArray();
                    int[] dest = Enumerable.Range(0, keep.Length).ToArray();
                    NDArray moved = np.moveaxis(a, keep, dest);
                    long reduced = 1;
                    for (int i = 0; i < normalized.Length; i++) reduced *= a.shape[normalized[i]];
                    long[] newShape = new long[keep.Length + 1];
                    for (int i = 0; i < keep.Length; i++) newShape[i] = a.shape[keep[i]];
                    newShape[keep.Length] = reduced;
                    staged = moved.reshape(newShape).copy();
                    axis = keep.Length;
                    n = (int)staged.shape[staged.ndim - 1];
                    outerSize = 1;
                    for (int i = 0; i < staged.ndim - 1; i++) outerSize *= staged.shape[i];
                }
            }

            int ndStaged = staged.ndim;

            // Output shape: q is prepended unless q is scalar.
            long[] outShape;
            int qLen = q.Length;
            int outerDims = ndStaged - 1;
            if (axisArr == null || axisArr.Length == 0)
            {
                // axis=None → result is a scalar (or 1-D with shape=(qLen,) when q is array).
                outShape = qIsScalar ? new long[0] : new long[] { qLen };
            }
            else
            {
                long[] keepShape = new long[outerDims];
                for (int i = 0; i < outerDims; i++) keepShape[i] = staged.shape[i];
                if (qIsScalar)
                {
                    outShape = keepShape;
                }
                else
                {
                    outShape = new long[outerDims + 1];
                    outShape[0] = qLen;
                    for (int i = 0; i < outerDims; i++) outShape[i + 1] = keepShape[i];
                }
            }

            NDArray resultRaw = new NDArray(outTypeCode, qIsScalar && outerDims == 0 ? new Shape() : new Shape(outShape));

            // Dispatch by input dtype. Sort is dtype-specific and inherently non-SIMD; the
            // per-dtype switch mirrors the existing pattern used by NDArray.unique / sortedness.
            unsafe
            {
                void* src = staged.Address;
                void* dst = resultRaw.Address;
                switch (a.typecode)
                {
                    case NPTypeCode.Boolean: ComputeForType<byte>((byte*)src, n, outerSize, q, qIsScalar, dst, outTypeCode, method); break;
                    case NPTypeCode.Byte:    ComputeForType<byte>((byte*)src, n, outerSize, q, qIsScalar, dst, outTypeCode, method); break;
                    case NPTypeCode.SByte:   ComputeForType<sbyte>((sbyte*)src, n, outerSize, q, qIsScalar, dst, outTypeCode, method); break;
                    case NPTypeCode.Int16:   ComputeForType<short>((short*)src, n, outerSize, q, qIsScalar, dst, outTypeCode, method); break;
                    case NPTypeCode.UInt16:  ComputeForType<ushort>((ushort*)src, n, outerSize, q, qIsScalar, dst, outTypeCode, method); break;
                    case NPTypeCode.Int32:   ComputeForType<int>((int*)src, n, outerSize, q, qIsScalar, dst, outTypeCode, method); break;
                    case NPTypeCode.UInt32:  ComputeForType<uint>((uint*)src, n, outerSize, q, qIsScalar, dst, outTypeCode, method); break;
                    case NPTypeCode.Int64:   ComputeForType<long>((long*)src, n, outerSize, q, qIsScalar, dst, outTypeCode, method); break;
                    case NPTypeCode.UInt64:  ComputeForType<ulong>((ulong*)src, n, outerSize, q, qIsScalar, dst, outTypeCode, method); break;
                    case NPTypeCode.Char:    ComputeForType<char>((char*)src, n, outerSize, q, qIsScalar, dst, outTypeCode, method); break;
                    case NPTypeCode.Half:    ComputeForHalf((Half*)src, n, outerSize, q, qIsScalar, dst, outTypeCode, method); break;
                    case NPTypeCode.Single:  ComputeForType<float>((float*)src, n, outerSize, q, qIsScalar, dst, outTypeCode, method); break;
                    case NPTypeCode.Double:  ComputeForType<double>((double*)src, n, outerSize, q, qIsScalar, dst, outTypeCode, method); break;
                    case NPTypeCode.Decimal: ComputeForType<decimal>((decimal*)src, n, outerSize, q, qIsScalar, dst, outTypeCode, method); break;
                    default:
                        throw new NotSupportedException($"np.quantile does not support dtype {a.typecode}");
                }
            }

            // Restore axis shape for keepdims (insert size-1 dims at reduced positions).
            NDArray result = ApplyKeepdims(resultRaw, a, axisArr, keepdims, qIsScalar);

            if (@out is not null)
            {
                if (!result.shape.SequenceEqual(@out.shape))
                    throw new ArgumentException(
                        $"Wrong shape of argument 'out', shape={string.Join(",", result.shape)} is required; " +
                        $"got shape={string.Join(",", @out.shape)}.");
                // copyto handles dtype casting.
                np.copyto(@out, result);
                return @out;
            }
            return result;
        }

        // Per-T inner driver. Each outer-row owns n contiguous elements; we partition that
        // row (in a scratch copy rented from ArrayPool to avoid GC pressure when callers
        // invoke us in tight loops) around just the indices the requested quantiles need —
        // O(n + k·n) introselect rather than O(n log n) full sort.
        private static unsafe void ComputeForType<T>(
            T* src, int n, long outer, double[] q, bool qIsScalar,
            void* dst, NPTypeCode outTypeCode, QuantileMethod method)
            where T : unmanaged, IComparable<T>
        {
            int qLen = q.Length;
            T[] scratch = ArrayPool<T>.Shared.Rent(n);
            int[] kSorted = BuildSortedTargetIndices(n, q, method);
            bool isFloat = typeof(T) == typeof(float) || typeof(T) == typeof(double);

            try
            {
                fixed (T* scratchPtr = scratch)
                {
                    for (long i = 0; i < outer; i++)
                    {
                        T* row = src + i * n;
                        Buffer.MemoryCopy(row, scratchPtr, (long)n * sizeof(T), (long)n * sizeof(T));

                        // Float dtypes: pre-scan for NaN so we can short-circuit + use the
                        // NaN-aware comparator only when needed.
                        bool sliceHasNaN = isFloat && HasNaN(scratchPtr, n);
                        if (!sliceHasNaN)
                        {
                            if (isFloat) PartitionWithNaNAtEnd(scratchPtr, n, kSorted);
                            else QuickSelect.PartitionAt(scratchPtr, n, kSorted);
                        }

                        for (int j = 0; j < qLen; j++)
                        {
                            long outIndex = qIsScalar ? i : (long)j * outer + i;
                            WriteResult(scratchPtr, n, q[j], sliceHasNaN, method, dst, outIndex, outTypeCode);
                        }
                    }
                }
            }
            finally
            {
                ArrayPool<T>.Shared.Return(scratch);
            }
        }

        // Half cannot satisfy `IComparable<Half>` portably (older BCL gap), and has no
        // arithmetic operators — promote to float for partition + lerp, cast back at write.
        private static unsafe void ComputeForHalf(
            Half* src, int n, long outer, double[] q, bool qIsScalar,
            void* dst, NPTypeCode outTypeCode, QuantileMethod method)
        {
            int qLen = q.Length;
            float[] scratch = ArrayPool<float>.Shared.Rent(n);
            int[] kSorted = BuildSortedTargetIndices(n, q, method);

            try
            {
                fixed (float* scratchPtr = scratch)
                {
                    for (long i = 0; i < outer; i++)
                    {
                        Half* row = src + i * n;
                        for (int k = 0; k < n; k++) scratchPtr[k] = (float)row[k];
                        bool sliceHasNaN = HasNaN(scratchPtr, n);
                        if (!sliceHasNaN)
                            PartitionWithNaNAtEnd(scratchPtr, n, kSorted);

                        for (int j = 0; j < qLen; j++)
                        {
                            long outIndex = qIsScalar ? i : (long)j * outer + i;
                            WriteResult(scratchPtr, n, q[j], sliceHasNaN, method, dst, outIndex, outTypeCode);
                        }
                    }
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(scratch);
            }
        }

        // Compute the unique, sorted list of buffer indices that the quantile interpolation
        // will read from after the partition. Discrete methods touch one index per q;
        // continuous methods touch the floor + ceil (≤ 2 per q). Deduplicating cuts the
        // number of QuickSelect passes when q values share indices.
        private static int[] BuildSortedTargetIndices(int n, double[] q, QuantileMethod method)
        {
            if (n <= 1) return Array.Empty<int>();
            var set = new HashSet<int>();
            for (int j = 0; j < q.Length; j++)
            {
                var (_, prev, next, _) = ComputeIndex(n, q[j], method);
                set.Add(prev);
                set.Add(next);
            }
            int[] arr = set.ToArray();
            Array.Sort(arr);
            return arr;
        }

        // Float-only NaN prescan. The buffer is on the stack-allocated scratch row, so
        // a tight `IsNaN` loop is fast enough and matches NumPy's `arr[-1]==NaN`
        // post-sort check semantically.
        private static unsafe bool HasNaN<T>(T* ptr, int n) where T : unmanaged
        {
            if (typeof(T) == typeof(double))
            {
                var p = (double*)ptr;
                for (int i = 0; i < n; i++) if (double.IsNaN(p[i])) return true;
            }
            else if (typeof(T) == typeof(float))
            {
                var p = (float*)ptr;
                for (int i = 0; i < n; i++) if (float.IsNaN(p[i])) return true;
            }
            return false;
        }

        // Partition wrapper for float/double — caller has already verified no NaN, so a
        // straight `IComparable<T>` partition is correct.
        private static unsafe void PartitionWithNaNAtEnd<T>(T* ptr, int n, int[] sortedKs) where T : unmanaged, IComparable<T>
            => QuickSelect.PartitionAt(ptr, n, sortedKs);

        // ── per-method virtual-index computation ────────────────────────────────────

        /// <summary>
        ///     Computes virtual_index, prev/next sorted indices, and the linear-interpolation
        ///     weight γ for a single (n, q, method) triple. Mirrors NumPy's
        ///     <c>_compute_virtual_index + _get_gamma</c> table.
        /// </summary>
        private static (double virtualIndex, int prevIdx, int nextIdx, double gamma)
            ComputeIndex(int n, double q, QuantileMethod method)
        {
            double vi;
            int prev, next;
            double gamma;

            switch (method)
            {
                case QuantileMethod.Linear:
                    vi = (n - 1) * q;
                    prev = (int)Math.Floor(vi);
                    next = prev + 1;
                    gamma = vi - prev;
                    break;

                case QuantileMethod.Lower:
                    vi = Math.Floor((n - 1) * q);
                    prev = (int)vi;
                    next = prev;
                    gamma = 0;
                    break;

                case QuantileMethod.Higher:
                    vi = Math.Ceiling((n - 1) * q);
                    prev = (int)vi;
                    next = prev;
                    gamma = 0;
                    break;

                case QuantileMethod.Nearest:
                    // NumPy: np.around uses banker's rounding.
                    vi = Math.Round((n - 1) * q, MidpointRounding.ToEven);
                    prev = (int)vi;
                    next = prev;
                    gamma = 0;
                    break;

                case QuantileMethod.Midpoint:
                {
                    vi = (n - 1) * q;
                    double lo = Math.Floor(vi);
                    double hi = Math.Ceiling(vi);
                    prev = (int)lo;
                    next = (int)hi;
                    // fix_gamma: 0.5 default, 0 when index hits an integer.
                    gamma = (vi == lo) ? 0.0 : 0.5;
                    break;
                }

                case QuantileMethod.InvertedCdf:
                {
                    // (n*q) - 1 — virtual index; rounded UP unless gamma is exactly 0.
                    double vRaw = n * q - 1.0;
                    double frac = vRaw - Math.Floor(vRaw);
                    // gamma_fun: when frac == 0 → use floor, else → use ceil.
                    if (frac == 0)
                        vi = Math.Floor(vRaw);
                    else
                        vi = Math.Floor(vRaw) + 1.0;
                    if (vi < 0) vi = 0;
                    prev = (int)vi;
                    next = prev;
                    gamma = 0;
                    break;
                }

                case QuantileMethod.ClosestObservation:
                {
                    // _closest_observation: index = (n*q) - 1 - 0.5
                    //  gamma_fun: gamma == 0 AND floor(index) % 2 == 1 → use floor, else ceil.
                    double idx = n * q - 1.0 - 0.5;
                    double fl = Math.Floor(idx);
                    double frac = idx - fl;
                    if (frac == 0 && ((long)fl % 2 + 2) % 2 == 1)
                        vi = fl;
                    else
                        vi = fl + 1.0;
                    if (vi < 0) vi = 0;
                    prev = (int)vi;
                    next = prev;
                    gamma = 0;
                    break;
                }

                case QuantileMethod.AveragedInvertedCdf:
                {
                    vi = n * q - 1.0;
                    prev = (int)Math.Floor(vi);
                    next = prev + 1;
                    double g = vi - prev;
                    // fix_gamma: default 1.0, 0.5 when gamma == 0 (average prev with next).
                    gamma = (g == 0) ? 0.5 : 1.0;
                    break;
                }

                case QuantileMethod.InterpolatedInvertedCdf:
                    vi = ComputeVirtualIndexAB(n, q, 0.0, 1.0);
                    prev = (int)Math.Floor(vi);
                    next = prev + 1;
                    gamma = vi - prev;
                    break;

                case QuantileMethod.Hazen:
                    vi = ComputeVirtualIndexAB(n, q, 0.5, 0.5);
                    prev = (int)Math.Floor(vi);
                    next = prev + 1;
                    gamma = vi - prev;
                    break;

                case QuantileMethod.Weibull:
                    vi = ComputeVirtualIndexAB(n, q, 0.0, 0.0);
                    prev = (int)Math.Floor(vi);
                    next = prev + 1;
                    gamma = vi - prev;
                    break;

                case QuantileMethod.MedianUnbiased:
                    vi = ComputeVirtualIndexAB(n, q, 1.0 / 3.0, 1.0 / 3.0);
                    prev = (int)Math.Floor(vi);
                    next = prev + 1;
                    gamma = vi - prev;
                    break;

                case QuantileMethod.NormalUnbiased:
                    vi = ComputeVirtualIndexAB(n, q, 3.0 / 8.0, 3.0 / 8.0);
                    prev = (int)Math.Floor(vi);
                    next = prev + 1;
                    gamma = vi - prev;
                    break;

                default:
                    throw new ArgumentOutOfRangeException(nameof(method));
            }

            // Clamp into [0, n-1] like NumPy's _get_indexes.
            if (prev < 0) prev = 0;
            if (prev > n - 1) prev = n - 1;
            if (next < 0) next = 0;
            if (next > n - 1) next = n - 1;
            return (vi, prev, next, gamma);
        }

        private static double ComputeVirtualIndexAB(int n, double q, double alpha, double beta) =>
            n * q + (alpha + q * (1.0 - alpha - beta)) - 1.0;

        // ── per-output-cell write ───────────────────────────────────────────────────

        /// <summary>
        ///     Pull <c>previous</c> and <c>next</c> samples from <paramref name="sorted"/>, lerp, then
        ///     cast to the output dtype and write to <c>dst[outIndex]</c>.
        ///     NaN-tainted slices write NaN/0 per dtype (float dtypes preserve NaN,
        ///     int dtypes do not; matches NumPy's silent overflow).
        /// </summary>
        private static unsafe void WriteResult<T>(
            T* sorted, int n, double q, bool sliceHasNaN, QuantileMethod method,
            void* dst, long outIndex, NPTypeCode outTypeCode)
            where T : unmanaged
        {
            if (sliceHasNaN)
            {
                WriteNaN(dst, outIndex, outTypeCode);
                return;
            }

            var (_, prevIdx, nextIdx, gamma) = ComputeIndex(n, q, method);

            // Discrete methods: take a single sample, no arithmetic — preserve integer dtype.
            if (IsDiscrete(method))
            {
                T sample = sorted[prevIdx];
                WriteSample(sample, dst, outIndex, outTypeCode);
                return;
            }

            // Continuous methods. We use double internally for arithmetic except for
            // float-input dtypes, where we keep the input precision to match NumPy bit-for-bit.
            if (typeof(T) == typeof(decimal))
            {
                decimal prev = ((decimal*)sorted)[prevIdx];
                decimal next = ((decimal*)sorted)[nextIdx];
                decimal v = (gamma == 0) ? prev : prev + (next - prev) * (decimal)gamma;
                WriteDecimal(v, dst, outIndex, outTypeCode);
                return;
            }

            if (typeof(T) == typeof(float))
            {
                float prev = ((float*)sorted)[prevIdx];
                float next = ((float*)sorted)[nextIdx];
                float v = prev + (next - prev) * (float)gamma;
                WriteSingle(v, dst, outIndex, outTypeCode);
                return;
            }

            // double, int, bool, etc. — arithmetic in double.
            double prevD = ToDouble(sorted[prevIdx]);
            double nextD = ToDouble(sorted[nextIdx]);
            double res = prevD + (nextD - prevD) * gamma;
            WriteDouble(res, dst, outIndex, outTypeCode);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double ToDouble<T>(T value) where T : unmanaged
        {
            if (typeof(T) == typeof(byte))    return Unsafe.As<T, byte>(ref value);
            if (typeof(T) == typeof(sbyte))   return Unsafe.As<T, sbyte>(ref value);
            if (typeof(T) == typeof(short))   return Unsafe.As<T, short>(ref value);
            if (typeof(T) == typeof(ushort))  return Unsafe.As<T, ushort>(ref value);
            if (typeof(T) == typeof(int))     return Unsafe.As<T, int>(ref value);
            if (typeof(T) == typeof(uint))    return Unsafe.As<T, uint>(ref value);
            if (typeof(T) == typeof(long))    return Unsafe.As<T, long>(ref value);
            if (typeof(T) == typeof(ulong))   return Unsafe.As<T, ulong>(ref value);
            if (typeof(T) == typeof(char))    return Unsafe.As<T, char>(ref value);
            if (typeof(T) == typeof(double))  return Unsafe.As<T, double>(ref value);
            if (typeof(T) == typeof(float))   return Unsafe.As<T, float>(ref value);
            if (typeof(T) == typeof(decimal)) return (double)Unsafe.As<T, decimal>(ref value);
            throw new NotSupportedException(typeof(T).Name);
        }

        // ── output-dtype writers ────────────────────────────────────────────────────

        private static unsafe void WriteDouble(double v, void* dst, long i, NPTypeCode tc)
        {
            switch (tc)
            {
                case NPTypeCode.Double:  ((double*)dst)[i] = v; break;
                case NPTypeCode.Single:  ((float*)dst)[i] = (float)v; break;
                case NPTypeCode.Half:    ((Half*)dst)[i] = (Half)v; break;
                case NPTypeCode.Decimal: ((decimal*)dst)[i] = double.IsFinite(v) ? (decimal)v : 0m; break;
                case NPTypeCode.Int64:   ((long*)dst)[i] = (long)v; break;
                case NPTypeCode.UInt64:  ((ulong*)dst)[i] = (ulong)v; break;
                case NPTypeCode.Int32:   ((int*)dst)[i] = (int)v; break;
                case NPTypeCode.UInt32:  ((uint*)dst)[i] = (uint)v; break;
                case NPTypeCode.Int16:   ((short*)dst)[i] = (short)v; break;
                case NPTypeCode.UInt16:  ((ushort*)dst)[i] = (ushort)v; break;
                case NPTypeCode.Byte:    ((byte*)dst)[i] = (byte)v; break;
                case NPTypeCode.SByte:   ((sbyte*)dst)[i] = (sbyte)v; break;
                case NPTypeCode.Boolean: ((byte*)dst)[i] = v != 0 ? (byte)1 : (byte)0; break;
                case NPTypeCode.Char:    ((char*)dst)[i] = (char)(ushort)v; break;
                default: throw new NotSupportedException(tc.ToString());
            }
        }

        private static unsafe void WriteSingle(float v, void* dst, long i, NPTypeCode tc)
        {
            if (tc == NPTypeCode.Single) { ((float*)dst)[i] = v; return; }
            WriteDouble(v, dst, i, tc);
        }

        private static unsafe void WriteDecimal(decimal v, void* dst, long i, NPTypeCode tc)
        {
            if (tc == NPTypeCode.Decimal) { ((decimal*)dst)[i] = v; return; }
            WriteDouble((double)v, dst, i, tc);
        }

        private static unsafe void WriteSample<T>(T sample, void* dst, long i, NPTypeCode tc) where T : unmanaged
        {
            // Discrete-method sample write: if the input and output dtype match, blit directly.
            // Otherwise route through double (only happens when the caller forces a wider out=
            // dtype, since discrete methods don't promote).
            if (typeof(T) == typeof(byte)    && tc == NPTypeCode.Byte)    { ((byte*)dst)[i] = Unsafe.As<T, byte>(ref sample); return; }
            if (typeof(T) == typeof(sbyte)   && tc == NPTypeCode.SByte)   { ((sbyte*)dst)[i] = Unsafe.As<T, sbyte>(ref sample); return; }
            if (typeof(T) == typeof(short)   && tc == NPTypeCode.Int16)   { ((short*)dst)[i] = Unsafe.As<T, short>(ref sample); return; }
            if (typeof(T) == typeof(ushort)  && tc == NPTypeCode.UInt16)  { ((ushort*)dst)[i] = Unsafe.As<T, ushort>(ref sample); return; }
            if (typeof(T) == typeof(int)     && tc == NPTypeCode.Int32)   { ((int*)dst)[i] = Unsafe.As<T, int>(ref sample); return; }
            if (typeof(T) == typeof(uint)    && tc == NPTypeCode.UInt32)  { ((uint*)dst)[i] = Unsafe.As<T, uint>(ref sample); return; }
            if (typeof(T) == typeof(long)    && tc == NPTypeCode.Int64)   { ((long*)dst)[i] = Unsafe.As<T, long>(ref sample); return; }
            if (typeof(T) == typeof(ulong)   && tc == NPTypeCode.UInt64)  { ((ulong*)dst)[i] = Unsafe.As<T, ulong>(ref sample); return; }
            if (typeof(T) == typeof(char)    && tc == NPTypeCode.Char)    { ((char*)dst)[i] = Unsafe.As<T, char>(ref sample); return; }
            if (typeof(T) == typeof(float)   && tc == NPTypeCode.Single)  { ((float*)dst)[i] = Unsafe.As<T, float>(ref sample); return; }
            if (typeof(T) == typeof(double)  && tc == NPTypeCode.Double)  { ((double*)dst)[i] = Unsafe.As<T, double>(ref sample); return; }
            if (typeof(T) == typeof(decimal) && tc == NPTypeCode.Decimal) { ((decimal*)dst)[i] = Unsafe.As<T, decimal>(ref sample); return; }

            // Cross-dtype: convert through double.
            if (typeof(T) == typeof(decimal))
                WriteDouble((double)Unsafe.As<T, decimal>(ref sample), dst, i, tc);
            else
                WriteDouble(ToDouble(sample), dst, i, tc);
        }

        private static unsafe void WriteNaN(void* dst, long i, NPTypeCode tc)
        {
            switch (tc)
            {
                case NPTypeCode.Double:  ((double*)dst)[i] = double.NaN; break;
                case NPTypeCode.Single:  ((float*)dst)[i] = float.NaN; break;
                case NPTypeCode.Half:    ((Half*)dst)[i] = Half.NaN; break;
                // Integer/bool/char have no NaN — NumPy silently writes 0 here (cast of NaN to int).
                default: WriteDouble(0.0, dst, i, tc); break;
            }
        }

        // ── helpers ─────────────────────────────────────────────────────────────────

        /// <summary>Output-dtype rule: int/bool → float64 for continuous methods, preserved for discrete.</summary>
        internal static NPTypeCode ResolveOutputDtype(NPTypeCode inTc, QuantileMethod method)
        {
            bool discrete = IsDiscrete(method);
            switch (inTc)
            {
                case NPTypeCode.Boolean:
                case NPTypeCode.Byte:
                case NPTypeCode.SByte:
                case NPTypeCode.Int16:
                case NPTypeCode.UInt16:
                case NPTypeCode.Int32:
                case NPTypeCode.UInt32:
                case NPTypeCode.Int64:
                case NPTypeCode.UInt64:
                case NPTypeCode.Char:
                    return discrete ? inTc : NPTypeCode.Double;
                case NPTypeCode.Half:
                case NPTypeCode.Single:
                case NPTypeCode.Double:
                case NPTypeCode.Decimal:
                    return inTc;
                default:
                    throw new NotSupportedException(inTc.ToString());
            }
        }

        internal static int[] NormalizeAxes(int[] axes, int ndim)
        {
            if (axes.Length == 0) return Array.Empty<int>();
            var seen = new HashSet<int>();
            int[] outAx = new int[axes.Length];
            for (int i = 0; i < axes.Length; i++)
            {
                int ax = axes[i];
                if (ax < 0) ax += ndim;
                if (ax < 0 || ax >= ndim)
                    throw new AxisError(axes[i], ndim);
                if (!seen.Add(ax))
                    throw new ArgumentException($"repeated axis in axis argument");
                outAx[i] = ax;
            }
            return outAx;
        }

        internal static NDArray ApplyKeepdims(NDArray result, NDArray source, int[] axisArr, bool keepdims, bool qIsScalar)
        {
            if (!keepdims) return result;

            int qDim = qIsScalar ? 0 : 1;
            int srcNd = source.ndim;

            long[] outShape;
            if (axisArr == null || axisArr.Length == 0)
            {
                // All source dims become 1.
                outShape = new long[qDim + srcNd];
                if (!qIsScalar) outShape[0] = result.shape[0];
                for (int i = 0; i < srcNd; i++) outShape[qDim + i] = 1;
            }
            else
            {
                int[] norm = NormalizeAxes(axisArr, srcNd);
                var axisSet = new HashSet<int>(norm);
                outShape = new long[qDim + srcNd];
                if (!qIsScalar) outShape[0] = result.shape[0];

                int srcIdx = qDim;
                for (int i = 0; i < srcNd; i++)
                {
                    if (axisSet.Contains(i))
                    {
                        outShape[qDim + i] = 1;
                    }
                    else
                    {
                        outShape[qDim + i] = result.shape[srcIdx++];
                    }
                }
            }
            return result.reshape(outShape);
        }
    }
}
