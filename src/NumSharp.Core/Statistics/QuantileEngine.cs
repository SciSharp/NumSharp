using System;
using System.Buffers;
using System.Collections.Generic;
using System.Linq;
using NumSharp.Backends;
using NumSharp.Backends.Kernels;

namespace NumSharp.Statistics
{
    /// <summary>
    ///     13 NumPy quantile estimation methods (Hyndman &amp; Fan 1996 + NumPy-only variations).
    ///     Mirrors numpy.lib._function_base_impl._QuantileMethods.
    /// </summary>
    public enum QuantileMethod
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
    ///     Orchestrator for <c>np.median</c>, <c>np.percentile</c>, <c>np.quantile</c>.
    ///     Responsibilities split:
    ///       <list type="bullet">
    ///         <item>This file: input validation, axis normalization, dtype-promotion rule,
    ///               output-shape computation, keepdims/out plumbing, scratch rental.</item>
    ///         <item><see cref="ILKernelGenerator.Quantile"/>: per-dtype IL-emitted kernel
    ///               that owns the outer row-loop and dispatches into a JIT-specialized
    ///               generic row processor. No per-dtype switch executes per call after
    ///               the first cache miss.</item>
    ///       </list>
    /// </summary>
    internal static class QuantileEngine
    {
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

        public static NDArray Compute(
            NDArray a, double[] q, int[] axisArr, NDArray @out,
            bool overwrite_input, QuantileMethod method, bool keepdims, bool qIsScalar)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            if (q is null) throw new ArgumentNullException(nameof(q));
            if (a.typecode == NPTypeCode.Complex)
                throw new ArgumentException("a must be an array of real numbers");

            NPTypeCode outTypeCode = ResolveOutputDtype(a.typecode, method);

            // ── Stage the data so the reduction axis is innermost (stride-1 rows). ──
            // The IL kernel below walks rows by adding (n * sizeof(T)) per iteration.
            NDArray staged;
            int n;
            long outerSize;

            if (axisArr == null || axisArr.Length == 0)
            {
                if (a.ndim == 1 && a.Shape.IsContiguous)        staged = a;
                else if (a.Shape.IsContiguous)                  staged = a.reshape((long)a.size);
                else                                            staged = a.flatten();
                n = (int)staged.size;
                outerSize = 1;
            }
            else
            {
                int[] normalized = NormalizeAxes(axisArr, a.ndim);

                if (normalized.Length == 1)
                {
                    int axis = normalized[0];
                    int nd = a.ndim;
                    if (nd == 1 || axis == nd - 1)
                        staged = a.Shape.IsContiguous ? a : a.copy();
                    else
                        staged = np.moveaxis(a, axis, nd - 1).copy();
                    n = (int)staged.shape[staged.ndim - 1];
                    outerSize = 1;
                    for (int i = 0; i < staged.ndim - 1; i++) outerSize *= staged.shape[i];
                }
                else
                {
                    // Multi-axis: merge the reduction axes into one at the back.
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
                    n = (int)staged.shape[staged.ndim - 1];
                    outerSize = 1;
                    for (int i = 0; i < staged.ndim - 1; i++) outerSize *= staged.shape[i];
                }
            }

            // ── Output shape: q's shape (if non-scalar) prepended to the reduced shape. ──
            int qLen = q.Length;
            int outerDims = staged.ndim - 1;
            long[] outShape;
            if (axisArr == null || axisArr.Length == 0)
            {
                outShape = qIsScalar ? Array.Empty<long>() : new long[] { qLen };
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

            // ── Pre-compute the sorted list of buffer indices the partition needs to touch. ──
            // For continuous methods that's (floor, ceil) per q; discrete methods touch one
            // index per q. Deduplicating shrinks the number of QuickSelect passes when q
            // values share indices (e.g. q=[25,50,75] often share neighbouring picks on
            // small n).
            int[] kSortedManaged = BuildSortedTargetIndices(n, q, method);

            int srcSize = TypeSize(a.typecode);
            byte[] scratchBytes = ArrayPool<byte>.Shared.Rent(checked(srcSize * n));
            double[] qCopy = q;   // q is already a flat array we own; no need to copy

            try
            {
                unsafe
                {
                    fixed (byte* scratchPtr = scratchBytes)
                    fixed (int* kPtr = kSortedManaged)
                    fixed (double* qPtr = qCopy)
                    {
                        long dstOuterStride = qIsScalar ? 0L : outerSize;
                        ILKernelGenerator.Quantile(
                            srcType: a.typecode,
                            outType: outTypeCode,
                            method: method,
                            srcBase: staged.Address,
                            scratchBase: scratchPtr,
                            outer: outerSize,
                            n: n,
                            kSorted: kPtr,
                            nKs: kSortedManaged.Length,
                            q: qPtr,
                            nQs: qCopy.Length,
                            dstBase: resultRaw.Address,
                            dstOuterStride: dstOuterStride);
                    }
                }
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(scratchBytes);
            }

            // ── keepdims / out= plumbing ──
            NDArray result = ApplyKeepdims(resultRaw, a, axisArr, keepdims, qIsScalar);

            if (@out is not null)
            {
                if (!result.shape.SequenceEqual(@out.shape))
                    throw new ArgumentException(
                        $"Wrong shape of argument 'out', shape={string.Join(",", result.shape)} is required; " +
                        $"got shape={string.Join(",", @out.shape)}.");
                np.copyto(@out, result);
                return @out;
            }
            return result;
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
                    if (axisSet.Contains(i)) outShape[qDim + i] = 1;
                    else outShape[qDim + i] = result.shape[srcIdx++];
                }
            }
            return result.reshape(outShape);
        }

        /// <summary>
        ///     Builds the unique, ascending list of buffer indices the partition will touch.
        ///     Same formula as <see cref="ILKernelGenerator.ComputeIndex"/> — duplicated here so
        ///     the engine can hand the IL kernel a pre-sized int[] without circular calls.
        /// </summary>
        private static int[] BuildSortedTargetIndices(int n, double[] q, QuantileMethod method)
        {
            if (n <= 1) return Array.Empty<int>();
            var set = new HashSet<int>();
            for (int j = 0; j < q.Length; j++)
            {
                ILKernelGenerator.ComputeIndex(n, q[j], method, out int prev, out int next, out _);
                set.Add(prev);
                set.Add(next);
            }
            int[] arr = set.ToArray();
            Array.Sort(arr);
            return arr;
        }

        private static int TypeSize(NPTypeCode tc) => tc switch
        {
            NPTypeCode.Boolean => 1,
            NPTypeCode.Byte    => 1,
            NPTypeCode.SByte   => 1,
            NPTypeCode.Int16   => 2,
            NPTypeCode.UInt16  => 2,
            NPTypeCode.Half    => 2,
            NPTypeCode.Char    => 2,
            NPTypeCode.Int32   => 4,
            NPTypeCode.UInt32  => 4,
            NPTypeCode.Single  => 4,
            NPTypeCode.Int64   => 8,
            NPTypeCode.UInt64  => 8,
            NPTypeCode.Double  => 8,
            NPTypeCode.Decimal => 16,
            _                  => throw new NotSupportedException(tc.ToString()),
        };
    }
}
