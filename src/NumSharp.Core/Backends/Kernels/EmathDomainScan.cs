using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using NumSharp.Backends;

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    ///     Fused single-pass, early-exit domain scans used by the <see cref="EmathModule"/> functions to
    ///     decide whether a real array must be promoted to complex128 (NumPy's <c>_fix_real_lt_zero</c> /
    ///     <c>_fix_real_abs_gt_1</c>). NumPy computes these as <c>any(x &lt; 0)</c> / <c>any(abs(x) &gt; 1)</c>
    ///     — a full boolean (and, for the abs case, a full <c>abs</c>) temporary plus a reduce pass. This
    ///     class fuses the comparison and the reduction into ONE streaming pass with no intermediate
    ///     allocation and an early exit at the first triggering element.
    ///
    ///     <para>
    ///     This is a SPECIALIZED FAST PATH, not a general kernel: it covers only the hot
    ///     <b>contiguous float32/float64</b> case — the dtypes that actually flow into
    ///     <c>sqrt</c>/<c>log</c>/<c>arccos</c>/… in practice — and returns <c>null</c> for every other
    ///     dtype or layout so the caller falls back to the correct-by-construction composition
    ///     (<c>np.any(x &lt; 0)</c> / <c>np.any(np.abs(x) &gt; 1)</c>). Float IEEE semantics are clean here:
    ///     <c>-0.0 &lt; 0</c> is false, a NaN never satisfies either predicate (all NaN comparisons are
    ///     false), and <c>±inf</c> satisfies <c>|x| &gt; 1</c> — all matching NumPy's element-wise operators
    ///     bit-for-bit (probed against 2.4.2). Integer/bool/char/half/decimal and non-contiguous views take
    ///     the composition, which reuses the already-fuzzed <c>np.abs</c>/comparison kernels (and so
    ///     inherits, e.g., NumPy's <c>abs(int.MinValue)</c> overflow exactly).
    ///     </para>
    /// </summary>
    internal static unsafe class EmathDomainScan
    {
        /// <summary>
        ///     Fast path for "does any element satisfy <c>x &lt; 0</c>?" — the trigger for the sqrt/log
        ///     family's complex promotion.
        /// </summary>
        /// <param name="a">The array to scan.</param>
        /// <returns>
        ///     <c>true</c>/<c>false</c> when <paramref name="a"/> is a contiguous float32/float64 array
        ///     (the fast path applies); <c>null</c> when it is not, signalling the caller to use the
        ///     composition fallback. An empty array yields <c>false</c> (no element triggers).
        /// </returns>
        internal static bool? AnyNegative(NDArray a)
        {
            if (!(a.Shape.IsContiguous || a.Shape.IsFContiguous))
                return null; // strided/broadcast/transposed → composition (correct for any layout)
            byte* p = (byte*)a.Address + a.Shape.offset * (long)a.dtypesize;
            switch (a.typecode)
            {
                case NPTypeCode.Double: return AnyNegContig<double>((double*)p, a.size);
                case NPTypeCode.Single: return AnyNegContig<float>((float*)p, a.size);
                default: return null; // int/bool/char/half/decimal → composition
            }
        }

        /// <summary>
        ///     Fast path for "does any element satisfy <c>|x| &gt; 1</c>?" — the trigger for the
        ///     arccos/arcsin/arctanh family's complex promotion.
        /// </summary>
        /// <param name="a">The array to scan.</param>
        /// <returns>
        ///     <c>true</c>/<c>false</c> when <paramref name="a"/> is a contiguous float32/float64 array;
        ///     <c>null</c> when the composition fallback should be used. An empty array yields <c>false</c>.
        /// </returns>
        internal static bool? AnyAbsGreaterThanOne(NDArray a)
        {
            if (!(a.Shape.IsContiguous || a.Shape.IsFContiguous))
                return null;
            byte* p = (byte*)a.Address + a.Shape.offset * (long)a.dtypesize;
            switch (a.typecode)
            {
                case NPTypeCode.Double: return AnyAbsGt1Contig<double>((double*)p, a.size);
                case NPTypeCode.Single: return AnyAbsGt1Contig<float>((float*)p, a.size);
                default: return null;
            }
        }

        /// <summary>
        ///     Contiguous SIMD scan returning <c>true</c> at the first lane with <c>value &lt; 0</c>.
        ///     Generic over <typeparamref name="T"/> so one body serves float32 and float64 — the JIT bakes
        ///     the vector width (V128/V256/V512) per instantiation. A NaN lane is never <c>&lt; 0</c> and a
        ///     <c>-0.0</c> lane is never <c>&lt; 0</c>, matching NumPy's <c>x &lt; 0</c> exactly.
        /// </summary>
        /// <param name="src">Pointer to the first element of the contiguous run.</param>
        /// <param name="n">Number of elements.</param>
        /// <returns><c>true</c> iff some element is strictly negative.</returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static bool AnyNegContig<T>(T* src, long n) where T : unmanaged, INumber<T>
        {
            long i = 0;
            if (Vector.IsHardwareAccelerated && Vector<T>.IsSupported)
            {
                int w = Vector<T>.Count;
                var zero = Vector<T>.Zero;
                // 4x-unrolled body: OR the four comparison masks (reinterpreted to bytes so the test is on
                // raw mask bits, immune to the NaN/±0 hazards a float compare/min would carry) and branch
                // ONCE per 4 vectors. Non-triggering runs stream near memory bandwidth (one well-predicted
                // not-taken branch per 4·w elements); a trigger still exits within a single block.
                long step = 4L * w;
                for (long end = n - step; i <= end; i += step)
                {
                    var m0 = Vector.LessThan(Vector.Load(src + i), zero);
                    var m1 = Vector.LessThan(Vector.Load(src + i + w), zero);
                    var m2 = Vector.LessThan(Vector.Load(src + i + 2 * w), zero);
                    var m3 = Vector.LessThan(Vector.Load(src + i + 3 * w), zero);
                    var acc = (Vector.AsVectorByte(m0) | Vector.AsVectorByte(m1))
                            | (Vector.AsVectorByte(m2) | Vector.AsVectorByte(m3));
                    if (!Vector.EqualsAll(acc, Vector<byte>.Zero))
                        return true;
                }
                // One-vector remainder for the tail below 4·w.
                for (long end = n - w; i <= end; i += w)
                    if (Vector.LessThanAny(Vector.Load(src + i), zero))
                        return true;
            }
            for (; i < n; i++)
                if (src[i] < T.Zero)
                    return true;
            return false;
        }

        /// <summary>
        ///     Contiguous SIMD scan returning <c>true</c> at the first lane with <c>|value| &gt; 1</c>.
        ///     <see cref="Vector.Abs"/> on a float vector is IEEE magnitude, so a NaN lane gives
        ///     <c>|NaN| &gt; 1 == false</c> (NaN comparisons are false) while a <c>±inf</c> lane gives
        ///     <c>inf &gt; 1 == true</c> — both matching NumPy's <c>abs(x) &gt; 1</c>. The comparison is
        ///     strict, so <c>|x| == 1</c> does not trigger.
        /// </summary>
        /// <param name="src">Pointer to the first element of the contiguous run.</param>
        /// <param name="n">Number of elements.</param>
        /// <returns><c>true</c> iff some element has magnitude strictly greater than one.</returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static bool AnyAbsGt1Contig<T>(T* src, long n) where T : unmanaged, INumber<T>
        {
            long i = 0;
            if (Vector.IsHardwareAccelerated && Vector<T>.IsSupported)
            {
                int w = Vector<T>.Count;
                var one = Vector<T>.One;
                // 4x-unrolled, branch-once-per-4-vectors (see AnyNegContig). Vector.Abs is IEEE magnitude,
                // so a NaN lane never triggers (|NaN| > 1 is false) and a ±inf lane does (|inf| > 1).
                long step = 4L * w;
                for (long end = n - step; i <= end; i += step)
                {
                    var m0 = Vector.GreaterThan(Vector.Abs(Vector.Load(src + i)), one);
                    var m1 = Vector.GreaterThan(Vector.Abs(Vector.Load(src + i + w)), one);
                    var m2 = Vector.GreaterThan(Vector.Abs(Vector.Load(src + i + 2 * w)), one);
                    var m3 = Vector.GreaterThan(Vector.Abs(Vector.Load(src + i + 3 * w)), one);
                    var acc = (Vector.AsVectorByte(m0) | Vector.AsVectorByte(m1))
                            | (Vector.AsVectorByte(m2) | Vector.AsVectorByte(m3));
                    if (!Vector.EqualsAll(acc, Vector<byte>.Zero))
                        return true;
                }
                for (long end = n - w; i <= end; i += w)
                    if (Vector.GreaterThanAny(Vector.Abs(Vector.Load(src + i)), one))
                        return true;
            }
            for (; i < n; i++)
            {
                // T.Abs is IEEE for float/double; |NaN| is NaN (> 1 is false), |inf| is inf (> 1 is true).
                if (T.Abs(src[i]) > T.One)
                    return true;
            }
            return false;
        }
    }
}
