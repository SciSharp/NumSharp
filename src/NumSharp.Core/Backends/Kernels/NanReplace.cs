using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using NumSharp.Backends;

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    ///     Fused single-pass "replace every NaN with a scalar identity" copy used by
    ///     <see cref="np.nancumsum(NDArray, int?, NPTypeCode?, NDArray)"/> /
    ///     <see cref="np.nancumprod(NDArray, int?, NPTypeCode?, NDArray)"/> for the hot contiguous
    ///     float32/float64 path.
    ///
    ///     NumPy computes the replacement as <c>a2 = a.copy(); np.copyto(a2, val, where=np.isnan(a))</c>
    ///     — a boolean-mask temp plus two extra passes. The composed <c>np.where(np.isnan(a), val, a)</c>
    ///     is the same shape (isnan writes a bool array, where re-reads it): on a 10M float64 that is
    ///     <c>isnan</c> (~6.5 ms) + <c>where</c> (~18 ms) ≈ the cost of the cumsum scan itself. This fuses
    ///     both into ONE streaming pass with no bool-mask temp: read the operand once, write the cleaned
    ///     copy once. <c>Vector.Equals(v, v)</c> is all-ones for finite/inf lanes and zero for NaN lanes
    ///     (NaN ≠ NaN), so <c>ConditionalSelect(Equals(v, v), v, identity)</c> keeps every non-NaN value
    ///     (±inf included, matching <c>isnan</c>) and drops NaN to the identity, branchlessly, 4× unrolled.
    ///
    ///     Only true C-contiguous float32/float64 operands reach here (dispatched from
    ///     <c>np._replace_nan_for_scan</c>); Half, Complex and every non-contiguous view take the composed
    ///     <c>where(isnan(a), val, a)</c> path instead — bit-identical, fully gated, and not on the hot
    ///     path (Half/Complex are scalar-bound by their scan, F-order/strided are the uncommon layouts).
    /// </summary>
    internal static unsafe class NanReplace
    {
        /// <summary>
        ///     Returns a fresh C-contiguous copy of the contiguous float32/float64 array
        ///     <paramref name="a"/> with every NaN replaced by <paramref name="identity"/> (0 for the
        ///     cumsum scan, 1 for the cumprod scan). ±inf and every finite value pass through unchanged.
        /// </summary>
        internal static NDArray ReplaceContiguousFloat(NDArray a, int identity)
        {
            var shape = a.Shape;
            long n = a.size;
            // Fresh C-contiguous output; every element is written below, so skip the zero-fill.
            var ret = new NDArray(a.typecode, new Shape((long[])shape.dimensions.Clone()), false);
            byte* src = (byte*)a.Address + shape.offset * (long)a.dtypesize;

            if (a.typecode == NPTypeCode.Double)
                Fill<double>((double*)src, (double*)ret.Address, n, identity);
            else
                Fill<float>((float*)src, (float*)ret.Address, n, identity);
            return ret;
        }

        /// <summary>
        ///     NaN → identity fill, SIMD 4× unrolled + scalar tail. Generic over <typeparamref name="T"/>
        ///     so one body serves both float widths — the JIT bakes the vector width (V128/V256/V512) per
        ///     instantiation, exactly as <see cref="FiniteScan"/> does.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void Fill<T>(T* src, T* dst, long n, int identity) where T : unmanaged, INumber<T>
        {
            T id = T.CreateChecked(identity);
            long i = 0;

            if (Vector.IsHardwareAccelerated && Vector<T>.IsSupported)
            {
                int w = Vector<T>.Count;
                long step = (long)w * 4;
                if (n >= step)
                {
                    var idv = new Vector<T>(id);
                    long end = n - step;
                    for (; i <= end; i += step)
                    {
                        var v0 = Vector.Load(src + i);
                        var v1 = Vector.Load(src + i + w);
                        var v2 = Vector.Load(src + i + 2 * w);
                        var v3 = Vector.Load(src + i + 3 * w);
                        // Equals(v, v): all-ones for finite/inf lanes, zero for NaN lanes (NaN != NaN).
                        Vector.ConditionalSelect(Vector.Equals(v0, v0), v0, idv).Store(dst + i);
                        Vector.ConditionalSelect(Vector.Equals(v1, v1), v1, idv).Store(dst + i + w);
                        Vector.ConditionalSelect(Vector.Equals(v2, v2), v2, idv).Store(dst + i + 2 * w);
                        Vector.ConditionalSelect(Vector.Equals(v3, v3), v3, idv).Store(dst + i + 3 * w);
                    }
                }
            }

            for (; i < n; i++)
            {
                T x = src[i];
                dst[i] = x == x ? x : id; // x != x only for NaN
            }
        }
    }
}
