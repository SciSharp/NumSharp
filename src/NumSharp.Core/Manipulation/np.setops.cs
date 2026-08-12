using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Rewrite the NaN lanes of a set-operation result to the exact bits NumPy produces, so
        ///     <see cref="union1d"/>/<see cref="setxor1d"/>/<see cref="setdiff1d"/> are bit-identical
        ///     to NumPy 2.4.2 even for NaN-bearing float input.
        /// </summary>
        /// <remarks>
        ///     NumPy's set operations route their surviving values through the SIMD quicksort/heapsort
        ///     that <c>np.unique</c>/<c>np.sort</c> use, and that sort CANONICALISES every NaN — signalling,
        ///     quiet, negative-signed, or payload-bearing — to the single positive quiet NaN, but ONLY for
        ///     float32/float64. float16 and complex128 NaNs are left exactly as they are (probed against
        ///     NumPy 2.4.2: f16 <c>0x7E00/0xFE00/0x7C01/…</c> and a complex negative-NaN real part
        ///     <c>0xFFF8…</c> all pass through unchanged). NumSharp's sort/unique do not canonicalise —
        ///     <c>np.sort</c> yields .NET's negative NaN (<c>0xFFF8…</c>) and <c>np.unique</c> preserves the
        ///     input payload — so a NaN that survives a set operation would diverge. This restores parity by
        ///     rewriting only the NaN lanes of a float32/float64 result to NumPy's canonical value.
        ///
        ///     <para><c>isin</c> returns bool (no NaN) and <c>intersect1d</c> never emits a NaN (NaN != NaN,
        ///     so a shared-value adjacency check never selects one), so neither needs this pass.</para>
        /// </remarks>
        private static NDArray CanonicalizeSetOpNaN(NDArray result)
        {
            // Only float32/float64 are canonicalised by NumPy's sort; f16/complex preserve, ints have no NaN.
            if (result.typecode != NPTypeCode.Single && result.typecode != NPTypeCode.Double)
                return result;
            if (result.size == 0)
                return result;

            NDArray mask = np.isnan(result);
            if (!np.any(mask))                              // common case: no NaN survived -> untouched
                return result;

            // The exact bits NumPy's float sort produces: 0x7FF8000000000000 (f64) / 0x7FC00000 (f32).
            NDArray canon = result.typecode == NPTypeCode.Double
                ? NDArray.Scalar(BitConverter.UInt64BitsToDouble(0x7FF8000000000000UL))
                : NDArray.Scalar(BitConverter.UInt32BitsToSingle(0x7FC00000U));
            return np.where(mask, canon, result);
        }
    }
}
