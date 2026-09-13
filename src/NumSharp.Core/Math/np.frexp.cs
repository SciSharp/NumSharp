using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Decompose each element of <paramref name="x"/> into a normalized fraction and an integer power
        ///     of two, element-wise: returns <c>(mantissa, exponent)</c> such that
        ///     <c>x == mantissa * 2^exponent</c>, with the mantissa in the half-open interval [0.5, 1). This is
        ///     the two-output inverse of <see cref="ldexp(NDArray, NDArray)"/>.
        ///     Mirrors NumPy's ufunc <c>frexp(x) -&gt; (mantissa, exponent)</c>.
        /// </summary>
        /// <param name="x">Input array. Integer/bool inputs promote to a float per NumPy's width rule
        /// (bool/int8/uint8 → float16, int16/uint16/char → float32, int32+ → float64); Half/Single/Double are
        /// preserved and Decimal is a NumSharp extension (computed through the double bridge). A complex input
        /// is rejected — frexp has no complex loop.</param>
        /// <param name="out1">Output location for the mantissa (NumPy ufunc out1 / <c>out=(out1,out2)</c>): must be
        /// same_kind-castable from the mantissa loop dtype and match <paramref name="x"/>'s shape; returned as-is.</param>
        /// <param name="out2">Output location for the exponent: must be same_kind-castable from int32 (int64 accepted)
        /// and match <paramref name="x"/>'s shape; returned as-is.</param>
        /// <param name="where">Boolean mask (NumPy ufunc <c>where=</c>): with <paramref name="out1"/>/<paramref name="out2"/>
        /// supplied, only mask-true elements are written and masked-off slots keep their prior contents.</param>
        /// <returns>
        ///     A tuple <c>(Mantissa, Exponent)</c>: the mantissa in the promoted float dtype, and the
        ///     <b>int32</b> exponent (NumPy's exponent is always int32, regardless of the mantissa width) — or the
        ///     supplied <paramref name="out1"/>/<paramref name="out2"/>. Both have the shape of <paramref name="x"/>;
        ///     an F-contiguous input yields F-contiguous outputs. Each is a 0-d scalar when <paramref name="x"/> is a scalar.
        /// </returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.frexp.html
        ///     <para>
        ///     Special values follow win-amd64 NumPy 2.4.2 exactly (its scalar C-runtime <c>npy_frexp</c>):
        ///     <c>frexp(±0) = (±0, 0)</c>, <c>frexp(±inf) = (±inf, -1)</c>, <c>frexp(NaN) = (NaN, -1)</c> — note
        ///     the exponent is <c>-1</c>, not 0, for ±inf/NaN — and a signalling NaN mantissa is quieted.
        ///     </para>
        /// </remarks>
        [NDScoped]
        public static (NDArray Mantissa, NDArray Exponent) frexp(NDArray x, NDArray out1 = null, NDArray out2 = null, NDArray where = null)
            // The engine returns C-contiguous outputs (or the supplied out1/out2); PreserveFContig (shared with
            // np.modf) relabels a fresh result column-major when the input was F-contiguous — but ONLY when no out
            // was supplied (a caller's out array owns its own layout), matching NumPy's layout propagation.
            => (out1 is null && out2 is null)
                ? PreserveFContig(x, x.TensorEngine.Frexp(x, null, null, where))
                : x.TensorEngine.Frexp(x, out1, out2, where);
    }
}
