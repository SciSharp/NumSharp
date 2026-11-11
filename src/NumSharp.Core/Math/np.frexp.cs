using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Decompose each element of <paramref name="x"/> into a normalized fraction and an integer power
        ///     of two, element-wise: returns <c>(mantissa, exponent)</c> such that
        ///     <c>x == mantissa * 2^exponent</c>, with the mantissa in the half-open interval [0.5, 1). This is
        ///     the two-output inverse of <see cref="ldexp(NDArray, NDArray, NDArray, NDArray, DType)"/>.
        ///     Mirrors NumPy's ufunc <c>frexp(x) -&gt; (mantissa, exponent)</c>.
        /// </summary>
        /// <param name="x">Input array. Integer/bool inputs promote to a float per NumPy's width rule
        /// (bool/int8/uint8 → float16, int16/uint16/char → float32, int32+ → float64); Half/Single/Double are
        /// preserved and Decimal is a NumSharp extension (computed through the double bridge). A complex input
        /// is rejected — frexp has no complex loop.</param>
        /// <param name="out">NumPy's <c>out=(o1, o2)</c>: the mantissa output (same_kind-castable from the mantissa loop
        /// dtype) and the exponent output (same_kind-castable from int32; int64 accepted), each of <paramref name="x"/>'s
        /// shape and returned as-is; either slot may be null (NumPy's <c>(None, e)</c>), <c>default</c> allocates both.</param>
        /// <param name="where">Boolean mask (NumPy ufunc <c>where=</c>): with outputs supplied, only mask-true elements
        /// are written and masked-off slots keep their prior contents.</param>
        /// <param name="dtype">NumPy's <c>dtype=</c>, which frexp accepts only as null: any dtype would apply to both
        /// outputs and no loop has a non-int32 exponent, so NumPy raises "No loop matching ..." (2.4.2).</param>
        /// <returns>
        ///     A tuple <c>(Mantissa, Exponent)</c>: the mantissa in the promoted float dtype, and the
        ///     <b>int32</b> exponent (NumPy's exponent is always int32, regardless of the mantissa width) — or the
        ///     supplied outputs. Both have the shape of <paramref name="x"/>;
        ///     an F-contiguous input yields F-contiguous outputs. Each is a 0-d scalar when <paramref name="x"/> is a scalar.
        /// </returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.frexp.html
        ///     <para>
        ///     Special values follow win-amd64 NumPy 2.4.2 exactly (its scalar C-runtime <c>npy_frexp</c>):
        ///     <c>frexp(±0) = (±0, 0)</c>, <c>frexp(±inf) = (±inf, -1)</c>, <c>frexp(NaN) = (NaN, -1)</c> — note
        ///     the exponent is <c>-1</c>, not 0, for ±inf/NaN — and a signalling NaN mantissa is quieted.
        ///     </para>
        ///     <para>NumPy's signature <c>frexp(x, /, out=(None, None), *, where=True, ..., dtype=None)</c>: the outputs are
        ///     ONE tuple parameter, as <see cref="divmod"/>'s (the former <c>out1</c> / <c>out2</c> parameters went on
        ///     2026-11-11). NumPy's single array as out (<c>'out' must be a tuple of arrays</c>) cannot be written in
        ///     C#.</para>
        /// </remarks>
        /// <exception cref="IncorrectTypeException">A complex input (no loop), or any <paramref name="dtype"/>.</exception>
        [NDScoped]
        public static (NDArray Mantissa, NDArray Exponent) frexp(NDArray x, (NDArray, NDArray) @out = default, NDArray where = null,
            DType dtype = null)
            // The engine lays fresh outputs out as NumPy's ufunc does (the K order of the input, parity QA-36: F stays
            // F, a permuted input keeps its memory order) and returns a supplied output itself (a caller's out array
            // owns its own layout).
            => x.TensorEngine.Frexp(x, @out.Item1, @out.Item2, where, dtype);
    }
}
