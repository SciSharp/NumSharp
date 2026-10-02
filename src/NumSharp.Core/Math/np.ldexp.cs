using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Compute <c>x1 * 2^x2</c> element-wise — the inverse of <see cref="frexp(NDArray)"/>. Given a
        ///     mantissa and an integer power of two it reconstructs the value (<c>ldexp(*frexp(x)) == x</c>).
        ///     Mirrors NumPy's ufunc <c>ldexp(x1, x2)</c>.
        /// </summary>
        /// <param name="x1">Mantissa/base values. Integer/bool inputs promote to a float per NumPy's width rule
        /// (bool/int8/uint8 → float16, int16/uint16/char → float32, int32+ → float64); Half/Single/Double are
        /// preserved and Decimal is a NumSharp extension. A complex mantissa is rejected.</param>
        /// <param name="x2">Integer exponents (a Python/C# int scalar converts implicitly). Accepted dtypes are
        /// bool and the integers that safely cast to int64 (int8..int64, uint8..uint32) plus char; <b>uint64,
        /// any float, and complex exponents are rejected</b> (NumPy's TypeError). The exponent does NOT widen
        /// <paramref name="x1"/>'s dtype — the result is purely <paramref name="x1"/>'s float tier. An exponent
        /// too large for the C-int range is clamped, so it overflows to ±inf / underflows to ±0 exactly as
        /// NumPy's int64 loop specifies. <paramref name="x1"/> and <paramref name="x2"/> broadcast to a common
        /// shape.</param>
        /// <param name="out">A location into which the result is stored (NumPy ufunc <c>out=</c>): must be
        /// same_kind-castable from the loop dtype (<paramref name="x1"/>'s float tier) and match the broadcast
        /// shape; the same instance is returned. A float32 result up-casts into a float64 <paramref name="out"/>.</param>
        /// <param name="where">Boolean mask (NumPy ufunc <c>where=</c>): only mask-true elements are written to
        /// <paramref name="out"/>; masked-off slots keep their prior contents. Must be boolean and broadcast to
        /// the result shape.</param>
        /// <returns><c>x1 * 2^x2</c> in <paramref name="x1"/>'s float tier, broadcast to the common shape (a
        /// scalar if both inputs are scalars). An F-contiguous input set yields an F-contiguous result.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.ldexp.html
        ///     <para>Bit-identical to win-amd64 NumPy 2.4.2 — <see cref="System.Math.ScaleB(double, int)"/> is
        ///     the C <c>ldexp</c>, including the IEEE special values (<c>ldexp(±inf, n) = ±inf</c>,
        ///     <c>ldexp(NaN, n) = NaN</c>, <c>ldexp(±0, n) = ±0</c>).</para>
        /// </remarks>
        [NDScoped]
        public static NDArray ldexp(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null)
            => x1.TensorEngine.Ldexp(x1, x2, @out, where);
    }
}
