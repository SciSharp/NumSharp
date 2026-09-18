namespace NumSharp
{
    public partial class np
    {
        /// <summary>
        /// Shift the bits of an integer to the left.
        /// Mirrors NumPy's ufunc signature: <c>left_shift(x1, x2, /, out=None, *, where=True, dtype=None)</c>.
        /// </summary>
        /// <param name="x1">Value operand (integer/bool/char only; bool promotes to int8).</param>
        /// <param name="x2">Shift-count operand (integer/bool/char only).</param>
        /// <param name="out">A location into which the result is stored; must be same_kind-castable from
        /// the promoted loop dtype (int32→int16/float64 work; int32→uint8/bool raise). Returned as-is.</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written; masked-off out
        /// slots keep prior contents.</param>
        /// <param name="dtype">Optional loop dtype override; must be an integer/char loop — a
        /// float/complex/decimal request raises NumPy's no-loop TypeError.</param>
        /// <returns>Array with bits shifted left (or <paramref name="out"/> when provided).</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.left_shift.html
        ///
        /// Bits are shifted to the left by appending x2 0s at the right of x1.
        /// Since the internal representation of numbers is in binary format,
        /// this operation is equivalent to multiplying x1 by 2**x2.
        ///
        /// Example:
        ///   np.left_shift(5, 2) = 20  # 0b101 -> 0b10100
        /// </remarks>
        public static NDArray left_shift(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, DType dtype = null)
            => x1.TensorEngine.LeftShift(x1, x2, dtype, @out, where);

        /// <summary>
        /// Shift the bits of an integer to the left by a scalar or array-like amount.
        /// </summary>
        /// <param name="x1">Value operand (integer/bool/char only).</param>
        /// <param name="x2">Shift-count (scalar or array-like).</param>
        /// <param name="out">Optional provided output (see the array overload).</param>
        /// <param name="where">Optional bool write mask.</param>
        /// <param name="dtype">Optional integer/char loop dtype override.</param>
        /// <returns>Array with bits shifted left (or <paramref name="out"/> when provided).</returns>
        // Scope: reclaims the np.asanyarray(x2) temp for a scalar/array-like shift count
        // (the a<<obj operator analog); an NDArray-input passthrough and a provided out stay untracked.
        [NDScoped]
        public static NDArray left_shift(NDArray x1, object x2, NDArray @out = null, NDArray where = null, DType dtype = null)
            => x1.TensorEngine.LeftShift(x1, np.asanyarray(x2), dtype, @out, where);
    }
}
