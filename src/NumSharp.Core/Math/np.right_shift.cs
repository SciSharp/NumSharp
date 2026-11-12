namespace NumSharp
{
    public partial class np
    {
        /// <summary>
        /// Shift the bits of an integer to the right.
        /// Mirrors NumPy's ufunc signature: <c>right_shift(x1, x2, /, out=None, *, where=True, dtype=None)</c>.
        /// </summary>
        /// <param name="x1">Value operand (integer/bool/char only; bool promotes to int8).</param>
        /// <param name="x2">Shift-count operand (integer/bool/char only).</param>
        /// <param name="out">A location into which the result is stored; must be same_kind-castable from
        /// the promoted loop dtype. Returned as-is.</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written; masked-off out
        /// slots keep prior contents.</param>
        /// <param name="dtype">Optional loop dtype override; must be an integer/char loop — a
        /// float/complex/decimal request raises NumPy's no-loop TypeError.</param>
        /// <returns>Array with bits shifted right (or <paramref name="out"/> when provided).</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.right_shift.html
        ///
        /// One signature, NumPy's ufunc form (2026-11-11): the former overload that took the shift count as an
        /// <c>object</c> read a boxed number as a STRONG array; a C# literal converts to a 0-d array marked as the
        /// Python scalar it stands for (NEP 50's weak scalar), and a value held in an <c>object</c> variable goes
        /// through <see cref="np.asanyarray(in object, DType, char?, string, bool?)"/>.
        ///
        /// Bits are shifted to the right by removing x2 bits from the right of x1.
        /// For unsigned integers, this is logical shift (zeros filled from left).
        /// For signed integers, this is arithmetic shift (sign bit extended).
        /// This operation is equivalent to floor division by 2**x2.
        ///
        /// Example:
        ///   np.right_shift(20, 2) = 5  # 0b10100 -> 0b101
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray right_shift(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, DType dtype = null)
            => x1.TensorEngine.RightShift(x1, x2, dtype, @out, where);

        /// <summary>
        ///     Shift the bits of an integer to the right, element-wise (NumPy's <c>np.bitwise_right_shift</c>,
        ///     the Array-API name NumPy 2.0 added: <c>np.bitwise_right_shift is np.right_shift</c>, the
        ///     same ufunc object, so this forwards to <see cref="right_shift(NDArray, NDArray, NDArray, NDArray, DType)"/> and
        ///     answers bit for bit as it does).
        /// </summary>
        /// <param name="x1">Value operand (integer / bool / char only).</param>
        /// <param name="x2">Shift-count operand (integer / bool / char only).</param>
        /// <param name="out">A location into which the result is stored (NumPy ufunc <c>out=</c>); <c>null</c> allocates the
        ///     result.</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed / written (NumPy ufunc <c>where=</c>);
        ///     <c>null</c> is NumPy's <c>where=True</c>.</param>
        /// <param name="dtype">The loop dtype (NumPy ufunc <c>dtype=</c>); must be an integer / bool loop - a float request raises NumPy's no-loop TypeError.</param>
        /// <returns>The result of <see cref="right_shift(NDArray, NDArray, NDArray, NDArray, DType)"/>: <paramref name="out"/>
        ///     itself when it was given.</returns>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.bitwise_right_shift.html
        ///     <para>One signature, the ufunc form of the original's. A C# literal operand converts to a 0-d array marked as
        ///     the Python scalar it stands for, so NEP 50's weak-scalar promotion holds without the original's
        ///     <c>object</c> overload (<c>np.bitwise_right_shift(int32_array, 2)</c> stays int32, as NumPy's).</para></remarks>
        public static NDArray bitwise_right_shift(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, DType dtype = null)
            => right_shift(x1, x2, @out, where, dtype);
    }
}
