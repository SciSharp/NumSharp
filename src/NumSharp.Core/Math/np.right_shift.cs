namespace NumSharp
{
    public partial class np
    {
        /// <summary>
        /// Shift the bits of an integer to the right.
        /// </summary>
        /// <param name="x1">Input array (integer types only).</param>
        /// <param name="x2">Number of bits to shift (integer types only).</param>
        /// <returns>Array with bits shifted right.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.right_shift.html
        ///
        /// Bits are shifted to the right by removing x2 bits from the right of x1.
        /// For unsigned integers, this is logical shift (zeros filled from left).
        /// For signed integers, this is arithmetic shift (sign bit extended).
        /// This operation is equivalent to floor division by 2**x2.
        ///
        /// Example:
        ///   np.right_shift(20, 2) = 5  # 0b10100 -> 0b101
        /// </remarks>
        public static NDArray right_shift(NDArray x1, NDArray x2) => x1.TensorEngine.RightShift(x1, x2);

        /// <summary>
        /// Shift the bits of an integer to the right by a scalar amount.
        /// </summary>
        /// <param name="x1">Input array (integer types only).</param>
        /// <param name="x2">Number of bits to shift.</param>
        /// <returns>Array with bits shifted right.</returns>
        public static NDArray right_shift(NDArray x1, int x2) => x1.TensorEngine.RightShift(x1, x2);
    }
}
