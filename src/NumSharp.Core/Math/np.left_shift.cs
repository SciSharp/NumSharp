namespace NumSharp
{
    public partial class np
    {
        /// <summary>
        /// Shift the bits of an integer to the left.
        /// </summary>
        /// <param name="x1">Input array (integer types only).</param>
        /// <param name="x2">Number of bits to shift (integer types only).</param>
        /// <returns>Array with bits shifted left.</returns>
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
        public static NDArray left_shift(NDArray x1, NDArray x2) => x1.TensorEngine.LeftShift(x1, x2);

        /// <summary>
        /// Shift the bits of an integer to the left by a scalar or array-like amount.
        /// </summary>
        /// <param name="x1">Input array (integer types only).</param>
        /// <param name="x2">Number of bits to shift (scalar or array-like).</param>
        /// <returns>Array with bits shifted left.</returns>
        public static NDArray left_shift(NDArray x1, object x2) => x1.TensorEngine.LeftShift(x1, np.asanyarray(x2));
    }
}
