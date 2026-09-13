namespace NumSharp {
    public static partial class np
    {
        /// <summary>
        ///     True if two arrays have the same shape and all elements are equal, False otherwise.
        /// </summary>
        /// <param name="a">First input array.</param>
        /// <param name="b">Second input array.</param>
        /// <param name="equal_nan">
        ///     When <c>false</c> (the NumPy default) NaN never compares equal, so an array that holds a NaN
        ///     is NOT equal even to itself — <c>np.array_equal(a, a)</c> is <c>false</c> whenever <paramref name="a"/>
        ///     contains a NaN. When <c>true</c>, NaNs at matching positions are treated as equal; for a complex
        ///     dtype a value counts as NaN when EITHER its real or imaginary component is NaN.
        /// </param>
        /// <returns>True if the arrays have identical shape and every element is equal.</returns>
        /// <remarks>
        ///     The shapes must match EXACTLY — unlike <see cref="array_equiv(NDArray,NDArray)"/> this performs
        ///     NO broadcasting. Mirrors <c>numpy.array_equal(a1, a2, equal_nan=False)</c> (NumPy 2.4.2).
        ///     https://numpy.org/doc/stable/reference/generated/numpy.array_equal.html
        /// </remarks>
        public static bool array_equal(NDArray a, NDArray b, bool equal_nan = false)
        {
            // A null "array" cannot equal a real one; two nulls are trivially equal (mirrors the
            // NDArray == null convention). Guarding here also keeps a.array_equal(...) from throwing.
            if (a is null)
                return b is null;
            return a.array_equal(b, equal_nan);
        }
    }
}
