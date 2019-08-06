namespace NumSharp {
    public static partial class np
    {
        /// <summary>
        ///     True if two arrays have the same shape and elements, False otherwise.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="b">Input array.</param>
        /// <returns>Returns True if the arrays are equal.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy-1.16.0/reference/generated/numpy.array_equal.html</remarks>
        public static bool array_equal(NDArray a, NDArray b)
        {
            return a.array_equal(b);
        }
    }
}
