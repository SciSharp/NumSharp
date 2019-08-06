namespace NumSharp {
    public static partial class np
    {
        /// <summary>
        ///     Returns True if two arrays are element-wise equal within a tolerance.
        ///     The tolerance values are positive, typically very small numbers.The
        ///     
        ///     relative difference (`rtol` * abs(`b`)) and the absolute difference
        ///     `atol` are added together to compare against the absolute difference
        ///     between `a` and `b`.
        ///     If either array contains one or more NaNs, False is returned.
        ///     Infs are treated as equal if they are in the same place and of the same
        ///     sign in both arrays.
        /// </summary>
        /// <param name="a">Input array to compare with b</param>
        /// <param name="b">Input array to compare with a.</param>
        /// <param name="rtol">The relative tolerance parameter(see Notes)</param>
        /// <param name="atol">The absolute tolerance parameter(see Notes)</param>
        /// <param name="equal_nan">Whether to compare NaN's as equal.  If True, NaN's in `a` will be
        ///considered equal to NaN's in `b` in the output array.</param>
        public static bool allclose(NDArray a, NDArray b, double rtol = 1.0E-5, double atol = 1.0E-8,
            bool equal_nan = false)
            => a.TensorEngine.AllClose(a, b, rtol, atol, equal_nan);
    }
}
