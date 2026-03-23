namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Return the sum of array elements over a given axis treating Not a Numbers (NaNs) as zero.
        /// </summary>
        /// <param name="a">Array containing numbers whose sum is desired. If a is not an array, a conversion is attempted.</param>
        /// <param name="axis">Axis or axes along which the sum is computed. The default is to compute the sum of the flattened array.</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one.</param>
        /// <returns>A new array containing the sum, with NaN values treated as zero.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.nansum.html
        /// Only applicable to float and double arrays.
        /// For integer arrays, this is equivalent to np.sum (no NaN values possible).
        /// </remarks>
        public static NDArray nansum(NDArray a, int? axis = null, bool keepdims = false)
            => a.TensorEngine.NanSum(a, axis, keepdims);
    }
}
