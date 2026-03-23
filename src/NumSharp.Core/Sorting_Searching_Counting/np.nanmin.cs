namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Return minimum of an array or minimum along an axis, ignoring any NaNs.
        /// </summary>
        /// <param name="a">Array containing numbers whose minimum is desired. If a is not an array, a conversion is attempted.</param>
        /// <param name="axis">Axis or axes along which the minimum is computed. The default is to compute the minimum of the flattened array.</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one.</param>
        /// <returns>A new array containing the minimum. If all values are NaN, returns NaN.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.nanmin.html
        /// Only applicable to float and double arrays.
        /// For integer arrays, this is equivalent to np.amin (no NaN values possible).
        /// </remarks>
        public static NDArray nanmin(NDArray a, int? axis = null, bool keepdims = false)
            => a.TensorEngine.NanMin(a, axis, keepdims);
    }
}
