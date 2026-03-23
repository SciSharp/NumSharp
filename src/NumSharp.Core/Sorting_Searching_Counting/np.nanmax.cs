namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Return maximum of an array or maximum along an axis, ignoring any NaNs.
        /// </summary>
        /// <param name="a">Array containing numbers whose maximum is desired. If a is not an array, a conversion is attempted.</param>
        /// <param name="axis">Axis or axes along which the maximum is computed. The default is to compute the maximum of the flattened array.</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one.</param>
        /// <returns>A new array containing the maximum. If all values are NaN, returns NaN.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.nanmax.html
        /// Only applicable to float and double arrays.
        /// For integer arrays, this is equivalent to np.amax (no NaN values possible).
        /// </remarks>
        public static NDArray nanmax(NDArray a, int? axis = null, bool keepdims = false)
            => a.TensorEngine.NanMax(a, axis, keepdims);
    }
}
