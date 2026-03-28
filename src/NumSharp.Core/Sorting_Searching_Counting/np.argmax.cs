namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Returns the indices of the maximum values along an axis.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="axis">By default, the index is into the flattened array, otherwise along the specified axis.</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one.</param>
        /// <returns>Array of indices into the array. It has the same shape as a.shape with the dimension along axis removed (unless keepdims is True).</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.argmax.html</remarks>
        public static NDArray argmax(NDArray a, int axis, bool keepdims = false)
            => a.TensorEngine.ArgMax(a, axis: axis, keepdims: keepdims);

        /// <summary>
        ///     Returns the index of the maximum value.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <returns>Index of the maximum value in the flattened array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.argmax.html</remarks>
        public static long argmax(NDArray a)
            => (long)a.TensorEngine.ArgMax(a);

        /// <summary>
        ///     Returns the indices of the minimum values along an axis.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="axis">By default, the index is into the flattened array, otherwise along the specified axis.</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one.</param>
        /// <returns>Array of indices into the array. It has the same shape as a.shape with the dimension along axis removed (unless keepdims is True).</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.argmin.html</remarks>
        public static NDArray argmin(NDArray a, int axis, bool keepdims = false)
            => a.TensorEngine.ArgMin(a, axis: axis, keepdims: keepdims);

        /// <summary>
        ///     Returns the index of the minimum value.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <returns>Index of the minimum value in the flattened array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.argmin.html</remarks>
        public static long argmin(NDArray a)
            => (long)a.TensorEngine.ArgMin(a);
    }
}
