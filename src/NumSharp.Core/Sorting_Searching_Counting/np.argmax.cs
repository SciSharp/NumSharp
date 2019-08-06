namespace NumSharp
{
    public static partial class np
    {

        /// <summary>
        ///     Returns the indices of the maximum values along an axis.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="axis">By default, the index is into the flattened array, otherwise along the specified axis.</param>
        /// <returns>Array of indices into the array. It has the same shape as a.shape with the dimension along axis removed.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html</remarks>
        public static NDArray argmax(NDArray a, int axis)
            => a.TensorEngine.ArgMax(a, axis: axis);

        /// <summary>
        ///     Returns the index of the maximum value.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <returns>Array of indices into the array. It has the same shape as a.shape with the dimension along axis removed.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html</remarks>
        public static int argmax(NDArray a)
            => a.TensorEngine.ArgMax(a);

        /// <summary>
        ///     Returns the indices of the minimum values along an axis.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="axis">By default, the index is into the flattened array, otherwise along the specified axis.</param>
        /// <returns>Array of indices into the array. It has the same shape as a.shape with the dimension along axis removed.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmin.html</remarks>
        public static NDArray argmin(NDArray a, int axis)
            => a.TensorEngine.ArgMin(a, axis: axis);

        /// <summary>
        ///     Returns the index of the minimum value.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <returns>Array of indices into the array. It has the same shape as a.shape with the dimension along axis removed.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmin.html</remarks>
        public static int argmin(NDArray a)
            => a.TensorEngine.ArgMin(a);
    }
}
