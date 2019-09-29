namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Interchange two axes of an array.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="axis1">First axis.</param>
        /// <param name="axis2">Second axis.</param>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.swapaxes.html</remarks>
        public static NDArray swapaxes(in NDArray a, int axis1, int axis2)
            => a.TensorEngine.SwapAxes(a, axis1, axis2);
    }
}