namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Interchange two axes of an array.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="axis1">First axis.</param>
        /// <param name="axis2">Second axis.</param>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.swapaxes.html</remarks>
        public NDArray swapaxes(int axis1, int axis2)
            => TensorEngine.SwapAxes(this, axis1, axis2);
    }
}
