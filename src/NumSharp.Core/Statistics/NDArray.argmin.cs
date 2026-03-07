namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Returns the index of the minimum value (flattened array).
        /// </summary>
        /// <returns>The index of the minimum value in the flattened array.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmin.html</remarks>
        public int argmin()
        {
            return np.argmin(this);
        }

        /// <summary>
        ///     Returns the indices of the minimum values along an axis.
        /// </summary>
        /// <param name="axis">The axis along which to operate. By default, the index is into the flattened array.</param>
        /// <returns>Array of indices into the array. It has the same shape as a.shape with the dimension along axis removed.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmin.html</remarks>
        public NDArray argmin(int axis)
        {
            return np.argmin(this, axis);
        }
    }
}
