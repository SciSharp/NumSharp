namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Returns the index of the minimum value (flattened array).
        /// </summary>
        /// <returns>The index of the minimum value in the flattened array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.argmin.html</remarks>
        public long argmin()
        {
            return np.argmin(this);
        }

        /// <summary>
        ///     Returns the indices of the minimum values along an axis.
        /// </summary>
        /// <param name="axis">The axis along which to operate. By default, the index is into the flattened array.</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one.</param>
        /// <returns>Array of indices into the array. It has the same shape as a.shape with the dimension along axis removed (unless keepdims is True).</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.argmin.html</remarks>
        public NDArray argmin(int axis, bool keepdims = false)
        {
            return np.argmin(this, axis, keepdims);
        }
    }
}
