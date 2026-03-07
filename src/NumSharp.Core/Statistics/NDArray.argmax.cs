namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Returns the index of the maximum value (flattened array).
        /// </summary>
        /// <returns>The index of the maximal value in the flattened array.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html</remarks>
        public int argmax()
        {
            return np.argmax(this);
        }

        /// <summary>
        ///     Returns the indices of the maximum values along an axis.
        /// </summary>
        /// <param name="axis">The axis along which to operate. By default, the index is into the flattened array.</param>
        /// <returns>Array of indices into the array. It has the same shape as a.shape with the dimension along axis removed.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html</remarks>
        public NDArray argmax(int axis)
        {
            return np.argmax(this, axis);
        }
    }
}
