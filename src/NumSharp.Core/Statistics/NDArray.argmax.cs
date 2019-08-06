namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Returns the indices of the maximum values along an axis.
        /// </summary>
        /// <returns>The index of the maximal value in the array.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html</remarks>
        public int argmax()
        {
            return np.argmax(this);
        }

        /// <summary>
        ///     Returns the indices of the maximum values along an axis.
        /// </summary>
        /// <returns>Array of indices into the array. It has the same shape as a.shape with the dimension along axis removed.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html</remarks>
        public int argmax(int axis)
        {
            return np.argmax(this, axis);
        }
    }
}
