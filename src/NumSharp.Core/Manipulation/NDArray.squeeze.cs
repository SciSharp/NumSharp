namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Remove axes of length one from this array. Refer to <see cref="np.squeeze(NDArray, int?)"/> for full
        ///     documentation.
        /// </summary>
        /// <param name="axis">
        ///     Selects a subset of the entries of length one to remove. The default (null) removes all length-one
        ///     axes. Selecting an axis whose length is not one raises.
        /// </param>
        /// <returns>A view of this array with the selected length-one axes removed. Shares memory with this array.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range.</exception>
        /// <exception cref="ValueError">The selected axis does not have length one.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ndarray.squeeze.html</remarks>
        [PlainReceiver]
        public NDArray squeeze(int? axis = null)
        {
            return np.squeeze(this, axis);
        }
    }
}
