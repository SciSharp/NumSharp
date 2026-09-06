namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Remove axes of length one from this array. Refer to <see cref="np.squeeze(NDArray)"/> for full
        ///     documentation.
        /// </summary>
        /// <param name="axis">
        ///     Selects a subset of the entries of length one to remove. The default (null) removes all length-one
        ///     axes. Selecting an axis whose length is not one raises.
        /// </param>
        /// <returns>A view of this array with the selected length-one axes removed. Shares memory with this array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ndarray.squeeze.html</remarks>
        public NDArray squeeze(int? axis = null)
            => axis is null ? np.squeeze(this) : np.squeeze(this, axis.Value);
    }
}
