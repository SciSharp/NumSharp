namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Find the index into this SORTED array where <paramref name="v"/> would be inserted to keep it sorted.
        ///     Refer to <see cref="np.searchsorted(NDArray, int, string, NDArray)"/> for full documentation.
        /// </summary>
        /// <param name="v">Value to insert.</param>
        /// <param name="side">If "left" (default), the first suitable location is given; if "right", the last.</param>
        /// <param name="sorter">Optional array of integer indices that sort this array (as from argsort).</param>
        /// <returns>The insertion index.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ndarray.searchsorted.html</remarks>
        public long searchsorted(int v, string side = "left", NDArray sorter = null)
            => np.searchsorted(this, v, side, sorter);

        /// <inheritdoc cref="searchsorted(int, string, NDArray)"/>
        public long searchsorted(double v, string side = "left", NDArray sorter = null)
            => np.searchsorted(this, v, side, sorter);

        /// <summary>
        ///     Find the indices into this SORTED array where each value in <paramref name="v"/> would be inserted
        ///     to keep it sorted. Refer to <see cref="np.searchsorted(NDArray, NDArray, string, NDArray)"/> for
        ///     full documentation.
        /// </summary>
        /// <param name="v">Values to insert.</param>
        /// <param name="side">If "left" (default), the first suitable location is given; if "right", the last.</param>
        /// <param name="sorter">Optional array of integer indices that sort this array (as from argsort).</param>
        /// <returns>An array of insertion indices with the same shape as <paramref name="v"/>.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ndarray.searchsorted.html</remarks>
        public NDArray searchsorted(NDArray v, string side = "left", NDArray sorter = null)
            => np.searchsorted(this, v, side, sorter);
    }
}
