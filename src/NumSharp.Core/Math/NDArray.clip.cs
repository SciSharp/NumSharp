namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Return an array whose values are limited to <c>[min, max]</c>. One of <paramref name="min"/> or
        ///     <paramref name="max"/> must be given. Refer to <see cref="np.clip(NDArray, NDArray, NDArray, NDArray, NumSharp.Backends.NPTypeCode?, NDArray, NDArray)"/>
        ///     for full documentation.
        /// </summary>
        /// <param name="min">Minimum value. If null, clipping is not performed on the lower interval edge.</param>
        /// <param name="max">Maximum value. If null, clipping is not performed on the upper interval edge.</param>
        /// <param name="out">The results will be placed in this array. It may be the input array for in-place clipping.</param>
        /// <returns>An array with the elements of this array, but where values &lt; min are replaced with min, and those &gt; max with max.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ndarray.clip.html</remarks>
        public NDArray clip(NDArray min = null, NDArray max = null, NDArray @out = null)
            => np.clip(this, a_min: min, a_max: max, @out: @out);
    }
}
