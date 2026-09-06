using System;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Return an array whose values are limited to <c>[min, max]</c>. If neither <paramref name="min"/>
        ///     nor <paramref name="max"/> is given the array is returned unchanged (a copy). Refer to
        ///     <see cref="np.clip(NDArray, NDArray, NDArray, NDArray, NumSharp.Backends.NPTypeCode?, NDArray, NDArray)"/>
        ///     for full documentation.
        /// </summary>
        /// <param name="min">Minimum value. If null, clipping is not performed on the lower interval edge.</param>
        /// <param name="max">Maximum value. If null, clipping is not performed on the upper interval edge.</param>
        /// <param name="out">The results will be placed in this array. It may be the input array for in-place clipping.</param>
        /// <param name="dtype">The dtype the returned array should be of (NumPy's <c>ndarray.clip(..., **kwargs)</c> passes <c>dtype</c> through to the ufunc). Null (default) keeps the promoted result dtype.</param>
        /// <returns>An array with the elements of this array, but where values &lt; min are replaced with min, and those &gt; max with max.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ndarray.clip.html</remarks>
        public NDArray clip(NDArray min = null, NDArray max = null, NDArray @out = null, Type dtype = null)
            => np.clip(this, a_min: min, a_max: max, @out: @out, dtype: dtype?.GetTypeCode());
    }
}
