using System;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Return an array whose values are limited to <c>[min, max]</c>. If neither <paramref name="min"/>
        ///     nor <paramref name="max"/> is given the array is returned unchanged (a copy; NumPy's <c>np.positive</c>,
        ///     which refuses a bool array). On an integer array an integer bound at or past the dtype's limit is
        ///     ignored, and a kept one that does not fit raises <see cref="OverflowException"/> instead of wrapping.
        ///     Refer to <see cref="np.clip(NDArray, NDArray, NDArray, NDArray, DType, NDArray, NDArray)"/>
        ///     for full documentation.
        /// </summary>
        /// <param name="min">Minimum value. If null, clipping is not performed on the lower interval edge.</param>
        /// <param name="max">Maximum value. If null, clipping is not performed on the upper interval edge.</param>
        /// <param name="out">The results will be placed in this array, which is returned as-is. It may be the input array for in-place clipping.</param>
        /// <param name="dtype">The dtype the returned array should be of (NumPy's <c>ndarray.clip(..., **kwargs)</c> passes <c>dtype</c> through to the ufunc). Null (default) keeps the promoted result dtype.</param>
        /// <returns>An array with the elements of this array, but where values &lt; min are replaced with min, and those &gt; max with max.</returns>
        /// <exception cref="OverflowException">A kept integer bound does not fit the integer dtype the computation runs in.</exception>
        /// <exception cref="ArgumentException">A <paramref name="dtype"/> or <paramref name="out"/> that breaks NumPy's same_kind casting rule, or an <paramref name="out"/> of another shape.</exception>
        /// <exception cref="ValueError"><paramref name="out"/> is read-only.</exception>
        /// <exception cref="TypeError">No bound is left and the computation would run in bool.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ndarray.clip.html</remarks>
        public NDArray clip(NDArray min = null, NDArray max = null, NDArray @out = null, DType dtype = null)
            => np.clip(this, a_min: min, a_max: max, @out: @out, dtype: dtype?.GetTypeCode());
    }
}
