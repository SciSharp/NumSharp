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
        ///     Refer to <see cref="np.clip(NDArray, NDArray, NDArray, NDArray, DType, NDArray, NDArray, NDArray)"/>
        ///     for full documentation.
        /// </summary>
        /// <param name="min">Minimum value. If null, clipping is not performed on the lower interval edge. An array bound broadcasts against this array.</param>
        /// <param name="max">Maximum value. If null, clipping is not performed on the upper interval edge.</param>
        /// <param name="out">The results will be placed in this array, which is returned as-is. It may be this array for in-place clipping, or overlap it in any other way (the result is computed as if from a copy).</param>
        /// <param name="dtype">The dtype the computation runs in (NumPy's <c>ndarray.clip(..., **kwargs)</c> passes <c>dtype</c> through to the ufunc). Null (default) keeps the promoted result dtype.</param>
        /// <param name="where">NumPy's ufunc <c>where=</c> mask (bool, broadcast with the operands): only its True positions are written.</param>
        /// <returns>An array with the elements of this array, but where values &lt; min are replaced with min, and those &gt; max with max.</returns>
        /// <exception cref="OverflowException">A kept integer bound does not fit the integer dtype the computation runs in.</exception>
        /// <exception cref="ArgumentException">A <paramref name="dtype"/> or <paramref name="out"/> that breaks NumPy's same_kind casting rule, or a non-bool <paramref name="where"/>.</exception>
        /// <exception cref="ValueError"><paramref name="out"/> is read-only, or the operands do not broadcast together / <paramref name="out"/> would be stretched.</exception>
        /// <exception cref="TypeError">No bound is left and the computation would run in bool.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.ndarray.clip.html
        ///     <para>NumPy's <c>ndarray.clip</c> forwards to <c>_methods._clip(a, min=None, max=None, out=None, **kwargs)</c>
        ///     directly, so the <c>a_min</c>/<c>min</c> keyword rule of <c>np.clip</c> does not apply here: <c>min</c> and
        ///     <c>max</c> are the only bound parameters.</para>
        /// </remarks>
        public NDArray clip(NDArray min = null, NDArray max = null, NDArray @out = null, DType dtype = null, NDArray where = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(this);   // the member keeps the plain-array meaning for a matrix object
            return np.clip(this, a_min: min, a_max: max, @out: @out, dtype: dtype, where: where);
        }
    }
}
