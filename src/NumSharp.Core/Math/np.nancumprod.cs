using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return the cumulative product of array elements over a given axis treating Not a Numbers
        ///     (NaNs) as one. The cumulative product does not change when NaNs are encountered and leading
        ///     NaNs are replaced by ones. Ones are returned for slices that are all-NaN or empty.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="axis">Axis along which the cumulative product is computed. The default (None) is to compute the cumprod over the flattened array.</param>
        /// <param name="typeCode">Type of the returned array and of the accumulator in which the elements are multiplied. If not specified, it defaults to the dtype of <paramref name="a"/>, unless <paramref name="a"/> has an integer dtype with a precision less than that of the default platform integer, in which case the default platform integer is used.</param>
        /// <param name="out">Alternate output array in which to place the result. It must have the same shape and buffer length as the expected output, but its dtype may differ (the result is cast into it with NumPy's unsafe casting) and a reference to <paramref name="out"/> is returned.</param>
        /// <returns>A new array holding the result unless <paramref name="out"/> is specified, in which case a reference to <paramref name="out"/> is returned.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.nancumprod.html
        /// Port of numpy/lib/_nanfunctions_impl.py::nancumprod: <c>_replace_nan(a, 1)</c> then <c>np.cumprod</c>.
        /// Only float-family dtypes can contain NaN, so integer/bool/decimal inputs are equivalent to <see cref="cumprod"/>.
        /// </remarks>
        public static NDArray nancumprod(NDArray a, int? axis = null, NPTypeCode? typeCode = null, NDArray @out = null)
        {
            // NumPy: a, mask = _replace_nan(a, 1); return np.cumprod(a, axis, dtype, out).
            return cumprod(_replace_nan_for_scan(a, 1), axis, typeCode, @out);
        }
    }
}
