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
        /// <param name="dtype">Type of the returned array and of the accumulator in which the elements are multiplied (a <see cref="Type"/>, <see cref="NPTypeCode"/>, dtype string or <see cref="DType"/> — all convert implicitly). If not specified, it defaults to the dtype of <paramref name="a"/>, unless <paramref name="a"/> has an integer dtype with a precision less than that of the default platform integer, in which case the default platform integer is used.</param>
        /// <param name="out">Alternate output array in which to place the result. It must have the same shape and buffer length as the expected output, but its dtype may differ (the result is cast into it with NumPy's unsafe casting) and a reference to <paramref name="out"/> is returned.</param>
        /// <returns>A new array holding the result unless <paramref name="out"/> is specified, in which case a reference to <paramref name="out"/> is returned.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.nancumprod.html
        /// Port of numpy/lib/_nanfunctions_impl.py::nancumprod: <c>_replace_nan(a, 1)</c> then <c>np.cumprod</c>.
        /// Only float-family dtypes can contain NaN, so integer/bool/decimal inputs are equivalent to <see cref="cumprod"/>.
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        [NDScoped]
        public static NDArray nancumprod(NDArray a, int? axis = null, DType dtype = null, NDArray @out = null)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Wrapped((a, axis, dtype, @out), static s => nancumprod(s.a, s.axis, s.dtype, s.@out), @out);
            // Text, or a string dtype= request: NumPy's _replace_nan leaves text alone and its accumulate validates the
            // axis (a 0-d input ravels first), then refuses with its verbatim text; see NDStringUfuncs.ReduceError.
            // A numeric dtype= over text: NumPy's _replace_nan leaves text alone (it masks only inexact arrays), and cumprod's
            // cast to the requested number type parses it — so a parsed 'nan' is MULTIPLIED, not replaced by 1 (probed:
            // np.nancumprod(['inf', 'nan'], dtype='f8') is [inf, nan]). The scope reclaims the parsed array.
            if (Backends.Strings.NDStringUfuncs.ParseForNumericReduce("nancumprod", a, axis, dtype) is { } parsed)
                return cumprod(parsed, axis, dtype, @out);
            Backends.Strings.NDStringUfuncs.ThrowIfReduce("nancumprod", a, axis, @out, dtype);
            // Boundary scope: the full-size NaN-replaced copy is reclaimed at exit; the scan
            // result (or the caller's @out — untracked, a Returns no-op) is yielded.
            // NumPy: a, mask = _replace_nan(a, 1); return np.cumprod(a, axis, dtype, out).
            return cumprod(_replace_nan_for_scan(a, 1), axis, dtype, @out);
        }
    }
}
