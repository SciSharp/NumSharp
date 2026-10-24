using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return the cumulative sum of array elements over a given axis treating Not a Numbers
        ///     (NaNs) as zero. The cumulative sum does not change when NaNs are encountered and leading
        ///     NaNs are replaced by zeros. Zeros are returned for slices that are all-NaN or empty.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="axis">Axis along which the cumulative sum is computed. The default (None) is to compute the cumsum over the flattened array.</param>
        /// <param name="dtype">Type of the returned array and of the accumulator in which the elements are summed (a <see cref="Type"/>, <see cref="NPTypeCode"/>, dtype string or <see cref="DType"/> — all convert implicitly). If not specified, it defaults to the dtype of <paramref name="a"/>, unless <paramref name="a"/> has an integer dtype with a precision less than that of the default platform integer, in which case the default platform integer is used.</param>
        /// <param name="out">Alternate output array in which to place the result. It must have the same shape and buffer length as the expected output, but its dtype may differ (the result is cast into it with NumPy's unsafe casting) and a reference to <paramref name="out"/> is returned.</param>
        /// <returns>A new array holding the result unless <paramref name="out"/> is specified, in which case a reference to <paramref name="out"/> is returned. The result has the same size as <paramref name="a"/>, and the same shape if <paramref name="axis"/> is not None or <paramref name="a"/> is 1-D.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.nancumsum.html
        /// Port of numpy/lib/_nanfunctions_impl.py::nancumsum: <c>_replace_nan(a, 0)</c> then <c>np.cumsum</c>.
        /// Only float-family dtypes can contain NaN, so integer/bool/decimal inputs are equivalent to <see cref="cumsum"/>.
        /// The whole function is <c>DefaultEngine.NanCumSum</c> (the np-layer dtype plan, P1b); this applies the matrix
        /// rule and calls it.
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>); a read-only <paramref name="out"/>; text a numeric loop cannot parse.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range.</exception>
        /// <exception cref="TypeError">A text operand or request NumPy's loop resolution refuses.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="out"/> does not have the result's shape.</exception>
        [NDScoped]
        public static NDArray nancumsum(NDArray a, int? axis = null, DType dtype = null, NDArray @out = null)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Wrapped((a, axis, dtype, @out), static s => nancumsum(s.a, s.axis, s.dtype, s.@out), @out);
            // NumPy's nancumsum is _replace_nan(a, 0) then np.cumsum(a, axis, dtype, out); the whole function, its text
            // routes included, is the engine's: DefaultEngine.NanCumSum.
            return DefaultEngine.NanCumSum(a, axis, dtype, @out);
        }
    }
}
