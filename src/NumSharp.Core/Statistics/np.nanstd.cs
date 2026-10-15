using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Compute the standard deviation along the specified axis, while ignoring NaNs: NumPy's <c>numpy.nanstd</c>, the
        ///     square root of <see cref="nanvar"/> taken as NumPy takes it.
        /// </summary>
        /// <param name="a">The array whose standard deviation is desired; any dtype and layout.</param>
        /// <param name="axis">The axis along which the standard deviation is computed (negative counts from the end); null
        /// (the default) computes it over the flattened array.</param>
        /// <param name="keepdims">If true, the reduced axes are left in the result as dimensions with size one.</param>
        /// <param name="ddof">Delta degrees of freedom: the variance's divisor is <c>N - ddof</c>, where <c>N</c> is the
        /// number of non-NaN elements of each slice. Default 0.</param>
        /// <param name="dtype">The type used to compute the variance (both sums run in it). For a float or complex input it
        /// must be inexact; null uses the input's dtype (np.std's float64 for an integer or bool input).</param>
        /// <returns>The standard deviation, in the variance's dtype; a 0-d array for a flat reduction without
        /// <paramref name="keepdims"/>. A slice with no degrees of freedom left is NaN.</returns>
        /// <exception cref="TypeError">A float or complex <paramref name="a"/> with a non-inexact <paramref name="dtype"/>:
        /// "If a is inexact, then dtype must be inexact".</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is outside <paramref name="a"/>'s dimensions (a 0-d array
        /// included).</exception>
        /// <exception cref="System.ArgumentException">An integer or bool input with an integer <paramref name="dtype"/>
        /// and an array result: NumPy's "Cannot cast ufunc 'sqrt' output ..." error, as np.std raises it.</exception>
        /// <remarks>
        ///     <para>
        ///     NumPy's <c>nanstd</c> is <c>sqrt(nanvar(...), out=var)</c> for an array and
        ///     <c>var.dtype.type(np.sqrt(var))</c> for a scalar; for an array that cannot hold NaN that is exactly
        ///     <see cref="std(NDArray, int, bool, int?, DType)"/>.
        ///     </para>
        ///     <para>https://numpy.org/doc/stable/reference/generated/numpy.nanstd.html</para>
        /// </remarks>
        /// <exception cref="TypeError">A <c>numpy.matrix</c> operand with <c>keepdims: true</c>: NumPy calls the matrix method, which has no such keyword. The text names that method, as in <c>matrix.sum() got an unexpected keyword argument 'keepdims'</c> for <c>np.sum</c>.</exception>
        [NDScoped]
        public static NDArray nanstd(NDArray a, int? axis = null, bool keepdims = false, int ddof = 0, DType dtype = null)
            => NDMatrixBoundary.Sees(a)   // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            ? NDMatrixBoundary.Reduced((a, axis, keepdims, ddof, dtype), static s => nanstd(s.a, s.axis, s.keepdims, s.ddof, s.dtype), axis, keepdims, (a.typecode is NPTypeCode.Half or NPTypeCode.Single or NPTypeCode.Double or NPTypeCode.Complex) ? "sum" : "var")
            : DefaultEngine.NanVarStdNumPy(a, axis, keepdims, ddof, dtype, takeSqrt: true);
    }
}
