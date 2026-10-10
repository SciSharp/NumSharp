using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Compute the variance along the specified axis, while ignoring NaNs: NumPy's <c>numpy.nanvar</c>, statement for
        ///     statement. NaNs count neither in the mean nor in the divisor <c>N - ddof</c>, where <c>N</c> is the number of
        ///     non-NaN elements of each slice.
        /// </summary>
        /// <param name="a">The array whose variance is desired; any dtype and layout.</param>
        /// <param name="axis">The axis along which the variance is computed (negative counts from the end); null (the
        /// default) computes the variance of the flattened array.</param>
        /// <param name="keepdims">If true, the reduced axes are left in the result as dimensions with size one.</param>
        /// <param name="ddof">Delta degrees of freedom: the divisor is <c>N - ddof</c>. Default 0.</param>
        /// <param name="dtype">The type used to compute the variance (both sums run in it). For a float or complex input it
        /// must be inexact; null uses the input's dtype (np.var's float64 for an integer or bool input).</param>
        /// <returns>The variance, in <paramref name="dtype"/> or the input's float dtype (float64 for a complex input and for
        /// an integer or bool one); a 0-d array for a flat reduction without <paramref name="keepdims"/>. A slice with no
        /// degrees of freedom left (all NaN, or <paramref name="ddof"/> at or past its non-NaN count) is NaN.</returns>
        /// <exception cref="TypeError">A float or complex <paramref name="a"/> with a non-inexact <paramref name="dtype"/>:
        /// "If a is inexact, then dtype must be inexact".</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is outside <paramref name="a"/>'s dimensions (a 0-d array
        /// included).</exception>
        /// <remarks>
        ///     <para>
        ///     An array that cannot hold NaN (integer, bool, char, decimal) has no NaN to ignore: the result is
        ///     <see cref="var(NDArray, int, bool, int?, DType)"/>'s, keywords included. For a float or complex array the
        ///     arithmetic is NumPy's <c>_nanfunctions_impl.nanvar</c> (see <c>DefaultEngine.NanVarStdNumPy</c>), which
        ///     differs from np.var in three observable ways: the deviations are rounded into the input's dtype before they
        ///     are squared (with <c>dtype=float64</c>, a float32 input squares float32 deviations), a complex square is the
        ///     real part of the complex product <c>x * conj(x)</c>, and a slice without degrees of freedom is NaN (np.var
        ///     gives +inf there).
        ///     </para>
        ///     <para>https://numpy.org/doc/stable/reference/generated/numpy.nanvar.html</para>
        /// </remarks>
        [NDScoped]
        public static NDArray nanvar(NDArray a, int? axis = null, bool keepdims = false, int ddof = 0, DType dtype = null)
            => NDMatrixBoundary.Sees(a)   // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            ? NDMatrixBoundary.Reduced((a, axis, keepdims, ddof, dtype), static s => nanvar(s.a, s.axis, s.keepdims, s.ddof, s.dtype), axis, keepdims, (a.typecode is NPTypeCode.Half or NPTypeCode.Single or NPTypeCode.Double or NPTypeCode.Complex) ? "sum" : "var")
            : DefaultEngine.NanVarStdNumPy(a, axis, keepdims, ddof, dtype, takeSqrt: false);
    }
}
