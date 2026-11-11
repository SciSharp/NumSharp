using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Compute tangent element-wise. <br></br>
        ///     Equivalent to np.sin(x)/np.cos(x) element-wise.
        ///     Mirrors NumPy's ufunc signature: <c>tan(x, /, out=None, *, where=True, dtype=None)</c>.
        /// </summary>
        /// <param name="x">Angle, in radians (2 \pi rad equals 360 degrees).</param>
        /// <param name="@out">A location into which the result is stored (joins the broadcast without being stretched, must be same_kind-castable from the loop dtype; returned as-is).</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written (NumPy ufunc where=).</param>
        /// <param name="dtype">Explicit loop dtype (NumPy ufunc dtype=): the computation runs at this precision; integer/bool requests raise NumPy's "No loop matching" error.</param>
        /// <returns>The tangent of each element of x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.tan.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray tan(NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null)
            => x.TensorEngine.Tan(x, dtype, @out, where);

        /// <summary>
        ///     Compute hyperbolic tangent element-wise. <br></br>
        ///     Equivalent to np.sinh(x)/np.cosh(x) or -1j * np.tan(1j*x).
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>The sine of each element of x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.tanh.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray tanh(NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null)
            => x.TensorEngine.Tanh(x, dtype, @out, where);

        /// <summary>
        ///     Compute trigonometric inverse tangent, element-wise. <br></br>
        ///     The inverse of tan, so that if y = tan(x) then x = arctan(y).
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>Return has the same shape as x. Its real part is in [-pi/2, pi/2] (arctan(+/-inf) returns +/-pi/2). This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.arctan.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray arctan(NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null)
            => x.TensorEngine.ATan(x, dtype, @out, where);

        /// <summary>
        ///     Compute Element-wise arc tangent of x1/x2 choosing the quadrant correctly. <br></br>
        ///     By IEEE convention, this function is defined for x2 = +/-0 and for either or both of x1 and x2 = +/-inf
        /// </summary>
        /// <param name="y">The y-coordinates (NumPy's <c>x1</c>).</param>
        /// <param name="x">The x-coordinates (NumPy's <c>x2</c>). If <c>y.shape != x.shape</c>, they must be broadcastable to a common shape (which becomes the shape of the output).</param>
        /// <param name="out">A location into which the result is stored (NumPy ufunc out=); <c>null</c> allocates the result.</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written (NumPy ufunc where=).</param>
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>The Array of angles in radians, in the range [-pi, pi]. This is a scalar if both x1 and x2 are scalars.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.arctan2.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray arctan2(NDArray y, NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null)
            => x.TensorEngine.ATan2(y, x, dtype, @out, where);

        /// <summary>
        ///     Trigonometric inverse tangent, element-wise (NumPy's <c>np.atan</c>,
        ///     the Array-API name NumPy 2.0 added: <c>np.atan is np.arctan</c>, the
        ///     same ufunc object, so this forwards to <see cref="arctan"/> and answers bit for bit as it does).
        /// </summary>
        /// <param name="x">Input array (a C# scalar converts to a 0-d array marked as the Python scalar it stands for).</param>
        /// <param name="out">A location into which the result is stored (NumPy ufunc <c>out=</c>); <c>null</c> allocates the
        ///     result.</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed / written (NumPy ufunc <c>where=</c>);
        ///     <c>null</c> is NumPy's <c>where=True</c>.</param>
        /// <param name="dtype">The loop dtype (NumPy ufunc <c>dtype=</c>); only non-integer dtypes have a loop.</param>
        /// <returns>The result of <see cref="arctan"/>: <paramref name="out"/> itself when it was given.</returns>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.atan.html
        ///     <para>One signature, the ufunc form of the original's; the original's own extra overloads (if any) are not
        ///     copied, so the alias never carries a second overload.</para></remarks>
        public static NDArray atan(NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null)
            => arctan(x, @out, where, dtype);

        /// <summary>
        ///     Element-wise arc tangent of y/x choosing the quadrant correctly (NumPy's <c>np.atan2</c>,
        ///     the Array-API name NumPy 2.0 added: <c>np.atan2 is np.arctan2</c>, the
        ///     same ufunc object, so this forwards to <see cref="arctan2(NDArray, NDArray, NDArray, NDArray, DType)"/> and
        ///     answers bit for bit as it does).
        /// </summary>
        /// <param name="y">The y-coordinates (NumPy's positional <c>x1</c>).</param>
        /// <param name="x">The x-coordinates (NumPy's positional <c>x2</c>); broadcast against <paramref name="y"/>.</param>
        /// <param name="out">A location into which the result is stored (NumPy ufunc <c>out=</c>); <c>null</c> allocates the
        ///     result.</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed / written (NumPy ufunc <c>where=</c>);
        ///     <c>null</c> is NumPy's <c>where=True</c>.</param>
        /// <param name="dtype">The loop dtype (NumPy ufunc <c>dtype=</c>); only non-integer dtypes have a loop.</param>
        /// <returns>The result of <see cref="arctan2(NDArray, NDArray, NDArray, NDArray, DType)"/>: <paramref name="out"/>
        ///     itself when it was given.</returns>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.atan2.html
        ///     <para>One signature, the ufunc form of the original's. A C# literal operand converts to a 0-d array marked as
        ///     the Python scalar it stands for, so NEP 50's weak-scalar promotion holds without the original's
        ///     <c>object</c> overload (<c>np.atan2(int32_array, 2)</c> stays int32, as NumPy's).</para></remarks>
        public static NDArray atan2(NDArray y, NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null)
            => arctan2(y, x, @out, where, dtype);
    }
}
