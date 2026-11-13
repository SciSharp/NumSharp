using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Cosine element-wise.
        ///     Mirrors NumPy's ufunc signature: <c>cos(x, /, out=None, *, where=True, dtype=None)</c>.
        /// </summary>
        /// <param name="x">Input array in radians.</param>
        /// <param name="@out">A location into which the result is stored (joins the broadcast without being stretched, must be same_kind-castable from the loop dtype; returned as-is).</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written (NumPy ufunc where=).</param>
        /// <param name="dtype">Explicit loop dtype (NumPy ufunc dtype=): the computation runs at this precision; integer/bool requests raise NumPy's "No loop matching" error.</param>
        /// <returns>The cosine of each element of x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.cos.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray cos(NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null)
            => x.TensorEngine.Cos(x, dtype, @out, where);

        /// <summary>
        ///     Hyperbolic cosine, element-wise. <br></br>
        ///     Equivalent to 1/2 * (np.exp(x) + np.exp(-x)) and np.cos(1j* x).
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>Output array of same shape as x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.cosh.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray cosh(NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null)
            => x.TensorEngine.Cosh(x, dtype, @out, where);

        /// <summary>
        ///     Trigonometric inverse cosine, element-wise. <br></br>
        ///     The inverse of cos so that, if y = cos(x), then x = arccos(y).
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>The angle of the ray intersecting the unit circle at the given x-coordinate in radians [0, pi]. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.arccos.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray arccos(NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null)
            => x.TensorEngine.ACos(x, dtype, @out, where);

        /// <summary>
        ///     Trigonometric inverse cosine, element-wise (NumPy's <c>np.acos</c>,
        ///     the Array-API name NumPy 2.0 added: <c>np.acos is np.arccos</c>, the
        ///     same ufunc object, so this forwards to <see cref="arccos"/> and answers bit for bit as it does).
        /// </summary>
        /// <param name="x">Input array (a C# scalar converts to a 0-d array marked as the Python scalar it stands for).</param>
        /// <param name="out">A location into which the result is stored (NumPy ufunc <c>out=</c>); <c>null</c> allocates the
        ///     result.</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed / written (NumPy ufunc <c>where=</c>);
        ///     <c>null</c> is NumPy's <c>where=True</c>.</param>
        /// <param name="dtype">The loop dtype (NumPy ufunc <c>dtype=</c>); only non-integer dtypes have a loop.</param>
        /// <returns>The result of <see cref="arccos"/>: <paramref name="out"/> itself when it was given.</returns>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.acos.html
        ///     <para>One signature, the ufunc form of the original's; the original's own extra overloads (if any) are not
        ///     copied, so the alias never carries a second overload.</para></remarks>
        public static NDArray acos(NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null)
            => arccos(x, @out, where, dtype);
    }
}
