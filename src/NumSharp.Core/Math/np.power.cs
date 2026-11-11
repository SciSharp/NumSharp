using System;
using NumSharp.Backends;

namespace NumSharp
{
    public partial class np
    {
        /// <summary>
        ///     First array elements raised to powers from second array, element-wise.
        ///     Mirrors NumPy's ufunc signature: <c>power(x1, x2, /, out=None, *, where=True, dtype=None)</c>.
        /// </summary>
        /// <param name="x1">The bases.</param>
        /// <param name="x2">The exponents (array).</param>
        /// <param name="out">A location into which the result is stored (joins the broadcast without being stretched, must be same_kind-castable from the loop dtype; returned as-is).</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written (NumPy ufunc where=).</param>
        /// <param name="dtype">Explicit loop dtype (NumPy ufunc dtype=): the computation runs in this dtype (power(2, -1, dtype: float64) = 0.5; inputs must be same_kind-castable to it).</param>
        /// <returns>The bases in x1 raised to the exponents in x2 (<paramref name="out"/> itself when given).</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.power.html
        ///     <para>One signature, NumPy's ufunc form (2026-11-11): the former overload that took the second operand as an
        ///     <c>object</c> read a boxed number as a STRONG array (parity QA-QB30); a C# literal converts to a 0-d array
        ///     marked as the Python scalar it stands for, so NEP 50's weak-scalar rule holds without it, and a value held in
        ///     an <c>object</c> variable goes through <see cref="np.asanyarray(in object, DType, string)"/>.</para>
        ///     <para>The former two-argument and <c>in</c> overloads called the same engine method with no out / where /
        ///     dtype, so <c>np.power(a, b)</c> computes exactly as before through this one.</para>
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray power(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, DType dtype = null)
            => x1.TensorEngine.Power(x1, x2, dtype, @out, where);

        /// <summary>
        ///     First array elements raised to powers from second array, element-wise (NumPy's <c>np.pow</c>,
        ///     the Array-API name NumPy 2.0 added: <c>np.pow is np.power</c>, the
        ///     same ufunc object, so this forwards to <see cref="power(NDArray, NDArray, NDArray, NDArray, DType)"/> and
        ///     answers bit for bit as it does).
        /// </summary>
        /// <param name="x1">The bases.</param>
        /// <param name="x2">The exponents; broadcast against <paramref name="x1"/>.</param>
        /// <param name="out">A location into which the result is stored (NumPy ufunc <c>out=</c>); <c>null</c> allocates the
        ///     result.</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed / written (NumPy ufunc <c>where=</c>);
        ///     <c>null</c> is NumPy's <c>where=True</c>.</param>
        /// <param name="dtype">The loop dtype (NumPy ufunc <c>dtype=</c>): the computation runs in this dtype.</param>
        /// <returns>The result of <see cref="power(NDArray, NDArray, NDArray, NDArray, DType)"/>: <paramref name="out"/>
        ///     itself when it was given.</returns>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.pow.html
        ///     <para>One signature, the ufunc form of the original's. A C# literal operand converts to a 0-d array marked as
        ///     the Python scalar it stands for, so NEP 50's weak-scalar promotion holds without the original's
        ///     <c>object</c> overload (<c>np.pow(int32_array, 2)</c> stays int32, as NumPy's).</para></remarks>
        public static NDArray pow(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, DType dtype = null)
            => power(x1, x2, @out, where, dtype);


        /// <summary>
        ///     Return the element-wise square of the input.
        /// </summary>
        /// <param name="x">Input data.</param>
        /// <returns>Element-wise x*x, of the same shape and dtype as x. Returns scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.power.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray square(NDArray x) => x.TensorEngine.Square(x, (Type)null);

        /// <summary>
        ///     Return the element-wise square of the input.
        ///     Mirrors NumPy's ufunc signature: <c>square(x, /, out=None, *, where=True, dtype=None)</c>.
        /// </summary>
        /// <param name="x">Input data.</param>
        /// <param name="out">A location into which the result is stored (joins the broadcast without being stretched, must be same_kind-castable from the loop dtype; returned as-is).</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written (NumPy ufunc where=).</param>
        /// <param name="dtype">Explicit loop dtype (NumPy ufunc dtype=): the input must be same_kind-castable to it.</param>
        /// <returns>Element-wise <c>x*x</c> (<paramref name="out"/> itself when given).</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.square.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray square(NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null)
            => x.TensorEngine.Square(x, dtype, @out, where);
    }
}
