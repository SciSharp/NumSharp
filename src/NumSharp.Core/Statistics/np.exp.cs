using System;
using NumSharp.Backends;

namespace NumSharp
{
    public partial class np
    {

        /// <summary>
        ///     Calculate the exponential of all elements in the input array.
        ///     Mirrors NumPy's ufunc signature: <c>exp(x, /, out=None, *, where=True, dtype=None)</c>.
        /// </summary>
        /// <param name="a">Input value.</param>
        /// <param name="out">A location into which the result is stored (joins the broadcast without being stretched, must be same_kind-castable from the loop dtype; returned as-is).</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written (NumPy ufunc where=).</param>
        /// <param name="dtype">Explicit loop dtype (NumPy ufunc dtype=): the computation runs at this precision; integer/bool requests raise NumPy's "No loop matching" error.</param>
        /// <returns><c>e**x</c> element-wise (an integer or bool input computes in its float tier), or <paramref name="out"/> itself when given.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.exp.html
        ///     <para>One signature, NumPy's ufunc form: the one-argument overload went with API-parity batch 3 (2026-11-11), so
        ///     <c>np.exp(x)</c> binds here with the defaults (the same loop, bits and dtype). A method group needs a lambda now
        ///     (<c>xs.Select(x => np.exp(x))</c>), as for every other ufunc.</para>
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray exp(NDArray a, NDArray @out = null, NDArray where = null, DType dtype = null)
            => a.TensorEngine.Exp(a, dtype, @out, where);

        /// <summary>
        ///     Mirrors NumPy's ufunc signature: <c>exp2(x, /, out=None, *, where=True, dtype=None)</c>.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="out">A location into which the result is stored (joins the broadcast without being stretched, must be same_kind-castable from the loop dtype; returned as-is).</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written (NumPy ufunc where=).</param>
        /// <param name="dtype">Explicit loop dtype (NumPy ufunc dtype=): selects the loop; the input must be same_kind-castable to it.</param>
        /// <returns><c>2**x</c> element-wise (an integer or bool input computes in its float tier), or <paramref name="out"/> itself when given.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.exp2.html
        ///     <para>One signature, NumPy's ufunc form: the one-argument overload went with API-parity batch 3 (2026-11-11), so
        ///     <c>np.exp2(x)</c> binds here with the defaults (the same loop, bits and dtype). A method group needs a lambda now
        ///     (<c>xs.Select(x => np.exp2(x))</c>), as for every other ufunc.</para>
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray exp2(NDArray a, NDArray @out = null, NDArray where = null, DType dtype = null)
            => a.TensorEngine.Exp2(a, dtype, @out, where);

        /// <summary>
        ///     Mirrors NumPy's ufunc signature: <c>expm1(x, /, out=None, *, where=True, dtype=None)</c>.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="out">A location into which the result is stored (joins the broadcast without being stretched, must be same_kind-castable from the loop dtype; returned as-is).</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written (NumPy ufunc where=).</param>
        /// <param name="dtype">Explicit loop dtype (NumPy ufunc dtype=): selects the loop; the input must be same_kind-castable to it.</param>
        /// <returns><c>exp(x) - 1</c> element-wise, accurate near zero, or <paramref name="out"/> itself when given.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.expm1.html
        ///     <para>One signature, NumPy's ufunc form: the one-argument overload went with API-parity batch 3 (2026-11-11), so
        ///     <c>np.expm1(x)</c> binds here with the defaults (the same loop, bits and dtype). A method group needs a lambda now
        ///     (<c>xs.Select(x => np.expm1(x))</c>), as for every other ufunc.</para>
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray expm1(NDArray a, NDArray @out = null, NDArray where = null, DType dtype = null)
            => a.TensorEngine.Expm1(a, dtype, @out, where);
    }
}
