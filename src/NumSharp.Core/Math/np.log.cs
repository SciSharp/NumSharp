using System;
using NumSharp.Backends;

namespace NumSharp
{
    public partial class np
    {

        /// <summary>
        ///     Natural logarithm, element-wise.
        ///     Mirrors NumPy's ufunc signature: <c>log(x, /, out=None, *, where=True, dtype=None)</c>.
        /// </summary>
        /// <param name="x">Input value.</param>
        /// <param name="out">A location into which the result is stored (joins the broadcast without being stretched, must be same_kind-castable from the loop dtype; returned as-is).</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written (NumPy ufunc where=).</param>
        /// <param name="dtype">Explicit loop dtype (NumPy ufunc dtype=): the computation runs at this precision; integer/bool requests raise NumPy's "No loop matching" error.</param>
        /// <returns>The natural logarithm of each element (NaN for a negative real input), or <paramref name="out"/> itself when given.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.log.html
        ///     <para>One signature, NumPy's ufunc form: the one-argument overload went with API-parity batch 3 (2026-11-11), so
        ///     <c>np.log(x)</c> binds here with the defaults (the same loop, bits and dtype). A method group needs a lambda now
        ///     (<c>xs.Select(x => np.log(x))</c>), as for every other ufunc.</para>
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray log(NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null)
            => x.TensorEngine.Log(x, dtype, @out, where);

        /// <summary>
        ///     Base-2 logarithm of x.
        /// </summary>
        /// <param name="x">Input value.</param>
        /// <param name="out">A location into which the result is stored (joins the broadcast without being stretched, must be same_kind-castable from the loop dtype; returned as-is), or null for a fresh result.</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed / written (NumPy ufunc <c>where=</c>); null is NumPy's <c>where=True</c>.</param>
        /// <param name="dtype">Explicit loop dtype (NumPy ufunc <c>dtype=</c>): the input must be same_kind-castable to it.</param>
        /// <returns>Base-2 logarithm of x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.log2.html
        ///     <para>One signature, NumPy's ufunc form: the one-argument overload went with API-parity batch 3 (2026-11-11), so
        ///     <c>np.log2(x)</c> binds here with the defaults (the same loop, bits and dtype). A method group needs a lambda now
        ///     (<c>xs.Select(x => np.log2(x))</c>), as for every other ufunc.</para>
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray log2(NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null) => x.TensorEngine.Log2(x, dtype, @out, where);

        /// <summary>
        ///     Return the base 10 logarithm of the input array, element-wise.
        /// </summary>
        /// <param name="x">Input value.</param>
        /// <param name="out">A location into which the result is stored (joins the broadcast without being stretched, must be same_kind-castable from the loop dtype; returned as-is), or null for a fresh result.</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed / written (NumPy ufunc <c>where=</c>); null is NumPy's <c>where=True</c>.</param>
        /// <param name="dtype">Explicit loop dtype (NumPy ufunc <c>dtype=</c>): the input must be same_kind-castable to it.</param>
        /// <returns>The logarithm to the base 10 of x, element-wise. NaNs are returned where x is negative. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.log10.html
        ///     <para>One signature, NumPy's ufunc form: the one-argument overload went with API-parity batch 3 (2026-11-11), so
        ///     <c>np.log10(x)</c> binds here with the defaults (the same loop, bits and dtype). A method group needs a lambda now
        ///     (<c>xs.Select(x => np.log10(x))</c>), as for every other ufunc.</para>
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray log10(NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null) => x.TensorEngine.Log10(x, dtype, @out, where);

        /// <summary>
        ///     Return the natural logarithm of one plus the input array, element-wise.<br></br>
        ///     Calculates log(1 + x).
        /// </summary>
        /// <param name="x">Input value.</param>
        /// <param name="out">A location into which the result is stored (joins the broadcast without being stretched, must be same_kind-castable from the loop dtype; returned as-is), or null for a fresh result.</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed / written (NumPy ufunc <c>where=</c>); null is NumPy's <c>where=True</c>.</param>
        /// <param name="dtype">Explicit loop dtype (NumPy ufunc <c>dtype=</c>): the input must be same_kind-castable to it.</param>
        /// <returns>Natural logarithm of 1 + x, element-wise. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.log1p.html
        ///     <para>One signature, NumPy's ufunc form: the one-argument overload went with API-parity batch 3 (2026-11-11), so
        ///     <c>np.log1p(x)</c> binds here with the defaults (the same loop, bits and dtype). A method group needs a lambda now
        ///     (<c>xs.Select(x => np.log1p(x))</c>), as for every other ufunc.</para>
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray log1p(NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null) => x.TensorEngine.Log1p(x, dtype, @out, where);
    }
}
