using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {

        /// <summary>
        ///     Calculate the absolute value element-wise.
        ///     Mirrors NumPy's ufunc signature: <c>absolute(x, /, out=None, *, where=True, dtype=None)</c>.
        /// </summary>
        /// <param name="a">Input value.</param>
        /// <param name="out">A location into which the result is stored (joins the broadcast without being stretched, must be same_kind-castable from the loop dtype; returned as-is).</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written (NumPy ufunc where=).</param>
        /// <param name="dtype">Explicit loop dtype (NumPy ufunc dtype=): the input must be same_kind-castable to it; for complex input it selects the magnitude dtype (float kinds only).</param>
        /// <returns>The absolute value of each element, in the input's dtype (a complex input gives its float magnitude), or <paramref name="out"/> itself when given.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.absolute.html
        ///     <para>One signature, NumPy's ufunc form: the one-argument overload went with API-parity batch 3 (2026-11-11), so
        ///     <c>np.absolute(x)</c> binds here with the defaults (the same loop, bits and dtype). A method group needs a lambda now
        ///     (<c>xs.Select(x => np.absolute(x))</c>), as for every other ufunc.</para>
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray absolute(NDArray a, NDArray @out = null, NDArray where = null, DType dtype = null)
            => a.TensorEngine.Abs(a, dtype, @out, where);

        /// <summary>
        ///     Calculate the absolute value element-wise (alias of <see cref="absolute(NDArray, NDArray, NDArray, DType)"/>).
        ///     Mirrors NumPy's ufunc signature: <c>absolute(x, /, out=None, *, where=True, dtype=None)</c>.
        /// </summary>
        /// <param name="a">Input value.</param>
        /// <param name="out">A location into which the result is stored (joins the broadcast without being stretched, must be same_kind-castable from the loop dtype; returned as-is).</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written (NumPy ufunc where=).</param>
        /// <param name="dtype">Explicit loop dtype (NumPy ufunc dtype=): the input must be same_kind-castable to it; for complex input it selects the magnitude dtype (float kinds only).</param>
        /// <returns>The absolute value of each element, in the input's dtype (a complex input gives its float magnitude), or <paramref name="out"/> itself when given.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.absolute.html
        ///     <para>One signature, NumPy's ufunc form: the one-argument overload went with API-parity batch 3 (2026-11-11), so
        ///     <c>np.abs(x)</c> binds here with the defaults (the same loop, bits and dtype). A method group needs a lambda now
        ///     (<c>xs.Select(x => np.abs(x))</c>), as for every other ufunc.</para>
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray abs(NDArray a, NDArray @out = null, NDArray where = null, DType dtype = null)
            => a.TensorEngine.Abs(a, dtype, @out, where);
    }
}
