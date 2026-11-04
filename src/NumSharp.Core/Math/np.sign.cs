using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Returns an element-wise indication of the sign of a number. <br></br>
        ///     The sign function returns -1 if x &lt; 0, 0 if x==0, 1 if x &gt; 0. nan is returned for nan inputs.
        /// </summary>
        /// <param name="x">Input values.</param>
        /// <param name="out">A location into which the result is stored (NumPy ufunc out=); <c>null</c> allocates the result.</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written (NumPy ufunc where=).</param>
        /// <param name="dtype">
        ///     The loop dtype (NumPy's <c>dtype=</c>): <paramref name="x"/> must cast to it under <c>same_kind</c>, so a float
        ///     input with an integer dtype is refused. <c>null</c> keeps the input's dtype (the sign of an integer array is
        ///     an integer array).
        /// </param>
        /// <returns>The sign of x, in the input's dtype unless <paramref name="dtype"/> names another. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.sign.html</remarks>
        /// <exception cref="ArgumentException">A <paramref name="dtype"/> the input cannot reach under <c>same_kind</c> (<c>Cannot cast
        ///     ufunc 'sign' input from dtype('float64') to dtype('int64') with casting rule 'same_kind'</c>; NumPy raises
        ///     <c>UFuncTypeError</c>).</exception>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray sign(NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null)
            => NDMatrixBoundary.Sees(x)   // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            ? NDMatrixBoundary.Wrapped((x, @out, where, dtype), static s => sign(s.x, s.@out, s.where, s.dtype), @out)
            : x.TensorEngine.Sign(x, dtype, @out, where);
    }
}
