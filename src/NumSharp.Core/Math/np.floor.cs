using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return the floor of the input, element-wise. <br></br>
        ///     The floor of the scalar x is the largest integer i, such that i &lt;= x. It is often denoted as \lfloor x \rfloor.
        /// </summary>
        /// <param name="x">Input array</param>
        /// <param name="out">A location into which the result is stored (NumPy ufunc out=); <c>null</c> allocates the result.</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written (NumPy ufunc where=).</param>
        /// <param name="dtype">
        ///     The loop dtype (NumPy's <c>dtype=</c>): <paramref name="x"/> must cast to it under <c>same_kind</c>, so a float
        ///     input with an integer dtype is refused. <c>null</c> keeps the input's dtype (an integer input stays integer,
        ///     as in NumPy 2.x).
        /// </param>
        /// <returns>The floor of each element in x, in the input's dtype unless <paramref name="dtype"/> names another. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.floor.html</remarks>
        /// <exception cref="TypeError">A <paramref name="dtype"/> the input cannot reach under <c>same_kind</c> (<c>Cannot cast
        ///     ufunc 'floor' input from dtype('float64') to dtype('int64') with casting rule 'same_kind'</c>; NumPy raises
        ///     <c>UFuncTypeError</c>).</exception>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray floor(NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null)
            => x.TensorEngine.Floor(x, dtype, @out, where);
    }
}
