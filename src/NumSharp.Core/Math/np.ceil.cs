using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return the ceiling of the input, element-wise.<br></br>
        ///     The ceil of the scalar x is the smallest integer i, such that i >= x. It is often denoted as \lceil x \rceil.
        /// </summary>
        /// <param name="x">Input data.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>The ceiling of each element in x, with float dtype. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ceil.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray ceil(NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null)
            => NDMatrixBoundary.Sees(x)   // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            ? NDMatrixBoundary.Wrapped((x, @out, where, dtype), static s => ceil(s.x, s.@out, s.where, s.dtype), @out)
            : x.TensorEngine.Ceil(x, dtype, @out, where);
    }
}
