using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Return the reciprocal of the argument, element-wise.
        /// Calculates 1/x.
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of.</param>
        /// <returns>Return array containing 1/x for each element in x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.reciprocal.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        [NDScoped]
        public static NDArray reciprocal(NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null)
            => NDMatrixBoundary.Sees(x)   // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            ? NDMatrixBoundary.Wrapped((x, @out, where, dtype), static s => reciprocal(s.x, s.@out, s.where, s.dtype), @out)
            : x.TensorEngine.Reciprocal(x, dtype, @out, where);
    }
}
