using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Return the truncated value of the input, element-wise.
        /// The truncated value of the scalar x is the nearest integer i which is closer to zero than x is.
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of.</param>
        /// <returns>The truncated value of each element in x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.trunc.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray trunc(NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null)
            => NDMatrixBoundary.Sees(x)   // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            ? NDMatrixBoundary.Wrapped((x, @out, where, dtype), static s => trunc(s.x, s.@out, s.where, s.dtype), @out)
            : x.TensorEngine.Truncate(x, dtype, @out, where);
    }
}
