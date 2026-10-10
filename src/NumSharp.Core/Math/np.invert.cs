using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Compute bit-wise inversion, or bit-wise NOT, element-wise.
        /// Computes the bit-wise NOT of the underlying binary representation of
        /// the integers in the input arrays.For signed integer inputs, the
        /// two's complement is returned. In a two's-complement system negative
        /// numbers are represented by the two's complement of the absolute value.
        /// </summary>
        /// <param name="x">Only integer and boolean types are handled.</param>
        /// <param name="outType">The dtype the returned ndarray should be of.</param>
        /// <returns>Result. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.invert.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray invert(NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null)
            => NDMatrixBoundary.Sees(x)   // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            ? NDMatrixBoundary.Wrapped((x, @out, where, dtype), static s => invert(s.x, s.@out, s.where, s.dtype), @out)
            : x.TensorEngine.Invert(x, dtype, @out, where);

        /// <summary>
        /// Compute bit-wise inversion, or bit-wise NOT, element-wise. Alias for invert.
        /// </summary>
        /// <param name="x">Only integer and boolean types are handled.</param>
        /// <param name="outType">The dtype the returned ndarray should be of.</param>
        /// <returns>Result. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.invert.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray bitwise_not(NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null)
            => NDMatrixBoundary.Sees(x)   // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            ? NDMatrixBoundary.Wrapped((x, @out, where, dtype), static s => bitwise_not(s.x, s.@out, s.where, s.dtype), @out)
            : invert(x, @out, where, dtype);
    }
}
