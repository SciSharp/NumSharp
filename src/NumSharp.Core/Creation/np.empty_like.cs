using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return a new array with the same shape and type as a given array.
        /// </summary>
        /// <param name="prototype">The shape and data-type of prototype define these same attributes of the returned array.</param>
        /// <param name="dtype">
        ///     Overrides the dtype of the result — one descriptor parameter, like NumPy's <c>dtype</c>: a C# <see cref="Type"/>,
        ///     an <see cref="NPTypeCode"/>, a NumPy dtype string (<c>"f4"</c>) or a <see cref="DType"/> all convert implicitly.
        /// </param>
        /// <param name="shape">Overrides the shape of the result.</param>
        /// <returns>Array of uninitialized (arbitrary) data with the same shape and type as prototype.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.empty_like.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray empty_like(NDArray prototype, DType dtype = null, Shape shape = default, string device = null)
            => NDMatrixBoundary.Sees(prototype)   // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            ? NDMatrixBoundary.Wrapped((prototype, dtype, shape, device), static s => empty_like(s.prototype, s.dtype, s.shape, s.device))
            : empty_like(prototype, dtype, shape, 'K', device);

        /// <summary>
        ///     Return a new array with the same shape and type as a given array.
        /// </summary>
        /// <param name="prototype">The shape and data-type of prototype define these same attributes of the returned array.</param>
        /// <param name="dtype">Overrides the dtype of the result (a <see cref="Type"/>, <see cref="NPTypeCode"/>, dtype string or <see cref="DType"/> — all convert implicitly).</param>
        /// <param name="shape">Overrides the shape of the result.</param>
        /// <param name="order">Memory layout: 'C', 'F', 'A' or 'K' (default, preserves prototype layout).</param>
        /// <param name="device">Target device. Only <c>"cpu"</c> and <c>null</c> are accepted (Array-API parity).</param>
        /// <returns>Array of uninitialized (arbitrary) data with the same shape and type as prototype.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.empty_like.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray empty_like(NDArray prototype, DType dtype, Shape shape, char order, string device = null)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(prototype))
                return NDMatrixBoundary.Wrapped((prototype, dtype, shape, order, device), static s => empty_like(s.prototype, s.dtype, s.shape, s.order, s.device));
            ValidateDevice(device);
            var dims = shape.IsEmpty ? (long[])prototype.Shape.dimensions.Clone() : (long[])shape;
            // PyArray_NewLikeArrayWithShape's layout: order='K' keeps the prototype's axis permutation (C when shape=
            // changes the rank), not merely C-or-F.
            var resolvedShape = Shape.NewLikeLayout(prototype.Shape, order, dims);
            return new NDArray(dtype ?? prototype.dtype, resolvedShape, false);
        }
    }
}
