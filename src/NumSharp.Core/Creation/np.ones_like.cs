using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return an array of ones with the same shape and type as a given array.
        /// </summary>
        /// <param name="a">Array of ones with the same shape and type as a.</param>
        /// <param name="dtype">Overrides the data type of the result.</param>
        /// <returns>Array of zeros with the same shape and type as `nd`.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ones_like.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray ones_like(NDArray a, DType dtype = null, string device = null) => NDMatrixBoundary.Sees(a)   // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            ? NDMatrixBoundary.Wrapped((a, dtype, device), static s => ones_like(s.a, s.dtype, s.device))
            : ones_like(a, dtype, 'K', device);

        /// <summary>
        ///     Return an array of ones with the same shape and type as a given array.
        /// </summary>
        /// <param name="a">Array of ones with the same shape and type as a.</param>
        /// <param name="dtype">Overrides the data type of the result.</param>
        /// <param name="order">Memory layout: 'C', 'F', 'A' or 'K' (default, preserves source layout).</param>
        /// <param name="device">Target device. Only <c>"cpu"</c> and <c>null</c> are accepted (Array-API parity).</param>
        /// <returns>Array of ones with the same shape and type as `nd`.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ones_like.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray ones_like(NDArray a, DType dtype, char order, string device = null)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Wrapped((a, dtype, order, device), static s => ones_like(s.a, s.dtype, s.order, s.device));
            // NumPy computes this on text (sorting, searching, truth tests); string computation is
            // the next phase of NumSharp's string support, so a string array is refused here.
            if (Backends.Strings.NDStringUfuncs.IsString(a)) throw Backends.Strings.NDStringUfuncs.NextPhase("ones_like", a);
            ValidateDevice(device);
            // PyArray_NewLikeArray's layout: order='K' keeps a's axis permutation, not merely C-or-F.
            var resolvedShape = Shape.NewLikeLayout(a.Shape, order, (long[])a.Shape.dimensions.Clone());
            return np.ones(resolvedShape, dtype ?? a.dtype);
        }
    }
}
