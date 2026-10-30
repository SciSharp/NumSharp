using System;

namespace NumSharp
{
    public static partial class np
    {
        ///  <summary>
        ///     Return an array of zeros with the same shape and type as a given array.
        ///  </summary>
        /// <param name="a">The shape and data-type of a define these same attributes of the returned array.</param>
        ///  <param name="dtype">Overrides the data type of the result.</param>
        ///  <returns>Array of zeros with the same shape and type as `nd`.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.zeros_like.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray zeros_like(NDArray a, DType dtype = null, string device = null) => NDMatrixBoundary.Sees(a)   // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            ? NDMatrixBoundary.Wrapped((a, dtype, device), static s => zeros_like(s.a, s.dtype, s.device))
            : zeros_like(a, dtype, 'K', device);

        ///  <summary>
        ///     Return an array of zeros with the same shape and type as a given array.
        ///  </summary>
        /// <param name="a">The shape and data-type of a define these same attributes of the returned array.</param>
        ///  <param name="dtype">Overrides the data type of the result.</param>
        ///  <param name="order">Memory layout: 'C', 'F', 'A' or 'K' (default, preserves source layout).</param>
        ///  <param name="device">Target device. Only <c>"cpu"</c> and <c>null</c> are accepted (Array-API parity).</param>
        ///  <returns>Array of zeros with the same shape and type as `nd`.
        ///     A <c>numpy.char.chararray</c> operand gives a chararray (NumPy allocates the result as the
        ///     subclass).</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.zeros_like.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="ValueError">A <c>numpy.char.chararray</c> operand with a dtype that is not string data:
        ///     NumPy's finalize error (<c>Can only create a chararray from string
        ///     data.</c>), before anything is filled.</exception>
        /// <exception cref="ValueError"><paramref name="order"/> is not one of C, F, A, K in either case (<c>order must be one
        ///     of 'C', 'F', 'A', or 'K' (got 'Z')</c>), reported before the device, as NumPy's empty_like converts its
        ///     arguments in order.</exception>
        /// <exception cref="ValueError"><paramref name="device"/> is neither null nor <c>"cpu"</c> (NumPy's <c>Device not
        ///     understood. Only "cpu" is allowed, but received: ...</c>).</exception>
        public static NDArray zeros_like(NDArray a, DType dtype, char order, string device = null)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Wrapped((a, dtype, order, device), static s => zeros_like(s.a, s.dtype, s.order, s.device));
            // a numpy.char.chararray operand: NumPy's res = empty_like(a, ...) allocates the result as the subclass, so its
            // dtype meets __array_finalize__ after the order and device checks and BEFORE the fill (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(a))
            {
                Shape.NewLikeLayout(a.Shape, order, (long[])a.Shape.dimensions.Clone());   // the order check, before the device
                ValidateDevice(device);
                NDCharArrayBoundary.RequireStringResult(dtype ?? a.dtype);
                return NDCharArrayBoundary.Kept((a, dtype, order, device), static s => zeros_like(s.a, s.dtype, s.order, s.device));
            }
            // NumPy's empty_like converts its arguments in order (npy_parse_arguments): the order's letter before the device,
            // so an unknown order is reported first (np.zeros_like(a, order='Z', device='cuda') raises the order's ValueError,
            // probed 2.4.2). The layout itself is resolved after the device, as before.
            OrderResolver.Convert(order);
            ValidateDevice(device);
            // PyArray_NewLikeArray's layout: order='K' keeps a's axis permutation (a transposed a gives a transposed
            // result, a broadcast one an F result), not merely C-or-F.
            var resolvedShape = Shape.NewLikeLayout(a.Shape, order, (long[])a.Shape.dimensions.Clone());
            return np.zeros(resolvedShape, dtype ?? a.dtype);
        }
    }
}
