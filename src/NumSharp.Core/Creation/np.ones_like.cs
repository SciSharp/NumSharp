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
        /// <param name="device">Target device. Only <c>"cpu"</c> and <c>null</c> are accepted (Array-API parity).</param>
        /// <returns>Array of ones with the same shape and type as <paramref name="a"/>, in <paramref name="a"/>'s memory
        ///     order (<c>order='K'</c>). A <c>numpy.char.chararray</c> operand gives a chararray.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ones_like.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="ValueError"><paramref name="device"/> is neither null nor <c>"cpu"</c>; a chararray operand with a
        ///     dtype that is not string data (<c>Can only create a chararray from string data.</c>).</exception>
        /// <exception cref="NullReferenceException"><paramref name="a"/> is null (its layout is read after the order and device checks).</exception>
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
        /// <returns>Array of ones with the same shape and type as `nd`.
        ///     A <c>numpy.char.chararray</c> operand gives a chararray (NumPy allocates the result as the
        ///     subclass).</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ones_like.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="ValueError">A <c>numpy.char.chararray</c> operand with a dtype that is not string data:
        ///     NumPy's finalize error (<c>Can only create a chararray from string
        ///     data.</c>), before anything is filled.</exception>
        /// <exception cref="ValueError"><paramref name="order"/> is not one of C, F, A, K in either case (<c>order must be one
        ///     of 'C', 'F', 'A', or 'K' (got 'Z')</c>), reported before the device, as NumPy's empty_like converts its
        ///     arguments in order.</exception>
        /// <exception cref="ValueError"><paramref name="device"/> is neither null nor <c>"cpu"</c> (NumPy's <c>Device not
        ///     understood. Only "cpu" is allowed, but received: ...</c>).</exception>
        /// <exception cref="NullReferenceException"><paramref name="a"/> is null (its layout is read after the order and device checks).</exception>
        public static NDArray ones_like(NDArray a, DType dtype, char order, string device = null)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Wrapped((a, dtype, order, device), static s => ones_like(s.a, s.dtype, s.order, s.device));
            // a numpy.char.chararray operand: NumPy's res = empty_like(a, ...) allocates the result as the subclass, so its
            // dtype meets __array_finalize__ after the order and device checks and BEFORE the fill (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(a))
            {
                Shape.NewLikeLayout(a.Shape, order, (long[])a.Shape.dimensions.Clone());   // the order check, before the device
                DeviceArguments.ValidateDevice(device);
                NDCharArrayBoundary.RequireStringResult(dtype ?? a.dtype);
                return NDCharArrayBoundary.Kept((a, dtype, order, device), static s => ones_like(s.a, s.dtype, s.order, s.device));
            }
            // NumPy's statements (empty_like's argument checks and layout, then the fill with 1) are the array object's.
            return NDArray.ones_like(a, dtype, order, device);
        }
    }
}
