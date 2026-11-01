using System;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return a full array with the same shape and type as a given array.
        /// </summary>
        /// <param name="a">The shape and data-type of a define these same attributes of the returned array.</param>
        /// <param name="fill_value">Fill value: a scalar, or an array that broadcasts into the result.</param>
        /// <param name="dtype">Overrides the data type of the result.</param>
        /// <param name="device">Target device. Only <c>"cpu"</c> and <c>null</c> are accepted (Array-API parity).</param>
        /// <returns>Array of fill_value with the same shape and type as a.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.full_like.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>);
        ///     or an array fill value that does not broadcast into the result.</exception>
        public static NDArray full_like(NDArray a, object fill_value, DType dtype = null, string device = null)
            => NDMatrixBoundary.Sees(a)   // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            ? NDMatrixBoundary.Wrapped((a, fill_value, dtype, device), static s => full_like(s.a, s.fill_value, s.dtype, s.device))
            : full_like(a, fill_value, dtype, 'K', device);

        /// <summary>
        ///     Return a full array with the same shape and type as a given array.
        /// </summary>
        /// <param name="a">The shape and data-type of a define these same attributes of the returned array.</param>
        /// <param name="fill_value">Fill value: a scalar, or an array (an <see cref="NDArray"/> of rank ≥ 1, a C# array) that
        ///     broadcasts into the result.</param>
        /// <param name="dtype">Overrides the data type of the result.</param>
        /// <param name="order">Memory layout: 'C', 'F', 'A' or 'K' (default, preserves source layout).</param>
        /// <param name="device">Target device. Only <c>"cpu"</c> and <c>null</c> are accepted (Array-API parity).</param>
        /// <returns>Array of fill_value with the same shape and type as a.
        ///     A <c>numpy.char.chararray</c> operand gives a chararray (NumPy allocates the result as the
        ///     subclass).</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.full_like.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="ValueError">A <c>numpy.char.chararray</c> operand with a dtype that is not string data:
        ///     NumPy's finalize error (<c>Can only create a chararray from string
        ///     data.</c>), before anything is filled; or an array fill value that does not broadcast into the result, or
        ///     whose text does not parse into a numeric result.</exception>
        /// <exception cref="ValueError"><paramref name="order"/> is not one of C, F, A, K in either case (<c>order must be one
        ///     of 'C', 'F', 'A', or 'K' (got 'Z')</c>), reported before the device, as NumPy's empty_like converts its
        ///     arguments in order.</exception>
        /// <exception cref="ValueError"><paramref name="device"/> is neither null nor <c>"cpu"</c> (NumPy's <c>Device not
        ///     understood. Only "cpu" is allowed, but received: ...</c>).</exception>
        public static NDArray full_like(NDArray a, object fill_value, DType dtype, char order, string device = null)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Wrapped((a, fill_value, dtype, order, device), static s => full_like(s.a, s.fill_value, s.dtype, s.order, s.device));
            // a numpy.char.chararray operand: NumPy's res = empty_like(a, ...) allocates the result as the subclass, so its
            // dtype meets __array_finalize__ after the order and device checks and BEFORE the fill (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(a))
            {
                Shape.NewLikeLayout(a.Shape, order, (long[])a.Shape.dimensions.Clone());   // the order check, before the device
                DeviceArguments.ValidateDevice(device);
                NDCharArrayBoundary.RequireStringResult(dtype ?? a.dtype);
                return NDCharArrayBoundary.Kept((a, fill_value, dtype, order, device), static s => full_like(s.a, s.fill_value, s.dtype, s.order, s.device));
            }
            // NumPy's empty_like converts its arguments in order (npy_parse_arguments): the order's letter before the device,
            // so an unknown order is reported first (np.zeros_like(a, order='Z', device='cuda') raises the order's ValueError,
            // probed 2.4.2). The layout itself is resolved after the device, as before.
            OrderResolver.Convert(order);
            DeviceArguments.ValidateDevice(device);
            return NDArray.full_like(a, fill_value, dtype, order);
        }
    }
}
