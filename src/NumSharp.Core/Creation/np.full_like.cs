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
        /// <param name="fill_value">Fill value.</param>
        /// <param name="dtype">Overrides the data type of the result.</param>
        /// <returns>Array of fill_value with the same shape and type as a.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.full_like.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray full_like(NDArray a, object fill_value, DType dtype = null, string device = null)
            => NDMatrixBoundary.Sees(a)   // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            ? NDMatrixBoundary.Wrapped((a, fill_value, dtype, device), static s => full_like(s.a, s.fill_value, s.dtype, s.device))
            : full_like(a, fill_value, dtype, 'K', device);

        /// <summary>
        ///     Return a full array with the same shape and type as a given array.
        /// </summary>
        /// <param name="a">The shape and data-type of a define these same attributes of the returned array.</param>
        /// <param name="fill_value">Fill value.</param>
        /// <param name="dtype">Overrides the data type of the result.</param>
        /// <param name="order">Memory layout: 'C', 'F', 'A' or 'K' (default, preserves source layout).</param>
        /// <param name="device">Target device. Only <c>"cpu"</c> and <c>null</c> are accepted (Array-API parity).</param>
        /// <returns>Array of fill_value with the same shape and type as a.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.full_like.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray full_like(NDArray a, object fill_value, DType dtype, char order, string device = null)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Wrapped((a, fill_value, dtype, order, device), static s => full_like(s.a, s.fill_value, s.dtype, s.order, s.device));
            ValidateDevice(device);
            // NumPy's *_like contract preserves a.dtype unless dtype= is explicitly supplied.
            // fill_value is CAST into that dtype; its CLR type never selects the result dtype.
            var typeCode = dtype != null ? dtype.GetTypeCode() : a.typecode;
            // PyArray_NewLikeArray's layout: order='K' keeps a's axis permutation, not merely C-or-F.
            var shape = Shape.NewLikeLayout(a.Shape, order, (long[])a.Shape.dimensions.Clone());

            // A string result (a's dtype, or a requested one — unsized means one character, NumPy's empty_like) is
            // `res = empty_like(...); copyto(res, fill_value, casting='unsafe')`: the copy truncates / converts the value.
            var resultDescr = dtype ?? a.dtype;
            if (Backends.Strings.NDStringStorage.IsString(resultDescr))
                return Backends.Strings.NDStringArrays.Full(shape, fill_value, resultDescr);

            // Allocates from shape.size directly (not via UnmanagedStorage.Allocate), so the
            // dimension guard has to be explicit here — see np.full.
            AllocationGuard.CheckDimensions(shape.dimensions, typeCode);

            // NumPy's `res = empty_like(...); copyto(res, fill_value, casting='unsafe')`: a Python int is range-checked in
            // the result's dtype (np.full_like(int8_array, 1000) raises), a Python float / complex cast unsafely.
            return new NDArray(new UnmanagedStorage(ArraySlice.Allocate(typeCode, shape.size, FullFillElement(fill_value, typeCode)), shape));
        }
    }
}
