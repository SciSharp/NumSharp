using System;
using NumSharp.Backends.Kernels;
using NumSharp.Generic;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Converts a flat index or array of flat indices into a tuple of coordinate
        ///     arrays. Inverse of <see cref="ravel_multi_index"/>.
        /// </summary>
        /// <param name="indices">
        ///     An integer array whose elements are indices into the flattened version of
        ///     an array of dimensions <paramref name="shape"/>. Cast to int64 internally.
        /// </param>
        /// <param name="shape">The shape of the array to use for unraveling.</param>
        /// <param name="order">
        ///     <c>'C'</c> (row-major, default) or <c>'F'</c> (column-major) — selects the
        ///     extraction order for the coordinate tuple.
        /// </param>
        /// <returns>
        ///     A tuple of <c>shape.Length</c> NDArrays. Each output array has the same
        ///     shape as <paramref name="indices"/>. Element dtype is always
        ///     <see cref="NPTypeCode.Int64"/>.
        /// </returns>
        /// <exception cref="ArgumentException">
        ///     <paramref name="shape"/> is empty, has non-positive dims, or the dims'
        ///     product overflows int64.
        /// </exception>
        /// <exception cref="ArgumentOutOfRangeException">
        ///     Any index in <paramref name="indices"/> is &lt; 0 or
        ///     &gt;= product of <paramref name="shape"/>.
        /// </exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.unravel_index.html</remarks>
        public static NDArray<long>[] unravel_index(NDArray indices, int[] shape, char order = 'C')
        {
            ValidateOrder(order, nameof(order));
            ValidateShapeForUnravel(shape, out long unravelSize);

            int ndim = shape.Length;

            // Cast indices to int64 contig. astype with copy: false returns the same
            // NDArray if dtype already matches AND it's contig; otherwise it materializes.
            var idx64 = indices.GetTypeCode == NPTypeCode.Int64
                ? (indices.Shape.IsContiguous ? indices : np.ascontiguousarray(indices))
                : indices.astype(NPTypeCode.Int64);

            // ARC: idx64 may alias `indices` (no-op cast on contig int64); the explicit
            // Dispose below only triggers when astype/ascontiguousarray actually allocated.
            bool ownIdx64 = !ReferenceEquals(idx64, indices);

            // Result NDArrays have the same DIMENSIONS as the input but with fresh
            // C-contiguous strides. Cloning indices.Shape directly would copy
            // broadcast / strided / sliced flags that, on read via
            // Shape.TransformOffset, would collapse the result to the unbroadcast
            // cell — the kernel writes correct sequential values to raw memory
            // but the user would see duplicates.
            var resultShape = new Shape((long[])indices.Shape.dimensions.Clone());
            var result = new NDArray<long>[ndim];
            for (int d = 0; d < ndim; d++)
                result[d] = new NDArray<long>(resultShape);

            long count = indices.size;
            if (count == 0)
            {
                if (ownIdx64) idx64.Dispose();
                return result;
            }

            try
            {
                ExecuteUnravelIndex(idx64, shape, unravelSize, result, order);
            }
            finally
            {
                if (ownIdx64) idx64.Dispose();
            }

            return result;
        }

        /// <summary>
        ///     Scalar convenience overload — converts a single flat index into a coord
        ///     array. Equivalent to <c>unravel_index(NDArray.Scalar(index), shape, order)</c>
        ///     but returns a <c>long[]</c> directly without NDArray wrapping.
        /// </summary>
        public static long[] unravel_index(long index, int[] shape, char order = 'C')
        {
            ValidateOrder(order, nameof(order));
            ValidateShapeForUnravel(shape, out long unravelSize);

            if (index < 0 || index >= unravelSize)
                throw new ArgumentOutOfRangeException(nameof(index),
                    $"index {index} is out of bounds for array with size {unravelSize}");

            int ndim = shape.Length;
            var coords = new long[ndim];

            if (order == 'C')
            {
                // C-order: from innermost dim down to outermost; skip last divmod
                // because val < dims[0] after extracting the previous coords.
                long val = index;
                for (int d = ndim - 1; d > 0; d--)
                {
                    long m = shape[d];
                    coords[d] = val % m;
                    val /= m;
                }
                coords[0] = val;
            }
            else
            {
                long val = index;
                for (int d = 0; d < ndim - 1; d++)
                {
                    long m = shape[d];
                    coords[d] = val % m;
                    val /= m;
                }
                coords[ndim - 1] = val;
            }

            return coords;
        }

        private static unsafe void ExecuteUnravelIndex(
            NDArray idx64, int[] shape, long unravelSize,
            NDArray<long>[] result, char order)
        {
            int ndim = shape.Length;
            long count = idx64.size;

            var kernel = DirectILKernelGenerator.GetUnravelIndexKernel();
            if (kernel == null)
                throw new NotSupportedException("np.unravel_index: IL kernel unavailable");

            // Stack-allocate dims as long[] and grab per-axis result pointers.
            Span<long> dims = stackalloc long[ndim];
            for (int d = 0; d < ndim; d++)
                dims[d] = shape[d];

            long** outCols = stackalloc long*[ndim];
            for (int d = 0; d < ndim; d++)
                outCols[d] = (long*)result[d].Address;

            long idxStart = order == 'C' ? ndim - 1 : 0;
            long idxStep = order == 'C' ? -1 : 1;

            long* idxPtr = (long*)idx64.Storage.Address + idx64.Shape.offset;

            fixed (long* dimsPtr = dims)
            {
                long status = kernel(idxPtr, count, dimsPtr, unravelSize, outCols, ndim, idxStart, idxStep);
                if (status < count)
                {
                    long badPos = status;
                    long badVal = idxPtr[badPos];
                    throw new ArgumentOutOfRangeException(nameof(idx64),
                        $"index {badVal} is out of bounds for array with size {unravelSize}");
                }
            }
        }

        private static void ValidateOrder(char order, string paramName)
        {
            if (order != 'C' && order != 'F')
                throw new ArgumentException($"only 'C' or 'F' order is permitted, got '{order}'", paramName);
        }

        private static void ValidateShapeForUnravel(int[] shape, out long unravelSize)
        {
            if (shape == null || shape.Length == 0)
                throw new ArgumentException("shape must contain at least one dimension.", nameof(shape));

            // NumPy parity: do NOT pre-reject zero or negative dims. Empty input
            // arrays must work against any shape, including (0, 3). Non-empty input
            // arrays against zero/negative shape are rejected naturally by the
            // OOB check (val < 0 || val >= unravelSize) inside the kernel — for
            // shape with zero or negative entries, unravelSize ≤ 0, so every
            // non-negative index trips the check and we throw the same message
            // NumPy does: "index N is out of bounds for array with size M".
            long s = 1L;
            for (int d = 0; d < shape.Length; d++)
            {
                long dim = shape[d];

                // Overflow check on accumulated size — only meaningful for positive
                // dims (with mixed signs the int64 math is well-defined and the OOB
                // check downstream catches any inconsistency).
                long next = unchecked(s * dim);
                if (dim != 0 && s != 0 && next / dim != s)
                    throw new ArgumentException(
                        "invalid shape: array size defined by shape is larger than the maximum possible size.",
                        nameof(shape));
                s = next;
            }
            unravelSize = s;
        }
    }
}
