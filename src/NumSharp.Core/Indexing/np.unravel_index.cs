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
        /// <exception cref="ArgumentException"><paramref name="shape"/> is empty, or <paramref name="order"/> is invalid.</exception>
        /// <exception cref="ValueError">The dims' product overflows int64 or is exactly -1 (NumPy's <c>dimensions are too
        ///     large; …</c>).</exception>
        /// <exception cref="TypeError"><paramref name="indices"/> is neither integer nor bool (<c>only int indices
        ///     permitted</c>; floats are refused, never truncated, and text is refused, never parsed).</exception>
        /// <exception cref="ArgumentOutOfRangeException">
        ///     Any index in <paramref name="indices"/> is &lt; 0 or
        ///     &gt;= product of <paramref name="shape"/>.
        /// </exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.unravel_index.html
        ///     <para>Scoped by <c>[NDScoped]</c>: the int64 / C-contiguous copy of <paramref name="indices"/> is released
        ///     on return, exceptions included, and the coordinate arrays are yielded. The body no longer tracks
        ///     ownership of that copy with a flag and two hand-written disposals.</para>
        /// </remarks>
        [NDScoped]
        public static NDArray<long>[] unravel_index(NDArray indices, int[] shape, char order = 'C')
        {
            using var plain = NDMatrixBoundary.PlainFor(indices);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            ValidateOrder(order, nameof(order));
            ValidateShapeForUnravel(shape, out long unravelSize);
            ThrowIfNotIntIndices(indices);

            int ndim = shape.Length;

            // Cast indices to int64 contig. astype with copy: false returns the same
            // NDArray if dtype already matches AND it's contig; otherwise it materializes.
            // A materialized copy is released by this method's scope (exceptions included) and is
            // kept reachable by it while the kernel reads it; an `indices` used as-is is the caller's
            // (built before the scope opened) and is never disposed.
            // C order on every route: the kernel reads the indices linearly beside C-order results, and astype's default
            // 'K' kept an F-order int16 / int32 index array F-ordered (its coordinates came back in memory order).
            var idx64 = indices.GetTypeCode == NPTypeCode.Int64
                ? (indices.Shape.IsContiguous ? indices : np.ascontiguousarray(indices))
                : indices.astype(NPTypeCode.Int64, copy: true, order: 'C');

            // Result NDArrays have the same DIMENSIONS as the input but with fresh
            // C-contiguous strides. Cloning indices.Shape directly would copy
            // broadcast / strided / sliced flags that, on read via
            // Shape.TransformOffset, would collapse the result to the unbroadcast
            // cell — the kernel writes correct sequential values to raw memory
            // but the user would see duplicates.
            var resultShape = Shape.Adopt((long[])indices.Shape.dimensions.Clone());
            var result = new NDArray<long>[ndim];
            for (int d = 0; d < ndim; d++)
                result[d] = new NDArray<long>(resultShape);

            long count = indices.size;
            if (count == 0)
            {
                return result;
            }

            ExecuteUnravelIndex(idx64, shape, unravelSize, result, order);

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

        /// <summary>
        ///     The shape checks of NumPy's <c>unravel_index</c>: at least one dimension (a NumSharp restriction), then the
        ///     product of the dimensions as <c>PyArray_OverflowMultiplyList</c> computes it.
        /// </summary>
        /// <param name="shape">The shape.</param>
        /// <param name="unravelSize">The product (0 as soon as a dimension is 0; negative dimensions multiply in).</param>
        /// <exception cref="ArgumentException"><paramref name="shape"/> is null or empty.</exception>
        /// <exception cref="ValueError">The product overflows int64, or is exactly -1 — NumPy's overflow sentinel, so shape
        ///     <c>(-1,)</c> is refused too (<c>dimensions are too large; …</c>).</exception>
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
            // NumPy's PyArray_OverflowMultiplyList: a zero dimension ends the product at 0 (later dimensions are not even
            // multiplied); an overflow returns -1 — and so does a product that IS -1 (shape (-1,)), which NumPy cannot tell
            // apart and refuses with the same text.
            long s = 1L;
            bool overflow = false;
            for (int d = 0; d < shape.Length; d++)
            {
                long dim = shape[d];
                if (dim == 0)
                {
                    s = 0;
                    break;
                }
                long next = unchecked(s * dim);
                if (next / dim != s)
                {
                    overflow = true;
                    break;
                }
                s = next;
            }
            if (overflow || s == -1)
                throw new ValueError(
                    "dimensions are too large; arrays and shapes with a total size greater than 'intp' are not supported.");
            unravelSize = s;
        }

        /// <summary>
        ///     NumPy's <c>astype_anyint</c>: index arrays must hold integers or bools — a float, complex, decimal or text array
        ///     is refused rather than truncated or parsed.
        /// </summary>
        /// <param name="indices">The index (or coordinate) array.</param>
        /// <exception cref="TypeError">Not integer or bool: <c>only int indices permitted</c>.</exception>
        /// <remarks>
        ///     NumSharp used to cast any dtype to int64 (<c>unravel_index(np.array(new[] { 1.7 }), …)</c> read index 1) and to
        ///     defer text to the next phase; NumPy refuses both before reading an element. NumSharp's char counts as an
        ///     integer (its uint16 storage).
        /// </remarks>
        internal static void ThrowIfNotIntIndices(NDArray indices)
        {
            NPTypeCode t = indices.typecode;
            if (t == NPTypeCode.Boolean || t == NPTypeCode.Char || t.IsInteger())
                return;
            throw new TypeError("only int indices permitted");
        }
    }
}
