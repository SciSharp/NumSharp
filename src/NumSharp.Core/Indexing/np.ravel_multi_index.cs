using System;
using NumSharp.Backends.Kernels;
using NumSharp.Generic;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Converts a tuple of coordinate arrays into an array of flat indices,
        ///     applying boundary modes per axis. Inverse of <see cref="unravel_index"/>.
        /// </summary>
        /// <param name="multi_index">
        ///     Tuple of integer arrays, one per dimension. All arrays must share the
        ///     same shape, which becomes the shape of the result.
        /// </param>
        /// <param name="dims">Shape of the array the indices are unravelling into.</param>
        /// <param name="mode">
        ///     Boundary mode: <c>"raise"</c> (default — throw on OOB), <c>"wrap"</c>
        ///     (modulo with sign correction), or <c>"clip"</c> (saturate). Applied to
        ///     every axis.
        /// </param>
        /// <param name="order">
        ///     <c>'C'</c> (row-major, default) or <c>'F'</c> (column-major). Selects the
        ///     stride ordering used to fold coordinates into a flat index.
        /// </param>
        /// <returns>1-D (or shape-preserving) <see cref="NDArray{T}"/> of <see cref="long"/>.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="multi_index"/> or <paramref name="dims"/> is null.</exception>
        /// <exception cref="ValueError">The coordinate count differs from <paramref name="dims"/>' length (<c>parameter
        ///     multi_index must be a sequence of length N</c>).</exception>
        /// <exception cref="TypeError">A coordinate array that is neither integer nor bool (<c>only int indices
        ///     permitted</c>).</exception>
        /// <exception cref="ValueError">A bad <paramref name="mode"/> (NumPy's <c>clipmode_parser</c> texts, see
        ///     <see cref="ClipModeConverter"/>), checked before the coordinate count.</exception>
        /// <exception cref="ArgumentException">An invalid order, coordinates that do not broadcast, or (mode
        ///     <c>raise</c>) a coordinate out of range.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ravel_multi_index.html</remarks>
        public static NDArray<long> ravel_multi_index(NDArray[] multi_index, int[] dims, string mode = "raise", char order = 'C')
        {
            using var plain = NDMatrixBoundary.PlainFor(multi_index);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            if (multi_index == null) throw new ArgumentNullException(nameof(multi_index));
            if (dims == null) throw new ArgumentNullException(nameof(dims));
            // arr_ravel_multi_index (compiled_base.c) converts the modes before it reads the coordinates
            // (PyArray_ConvertClipmodeSequence ahead of int_sequence_to_arrays): a bad mode wins over a coordinate count
            // that does not match. One mode string serves every dimension.
            var modes = new int[dims.Length];
            int singleMode = ClipModeConverter(mode);
            for (int d = 0; d < modes.Length; d++) modes[d] = singleMode;
            ValidateRavelInputs(multi_index, dims);

            return RavelMultiIndexImpl(multi_index, dims, modes, order);
        }

        /// <summary>
        ///     Per-axis mode overload. <paramref name="modes"/> length must match
        ///     <paramref name="dims"/> length.
        /// </summary>
        /// <param name="multi_index">Tuple of integer arrays, one per dimension.</param>
        /// <param name="dims">Shape of the array the indices are unravelling into.</param>
        /// <param name="modes">One boundary mode per dimension (NumPy's <c>mode</c> tuple); <c>null</c> is NumPy's
        ///     <c>None</c>, which raises on every dimension.</param>
        /// <param name="order"><c>'C'</c> (default) or <c>'F'</c>.</param>
        /// <returns>The flat indices.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="multi_index"/> or <paramref name="dims"/> is null.</exception>
        /// <exception cref="ValueError">A mode count that differs from the dimensions (<c>list of clipmodes has wrong length
        ///     (1 instead of 2)</c>) or a bad mode (<see cref="ClipModeConverter"/>), both checked before the coordinate
        ///     count; then a coordinate count that differs from <paramref name="dims"/>' length.</exception>
        /// <exception cref="TypeError">A coordinate array that is neither integer nor bool (<c>only int indices
        ///     permitted</c>).</exception>
        /// <exception cref="ArgumentException">An invalid order, coordinates that do not broadcast, or (mode
        ///     <c>raise</c>) a coordinate out of range.</exception>
        public static NDArray<long> ravel_multi_index(NDArray[] multi_index, int[] dims, string[] modes, char order = 'C')
        {
            using var plain = NDMatrixBoundary.PlainFor(multi_index);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            if (multi_index == null) throw new ArgumentNullException(nameof(multi_index));
            if (dims == null) throw new ArgumentNullException(nameof(dims));
            // NumPy's PyArray_ConvertClipmodeSequence: the tuple's length, then each mode in order — all before the
            // coordinates are read.
            var modeInts = ConvertClipmodeSequence(modes, dims.Length);
            ValidateRavelInputs(multi_index, dims);

            return RavelMultiIndexImpl(multi_index, dims, modeInts, order);
        }

        /// <summary>
        ///     Scalar convenience overload — folds a single coordinate tuple into a
        ///     flat index. Equivalent to wrapping each coord in a 0-d NDArray but
        ///     returns a <see cref="long"/> directly.
        /// </summary>
        public static long ravel_multi_index(long[] coords, int[] dims, string mode = "raise", char order = 'C')
        {
            if (coords == null) throw new ArgumentNullException(nameof(coords));
            if (dims == null) throw new ArgumentNullException(nameof(dims));
            // The mode first, as NumPy converts it before it reads the coordinates.
            int modeInt = ClipModeConverter(mode);
            if (coords.Length != dims.Length)
                throw new ArgumentException(
                    $"coords length ({coords.Length}) must match dims length ({dims.Length}).",
                    nameof(coords));

            int ndim = dims.Length;

            Span<long> ravelStrides = stackalloc long[ndim];
            ComputeRavelStrides(dims, order, ravelStrides);

            long raveled = 0;
            for (int d = 0; d < ndim; d++)
            {
                long j = coords[d];
                long m = dims[d];
                j = ApplyMode(j, m, modeInt, d);
                raveled += j * ravelStrides[d];
            }
            return raveled;
        }

        /// <summary>
        ///     The shared core of <c>ravel_multi_index</c>: NumPy's per-array integer check (<c>astype_anyint</c>), the broadcast
        ///     of the coordinate arrays, the int64 casts and the folding kernel.
        /// </summary>
        /// <param name="multi_index">The coordinate arrays (one per dimension; the count is already checked).</param>
        /// <param name="dims">The shape being indexed.</param>
        /// <param name="modes">The parsed per-axis modes.</param>
        /// <param name="order"><c>'C'</c> or <c>'F'</c>.</param>
        /// <returns>The flat indices, in the coordinates' broadcast shape.</returns>
        /// <exception cref="TypeError">A coordinate array that is neither integer nor bool (<c>only int indices permitted</c>:
        ///     a float, complex, decimal or text array — NumPy never truncates or parses coordinates).</exception>
        /// <exception cref="ArgumentException">An invalid order, coordinates that do not broadcast, or (mode <c>raise</c>) a
        ///     coordinate out of range.</exception>
        private static unsafe NDArray<long> RavelMultiIndexImpl(NDArray[] multi_index, int[] dims, int[] modes, char order)
        {
            ValidateOrder(order, nameof(order));
            foreach (var coords in multi_index)
                ThrowIfNotIntIndices(coords);

            int ndim = multi_index.Length;

            // NumPy broadcasts the coord arrays against each other before folding —
            // e.g. (np.array([1,2,3]), np.array(2)) is legal and the result has the
            // broadcast shape. Use the existing np.broadcast_arrays which throws a
            // ValueError-equivalent on incompatible shapes (matches NumPy's
            // "operands could not be broadcast together" diagnostic). For the
            // common case of all-same-shape coords this is a no-op view.
            var broadcasted = np.broadcast_arrays(multi_index);
            var refShape = broadcasted[0].Shape;

            long count = broadcasted[0].size;

            // CRITICAL: construct the result with fresh C-contiguous strides built
            // from the broadcast dimensions alone — using `refShape` directly would
            // inherit the broadcast view's stride=0 axes, and subsequent reads via
            // NDArray<T>.GetAtIndex (which translates through Shape.TransformOffset)
            // would collapse to the unbroadcast cell, returning duplicated values
            // even though the kernel wrote the correct sequence to raw memory.
            var resultShape = Shape.Adopt((long[])refShape.dimensions.Clone());
            var result = new NDArray<long>(resultShape);

            if (count == 0)
                return result;

            // Cast each broadcast result to contig int64. The broadcast views above
            // are non-contig stride-0 layouts, so the IsContiguous gate falls
            // through to ascontiguousarray which materialises the proper buffer for
            // the IL kernel to walk linearly.
            //
            // ARC: track which entries we allocated so we can Dispose them in the
            // finally block — the materialised copies own their storage and must be
            // explicitly released (see commits 392529f2 / 294d4329 / 4ad62bb3 for
            // the established pattern).
            var casts = new NDArray[ndim];
            var ownCast = new bool[ndim];
            try
            {
                for (int d = 0; d < ndim; d++)
                {
                    var src = broadcasted[d];
                    NDArray c;
                    if (src.GetTypeCode == NPTypeCode.Int64 && src.Shape.IsContiguous)
                    {
                        c = src;
                        ownCast[d] = false;
                    }
                    else
                    {
                        // order 'C' explicitly: astype's default 'K' now keeps a broadcast
                        // source's perm layout (stride-0 axes innermost — NumPy astype parity),
                        // but the IL kernel below walks these buffers LINEARLY in C order, and
                        // NumPy's own ravel_multi_index materialises via PyArray_ContiguousFromAny.
                        c = src.GetTypeCode == NPTypeCode.Int64
                            ? np.ascontiguousarray(src)
                            : src.astype(NPTypeCode.Int64, true, 'C');
                        ownCast[d] = !ReferenceEquals(c, src);
                    }
                    casts[d] = c;
                }

                ExecuteRavelMultiIndex(casts, dims, modes, order, result);
            }
            finally
            {
                for (int d = 0; d < ndim; d++)
                    if (ownCast[d]) casts[d]?.Dispose();
            }

            return result;
        }

        private static unsafe void ExecuteRavelMultiIndex(
            NDArray[] casts, int[] dims, int[] modes, char order, NDArray<long> result)
        {
            int ndim = casts.Length;
            long count = result.size;

            var kernel = DirectILKernelGenerator.GetRavelMultiIndexKernel();
            if (kernel == null)
                throw new NotSupportedException("np.ravel_multi_index: IL kernel unavailable");

            Span<long> dimsSpan = stackalloc long[ndim];
            Span<long> ravelStrides = stackalloc long[ndim];
            for (int d = 0; d < ndim; d++) dimsSpan[d] = dims[d];
            ComputeRavelStrides(dims, order, ravelStrides);

            long** coordPtrs = stackalloc long*[ndim];
            for (int d = 0; d < ndim; d++)
            {
                var s = casts[d].Shape;
                coordPtrs[d] = (long*)casts[d].Storage.Address + s.offset;
            }

            long* outPtr = (long*)result.Address;

            fixed (long* dimsPtr = dimsSpan)
            fixed (long* stridesPtr = ravelStrides)
            fixed (int* modesPtr = modes)
            {
                long status = kernel(coordPtrs, count, dimsPtr, stridesPtr, modesPtr, ndim, outPtr);
                if (status < count)
                    throw new ArgumentException("invalid entry in coordinates array", nameof(casts));
            }
        }

        private static void ComputeRavelStrides(int[] dims, char order, Span<long> ravelStrides)
        {
            int ndim = dims.Length;
            long s = 1;
            if (order == 'C')
            {
                for (int d = ndim - 1; d >= 0; d--)
                {
                    ravelStrides[d] = s;
                    long next = unchecked(s * dims[d]);
                    if (dims[d] != 0 && next / dims[d] != s)
                        throw new ArgumentException(
                            "invalid dims: array size defined by dims is larger than the maximum possible size.",
                            nameof(dims));
                    s = next;
                }
            }
            else
            {
                for (int d = 0; d < ndim; d++)
                {
                    ravelStrides[d] = s;
                    long next = unchecked(s * dims[d]);
                    if (dims[d] != 0 && next / dims[d] != s)
                        throw new ArgumentException(
                            "invalid dims: array size defined by dims is larger than the maximum possible size.",
                            nameof(dims));
                    s = next;
                }
            }
        }

        /// <summary>The coordinate checks <c>ravel_multi_index</c> makes after it converts the modes (NumPy's order): both
        ///     present, one array per dimension.</summary>
        /// <param name="multi_index">The coordinate arrays.</param>
        /// <param name="dims">The shape.</param>
        /// <returns>The number of dimensions.</returns>
        /// <exception cref="ArgumentNullException">Either argument is null.</exception>
        /// <exception cref="ValueError">The counts differ (NumPy's <c>parameter multi_index must be a sequence of length
        ///     N</c>).</exception>
        /// <exception cref="ArgumentException"><paramref name="dims"/> is empty.</exception>
        private static int ValidateRavelInputs(NDArray[] multi_index, int[] dims)
        {
            if (multi_index == null) throw new ArgumentNullException(nameof(multi_index));
            if (dims == null) throw new ArgumentNullException(nameof(dims));
            // NumPy's int_sequence_to_arrays: a ValueError naming the expected length.
            if (multi_index.Length != dims.Length)
                throw new ValueError($"parameter multi_index must be a sequence of length {dims.Length}");
            if (dims.Length == 0)
                throw new ArgumentException("dims must contain at least one dimension.", nameof(dims));
            return dims.Length;
        }

        private static long ApplyMode(long j, long m, int mode, int dim)
        {
            switch (mode)
            {
                case 0: // raise
                    if (j < 0 || j >= m)
                        throw new ArgumentException("invalid entry in coordinates array");
                    return j;
                case 1: // wrap (NumPy staged path)
                    if (j < 0)
                    {
                        j += m;
                        if (j < 0)
                        {
                            j %= m;
                            if (j != 0) j += m;
                        }
                    }
                    else if (j >= m)
                    {
                        j -= m;
                        if (j >= m) j %= m;
                    }
                    return j;
                case 2: // clip
                    if (j < 0) return 0;
                    if (j >= m) return m - 1;
                    return j;
                default:
                    return j;
            }
        }
    }
}
