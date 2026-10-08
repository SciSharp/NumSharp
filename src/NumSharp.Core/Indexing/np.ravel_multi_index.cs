using System;
using System.Text;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Generic;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Converts a tuple of coordinate arrays into an array of flat indices, applying boundary modes per axis —
        ///     NumPy's <c>arr_ravel_multi_index</c> (<c>numpy/_core/src/multiarray/compiled_base.c</c>), statement for
        ///     statement. Inverse of <see cref="unravel_index(NDArray, int[], char)"/>.
        /// </summary>
        /// <param name="multi_index">
        ///     The coordinate arrays, one per dimension of <paramref name="dims"/>: integer or bool, any layout; they
        ///     broadcast against each other, and the broadcast shape is the result's shape.
        /// </param>
        /// <param name="dims">The shape of the array the coordinates index. An empty shape (NumPy's <c>()</c>) takes no
        ///     coordinate array and gives the flat index 0.</param>
        /// <param name="mode">
        ///     Boundary mode for every axis: <c>"raise"</c> (default — an out-of-range coordinate raises),
        ///     <c>"wrap"</c> (modulo with sign correction) or <c>"clip"</c> (saturate).
        /// </param>
        /// <param name="order">
        ///     <c>'C'</c> (row-major, default) or <c>'F'</c> (column-major), in either case: the stride ordering used to
        ///     fold coordinates into a flat index.
        /// </param>
        /// <returns>
        ///     The int64 flat indices, in the coordinates' broadcast shape, laid out as NumPy's iterator allocates them
        ///     (KEEPORDER over the coordinate arrays: F-ordered coordinates give an F-ordered result, a transposed set the
        ///     same permutation, mixed or broadcast ones C order). Coordinates that are all 0-d give NumPy's scalar: a
        ///     fresh read-only 0-d array.
        /// </returns>
        /// <exception cref="ArgumentNullException"><paramref name="multi_index"/> or <paramref name="dims"/> is null.</exception>
        /// <exception cref="ValueError">
        ///     In NumPy's order: an order letter that is not C / F / A / K (<c>order must be one of 'C', 'F', 'A', or 'K'
        ///     (got 'X')</c>); a bad <paramref name="mode"/> (NumPy's <c>clipmode_parser</c> texts, see
        ///     <see cref="ClipModeConverter"/>); an <c>'A'</c> or <c>'K'</c> order (<c>only 'C' or 'F' order is
        ///     permitted</c>); a dims product that overflows int64 (<c>invalid dims: array size defined by dims is larger
        ///     than the maximum possible size.</c>); a coordinate count that differs from <paramref name="dims"/>' length
        ///     (<c>parameter multi_index must be a sequence of length N</c>); a zero dimension while there is a coordinate
        ///     to fold (<c>cannot unravel if shape has zero entries (is empty).</c>, whatever the mode); and (mode
        ///     <c>raise</c>) a coordinate out of range (<c>invalid entry in coordinates array</c>).
        /// </exception>
        /// <exception cref="TypeError">A coordinate array that is neither integer nor bool (<c>only int indices
        ///     permitted</c>), checked in order after the count.</exception>
        /// <exception cref="IncorrectShapeException">The coordinate arrays do not broadcast together (NumPy's
        ///     <c>ValueError</c> text from its iterator, every coordinate shape listed: <c>operands could not be broadcast
        ///     together with shapes (2,) (3,) </c>).</exception>
        /// <exception cref="NotSupportedException">Runtime code generation is unavailable (the IL kernel cannot be
        ///     emitted).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.ravel_multi_index.html
        ///     <para><b>BREAKING (NumPy parity):</b> the result used to be C-ordered for any coordinate layout, and an
        ///     <c>'A'</c> / <c>'K'</c> order and a dims product overflow were refused only after the coordinates were
        ///     read. A lowercase order letter is accepted, an empty <paramref name="dims"/> is legal, a zero dimension in
        ///     wrap or clip mode raises NumPy's ValueError (it used to divide by zero), and the errors are
        ///     <see cref="ValueError"/> (an <see cref="ArgumentException"/>).</para>
        ///     <para><b>Deliberate divergence:</b> NumPy refuses more than 63 dimensions (<c>too many dimensions passed to
        ///     ravel_multi_index</c>, its fixed iterator operand limit); NumSharp has no dimension cap.</para>
        /// </remarks>
        public static NDArray<long> ravel_multi_index(NDArray[] multi_index, int[] dims, string mode = "raise", char order = 'C')
        {
            using var plain = NDMatrixBoundary.PlainFor(multi_index);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            if (multi_index == null) throw new ArgumentNullException(nameof(multi_index));
            if (dims == null) throw new ArgumentNullException(nameof(dims));
            // arr_ravel_multi_index's order: the argument parse converts the order (a bad letter first); then
            // PyArray_ConvertClipmodeSequence converts the modes (one mode string serves every dimension); then the
            // ravel strides refuse an 'A' / 'K' order and an overflowing product; only then are the coordinates read.
            NPY_ORDER npyOrder = OrderResolver.Convert(order);
            var modes = new int[dims.Length];
            int singleMode = ClipModeConverter(mode);
            for (int d = 0; d < modes.Length; d++) modes[d] = singleMode;
            var ravelStrides = RavelStrides(dims, npyOrder);
            ValidateRavelInputs(multi_index, dims);

            return RavelMultiIndexImpl(multi_index, dims, modes, ravelStrides);
        }

        /// <summary>
        ///     Per-axis mode overload. <paramref name="modes"/> length must match
        ///     <paramref name="dims"/> length.
        /// </summary>
        /// <param name="multi_index">Tuple of integer arrays, one per dimension.</param>
        /// <param name="dims">Shape of the array the indices are unravelling into.</param>
        /// <param name="modes">One boundary mode per dimension (NumPy's <c>mode</c> tuple); <c>null</c> is NumPy's
        ///     <c>None</c>, which raises on every dimension.</param>
        /// <param name="order"><c>'C'</c> (default) or <c>'F'</c>, in either case.</param>
        /// <returns>The flat indices, laid out as <see cref="ravel_multi_index(NDArray[], int[], string, char)"/> lays them
        ///     out.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="multi_index"/> or <paramref name="dims"/> is null.</exception>
        /// <exception cref="ValueError">In NumPy's order: a bad order letter; a mode count that differs from the dimensions
        ///     (<c>list of clipmodes has wrong length (1 instead of 2)</c>) or a bad mode (<see cref="ClipModeConverter"/>);
        ///     an <c>'A'</c> / <c>'K'</c> order; a dims product overflow; a coordinate count that differs from
        ///     <paramref name="dims"/>' length; a zero dimension with a coordinate to fold; (mode <c>raise</c>) a coordinate
        ///     out of range.</exception>
        /// <exception cref="TypeError">A coordinate array that is neither integer nor bool (<c>only int indices
        ///     permitted</c>).</exception>
        /// <exception cref="IncorrectShapeException">The coordinate arrays do not broadcast together (NumPy's iterator
        ///     text).</exception>
        /// <exception cref="NotSupportedException">The IL kernel cannot be emitted.</exception>
        public static NDArray<long> ravel_multi_index(NDArray[] multi_index, int[] dims, string[] modes, char order = 'C')
        {
            using var plain = NDMatrixBoundary.PlainFor(multi_index);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            if (multi_index == null) throw new ArgumentNullException(nameof(multi_index));
            if (dims == null) throw new ArgumentNullException(nameof(dims));
            // NumPy's order: the order letter (argument parse), then PyArray_ConvertClipmodeSequence — the tuple's
            // length, then each mode in order —, then the ravel strides, all before the coordinates are read.
            NPY_ORDER npyOrder = OrderResolver.Convert(order);
            var modeInts = ConvertClipmodeSequence(modes, dims.Length);
            var ravelStrides = RavelStrides(dims, npyOrder);
            ValidateRavelInputs(multi_index, dims);

            return RavelMultiIndexImpl(multi_index, dims, modeInts, ravelStrides);
        }

        /// <summary>
        ///     Scalar convenience overload — folds a single coordinate tuple into a
        ///     flat index. Equivalent to wrapping each coord in a 0-d NDArray but
        ///     returns a <see cref="long"/> directly.
        /// </summary>
        /// <param name="coords">One coordinate per dimension.</param>
        /// <param name="dims">The shape the coordinates index; an empty shape takes no coordinate and gives 0.</param>
        /// <param name="mode"><c>"raise"</c> (default), <c>"wrap"</c> or <c>"clip"</c>, for every axis.</param>
        /// <param name="order"><c>'C'</c> (default) or <c>'F'</c>, in either case.</param>
        /// <returns>The flat index.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="coords"/> or <paramref name="dims"/> is null.</exception>
        /// <exception cref="ValueError">In NumPy's order: a bad order letter; a bad mode; an <c>'A'</c> / <c>'K'</c> order;
        ///     a dims product overflow; a coordinate count that differs from <paramref name="dims"/>' length
        ///     (<c>parameter multi_index must be a sequence of length N</c>); a zero dimension (<c>cannot unravel if shape
        ///     has zero entries (is empty).</c>); (mode <c>raise</c>) a coordinate out of range (<c>invalid entry in
        ///     coordinates array</c>).</exception>
        public static long ravel_multi_index(long[] coords, int[] dims, string mode = "raise", char order = 'C')
        {
            if (coords == null) throw new ArgumentNullException(nameof(coords));
            if (dims == null) throw new ArgumentNullException(nameof(dims));
            // NumPy's order, as in the array form: the order letter, the mode, the ravel strides, then the coordinates.
            NPY_ORDER npyOrder = OrderResolver.Convert(order);
            int modeInt = ClipModeConverter(mode);
            var ravelStrides = RavelStrides(dims, npyOrder);
            if (coords.Length != dims.Length)
                throw new ValueError($"parameter multi_index must be a sequence of length {dims.Length}");

            int ndim = dims.Length;
            // A scalar tuple is one coordinate to fold, so NumPy's zero-dimension check applies (in every mode).
            ThrowIfZeroDimension(dims);

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
        ///     The shared core of <c>ravel_multi_index</c>: NumPy's per-array integer check (<c>astype_anyint</c>), its
        ///     iterator's broadcast and output allocation (KEEPORDER over the coordinate arrays), the int64 casts and the
        ///     folding kernel.
        /// </summary>
        /// <param name="multi_index">The coordinate arrays (one per dimension; the count is already checked).</param>
        /// <param name="dims">The shape being indexed.</param>
        /// <param name="modes">The parsed per-axis modes.</param>
        /// <param name="ravelStrides">The C- or F-order ravel strides (<see cref="RavelStrides"/>).</param>
        /// <returns>The flat indices, in the coordinates' broadcast shape and NumPy's layout.</returns>
        /// <exception cref="TypeError">A coordinate array that is neither integer nor bool (<c>only int indices permitted</c>:
        ///     a float, complex, decimal or text array — NumPy never truncates or parses coordinates).</exception>
        /// <exception cref="IncorrectShapeException">The coordinate arrays do not broadcast together.</exception>
        /// <exception cref="ValueError">A zero dimension while there is a coordinate to fold, or (mode <c>raise</c>) a
        ///     coordinate out of range.</exception>
        /// <exception cref="NotSupportedException">The IL kernel cannot be emitted.</exception>
        private static unsafe NDArray<long> RavelMultiIndexImpl(NDArray[] multi_index, int[] dims, int[] modes, long[] ravelStrides)
        {
            foreach (var coords in multi_index)
                ThrowIfNotIntIndices(coords);

            int ndim = multi_index.Length;

            // dims=(): NumPy's iterator has the allocated output as its only operand — a 0-d iteration of one element,
            // whose sum of no products is 0, returned as np.int64(0).
            if (ndim == 0)
            {
                var zero = new NDArray<long>(Shape.NewScalar(), true);
                zero.MarkReductionScalar();
                return zero;
            }

            // NumPy's NpyIter_MultiNew over the coordinates plus an ALLOCATE output (KEEPORDER, no stride negation since an
            // output is allocated): the coordinates broadcast together, and the output is laid out dense along the
            // iterator's axis order — Shape.NpyIterOutputShape over the coordinates' own (unbroadcast) shapes.
            var bdims = BroadcastCoordinates(multi_index);
            var shapes = new Shape[ndim];
            for (int d = 0; d < ndim; d++)
                shapes[d] = multi_index[d].Shape;
            Shape layout = Shape.NpyIterOutputShape(bdims, shapes);

            // ravel_multi_index_loop: a zero dimension is refused before any coordinate is folded (in every mode — wrap
            // would divide by it), once there is a coordinate to fold. NumPy allocates its output first; the allocation is
            // unobservable, so the check runs before it here and a refused call leaves nothing to release.
            long count = layout.size;
            if (count != 0)
                ThrowIfZeroDimension(dims);

            var result = new NDArray<long>(layout, false);
            if (count == 0)
                return result;

            // The kernel walks every coordinate buffer and the result linearly, so each coordinate must be laid out like
            // the result: an int64 array of the broadcast dims already in that layout (every C-contiguous one when the
            // result is C) is read in place, any other is broadcast and cast (NumPy's wrapping same_kind cast for uint64)
            // into a copy of that layout. The copies are released on every path, exceptions included, and keep the read
            // pointers valid until the kernel is done.
            var copies = new NDArray[ndim];
            try
            {
                // One buffer address per dimension, pointer-sized (nint) so the kernel's long** reads them on any platform;
                // on the stack for any realistic rank (NumSharp has no dimension cap, so a huge one takes the heap).
                Span<nint> coordPtrs = ndim <= 64 ? stackalloc nint[ndim] : new nint[ndim];
                for (int d = 0; d < ndim; d++)
                {
                    var src = multi_index[d];
                    if (src.typecode == NPTypeCode.Int64 && HasDims(src.Shape, bdims) && IsLaidOutAs(src.Shape, layout))
                    {
                        coordPtrs[d] = (nint)((long*)src.Storage.Address + src.Shape.offset);
                    }
                    else
                    {
                        var copy = NDArray.ForOverwrite(NPTypeCode.Int64, layout);
                        copies[d] = copy;
                        NDIter.Copy(copy, src);
                        coordPtrs[d] = (nint)copy.Storage.Address;
                    }
                }

                ExecuteRavelMultiIndex(coordPtrs, count, dims, ravelStrides, modes, (long*)result.Storage.Address);
                // The in-place route reads the caller's buffers through raw pointers: keep them reachable until the kernel
                // is done.
                GC.KeepAlive(multi_index);
            }
            catch
            {
                // A refused coordinate (mode raise) or a failed cast: the result is never handed out, so its buffer goes
                // back to the pool now rather than through the finalizer.
                result.Dispose();
                throw;
            }
            finally
            {
                for (int d = 0; d < ndim; d++)
                    copies[d]?.Dispose();
            }

            // NumPy's PyArray_Return: an all-0-d result is a numpy scalar — read-only, owning its value.
            if (result.ndim == 0)
                result.MarkReductionScalar();
            return result;
        }

        /// <summary>
        ///     The broadcast of the coordinate arrays' dimensions as NumPy's iterator computes it (right-aligned, an
        ///     extent-1 axis stretches), with its error text.
        /// </summary>
        /// <param name="multi_index">The coordinate arrays.</param>
        /// <returns>A fresh array of the broadcast dimensions (the result shape adopts it).</returns>
        /// <exception cref="IncorrectShapeException">Two extents disagree and neither is 1: NumPy's iterator text, every
        ///     coordinate shape followed by a space (<c>operands could not be broadcast together with shapes (2,) (3,)
        ///     </c>).</exception>
        private static long[] BroadcastCoordinates(NDArray[] multi_index)
        {
            int nd = 0;
            foreach (var c in multi_index)
                nd = Math.Max(nd, c.ndim);

            var bdims = new long[nd];
            for (int k = 0; k < nd; k++)
                bdims[k] = 1;

            foreach (var c in multi_index)
            {
                var cd = c.Shape.dimensions;
                int off = nd - cd.Length;
                for (int k = 0; k < cd.Length; k++)
                {
                    long v = cd[k];
                    if (v == 1)
                        continue;
                    if (bdims[off + k] == 1)
                        bdims[off + k] = v;
                    else if (bdims[off + k] != v)
                    {
                        var sb = new StringBuilder("operands could not be broadcast together with shapes ");
                        foreach (var op in multi_index)
                            sb.Append(op.Shape.ToPythonTuple()).Append(' ');
                        throw new IncorrectShapeException(sb.ToString());
                    }
                }
            }

            return bdims;
        }

        /// <summary>Whether <paramref name="s"/> has exactly the dimensions <paramref name="dims"/> (no broadcast needed).</summary>
        /// <param name="s">An operand's shape.</param>
        /// <param name="dims">The broadcast dimensions.</param>
        /// <returns>True when the rank and every extent agree.</returns>
        private static bool HasDims(Shape s, long[] dims)
        {
            var sd = s.dimensions;
            if (sd.Length != dims.Length)
                return false;
            for (int k = 0; k < sd.Length; k++)
                if (sd[k] != dims[k])
                    return false;
            return true;
        }

        /// <summary>
        ///     NumPy's <c>ravel_multi_index_loop</c> over <paramref name="count"/> rows: the IL kernel reads every coordinate
        ///     buffer and writes the result linearly.
        /// </summary>
        /// <param name="coordPtrs">One int64 buffer address per dimension, laid out like the result.</param>
        /// <param name="count">The number of flat indices.</param>
        /// <param name="dims">The shape being indexed.</param>
        /// <param name="ravelStrides">The C- or F-order ravel strides.</param>
        /// <param name="modes">The per-axis modes.</param>
        /// <param name="outPtr">The result buffer.</param>
        /// <exception cref="ValueError">(mode <c>raise</c>) a coordinate out of range (<c>invalid entry in coordinates
        ///     array</c> — NumPy names neither the axis nor the value).</exception>
        /// <exception cref="NotSupportedException">The IL kernel cannot be emitted.</exception>
        private static unsafe void ExecuteRavelMultiIndex(
            ReadOnlySpan<nint> coordPtrs, long count, int[] dims, long[] ravelStrides, int[] modes, long* outPtr)
        {
            int ndim = coordPtrs.Length;

            var kernel = DirectILKernelGenerator.GetRavelMultiIndexKernel();
            if (kernel == null)
                throw new NotSupportedException("np.ravel_multi_index: IL kernel unavailable");

            Span<long> dimsLong = ndim <= 64 ? stackalloc long[ndim] : new long[ndim];
            for (int d = 0; d < ndim; d++) dimsLong[d] = dims[d];

            fixed (nint* ptrs = coordPtrs)
            fixed (long* dimsPtr = dimsLong)
            fixed (long* stridesPtr = ravelStrides)
            fixed (int* modesPtr = modes)
            {
                // coordPtrs holds the buffer addresses as pointer-sized integers: the kernel's long** reads them as pointers.
                long status = kernel((long**)ptrs, count, dimsPtr, stridesPtr, modesPtr, ndim, outPtr);
                if (status < count)
                    throw new ValueError("invalid entry in coordinates array");
            }
        }

        /// <summary>
        ///     The ravel strides <c>arr_ravel_multi_index</c> computes before it reads the coordinates, with NumPy's two
        ///     refusals: the order, and the dims product (checked on every multiplication, the last one included).
        /// </summary>
        /// <param name="dims">The shape being indexed.</param>
        /// <param name="order">The converted order.</param>
        /// <returns>One stride per dimension (C: the last dimension's is 1; F: the first one's).</returns>
        /// <exception cref="ValueError"><c>'A'</c> / <c>'K'</c> (<c>only 'C' or 'F' order is permitted</c>), or a product
        ///     that overflows int64 (<c>invalid dims: array size defined by dims is larger than the maximum possible
        ///     size.</c>).</exception>
        private static long[] RavelStrides(int[] dims, NPY_ORDER order)
        {
            bool cOrder = RequireCOrFOrder(order);
            int ndim = dims.Length;
            var ravelStrides = new long[ndim];
            long s = 1;
            for (int i = 0; i < ndim; i++)
            {
                int d = cOrder ? ndim - 1 - i : i;
                ravelStrides[d] = s;
                long next = unchecked(s * dims[d]);
                if (dims[d] != 0 && next / dims[d] != s)
                    throw new ValueError("invalid dims: array size defined by dims is larger than the maximum possible size.");
                s = next;
            }
            return ravelStrides;
        }

        /// <summary>
        ///     <c>ravel_multi_index_loop</c>'s first check: a shape with a zero dimension has no element to index.
        /// </summary>
        /// <param name="dims">The shape being indexed.</param>
        /// <exception cref="ValueError">A zero dimension (<c>cannot unravel if shape has zero entries (is empty).</c>).</exception>
        /// <remarks>NumPy makes it only when there is a coordinate to fold (an empty broadcast skips the loop), in every
        ///     mode: wrap would otherwise divide by the zero dimension.</remarks>
        private static void ThrowIfZeroDimension(int[] dims)
        {
            for (int d = 0; d < dims.Length; d++)
                if (dims[d] == 0)
                    throw new ValueError("cannot unravel if shape has zero entries (is empty).");
        }

        /// <summary>The coordinate checks <c>ravel_multi_index</c> makes after it converts the modes and the strides
        ///     (NumPy's order): both present, one array per dimension.</summary>
        /// <param name="multi_index">The coordinate arrays.</param>
        /// <param name="dims">The shape.</param>
        /// <returns>The number of dimensions.</returns>
        /// <exception cref="ArgumentNullException">Either argument is null.</exception>
        /// <exception cref="ValueError">The counts differ (NumPy's <c>parameter multi_index must be a sequence of length
        ///     N</c>).</exception>
        private static int ValidateRavelInputs(NDArray[] multi_index, int[] dims)
        {
            if (multi_index == null) throw new ArgumentNullException(nameof(multi_index));
            if (dims == null) throw new ArgumentNullException(nameof(dims));
            // NumPy's int_sequence_to_arrays: a ValueError naming the expected length. dims=() takes no coordinate array
            // (and gives the flat index 0).
            if (multi_index.Length != dims.Length)
                throw new ValueError($"parameter multi_index must be a sequence of length {dims.Length}");
            return dims.Length;
        }

        /// <summary>
        ///     One coordinate under its boundary mode, as <c>ravel_multi_index_loop</c> treats it.
        /// </summary>
        /// <param name="j">The coordinate.</param>
        /// <param name="m">The dimension (not zero: <see cref="ThrowIfZeroDimension"/> ran first).</param>
        /// <param name="mode">0 raise, 1 wrap, 2 clip.</param>
        /// <param name="dim">The axis (unused in the text: NumPy names neither the axis nor the value).</param>
        /// <returns>The coordinate brought into range (wrap / clip) or unchanged (raise, in range).</returns>
        /// <exception cref="ValueError">(raise) the coordinate is outside <c>[0, m)</c>: <c>invalid entry in coordinates
        ///     array</c>.</exception>
        private static long ApplyMode(long j, long m, int mode, int dim)
        {
            switch (mode)
            {
                case 0: // raise
                    if (j < 0 || j >= m)
                        throw new ValueError("invalid entry in coordinates array");
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
