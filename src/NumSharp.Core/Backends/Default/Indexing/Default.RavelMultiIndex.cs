// NumPy's np.ravel_multi_index (arr_ravel_multi_index, numpy/_core/src/multiarray/compiled_base.c) as DefaultEngine's
// member: the argument checks in NumPy's order, astype_anyint (ThrowIfNotIntIndices, which np.unravel_index's member
// shares), the coordinates' broadcast and NumPy's output layout, the int64 casts and the folding kernel, with the
// helpers the scalar overload of np.ravel_multi_index shares (RavelStrides, ThrowIfZeroDimension).

using System;
using System.Text;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Generic;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <inheritdoc />
        /// <remarks>
        ///     <para><b>Layout:</b> NumPy's NpyIter_MultiNew over the coordinates plus an ALLOCATE output (KEEPORDER,
        ///     no stride negation) lays the result out dense along the iterator's axis order. DefaultEngine reads an
        ///     int64 coordinate array already laid out like the result in place and broadcasts / casts any other into a
        ///     copy of that layout (NumPy's wrapping same_kind cast for uint64), so one IL kernel walks every buffer
        ///     linearly.</para>
        ///     <para><b>Lifetime:</b> the copies are released on every path, exceptions included; a refused call (a
        ///     coordinate out of range in mode raise, a failed cast) returns the result's buffer to the pool at
        ///     once.</para>
        /// </remarks>
        public override unsafe NDArray<long> RavelMultiIndex(NDArray[] multi_index, int[] dims, object mode = null, char order = 'C')
        {
            if (multi_index == null) throw new ArgumentNullException(nameof(multi_index));
            if (dims == null) throw new ArgumentNullException(nameof(dims));
            // arr_ravel_multi_index's order: the argument parse converts the order (a bad letter first); then
            // PyArray_ConvertClipmodeSequence converts the mode argument (one mode string serves every dimension, a
            // sequence gives each its own: the sequence's length, then each mode in order); then the ravel strides
            // refuse an 'A' / 'K' order and an overflowing product; only then are the coordinates read.
            NPY_ORDER npyOrder = OrderResolver.Convert(order);
            int[] modes = ClipModes.ConvertClipmodeSequence(mode, dims.Length);
            long[] ravelStrides = RavelStrides(dims, npyOrder);
            ValidateRavelInputs(multi_index, dims);

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
                    if (src.typecode == NPTypeCode.Int64 && HasDims(src.Shape, bdims) && src.Shape.IsLaidOutAs(layout))
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
        ///     NumPy's <c>astype_anyint</c>: index arrays must hold integers or bools — a float, complex, decimal or text array
        ///     is refused rather than truncated or parsed, and so is a null entry (NumPy's <c>None</c>).
        /// </summary>
        /// <param name="indices">The index (or coordinate) array; null is NumPy's <c>None</c>.</param>
        /// <exception cref="TypeError">Null, or not integer or bool: <c>only int indices permitted</c>.</exception>
        /// <remarks>
        ///     NumSharp used to cast any dtype to int64 (<c>unravel_index(np.array(new[] { 1.7 }), …)</c> read index 1) and to
        ///     refuse text as a later phase's work; NumPy refuses both before reading an element. NumSharp's char counts as an
        ///     integer (its uint16 storage). A null entry of np.ravel_multi_index's coordinate list used to leak a
        ///     NullReferenceException; NumPy's <c>None</c> there becomes a 0-d object array, which this same check refuses, in
        ///     the list's order (np.unravel_index refuses a null index array with ArgumentNullException first).
        ///     <para>The test is NumPy's <c>PyArray_ISINTEGER(ret) || PyArray_ISBOOL(ret)</c> asked of the descriptor: its
        ///     kind is <c>'i'</c> or <c>'u'</c> (every integer class, NumSharp's char among them) or <c>'b'</c>; a float,
        ///     complex, decimal, text or datetime descriptor is none of them.</para>
        /// </remarks>
        private static void ThrowIfNotIntIndices(NDArray indices)
        {
            // A null entry is NumPy's None: astype_anyint converts it to a 0-d object array, which is not integer.
            if (indices is not null && indices.dtype.kind is 'b' or 'i' or 'u')
                return;
            throw new TypeError("only int indices permitted");
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
        internal static long[] RavelStrides(int[] dims, NPY_ORDER order)
        {
            bool cOrder = OrderResolver.RequireCOrFOrder(order);
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
        internal static void ThrowIfZeroDimension(int[] dims)
        {
            for (int d = 0; d < dims.Length; d++)
                if (dims[d] == 0)
                    throw new ValueError("cannot unravel if shape has zero entries (is empty).");
        }
    }
}
