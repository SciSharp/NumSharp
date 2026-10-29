// NumPy's np.unravel_index (arr_unravel_index, numpy/_core/src/multiarray/compiled_base.c) as DefaultEngine's member: the
// argument checks in NumPy's order, the result buffer laid out along the indices' iteration order, the unravel kernel and
// the coordinate views. The file sits beside its facade while it calls np.* (the helpers np.ravel_multi_index shares):
// ratchet R2 counts np.* calls against every file under Backends/ (the np-layer dtype plan, C3); it moves under
// Backends/Default/ in P6.

using System;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Generic;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <inheritdoc />
        /// <remarks>
        ///     <para><b>Iteration order:</b> NumPy walks the indices with <c>NpyIter</c> (KEEPORDER,
        ///     <c>DONT_NEGATE_STRIDES</c>) and writes the coordinates of the k-th element it visits into the k-th row of
        ///     the buffer, which <c>NpyIter_CreateCompatibleStrides</c> lays out along that same order. DefaultEngine copies a
        ///     non-int64 or differently laid out index array into that layout (an int64 array already laid out so — every
        ///     C-contiguous one — is read in place) and runs one IL kernel over it linearly, so the rows, the strides and
        ///     the out-of-range index that is reported first are NumPy's.</para>
        ///     <para><b>Lifetime:</b> this member's scope releases the interleaved buffer's own handle and any int64 copy of
        ///     <paramref name="indices"/> on return, exceptions included; the coordinate arrays are yielded and each holds
        ///     a counted reference on the buffer, so disposing one leaves the others valid.</para>
        /// </remarks>
        [NDScoped]
        public override unsafe NDArray<long>[] UnravelIndex(NDArray indices, int[] shape, char order = 'C')
        {
            if (indices is null) throw new ArgumentNullException(nameof(indices));
            if (shape is null) throw new ArgumentNullException(nameof(shape));

            // NumPy's statement order: the argument parse converts the order (a bad letter is refused before anything
            // else), PyArray_OverflowMultiplyList checks the dimensions, astype_anyint refuses a non-integer index array,
            // and only then — once the result buffer exists, before any index is read — is an 'A' / 'K' order refused.
            NPY_ORDER npyOrder = OrderResolver.Convert(order);
            long unravelSize = np.UnravelSize(shape);
            np.ThrowIfNotIntIndices(indices);
            bool cOrder = np.RequireCOrFOrder(npyOrder);

            int ndims = shape.Length;
            long count = indices.size;

            // NumPy's iterator (KEEPORDER, DONT_NEGATE_STRIDES) visits the indices along their |stride| order, and
            // NpyIter_CreateCompatibleStrides lays the result out dense along that order, so the coordinates of the k-th
            // element visited are the k-th row. Shape.NpyIterOutputShape is that layout for one operand: a broadcast or
            // extent-1 axis abstains from the sort, C order wins every tie, and no axis is reversed.
            Shape idxShape = indices.Shape;
            Shape layout = Shape.NpyIterOutputShape((long[])idxShape.dimensions.Clone(), new ReadOnlySpan<Shape>(in idxShape));

            if (ndims == 0)
            {
                // shape=(): the empty product makes 0 the one valid index. NumPy's loop checks every index against [0, 1)
                // and writes no coordinate, and only AFTER the loop does it refuse an index array of any rank (gh-580),
                // so an out-of-range index reports its own error first. The kernel needs a dimension to run, so the
                // indices are checked as an unravel into (1,), whose bounds are the same; the coordinates it writes are
                // discarded with the scratch buffer.
                if (count > 0)
                {
                    var scratch = new NDArray(NPTypeCode.Int64, new Shape(count), false);
                    UnravelInto(indices, layout, (long*)scratch.Storage.Address, UnitDimension, 1, cOrder);
                }
                if (indices.ndim != 0)
                    throw new ValueError("multiple indices are not supported for 0d arrays");
                return Array.Empty<NDArray<long>>();
            }

            // NumPy's ret_arr: indices.shape + (ndims,), the leading axes strided along the iteration order (scaled by
            // the row length) and the coordinate axis innermost. The kernel writes every element before anyone reads it.
            int idxNdim = indices.ndim;
            var bufDims = new long[idxNdim + 1];
            var bufStrides = new long[idxNdim + 1];
            for (int k = 0; k < idxNdim; k++)
            {
                bufDims[k] = layout.dimensions[k];
                bufStrides[k] = layout.strides[k] * ndims;
            }
            bufDims[idxNdim] = ndims;
            bufStrides[idxNdim] = 1;
            var coords = NDArray.ForOverwrite(NPTypeCode.Int64, Shape.Adopt(bufDims, bufStrides));

            if (count > 0)
                UnravelInto(indices, layout, (long*)coords.Storage.Address, shape, unravelSize, cOrder);

            var result = new NDArray<long>[ndims];
            if (idxNdim == 0)
            {
                // A 0-d index array: NumPy's PyArray_Return turns each 0-d view into a numpy SCALAR, which owns its value
                // and cannot be written — fresh 0-d arrays with a scalar's flags (MarkReductionScalar: num=263).
                long* row = (long*)coords.Storage.Address;
                for (int j = 0; j < ndims; j++)
                {
                    var scalar = new NDArray<long>(Shape.NewScalar(), false);
                    *(long*)scalar.Storage.Address = row[j];
                    scalar.MarkReductionScalar();
                    result[j] = scalar;
                }
                return result;
            }

            // NumPy's PyArray_NewFromDescrAndBase(ret_dims, ret_strides, data + j*8, base=ret_arr): coordinate j is the
            // buffer's leading axes at element offset j. Each view holds its own counted reference on the buffer, so the
            // scope may release coords' handle at exit; an empty result keeps the offset pattern and is never read.
            var viewStrides = new long[idxNdim];
            Array.Copy(bufStrides, viewStrides, idxNdim);
            long bufferSize = count * ndims;
            for (int j = 0; j < ndims; j++)
            {
                var view = new Shape(layout.dimensions, viewStrides, j, bufferSize);
                result[j] = new NDArray<long>(coords.Storage.Alias(view)) { TensorEngine = coords.TensorEngine };
            }

            return result;
        }

        /// <summary>The one-dimensional unravel shape <c>(1,)</c> that checks an index against <c>shape=()</c>'s bounds.</summary>
        private static readonly int[] UnitDimension = { 1 };

        /// <summary>
        ///     Runs the unravel kernel over <paramref name="indices"/> in NumPy's iteration order, writing element k's
        ///     coordinates into row k of <paramref name="out"/>.
        /// </summary>
        /// <param name="indices">The (integer or bool, non-empty) index array.</param>
        /// <param name="layout">The dense layout of <paramref name="indices"/>' dimensions along their iteration order
        ///     (<see cref="Shape.NpyIterOutputShape"/> of the index array).</param>
        /// <param name="out">The <c>size * len(shape)</c> interleaved coordinate rows.</param>
        /// <param name="shape">The unravel dimensions (at least one).</param>
        /// <param name="unravelSize">Their product: the exclusive upper bound of a valid index.</param>
        /// <param name="cOrder">C order (true) or F order (false).</param>
        /// <exception cref="ValueError">An index outside <c>[0, unravelSize)</c>: the first one in the iteration
        ///     order.</exception>
        /// <exception cref="NotSupportedException">The IL kernel cannot be emitted.</exception>
        /// <remarks>
        ///     An int64 index array whose layout already IS <paramref name="layout"/> (every C-contiguous one, an F-ordered
        ///     one, a dense permutation) is read in place; any other is copied — and cast, with NumPy's wrapping same_kind
        ///     cast for uint64 — into that layout first, so the kernel can read it linearly. The copy is released before
        ///     this method returns, exceptions included, and keeps the read pointer valid until the kernel is done.
        /// </remarks>
        private static unsafe void UnravelInto(NDArray indices, Shape layout, long* @out, int[] shape, long unravelSize, bool cOrder)
        {
            long count = indices.size;
            NDArray copy = null;
            try
            {
                long* idx;
                if (indices.typecode == NPTypeCode.Int64 && np.IsLaidOutAs(indices.Shape, layout))
                {
                    idx = (long*)indices.Storage.Address + indices.Shape.offset;
                }
                else
                {
                    copy = NDArray.ForOverwrite(NPTypeCode.Int64, layout);
                    NDIter.Copy(copy, indices);
                    idx = (long*)copy.Storage.Address;
                }

                RunUnravelKernel(idx, count, shape, unravelSize, @out, cOrder);
                // The in-place route reads indices' buffer through a raw pointer: keep the array reachable until the
                // kernel is done (a GC finalizing it mid-scan would hand the buffer back to the pool).
                GC.KeepAlive(indices);
            }
            finally
            {
                copy?.Dispose();
            }
        }

        /// <summary>
        ///     NumPy's <c>unravel_index_loop</c>: an IL kernel over <paramref name="count"/> int64 indices read linearly —
        ///     the reciprocal kernel when every dimension is positive (a multiply-high and two shifts per peeled dimension,
        ///     exact for every index), the divmod kernel otherwise (a zero or negative dimension).
        /// </summary>
        /// <param name="idx">The indices, in iteration order.</param>
        /// <param name="count">How many.</param>
        /// <param name="shape">The unravel dimensions (at least one).</param>
        /// <param name="unravelSize">Their product.</param>
        /// <param name="out">The interleaved rows (<c>count * len(shape)</c>).</param>
        /// <param name="cOrder">C order (true) or F order (false).</param>
        /// <exception cref="ValueError">An index outside <c>[0, unravelSize)</c>, reported with NumPy's text and the
        ///     index's own value.</exception>
        /// <exception cref="NotSupportedException">The IL kernel cannot be emitted.</exception>
        private static unsafe void RunUnravelKernel(long* idx, long count, int[] shape, long unravelSize, long* @out, bool cOrder)
        {
            int ndim = shape.Length;
            long idxStart = cOrder ? ndim - 1 : 0, idxStep = cOrder ? -1 : 1;
            long status;

            // Every dimension positive (the common case): the reciprocal kernel, which divides by each dimension with a
            // precomputed invariant multiplier instead of the hardware divide. A zero dimension (no valid index: the
            // product refuses every one before any division) or a negative one (C's truncating division of a negative
            // running quotient, which the unsigned reciprocal does not reproduce) takes the divmod kernel.
            bool allPositive = true;
            for (int d = 0; d < ndim && allPositive; d++)
                allPositive = shape[d] > 0;

            if (allPositive && DirectILKernelGenerator.GetUnravelIndexMagicKernel() is { } magicKernel)
            {
                // [d, m', sh1, sh2] per dimension; a shape of more than 16 dimensions (NumSharp has no dimension cap)
                // takes the heap instead of the stack.
                Span<long> table = ndim <= 16 ? stackalloc long[ndim * 4] : new long[ndim * 4];
                for (int d = 0; d < ndim; d++)
                    DirectILKernelGenerator.FillUnravelDivisor(shape[d], table.Slice(d * 4, 4));
                fixed (long* tablePtr = table)
                    status = magicKernel(idx, count, tablePtr, unravelSize, @out, ndim, idxStart, idxStep);
            }
            else
            {
                var kernel = DirectILKernelGenerator.GetUnravelIndexKernel();
                if (kernel == null)
                    throw new NotSupportedException("np.unravel_index: IL kernel unavailable");

                // The dims as the kernel's int64s; a shape of more than 64 dimensions takes the heap instead of the stack.
                Span<long> dims = ndim <= 64 ? stackalloc long[ndim] : new long[ndim];
                for (int d = 0; d < ndim; d++)
                    dims[d] = shape[d];
                fixed (long* dimsPtr = dims)
                    status = kernel(idx, count, dimsPtr, unravelSize, @out, ndim, idxStart, idxStep);
            }

            if (status < count)
                throw new ValueError($"index {idx[status]} is out of bounds for array with size {unravelSize}");
        }
    }
}
