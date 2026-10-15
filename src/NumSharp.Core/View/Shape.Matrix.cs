using System;

namespace NumSharp
{
    public readonly partial struct Shape
    {
        /// <summary>
        ///     The shape NumPy's <c>matrix.__array_finalize__</c> gives an array of this shape: a matrix is two-dimensional,
        ///     so every other rank is folded onto two axes WITHOUT moving an element. Use it wherever an array becomes a
        ///     <see cref="NDMatrix"/>; it is the one place that decides which arrays can be a matrix at all.
        /// </summary>
        /// <returns>
        ///     This shape when it already has two axes. Otherwise a new shape over the same buffer (same offset, same
        ///     buffer size): a 0-d shape becomes <c>(1, 1)</c>, a 1-d shape <c>(N,)</c> becomes the row <c>(1, N)</c>, and
        ///     a shape with more than two axes drops every axis whose extent is 0 or 1 and then follows the same rules.
        /// </returns>
        /// <exception cref="ValueError">More than two axes have an extent above 1 (NumPy's
        ///     <c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="IncorrectShapeException">The array has more than two axes and one of them has extent 0. NumPy
        ///     drops that axis with the extent-1 axes and then fails to reshape the empty array to the remaining extents
        ///     (<c>cannot reshape array of size 0 into shape (1,3)</c>).</exception>
        /// <remarks>
        ///     <para>
        ///     NumPy's rule, statement for statement (<c>numpy/matrixlib/defmatrix.py</c>): with more than two axes
        ///     <c>newshape = tuple(x for x in self.shape if x > 1)</c>; two extents left are the matrix shape, more is an
        ///     error, and fewer fall through to the low-rank rule (<c>()</c> to <c>(1, 1)</c>, <c>(N,)</c> to
        ///     <c>(1, N)</c>). NumPy then assigns <c>self.shape</c> in place, which is a no-copy reshape.
        ///     </para>
        ///     <para>
        ///     The strides are the ones that no-copy reshape computes (<c>_attempt_nocopy_reshape</c>), so
        ///     <c>ndarray.strides</c> of a matrix matches NumPy's: a kept axis keeps its stride, the axis prepended to a
        ///     row gets <c>N * stride</c>, and a matrix made from one element gets strides of one element.
        ///     </para>
        ///     <para>
        ///     The layout flags (C / F contiguity, BROADCASTED) are computed again from the new strides. The WRITEABLE
        ///     and ALIGNED bits are this shape's own, because an in-place shape assignment changes neither in NumPy. The
        ///     OWNDATA mirror is NOT carried: the storage that installs the shape sets it
        ///     (<see cref="Backends.UnmanagedStorage.SetShapeUnsafe(ref Shape)"/> for an in-place change,
        ///     <see cref="Backends.UnmanagedStorage.Alias(Shape)"/> for a view).
        ///     </para>
        /// </remarks>
        internal readonly Shape AsMatrix()
        {
            long[] dims = dimensions;
            int nd = dims.Length;
            if (nd == 2)
                return this;

            long keptSize = bufferSize > 0 ? bufferSize : size;

            if (nd > 2)
            {
                // newshape = tuple(x for x in self.shape if x > 1): an extent of 0 is dropped WITH the extent-1 axes.
                int keep = 0;
                for (int i = 0; i < nd; i++)
                    if (dims[i] > 1)
                        keep++;

                if (keep > 2)
                    throw new ValueError("shape too large to be a matrix.");

                var kd = new long[keep];
                var ks = new long[keep];
                for (int i = 0, j = 0; i < nd; i++)
                {
                    if (dims[i] > 1)
                    {
                        kd[j] = dims[i];
                        ks[j] = strides[i];
                        j++;
                    }
                }

                // NumPy now assigns self.shape = (k0, k1) | (1, k0) | (1, 1). The assignment is a reshape, and a
                // reshape cannot change the element count: that is how a dropped extent-0 axis becomes an error.
                long t0 = keep == 2 ? kd[0] : 1;
                long t1 = keep == 2 ? kd[1] : keep == 1 ? kd[0] : 1;
                if (t0 * t1 != size)
                    throw new IncorrectShapeException($"cannot reshape array of size {size} into shape ({t0},{t1})");

                if (keep == 2)
                    return WithAccessFlagsOf(new Shape(kd, ks, offset, keptSize));
                if (keep == 1)
                    return WithAccessFlagsOf(new Shape(new[] { 1L, kd[0] }, new[] { kd[0] * ks[0], ks[0] }, offset, keptSize));
                return WithAccessFlagsOf(new Shape(new[] { 1L, 1L }, new[] { 1L, 1L }, offset, keptSize));
            }

            if (nd == 1)
                return WithAccessFlagsOf(new Shape(new[] { 1L, dims[0] }, new[] { dims[0] * strides[0], strides[0] }, offset, keptSize));

            // 0-d: NumPy's no-copy reshape gives every axis of a one-element result the item size as its stride.
            return WithAccessFlagsOf(new Shape(new[] { 1L, 1L }, new[] { 1L, 1L }, offset, keptSize));
        }

        /// <summary>
        ///     The column shape <c>(N, 1)</c> over the same elements as this one-dimensional shape <c>(N,)</c>. NumPy's
        ///     <c>matrix.__getitem__</c> uses it when a column was selected with an integer
        ///     (<c>m[:, 0]</c>, <c>m[[0, 1, 0], 1]</c>), where the plain rule <see cref="AsMatrix"/> would give a row.
        /// </summary>
        /// <returns>A new shape over the same buffer (same offset, same buffer size) with a trailing axis of extent 1.</returns>
        /// <exception cref="InvalidOperationException">This shape does not have exactly one axis.</exception>
        /// <remarks>
        ///     NumPy assigns <c>out.shape = (sh, 1)</c> in place. Its no-copy reshape gives the trailing extent-1 axis the
        ///     stride of the axis before it, so a column cut out of a C-contiguous matrix reports strides
        ///     <c>(s, s)</c>, not <c>(s, itemsize)</c>. The flags follow the rule of <see cref="AsMatrix"/>: layout bits
        ///     computed again, WRITEABLE and ALIGNED kept.
        /// </remarks>
        internal readonly Shape AsMatrixColumn()
        {
            if (dimensions.Length != 1)
                throw new InvalidOperationException($"A matrix column is made from a one-dimensional shape, not from {ToPythonTuple()}.");

            long keptSize = bufferSize > 0 ? bufferSize : size;
            return WithAccessFlagsOf(new Shape(new[] { dimensions[0], 1L }, new[] { strides[0], strides[0] }, offset, keptSize));
        }

        /// <summary>
        ///     Gives <paramref name="reshaped"/> this shape's WRITEABLE and ALIGNED bits. Use it after a no-copy
        ///     reshape built a shape from dimensions and strides alone, where those two bits came out as defaults.
        /// </summary>
        /// <param name="reshaped">A shape over the same memory as this one, with freshly computed flags.</param>
        /// <returns><paramref name="reshaped"/> with the two access bits of this shape; every other bit unchanged.</returns>
        /// <remarks>
        ///     A broadcast source (stride 0) computes WRITEABLE false by itself, and the reshaped form is broadcast
        ///     too, so both sides agree there. The case this method exists for is the other direction: a read-only
        ///     array that is NOT broadcast (<c>setflags(write: false)</c>, a read-only memory map, <c>diagonal()</c>),
        ///     whose reshaped form would otherwise come back writeable.
        /// </remarks>
        private readonly Shape WithAccessFlagsOf(Shape reshaped)
        {
            const ArrayFlags Access = ArrayFlags.WRITEABLE | ArrayFlags.ALIGNED;
            var own = (ArrayFlags)_flags & Access;
            if (((ArrayFlags)reshaped._flags & Access) == own)
                return reshaped;
            return reshaped.WithFlags(flagsToSet: own, flagsToClear: Access & ~own);
        }
    }
}
