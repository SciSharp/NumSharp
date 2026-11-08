using System;

namespace NumSharp
{
    public static partial class np
    {
        // =====================================================================
        //  np.broadcast_shapes — broadcast the input shapes into a single shape.
        //
        //  Port of NumPy 2.x numpy.broadcast_shapes (numpy/lib/_stride_tricks_impl.py):
        //
        //      arrays = [np.empty(x, dtype=_size0_dtype) for x in args]
        //      return _broadcast_shape(*arrays)        # == np.broadcast(*arrays).shape
        //
        //  NumPy materialises one zero-byte array per shape and asks np.broadcast for the joint
        //  shape. NumSharp computes the joint shape DIRECTLY — no allocation — because the answer is
        //  a pure function of the dimension extents: Shape.BroadcastShape is _broadcast_shape, the
        //  multi-iterator's scan INCLUDING its error text. The scan walks the axes from the first and,
        //  within an axis, the arguments left to right; a conflict names the argument that set that
        //  axis's extent and the one that disagrees — never the running/accumulated shape — on the
        //  leftmost axis that has a conflict. Shape.ResolveReturnShape runs the same scan with the
        //  generic family message, without the argument numbers.
        // =====================================================================

        /// <summary>
        ///     Broadcasts the input shapes into the single shape they would all broadcast to, WITHOUT
        ///     allocating any array — the NumPy 2.x <c>numpy.broadcast_shapes</c>. Use it to size an
        ///     output up front (e.g. before an <c>out=</c> allocation) when you have only the shapes,
        ///     not the arrays. An <c>int</c> argument is read as the one-dimensional shape <c>(n,)</c>
        ///     exactly as NumPy treats a bare int, so <c>broadcast_shapes(3, 1)</c> is <c>(3,)</c>.
        /// </summary>
        /// <param name="shapes">
        ///     The shapes to broadcast against each other. Each entry is a <see cref="Shape"/>; the
        ///     rich implicit conversions on <see cref="Shape"/> let NumPy call sites port verbatim —
        ///     a tuple literal <c>(1, 2)</c>, a bare <c>int</c> (→ <c>(n,)</c>), an <c>int[]</c>/
        ///     <c>long[]</c> (e.g. <c>a.shape</c>), or an explicit <c>new Shape(...)</c> all bind. No
        ///     argument at all returns the scalar shape <c>()</c> (NumPy's empty-tuple result).
        /// </param>
        /// <returns>
        ///     The broadcast shape. A scalar shape <c>()</c> when <paramref name="shapes"/> is empty or
        ///     every entry is scalar. A stretched extent of 0 wins over 1 (so <c>(0,)</c> broadcast
        ///     with <c>(1,)</c> is <c>(0,)</c>), matching NumPy — a zero-length axis is not size-1.
        /// </returns>
        /// <exception cref="ValueError">
        ///     If any dimension is negative — <c>"negative dimensions are not allowed"</c>, verbatim
        ///     with NumPy (which reaches it through the <c>np.empty</c> it would have allocated).
        /// </exception>
        /// <exception cref="IncorrectShapeException">
        ///     If the shapes are not broadcast-compatible. The message is NumPy's verbatim, naming the
        ///     argument that first established the conflicting extent and the argument that clashes
        ///     with it (e.g. <c>"...  Mismatch is between arg 0 with shape (3,) and arg 1 with shape
        ///     (4,)."</c>).
        /// </exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.broadcast_shapes.html</remarks>
        public static Shape broadcast_shapes(params Shape[] shapes)
        {
            // No operands broadcast to the scalar shape (NumPy's empty tuple). Guard here so the loop
            // below never has to special-case nd == 0 for the result construction.
            if (shapes == null || shapes.Length == 0)
                return new Shape();

            int len = shapes.Length;

            // NumPy validates negatives up front: broadcast_shapes builds np.empty(x) per shape, and a
            // negative extent is rejected there before any broadcasting. Reproduce that ordering — a
            // negative dimension is reported even when the shapes would ALSO fail to broadcast.
            for (int a = 0; a < len; a++)
            {
                long[] dims = shapes[a].dimensions;
                if (dims == null)
                    continue;
                for (int d = 0; d < dims.Length; d++)
                    if (dims[d] < 0)
                        throw new ValueError(AllocationGuard.NegativeDimensionsMessage);
            }

            // The broadcast itself is NumPy's _broadcast_shape over the (zero-byte) arrays: np.broadcast's multi-iterator,
            // axis by axis, naming the operand that set an axis's length and the first one that disagrees with it, in groups
            // of 64 operands (Shape.BroadcastShape).
            return Shape.BroadcastShape(shapes);
        }
    }
}
