using System;
using System.Text;

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
        //  a pure function of the dimension extents. The accumulation loop below reproduces
        //  np.broadcast's multi-iterator exactly, INCLUDING its error text: the multi-iterator adds
        //  arguments left to right, and when a non-1 extent conflicts with the extent an EARLIER
        //  argument already established for that (right-aligned) dimension, it names that earlier
        //  argument and the current one — never the running/accumulated shape. Getting the two named
        //  args right is the only reason this is not a one-liner over Shape.ResolveReturnShape (which
        //  throws the generic family message without arg indices).
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

            // The broadcast rank is the largest input rank; shorter shapes are right-aligned (their
            // missing leading axes act as size 1).
            int nd = 0;
            for (int a = 0; a < len; a++)
                nd = Math.Max(nd, shapes[a].NDim);

            var result = new long[nd];       // running broadcast extents, seeded to the size-1 identity
            var setter = new int[nd];        // which arg first pushed result[dim] above 1 (for the error)
            for (int i = 0; i < nd; i++)
            {
                result[i] = 1;
                setter[i] = -1;
            }

            // Add arguments left to right, exactly as np.broadcast's multi-iterator does, so the
            // FIRST conflict encountered — and the pair of args it names — matches NumPy.
            for (int a = 0; a < len; a++)
            {
                Shape shape = shapes[a];
                int snd = shape.NDim;
                int off = nd - snd;          // right-alignment offset into the result
                long[] dims = shape.dimensions;
                for (int d = 0; d < snd; d++)
                {
                    long extent = dims[d];
                    int rd = off + d;
                    long cur = result[rd];

                    // A size-1 axis always yields to the other operand and never establishes an extent.
                    if (extent == 1)
                        continue;

                    if (cur == 1)
                    {
                        result[rd] = extent;
                        setter[rd] = a;      // this arg owns the extent; a later clash points back here
                    }
                    else if (cur != extent)
                    {
                        // Conflict: name the establishing arg and this one, with their ORIGINAL shapes,
                        // in NumPy's exact wording (two spaces before "Mismatch", Python-repr tuples).
                        throw new IncorrectShapeException(
                            "shape mismatch: objects cannot be broadcast to a single shape.  " +
                            $"Mismatch is between arg {setter[rd]} with shape {PyReprShape(shapes[setter[rd]].dimensions)} " +
                            $"and arg {a} with shape {PyReprShape(shape.dimensions)}.");
                    }
                    // cur == extent: compatible, nothing to record.
                }
            }

            return new Shape(result);
        }

        /// <summary>
        ///     Renders dimension extents as a Python tuple repr — <c>()</c> for rank 0, <c>(4,)</c>
        ///     with a trailing comma for rank 1, <c>(2, 3)</c> with a space after each comma otherwise.
        ///     This is the exact spelling NumPy's broadcast error uses; it is neither
        ///     <see cref="Shape.ToString"/> (no trailing comma on a 1-tuple) nor
        ///     <c>Shape.ToPythonTuple</c> (no space after the comma), hence the dedicated helper.
        /// </summary>
        /// <param name="dims">The dimension extents to render; a null or empty array renders <c>()</c>.</param>
        /// <returns>The Python-repr tuple string for <paramref name="dims"/>.</returns>
        private static string PyReprShape(long[] dims)
        {
            if (dims == null || dims.Length == 0)
                return "()";
            if (dims.Length == 1)
                return "(" + dims[0] + ",)";
            var sb = new StringBuilder("(");
            for (int i = 0; i < dims.Length; i++)
            {
                if (i > 0)
                    sb.Append(", ");
                sb.Append(dims[i]);
            }
            sb.Append(')');
            return sb.ToString();
        }
    }
}
