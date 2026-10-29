using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Broadcast an shape against an other new shape.
        /// </summary>
        /// <param name="from">The shape that is to be broadcasted</param>
        /// <param name="against">The shape that'll be used to broadcast <paramref name="from"/> shape</param>
        /// <returns>A readonly view on the original array with the given shape. It is typically not contiguous. Furthermore, more than one element of a broadcasted array may refer to a single memory location.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.broadcast_to.html</remarks>
        /// <exception cref="IncorrectShapeException"><paramref name="from"/> does not broadcast to
        ///     <paramref name="against"/> one way: it has more axes, or an axis of a length other than 1 that differs
        ///     from the target's.</exception>
        public static Shape broadcast_to(Shape from, Shape against)
        {
            return Shape.BroadcastTo(from, against);
        }

        /// <summary>
        ///     Broadcast an array to a new shape.
        /// </summary>
        /// <param name="from">The NDArray to broadcast.</param>
        /// <param name="against">The shape to broadcast against.</param>
        /// <returns>These arrays are views on the original arrays. They are typically not contiguous. Furthermore, more than one element of a broadcasted array may refer to a single memory location. If you need to write to the arrays, make copies first.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.broadcast_to.html</remarks>
        /// <exception cref="IncorrectShapeException"><paramref name="from"/> does not broadcast to
        ///     <paramref name="against"/> one way: it has more axes, or an axis of a length other than 1 that differs
        ///     from the target's.</exception>
        public static NDArray broadcast_to(UnmanagedStorage from, Shape against)
        {
            return NDArray.BroadcastTo(from, against);
        }

        /// <summary>
        ///     Broadcast an array to a new shape.
        /// </summary>
        /// <param name="from">The NDArray to broadcast.</param>
        /// <param name="against">The shape to broadcast against.</param>
        /// <returns>These arrays are views on the original arrays. They are typically not contiguous. Furthermore, more than one element of a broadcasted array may refer to a single memory location. If you need to write to the arrays, make copies first.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.broadcast_to.html</remarks>
        /// <exception cref="IncorrectShapeException"><paramref name="from"/> does not broadcast to
        ///     <paramref name="against"/> one way: it has more axes, or an axis of a length other than 1 that differs
        ///     from the target's.</exception>
        public static NDArray broadcast_to(NDArray from, Shape against)
        {
            using var plain = NDMatrixBoundary.PlainFor(from);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return NDArray.BroadcastTo(from, against);
        }

        /// <summary>
        ///     Broadcast an shape against an other new shape.
        /// </summary>
        /// <param name="from">The shape that is to be broadcasted</param>
        /// <param name="against">The shape that'll be used to broadcast <paramref name="from"/> shape</param>
        /// <returns>A readonly view on the original array with the given shape. It is typically not contiguous. Furthermore, more than one element of a broadcasted array may refer to a single memory location.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.broadcast_to.html</remarks>
        /// <exception cref="IncorrectShapeException"><paramref name="from"/> does not broadcast to
        ///     <paramref name="against"/> one way: it has more axes, or an axis of a length other than 1 that differs
        ///     from the target's.</exception>
        public static Shape broadcast_to(Shape from, NDArray against)
        {
            using var plain = NDMatrixBoundary.PlainFor(against);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return Shape.BroadcastTo(from, against.Shape);
        }

        /// <summary>
        ///     Broadcast an array to a new shape.
        /// </summary>
        /// <param name="from">The UnmanagedStorage to broadcast.</param>
        /// <param name="against">The shape to broadcast against.</param>
        /// <returns>These arrays are views on the original arrays. They are typically not contiguous. Furthermore, more than one element of a broadcasted array may refer to a single memory location. If you need to write to the arrays, make copies first.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.broadcast_to.html</remarks>
        /// <exception cref="IncorrectShapeException"><paramref name="from"/> does not broadcast to
        ///     <paramref name="against"/> one way: it has more axes, or an axis of a length other than 1 that differs
        ///     from the target's.</exception>
        public static NDArray broadcast_to(UnmanagedStorage from, NDArray against)
        {
            using var plain = NDMatrixBoundary.PlainFor(against);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return NDArray.BroadcastTo(from, against.Shape);
        }

        /// <summary>
        ///     Broadcast an array to a new shape.
        /// </summary>
        /// <param name="from">The NDArray to broadcast.</param>
        /// <param name="against">The shape to broadcast against.</param>
        /// <returns>These arrays are views on the original arrays. They are typically not contiguous. Furthermore, more than one element of a broadcasted array may refer to a single memory location. If you need to write to the arrays, make copies first.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.broadcast_to.html</remarks>
        /// <exception cref="IncorrectShapeException"><paramref name="from"/> does not broadcast to
        ///     <paramref name="against"/> one way: it has more axes, or an axis of a length other than 1 that differs
        ///     from the target's.</exception>
        public static NDArray broadcast_to(NDArray from, NDArray against)
        {
            using var plain = NDMatrixBoundary.PlainFor(from, against);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return NDArray.BroadcastTo(from, against.Shape);
        }


        /// <summary>
        ///     Broadcast an shape against an other new shape.
        /// </summary>
        /// <param name="from">The shape that is to be broadcasted</param>
        /// <param name="against">The shape that'll be used to broadcast <paramref name="from"/> shape</param>
        /// <returns>A readonly view on the original array with the given shape. It is typically not contiguous. Furthermore, more than one element of a broadcasted array may refer to a single memory location.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.broadcast_to.html</remarks>
        /// <exception cref="IncorrectShapeException"><paramref name="from"/> does not broadcast to
        ///     <paramref name="against"/> one way: it has more axes, or an axis of a length other than 1 that differs
        ///     from the target's.</exception>
        public static Shape broadcast_to(Shape from, UnmanagedStorage against)
        {
            return Shape.BroadcastTo(from, against.Shape);
        }

        /// <summary>
        ///     Broadcast an array to a new shape.
        /// </summary>
        /// <param name="from">The UnmanagedStorage to broadcast.</param>
        /// <param name="against">The shape to broadcast against.</param>
        /// <returns>These arrays are views on the original arrays. They are typically not contiguous. Furthermore, more than one element of a broadcasted array may refer to a single memory location. If you need to write to the arrays, make copies first.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.broadcast_to.html</remarks>
        /// <exception cref="IncorrectShapeException"><paramref name="from"/> does not broadcast to
        ///     <paramref name="against"/> one way: it has more axes, or an axis of a length other than 1 that differs
        ///     from the target's.</exception>
        public static NDArray broadcast_to(UnmanagedStorage from, UnmanagedStorage against)
        {
            return NDArray.BroadcastTo(from, against.Shape);
        }

        /// <summary>
        ///     Broadcast an array to a new shape.
        /// </summary>
        /// <param name="from">The NDArray to broadcast.</param>
        /// <param name="against">The shape to broadcast against.</param>
        /// <returns>These arrays are views on the original arrays. They are typically not contiguous. Furthermore, more than one element of a broadcasted array may refer to a single memory location. If you need to write to the arrays, make copies first.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.broadcast_to.html</remarks>
        /// <exception cref="IncorrectShapeException"><paramref name="from"/> does not broadcast to
        ///     <paramref name="against"/> one way: it has more axes, or an axis of a length other than 1 that differs
        ///     from the target's.</exception>
        public static NDArray broadcast_to(NDArray from, UnmanagedStorage against)
        {
            using var plain = NDMatrixBoundary.PlainFor(from);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return NDArray.BroadcastTo(from, against.Shape);
        }
    }
}
