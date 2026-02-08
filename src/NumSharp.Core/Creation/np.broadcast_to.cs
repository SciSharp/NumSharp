using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Validates that <paramref name="from"/> can be unilaterally broadcast to <paramref name="target"/>.
        ///     NumPy's broadcast_to only stretches dimensions of the source that are size 1 to match the target.
        ///     If any source dimension is larger than the target, or if the source has more dimensions than the target,
        ///     it raises ValueError.
        /// </summary>
        private static void ValidateBroadcastTo(Shape from, Shape target)
        {
            if (from.NDim > target.NDim)
                throw new IncorrectShapeException(
                    $"operands could not be broadcast together with remapped shapes " +
                    $"[original->remapped]: ({string.Join(",", from.dimensions)}) and requested shape ({string.Join(",", target.dimensions)})");

            // Right-align: iterate from the rightmost dimension
            for (int i = 0; i < from.NDim; i++)
            {
                int fromDim = from.dimensions[from.NDim - 1 - i];
                int targetDim = target.dimensions[target.NDim - 1 - i];

                if (fromDim != 1 && fromDim != targetDim)
                    throw new IncorrectShapeException(
                        $"operands could not be broadcast together with remapped shapes " +
                        $"[original->remapped]: ({string.Join(",", from.dimensions)}) and requested shape ({string.Join(",", target.dimensions)})");
            }
        }

        /// <summary>
        ///     Broadcast an shape against an other new shape.
        /// </summary>
        /// <param name="from">The shape that is to be broadcasted</param>
        /// <param name="against">The shape that'll be used to broadcast <paramref name="from"/> shape</param>
        /// <returns>A readonly view on the original array with the given shape. It is typically not contiguous. Furthermore, more than one element of a broadcasted array may refer to a single memory location.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.broadcast_to.html</remarks>
        public static Shape broadcast_to(Shape from, Shape against)
        {
            ValidateBroadcastTo(from, against);
            return DefaultEngine.Broadcast(from, against).LeftShape;
        }

        /// <summary>
        ///     Broadcast an array to a new shape.
        /// </summary>
        /// <param name="from">The NDArray to broadcast.</param>
        /// <param name="against">The shape to broadcast against.</param>
        /// <returns>These arrays are views on the original arrays. They are typically not contiguous. Furthermore, more than one element of a broadcasted array may refer to a single memory location. If you need to write to the arrays, make copies first.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.broadcast_to.html</remarks>
        public static NDArray broadcast_to(UnmanagedStorage from, Shape against)
        {
            ValidateBroadcastTo(from.Shape, against);
            return new NDArray(UnmanagedStorage.CreateBroadcastedUnsafe(from, DefaultEngine.Broadcast(from.Shape, against).LeftShape));
        }

        /// <summary>
        ///     Broadcast an array to a new shape.
        /// </summary>
        /// <param name="from">The NDArray to broadcast.</param>
        /// <param name="against">The shape to broadcast against.</param>
        /// <returns>These arrays are views on the original arrays. They are typically not contiguous. Furthermore, more than one element of a broadcasted array may refer to a single memory location. If you need to write to the arrays, make copies first.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.broadcast_to.html</remarks>
        public static NDArray broadcast_to(NDArray from, Shape against)
        {
            ValidateBroadcastTo(from.Shape, against);
            return new NDArray(UnmanagedStorage.CreateBroadcastedUnsafe(from.Storage, DefaultEngine.Broadcast(from.Shape, against).LeftShape));
        }

        /// <summary>
        ///     Broadcast an shape against an other new shape.
        /// </summary>
        /// <param name="from">The shape that is to be broadcasted</param>
        /// <param name="against">The shape that'll be used to broadcast <paramref name="from"/> shape</param>
        /// <returns>A readonly view on the original array with the given shape. It is typically not contiguous. Furthermore, more than one element of a broadcasted array may refer to a single memory location.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.broadcast_to.html</remarks>
        public static Shape broadcast_to(Shape from, NDArray against)
        {
            ValidateBroadcastTo(from, against.Shape);
            return DefaultEngine.Broadcast(from, against.Shape).LeftShape;
        }

        /// <summary>
        ///     Broadcast an array to a new shape.
        /// </summary>
        /// <param name="from">The UnmanagedStorage to broadcast.</param>
        /// <param name="against">The shape to broadcast against.</param>
        /// <returns>These arrays are views on the original arrays. They are typically not contiguous. Furthermore, more than one element of a broadcasted array may refer to a single memory location. If you need to write to the arrays, make copies first.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.broadcast_to.html</remarks>
        public static NDArray broadcast_to(UnmanagedStorage from, NDArray against)
        {
            ValidateBroadcastTo(from.Shape, against.Shape);
            return new NDArray(UnmanagedStorage.CreateBroadcastedUnsafe(from, DefaultEngine.Broadcast(from.Shape, against.Shape).LeftShape));
        }

        /// <summary>
        ///     Broadcast an array to a new shape.
        /// </summary>
        /// <param name="from">The NDArray to broadcast.</param>
        /// <param name="against">The shape to broadcast against.</param>
        /// <returns>These arrays are views on the original arrays. They are typically not contiguous. Furthermore, more than one element of a broadcasted array may refer to a single memory location. If you need to write to the arrays, make copies first.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.broadcast_to.html</remarks>
        public static NDArray broadcast_to(NDArray from, NDArray against)
        {
            ValidateBroadcastTo(from.Shape, against.Shape);
            return new NDArray(UnmanagedStorage.CreateBroadcastedUnsafe(from.Storage, DefaultEngine.Broadcast(from.Shape, against.Shape).LeftShape));
        }


        /// <summary>
        ///     Broadcast an shape against an other new shape.
        /// </summary>
        /// <param name="from">The shape that is to be broadcasted</param>
        /// <param name="against">The shape that'll be used to broadcast <paramref name="from"/> shape</param>
        /// <returns>A readonly view on the original array with the given shape. It is typically not contiguous. Furthermore, more than one element of a broadcasted array may refer to a single memory location.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.broadcast_to.html</remarks>
        public static Shape broadcast_to(Shape from, UnmanagedStorage against)
        {
            ValidateBroadcastTo(from, against.Shape);
            return DefaultEngine.Broadcast(from, against.Shape).LeftShape;
        }

        /// <summary>
        ///     Broadcast an array to a new shape.
        /// </summary>
        /// <param name="from">The UnmanagedStorage to broadcast.</param>
        /// <param name="against">The shape to broadcast against.</param>
        /// <returns>These arrays are views on the original arrays. They are typically not contiguous. Furthermore, more than one element of a broadcasted array may refer to a single memory location. If you need to write to the arrays, make copies first.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.broadcast_to.html</remarks>
        public static NDArray broadcast_to(UnmanagedStorage from, UnmanagedStorage against)
        {
            ValidateBroadcastTo(from.Shape, against.Shape);
            return new NDArray(UnmanagedStorage.CreateBroadcastedUnsafe(from, DefaultEngine.Broadcast(from.Shape, against.Shape).LeftShape));
        }

        /// <summary>
        ///     Broadcast an array to a new shape.
        /// </summary>
        /// <param name="from">The NDArray to broadcast.</param>
        /// <param name="against">The shape to broadcast against.</param>
        /// <returns>These arrays are views on the original arrays. They are typically not contiguous. Furthermore, more than one element of a broadcasted array may refer to a single memory location. If you need to write to the arrays, make copies first.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.broadcast_to.html</remarks>
        public static NDArray broadcast_to(NDArray from, UnmanagedStorage against)
        {
            ValidateBroadcastTo(from.Shape, against.Shape);
            return new NDArray(UnmanagedStorage.CreateBroadcastedUnsafe(from.Storage, DefaultEngine.Broadcast(from.Shape, against.Shape).LeftShape));
        }
    }
}
