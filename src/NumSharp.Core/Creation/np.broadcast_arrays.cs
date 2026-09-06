using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Broadcast any number of arrays against each other.
        /// </summary>
        /// <param name="ndArrays">The arrays to broadcast.</param>
        /// <returns>These arrays are views on the original arrays. They are typically not contiguous. Furthermore, more than one element of a broadcasted array may refer to a single memory location. If you need to write to the arrays, make copies first.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.broadcast_arrays.html</remarks>
        public static NDArray[] broadcast_arrays(params NDArray[] ndArrays)
        {
            int len = ndArrays.Length;
            int i;
            var inputShapes = new Shape[len];
            for (i = 0; i < len; i++)
                inputShapes[i] = ndArrays[i].Shape;
            var outputShapes = Shape.Broadcast(inputShapes);

            var list = new NDArray[len];
            for (i = 0; i < len; i++)
                // NumPy 2.4.2 parity: broadcast_arrays results are WRITEABLE (unlike broadcast_to),
                // even for stretched (stride-0) dimensions. NumPy emits a FutureWarning that a future
                // version will change this, but 2.4.2 still returns writeable=True. Force the flag on so
                // even a broadcasted subshape (which Shape.Broadcast marks read-only) stays writeable.
                list[i] = new NDArray(UnmanagedStorage.CreateBroadcastedUnsafe(ndArrays[i].Storage, outputShapes[i].WithFlags(flagsToSet: ArrayFlags.WRITEABLE)));

            return list;
        }

        /// <summary>
        ///     Broadcast two arrays against each other.
        /// </summary>
        /// <param name="lhs">An array to broadcast.</param>
        /// <param name="rhs">An array to broadcast.</param>
        /// <returns>These arrays are views on the original arrays. They are typically not contiguous. Furthermore, more than one element of a broadcasted array may refer to a single memory location. If you need to write to the arrays, make copies first.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.broadcast_arrays.html</remarks>
        public static (NDArray Lhs, NDArray Rhs) broadcast_arrays(NDArray lhs, NDArray rhs)
        {
            var (leftShape, rightShape) = Shape.Broadcast(lhs.Shape, rhs.Shape);
            // NumPy 2.4.2 parity: broadcast_arrays results are WRITEABLE (see the params overload above).
            return (new NDArray(UnmanagedStorage.CreateBroadcastedUnsafe(lhs.Storage, leftShape.WithFlags(flagsToSet: ArrayFlags.WRITEABLE))),
                new NDArray(UnmanagedStorage.CreateBroadcastedUnsafe(rhs.Storage, rightShape.WithFlags(flagsToSet: ArrayFlags.WRITEABLE))));
        }
    }
}
