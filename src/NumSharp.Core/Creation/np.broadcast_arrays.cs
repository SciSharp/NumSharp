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
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.broadcast_arrays.html</remarks>
        public static NDArray[] broadcast_arrays(params NDArray[] ndArrays)
        {
            int len = ndArrays.Length;
            int i;
            var inputShapes = new Shape[len];
            for (i = 0; i < len; i++) 
                inputShapes[i] = ndArrays[i].Shape;
            var outputShapes = DefaultEngine.Broadcast(inputShapes);

            var list = new NDArray[len];
            for (i = 0; i < len; i++) 
                list[i] = new NDArray(UnmanagedStorage.CreateBroadcastedUnsafe(ndArrays[i].Storage, outputShapes[i]));

            return list;
        }

        /// <summary>
        ///     Broadcast two arrays against each other.
        /// </summary>
        /// <param name="lhs">An array to broadcast.</param>
        /// <param name="rhs">An array to broadcast.</param>
        /// <returns>These arrays are views on the original arrays. They are typically not contiguous. Furthermore, more than one element of a broadcasted array may refer to a single memory location. If you need to write to the arrays, make copies first.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.broadcast_arrays.html</remarks>
        public static (NDArray Lhs, NDArray Rhs) broadcast_arrays(NDArray lhs, NDArray rhs)
        {
            var (leftShape, rightShape) = DefaultEngine.Broadcast(lhs.Shape, rhs.Shape);
            return (new NDArray(UnmanagedStorage.CreateBroadcastedUnsafe(lhs.Storage, leftShape)),
                new NDArray(UnmanagedStorage.CreateBroadcastedUnsafe(rhs.Storage, rightShape)));
        }
    }
}
