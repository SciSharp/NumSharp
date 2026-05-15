using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return a contiguous flattened array. A 1-D array, containing the elements of the input, is returned
        /// </summary>
        /// <param name="a">Input array. The elements in a are read in the order specified by order, and packed as a 1-D array.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ravel.html</remarks>
        /// <remarks><br></br>If this array's <see cref="Shape"/> is a sliced or broadcasted, the a copy will be made.</remarks>
        public static NDArray ravel(NDArray a) => ravel(a, 'C');

        /// <summary>
        ///     Return a contiguous flattened array. A 1-D array, containing the elements of the input, is returned
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="order">
        ///     The order in which to read the elements.
        ///     'C' - row-major, 'F' - column-major,
        ///     'A' - 'F' if a is F-contiguous (and not C-contiguous) else 'C',
        ///     'K' - memory order.
        /// </param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ravel.html</remarks>
        public static NDArray ravel(NDArray a, char order)
        {
            char physical = OrderResolver.Resolve(order, a.Shape);

            if (physical == 'F' && a.Shape.NDim > 1 && a.size > 1)
            {
                // F-order ravel: read column-major.
                // When the source is F-contiguous, strides[0]==1 and memory is dense, so a
                // linear walk from `offset` for `size` elements is exactly the F-order
                // read-out — return a 1-D view sharing the underlying buffer (no copy).
                // NumPy: np.shares_memory(np.ravel(aF, 'F'), aF) == True.
                if (a.Shape.IsFContiguous)
                {
                    var vec = new Shape(new long[] { a.size }, new long[] { 1 }, a.Shape.offset, a.Shape.bufferSize);
                    return new NDArray(a.Storage.Alias(vec)) { TensorEngine = a.TensorEngine };
                }
                // Non-F-contiguous source: must materialize column-major into fresh memory.
                return a.flatten('F');
            }

            // C-order: view when possible, otherwise materialize a C-contiguous copy.
            if (!a.Shape.IsContiguous)
                return new NDArray(new UnmanagedStorage(a.Storage.CloneData(), Shape.Vector(a.size))) { TensorEngine = a.TensorEngine };

            return a.reshape(Shape.Vector(a.size));
        }
    }
}
