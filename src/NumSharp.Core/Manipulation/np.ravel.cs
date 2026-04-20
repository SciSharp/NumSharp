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
                // F-order ravel: read column-major; same values as flatten('F').
                return a.flatten('F');
            }

            // C-order: view when possible, otherwise materialize a C-contiguous copy.
            if (!a.Shape.IsContiguous)
                return new NDArray(new UnmanagedStorage(a.Storage.CloneData(), Shape.Vector(a.size)));

            return a.reshape(Shape.Vector(a.size));
        }
    }
}
