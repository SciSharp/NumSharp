using NumSharp.Backends;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Return a copy of the array collapsed into one dimension.
        /// </summary>
        /// <param name="order">
        ///     The order in which to read the elements.
        ///     'C' - row-major (C-style), 'F' - column-major (Fortran-style),
        ///     'A' - 'F' if this is F-contiguous (and not C-contiguous) else 'C',
        ///     'K' - memory order (reads the elements in the order they occur in memory).
        /// </param>
        /// <returns>A copy of the input array, flattened to one dimension.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html
        ///     NumPy: flatten() ALWAYS returns a copy. Use ravel() for a view when possible.
        /// </remarks>
        public NDArray flatten(char order = 'C')
        {
            char physical = OrderResolver.Resolve(order, this.Shape);

            if (physical == 'F' && this.Shape.NDim > 1 && this.size > 1)
            {
                // F-order flatten: the memory of a fresh F-contiguous copy contains
                // the values in column-major read-out order; interpret it as 1-D.
                var fcopy = this.copy('F');
                return new NDArray(new UnmanagedStorage(fcopy.Array.Clone(), Shape.Vector(size)));
            }

            return new NDArray(new UnmanagedStorage(Storage.CloneData(), Shape.Vector(size)));
        }
    }
}
