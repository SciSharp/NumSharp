using NumSharp.Backends;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Return a copy of the array collapsed into one dimension.
        /// </summary>
        /// <param name="order">
        ///     The order in which to read the elements. 'C' means row-major (C-style),
        ///     'F' means column-major (Fortran-style). NumSharp only supports 'C' order;
        ///     this parameter is accepted for API compatibility but 'F' is ignored.
        /// </param>
        /// <returns>A copy of the input array, flattened to one dimension.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html
        ///     NumPy: flatten() ALWAYS returns a copy. Use ravel() for a view when possible.
        /// </remarks>
        public NDArray flatten(char order = 'C')
        {
            // NumPy: flatten() ALWAYS returns a copy, regardless of memory layout.
            // For non-contiguous arrays (broadcast, sliced, transposed), CloneData()
            // correctly copies elements in logical (C-order) sequence.
            // Note: 'order' parameter is accepted for API compatibility but NumSharp
            // only supports C-order (row-major). F-order is silently treated as C-order.
            return new NDArray(new UnmanagedStorage(Storage.CloneData(), Shape.Vector(size)));
        }
    }
}
