namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Return a contiguous flattened array. A 1-D array, containing the elements of the input, is returned
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ravel.html</remarks>
        /// <remarks><br></br>If this array's <see cref="Shape"/> is a slice, the a copy will be made.</remarks>
        public NDArray ravel() => np.ravel(this, 'C');

        /// <summary>
        ///     Return a contiguous flattened array. A 1-D array, containing the elements of the input, is returned
        /// </summary>
        /// <param name="order">
        ///     The order in which to read the elements.
        ///     'C' - row-major, 'F' - column-major,
        ///     'A' - 'F' if F-contiguous (and not C-contiguous) else 'C',
        ///     'K' - memory order.
        /// </param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ravel.html</remarks>
        public NDArray ravel(char order) => np.ravel(this, order);
    }
}
