namespace NumSharp
{
    public partial class np
    {
        /// <summary>
        ///     Return an array copy of the given object.
        /// </summary>
        /// <param name="a">Input data.</param>
        /// <param name="order">
        ///     Controls the memory layout of the copy.
        ///     'C' - row-major, 'F' - column-major, 'A' - 'F' if source is F-contiguous else 'C',
        ///     'K' - match source layout as closely as possible.
        /// </param>
        /// <returns>Array interpretation of a.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.copy.html</remarks>
        public static NDArray copy(NDArray a, char order = 'K') => a.copy(order);
    }
}
