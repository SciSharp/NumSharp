namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Move axes of an array to new positions.
        ///     Other axes remain in their original order.
        /// </summary>
        /// <param name="a">The array whose axes should be reordered.</param>
        /// <param name="source">Original positions of the axes to move. These must be unique (distinct).</param>
        /// <param name="destination">Destination positions for each of the original axes. These must also be unique (distinct).</param>
        /// <returns>Array with moved axes.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.moveaxis.html</remarks>
        public static NDArray moveaxis(in NDArray a, int source, int destination)
            => a.TensorEngine.MoveAxis(a, new[] {source}, new[] {destination});

        /// <summary>
        ///     Move axes of an array to new positions.
        ///     Other axes remain in their original order.
        /// </summary>
        /// <param name="a">The array whose axes should be reordered.</param>
        /// <param name="source">Original positions of the axes to move. These must be unique (distinct).</param>
        /// <param name="destination">Destination positions for each of the original axes. These must also be unique (distinct).</param>
        /// <returns>Array with moved axes.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.moveaxis.html</remarks>
        public static NDArray moveaxis(in NDArray a, int[] source, int destination)
            => a.TensorEngine.MoveAxis(a, source, new[] {destination});

        /// <summary>
        ///     Move axes of an array to new positions.
        ///     Other axes remain in their original order.
        /// </summary>
        /// <param name="a">The array whose axes should be reordered.</param>
        /// <param name="source">Original positions of the axes to move. These must be unique (distinct).</param>
        /// <param name="destination">Destination positions for each of the original axes. These must also be unique (distinct).</param>
        /// <returns>Array with moved axes.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.moveaxis.html</remarks>
        public static NDArray moveaxis(in NDArray a, int source, int[] destination)
            => a.TensorEngine.MoveAxis(a, new[] {source}, destination);

        /// <summary>
        ///     Move axes of an array to new positions.
        ///     Other axes remain in their original order.
        /// </summary>
        /// <param name="a">The array whose axes should be reordered.</param>
        /// <param name="source">Original positions of the axes to move. These must be unique (distinct).</param>
        /// <param name="destination">Destination positions for each of the original axes. These must also be unique (distinct).</param>
        /// <returns>Array with moved axes.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.moveaxis.html</remarks>
        public static NDArray moveaxis(in NDArray a, int[] source, int[] destination)
            => a.TensorEngine.MoveAxis(a, source, destination);
    }
}