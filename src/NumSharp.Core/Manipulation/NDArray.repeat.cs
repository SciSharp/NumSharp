namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Repeat each element of the array after itself. Refer to <see cref="np.repeat(NDArray, int, int?)"/>
        ///     for full documentation.
        /// </summary>
        /// <param name="repeats">The number of repetitions for each element.</param>
        /// <param name="axis">The axis along which to repeat values. The default (null) flattens the input array and returns a flat output array.</param>
        /// <returns>Output array which has the same shape as this array, except along the given axis.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ndarray.repeat.html</remarks>
        public NDArray repeat(int repeats, int? axis = null)
            => np.repeat(this, repeats, axis);

        /// <inheritdoc cref="repeat(int, int?)"/>
        public NDArray repeat(long repeats, int? axis = null)
            => np.repeat(this, repeats, axis);

        /// <summary>
        ///     Repeat each element of the array by the per-element counts in <paramref name="repeats"/>. Refer to
        ///     <see cref="np.repeat(NDArray, NDArray, int?)"/> for full documentation.
        /// </summary>
        /// <param name="repeats">Per-element repetition counts, broadcast to the shape along the given axis.</param>
        /// <param name="axis">The axis along which to repeat values. The default (null) flattens the input array and returns a flat output array.</param>
        /// <returns>Output array which has the same shape as this array, except along the given axis.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ndarray.repeat.html</remarks>
        public NDArray repeat(NDArray repeats, int? axis = null)
            => np.repeat(this, repeats, axis);
    }
}
