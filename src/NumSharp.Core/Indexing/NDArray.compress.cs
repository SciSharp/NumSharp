namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Return selected slices of this array along the given axis. Refer to
        ///     <see cref="np.compress(NDArray, NDArray, int?, NDArray)"/> for full documentation.
        /// </summary>
        /// <param name="condition">1-D boolean array selecting which entries to return. If longer than the axis length the extra entries are treated as false.</param>
        /// <param name="axis">Axis along which to take slices. The default (null) works over the flattened array.</param>
        /// <param name="out">Output array whose type is preserved and which must be of the right shape to hold the output.</param>
        /// <returns>A copy of the selected slices along the given axis.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ndarray.compress.html</remarks>
        public NDArray compress(NDArray condition, int? axis = null, NDArray @out = null)
            => np.compress(condition, this, axis, @out);
    }
}
