namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Return specified diagonals. Refer to <see cref="np.diagonal(NDArray, int, int, int)"/> for full documentation.
        /// </summary>
        /// <param name="offset">Offset of the diagonal from the main diagonal. Can be positive or negative. Defaults to 0.</param>
        /// <param name="axis1">Axis to be used as the first axis of the 2-D sub-arrays from which the diagonals should be taken. Defaults to 0.</param>
        /// <param name="axis2">Axis to be used as the second axis of the 2-D sub-arrays from which the diagonals should be taken. Defaults to 1.</param>
        /// <returns>A view onto the requested diagonal(s).</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ndarray.diagonal.html</remarks>
        public NDArray diagonal(int offset = 0, int axis1 = 0, int axis2 = 1)
            => np.diagonal(this, offset, axis1, axis2);
    }
}
