namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Returns the index of the maximum value (flattened array).
        /// </summary>
        /// <returns>The index of the maximal value in the flattened array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.argmax.html</remarks>
        public long argmax()
        {
            using var plain = NDMatrixBoundary.PlainFor(this);   // the member keeps the plain-array meaning for a matrix object
            return np.argmax(this);
        }

        /// <summary>
        ///     Returns the indices of the maximum values along an axis.
        /// </summary>
        /// <param name="axis">The axis along which to operate. By default, the index is into the flattened array.</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one.</param>
        /// <returns>Array of indices into the array. It has the same shape as a.shape with the dimension along axis removed (unless keepdims is True).</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.argmax.html</remarks>
        public NDArray argmax(int axis, bool keepdims = false)
        {
            using var plain = NDMatrixBoundary.PlainFor(this);   // the member keeps the plain-array meaning for a matrix object
            return np.argmax(this, axis, keepdims);
        }

        /// <summary>
        ///     NumPy's <c>ndarray.argmax(axis=None, out=None, *, keepdims=False)</c>: the int64 indices of the maximum
        ///     values along <paramref name="axis"/> (the first occurrence), or of the C-order flattened array.
        /// </summary>
        /// <param name="axis">The axis along which to operate, or null for the flattened array.</param>
        /// <param name="out">If provided, the indices are written into it and it is returned (see
        ///     <see cref="np.argmax(NDArray, int?, NDArray, bool)"/>).</param>
        /// <param name="keepdims">If true, the reduced axis is left in the result with length one.</param>
        /// <returns>The int64 indices, or <paramref name="out"/>.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is outside the array's rank.</exception>
        /// <exception cref="ValueError">The reduced length is 0, or <paramref name="out"/> has another shape.</exception>
        /// <exception cref="TypeError"><paramref name="out"/>'s dtype does not cast safely to int64.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ndarray.argmax.html</remarks>
        public NDArray argmax(int? axis = null, NDArray @out = null, bool keepdims = false)
        {
            using var plain = NDMatrixBoundary.PlainFor(this);   // the member keeps the plain-array meaning for a matrix object
            return np.argmax(this, axis, @out, keepdims);
        }
    }
}
