namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Returns the index of the minimum value (flattened array).
        /// </summary>
        /// <returns>The index of the minimum value in the flattened array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.argmin.html</remarks>
        [PlainReceiver]
        public long argmin()
        {
            return np.argmin(this);
        }

        /// <summary>
        ///     NumPy's <c>ndarray.argmin(axis=None, out=None, *, keepdims=False)</c>: the int64 indices of the minimum
        ///     values along <paramref name="axis"/> (the first occurrence), or of the C-order flattened array.
        /// </summary>
        /// <param name="axis">The axis along which to operate, or null for the flattened array (ONE axis: a tuple is NumPy's
        ///     TypeError).</param>
        /// <param name="out">If provided, the indices are written into it and it is returned (see
        ///     <see cref="np.argmin(NDArray, NDAxis?, NDArray, bool)"/>).</param>
        /// <param name="keepdims">If true, the reduced axis is left in the result with length one.</param>
        /// <returns>The int64 indices, or <paramref name="out"/>.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is outside the array's rank.</exception>
        /// <exception cref="ValueError">The reduced length is 0, or <paramref name="out"/> has another shape.</exception>
        /// <exception cref="TypeError"><paramref name="out"/>'s dtype does not cast safely to int64; <paramref name="axis"/> is a
        ///     tuple (<c>'tuple' object cannot be interpreted as an integer</c>); <paramref name="out"/> is a C# scalar
        ///     (<c>output must be an array</c>: the second positional parameter is out=, so write <c>keepdims:</c> by
        ///     name).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ndarray.argmin.html</remarks>
        [PlainReceiver]
        public NDArray argmin(NDAxis? axis = null, NDArray @out = null, bool keepdims = false)
        {
            return np.argmin(this, axis, @out, keepdims);
        }

        /// <summary>
        ///     <see cref="argmin(NDAxis?, NDArray, bool)"/> with the axis as an <c>int?</c>: the overload a C# integer,
        ///     an <c>int?</c>, a bare <c>null</c> and <c>default</c> bind (NumPy's axis here is ONE integer, so there
        ///     is no <c>int[]</c> form, and <c>null</c> / <c>default</c> stay None), forwarding to the NDAxis form, so
        ///     every answer and error is that overload's.
        /// </summary>
        /// <param name="axis">The axis, or <c>null</c> for the flattened array.</param>
        /// <inheritdoc cref="argmin(NDAxis?, NDArray, bool)"/>
        public NDArray argmin(int? axis, NDArray @out = null, bool keepdims = false)
            => argmin(NDAxis.FromNullable(axis), @out, keepdims);
    }
}
