namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Remove axes of length one from this array (NumPy's <c>ndarray.squeeze(axis=None)</c>). Refer to
        ///     <see cref="np.squeeze(NDArray, NDAxis?)"/> for full documentation.
        /// </summary>
        /// <param name="axis">
        ///     Selects the entries of length one to remove: <c>null</c> (None, the default) removes all length-one axes; one
        ///     axis or a tuple (<c>(0, 2)</c>, <c>new[] { 0, 2 }</c>) removes those, and the empty tuple none. Selecting an
        ///     axis whose length is not one raises.
        /// </param>
        /// <returns>A view of this array with the selected length-one axes removed. Shares memory with this array.</returns>
        /// <exception cref="AxisError">An axis is out of range.</exception>
        /// <exception cref="ValueError">A selected axis does not have length one, or a tuple names an axis twice
        ///     (<c>duplicate value in 'axis'</c>).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ndarray.squeeze.html. The axis is an
        ///     <see cref="NDAxis"/> since API-parity batch 4 (2026-11-12); an <c>int?</c> argument converts.</remarks>
        [PlainReceiver]
        public NDArray squeeze(NDAxis? axis = null)
        {
            return np.squeeze(this, axis);
        }

        /// <summary>
        ///     <see cref="squeeze(NDAxis?)"/> with the axis as one <c>int</c>: the overload a C# integer binds (its
        ///     identity conversion beats NDAxis' implicit one), forwarding to the NDAxis form, so every answer, error,
        ///     layout and subclass result is that overload's.
        /// </summary>
        /// <param name="axis">The axis (NumPy's <c>axis=k</c>; a negative value counts from the end).</param>
        /// <inheritdoc cref="squeeze(NDAxis?)"/>
        public NDArray squeeze(int axis)
            => squeeze((NDAxis?)axis);

        /// <summary>
        ///     <see cref="squeeze(NDAxis?)"/> with the axes as an <c>int[]</c>: the overload an <c>int[]</c> binds, and
        ///     a bare <c>null</c> too (null converts to <c>int[]</c> better than to NDAxis), forwarding to the NDAxis
        ///     form: a null array is NumPy's None, any other array NumPy's tuple (<c>new int[0]</c> is <c>()</c>).
        /// </summary>
        /// <param name="axis">The axes as NumPy's tuple, or <c>null</c> for None.</param>
        /// <inheritdoc cref="squeeze(NDAxis?)"/>
        public NDArray squeeze(int[] axis)
            => squeeze((NDAxis?)axis);
    }
}
