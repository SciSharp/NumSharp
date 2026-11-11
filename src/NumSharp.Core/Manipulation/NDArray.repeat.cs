namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Repeat each element of the array by the per-element counts in <paramref name="repeats"/>. Refer to
        ///     <see cref="np.repeat(NDArray, NDArray, int?)"/> for full documentation.
        /// </summary>
        /// <param name="repeats">Per-element repetition counts, broadcast to the shape along the given axis.</param>
        /// <param name="axis">The axis along which to repeat values. The default (null) flattens the input array and returns a flat output array.</param>
        /// <returns>Output array which has the same shape as this array, except along the given axis.</returns>
        /// <exception cref="ArgumentException">A negative count, or counts that do not match the axis.</exception>
        /// <exception cref="TypeError">Counts that do not cast 'safely' to int64.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ndarray.repeat.html
        ///     <para>One signature, NumPy's <c>a.repeat(repeats, axis=None)</c>: the <c>int</c> / <c>long</c> overloads
        ///     went with API-parity batch 3 (2026-11-11); <c>a.repeat(3)</c> binds here (3 converts to a 0-d array).</para>
        /// </remarks>
        [PlainReceiver]
        public NDArray repeat(NDArray repeats, int? axis = null)
        {
            return np.repeat(this, repeats, axis);
        }
    }
}
