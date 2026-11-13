namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Return the elements at <paramref name="indices"/> taken along an axis. Refer to
        ///     <see cref="np.take(NDArray, NDArray, int?, NDArray, string)"/> for full documentation.
        /// </summary>
        /// <param name="indices">Indices of the values to extract. Integer arrays and a single C# integer (a 0-d index)
        ///     convert implicitly.</param>
        /// <param name="axis">The axis over which to select values. The default (null) works over the flattened array.</param>
        /// <param name="out">Optional output array of the right shape to hold the result.</param>
        /// <param name="mode">How out-of-bounds indices behave: "raise" (default), "wrap", or "clip".</param>
        /// <returns>The returned array has the same type as this array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ndarray.take.html</remarks>
        [PlainReceiver]
        public NDArray take(NDArray indices, int? axis = null, NDArray @out = null, string mode = "raise")
        {
            return np.take(this, indices, axis, @out, mode);
        }
    }
}
