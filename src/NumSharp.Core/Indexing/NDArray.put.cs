namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Set <c>this.flat[n] = values[n]</c> for all <c>n</c> in <paramref name="indices"/> — an in-place
        ///     operation (returns nothing). Refer to <see cref="np.put(NDArray, NDArray, NDArray, string)"/> for
        ///     full documentation.
        /// </summary>
        /// <param name="indices">Target indices into the flattened array. Scalars are accepted via implicit conversion.</param>
        /// <param name="values">Values to place at <paramref name="indices"/>. Broadcast/repeated to match the number of indices if necessary.</param>
        /// <param name="mode">How out-of-bounds indices behave: "raise" (default), "wrap", or "clip".</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ndarray.put.html</remarks>
        public void put(NDArray indices, NDArray values, string mode = "raise")
            => np.put(this, indices, values, mode);
    }
}
