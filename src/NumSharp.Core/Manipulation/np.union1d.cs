namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Find the union of two arrays.<br></br>
        ///     Return the unique, sorted array of values that are in either of the two input arrays.
        /// </summary>
        /// <param name="ar1">Input array (flattened if not already 1-D).</param>
        /// <param name="ar2">Input array (flattened if not already 1-D).</param>
        /// <returns>Unique, sorted union of the input arrays.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.union1d.html</remarks>
        [NDScoped]
        public static NDArray union1d(NDArray ar1, NDArray ar2)
        {
            // Boundary scope: the ravel views, the concatenate buffer and unique's internals are
            // reclaimed at exit; only the final (possibly NaN-canonicalised) array is yielded.
            // NumPy: unique(concatenate((ar1, ar2), axis=None)) — axis=None flattens both operands.
            // NumPy's unique-sort canonicalises a surviving float32/float64 NaN; match it (see np.setops.cs).
            return CanonicalizeSetOpNaN(np.unique(np.concatenate((np.ravel(ar1), np.ravel(ar2)), 0)));
        }
    }
}
