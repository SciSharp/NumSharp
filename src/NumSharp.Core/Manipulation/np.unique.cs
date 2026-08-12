namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Find the unique elements of an array (the bare-return form of <c>np.unique</c>).<br></br>
        ///
        ///     Returns the sorted unique elements. This overload mirrors NumPy's single-array return
        ///     (all three <c>return_*</c> flags False): <c>np.unique(ar)</c>, <c>np.unique(ar, axis=0)</c>,
        ///     <c>np.unique(ar, equal_nan=False)</c> and <c>np.unique(ar, sorted=False)</c> all land here.
        ///     Pass any <c>return_index</c>/<c>return_inverse</c>/<c>return_counts</c> to reach the
        ///     <see cref="unique(NDArray, bool, bool, bool, int?, bool, bool)"/> tuple overload.
        /// </summary>
        /// <param name="ar">Input array. Unless <paramref name="axis"/> is given, it is flattened first.</param>
        /// <param name="axis">Axis to operate on. If <c>null</c> (default), the array is flattened.</param>
        /// <param name="equal_nan">If <c>true</c> (default), all NaNs collapse to a single output value;
        ///   if <c>false</c>, each NaN is distinct.</param>
        /// <param name="sorted">Accepted for NumPy 2.3 parity; NumSharp always returns sorted output
        ///   (see the tuple overload's remarks).</param>
        /// <returns>The sorted unique values.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.unique.html</remarks>
        public static NDArray unique(NDArray ar, int? axis = null, bool equal_nan = true, bool sorted = true)
            => ar.unique(axis, equal_nan, sorted);

        /// <summary>
        ///     Find the unique elements of an array with full NumPy keyword argument support.
        ///
        ///     Returns sorted unique elements; optionally returns first-occurrence indices,
        ///     reconstruction indices, and counts. Supports axis-aware uniqueness.
        /// </summary>
        /// <param name="ar">Input array.</param>
        /// <param name="return_index">If True, also return indices of <c>ar</c>
        ///   (along the specified axis, if provided) that result in the unique array.</param>
        /// <param name="return_inverse">If True, also return the indices of the unique array
        ///   that can be used to reconstruct <c>ar</c>.</param>
        /// <param name="return_counts">If True, also return the number of times each unique
        ///   item appears in <c>ar</c>.</param>
        /// <param name="axis">The axis to operate on. If <c>null</c>, the array is flattened first.</param>
        /// <param name="equal_nan">If True (default), all NaN values are considered equal so
        ///   only one appears in the output. If False, each NaN is treated as unique.</param>
        /// <param name="sorted">If True (default), the unique elements are sorted (NumPy 2.3).
        ///   NumSharp always returns sorted output: NumPy's <c>sorted=False</c> hash-iteration order
        ///   for integer/complex values is platform-specific and not reproducible in C#, so this
        ///   parameter exists for API parity but does not change the result (spec-compliant — the
        ///   Array API leaves the <c>sorted=False</c> order unspecified). See <see cref="unique_values"/>.</param>
        /// <returns>An array of NDArrays in order: [values, index?, inverse?, counts?].
        ///   The first element is always the sorted unique values; remaining elements are
        ///   present only when the corresponding flag is True.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.unique.html</remarks>
        public static NDArray[] unique(NDArray ar,
            bool return_index,
            bool return_inverse = false,
            bool return_counts = false,
            int? axis = null,
            bool equal_nan = true,
            bool sorted = true)
            => ar.unique(return_index, return_inverse, return_counts, axis, equal_nan, sorted);
    }
}
