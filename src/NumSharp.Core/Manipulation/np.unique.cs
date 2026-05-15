namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Find the unique elements of an array.<br></br>
        ///
        ///     Returns the sorted unique elements of an array.There are three optional outputs in addition to the unique elements:<br></br>
        ///     * the indices of the input array that give the unique values<br></br>
        ///     * the indices of the unique array that reconstruct the input array<br></br>
        ///     * the number of times each unique value comes up in the input array<br></br>
        /// </summary>
        /// <returns>The sorted unique values.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.unique.html</remarks>
        public static NDArray unique(NDArray ar)
            => ar.unique();

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
        /// <returns>An array of NDArrays in order: [values, index?, inverse?, counts?].
        ///   The first element is always the sorted unique values; remaining elements are
        ///   present only when the corresponding flag is True.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.unique.html</remarks>
        public static NDArray[] unique(NDArray ar,
            bool return_index,
            bool return_inverse = false,
            bool return_counts = false,
            int? axis = null,
            bool equal_nan = true)
            => ar.unique(return_index, return_inverse, return_counts, axis, equal_nan);
    }
}
