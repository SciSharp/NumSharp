namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Efficient multi-dimensional iterator object to iterate over arrays.
        /// </summary>
        /// <param name="op">The array to iterate over.</param>
        /// <param name="flags">
        ///     Flags to control iterator behavior:
        ///     - MultiIndex: Track N-D coordinates via multi_index property
        ///     - ReadWrite: Allow modifying array elements via value property
        ///     - ReadOnly: Read-only iteration (default)
        /// </param>
        /// <returns>An NdIter object for iterating over the array.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.nditer.html
        ///
        ///     The iterator provides:
        ///     - finished: whether iteration is complete
        ///     - iternext(): advance to next element
        ///     - value: current element value (get/set with ReadWrite flag)
        ///     - multi_index: current N-D coordinates (with MultiIndex flag)
        ///     - index: current flat index
        ///     - foreach enumeration support
        /// </remarks>
        /// <example>
        ///     // Basic iteration
        ///     var it = np.nditer(arr);
        ///     while (!it.finished)
        ///     {
        ///         Console.WriteLine(it.value);
        ///         it.iternext();
        ///     }
        ///
        ///     // With multi-index
        ///     var it = np.nditer(arr, NdIterFlags.MultiIndex);
        ///     while (!it.finished)
        ///     {
        ///         Console.WriteLine($"arr[{string.Join(",", it.multi_index)}] = {it.value}");
        ///         it.iternext();
        ///     }
        ///
        ///     // Read-write modification
        ///     using var it = np.nditer(arr, NdIterFlags.ReadWrite);
        ///     while (!it.finished)
        ///     {
        ///         it.value = (int)it.value * 2;
        ///         it.iternext();
        ///     }
        ///
        ///     // Foreach enumeration
        ///     foreach (var x in np.nditer(arr))
        ///         Console.WriteLine(x);
        /// </example>
        public static NdIter nditer(NDArray op, NdIterFlags flags = NdIterFlags.None)
        {
            return new NdIter(op, flags);
        }

        /// <summary>
        ///     Efficient multi-dimensional iterator with typed access.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="op">The array to iterate over.</param>
        /// <param name="flags">Flags to control iterator behavior.</param>
        /// <returns>A typed NdIter object.</returns>
        /// <example>
        ///     var it = np.nditer&lt;int&gt;(arr, NdIterFlags.ReadWrite);
        ///     while (!it.finished)
        ///     {
        ///         it.value *= 2;  // No casting needed
        ///         it.iternext();
        ///     }
        /// </example>
        public static NdIter<T> nditer<T>(NDArray op, NdIterFlags flags = NdIterFlags.None) where T : unmanaged
        {
            return new NdIter<T>(op, flags);
        }
    }
}
