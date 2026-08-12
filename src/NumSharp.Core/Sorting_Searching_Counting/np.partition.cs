using NumSharp.Backends.Sorting;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return a partitioned copy of an array: the element at index <paramref name="kth"/>
        ///     lands in its final sorted position, everything smaller lies before it and everything
        ///     equal or greater behind it — the order WITHIN the two sides is undefined
        ///     (NumPy <c>np.partition</c>).
        /// </summary>
        /// <param name="a">Array to be partitioned.</param>
        /// <param name="kth">Element index to partition by; negative wraps from the end.</param>
        /// <param name="axis">Axis to partition along. -1 (default) = last axis; null flattens first.</param>
        /// <param name="kind">Selection algorithm — only 'introselect' exists, exactly like NumPy;
        ///     anything else raises NumPy's verbatim ValueError.</param>
        /// <param name="order">Structured-dtype field order — NumSharp has no structured dtypes, so any
        ///     non-null value raises NumPy's "Cannot specify order when the array has no fields."</param>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.partition.html<br></br>
        ///     Validation follows NumPy's probed order: kind → order → axis → kth ("kth(=N) out of
        ///     bounds (M)" reports the post-wrap value; an EMPTY array skips the kth bounds check).
        ///     NaN floats partition to the end (original bit patterns preserved); complex uses NumPy's
        ///     lexicographic real-then-imag ordering. The result is a fresh C-contiguous copy (house
        ///     np.sort convention — NumPy's copy(order='K') keeps F-order for F-inputs; values identical).
        /// </remarks>
        public static NDArray partition(NDArray a, int kth, int? axis = -1, string kind = "introselect", string order = null)
            => AxisPartition.Partition(a, new[] { kth }, axis, kind, order);

        /// <summary>
        ///     Return a partitioned copy of an array, partitioning around EVERY index in
        ///     <paramref name="kth"/> at once (each lands in its final sorted position; the ranges
        ///     between them are mutually ordered). NumPy <c>np.partition</c> with a kth sequence.
        /// </summary>
        /// <param name="a">Array to be partitioned.</param>
        /// <param name="kth">Element indices to partition by; negatives wrap. NumSharp reads an empty
        ///     array as NumPy's <c>np.array([], dtype=intp)</c> — a valid no-op returning a plain copy
        ///     (Python's bare <c>[]</c> is float64 and raises TypeError; a typed int[] is never ambiguous).</param>
        /// <param name="axis">Axis to partition along. -1 (default) = last axis; null flattens first.</param>
        /// <param name="kind">Selection algorithm — only 'introselect' exists.</param>
        /// <param name="order">Must stay null (no structured dtypes).</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.partition.html</remarks>
        public static NDArray partition(NDArray a, int[] kth, int? axis = -1, string kind = "introselect", string order = null)
            => AxisPartition.Partition(a, kth, axis, kind, order);

        /// <summary>
        ///     Return a partitioned copy with the kth indices given as an ARRAY — NumPy's array-kth
        ///     form, which is what makes its kth-dtype rejections reachable: a bool kth raises
        ///     "Booleans unacceptable as partition index", a non-integer kth "Partition index must be
        ///     integer" (TypeError), a &gt;1-D kth "object too deep for desired array" (checked BEFORE
        ///     the axis, like NumPy). Integer kth values cast to intp with NumPy's modular wrap
        ///     (uint64 past 2^63 goes negative; 2^64-1 is a legal -1). 0-d and any layout accepted.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.partition.html</remarks>
        public static NDArray partition(NDArray a, NDArray kth, int? axis = -1, string kind = "introselect", string order = null)
            => AxisPartition.Partition(a, kth, axis, kind, order);
    }

    public partial class NDArray
    {
        /// <summary>
        ///     Partition this array in place along <paramref name="axis"/> so the element at
        ///     <paramref name="kth"/> lands in its final sorted position (NumPy <c>ndarray.partition</c>;
        ///     null axis flattens in place — the same NumSharp extension <c>ndarray.sort</c> carries).
        /// </summary>
        public void partition(int kth, int? axis = -1, string kind = "introselect", string order = null)
            => AxisPartition.PartitionInPlace(this, new[] { kth }, axis, kind, order);

        /// <summary>
        ///     Partition this array in place around every index in <paramref name="kth"/> at once
        ///     (NumPy <c>ndarray.partition</c> with a kth sequence).
        /// </summary>
        public void partition(int[] kth, int? axis = -1, string kind = "introselect", string order = null)
            => AxisPartition.PartitionInPlace(this, kth, axis, kind, order);

        /// <summary>
        ///     Partition this array in place with the kth indices given as an ARRAY (NumPy's array-kth
        ///     form — see <see cref="np.partition(NDArray, NDArray, int?, string, string)"/> for its
        ///     dtype/too-deep rejections and wrap semantics).
        /// </summary>
        public void partition(NDArray kth, int? axis = -1, string kind = "introselect", string order = null)
            => AxisPartition.PartitionInPlace(this, kth, axis, kind, order);
    }
}
