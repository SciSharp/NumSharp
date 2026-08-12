using NumSharp.Backends.Sorting;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Perform an indirect partition along the given axis: returns int64 indices of the same
        ///     shape that index <paramref name="a"/> along <paramref name="axis"/> in partitioned
        ///     order — <c>a[result[kth]]</c> is the element that would sit at <paramref name="kth"/>
        ///     in a sorted array (NumPy <c>np.argpartition</c>). The input is only read.
        /// </summary>
        /// <param name="a">Array to partition indirectly.</param>
        /// <param name="kth">Element index to partition by; negative wraps from the end.</param>
        /// <param name="axis">Axis to partition along. -1 (default) = last axis; null flattens first
        ///     (indices then address the flattened array).</param>
        /// <param name="kind">Selection algorithm — only 'introselect' exists.</param>
        /// <param name="order">Must stay null (no structured dtypes).</param>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.argpartition.html<br></br>
        ///     A 0-d input ravels to shape (1,) first — NumPy's arg-side quirk (its axis/kth errors
        ///     then report dimension/size 1, and the result is <c>[0]</c>), unlike np.partition which
        ///     raises AxisError on 0-d. NaN floats partition to the end (their indices keep encounter
        ///     order, the same policy as argsort's NaN tail).
        /// </remarks>
        public static NDArray argpartition(NDArray a, int kth, int? axis = -1, string kind = "introselect", string order = null)
            => AxisPartition.ArgPartition(a, new[] { kth }, axis, kind, order);

        /// <summary>
        ///     Indirect partition around EVERY index in <paramref name="kth"/> at once (NumPy
        ///     <c>np.argpartition</c> with a kth sequence). Returns int64 indices; the input is only read.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.argpartition.html</remarks>
        public static NDArray argpartition(NDArray a, int[] kth, int? axis = -1, string kind = "introselect", string order = null)
            => AxisPartition.ArgPartition(a, kth, axis, kind, order);
    }

    public partial class NDArray
    {
        /// <summary>
        ///     Returns the int64 indices that would partition this array along <paramref name="axis"/>
        ///     (NumPy <c>ndarray.argpartition</c>). This array is only read.
        /// </summary>
        public NDArray argpartition(int kth, int? axis = -1, string kind = "introselect", string order = null)
            => AxisPartition.ArgPartition(this, new[] { kth }, axis, kind, order);

        /// <summary>
        ///     Returns the int64 indices that would partition this array around every index in
        ///     <paramref name="kth"/> at once (NumPy <c>ndarray.argpartition</c>).
        /// </summary>
        public NDArray argpartition(int[] kth, int? axis = -1, string kind = "introselect", string order = null)
            => AxisPartition.ArgPartition(this, kth, axis, kind, order);
    }
}
