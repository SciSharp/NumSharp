using NumSharp.Backends.Sorting;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return a sorted copy of an array, sorted along <paramref name="axis"/>
        ///     (default last; <c>null</c> flattens first). NumPy <c>np.sort</c>.
        /// </summary>
        /// <param name="a">Array to sort.</param>
        /// <param name="axis">Axis to sort along. -1 = last axis. null = sort the flattened array.</param>
        /// <param name="kind">
        ///     Sort algorithm name (NumPy compatibility). All kinds produce identical sorted output;
        ///     the kernel is a stable LSD radix (numeric) or BCL introsort (Half/Complex/Decimal).
        /// </param>
        /// <remarks>
        ///     NaN floats sort to the end; complex sorts lexicographically (real then imaginary),
        ///     any-NaN-part last — matching NumPy 2.4.2.
        /// </remarks>
        public static NDArray sort(NDArray a, int? axis = -1, string kind = null)
            => AxisSort.Sort(a, axis);
    }

    public partial class NDArray
    {
        /// <summary>
        ///     Sort this array in place along <paramref name="axis"/> (default last;
        ///     <c>null</c> flattens). NumPy <c>ndarray.sort</c>.
        /// </summary>
        public void sort(int? axis = -1, string kind = null)
            => AxisSort.SortInPlace(this, axis);
    }
}
