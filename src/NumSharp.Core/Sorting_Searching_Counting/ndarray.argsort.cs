using System;
using NumSharp.Backends.Sorting;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Returns the indices that would sort an array along the given axis.
        ///
        ///     Indirect sort: returns an int64 array of the same shape whose values index this
        ///     array along <paramref name="axis"/> in sorted order (NumPy <c>np.argsort</c>).
        ///     Stable (ties resolve in ascending index order). Floating NaN sorts to the end.
        /// </summary>
        /// <remarks>
        ///     Implementation: NpyIter drives the all-but-axis loop; each 1-D line is argsorted by
        ///     a stable LSD radix kernel (<see cref="AxisSort"/>). The generic parameter is retained
        ///     for source compatibility — the element type is taken from the array's own dtype.
        /// </remarks>
        public NDArray argsort<T>(int axis = -1) where T : unmanaged
            => AxisSort.ArgSort(this, axis);

        /// <summary>
        ///     Returns the indices that would sort this array along <paramref name="axis"/>
        ///     (null flattens). NumPy <c>np.argsort</c>.
        /// </summary>
        public NDArray argsort(int? axis = -1)
            => AxisSort.ArgSort(this, axis);
    }
}
