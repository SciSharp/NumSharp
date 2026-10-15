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
        ///     Implementation: NDIter drives the all-but-axis loop; each 1-D line is argsorted by
        ///     a stable LSD radix kernel (<see cref="AxisSort"/>). The generic parameter is retained
        ///     for source compatibility — the element type is taken from the array's own dtype.
        /// </remarks>
        public NDArray argsort<T>(int axis = -1) where T : unmanaged
            => AxisSort.ArgSort(this, axis);

        /// <summary>
        ///     Returns the indices that would sort this array along <paramref name="axis"/> (null flattens). NumPy's
        ///     <c>ndarray.argsort(axis=-1, kind=None, order=None, *, stable=None)</c>.
        /// </summary>
        /// <param name="axis">Axis to sort along; null sorts the flattened array.</param>
        /// <param name="kind">
        ///     The sort algorithm, as NumPy parses it: the first letter decides, case-insensitively — <c>'q'</c>uicksort
        ///     (the default), <c>'h'</c>eapsort, <c>'m'</c>ergesort or <c>'s'</c>table. Over <c>S</c> / <c>U</c> text the
        ///     default and heapsort kinds give NumPy's introsort permutation (equal strings in its order, which is not their
        ///     input order once a line has more than 16 elements) and the stable kinds the stable permutation; NumSharp's
        ///     numeric argsort gives the stable permutation for every kind.
        /// </param>
        /// <param name="order">Field names; any value is refused (no NumSharp dtype has fields).</param>
        /// <param name="stable">NumPy's keyword-only <c>stable</c>: true asks for the stable kind. It cannot be combined
        ///     with <paramref name="kind"/> (false included).</param>
        /// <returns>A fresh, C-contiguous int64 array of indices.</returns>
        /// <exception cref="ValueError">An unknown <paramref name="kind"/>, <paramref name="kind"/> given with
        ///     <paramref name="stable"/>, or an <paramref name="order"/>.</exception>
        /// <exception cref="AxisError">The axis is out of bounds (a 0-d array ravels first, so it accepts 0 and -1).</exception>
        /// <exception cref="RuntimeError">A <c>StringDType</c> array with the heapsort kind (<c>unsupported sort kind</c>).</exception>
        /// <exception cref="StringComputationNotSupportedException">A <c>StringDType</c> array whose lines hold two or more
        ///     elements (the <c>StringDType</c> part of NumSharp's next phase).</exception>
        public NDArray argsort(int? axis = -1, string kind = null, object order = null, bool? stable = null)
            => AxisSort.ArgSort(this, axis, SortKindArguments.Resolve(kind, order, stable));
    }
}
