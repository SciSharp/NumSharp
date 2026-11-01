using NumSharp.Backends.Sorting;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's <c>fromnumeric.sort</c> and <c>PyArray_Sort</c>: the kind / stable / order arguments in NumPy's order
        ///     (<see cref="SortKindArguments.Resolve"/>, before the array is read), then the sorted copy
        ///     (<see cref="AxisSort.Sort(NDArray, int?, NPY_SORTKIND)"/>: <c>copy(order='K')</c> sorted in place, the
        ///     text and StringDType routes included). Moved from the public <c>np.sort</c> by the np-layer dtype plan's
        ///     phase PE (E5); AxisSort stays the kernel, and the public function keeps its class boundaries.
        /// </remarks>
        public override NDArray Sort(NDArray a, int? axis = -1, string kind = null, object order = null, bool? stable = null)
            => AxisSort.Sort(a, axis, SortKindArguments.Resolve(kind, order, stable));

        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's <c>ndarray.sort</c> (<c>array_sort</c> over <c>PyArray_Sort</c>): the arguments resolved first, then
        ///     <see cref="AxisSort.SortInPlace(NDArray, int?, NPY_SORTKIND)"/> (the axis, then writeability, then the
        ///     sort). Moved from <c>NDArray.sort</c> by the np-layer dtype plan's phase PE (E5).
        /// </remarks>
        public override void SortInPlace(NDArray a, int? axis = -1, string kind = null, object order = null, bool? stable = null)
            => AxisSort.SortInPlace(a, axis, SortKindArguments.Resolve(kind, order, stable));

        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's <c>PyArray_ArgSort</c> behind <c>np.argsort</c> and <c>ndarray.argsort</c>: the arguments resolved
        ///     first, then <see cref="AxisSort.ArgSort(NDArray, int?, NPY_SORTKIND)"/> (StringDType's heapsort refusal before
        ///     <c>PyArray_CheckAxis</c>, which ravels a 0-d array). Moved from <c>NDArray.argsort</c> by the np-layer dtype
        ///     plan's phase PE (E5).
        /// </remarks>
        public override NDArray ArgSort(NDArray a, int? axis = -1, string kind = null, object order = null, bool? stable = null)
            => AxisSort.ArgSort(a, axis, SortKindArguments.Resolve(kind, order, stable));

        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's <c>fromnumeric.partition</c> and <c>PyArray_Partition</c> over
        ///     <see cref="AxisPartition.Partition(NDArray, int[], int?, string, string)"/>, which checks the kind, the
        ///     order, the axis and then the kth in NumPy's order. The kth forms dispatch by
        ///     <see cref="PartitionKth"/>. Moved from the public <c>np.partition</c> by the np-layer dtype plan's phase PE
        ///     (E5); AxisPartition stays the kernel, and the public function keeps its class boundaries.
        /// </remarks>
        public override NDArray Partition(NDArray a, object kth, int? axis = -1, string kind = "introselect", string order = null)
        {
            switch (PartitionKth(kth, out int[] positions, out NDArray index, out NDArray converted))
            {
                case KthForm.Positions:
                    return AxisPartition.Partition(a, positions, axis, kind, order);
                default:
                    // An index array (the caller's, or the one np.asanyarray made, disposed here: the kernel reads the
                    // positions out of it and keeps none of it).
                    using (converted)
                        return AxisPartition.Partition(a, index, axis, kind, order);
            }
        }

        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's <c>ndarray.partition</c> (<c>array_partition</c> over <c>PyArray_Partition</c>) over
        ///     <see cref="AxisPartition.PartitionInPlace(NDArray, int[], int?, string, string)"/>: the kind, the order, the
        ///     axis, writeability, then the kth. Moved from <c>NDArray.partition</c> by the np-layer dtype plan's phase PE
        ///     (E5).
        /// </remarks>
        public override void PartitionInPlace(NDArray a, object kth, int? axis = -1, string kind = "introselect", string order = null)
        {
            switch (PartitionKth(kth, out int[] positions, out NDArray index, out NDArray converted))
            {
                case KthForm.Positions:
                    AxisPartition.PartitionInPlace(a, positions, axis, kind, order);
                    return;
                default:
                    using (converted)
                        AxisPartition.PartitionInPlace(a, index, axis, kind, order);
                    return;
            }
        }

        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's <c>PyArray_ArgPartition</c> behind <c>np.argpartition</c> and <c>ndarray.argpartition</c>, over
        ///     <see cref="AxisPartition.ArgPartition(NDArray, int[], int?, string, string)"/> (a 0-d array ravels to one
        ///     element first). Moved from the public <c>np.argpartition</c> and <c>NDArray.argpartition</c> by the np-layer
        ///     dtype plan's phase PE (E5).
        /// </remarks>
        public override NDArray ArgPartition(NDArray a, object kth, int? axis = -1, string kind = "introselect", string order = null)
        {
            switch (PartitionKth(kth, out int[] positions, out NDArray index, out NDArray converted))
            {
                case KthForm.Positions:
                    return AxisPartition.ArgPartition(a, positions, axis, kind, order);
                default:
                    using (converted)
                        return AxisPartition.ArgPartition(a, index, axis, kind, order);
            }
        }

        /// <summary>Which of AxisPartition's two kth forms a partition member's kth takes.</summary>
        private enum KthForm
        {
            /// <summary>C# positions: an int (one position) or an int[] (none, one or several), or null.</summary>
            Positions,

            /// <summary>An index array, whose dtype NumPy checks (bool and non-integer dtypes are refused).</summary>
            IndexArray,
        }

        /// <summary>
        ///     Sorts a partition member's <c>kth</c> (NumPy's array_like of integers) into the form AxisPartition takes: a C#
        ///     <see cref="int"/> or <see cref="int"/>[] as positions, an <see cref="NDArray"/> as an index array, and any other
        ///     object converted with <c>np.asanyarray</c> to an index array (so a <c>long</c>, a
        ///     <c>long[]</c> or a list reach NumPy's dtype checks as they would in NumPy).
        /// </summary>
        /// <param name="kth">The kth argument as the caller passed it.</param>
        /// <param name="positions">The positions, for <see cref="KthForm.Positions"/>: <paramref name="kth"/> as an
        ///     int[] (a one-element array for an int), or null for a null kth.</param>
        /// <param name="index">The index array, for <see cref="KthForm.IndexArray"/>.</param>
        /// <param name="converted">The index array this method made (np.asanyarray of another object), which the caller
        ///     disposes after the kernel has read it; null when the index array is the caller's own.</param>
        /// <returns>The form.</returns>
        /// <remarks>
        ///     A null kth takes the positions form, whose ArgumentNullException fires at NumPy's kth stage, after the kind,
        ///     the order and the axis (NumPy's kth=None raises its TypeError there); the index-array form lets a null ride
        ///     through to the same stage, so the two C# spellings of a null kth behave alike.
        /// </remarks>
        private static KthForm PartitionKth(object kth, out int[] positions, out NDArray index, out NDArray converted)
        {
            positions = null;
            index = null;
            converted = null;
            switch (kth)
            {
                case null:
                    return KthForm.Positions;
                case int k:
                    positions = new[] { k };
                    return KthForm.Positions;
                case int[] ks:
                    positions = ks;
                    return KthForm.Positions;
                case NDArray nd:
                    index = nd;
                    return KthForm.IndexArray;
                default:
                    // np.asanyarray(kth)'s statement (an NDArray kth is the arm above, so no class boundary applies).
                    index = converted = NDArray.asanyarray(kth, null, 'K');
                    return KthForm.IndexArray;
            }
        }
    }
}
