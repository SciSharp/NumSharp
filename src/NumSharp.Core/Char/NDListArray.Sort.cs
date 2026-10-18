using System;
using System.Collections.Generic;
using NumSharp.Backends;
using NumSharp.Backends.Sorting;
using NumSharp.Backends.Strings;

// =============================================================================
// NDListArray.Sort.cs — sorting, selection and search over NumPy's object array of lists: sort / argsort / partition /
// argpartition / searchsorted / argmax / argmin
// =============================================================================
//
// What NumPy runs for an OBJECT array (numpy/_core/src/multiarray/item_selection.c, npysort; probed against 2.4.2):
//
//   * sort / argsort take the GENERIC sorts with OBJECT_compare (Py_LT, then Py_GT): the default kind is npy_quicksort /
//     npy_aquicksort (an introsort: median of three, insertion sort below 17 elements, a heapsort fallback once the depth
//     budget 2 * msb(n) is spent — NOT stable, so the order of equal lists is the algorithm's), 'heapsort' maps to the
//     default (PyArray_Sort clears the heapsort bit), and the stable kind is npy_timsort (stable);
//   * partition / argpartition have no selection function for objects: PyArray_(Arg)Partition sorts the whole line with
//     the default kind, after their own checks (the kind must be 'introselect', kth integers within the axis);
//   * searchsorted is npy_binsearch with OBJECT_compare over a 1-D haystack;
//   * argmax / argmin are OBJECT_argmax / OBJECT_argmin: the FIRST element no later one is greater (smaller) than.
//
// Lists of one heap are totally ordered (one piece class), so every one of these is an algorithm over the lists' dense
// RANKS: equal lists one rank, the order of the ranks the order of the lists. The default-kind permutation is NumPy's
// introsort ported statement for statement over (rank, index) records (TextSort.IntroArgSortRecords, which the string
// sorts use: NumPy's string_aquicksort_ and npy_aquicksort_impl give the same permutation for a consistent comparison);
// the stable kind is the stable order of the ranks; argmax / argmin are NDArray's over the ranks (its first-occurrence
// rule is OBJECT_argmax's). A sort moves ids (NumPy's pointers) in place.
// =============================================================================

namespace NumSharp
{
    public sealed partial class NDListArray
    {
        // =============================================================================================================
        // sort / argsort
        // =============================================================================================================

        /// <summary>
        ///     NumPy's in-place <c>sort(axis=-1, kind=None, order=None, *, stable=None)</c>: every line along
        ///     <paramref name="axis"/> ordered by Python's list order — the default kind with NumPy's introsort (the order of
        ///     EQUAL lists is the algorithm's, not their input order), the stable kind keeping equal lists in order.
        /// </summary>
        /// <param name="axis">The axis (negative from the end).</param>
        /// <param name="kind"><c>'quicksort'</c> / <c>'heapsort'</c> (the default sort for objects), <c>'mergesort'</c> /
        ///     <c>'stable'</c>; NumPy's parser reads the first letter.</param>
        /// <param name="order">Must be null (an object array has no fields).</param>
        /// <param name="stable">NumPy's keyword-only <c>stable=</c>: true selects the stable kind.</param>
        /// <exception cref="ValueError">An unknown kind, <paramref name="kind"/> with <paramref name="stable"/>, an
        ///     <paramref name="order"/>, or a read-only array (<c>sort array is read-only</c>).</exception>
        /// <exception cref="AxisError">An axis outside the array.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public void sort(int axis = -1, string kind = null, object order = null, bool? stable = null)
        {
            ThrowIfDisposed();
            var sortKind = SortKindArguments.Resolve(kind, order, stable);
            int ax = DefaultEngine.check_and_adjust_axis(ndim, axis);
            if (!Ids.Shape.IsWriteable)
                throw new ValueError("sort array is read-only");
            SortInPlace(ax, sortKind == NPY_SORTKIND.NPY_STABLESORT);
        }

        /// <summary>
        ///     NumPy's <c>argsort(axis=-1, kind=None, order=None, *, stable=None)</c>: the int64 indices that sort every line
        ///     (C-ordered, the array's shape); <paramref name="axis"/> null sorts the flattened (C-order) array.
        /// </summary>
        /// <param name="axis">The axis, or null for the flattened array.</param>
        /// <param name="kind">See <see cref="sort"/>.</param>
        /// <param name="order">Must be null.</param>
        /// <param name="stable">NumPy's <c>stable=</c>.</param>
        /// <returns>The indices.</returns>
        /// <exception cref="ValueError">An unknown kind, <paramref name="kind"/> with <paramref name="stable"/>, or an
        ///     <paramref name="order"/>.</exception>
        /// <exception cref="AxisError">An axis outside the array.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDArray argsort(int? axis = -1, string kind = null, object order = null, bool? stable = null)
        {
            ThrowIfDisposed();
            var sortKind = SortKindArguments.Resolve(kind, order, stable);
            return ArgSortCore(axis, sortKind == NPY_SORTKIND.NPY_STABLESORT);
        }

        // =============================================================================================================
        // partition / argpartition
        // =============================================================================================================

        /// <summary>
        ///     NumPy's in-place <c>partition(kth, axis=-1, kind='introselect', order=None)</c>: an object array has no
        ///     selection function, so after NumPy's checks every line is SORTED with the default kind (each kth element is
        ///     then in its final place, as the partition promises).
        /// </summary>
        /// <param name="kth">The index (negative from the end) whose element lands in its sorted place.</param>
        /// <param name="axis">The axis.</param>
        /// <param name="kind">Must be <c>'introselect'</c>.</param>
        /// <param name="order">Must be null.</param>
        /// <exception cref="ValueError"><c>select kind must be 'introselect' (got 'x')</c>, an order, a read-only array
        ///     (<c>partition array is read-only</c>), or a kth outside the axis (<c>kth(=5) out of bounds (4)</c>).</exception>
        /// <exception cref="AxisError">An axis outside the array.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public void partition(long kth, int axis = -1, string kind = "introselect", string order = null)
            => PartitionCore(new[] { kth }, axis, kind, order);

        /// <summary>NumPy's in-place <c>partition</c> with several kth indices (see <see cref="partition(long, int, string, string)"/>).</summary>
        /// <param name="kth">The indices.</param>
        /// <param name="axis">The axis.</param>
        /// <param name="kind">Must be <c>'introselect'</c>.</param>
        /// <param name="order">Must be null.</param>
        /// <exception cref="ArgumentNullException"><paramref name="kth"/> is null.</exception>
        /// <exception cref="ValueError">See <see cref="partition(long, int, string, string)"/>.</exception>
        /// <exception cref="AxisError">An axis outside the array.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public void partition(long[] kth, int axis = -1, string kind = "introselect", string order = null)
            => PartitionCore(kth, axis, kind, order);

        /// <summary>NumPy's in-place <c>partition</c> with an array kth: NumPy's dtype rules (bool refused, a non-integer
        ///     array a TypeError, more than one dimension refused).</summary>
        /// <param name="kth">The kth array.</param>
        /// <param name="axis">The axis.</param>
        /// <param name="kind">Must be <c>'introselect'</c>.</param>
        /// <param name="order">Must be null.</param>
        /// <exception cref="ArgumentNullException"><paramref name="kth"/> is null.</exception>
        /// <exception cref="ValueError">See <see cref="partition(long, int, string, string)"/>; <c>Booleans unacceptable as
        ///     partition index</c>; <c>object too deep for desired array</c>.</exception>
        /// <exception cref="TypeError"><c>Partition index must be integer</c>.</exception>
        /// <exception cref="AxisError">An axis outside the array.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public void partition(NDArray kth, int axis = -1, string kind = "introselect", string order = null)
        {
            ThrowIfDisposed();
            CheckSelectKind(kind, order);
            CheckKthDepth(kth);
            PartitionCore(KthValues(kth), axis, kind, order);
        }

        /// <summary>
        ///     NumPy's <c>argpartition(kth, axis=-1, kind='introselect', order=None)</c>: after NumPy's checks, the default
        ///     argsort of every line (an object array has no selection function); a 0-d array is ravelled to one element
        ///     first, as NumPy's argument-side axis check does.
        /// </summary>
        /// <param name="kth">The index.</param>
        /// <param name="axis">The axis, or null for the flattened array.</param>
        /// <param name="kind">Must be <c>'introselect'</c>.</param>
        /// <param name="order">Must be null.</param>
        /// <returns>The int64 indices.</returns>
        /// <exception cref="ValueError">The kind, an order, or a kth outside the axis.</exception>
        /// <exception cref="AxisError">An axis outside the array.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDArray argpartition(long kth, int? axis = -1, string kind = "introselect", string order = null)
            => ArgPartitionCore(new[] { kth }, axis, kind, order);

        /// <summary>NumPy's <c>argpartition</c> with several kth indices.</summary>
        /// <param name="kth">The indices.</param>
        /// <param name="axis">The axis, or null for the flattened array.</param>
        /// <param name="kind">Must be <c>'introselect'</c>.</param>
        /// <param name="order">Must be null.</param>
        /// <returns>The int64 indices.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="kth"/> is null.</exception>
        /// <exception cref="ValueError">The kind, an order, or a kth outside the axis.</exception>
        /// <exception cref="AxisError">An axis outside the array.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDArray argpartition(long[] kth, int? axis = -1, string kind = "introselect", string order = null)
            => ArgPartitionCore(kth, axis, kind, order);

        /// <summary>NumPy's <c>argpartition</c> with an array kth (NumPy's dtype rules).</summary>
        /// <param name="kth">The kth array.</param>
        /// <param name="axis">The axis, or null for the flattened array.</param>
        /// <param name="kind">Must be <c>'introselect'</c>.</param>
        /// <param name="order">Must be null.</param>
        /// <returns>The int64 indices.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="kth"/> is null.</exception>
        /// <exception cref="ValueError">The kind, an order, a bool kth, a too-deep kth, or a kth outside the axis.</exception>
        /// <exception cref="TypeError">A non-integer kth array.</exception>
        /// <exception cref="AxisError">An axis outside the array.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDArray argpartition(NDArray kth, int? axis = -1, string kind = "introselect", string order = null)
        {
            ThrowIfDisposed();
            CheckSelectKind(kind, order);
            CheckKthDepth(kth);
            return ArgPartitionCore(KthValues(kth), axis, kind, order);
        }

        // =============================================================================================================
        // searchsorted
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>searchsorted(v, side='left', sorter=None)</c> over a 1-D array of lists sorted in Python's list
        ///     order: for each list of <paramref name="v"/>, the index where it would be inserted (before equal lists for
        ///     <c>'left'</c>, after them for <c>'right'</c>), through <paramref name="sorter"/> when given.
        /// </summary>
        /// <param name="v">The lists to place (any shape: the result has its shape).</param>
        /// <param name="side"><c>"left"</c> or <c>"right"</c>.</param>
        /// <param name="sorter">int64 indices that sort this array, or null.</param>
        /// <returns>The int64 insertion points.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="v"/> is null.</exception>
        /// <exception cref="ValueError">An unknown side, a haystack of more than one dimension (<c>object too deep for desired
        ///     array</c>), a sorter of another size, or a sorter index outside the array (<c>Sorter index out of
        ///     range.</c>).</exception>
        /// <exception cref="TypeError">A str list item compared with a bytes one; a sorter that is not 1-D integers.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDArray searchsorted(NDListArray v, string side = "left", NDArray sorter = null)
        {
            ThrowIfDisposed();
            if (v is null)
                throw new ArgumentNullException(nameof(v));
            v.ThrowIfDisposed();
            if (v.IsStandIn)
            {
                // np.asarray(v) of a Python list: the needles are its items.
                using var items = v.ItemsArray(v.FlatId(0));
                return searchsorted((object)items, side, sorter);
            }
            bool right = SearchSide(side);
            CheckHaystack(sorter);
            var result = new NDArray(NPTypeCode.Int64, Shape.Adopt((long[])v.Ids.Shape.dimensions.Clone()), false);
            try
            {
                using var keys = v.Ids.flatten();
                using var hay = Ids.flatten();
                long[] order = SorterIndices(sorter, hay.size);
                long n = hay.size;
                // npy_binsearch / npy_argbinsearch (binsearch.cpp): the bounds are CARRIED from key to key — a key not below
                // the previous one keeps the lower bound, any other restarts from 0 with the upper bound one past the last
                // answer — so over an unsorted haystack the answers are NumPy's, not a fresh search's. The comparisons are
                // OBJECT_compare's (Py_LT, then Py_GT), side 'left' taking compare < 0 and 'right' compare <= 0.
                long minIdx = 0, maxIdx = n;
                long lastKey = keys.size > 0 ? keys.GetInt64(0) : 0;
                for (long i = 0; i < keys.size; i++)
                {
                    long key = keys.GetInt64(i);
                    int step = CompareLists(v.Heap, lastKey, v.Heap, key);
                    if (right ? step <= 0 : step < 0)
                    {
                        maxIdx = n;
                    }
                    else
                    {
                        minIdx = 0;
                        maxIdx = maxIdx < n ? maxIdx + 1 : n;
                    }
                    lastKey = key;
                    while (minIdx < maxIdx)
                    {
                        long mid = minIdx + ((maxIdx - minIdx) >> 1);
                        long at = order is null ? mid : order[mid];
                        if (at < 0 || at >= n)
                            throw new ValueError("Sorter index out of range.");
                        int c = CompareLists(Heap, hay.GetInt64(at), v.Heap, key);
                        if (right ? c <= 0 : c < 0)
                            minIdx = mid + 1;
                        else
                            maxIdx = mid;
                    }
                    result.SetAtIndex(minIdx, i);
                }
            }
            catch
            {
                result.Dispose();
                throw;
            }
            return result;
        }

        /// <summary>
        ///     NumPy's <c>searchsorted</c> with a value that is not a list array: NumPy converts it with <c>np.asarray</c>
        ///     and compares the array's lists with its elements, CPython's TypeError for the first comparison (none is made
        ///     when the haystack or the value is empty: zeros of the value's shape).
        /// </summary>
        /// <param name="v">The value (a C# value as a Python one).</param>
        /// <param name="side"><c>"left"</c> or <c>"right"</c>.</param>
        /// <param name="sorter">int64 indices, or null.</param>
        /// <returns>Zeros of the value's shape when nothing is compared.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="v"/> is null.</exception>
        /// <exception cref="ValueError">An unknown side, a haystack of more than one dimension, a bad sorter.</exception>
        /// <exception cref="TypeError"><c>'&lt;' not supported between instances of 'list' and 'str'</c>.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDArray searchsorted(object v, string side = "left", NDArray sorter = null)
        {
            if (v is NDListArray lists)
                return searchsorted(lists, side, sorter);
            ThrowIfDisposed();
            if (v is null)
                throw new ArgumentNullException(nameof(v));
            bool right = SearchSide(side);
            CheckHaystack(sorter);
            NDArray keys = v as NDArray;
            NDArray owned = keys is null ? PythonOperand(v) : null;
            using (owned)
            {
                keys ??= owned;
                SorterIndices(sorter, size);
                if (size == 0 || keys.size == 0)
                    return new NDArray(NPTypeCode.Int64, Shape.Adopt((long[])keys.Shape.dimensions.Clone()), true);
                // OBJECT_compare(list, key) runs Py_LT first.
                _ = right;
                throw new TypeError($"'<' not supported between instances of 'list' and '{PyElementTypeName(keys.dtype)}'");
            }
        }

        // =============================================================================================================
        // argmax / argmin
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>argmax(axis=None, out=None, *, keepdims=False)</c>: the index of the FIRST greatest list (Python's
        ///     list order) of the flattened array or along <paramref name="axis"/> — NDArray's argmax over the lists' ranks,
        ///     whose first-occurrence rule is OBJECT_argmax's.
        /// </summary>
        /// <param name="axis">The axis, or null for the flattened array.</param>
        /// <param name="out">NumPy's <c>out=</c> (an int64-castable array of the result's shape).</param>
        /// <param name="keepdims">Keep the reduced axis as length 1.</param>
        /// <returns>A 0-d int64 for <c>axis=None</c>, else the int64 indices (or <paramref name="out"/>).</returns>
        /// <exception cref="ValueError">An empty array (<c>attempt to get argmax of an empty sequence</c>), or NDArray's out=
        ///     errors.</exception>
        /// <exception cref="AxisError">An axis outside the array.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDArray argmax(int? axis = null, NDArray @out = null, bool keepdims = false)
        {
            using var ranks = Ranks();
            return ranks.argmax(axis, @out, keepdims);
        }

        /// <summary>NumPy's <c>argmin(axis=None, out=None, *, keepdims=False)</c>: the index of the FIRST smallest list (see
        ///     <see cref="argmax"/>).</summary>
        /// <param name="axis">The axis, or null for the flattened array.</param>
        /// <param name="out">NumPy's <c>out=</c>.</param>
        /// <param name="keepdims">Keep the reduced axis as length 1.</param>
        /// <returns>A 0-d int64 for <c>axis=None</c>, else the int64 indices (or <paramref name="out"/>).</returns>
        /// <exception cref="ValueError">An empty array (<c>attempt to get argmin of an empty sequence</c>).</exception>
        /// <exception cref="AxisError">An axis outside the array.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDArray argmin(int? axis = null, NDArray @out = null, bool keepdims = false)
        {
            using var ranks = Ranks();
            return ranks.argmin(axis, @out, keepdims);
        }

        // =============================================================================================================
        // Internals
        // =============================================================================================================

        /// <summary>
        ///     Each element's dense rank in Python's list order (equal lists one rank), as a fresh C-ordered int64 array of
        ///     the outer shape: the integers every sorting member orders by.
        /// </summary>
        /// <returns>The ranks.</returns>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        internal NDArray Ranks()
        {
            ThrowIfDisposed();
            using var flat = Ids.flatten();
            long n = flat.size;
            var result = new NDArray(NPTypeCode.Int64, Shape.Adopt((long[])Ids.Shape.dimensions.Clone()), false);
            if (n == 0)
                return result;
            // The distinct lists ranked once (many elements can point at one list after views, repeats and assignments).
            var distinct = new List<long>();
            var slot = new Dictionary<long, int>();
            for (long i = 0; i < n; i++)
            {
                long id = flat.GetInt64(i);
                if (!slot.ContainsKey(id))
                {
                    slot[id] = distinct.Count;
                    distinct.Add(id);
                }
            }
            var ids = distinct.ToArray();
            var ranks = new long[ids.Length];
            DenseRanks(Heap, ids, ranks, new long[ids.Length]);
            for (long i = 0; i < n; i++)
                result.SetAtIndex(ranks[slot[flat.GetInt64(i)]], i);
            return result;
        }

        /// <summary>
        ///     The permutation that sorts one line of ranks: NumPy's introsort (the default kind) or the stable order.
        /// </summary>
        /// <param name="ranks">The line's ranks, in line order.</param>
        /// <param name="stable">True for the stable kind.</param>
        /// <returns>For each sorted position, the line index of its element.</returns>
        private static unsafe long[] SortPermutation(long[] ranks, bool stable)
        {
            long n = ranks.Length;
            var perm = new long[n];
            if (stable)
            {
                for (long i = 0; i < n; i++)
                    perm[i] = i;
                // The stable order: by rank, equal ranks in line order.
                Array.Sort(perm, (x, y) => ranks[x] != ranks[y] ? ranks[x].CompareTo(ranks[y]) : x.CompareTo(y));
                return perm;
            }
            // npy_aquicksort over (rank, index) records in NumPy's initial order (index i at record i).
            var rec = new ulong[2 * n];
            for (long i = 0; i < n; i++)
            {
                rec[2 * i] = (ulong)ranks[i];
                rec[2 * i + 1] = (ulong)i;
            }
            fixed (ulong* p = rec)
                TextSort.IntroArgSortRecords(p, n);
            for (long i = 0; i < n; i++)
                perm[i] = (long)rec[2 * i + 1];
            return perm;
        }

        /// <summary>
        ///     Calls <paramref name="line"/> for every line of an array along <paramref name="axis"/>, the other axes walked in
        ///     C order: the line's first element (in elements from the storage's start, the view's offset included), its
        ///     element stride and its length.
        /// </summary>
        /// <param name="shape">The array's shape.</param>
        /// <param name="axis">The normalized line axis.</param>
        /// <param name="line">The callback: (first element, element stride, length).</param>
        private static void ForEachLine(Shape shape, int axis, Action<long, long, long> line)
        {
            var dims = shape.dimensions;
            var strides = shape.strides;
            int nd = dims.Length;
            long n = dims[axis];
            long outer = 1;
            for (int d = 0; d < nd; d++)
                if (d != axis)
                    outer *= dims[d];
            if (outer == 0 || n == 0)
                return;
            var index = new long[nd];
            long offset = shape.offset;
            for (long k = 0; k < outer; k++)
            {
                line(offset, strides[axis], n);
                // The odometer over every axis but the line axis, last axis fastest.
                for (int d = nd - 1; d >= 0; d--)
                {
                    if (d == axis)
                        continue;
                    index[d]++;
                    offset += strides[d];
                    if (index[d] < dims[d])
                        break;
                    offset -= index[d] * strides[d];
                    index[d] = 0;
                }
            }
        }

        /// <summary>Sorts every line of the ids along <paramref name="axis"/> in place (NumPy moves the pointers).</summary>
        /// <param name="axis">The normalized axis.</param>
        /// <param name="stable">True for the stable kind.</param>
        private unsafe void SortInPlace(int axis, bool stable)
        {
            if (size == 0 || Ids.Shape.dimensions[axis] <= 1)
                return;
            using var ranks = Ranks();
            long* ids = (long*)Ids.Address;
            long* r = (long*)ranks.Address;
            var rankShape = ranks.Shape;
            var rankLines = new List<long>();
            ForEachLine(rankShape, axis, (start, stride, n) => rankLines.Add(start));
            int lineNo = 0;
            ForEachLine(Ids.Shape, axis, (start, stride, n) =>
            {
                long rankStart = rankLines[lineNo++];
                long rankStride = rankShape.strides[axis];
                var lineRanks = new long[n];
                var lineIds = new long[n];
                for (long j = 0; j < n; j++)
                {
                    lineRanks[j] = r[rankStart + j * rankStride];
                    lineIds[j] = ids[start + j * stride];
                }
                var perm = SortPermutation(lineRanks, stable);
                for (long j = 0; j < n; j++)
                    ids[start + j * stride] = lineIds[perm[j]];
            });
        }

        /// <summary>The argsort of every line (C-ordered int64 result of the array's shape), or of the flattened array.</summary>
        /// <param name="axis">The axis, or null for the flattened array.</param>
        /// <param name="stable">True for the stable kind.</param>
        /// <returns>The indices.</returns>
        /// <exception cref="AxisError">An axis outside the array.</exception>
        private unsafe NDArray ArgSortCore(int? axis, bool stable)
        {
            using var ranks = Ranks();
            NDArray source = ranks;
            NDArray flatRanks = null;
            int ax;
            if (axis is null)
            {
                flatRanks = ranks.ravel();
                source = flatRanks;
                ax = 0;
            }
            else if (ndim == 0)
            {
                // PyArray_ArgSort's PyArray_CheckAxis ravels a 0-d array to one element first: axis 0 / -1 of a 1-D array
                // (any other is the AxisError of dimension 1), and the result is that 1-D array's argsort, [0].
                DefaultEngine.check_and_adjust_axis(1, axis.Value);
                return new NDArray(NPTypeCode.Int64, Shape.Vector(1), true);
            }
            else
            {
                ax = DefaultEngine.check_and_adjust_axis(ndim, axis.Value);
            }
            using (flatRanks)
            {
                var result = new NDArray(NPTypeCode.Int64, Shape.Adopt((long[])source.Shape.dimensions.Clone()), false);
                long* r = (long*)source.Address;
                long* o = (long*)result.Address;
                var outLines = new List<long>();
                ForEachLine(result.Shape, ax, (start, stride, n) => outLines.Add(start));
                int lineNo = 0;
                long outStride = result.Shape.strides.Length > 0 ? result.Shape.strides[ax] : 0;
                ForEachLine(source.Shape, ax, (start, stride, n) =>
                {
                    long outStart = outLines[lineNo++];
                    var lineRanks = new long[n];
                    for (long j = 0; j < n; j++)
                        lineRanks[j] = r[start + j * stride];
                    var perm = SortPermutation(lineRanks, stable);
                    for (long j = 0; j < n; j++)
                        o[outStart + j * outStride] = perm[j];
                });
                return result;
            }
        }

        /// <summary>The partition's checks in NumPy's order, then the default sort of every line.</summary>
        /// <param name="kth">The kth values.</param>
        /// <param name="axis">The axis.</param>
        /// <param name="kind">Must be <c>'introselect'</c>.</param>
        /// <param name="order">Must be null.</param>
        /// <exception cref="ValueError">The kind, an order, a read-only array, a kth outside the axis.</exception>
        /// <exception cref="AxisError">An axis outside the array.</exception>
        private void PartitionCore(long[] kth, int axis, string kind, string order)
        {
            ThrowIfDisposed();
            CheckSelectKind(kind, order);
            int ax = DefaultEngine.check_and_adjust_axis(ndim, axis);
            if (!Ids.Shape.IsWriteable)
                throw new ValueError("partition array is read-only");
            PrepKth(kth, Ids.Shape.dimensions[ax], size);
            SortInPlace(ax, stable: false);
        }

        /// <summary>The argpartition's checks in NumPy's order, then the default argsort.</summary>
        /// <param name="kth">The kth values.</param>
        /// <param name="axis">The axis, or null for the flattened array.</param>
        /// <param name="kind">Must be <c>'introselect'</c>.</param>
        /// <param name="order">Must be null.</param>
        /// <returns>The indices.</returns>
        /// <exception cref="ValueError">The kind, an order, a kth outside the axis.</exception>
        /// <exception cref="AxisError">An axis outside the array.</exception>
        private NDArray ArgPartitionCore(long[] kth, int? axis, string kind, string order)
        {
            ThrowIfDisposed();
            CheckSelectKind(kind, order);
            if (ndim == 0)
            {
                // PyArray_ArgPartition's PyArray_CheckAxis ravels a 0-d array to one element first.
                if (axis is { } a0 && a0 != 0 && a0 != -1)
                    DefaultEngine.check_and_adjust_axis(1, a0);
                PrepKth(kth, 1, 1);
                var one = new NDArray(NPTypeCode.Int64, Shape.Vector(1), true);
                return one;
            }
            long n = axis is null ? size : Ids.Shape.dimensions[DefaultEngine.check_and_adjust_axis(ndim, axis.Value)];
            PrepKth(kth, n, size);
            return ArgSortCore(axis, stable: false);
        }

        /// <summary>The selection kind and order checks NumPy's argument parsing makes first.</summary>
        /// <param name="kind">Must be <c>'introselect'</c> (null is the default).</param>
        /// <param name="order">Must be null.</param>
        /// <exception cref="ValueError"><c>select kind must be 'introselect' (got 'x')</c>, or an order.</exception>
        private static void CheckSelectKind(string kind, string order)
        {
            if (kind is not null && kind != "introselect")
                throw new ValueError($"select kind must be 'introselect' (got '{kind}')");
            if (order is not null)
                throw new ValueError(SortKindArguments.OrderWithoutFieldsMessage);
        }

        /// <summary>NumPy's array-kth conversion's first stage: more than one dimension is refused before the axis.</summary>
        /// <param name="kth">The kth array.</param>
        /// <exception cref="ArgumentNullException"><paramref name="kth"/> is null.</exception>
        /// <exception cref="ValueError"><c>object too deep for desired array</c>.</exception>
        private static void CheckKthDepth(NDArray kth)
        {
            if (kth is null)
                throw new ArgumentNullException(nameof(kth));
            if (kth.ndim > 1)
                throw new ValueError("object too deep for desired array");
        }

        /// <summary>The values of an array kth: bool refused, a non-integer dtype a TypeError, integers cast to int64.</summary>
        /// <param name="kth">The kth array.</param>
        /// <returns>The values.</returns>
        /// <exception cref="ValueError"><c>Booleans unacceptable as partition index</c>.</exception>
        /// <exception cref="TypeError"><c>Partition index must be integer</c>.</exception>
        private static long[] KthValues(NDArray kth)
        {
            if (kth.typecode == NPTypeCode.Boolean)
                throw new ValueError("Booleans unacceptable as partition index");
            if (kth.dtype.kind is not ('i' or 'u') && kth.typecode != NPTypeCode.Char)
                throw new TypeError("Partition index must be integer");
            // The int64 conversion and its flat view are both this method's: each is disposed (a ravel view of the converted
            // array would otherwise leave the conversion to the finalizer).
            using var wide = kth.astype(NPTypeCode.Int64);
            using var flat = wide.ravel();
            var values = new long[flat.size];
            for (long i = 0; i < values.Length; i++)
                values[i] = flat.GetInt64(i);
            return values;
        }

        /// <summary><c>partition_prep_kth_array</c>: each kth wrapped once and checked against the axis length (unless the
        ///     array is empty).</summary>
        /// <param name="kth">The kth values.</param>
        /// <param name="axisLength">The partitioned length.</param>
        /// <param name="arraySize">The array's size (an empty array skips the bounds check).</param>
        /// <exception cref="ArgumentNullException"><paramref name="kth"/> is null.</exception>
        /// <exception cref="ValueError"><c>kth(=5) out of bounds (4)</c>, the wrapped value.</exception>
        private static void PrepKth(long[] kth, long axisLength, long arraySize)
        {
            if (kth is null)
                throw new ArgumentNullException(nameof(kth));
            foreach (long raw in kth)
            {
                long k = raw < 0 ? raw + axisLength : raw;
                if (arraySize != 0 && (k < 0 || k >= axisLength))
                    throw new ValueError($"kth(={k}) out of bounds ({axisLength})");
            }
        }

        /// <summary>NumPy's <c>searchside_parser</c>.</summary>
        /// <param name="side">The side name.</param>
        /// <returns>True for <c>"right"</c>.</returns>
        /// <exception cref="ValueError">Not exactly <c>"left"</c> or <c>"right"</c>.</exception>
        private static bool SearchSide(string side)
        {
            if (side == "left")
                return false;
            if (side == "right")
                return true;
            if (!string.IsNullOrEmpty(side) && side[0] is 'l' or 'L' or 'r' or 'R')
                throw new ValueError("search side must be one of 'left' or 'right'");
            throw new ValueError($"search side must be 'left' or 'right' (got '{side}')");
        }

        /// <summary>The haystack's rank check (<c>PyArray_SearchSorted</c> converts it to 1-D).</summary>
        /// <param name="sorter">The sorter (checked by <see cref="SorterIndices"/>).</param>
        /// <exception cref="ValueError">More than one dimension (<c>object too deep for desired array</c>), or none (<c>object
        ///     of too small depth for desired array</c>).</exception>
        private void CheckHaystack(NDArray sorter)
        {
            // PyArray_CheckFromAny(op1, dtype, 1, 1, ...): exactly one dimension.
            if (ndim > 1)
                throw new ValueError("object too deep for desired array");
            if (ndim == 0)
                throw new ValueError("object of too small depth for desired array");
        }

        /// <summary>The sorter's checks (1-D integers as long as the haystack) and its values.</summary>
        /// <param name="sorter">The sorter, or null.</param>
        /// <param name="haystackSize">The haystack's size.</param>
        /// <returns>The sorter's values, or null without a sorter.</returns>
        /// <exception cref="TypeError">Not 1-D (<c>could not parse sorter argument</c>), or not integers.</exception>
        /// <exception cref="ValueError">uint64, or another size (<c>sorter.size must equal a.size</c>).</exception>
        private static long[] SorterIndices(NDArray sorter, long haystackSize)
        {
            if (sorter is null)
                return null;
            if (sorter.ndim != 1)
                throw new TypeError("could not parse sorter argument");
            if (sorter.dtype.kind is not ('i' or 'u') && sorter.typecode != NPTypeCode.Char)
                throw new TypeError("sorter must only contain integers");
            if (sorter.typecode == NPTypeCode.UInt64)
                throw new ValueError("could not parse sorter argument");
            if (sorter.size != haystackSize)
                throw new ValueError("sorter.size must equal a.size");
            using var flat = sorter.astype(NPTypeCode.Int64);
            var values = new long[flat.size];
            for (long i = 0; i < values.Length; i++)
                values[i] = flat.GetInt64(i);
            return values;
        }
    }
}
