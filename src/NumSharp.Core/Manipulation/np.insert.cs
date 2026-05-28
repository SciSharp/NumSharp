using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        // ============================== np.insert ==============================
        // Insert values along the given axis before the given indices.
        //
        // NumPy 2.4.2 reference: numpy/lib/_function_base_impl.py::insert
        // Two execution branches mirror NumPy's:
        //
        //   * SINGLE-INDEX path — obj is a scalar int or a 1-element int array.
        //     The values shape is normalised (cast to arr.dtype, ndmin=arr.ndim)
        //     and the result is produced by 3-way concatenate:
        //         arr[..., :idx, ...] | values | arr[..., idx:, ...]
        //     A subtle NumPy quirk drives the shape coercion: scalar obj
        //     ("a[:,0,:] = ..." semantics) versus 1-elem-array obj
        //     ("a[:,[0],:] = ..." semantics) differ in how values broadcast,
        //     so when obj is scalar we np.moveaxis(values, 0, axis) to fold
        //     the first axis of values onto the insertion axis.
        //
        //   * MULTI-INDEX path — obj is a multi-element int array, a bool mask
        //     (converted via flatnonzero), or a non-trivial slice. Algorithm:
        //         1. Sort indices (stable). Reorder values along axis by the
        //            sort order so insertions land in monotonic axis position.
        //         2. Split arr along axis at the sorted indices into len(idx)+1
        //            chunks. Build a 2*N+1 interleaved sequence:
        //                chunk_0 | val_0 | chunk_1 | val_1 | ... | chunk_N
        //         3. Concatenate all chunks along axis.
        //     Equivalent to NumPy's "build a keep-mask and scatter-place"
        //     algorithm; the concatenate form lets the IL contig-cast kernel
        //     fire instead of fancy-index scatter for each value slot.
        //
        // axis=None ravels arr first (axis becomes 0 of the flattened array).

        /// <summary>
        ///     Insert <paramref name="values"/> along <paramref name="axis"/>
        ///     before the position <paramref name="obj"/>. Scalar-obj path.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.insert.html</remarks>
        public static NDArray insert(NDArray arr, int obj, NDArray values, int? axis = null)
            => insert(arr, (long)obj, values, axis);

        /// <summary>
        ///     Long-index overload of <see cref="insert(NDArray, int, NDArray, int?)"/>.
        /// </summary>
        public static NDArray insert(NDArray arr, long obj, NDArray values, int? axis = null)
        {
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            if (values is null) throw new ArgumentNullException(nameof(values));

            var ctx = PrepareAxisContext(arr, axis);
            return InsertSingleIndex(ctx.work, obj, values, ctx.axis, scalarObj: true);
        }

        /// <summary>
        ///     Scalar-obj overload accepting a scalar value (NumPy: <c>np.insert(a, 1, 99)</c>).
        ///     Broadcasts the value to the required shape using <c>arr.dtype</c>.
        /// </summary>
        public static NDArray insert(NDArray arr, long obj, object value, int? axis = null)
        {
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            var asArr = ScalarValueAsArray(value, arr.dtype);
            try { return insert(arr, obj, asArr, axis); }
            finally { asArr.Dispose(); }
        }

        /// <summary>
        ///     int-overload accepting a scalar value.
        /// </summary>
        public static NDArray insert(NDArray arr, int obj, object value, int? axis = null)
            => insert(arr, (long)obj, value, axis);

        /// <summary>
        ///     Slice-obj overload. <paramref name="obj"/> is expanded via
        ///     Python <c>slice.indices(N)</c> into an indices array, then the
        ///     multi-index branch runs.
        /// </summary>
        public static NDArray insert(NDArray arr, Slice obj, NDArray values, int? axis = null)
        {
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            if (obj is null) throw new ArgumentNullException(nameof(obj));
            if (values is null) throw new ArgumentNullException(nameof(values));

            var ctx = PrepareAxisContext(arr, axis);
            long N = ctx.work.shape[ctx.axis];
            var indices = InsertExpandSliceObj(obj, N);
            // Slice obj is never treated as scalar for the broadcast quirk
            // (matches NumPy: ``isinstance(obj, slice)`` short-circuits the
            // scalar check).
            return InsertMultiIndex(ctx.work, indices, values, ctx.axis);
        }

        /// <summary>
        ///     Slice-obj overload accepting a scalar value.
        /// </summary>
        public static NDArray insert(NDArray arr, Slice obj, object value, int? axis = null)
        {
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            var asArr = ScalarValueAsArray(value, arr.dtype);
            try { return insert(arr, obj, asArr, axis); }
            finally { asArr.Dispose(); }
        }

        /// <summary>
        ///     int[]-obj overload. Always routes through the multi-index branch
        ///     (NumPy parity: <c>np.insert(arr, [1], v) != np.insert(arr, 1, v)</c>
        ///     when v has multiple axes, even though both have one insertion point).
        /// </summary>
        public static NDArray insert(NDArray arr, int[] obj, NDArray values, int? axis = null)
        {
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            if (obj is null) throw new ArgumentNullException(nameof(obj));
            if (values is null) throw new ArgumentNullException(nameof(values));

            var longs = new long[obj.Length];
            for (int i = 0; i < obj.Length; i++) longs[i] = obj[i];

            var ctx = PrepareAxisContext(arr, axis);
            return InsertMultiIndex(ctx.work, longs, values, ctx.axis);
        }

        /// <summary>
        ///     int[]-obj overload accepting a scalar value.
        /// </summary>
        public static NDArray insert(NDArray arr, int[] obj, object value, int? axis = null)
        {
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            var asArr = ScalarValueAsArray(value, arr.dtype);
            try { return insert(arr, obj, asArr, axis); }
            finally { asArr.Dispose(); }
        }

        /// <summary>
        ///     long[]-obj overload.
        /// </summary>
        public static NDArray insert(NDArray arr, long[] obj, NDArray values, int? axis = null)
        {
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            if (obj is null) throw new ArgumentNullException(nameof(obj));
            if (values is null) throw new ArgumentNullException(nameof(values));

            var ctx = PrepareAxisContext(arr, axis);
            return InsertMultiIndex(ctx.work, (long[])obj.Clone(), values, ctx.axis);
        }

        /// <summary>
        ///     long[]-obj overload accepting a scalar value.
        /// </summary>
        public static NDArray insert(NDArray arr, long[] obj, object value, int? axis = null)
        {
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            var asArr = ScalarValueAsArray(value, arr.dtype);
            try { return insert(arr, obj, asArr, axis); }
            finally { asArr.Dispose(); }
        }

        /// <summary>
        ///     NDArray-obj overload accepting a scalar value.
        /// </summary>
        public static NDArray insert(NDArray arr, NDArray obj, object value, int? axis = null)
        {
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            var asArr = ScalarValueAsArray(value, arr.dtype);
            try { return insert(arr, obj, asArr, axis); }
            finally { asArr.Dispose(); }
        }

        /// <summary>
        ///     NDArray-typed obj. Dispatches:
        ///     <list type="bullet">
        ///       <item>0-D / 1-element integer ⇒ scalar single-index path.</item>
        ///       <item>1-D bool ⇒ <see cref="np.flatnonzero"/> ⇒ multi-index path.</item>
        ///       <item>1-D integer ⇒ multi-index path.</item>
        ///       <item>&gt; 1-D ⇒ <see cref="ArgumentException"/> matching NumPy's
        ///             "index array argument obj to insert must be one dimensional or scalar".</item>
        ///     </list>
        /// </summary>
        public static NDArray insert(NDArray arr, NDArray obj, NDArray values, int? axis = null)
        {
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            if (obj is null) throw new ArgumentNullException(nameof(obj));
            if (values is null) throw new ArgumentNullException(nameof(values));

            var ctx = PrepareAxisContext(arr, axis);

            if (obj.GetTypeCode == NPTypeCode.Boolean)
            {
                if (obj.ndim != 1)
                    throw new ArgumentException(
                        "boolean array argument obj to insert must be one dimensional",
                        nameof(obj));
                using var nzIdx = np.flatnonzero(obj);
                long[] indices = ToInt64Vector(nzIdx, "insert");
                return InsertMultiIndex(ctx.work, indices, values, ctx.axis);
            }

            if (obj.ndim > 1)
                throw new ArgumentException(
                    "index array argument obj to insert must be one dimensional or scalar",
                    nameof(obj));

            // 0-D integer ⇒ scalar path (true scalar semantics).
            if (obj.ndim == 0)
            {
                long idx = ToInt64Scalar(obj, "insert");
                return InsertSingleIndex(ctx.work, idx, values, ctx.axis, scalarObj: true);
            }

            // 1-D with size==1 still uses the single-index broadcast path but
            // with scalarObj=false (mirrors NumPy: indices.ndim == 1 prevents
            // the moveaxis quirk).
            long[] objIndices = ToInt64Vector(obj, "insert");
            if (objIndices.Length == 1)
                return InsertSingleIndex(ctx.work, objIndices[0], values, ctx.axis, scalarObj: false);

            return InsertMultiIndex(ctx.work, objIndices, values, ctx.axis);
        }

        // ---------------------------- impl ----------------------------

        /// <summary>
        ///     Single-index path: cast values to arr.dtype, normalise shape
        ///     (ndmin=arr.ndim, optional moveaxis for scalar-obj quirk),
        ///     concatenate pre + values + post along axis.
        /// </summary>
        private static NDArray InsertSingleIndex(
            NDArray arr, long index, NDArray values, int axis, bool scalarObj)
        {
            long N = arr.shape[axis];
            if (index < -N || index > N) // Insert allows index == N (append-at-end).
                throw new IndexOutOfRangeException(
                    $"index {index} is out of bounds for axis {axis} with size {N}");
            if (index < 0) index += N;

            // Cast values to arr.dtype (NumPy: values = array(values, dtype=arr.dtype))
            // and ensure ndmin = arr.ndim. We need to potentially moveaxis the first
            // axis to the insertion axis (scalarObj quirk).
            using var coerced = CoerceValuesForSingleIndex(arr, values, axis, scalarObj);

            long numNew = coerced.shape[axis];
            if (numNew == 0)
                return DuplicateArray(arr);

            var pre = SliceAlongAxis(arr, axis, 0, index);
            var post = SliceAlongAxis(arr, axis, index, N);
            try
            {
                // Build the chunk list, dropping any empty-along-axis ones so
                // concatenate doesn't have to filter them itself.
                if (pre.shape[axis] == 0)
                {
                    if (post.shape[axis] == 0)
                        return np.concatenate(new[] { coerced }, axis);
                    return np.concatenate(new[] { coerced, post }, axis);
                }
                if (post.shape[axis] == 0)
                    return np.concatenate(new[] { pre, coerced }, axis);
                return np.concatenate(new[] { pre, coerced, post }, axis);
            }
            finally { pre.Dispose(); post.Dispose(); }
        }

        /// <summary>
        ///     Multi-index path. Normalise indices (negative → positive,
        ///     bounds-check), stable-sort and reorder values, then build the
        ///     interleaved chunk list for a single concatenate call.
        /// </summary>
        private static NDArray InsertMultiIndex(
            NDArray arr, long[] indices, NDArray values, int axis)
        {
            long N = arr.shape[axis];

            // Empty indices ⇒ NumPy returns arr.copy() (NumPy: indices.size == 0 and
            // not isinstance(obj, np.ndarray); we always normalise empty regardless,
            // which matches NumPy when obj wasn't an ndarray with shape (0,)).
            if (indices.Length == 0)
                return DuplicateArray(arr);

            // Normalize and validate indices.
            for (int i = 0; i < indices.Length; i++)
            {
                long idx = indices[i];
                if (idx < -N || idx > N)
                    throw new IndexOutOfRangeException(
                        $"index {indices[i]} is out of bounds for axis {axis} with size {N}");
                if (idx < 0) idx += N;
                indices[i] = idx;
            }

            int numNew = indices.Length;

            // Coerce values to arr.dtype with axis-dim broadcast to numNew. The
            // multi-index path never applies the moveaxis quirk.
            using var coerced = CoerceValuesForMultiIndex(arr, values, axis, numNew);

            // Stable sort of indices, recording the permutation. Used to reorder
            // both the indices list (for chunk splitting) and the values along
            // axis (so the reordered values match the sorted insertion positions).
            int[] order = StableArgSort(indices);
            long[] sortedIndices = new long[numNew];
            for (int i = 0; i < numNew; i++) sortedIndices[i] = indices[order[i]];

            // Reorder values along axis to align with the sorted insertion order.
            // values[..., order, ...] — done via np.take if the order is non-trivial.
            // Skip the take if the order is already identity (early exit).
            NDArray reorderedValues = null;
            bool ownReordered = false;
            bool needReorder = false;
            for (int i = 0; i < numNew; i++)
            {
                if (order[i] != i) { needReorder = true; break; }
            }

            if (needReorder)
            {
                var orderArr = np.array(LongArrayFromInt(order));
                reorderedValues = np.take(coerced, orderArr, axis: axis);
                ownReordered = true;
                orderArr.Dispose();
            }
            else
            {
                reorderedValues = coerced;
            }

            try
            {
                // Build interleaved chunk list: [arr[:s0], val0, arr[s0:s1], val1, ..., arr[s_{N-1}:end]]
                // For each sorted index s_k, take values[..., k:k+1, ...] as the value chunk.
                int totalChunks = numNew * 2 + 1;
                var chunks = new NDArray[totalChunks];
                var disposable = new NDArray[totalChunks];
                int slot = 0;
                long prevSplit = 0;

                try
                {
                    for (int k = 0; k < numNew; k++)
                    {
                        long s = sortedIndices[k];
                        // arr chunk [prevSplit, s)
                        var arrChunk = SliceAlongAxis(arr, axis, prevSplit, s);
                        chunks[slot] = arrChunk;
                        disposable[slot] = arrChunk;
                        slot++;

                        // values chunk along axis: [k, k+1)
                        var valChunk = SliceAlongAxis(reorderedValues, axis, k, k + 1);
                        chunks[slot] = valChunk;
                        disposable[slot] = valChunk;
                        slot++;

                        prevSplit = s;
                    }

                    // Tail: arr[prevSplit:N]
                    var arrTail = SliceAlongAxis(arr, axis, prevSplit, N);
                    chunks[slot] = arrTail;
                    disposable[slot] = arrTail;
                    slot++;

                    return np.concatenate(chunks, axis);
                }
                finally
                {
                    for (int i = 0; i < disposable.Length; i++)
                        disposable[i]?.Dispose();
                }
            }
            finally
            {
                if (ownReordered) reorderedValues.Dispose();
            }
        }

        // ---------------------------- helpers ----------------------------

        /// <summary>
        ///     Wraps a scalar value as an NDArray of <paramref name="dtype"/>.
        ///     Used by the <c>insert(arr, obj, value, axis)</c> overloads that
        ///     accept an unboxed scalar so callers don't have to construct
        ///     a 0-D ndarray themselves.
        /// </summary>
        private static NDArray ScalarValueAsArray(object value, Type dtype)
        {
            if (value is NDArray nd) return nd; // (caller should disposes the wrapper, but the as-is path is safe)
            var asArr = np.asanyarray(value);
            if (!Equals(asArr.dtype, dtype))
            {
                var casted = asArr.astype(dtype, copy: true);
                asArr.Dispose();
                return casted;
            }
            return asArr;
        }

        /// <summary>
        ///     Coerces <paramref name="values"/> to <paramref name="arr"/>'s dtype
        ///     and aligns its shape for the single-index path. Matches NumPy's
        ///     <c>values = array(values, copy=None, ndmin=arr.ndim, dtype=arr.dtype);
        ///     if scalar_obj: values = np.moveaxis(values, 0, axis); numnew = values.shape[axis]</c>.
        ///     The final shape is broadcast to <c>arr.shape</c> with the axis dim
        ///     replaced by <c>numnew</c>, so the concatenate path can splice it
        ///     in without further shape gymnastics.
        /// </summary>
        private static NDArray CoerceValuesForSingleIndex(
            NDArray arr, NDArray values, int axis, bool scalarObj)
        {
            // Cast dtype if needed.
            NDArray v = values;
            bool owned = false;
            if (v.GetTypeCode != arr.GetTypeCode)
            {
                v = values.astype(arr.GetTypeCode, copy: true);
                owned = true;
            }

            // ndmin = arr.ndim: prepend 1s.
            int targetNDim = Math.Max(arr.ndim, 1);
            if (v.ndim < targetNDim)
            {
                var newDims = new long[targetNDim];
                int prefix = targetNDim - v.ndim;
                for (int i = 0; i < prefix; i++) newDims[i] = 1;
                for (int i = 0; i < v.ndim; i++) newDims[prefix + i] = v.shape[i];
                var reshaped = v.reshape(new Shape(newDims));
                if (owned) v.Dispose();
                v = reshaped;
                owned = true;
            }

            // Scalar-obj quirk: moveaxis(values, 0, axis) — moves the first axis
            // of values to the insertion axis position. This re-shapes values so
            // that broadcasting across non-axis dims behaves like
            // ``a[:, 0, :] = ...`` rather than ``a[:, [0], :] = ...``.
            if (scalarObj && v.ndim > 1 && axis != 0)
            {
                var moved = np.moveaxis(v, 0, axis);
                if (owned) v.Dispose();
                v = moved;
                owned = true;
            }

            // Broadcast to arr.shape with axis dim = v.shape[axis].
            long numNew = arr.ndim == 0 ? v.size : v.shape[axis];
            var broadcasted = BroadcastValuesToInsertSlot(arr, v, axis, numNew);
            if (owned && !ReferenceEquals(broadcasted, v)) v.Dispose();
            else if (!owned && ReferenceEquals(broadcasted, v))
                broadcasted = new NDArray(v.Storage) { TensorEngine = v.TensorEngine };
            return broadcasted;
        }

        /// <summary>
        ///     Broadcasts <paramref name="values"/> to the shape
        ///     <c>arr.shape[..axis] + (numNew,) + arr.shape[axis+1..]</c>. The
        ///     result is always C-contiguous so the chunk-splitting + concatenate
        ///     pipeline can slice it without strided-view surprises.
        /// </summary>
        private static NDArray BroadcastValuesToInsertSlot(
            NDArray arr, NDArray values, int axis, long numNew)
        {
            // Already matches? Return as-is.
            if (values.ndim == arr.ndim)
            {
                bool same = true;
                for (int i = 0; i < arr.ndim; i++)
                {
                    long expected = i == axis ? numNew : arr.shape[i];
                    if (values.shape[i] != expected) { same = false; break; }
                }
                if (same) return values;
            }

            // Build target shape and broadcast.
            var targetDims = new long[arr.ndim];
            for (int i = 0; i < arr.ndim; i++) targetDims[i] = arr.shape[i];
            targetDims[axis] = numNew;

            var bcast = np.broadcast_to(values, new Shape(targetDims));
            // Materialise to a contig owning copy — broadcast_to returns a
            // stride-0 read-only view that confuses np.take and SliceAlongAxis.
            var contig = np.ascontiguousarray(bcast);
            if (!ReferenceEquals(contig, bcast)) bcast.Dispose();
            return contig;
        }

        /// <summary>
        ///     Coerces <paramref name="values"/> to <paramref name="arr"/>'s dtype
        ///     and broadcasts its shape to match
        ///     <c>arr.shape[..axis] + (numNew,) + arr.shape[axis+1..]</c>.
        ///     Used by the multi-index path before sort/reorder.
        /// </summary>
        private static NDArray CoerceValuesForMultiIndex(
            NDArray arr, NDArray values, int axis, int numNew)
        {
            // Cast dtype.
            NDArray v = values;
            bool owned = false;
            if (v.GetTypeCode != arr.GetTypeCode)
            {
                v = values.astype(arr.GetTypeCode, copy: true);
                owned = true;
            }

            // Build target shape: arr.shape with axis dim = numNew.
            var targetDims = new long[arr.ndim];
            for (int i = 0; i < arr.ndim; i++) targetDims[i] = arr.shape[i];
            targetDims[axis] = numNew;
            var targetShape = new Shape(targetDims);

            // Shape already matches? Just hand it back (still wrapped).
            if (v.ndim == arr.ndim)
            {
                bool same = true;
                for (int i = 0; i < arr.ndim; i++)
                {
                    if (v.shape[i] != targetDims[i]) { same = false; break; }
                }
                if (same)
                {
                    if (!owned)
                        v = new NDArray(v.Storage) { TensorEngine = v.TensorEngine };
                    return v;
                }
            }

            // Broadcast. For 1-D values whose length equals numNew on an N-D
            // arr, NumPy broadcasts across non-axis dims (e.g. shape (2,) into
            // (3, 6) at axis=1 ⇒ broadcasts to (3, 2) at axis=1, but our target
            // is (3, 2) so we reshape (2,) → (1, 2) → (3, 2) via broadcast_to).
            //
            // Strategy: prepend ones to ndim, then broadcast_to the target.
            int targetNDim = arr.ndim;
            if (v.ndim < targetNDim)
            {
                var newDims = new long[targetNDim];
                int prefix = targetNDim - v.ndim;
                for (int i = 0; i < prefix; i++) newDims[i] = 1;
                for (int i = 0; i < v.ndim; i++) newDims[prefix + i] = v.shape[i];
                var reshaped = v.reshape(new Shape(newDims));
                if (owned) v.Dispose();
                v = reshaped;
                owned = true;
            }

            // Broadcast to target. broadcast_to handles size-1 stretching.
            var bcast = np.broadcast_to(v, targetShape);
            // broadcast_to returns a stride-0 read-only view; materialize a
            // contig owning copy so np.take and our SliceAlongAxis loop hit
            // the fast paths and we don't run into the "broadcast view + slice
            // by stepped-axis" quirks.
            var contig = np.ascontiguousarray(bcast);
            if (!ReferenceEquals(contig, bcast))
                bcast.Dispose();
            else
                contig = bcast;
            if (owned) v.Dispose();
            return contig;
        }

        /// <summary>
        ///     Expands a Python-style slice into the concrete integer indices
        ///     it would generate when used as the obj of np.insert. Mirrors
        ///     NumPy's <c>arange(*obj.indices(N), dtype=intp)</c>.
        /// </summary>
        internal static long[] InsertExpandSliceObj(Slice obj, long N)
        {
            var (start, stop, step) = PythonSliceIndices(obj, N);
            long count;
            if (step > 0)
                count = stop > start ? (stop - start + step - 1) / step : 0;
            else
                count = stop < start ? (start - stop + (-step) - 1) / (-step) : 0;

            var result = new long[count];
            for (long i = 0; i < count; i++) result[i] = start + i * step;
            return result;
        }

        /// <summary>
        ///     Stable argsort returning a permutation that orders
        ///     <paramref name="keys"/> in ascending order. Implemented as a
        ///     direct mergesort over int[] (numbers of insertions in any real
        ///     use are tiny; avoids the boxing detour through NDArray + argsort).
        /// </summary>
        private static int[] StableArgSort(long[] keys)
        {
            int n = keys.Length;
            var idx = new int[n];
            for (int i = 0; i < n; i++) idx[i] = i;

            if (n <= 1) return idx;

            var tmp = new int[n];
            MergeSortHelper(idx, tmp, keys, 0, n);
            return idx;
        }

        private static void MergeSortHelper(int[] arr, int[] tmp, long[] keys, int lo, int hi)
        {
            if (hi - lo <= 1) return;
            int mid = (lo + hi) >> 1;
            MergeSortHelper(arr, tmp, keys, lo, mid);
            MergeSortHelper(arr, tmp, keys, mid, hi);
            // Merge two sorted runs [lo, mid) and [mid, hi).
            int i = lo, j = mid, k = lo;
            while (i < mid && j < hi)
            {
                // <= for stability — equal-key items keep their original order.
                if (keys[arr[i]] <= keys[arr[j]])
                    tmp[k++] = arr[i++];
                else
                    tmp[k++] = arr[j++];
            }
            while (i < mid) tmp[k++] = arr[i++];
            while (j < hi) tmp[k++] = arr[j++];
            for (int q = lo; q < hi; q++) arr[q] = tmp[q];
        }

        private static long[] LongArrayFromInt(int[] src)
        {
            var result = new long[src.Length];
            for (int i = 0; i < src.Length; i++) result[i] = src[i];
            return result;
        }
    }
}
