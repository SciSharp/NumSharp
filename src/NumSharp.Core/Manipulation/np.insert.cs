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
        [NDScoped] // reclaims PrepareAxisContext's ravel work array and the pre-order intermediate WithSourceOrder supersedes (np.delete's exact pattern)
        public static NDArray insert(NDArray arr, long obj, NDArray values, int? axis = null)
        {
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            if (values is null) throw new ArgumentNullException(nameof(values));

            var ctx = PrepareAxisContext(arr, axis);
            return WithSourceOrder(arr, InsertSingleIndex(ctx.work, obj, values, ctx.axis, scalarObj: true));
        }

        /// <summary>
        ///     Scalar-obj overload accepting a scalar value (NumPy: <c>np.insert(a, 1, 99)</c>).
        ///     Broadcasts the value to the required shape using <c>arr.dtype</c>.
        /// </summary>
        /// <param name="arr">Input array.</param>
        /// <param name="obj">The index before which the value is inserted; negative values count from the end.</param>
        /// <param name="value">The value(s) to insert: a scalar, a C# array or collection, or an
        ///     <see cref="NDArray"/> whose static type is <see cref="object"/>. Cast to <paramref name="arr"/>'s
        ///     dtype.</param>
        /// <param name="axis">The axis to insert along; <c>null</c> inserts into the flattened array.</param>
        /// <returns>A fresh array with the value(s) inserted (F-ordered when <paramref name="arr"/> is).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="arr"/>, <paramref name="obj"/> or <paramref name="value"/> is null.</exception>
        /// <exception cref="IndexOutOfRangeException">An insertion index is outside <c>[-N, N]</c>.</exception>
        /// <remarks>
        ///     Scoped by <c>[NDScoped]</c>: the conversion of <paramref name="value"/> is built in THIS frame,
        ///     before the array overload's scope opens, so this overload releases it with a scope of its own. An
        ///     <see cref="NDArray"/> passed as <paramref name="value"/> comes back unchanged and was built before
        ///     the scope opened, so it is never disposed. (The hand-written version disposed the conversion
        ///     unconditionally, which released the caller's own array.)
        /// </remarks>
        [NDScoped]
        public static NDArray insert(NDArray arr, long obj, object value, int? axis = null)
        {
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            var asArr = ScalarValueAsArray(value, arr.dtype);
            return insert(arr, obj, asArr, axis);
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
        [NDScoped] // same transient set as the long-obj overload
        public static NDArray insert(NDArray arr, Slice obj, NDArray values, int? axis = null)
        {
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            if (obj is null) throw new ArgumentNullException(nameof(obj));
            if (values is null) throw new ArgumentNullException(nameof(values));

            var ctx = PrepareAxisContext(arr, axis);
            long N = ctx.work.Shape.dimensions[ctx.axis];
            var indices = InsertExpandSliceObj(obj, N);
            // Slice obj is never treated as scalar for the broadcast quirk
            // (matches NumPy: ``isinstance(obj, slice)`` short-circuits the
            // scalar check).
            return WithSourceOrder(arr, InsertMultiIndex(ctx.work, indices, values, ctx.axis));
        }

        /// <summary>
        ///     Slice-obj overload accepting a scalar value.
        /// </summary>
        /// <param name="arr">Input array.</param>
        /// <param name="obj">The insertion positions, expanded with Python's <c>slice.indices(N)</c>.</param>
        /// <param name="value">The value(s) to insert: a scalar, a C# array or collection, or an
        ///     <see cref="NDArray"/> whose static type is <see cref="object"/>. Cast to <paramref name="arr"/>'s
        ///     dtype.</param>
        /// <param name="axis">The axis to insert along; <c>null</c> inserts into the flattened array.</param>
        /// <returns>A fresh array with the value(s) inserted (F-ordered when <paramref name="arr"/> is).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="arr"/>, <paramref name="obj"/> or <paramref name="value"/> is null.</exception>
        /// <exception cref="IndexOutOfRangeException">An insertion index is outside <c>[-N, N]</c>.</exception>
        /// <remarks>
        ///     Scoped by <c>[NDScoped]</c>: the conversion of <paramref name="value"/> is built in THIS frame,
        ///     before the array overload's scope opens, so this overload releases it with a scope of its own. An
        ///     <see cref="NDArray"/> passed as <paramref name="value"/> comes back unchanged and was built before
        ///     the scope opened, so it is never disposed. (The hand-written version disposed the conversion
        ///     unconditionally, which released the caller's own array.)
        /// </remarks>
        [NDScoped]
        public static NDArray insert(NDArray arr, Slice obj, object value, int? axis = null)
        {
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            var asArr = ScalarValueAsArray(value, arr.dtype);
            return insert(arr, obj, asArr, axis);
        }

        /// <summary>
        ///     int[]-obj overload. Always routes through the multi-index branch
        ///     (NumPy parity: <c>np.insert(arr, [1], v) != np.insert(arr, 1, v)</c>
        ///     when v has multiple axes, even though both have one insertion point).
        /// </summary>
        [NDScoped] // same transient set as the long-obj overload
        public static NDArray insert(NDArray arr, int[] obj, NDArray values, int? axis = null)
        {
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            if (obj is null) throw new ArgumentNullException(nameof(obj));
            if (values is null) throw new ArgumentNullException(nameof(values));

            var longs = new long[obj.Length];
            for (int i = 0; i < obj.Length; i++) longs[i] = obj[i];

            var ctx = PrepareAxisContext(arr, axis);
            return WithSourceOrder(arr, InsertMultiIndex(ctx.work, longs, values, ctx.axis));
        }

        /// <summary>
        ///     int[]-obj overload accepting a scalar value.
        /// </summary>
        /// <param name="arr">Input array.</param>
        /// <param name="obj">The insertion positions (each before the original element at that index).</param>
        /// <param name="value">The value(s) to insert: a scalar, a C# array or collection, or an
        ///     <see cref="NDArray"/> whose static type is <see cref="object"/>. Cast to <paramref name="arr"/>'s
        ///     dtype.</param>
        /// <param name="axis">The axis to insert along; <c>null</c> inserts into the flattened array.</param>
        /// <returns>A fresh array with the value(s) inserted (F-ordered when <paramref name="arr"/> is).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="arr"/>, <paramref name="obj"/> or <paramref name="value"/> is null.</exception>
        /// <exception cref="IndexOutOfRangeException">An insertion index is outside <c>[-N, N]</c>.</exception>
        /// <remarks>
        ///     Scoped by <c>[NDScoped]</c>: the conversion of <paramref name="value"/> is built in THIS frame,
        ///     before the array overload's scope opens, so this overload releases it with a scope of its own. An
        ///     <see cref="NDArray"/> passed as <paramref name="value"/> comes back unchanged and was built before
        ///     the scope opened, so it is never disposed. (The hand-written version disposed the conversion
        ///     unconditionally, which released the caller's own array.)
        /// </remarks>
        [NDScoped]
        public static NDArray insert(NDArray arr, int[] obj, object value, int? axis = null)
        {
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            var asArr = ScalarValueAsArray(value, arr.dtype);
            return insert(arr, obj, asArr, axis);
        }

        /// <summary>
        ///     long[]-obj overload.
        /// </summary>
        [NDScoped] // same transient set as the long-obj overload
        public static NDArray insert(NDArray arr, long[] obj, NDArray values, int? axis = null)
        {
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            if (obj is null) throw new ArgumentNullException(nameof(obj));
            if (values is null) throw new ArgumentNullException(nameof(values));

            var ctx = PrepareAxisContext(arr, axis);
            return WithSourceOrder(arr, InsertMultiIndex(ctx.work, (long[])obj.Clone(), values, ctx.axis));
        }

        /// <summary>
        ///     long[]-obj overload accepting a scalar value.
        /// </summary>
        /// <param name="arr">Input array.</param>
        /// <param name="obj">The insertion positions (each before the original element at that index).</param>
        /// <param name="value">The value(s) to insert: a scalar, a C# array or collection, or an
        ///     <see cref="NDArray"/> whose static type is <see cref="object"/>. Cast to <paramref name="arr"/>'s
        ///     dtype.</param>
        /// <param name="axis">The axis to insert along; <c>null</c> inserts into the flattened array.</param>
        /// <returns>A fresh array with the value(s) inserted (F-ordered when <paramref name="arr"/> is).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="arr"/>, <paramref name="obj"/> or <paramref name="value"/> is null.</exception>
        /// <exception cref="IndexOutOfRangeException">An insertion index is outside <c>[-N, N]</c>.</exception>
        /// <remarks>
        ///     Scoped by <c>[NDScoped]</c>: the conversion of <paramref name="value"/> is built in THIS frame,
        ///     before the array overload's scope opens, so this overload releases it with a scope of its own. An
        ///     <see cref="NDArray"/> passed as <paramref name="value"/> comes back unchanged and was built before
        ///     the scope opened, so it is never disposed. (The hand-written version disposed the conversion
        ///     unconditionally, which released the caller's own array.)
        /// </remarks>
        [NDScoped]
        public static NDArray insert(NDArray arr, long[] obj, object value, int? axis = null)
        {
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            var asArr = ScalarValueAsArray(value, arr.dtype);
            return insert(arr, obj, asArr, axis);
        }

        /// <summary>
        ///     NDArray-obj overload accepting a scalar value.
        /// </summary>
        /// <param name="arr">Input array.</param>
        /// <param name="obj">A 0-d or 1-D integer index array, or a 1-D boolean mask of insertion positions.</param>
        /// <param name="value">The value(s) to insert: a scalar, a C# array or collection, or an
        ///     <see cref="NDArray"/> whose static type is <see cref="object"/>. Cast to <paramref name="arr"/>'s
        ///     dtype.</param>
        /// <param name="axis">The axis to insert along; <c>null</c> inserts into the flattened array.</param>
        /// <returns>A fresh array with the value(s) inserted (F-ordered when <paramref name="arr"/> is).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="arr"/>, <paramref name="obj"/> or <paramref name="value"/> is null.</exception>
        /// <exception cref="ArgumentException"><paramref name="obj"/> is a boolean mask that is not 1-D, or an index
        ///     array of more than one dimension.</exception>
        /// <exception cref="IndexOutOfRangeException">An insertion index is outside <c>[-N, N]</c>.</exception>
        /// <remarks>
        ///     Scoped by <c>[NDScoped]</c>: the conversion of <paramref name="value"/> is built in THIS frame,
        ///     before the array overload's scope opens, so this overload releases it with a scope of its own. An
        ///     <see cref="NDArray"/> passed as <paramref name="value"/> comes back unchanged and was built before
        ///     the scope opened, so it is never disposed. (The hand-written version disposed the conversion
        ///     unconditionally, which released the caller's own array.)
        /// </remarks>
        [NDScoped]
        public static NDArray insert(NDArray arr, NDArray obj, object value, int? axis = null)
        {
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            var asArr = ScalarValueAsArray(value, arr.dtype);
            return insert(arr, obj, asArr, axis);
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
        /// <param name="arr">Input array.</param>
        /// <param name="obj">A 0-d or 1-D integer index array, or a 1-D boolean mask of insertion positions.</param>
        /// <param name="values">The values to insert, cast to <paramref name="arr"/>'s dtype and broadcast to the
        ///     insertion slot.</param>
        /// <param name="axis">The axis to insert along; <c>null</c> inserts into the flattened array.</param>
        /// <returns>A fresh array with the values inserted (F-ordered when <paramref name="arr"/> is).</returns>
        /// <exception cref="ArgumentNullException">An argument is null.</exception>
        /// <exception cref="ArgumentException"><paramref name="obj"/> is a boolean mask that is not 1-D, or an index
        ///     array of more than one dimension.</exception>
        /// <exception cref="IndexOutOfRangeException">An insertion index is outside <c>[-N, N]</c>.</exception>
        /// <remarks>Scoped by <c>[NDScoped]</c>: the same transient set as the long-obj overload, plus the
        ///     bool-mask flatnonzero temp, is released on return.</remarks>
        [NDScoped] // same transient set as the long-obj overload, plus the bool-mask flatnonzero temp
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
                // nzIdx is released by this method's scope (exceptions included).
                var nzIdx = np.flatnonzero(obj);
                long[] indices = ToInt64Vector(nzIdx, "insert");
                return WithSourceOrder(arr, InsertMultiIndex(ctx.work, indices, values, ctx.axis));
            }

            if (obj.ndim > 1)
                throw new ArgumentException(
                    "index array argument obj to insert must be one dimensional or scalar",
                    nameof(obj));

            // 0-D integer ⇒ scalar path (true scalar semantics).
            if (obj.ndim == 0)
            {
                long idx = ToInt64Scalar(obj, "insert");
                return WithSourceOrder(arr, InsertSingleIndex(ctx.work, idx, values, ctx.axis, scalarObj: true));
            }

            // 1-D with size==1 still uses the single-index broadcast path but
            // with scalarObj=false (mirrors NumPy: indices.ndim == 1 prevents
            // the moveaxis quirk).
            long[] objIndices = ToInt64Vector(obj, "insert");
            if (objIndices.Length == 1)
                return WithSourceOrder(arr, InsertSingleIndex(ctx.work, objIndices[0], values, ctx.axis, scalarObj: false));

            return WithSourceOrder(arr, InsertMultiIndex(ctx.work, objIndices, values, ctx.axis));
        }

        // ---------------------------- impl ----------------------------

        /// <summary>
        ///     Single-index path: cast values to arr.dtype, normalise shape
        ///     (ndmin=arr.ndim, optional moveaxis for scalar-obj quirk),
        ///     concatenate pre + values + post along axis.
        /// </summary>
        /// <param name="arr">The working array (the input, or its ravel for <c>axis=None</c>).</param>
        /// <param name="index">The insertion index; negative values count from the end, <c>N</c> appends.</param>
        /// <param name="values">The values to insert.</param>
        /// <param name="axis">The normalized axis.</param>
        /// <param name="scalarObj">True when NumPy's scalar-obj quirk applies (moveaxis of the values' first axis).</param>
        /// <returns>A fresh array with the values inserted.</returns>
        /// <exception cref="IndexOutOfRangeException"><paramref name="index"/> is outside <c>[-N, N]</c>.</exception>
        /// <remarks>Covered by <c>[NDScopedCovered]</c>: its callers are the <c>[NDScoped]</c> <c>insert</c> overloads,
        ///     so the coerced values and the pre/post views are released by the caller's scope, exceptions
        ///     included.</remarks>
        [NDScopedCovered]
        private static NDArray InsertSingleIndex(
            NDArray arr, long index, NDArray values, int axis, bool scalarObj)
        {
            long N = arr.Shape.dimensions[axis];
            if (index < -N || index > N) // Insert allows index == N (append-at-end).
                throw new IndexOutOfRangeException(
                    $"index {index} is out of bounds for axis {axis} with size {N}");
            if (index < 0) index += N;

            // Cast values to arr.dtype (NumPy: values = array(values, dtype=arr.dtype))
            // and ensure ndmin = arr.ndim. We need to potentially moveaxis the first
            // axis to the insertion axis (scalarObj quirk).
            var coerced = CoerceValuesForSingleIndex(arr, values, axis, scalarObj);

            long numNew = coerced.Shape.dimensions[axis];
            if (numNew == 0)
                return DuplicateArray(arr);

            var pre = SliceAlongAxis(arr, axis, 0, index);
            var post = SliceAlongAxis(arr, axis, index, N);
            // Build the chunk list, dropping any empty-along-axis ones so
            // concatenate doesn't have to filter them itself.
            if (pre.Shape.dimensions[axis] == 0)
            {
                if (post.Shape.dimensions[axis] == 0)
                    return np.concatenate(new[] { coerced }, axis);
                return np.concatenate(new[] { coerced, post }, axis);
            }
            if (post.Shape.dimensions[axis] == 0)
                return np.concatenate(new[] { pre, coerced }, axis);
            return np.concatenate(new[] { pre, coerced, post }, axis);
        }

        /// <summary>
        ///     Multi-index path. Normalise indices (negative → positive,
        ///     bounds-check), stable-sort and reorder values, then build the
        ///     interleaved chunk list for a single concatenate call.
        /// </summary>
        /// <param name="arr">The working array (the input, or its ravel for <c>axis=None</c>).</param>
        /// <param name="indices">The insertion indices; normalized in place (the caller passes an array it owns).</param>
        /// <param name="values">The values to insert, one slot per index after broadcasting.</param>
        /// <param name="axis">The normalized axis.</param>
        /// <returns>A fresh array with the values interleaved at the sorted insertion positions.</returns>
        /// <exception cref="IndexOutOfRangeException">An index is outside <c>[-N, N]</c>.</exception>
        /// <remarks>Covered by <c>[NDScopedCovered]</c>: its callers are the <c>[NDScoped]</c> <c>insert</c>
        ///     overloads, so the coerced values, the reordering <c>take</c>, its order array and every chunk view are
        ///     released by the caller's scope, exceptions included. The chunk views share their bases' buffers, so
        ///     holding them until that scope closes keeps no extra memory alive.</remarks>
        [NDScopedCovered]
        private static NDArray InsertMultiIndex(
            NDArray arr, long[] indices, NDArray values, int axis)
        {
            long N = arr.Shape.dimensions[axis];

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
            var coerced = CoerceValuesForMultiIndex(arr, values, axis, numNew);

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
            bool needReorder = false;
            for (int i = 0; i < numNew; i++)
            {
                if (order[i] != i) { needReorder = true; break; }
            }

            if (needReorder)
            {
                var orderArr = np.array(LongArrayFromInt(order));
                reorderedValues = np.take(coerced, orderArr, axis: axis);
            }
            else
            {
                reorderedValues = coerced;
            }

            // Build interleaved chunk list: [arr[:s0], val0, arr[s0:s1], val1, ..., arr[s_{N-1}:end]]
            // For each sorted index s_k, take values[..., k:k+1, ...] as the value chunk.
            int totalChunks = numNew * 2 + 1;
            var chunks = new NDArray[totalChunks];
            int slot = 0;
            long prevSplit = 0;

            for (int k = 0; k < numNew; k++)
            {
                long s = sortedIndices[k];
                // arr chunk [prevSplit, s)
                var arrChunk = SliceAlongAxis(arr, axis, prevSplit, s);
                chunks[slot] = arrChunk;
                slot++;

                // values chunk along axis: [k, k+1)
                var valChunk = SliceAlongAxis(reorderedValues, axis, k, k + 1);
                chunks[slot] = valChunk;
                slot++;

                prevSplit = s;
            }

            // Tail: arr[prevSplit:N]
            var arrTail = SliceAlongAxis(arr, axis, prevSplit, N);
            chunks[slot] = arrTail;
            slot++;

            return np.concatenate(chunks, axis);
        }

        // ---------------------------- helpers ----------------------------

        /// <summary>
        ///     Wraps a scalar value as an NDArray of <paramref name="dtype"/>.
        ///     Used by the <c>insert(arr, obj, value, axis)</c> overloads that
        ///     accept an unboxed scalar so callers don't have to construct
        ///     a 0-D ndarray themselves.
        /// </summary>
        /// <param name="value">The scalar, C# array or collection, or an <see cref="NDArray"/> passed as object.</param>
        /// <param name="dtype">The target array's dtype.</param>
        /// <returns><paramref name="value"/> itself when it is an <see cref="NDArray"/>; otherwise its conversion,
        ///     cast to <paramref name="dtype"/> when the inferred dtype differs.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="value"/> is null.</exception>
        /// <remarks>Covered by <c>[NDScopedCovered]</c>: its callers are the <c>[NDScoped]</c> object-value
        ///     <c>insert</c> overloads, so the conversion that a cast supersedes, and the returned conversion itself,
        ///     are released by the caller's scope. An <see cref="NDArray"/> returned as-is is the caller's and is never
        ///     disposed.</remarks>
        [NDScopedCovered]
        private static NDArray ScalarValueAsArray(object value, DType dtype)
        {
            if (value is NDArray nd) return nd; // the caller's array, built before the scope opened: never disposed
            var asArr = np.asanyarray(value);
            if (!Equals(asArr.dtype, dtype))
            {
                var casted = asArr.astype(dtype, copy: true);
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
        /// <param name="arr">The working array whose dtype and shape the values adopt.</param>
        /// <param name="values">The values to insert.</param>
        /// <param name="axis">The normalized insertion axis.</param>
        /// <param name="scalarObj">True when NumPy's scalar-obj quirk applies (moveaxis of the values' first axis).</param>
        /// <returns>The values, cast and broadcast to the insertion slot; <paramref name="values"/> itself when it
        ///     already has the dtype and the slot's shape.</returns>
        /// <remarks>Covered by <c>[NDScopedCovered]</c>: its only caller is the covered <see cref="InsertSingleIndex"/>,
        ///     under the <c>[NDScoped]</c> <c>insert</c> overloads. The cast, the ndmin reshape, the moveaxis view and the
        ///     broadcast copy are released by that scope, and <paramref name="values"/> is never disposed. (The
        ///     hand-written version tracked ownership with a flag and wrapped a passed-through
        ///     <paramref name="values"/> in a second NDArray so the caller's <c>using</c> would not dispose the
        ///     caller's array; under the scope neither is needed.)</remarks>
        [NDScopedCovered]
        private static NDArray CoerceValuesForSingleIndex(
            NDArray arr, NDArray values, int axis, bool scalarObj)
        {
            // Cast dtype if needed.
            NDArray v = values;
            if (v.GetTypeCode != arr.GetTypeCode)
            {
                v = values.astype(arr.GetTypeCode, copy: true);
            }

            // ndmin = arr.ndim: prepend 1s.
            int targetNDim = Math.Max(arr.ndim, 1);
            if (v.ndim < targetNDim)
            {
                var newDims = new long[targetNDim];
                int prefix = targetNDim - v.ndim;
                for (int i = 0; i < prefix; i++) newDims[i] = 1;
                for (int i = 0; i < v.ndim; i++) newDims[prefix + i] = v.Shape.dimensions[i];
                var reshaped = v.reshape(Shape.Adopt(newDims));
                v = reshaped;
            }

            // Scalar-obj quirk: moveaxis(values, 0, axis) — moves the first axis
            // of values to the insertion axis position. This re-shapes values so
            // that broadcasting across non-axis dims behaves like
            // ``a[:, 0, :] = ...`` rather than ``a[:, [0], :] = ...``.
            if (scalarObj && v.ndim > 1 && axis != 0)
            {
                var moved = np.moveaxis(v, 0, axis);
                v = moved;
            }

            // Broadcast to arr.shape with axis dim = v.shape[axis].
            long numNew = arr.ndim == 0 ? v.size : v.Shape.dimensions[axis];
            var broadcasted = BroadcastValuesToInsertSlot(arr, v, axis, numNew);
            return broadcasted;
        }

        /// <summary>
        ///     Broadcasts <paramref name="values"/> to the shape
        ///     <c>arr.shape[..axis] + (numNew,) + arr.shape[axis+1..]</c>. The
        ///     result is always C-contiguous so the chunk-splitting + concatenate
        ///     pipeline can slice it without strided-view surprises.
        /// </summary>
        /// <param name="arr">The working array whose shape (except along <paramref name="axis"/>) is the target.</param>
        /// <param name="values">The dtype-matched values.</param>
        /// <param name="axis">The normalized insertion axis.</param>
        /// <param name="numNew">The extent of the insertion slot along <paramref name="axis"/>.</param>
        /// <returns><paramref name="values"/> itself when it already has the target shape; otherwise a C-contiguous
        ///     copy of its broadcast.</returns>
        /// <exception cref="IncorrectShapeException"><paramref name="values"/> cannot broadcast to the slot.</exception>
        /// <remarks>Covered by <c>[NDScopedCovered]</c>: its only caller is the covered
        ///     <see cref="CoerceValuesForSingleIndex"/>, so the stride-0 broadcast view is released by the
        ///     <c>insert</c> overload's scope.</remarks>
        [NDScopedCovered]
        private static NDArray BroadcastValuesToInsertSlot(
            NDArray arr, NDArray values, int axis, long numNew)
        {
            // Already matches? Return as-is.
            if (values.ndim == arr.ndim)
            {
                bool same = true;
                for (int i = 0; i < arr.ndim; i++)
                {
                    long expected = i == axis ? numNew : arr.Shape.dimensions[i];
                    if (values.Shape.dimensions[i] != expected) { same = false; break; }
                }
                if (same) return values;
            }

            // Build target shape and broadcast.
            var targetDims = new long[arr.ndim];
            for (int i = 0; i < arr.ndim; i++) targetDims[i] = arr.Shape.dimensions[i];
            targetDims[axis] = numNew;

            var bcast = np.broadcast_to(values, Shape.Adopt(targetDims));
            // Materialise to a contig owning copy — broadcast_to returns a
            // stride-0 read-only view that confuses np.take and SliceAlongAxis.
            var contig = np.ascontiguousarray(bcast);
            return contig;
        }

        /// <summary>
        ///     Coerces <paramref name="values"/> to <paramref name="arr"/>'s dtype
        ///     and broadcasts its shape to match
        ///     <c>arr.shape[..axis] + (numNew,) + arr.shape[axis+1..]</c>.
        ///     Used by the multi-index path before sort/reorder.
        /// </summary>
        /// <param name="arr">The working array whose dtype and shape the values adopt.</param>
        /// <param name="values">The values to insert.</param>
        /// <param name="axis">The normalized insertion axis.</param>
        /// <param name="numNew">The number of insertion positions (the slot's extent along <paramref name="axis"/>).</param>
        /// <returns>The values, cast and broadcast to the slot; <paramref name="values"/> itself when it already has
        ///     the dtype and the slot's shape.</returns>
        /// <exception cref="IncorrectShapeException"><paramref name="values"/> cannot broadcast to the slot.</exception>
        /// <remarks>Covered by <c>[NDScopedCovered]</c>: its only caller is the covered <see cref="InsertMultiIndex"/>,
        ///     under the <c>[NDScoped]</c> <c>insert</c> overloads. The cast, the ndmin reshape, the broadcast view and
        ///     its contiguous copy are released by that scope, and <paramref name="values"/> is never disposed. (The
        ///     hand-written version tracked ownership with a flag and wrapped a passed-through
        ///     <paramref name="values"/> in a second NDArray so the caller's <c>using</c> would not dispose the
        ///     caller's array; under the scope neither is needed.)</remarks>
        [NDScopedCovered]
        private static NDArray CoerceValuesForMultiIndex(
            NDArray arr, NDArray values, int axis, int numNew)
        {
            // Cast dtype.
            NDArray v = values;
            if (v.GetTypeCode != arr.GetTypeCode)
            {
                v = values.astype(arr.GetTypeCode, copy: true);
            }

            // Build target shape: arr.shape with axis dim = numNew.
            var targetDims = new long[arr.ndim];
            for (int i = 0; i < arr.ndim; i++) targetDims[i] = arr.Shape.dimensions[i];
            targetDims[axis] = numNew;
            var targetShape = Shape.Adopt(targetDims);

            // Shape already matches? Just hand it back (the caller's values or the cast copy).
            if (v.ndim == arr.ndim)
            {
                bool same = true;
                for (int i = 0; i < arr.ndim; i++)
                {
                    if (v.Shape.dimensions[i] != targetDims[i]) { same = false; break; }
                }
                if (same)
                {
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
                for (int i = 0; i < v.ndim; i++) newDims[prefix + i] = v.Shape.dimensions[i];
                var reshaped = v.reshape(Shape.Adopt(newDims));
                v = reshaped;
            }

            // Broadcast to target. broadcast_to handles size-1 stretching.
            var bcast = np.broadcast_to(v, targetShape);
            // broadcast_to returns a stride-0 read-only view; materialize a
            // contig owning copy so np.take and our SliceAlongAxis loop hit
            // the fast paths and we don't run into the "broadcast view + slice
            // by stepped-axis" quirks.
            var contig = np.ascontiguousarray(bcast);
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
