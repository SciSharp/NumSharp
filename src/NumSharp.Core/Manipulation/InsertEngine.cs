using System;
using NumSharp.Backends;
// NDExprTypeRules: NEP 50's check of a Python int against an integer dtype (the multi-index branch's `indices < 0` add).
using NumSharp.Backends.Iteration;
// The moved statements call np's internal insert / delete helpers (PrepareAxisContext, WithSourceOrder, SliceAlongAxis,
// DuplicateArray, ToInt64Scalar, ToInt64Vector, PythonSliceIndices) unqualified, as they did inside np; importing np's
// statics keeps them verbatim.
using static NumSharp.np;

namespace NumSharp
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
    //     (converted via flatnonzero), or a non-trivial slice. NumPy's
    //     statements decide the POSITIONS, in the index array's dtype:
    //         indices[indices < 0] += N
    //         indices[order] += arange(numnew)      (order: a stable argsort)
    //         old_mask[indices] = False             (M = N + numnew slots)
    //     so an index below -N lands at M + i, two indices may name one slot
    //     (the later value wins), and an index past N is an IndexError on the
    //     shifted value. The values must fit new[..., indices, ...] (MapIter's
    //     rule) and the array the slots old_mask keeps (one slab broadcasts).
    //     The data then moves in ONE concatenate, in position order:
    //                run_0 | val_0 | run_1 | val_1 | ... | run_n
    //     instead of NumPy's two fancy assignments; the concatenate form lets
    //     the IL contig-cast kernel fire instead of fancy-index scatter for
    //     each value slot.
    //
    // axis=None ravels arr first (axis becomes 0 of the flattened array).

    /// <summary>
    ///     NumPy's <c>numpy.insert</c> (<c>numpy/lib/_function_base_impl.py</c>) after its facade's class boundary: the
    ///     axis context (<c>axis=None</c> ravels), the positions in every C# form (a number, a slice, a sequence, an index
    ///     array or a boolean mask), the single-index and multi-index branches, the values' conversion into the array's
    ///     dtype, and the refusals of an index array that is neither integer nor boolean.
    /// </summary>
    /// <remarks>
    ///     An internal operation class (the np-layer dtype plan's D1: NumPy writes insert in Python), moved from the public
    ///     <c>np.insert</c> by P2; its twelve overloads keep their signatures, documentation, class boundaries, null checks
    ///     and one call here. It sits beside the facade, not under <c>Backends/</c>, because its statements call the public
    ///     np.* functions NumPy's call (np.concatenate, np.take, np.flatnonzero, np.moveaxis, np.broadcast_to, ...), which
    ///     ratchet R2 counts under <c>Backends/</c>. <see cref="IsIntegerIndexDtype"/> serves np.delete too.
    /// </remarks>
    internal static class InsertEngine
    {
        /// <summary>
        ///     <c>insert(arr, obj, values, axis)</c> with one position given as a number: the axis context, then the
        ///     single-index branch with NumPy's scalar-obj <c>moveaxis</c> of the values.
        /// </summary>
        /// <param name="arr">Input array (not null).</param>
        /// <param name="obj">The index before which the values are inserted; negative values count from the end.</param>
        /// <param name="values">The values to insert, or null (Python's None, converted after the index check).</param>
        /// <param name="axis">The axis to insert along; <c>null</c> inserts into the flattened array.</param>
        /// <returns>A fresh array with the values inserted (F-ordered when <paramref name="arr"/> is).</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="IndexOutOfRangeException"><paramref name="obj"/> is outside <c>[-N, N]</c>.</exception>
        /// <exception cref="OverflowException">A C# number literal outside the array dtype's range (NumPy's OverflowError,
        ///     <c>Python integer 1000 out of bounds for int8</c>).</exception>
        /// <exception cref="TypeError">None into an integer array (<c>int()</c>'s TypeError).</exception>
        /// <exception cref="ValueError">The values do not broadcast into the insertion slot (NumPy's view-assignment text,
        ///     <c>could not broadcast input array from shape (3,1) into shape (2,1)</c>); text values that do not parse as a
        ///     numeric array's number type.</exception>
        /// <remarks>Covered by <c>[NDScopedCovered]</c>: its only callers are the <c>[NDScoped]</c> np.insert overloads,
        ///     whose scope releases the ravel work array, the converted values and the intermediates.</remarks>
        [NDScopedCovered]
        internal static NDArray Insert(NDArray arr, long obj, NDArray values, int? axis)
        {
            var ctx = PrepareAxisContext(arr, axis);
            return WithSourceOrder(arr, InsertSingleIndex(ctx.work, obj, values, ctx.axis, scalarObj: true, obj));
        }

        /// <summary>
        ///     <c>insert(arr, obj, values, axis)</c> with the positions as a slice: the axis context, the slice expanded with
        ///     Python's <c>slice.indices(N)</c>, then the single-index branch for one position (without the scalar-obj
        ///     <c>moveaxis</c>, so every value is inserted there) or the multi-index branch.
        /// </summary>
        /// <param name="arr">Input array (not null).</param>
        /// <param name="obj">The insertion positions (not null).</param>
        /// <param name="values">The values to insert, or null (Python's None).</param>
        /// <param name="axis">The axis to insert along; <c>null</c> inserts into the flattened array.</param>
        /// <returns>A fresh array with the values inserted (F-ordered when <paramref name="arr"/> is).</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="OverflowException">A C# number literal outside the array dtype's range (NumPy's OverflowError).</exception>
        /// <exception cref="TypeError">None into an integer array (<c>int()</c>'s TypeError).</exception>
        /// <exception cref="ValueError">The values do not fit the insertion slots: for one position NumPy's view-assignment
        ///     text (<c>could not broadcast input array from shape ... into shape ...</c>), for several MapIter's (<c>shape
        ///     mismatch: value array of shape ... could not be broadcast to indexing result of shape ...</c>); the array
        ///     does not fill the slots left when two positions name one slot (<see cref="InsertMultiIndex"/>); text values
        ///     that do not parse as a numeric array's number type.</exception>
        /// <remarks>Covered by <c>[NDScopedCovered]</c>, as <see cref="Insert(NDArray, long, NDArray, int?)"/>.</remarks>
        [NDScopedCovered]
        internal static NDArray Insert(NDArray arr, Slice obj, NDArray values, int? axis)
        {
            var ctx = PrepareAxisContext(arr, axis);
            long N = ctx.work.Shape.dimensions[ctx.axis];
            var indices = InsertExpandSliceObj(obj, N);
            // NumPy's `if indices.size == 1:` holds for a slice's arange too, so one position takes the single-index branch.
            // A slice obj is never scalar for the broadcast quirk (its arange is 1-D, indices.ndim == 1): the values keep
            // their shape there, so np.insert(arange(4), slice(1, 2), [7, 8]) inserts both values at 1 (probed 2.4.2), where
            // the multi-index branch would refuse two values for one position.
            if (indices.Length == 1)
                return WithSourceOrder(arr, InsertSingleIndex(ctx.work, indices[0], values, ctx.axis, scalarObj: false, obj));
            return WithSourceOrder(arr, InsertMultiIndex(ctx.work, indices, NPTypeCode.Int64, values, ctx.axis));
        }

        /// <summary>
        ///     <c>insert(arr, obj, values, axis)</c> with the positions as an <c>int[]</c> (NumPy's sequence obj): widened to
        ///     <c>long</c>, then the single-index branch for one position (without the scalar-obj <c>moveaxis</c>) or the
        ///     multi-index branch.
        /// </summary>
        /// <param name="arr">Input array (not null).</param>
        /// <param name="obj">The insertion positions (not null).</param>
        /// <param name="values">The values to insert, or null (Python's None).</param>
        /// <param name="axis">The axis to insert along; <c>null</c> inserts into the flattened array.</param>
        /// <returns>A fresh array with the values inserted (F-ordered when <paramref name="arr"/> is).</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="IndexOutOfRangeException">One position outside <c>[-N, N]</c> (NumPy's text prints the
        ///     sequence, <c>index [13] is out of bounds ...</c>); of several, a SHIFTED position outside <c>[-M, M)</c>,
        ///     <c>M = N + numnew</c> (<c>index 14 is out of bounds for axis 0 with size 14</c>).</exception>
        /// <exception cref="OverflowException">A C# number literal outside the array dtype's range (NumPy's OverflowError).</exception>
        /// <exception cref="TypeError">None into an integer array (<c>int()</c>'s TypeError).</exception>
        /// <exception cref="ValueError">The values do not fit the insertion slots: for one position NumPy's view-assignment
        ///     text (<c>could not broadcast input array from shape ... into shape ...</c>), for several MapIter's (<c>shape
        ///     mismatch: value array of shape ... could not be broadcast to indexing result of shape ...</c>); the array
        ///     does not fill the slots left when two positions name one slot (<see cref="InsertMultiIndex"/>); text values
        ///     that do not parse as a numeric array's number type.</exception>
        /// <remarks>Covered by <c>[NDScopedCovered]</c>, as <see cref="Insert(NDArray, long, NDArray, int?)"/>.</remarks>
        [NDScopedCovered]
        internal static NDArray Insert(NDArray arr, int[] obj, NDArray values, int? axis)
        {
            var longs = new long[obj.Length];
            for (int i = 0; i < obj.Length; i++) longs[i] = obj[i];

            var ctx = PrepareAxisContext(arr, axis);
            // One position: the single-index branch, whose IndexError prints the sequence as a Python list (index [13]).
            if (longs.Length == 1)
                return WithSourceOrder(arr, InsertSingleIndex(ctx.work, longs[0], values, ctx.axis, scalarObj: false, longs));
            return WithSourceOrder(arr, InsertMultiIndex(ctx.work, longs, NPTypeCode.Int64, values, ctx.axis));
        }

        /// <summary>
        ///     <c>insert(arr, obj, values, axis)</c> with the positions as a <c>long[]</c>: the single-index branch for one
        ///     position (without the scalar-obj <c>moveaxis</c>) or the multi-index branch over a copy of the positions.
        /// </summary>
        /// <param name="arr">Input array (not null).</param>
        /// <param name="obj">The insertion positions (not null; never written: the multi-index branch normalizes a copy).</param>
        /// <param name="values">The values to insert, or null (Python's None).</param>
        /// <param name="axis">The axis to insert along; <c>null</c> inserts into the flattened array.</param>
        /// <returns>A fresh array with the values inserted (F-ordered when <paramref name="arr"/> is).</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="IndexOutOfRangeException">One position outside <c>[-N, N]</c> (NumPy's text prints the
        ///     sequence, <c>index [13] is out of bounds ...</c>); of several, a SHIFTED position outside <c>[-M, M)</c>,
        ///     <c>M = N + numnew</c> (<c>index 14 is out of bounds for axis 0 with size 14</c>).</exception>
        /// <exception cref="OverflowException">A C# number literal outside the array dtype's range (NumPy's OverflowError).</exception>
        /// <exception cref="TypeError">None into an integer array (<c>int()</c>'s TypeError).</exception>
        /// <exception cref="ValueError">The values do not fit the insertion slots: for one position NumPy's view-assignment
        ///     text (<c>could not broadcast input array from shape ... into shape ...</c>), for several MapIter's (<c>shape
        ///     mismatch: value array of shape ... could not be broadcast to indexing result of shape ...</c>); the array
        ///     does not fill the slots left when two positions name one slot (<see cref="InsertMultiIndex"/>); text values
        ///     that do not parse as a numeric array's number type.</exception>
        /// <remarks>Covered by <c>[NDScopedCovered]</c>, as <see cref="Insert(NDArray, long, NDArray, int?)"/>.</remarks>
        [NDScopedCovered]
        internal static NDArray Insert(NDArray arr, long[] obj, NDArray values, int? axis)
        {
            var ctx = PrepareAxisContext(arr, axis);
            // One position: the single-index branch, whose IndexError prints the sequence as a Python list (index [13]).
            if (obj.Length == 1)
                return WithSourceOrder(arr, InsertSingleIndex(ctx.work, obj[0], values, ctx.axis, scalarObj: false, obj));
            return WithSourceOrder(arr, InsertMultiIndex(ctx.work, (long[])obj.Clone(), NPTypeCode.Int64, values, ctx.axis));
        }

        /// <summary>
        ///     <c>insert(arr, obj, values, axis)</c> with the positions as an array: a 1-D boolean mask is its
        ///     <c>flatnonzero</c> (the single-index branch without the <c>moveaxis</c> for one True, else the multi-index
        ///     branch); an index array of more than one dimension is NumPy's ValueError;
        ///     an index array that is not integer takes NumPy's statements up to their refusal; a 0-d integer index is the
        ///     scalar single-index branch, a one-element one the single-index branch without the <c>moveaxis</c>, and any other
        ///     the multi-index branch.
        /// </summary>
        /// <param name="arr">Input array (not null).</param>
        /// <param name="obj">A 0-d or 1-D integer index array, or a 1-D boolean mask (not null).</param>
        /// <param name="values">The values to insert, or null (Python's None).</param>
        /// <param name="axis">The axis to insert along; <c>null</c> inserts into the flattened array.</param>
        /// <returns>A fresh array with the values inserted (F-ordered when <paramref name="arr"/> is).</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="ValueError"><paramref name="obj"/> is a boolean mask that is not 1-D, or an index array of more
        ///     than one dimension (NumPy's two texts); the values do not fit the insertion slots (one position: NumPy's
        ///     view-assignment text; several: MapIter's <c>shape mismatch: ...</c>, or the array does not fill the slots
        ///     left when two positions name one slot); text values that do not parse as a numeric array's number
        ///     type.</exception>
        /// <exception cref="TypeError">A one-element text or complex <paramref name="obj"/>, a text <paramref name="obj"/> of
        ///     any other size, or a one-element float within bounds (see <see cref="InsertNonIntegerObj"/>); None into an
        ///     integer array (<c>int()</c>'s TypeError).</exception>
        /// <exception cref="IndexError">A float or complex <paramref name="obj"/> of any size but one, or a one-element float
        ///     out of bounds.</exception>
        /// <exception cref="IndexOutOfRangeException">One integer position outside <c>[-N, N]</c>, or a mask's one True
        ///     past <c>N</c> (NumPy's text prints the array or the mask, <c>index [13] ...</c>); of several, a SHIFTED
        ///     position outside <c>[-M, M)</c>, <c>M = N + numnew</c>.</exception>
        /// <exception cref="OverflowException">A C# number literal outside the array dtype's range (NumPy's OverflowError),
        ///     or, for several positions, an axis length the index dtype cannot hold (<c>Python integer 300 out of bounds
        ///     for uint8</c>).</exception>
        /// <exception cref="ArgumentException">Several positions in an UNSIGNED index array: NumPy's in-place shift cannot cast
        ///     its int64 result back (<c>Cannot cast ufunc 'add' output from dtype('int64') to dtype('uint8') with casting
        ///     rule 'same_kind'</c>, NumSharp's ufunc output-cast refusal); an empty one too.</exception>
        /// <remarks>Covered by <c>[NDScopedCovered]</c>, as <see cref="Insert(NDArray, long, NDArray, int?)"/>; the
        ///     boolean mask's flatnonzero temp is released by the same scope.</remarks>
        [NDScopedCovered]
        internal static NDArray Insert(NDArray arr, NDArray obj, NDArray values, int? axis)
        {
            var ctx = PrepareAxisContext(arr, axis);

            if (obj.GetTypeCode == NPTypeCode.Boolean)
            {
                if (obj.ndim != 1)
                    throw new ValueError("boolean array argument obj to insert must be one dimensional");
                // nzIdx is released by this method's scope (exceptions included).
                var nzIdx = np.flatnonzero(obj);
                long[] indices = ToInt64Vector(nzIdx, "insert");
                // A mask with one True is one position (NumPy's `if indices.size == 1:` follows the flatnonzero): the
                // single-index branch, the values keeping their shape (flatnonzero is 1-D: no moveaxis), and an
                // IndexError that prints the whole mask, as NumPy's f"index {obj} ..." does.
                if (indices.Length == 1)
                    return WithSourceOrder(arr, InsertSingleIndex(ctx.work, indices[0], values, ctx.axis, scalarObj: false, obj));
                return WithSourceOrder(arr, InsertMultiIndex(ctx.work, indices, NPTypeCode.Int64, values, ctx.axis));
            }

            if (obj.ndim > 1)
                throw new ValueError("index array argument obj to insert must be one dimensional or scalar");

            // An index array that is not integer: NumPy compares it with 0 before it indexes with it.
            if (!IsIntegerIndexDtype(obj.typecode))
                throw InsertNonIntegerObj(ctx.work, obj, values, ctx.axis);

            // 0-D integer ⇒ scalar path (true scalar semantics); its IndexError prints the array (index 13).
            if (obj.ndim == 0)
            {
                long idx = ToInt64Scalar(obj, "insert");
                return WithSourceOrder(arr, InsertSingleIndex(ctx.work, idx, values, ctx.axis, scalarObj: true, obj));
            }

            // 1-D with size==1 still uses the single-index broadcast path but
            // with scalarObj=false (mirrors NumPy: indices.ndim == 1 prevents
            // the moveaxis quirk). Its IndexError prints the array (index [13]).
            long[] objIndices = ToInt64Vector(obj, "insert");
            if (objIndices.Length == 1)
                return WithSourceOrder(arr, InsertSingleIndex(ctx.work, objIndices[0], values, ctx.axis, scalarObj: false, obj));

            return WithSourceOrder(arr, InsertMultiIndex(ctx.work, objIndices, obj.typecode, values, ctx.axis));
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
        /// <param name="obj">The caller's positions argument as the IndexError prints it (<see cref="PythonStr"/>): the
        ///     <c>long</c> of a number, the <c>long[]</c> of a sequence, the index array or boolean mask itself, or the
        ///     <see cref="Slice"/>.</param>
        /// <returns>A fresh array with the values inserted.</returns>
        /// <exception cref="IndexOutOfRangeException"><paramref name="index"/> is outside <c>[-N, N]</c>: NumPy's
        ///     <c>f"index {obj} is out of bounds for axis {axis} with size {N}"</c>, which prints the ARGUMENT (<c>index [13]</c>
        ///     for a one-element sequence or array, the whole mask for a boolean mask), not the position.</exception>
        /// <exception cref="ValueError">The values do not broadcast into the slot (see
        ///     <see cref="BroadcastValuesToInsertSlot"/>), or text values do not parse as the array's number type.</exception>
        /// <remarks>Covered by <c>[NDScopedCovered]</c>: its callers are the <c>[NDScoped]</c> <c>insert</c> overloads,
        ///     so the coerced values and the pre/post views are released by the caller's scope, exceptions
        ///     included.</remarks>
        [NDScopedCovered]
        private static NDArray InsertSingleIndex(
            NDArray arr, long index, NDArray values, int axis, bool scalarObj, object obj)
        {
            long N = arr.Shape.dimensions[axis];
            // Insert allows index == N (append-at-end). The text is built only here, on the error path: a mask's str() is
            // NumPy's array printing.
            if (index < -N || index > N)
                throw new IndexOutOfRangeException(
                    $"index {PythonStr(obj)} is out of bounds for axis {axis} with size {N}");
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
        ///     NumPy's multi-index branch, statement for statement on the positions —
        ///     <c>indices[indices &lt; 0] += N</c>; <c>order = indices.argsort(kind='mergesort')</c>;
        ///     <c>indices[order] += np.arange(numnew)</c>; <c>old_mask = ones(N + numnew, dtype=bool)</c>;
        ///     <c>old_mask[indices] = False</c>; <c>new[..., indices, ...] = values</c>; <c>new[..., old_mask, ...] = arr</c> —
        ///     with the data moved by ONE concatenate of the array's runs and the value slabs in position order instead of
        ///     the two fancy assignments.
        /// </summary>
        /// <param name="arr">The working array (the input, or its ravel for <c>axis=None</c>).</param>
        /// <param name="indices">NumPy's index array, as <c>long</c> values; rewritten in place (the caller passes an array
        ///     it owns).</param>
        /// <param name="indexType">The dtype NumPy's index array has: int64 for a slice (<c>arange(..., dtype=intp)</c>), a
        ///     sequence or a mask's <c>flatnonzero</c>; the array's own dtype for an index array.</param>
        /// <param name="values">The values to insert; null is Python's None.</param>
        /// <param name="axis">The normalized axis.</param>
        /// <returns>A fresh array with the values at NumPy's positions and the array's slabs in the slots left over.</returns>
        /// <exception cref="OverflowException">The axis length does not fit the index dtype: <c>indices &lt; 0</c> meets the
        ///     Python int <c>N</c>, which NEP 50 packs into that dtype (<c>Python integer 300 out of bounds for uint8</c>);
        ///     also a C# number literal outside the array dtype's range.</exception>
        /// <exception cref="ArgumentException">An unsigned index dtype: the in-place <c>indices[order] += np.arange(numnew)</c>
        ///     cannot cast its int64 result back (<c>Cannot cast ufunc 'add' output from dtype('int64') to dtype('uint8')
        ///     with casting rule 'same_kind'</c>; NumSharp's ufunc output-cast refusal, NumPy's UFuncTypeError).</exception>
        /// <exception cref="IndexOutOfRangeException">A shifted position is outside <c>[-M, M)</c>, <c>M = N + numnew</c>:
        ///     <c>index 14 is out of bounds for axis 0 with size 14</c> (the mask's axis 0, whatever the insertion axis),
        ///     the first such index in the array's order.</exception>
        /// <exception cref="ValueError">The values do not broadcast to the indexing result (MapIter's <c>shape mismatch: value
        ///     array of shape ... could not be broadcast to indexing result of shape ...</c>, an empty index included), or the
        ///     array's slabs do not fill the slots the mask keeps (<c>NumPy boolean array indexing assignment cannot assign 2
        ///     input values to the 3 output values where the mask is true</c> for a 1-D array, MapIter's text otherwise);
        ///     text values that do not parse.</exception>
        /// <exception cref="TypeError">None into an integer array (<c>int()</c>'s TypeError).</exception>
        /// <remarks>
        ///     <para>
        ///     NumPy's shift makes an index below <c>-N</c> land at <c>M + i</c> (<c>np.insert(arange(12), [-13, 1], [100,
        ///     200])</c> puts 100 LAST), lets two indices name one position (the later value wins and the mask keeps one slot
        ///     more, which only a one-slab array fills), and turns every index past <c>N</c> into an IndexError. The positions
        ///     are therefore computed first and the result is built from them.
        ///     </para>
        ///     <para>
        ///     Covered by <c>[NDScopedCovered]</c>: its callers are the <c>[NDScoped]</c> <c>insert</c> overloads, so the
        ///     coerced values, the reordering <c>take</c>, its order array, the broadcast runs and every chunk view are
        ///     released by the caller's scope, exceptions included. The chunk views share their bases' buffers, so holding
        ///     them until that scope closes keeps no extra memory alive.
        ///     </para>
        /// </remarks>
        [NDScopedCovered]
        private static NDArray InsertMultiIndex(
            NDArray arr, long[] indices, NPTypeCode indexType, NDArray values, int axis)
        {
            long N = arr.Shape.dimensions[axis];
            int numNew = indices.Length;

            // indices[indices < 0] += N. The in-place add meets the Python int N, which NEP 50 packs into the index dtype
            // first, so an N that dtype cannot hold raises before anything else, even when no index is negative
            // (np.insert(np.arange(300), np.array([0, 2], np.uint8), [1, 2]) raises "Python integer 300 out of bounds for
            // uint8", probed 2.4.2). A negative index plus N cannot overflow a signed dtype: the two have opposite signs.
            NDExprTypeRules.CheckIntLiteralFits(N, indexType);
            for (int i = 0; i < numNew; i++)
                if (indices[i] < 0)
                    indices[i] += N;

            // order = indices.argsort(kind='mergesort'); indices[order] += np.arange(numnew). The in-place add casts its
            // int64 result back into the index dtype under same_kind: an unsigned dtype refuses it (an empty index array
            // too), a narrower signed dtype wraps.
            int[] order = StableArgSort(indices);
            DefaultEngine.ValidateOutCast(NPTypeCode.Int64, indexType, "add");
            for (int k = 0; k < numNew; k++)
                indices[order[k]] = WrapToIndexType(indices[order[k]] + k, indexType);

            // old_mask = ones(N + numnew, dtype=bool); old_mask[indices] = False. The fancy setitem checks the indices in
            // the array's order against the mask's length M and wraps a negative one (an original index below -N).
            long M = N + numNew;
            var positions = new long[numNew];
            for (int i = 0; i < numNew; i++)
            {
                long p = indices[i];
                if (p < -M || p >= M)
                    throw new IndexOutOfRangeException($"index {p} is out of bounds for axis 0 with size {M}");
                positions[i] = p < 0 ? p + M : p;
            }

            // new[tuple(slobj)] = values: converted into arr's dtype, then MapIter's broadcast to the indexing result
            // (arr's shape with numnew along the axis), so an empty index still refuses values that do not fit (0,).
            var coerced = CoerceValuesForMultiIndex(arr, values, axis, numNew);

            // The positions in increasing order. Where several indices name one position the LAST of them wins (the fancy
            // setitem writes in the index array's order: a stable sort keeps that order within equal positions).
            int[] byPosition = StableArgSort(positions);
            var slots = new long[numNew];
            var writers = new int[numNew];
            int distinct = 0;
            for (int i = 0; i < numNew; i++)
            {
                long p = positions[byPosition[i]];
                if (distinct > 0 && slots[distinct - 1] == p)
                {
                    writers[distinct - 1] = byPosition[i];
                    continue;
                }
                slots[distinct] = p;
                writers[distinct] = byPosition[i];
                distinct++;
            }

            // new[tuple(slobj2)] = arr: the M - distinct slots the mask keeps receive the array's N slabs along the axis. One
            // slab broadcasts to any number of slots; any other mismatch is the boolean assignment's refusal, whose text
            // is array_assign_boolean_subscript's for a 1-D array (a lone 1-D mask) and MapIter's otherwise.
            long count = M - distinct;
            if (count != N && N != 1)
            {
                if (arr.ndim == 1)
                    throw new ValueError($"NumPy boolean array indexing assignment cannot assign {N} input values to the " +
                                         $"{count} output values where the mask is true");
                var resultDims = (long[])arr.Shape.dimensions.Clone();
                resultDims[axis] = count;
                throw new ValueError($"shape mismatch: value array of shape {arr.Shape.ToPythonTuple()} could not be " +
                                     $"broadcast to indexing result of shape {Shape.Adopt(resultDims).ToPythonTuple()}");
            }

            // No value slot (an empty index): the array alone, NumPy's copy.
            if (distinct == 0)
                return DuplicateArray(arr);

            // The value slabs in slot order: values[..., writers, ...] (np.take along the axis) unless they already are.
            bool inOrder = distinct == numNew;
            for (int j = 0; inOrder && j < distinct; j++)
                inOrder = writers[j] == j;
            NDArray slabs = coerced;
            if (!inOrder)
            {
                var writerOrder = new long[distinct];
                for (int j = 0; j < distinct; j++) writerOrder[j] = writers[j];
                slabs = np.take(coerced, np.array(writerOrder), axis: axis);
            }

            // One concatenate in position order: a run of the array's slabs before each value slot, then the slot's slab.
            // With one slab broadcast over a different number of slots, every run is that slab broadcast to its length.
            bool broadcastArr = count != N;
            var chunks = new System.Collections.Generic.List<NDArray>(2 * distinct + 1);
            long nextPos = 0, nextSlab = 0;
            for (int j = 0; j < distinct; j++)
            {
                long run = slots[j] - nextPos;
                if (run > 0)
                {
                    chunks.Add(ArrayRun(arr, axis, nextSlab, run, broadcastArr));
                    nextSlab += run;
                }
                chunks.Add(SliceAlongAxis(slabs, axis, j, j + 1));
                nextPos = slots[j] + 1;
            }
            if (M - nextPos > 0)
                chunks.Add(ArrayRun(arr, axis, nextSlab, M - nextPos, broadcastArr));

            return np.concatenate(chunks.ToArray(), axis);
        }

        /// <summary>
        ///     A run of <paramref name="length"/> of the array's slabs along <paramref name="axis"/> for the multi-index
        ///     result: the next slabs in order, or the array's ONE slab broadcast to the run when the mask keeps another
        ///     number of slots than the array has (NumPy's boolean assignment broadcasts a one-slab value).
        /// </summary>
        /// <param name="arr">The working array.</param>
        /// <param name="axis">The insertion axis.</param>
        /// <param name="start">The first slab of the run (ignored when broadcasting).</param>
        /// <param name="length">The run's length (at least 1).</param>
        /// <param name="broadcast">True to broadcast the array's single slab.</param>
        /// <returns>A view: a slice of <paramref name="arr"/>, or a read-only stride-0 broadcast of it.</returns>
        /// <remarks>Covered by <c>[NDScopedCovered]</c>: its only caller is the covered <see cref="InsertMultiIndex"/>, so the
        ///     view is released by the <c>insert</c> overload's scope.</remarks>
        [NDScopedCovered]
        private static NDArray ArrayRun(NDArray arr, int axis, long start, long length, bool broadcast)
        {
            if (!broadcast)
                return SliceAlongAxis(arr, axis, start, start + length);
            var dims = (long[])arr.Shape.dimensions.Clone();
            dims[axis] = length;
            return np.broadcast_to(arr, Shape.Adopt(dims));
        }

        /// <summary>
        ///     A position wrapped into NumPy's index dtype, as the cast back of <c>indices[order] += np.arange(numnew)</c>
        ///     wraps it: int8 / int16 / int32 keep the low bits (two's complement), int64 is the value itself.
        /// </summary>
        /// <param name="value">The shifted position, computed in int64.</param>
        /// <param name="indexType">The index dtype (signed: the unsigned ones were refused by the add's cast).</param>
        /// <returns>The value as the index dtype holds it, widened back to <c>long</c>.</returns>
        private static long WrapToIndexType(long value, NPTypeCode indexType) => indexType switch
        {
            NPTypeCode.SByte => unchecked((sbyte)value),
            NPTypeCode.Int16 => unchecked((short)value),
            NPTypeCode.Int32 => unchecked((int)value),
            _ => value,
        };

        // ---------------------------- helpers ----------------------------

        /// <summary>
        ///     Python's <c>str(obj)</c> of np.insert's positions argument, as the single-index branch's IndexError prints it
        ///     (<c>f"index {obj} is out of bounds ..."</c>): a number as itself (<c>index 13</c>), a sequence as a Python list
        ///     (<c>index [13]</c>; a C# <c>int[]</c> arrives widened to <c>long[]</c>), an index array or a boolean mask as
        ///     NumPy's <c>ndarray.__str__</c> (<c>[13]</c>, <c>13</c> for a 0-d array, <c>[False  True]</c>).
        /// </summary>
        /// <param name="obj">The argument: a boxed <c>long</c>, a <c>long[]</c>, an <see cref="NDArray"/>, or a
        ///     <see cref="Slice"/>.</param>
        /// <returns>The text.</returns>
        /// <remarks>A <see cref="Slice"/> never reaches the error: <c>slice.indices(N)</c> clamps its positions to
        ///     <c>[0, N]</c>, so its one position is always in bounds; it falls to <see cref="object.ToString"/>.</remarks>
        private static string PythonStr(object obj) => obj switch
        {
            long number => number.ToString(System.Globalization.CultureInfo.InvariantCulture),
            // repr() of a Python list of ints: "[a, b]" (only a one-element list reaches the single-index branch).
            long[] list => "[" + string.Join(", ", Array.ConvertAll(list,
                v => v.ToString(System.Globalization.CultureInfo.InvariantCulture))) + "]",
            NDArray array => array.ToString(),
            _ => obj?.ToString(),
        };

        /// <summary>
        ///     Whether NumPy indexes with an array of this dtype as integers (dtype kind <c>'u'</c> or <c>'i'</c>; NumSharp's
        ///     <c>char</c> rides the integer route, as everywhere in the library).
        /// </summary>
        /// <param name="typeCode">The index array's type code.</param>
        /// <returns>True for the eight integer types and <c>char</c>.</returns>
        /// <remarks>Internal: np.delete's index-array test uses the same set.</remarks>
        internal static bool IsIntegerIndexDtype(NPTypeCode typeCode)
            => typeCode is NPTypeCode.Byte or NPTypeCode.SByte or NPTypeCode.Int16 or NPTypeCode.UInt16
                or NPTypeCode.Int32 or NPTypeCode.UInt32 or NPTypeCode.Int64 or NPTypeCode.UInt64 or NPTypeCode.Char;

        /// <summary>
        ///     NumPy's <c>np.insert</c> with an index array that is neither boolean nor integer — text, a float, complex
        ///     (and NumSharp's decimal, read as a float) — up to the error NumPy reaches:
        ///     <list type="bullet">
        ///       <item><b>One element</b>: <c>index = indices.item()</c> is a Python <c>str</c> / <c>bytes</c> / <c>complex</c>
        ///             (<c>'&lt;' not supported between instances of 'str' and 'int'</c>) or a float, which is bounds-checked
        ///             (<c>index [5.5] is out of bounds for axis 0 with size 3</c>, the obj printed as NumPy's <c>str</c>), then
        ///             the values are converted to the array's dtype, and the slice it builds refuses the float (<c>slice indices
        ///             must be integers or None or have an __index__ method</c>).</item>
        ///       <item><b>Several or no elements</b>: <c>indices[indices &lt; 0] += N</c> has no loop for text and a Python int
        ///             (<c>ufunc 'less' did not contain a loop …</c>); a number passes, and the keep-mask indexing refuses it
        ///             (<c>arrays used as indices must be of integer (or boolean) type</c>).</item>
        ///     </list>
        /// </summary>
        /// <param name="arr">The working array (the input, or its ravel for <c>axis=None</c>).</param>
        /// <param name="obj">The non-integer, non-boolean index array (at most 1-D).</param>
        /// <param name="values">The values to insert (converted, as NumPy does, before the slice error).</param>
        /// <param name="axis">The normalized axis.</param>
        /// <returns>The exception to throw.</returns>
        /// <exception cref="ValueError">Thrown (not returned) when converting the values parses text that is not a number
        ///     — NumPy's step before its slice error.</exception>
        /// <exception cref="UnicodeError">Thrown (not returned) when the values do not convert between the string classes.</exception>
        /// <remarks>Covered by <c>[NDScopedCovered]</c>: its only caller is the covered
        ///     <see cref="Insert(NDArray, NDArray, NDArray, int?)"/>, under the <c>[NDScoped]</c> np.insert overloads, so the
        ///     values converted for NumPy's step before the slice error are released by that scope.</remarks>
        [NDScopedCovered]
        private static Exception InsertNonIntegerObj(NDArray arr, NDArray obj, NDArray values, int axis)
        {
            bool text = obj.typecode == NPTypeCode.String;
            if (obj.size == 1)
            {
                if (text)
                {
                    // `indices.item()` is a Python object: a fixed-width element is a str / bytes, a StringDType element a
                    // str or its NA object (None; a NaN-like NA is a float, which takes the float route below).
                    string pyType = Backends.Strings.NDStringUfuncs.IsVString(obj)
                        ? obj.GetAtIndex(0) switch { null => "NoneType", string => "str", _ => null }
                        : obj.Storage.StringSlice.IsUnicode ? "str" : "bytes";
                    if (pyType is not null)
                        return new TypeError($"'<' not supported between instances of '{pyType}' and 'int'");
                }
                if (obj.typecode == NPTypeCode.Complex)
                    return new TypeError("'<' not supported between instances of 'complex' and 'int'");

                // A Python float: NaN fails both comparisons and so passes the bounds check, as in NumPy.
                double index = Convert.ToDouble(obj.GetAtIndex(0));
                long N = arr.Shape.dimensions[axis];
                if (index < -N || index > N)
                    return new IndexError($"index {obj} is out of bounds for axis {axis} with size {N}");
                // values = array(values, ndmin=arr.ndim, dtype=arr.dtype) (and the 0-d index's moveaxis) runs before the
                // slice is built; the values are never assigned, so their broadcast is not checked (a column of two into
                // one position still reports the slice's TypeError, probed 2.4.2).
                ConvertValuesForSingleIndex(arr, values, axis, scalarObj: obj.ndim == 0);
                return new TypeError("slice indices must be integers or None or have an __index__ method");
            }

            // `indices < 0`: text has no loop with a Python int (NumPy's weak int class), even for no element.
            if (text)
                return new TypeError($"ufunc 'less' did not contain a loop with signature matching types ({obj.dtype.Meta}, {DTypeRegistry.PyLong}) -> None");
            return new IndexError("arrays used as indices must be of integer (or boolean) type");
        }

        /// <summary>
        ///     The array an object value stands for, for the <c>insert(arr, obj, value, axis)</c> overloads, NOT yet converted
        ///     to the target array's dtype: the NDArray overload converts it where NumPy converts its values
        ///     (<c>array(values, dtype=arr.dtype)</c> after the single-index bounds check, the fancy setitem of the
        ///     multi-index branch), so an axis or index error comes first, as in NumPy.
        /// </summary>
        /// <param name="value">The scalar, C# array or collection, or an <see cref="NDArray"/> passed as object; <c>null</c>
        ///     is Python's <c>None</c>.</param>
        /// <param name="dtype">The target array's dtype.</param>
        /// <returns><paramref name="value"/> itself when it is an <see cref="NDArray"/>; <c>null</c> for <c>None</c>; a C#
        ///     number, bool or string as the marked 0-d array its implicit conversion makes
        ///     (<see cref="NDArray.FromPythonOperand"/>: a Python int / float / complex / bool / str, or a NumPy scalar for
        ///     <see cref="Half"/> and <c>char</c>), which <c>CoerceValuesForSingleIndex</c> / <c>CoerceValuesForMultiIndex</c>
        ///     pack with NumPy's <c>PyArray_Pack</c> rules; a C# array or collection as <c>np.asanyarray(value)</c>, cast
        ///     there too. A <c>StringDType</c> target, and a <see cref="System.Numerics.BigInteger"/> (a Python int past the
        ///     machine types, which has no marked 0-d form), are converted here, as before.</returns>
        /// <exception cref="OverflowException">A <see cref="System.Numerics.BigInteger"/> outside the target dtype's range
        ///     (NumPy's OverflowError: raised here, before the index checks, where NumPy raises it after them).</exception>
        /// <exception cref="ValueError">A <c>StringDType(coerce=False)</c> target and a non-text value (its setitem refusal,
        ///     raised here).</exception>
        /// <remarks>Covered by <c>[NDScopedCovered]</c>: its callers are the <c>[NDScoped]</c> object-value
        ///     <c>insert</c> overloads, so the array it builds is released by the caller's scope. An <see cref="NDArray"/>
        ///     returned as-is is the caller's and is never disposed.</remarks>
        [NDScopedCovered]
        internal static NDArray ScalarValueAsArray(object value, DType dtype)
        {
            // None stays null: the array overload's value coercion converts it after the index checks.
            if (value is null) return null;
            if (value is NDArray nd) return nd; // the caller's array, built before the scope opened: never disposed
            // A StringDType target: NumPy's array(values, dtype=arr.dtype) stores Python objects by setitem.
            if (dtype.Meta is VStringDTypeMeta)
                return Backends.Strings.NDVStringMove.StoreValue(value, dtype);
            // A Python int past the machine types has no marked 0-d array, so it is packed now (PyArray_Pack's range check),
            // the one value converted before NumPy's index checks.
            if (value is System.Numerics.BigInteger && !Backends.Strings.NDStringStorage.IsString(dtype))
                return NDArray.AsStoreValue(value, dtype.GetTypeCode());
            // Every other value keeps its kind until the array overload converts it, after the axis and index checks:
            // a C# number is a Python (or NumPy) scalar in the boundary map, which PyArray_Pack packs there (a Python int
            // range-checked, a float truncated then range-checked, a complex refused by a real dtype); a C# string into a
            // NUMERIC array is a Python str, which PyArray_Pack parses there with the dtype's setitem (its errors quote the
            // str '1.5', not np.str_('1.5')); a C# array or collection is cast there. The 0-d or array belongs to the
            // caller's scope.
            return NDArray.FromPythonOperand(value);
        }

        /// <summary>
        ///     Coerces <paramref name="values"/> to <paramref name="arr"/>'s dtype
        ///     and aligns its shape for the single-index path. Matches NumPy's
        ///     <c>values = array(values, copy=None, ndmin=arr.ndim, dtype=arr.dtype);
        ///     if scalar_obj: values = np.moveaxis(values, 0, axis); numnew = values.shape[axis]</c>
        ///     (<see cref="ConvertValuesForSingleIndex"/>). The final shape is broadcast to <c>arr.shape</c> with the axis
        ///     dim replaced by <c>numnew</c> (<see cref="BroadcastValuesToInsertSlot"/>, the rule of NumPy's
        ///     <c>new[..., index:index+numnew, ...] = values</c>), so the concatenate path can splice it in without further
        ///     shape gymnastics.
        /// </summary>
        /// <param name="arr">The working array whose dtype and shape the values adopt.</param>
        /// <param name="values">The values to insert.</param>
        /// <param name="axis">The normalized insertion axis.</param>
        /// <param name="scalarObj">True when NumPy's scalar-obj quirk applies (moveaxis of the values' first axis).</param>
        /// <returns>The values, cast and broadcast to the insertion slot; <paramref name="values"/> itself when it
        ///     already has the dtype and the slot's shape.</returns>
        /// <exception cref="ValueError">The values do not broadcast into the slot: NumPy's view-assignment text (see
        ///     <see cref="BroadcastValuesToInsertSlot"/>); text values that do not parse as the array's number type.</exception>
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
            var v = ConvertValuesForSingleIndex(arr, values, axis, scalarObj);

            // Broadcast to arr.shape with axis dim = v.shape[axis].
            long numNew = arr.ndim == 0 ? v.size : v.Shape.dimensions[axis];
            var broadcasted = BroadcastValuesToInsertSlot(arr, v, axis, numNew);
            return broadcasted;
        }

        /// <summary>
        ///     NumPy's first two statements of the single-index branch, without the assignment that broadcasts the result:
        ///     <c>values = array(values, copy=None, ndmin=arr.ndim, dtype=arr.dtype)</c>, then for a 0-d index
        ///     <c>values = np.moveaxis(values, 0, axis)</c>.
        /// </summary>
        /// <param name="arr">The working array whose dtype and rank the values adopt.</param>
        /// <param name="values">The values to insert; null is Python's None.</param>
        /// <param name="axis">The normalized insertion axis.</param>
        /// <param name="scalarObj">True when NumPy's scalar-obj quirk applies (moveaxis of the values' first axis).</param>
        /// <returns>The converted values with at least <c>arr.ndim</c> axes, the first one moved to <paramref name="axis"/>
        ///     for a scalar obj; <paramref name="values"/> itself when nothing changed.</returns>
        /// <exception cref="OverflowException">A C# number literal outside the array dtype's range.</exception>
        /// <exception cref="TypeError">None into an integer array (<c>int()</c>'s TypeError), or a complex literal into a
        ///     real one.</exception>
        /// <exception cref="ValueError">Text values that do not parse as the array's number type.</exception>
        /// <remarks>
        ///     Separate from the broadcast because NumPy can fail between the two: with a one-element FLOAT index the
        ///     conversion runs, and then the slice NumPy builds from the float refuses it ("slice indices must be integers
        ///     ...", <see cref="InsertNonIntegerObj"/>) before the values are assigned anywhere, so a value that does not
        ///     broadcast is never reported there. Covered by <c>[NDScopedCovered]</c>: its callers are covered members under
        ///     the <c>[NDScoped]</c> <c>insert</c> overloads, which release the conversions.
        /// </remarks>
        [NDScopedCovered]
        private static NDArray ConvertValuesForSingleIndex(NDArray arr, NDArray values, int axis, bool scalarObj)
        {
            // Cast dtype if needed — NumPy's array(values, dtype=arr.dtype): by ELEMENT type, so a string value of
            // another width is truncated / NUL-padded to arr's (insert('QQQQQQ') into <U4 stores 'QQQQ'). A C# literal
            // (NDArray.WasPythonLiteral) is NumPy's Python scalar, which array(..., dtype=) packs with PyArray_Pack's range
            // checks (np.insert(int8, 1, 1000) raises) after the index check; the packed 0-d is the scope's. A StringDType
            // target stores a literal by setitem and casts an ndarray (NDVStringMove.StoreValue). A null is Python's None,
            // array(None, dtype=arr.dtype): NaN / nan+nanj / False / 'None' / a StringDType NA object, or int(None)'s TypeError.
            NDArray v = values is null
                ? NDArray.NoneForStore(arr.dtype)
                : arr.Storage.IsVString
                    ? Backends.Strings.NDVStringMove.StoreValue(values, arr.dtype)
                    : NDArray.PackForStore(values, arr.typecode, pyArrayPackRoute: false);
            if (!v.HasSameElementType(arr))
            {
                v = v.astype(arr.dtype, copy: true);
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

            return v;
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
        /// <exception cref="ValueError"><paramref name="values"/> cannot broadcast to the slot: NumPy's assignment text,
        ///     <c>could not broadcast input array from shape (2,1) into shape (2,)</c> (the shape after the leading
        ///     extent-1 axes beyond the slot's rank are stripped).</exception>
        /// <remarks>Covered by <c>[NDScopedCovered]</c>: its only caller is the covered
        ///     <see cref="CoerceValuesForSingleIndex"/>, so the stride-0 broadcast view (and the stripped view of a value
        ///     with extra leading axes) is released by the <c>insert</c> overload's scope.</remarks>
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

            // NumPy writes the values with new[..., index:index+numnew, ...] = values, a view assignment
            // (PyArray_AssignArray): leading extent-1 axes beyond the slot's rank are stripped, then the values must
            // broadcast to the slot, else "could not broadcast input array from shape (2,1) into shape (2,)", the stripped
            // shape printed. np.broadcast_to alone refused every extra axis and raised its own remapped-shapes text.
            Shape.AssignmentBroadcastStrides(targetDims, values.Shape.dimensions, values.Shape.strides, "input array",
                stripLeadingUnitAxes: true);
            if (values.ndim > targetDims.Length)
                values = new NDArray(values.Storage.AliasWithoutLeadingAxes(targetDims.Length));

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
        /// <exception cref="ValueError"><paramref name="values"/> cannot broadcast to the indexing result: MapIter's text,
        ///     <c>shape mismatch: value array of shape (3,) could not be broadcast to indexing result of shape (2,)</c> (see
        ///     <see cref="NDArray.PrepareAssignedValue"/>); text values that do not parse as the array's number type.</exception>
        /// <exception cref="OverflowException">A C# number literal outside the array dtype's range.</exception>
        /// <exception cref="TypeError">None into an integer array (<c>int()</c>'s TypeError).</exception>
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
            // Cast dtype — by ELEMENT type (a string's class and width), NumPy's array(values, dtype=arr.dtype). A C#
            // literal (NDArray.WasPythonLiteral) is NumPy's Python scalar: NumPy assigns it through new[tuple(slobj)] =
            // values, whose fancy setitem packs it with PyArray_Pack's range checks; the packed 0-d is the scope's. A
            // StringDType target stores a literal by setitem and casts an ndarray (NDVStringMove.StoreValue). A null is
            // Python's None, converted into arr's descriptor by that same setitem (int(None)'s TypeError for integers).
            NDArray v = values is null
                ? NDArray.NoneForStore(arr.dtype)
                : arr.Storage.IsVString
                    ? Backends.Strings.NDVStringMove.StoreValue(values, arr.dtype)
                    : NDArray.PackForStore(values, arr.typecode, pyArrayPackRoute: false);
            if (!v.HasSameElementType(arr))
            {
                v = v.astype(arr.dtype, copy: true);
            }

            // Build target shape: arr.shape with axis dim = numNew.
            var targetDims = new long[arr.ndim];
            for (int i = 0; i < arr.ndim; i++) targetDims[i] = arr.Shape.dimensions[i];
            targetDims[axis] = numNew;
            var targetShape = Shape.Adopt(targetDims);

            // NumPy writes them with new[..., indices, ...] = values, a fancy assignment: MapIter drops a value's extra
            // leading axes (they must hold one element in total) and requires a right-aligned broadcast to the indexing
            // result, else "shape mismatch: value array of shape (3,) could not be broadcast to indexing result of shape
            // (2,)" — an empty index included, whose result (..., 0, ...) only a one-element value fits. np.broadcast_to
            // below raised its own remapped-shapes text and refused every extra axis.
            v = NDArray.PrepareAssignedValue(v, targetDims);

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
        /// <param name="obj">The slice (a zero step is read as 1 by <c>PythonSliceIndices</c>).</param>
        /// <param name="N">The length of the insertion axis.</param>
        /// <returns>The positions in the slice's order (descending for a negative step), possibly none.</returns>
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
        /// <param name="keys">The keys (not modified).</param>
        /// <returns>The permutation: <c>keys[result[0]] &lt;= keys[result[1]] &lt;= ...</c>, equal keys in their input
        ///     order (NumPy's <c>argsort(kind='mergesort')</c>).</returns>
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

        /// <summary>
        ///     The recursive merge step of <see cref="StableArgSort"/>: sorts the permutation entries <c>[lo, hi)</c> by their
        ///     keys, stably.
        /// </summary>
        /// <param name="arr">The permutation, sorted in place over <c>[lo, hi)</c>.</param>
        /// <param name="tmp">Scratch of the permutation's length (its <c>[lo, hi)</c> is overwritten).</param>
        /// <param name="keys">The keys the permutation's entries index.</param>
        /// <param name="lo">The first entry of the run.</param>
        /// <param name="hi">One past the last entry of the run.</param>
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

        /// <summary>
        ///     Widens a permutation to <c>long</c>, the element type of the index array <c>np.take</c> reorders the values
        ///     with.
        /// </summary>
        /// <param name="src">The permutation.</param>
        /// <returns>A fresh array of the same values.</returns>
        private static long[] LongArrayFromInt(int[] src)
        {
            var result = new long[src.Length];
            for (int i = 0; i < src.Length; i++) result[i] = src[i];
            return result;
        }
    }
}
