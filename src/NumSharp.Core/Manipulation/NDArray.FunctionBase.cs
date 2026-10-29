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

    // NDArray.FunctionBase.cs - the array object's functions of numpy/lib/_function_base_impl.py: insert and delete.
    //
    // np.insert and np.delete join and split an array by position and compute no value, so their statements are internal
    // members of the array object, not TensorEngine members (the np-layer dtype plan's section 1 "Which O", E7):
    //   * insert (NumPy's numpy.insert after the facade's class boundary): the axis context (axis=None ravels), the
    //     positions in every C# form (a number, a slice, a sequence, an index array or a boolean mask), the single-index
    //     and multi-index branches, the values' conversion into the array's dtype, and the refusals of an index array that
    //     is neither integer nor boolean;
    //   * delete (NumPy's numpy.delete after the facade's class boundaries and null checks): the axis context, the scalar,
    //     slice, index-array and boolean-mask paths and the source order of the result.
    //
    // History: the np-layer dtype plan's P2 moved insert's statements from np.insert into the internal operation class
    // InsertEngine; its phase PE (E1) made that class this NDArray partial and moved delete's statements here from
    // np.delete beside them. Both facades keep their signatures, documentation, class boundaries, null checks and one call
    // here. The file sits beside the facades, not under Backends/, because its statements call the public np.* functions
    // NumPy's call (np.concatenate, np.take, np.flatnonzero, np.moveaxis, np.broadcast_to, np.compress, ...), which ratchet
    // R2 counts under Backends/. Every member is internal or private: the array object's functions are not overridable.
    // In an NDArray partial a simple name resolves to NDArray's members before np's statics, so the moved code spells
    // System.Array in full (NDArray has an Array property); no other unqualified name of it is an NDArray member.
    public partial class NDArray
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
            long[] list => "[" + string.Join(", ", System.Array.ConvertAll(list,
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

        // ============================== delete ==============================

        /// <summary>
        ///     <c>delete(arr, obj, axis)</c> with one position given as a number, after np.delete's class boundaries and null
        ///     check: the axis context, then the scalar-index path in the source order.
        /// </summary>
        /// <param name="arr">Input array (not null).</param>
        /// <param name="obj">The index to remove; negative values count from the end.</param>
        /// <param name="axis">The axis to delete along; <c>null</c> deletes from the flattened array.</param>
        /// <returns>A fresh array without the sub-array at <paramref name="obj"/> (F-ordered when <paramref name="arr"/>
        ///     is).</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="IndexOutOfRangeException"><paramref name="obj"/> is outside <c>[-N, N)</c>.</exception>
        /// <remarks>Covered by <c>[NDScopedCovered]</c>: its only callers are the <c>[NDScoped]</c> np.delete overloads,
        ///     whose scope releases the ravel work array and the concatenated chunks.</remarks>
        [NDScopedCovered]
        internal static NDArray Delete(NDArray arr, long obj, int? axis)
        {
            var ctx = PrepareAxisContext(arr, axis);
            return WithSourceOrder(arr, DeleteSingleIndex(ctx.work, obj, ctx.axis));
        }

        /// <summary>
        ///     <c>delete(arr, obj, axis)</c> with a slice of positions, after np.delete's class boundaries and null checks:
        ///     the axis context, then the slice path in the source order.
        /// </summary>
        /// <param name="arr">Input array (not null).</param>
        /// <param name="obj">The positions to remove (not null).</param>
        /// <param name="axis">The axis to delete along; <c>null</c> deletes from the flattened array.</param>
        /// <returns>A fresh array without the sliced positions.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="ArgumentException">A slice step of zero.</exception>
        /// <remarks>Covered by <c>[NDScopedCovered]</c> (see the number overload).</remarks>
        [NDScopedCovered]
        internal static NDArray Delete(NDArray arr, Slice obj, int? axis)
        {
            var ctx = PrepareAxisContext(arr, axis);
            return WithSourceOrder(arr, DeleteSlice(ctx.work, obj, ctx.axis));
        }

        /// <summary>
        ///     <c>delete(arr, obj, axis)</c> with an <see cref="int"/> sequence of positions, after np.delete's class
        ///     boundaries and null checks: the positions widened to <see cref="long"/>, the axis context, then the
        ///     index-array path in the source order.
        /// </summary>
        /// <param name="arr">Input array (not null).</param>
        /// <param name="obj">The positions to remove (not null; negative values count from the end).</param>
        /// <param name="axis">The axis to delete along; <c>null</c> deletes from the flattened array.</param>
        /// <returns>A fresh array without the listed positions (duplicates remove once).</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="IndexOutOfRangeException">A position outside <c>[-N, N)</c>.</exception>
        /// <remarks>Covered by <c>[NDScopedCovered]</c> (see the number overload).</remarks>
        [NDScopedCovered]
        internal static NDArray Delete(NDArray arr, int[] obj, int? axis)
        {
            var longs = new long[obj.Length];
            for (int i = 0; i < obj.Length; i++) longs[i] = obj[i];
            var ctx = PrepareAxisContext(arr, axis);
            return WithSourceOrder(arr, DeleteIndexArray(ctx.work, longs, ctx.axis));
        }

        /// <summary>
        ///     <c>delete(arr, obj, axis)</c> with a <see cref="long"/> sequence of positions, after np.delete's class
        ///     boundaries and null checks: the axis context, then the index-array path (over a copy of the caller's array,
        ///     which the path sorts in place) in the source order.
        /// </summary>
        /// <param name="arr">Input array (not null).</param>
        /// <param name="obj">The positions to remove (not null; negative values count from the end).</param>
        /// <param name="axis">The axis to delete along; <c>null</c> deletes from the flattened array.</param>
        /// <returns>A fresh array without the listed positions (duplicates remove once).</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="IndexOutOfRangeException">A position outside <c>[-N, N)</c>.</exception>
        /// <remarks>Covered by <c>[NDScopedCovered]</c> (see the number overload).</remarks>
        [NDScopedCovered]
        internal static NDArray Delete(NDArray arr, long[] obj, int? axis)
        {
            var ctx = PrepareAxisContext(arr, axis);
            return WithSourceOrder(arr, DeleteIndexArray(ctx.work, (long[])obj.Clone(), ctx.axis));
        }

        /// <summary>
        ///     <c>delete(arr, obj, axis)</c> with a boolean mask, after np.delete's class boundaries and null checks: the
        ///     axis context, the mask's length check, then <c>np.compress</c> of the kept positions in the source order.
        /// </summary>
        /// <param name="arr">Input array (not null).</param>
        /// <param name="obj">The mask of positions to remove (not null), as long as the axis.</param>
        /// <param name="axis">The axis to delete along; <c>null</c> deletes from the flattened array.</param>
        /// <returns>A fresh array of the positions where <paramref name="obj"/> is false.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="ArgumentException">The mask's length differs from the axis length (NumPy's ValueError
        ///     text).</exception>
        /// <remarks>Covered by <c>[NDScopedCovered]</c>: the keep mask is released by the caller's scope, exceptions
        ///     included.</remarks>
        [NDScopedCovered]
        internal static NDArray Delete(NDArray arr, bool[] obj, int? axis)
        {
            var ctx = PrepareAxisContext(arr, axis);
            long N = ctx.work.Shape.dimensions[ctx.axis];
            if (obj.Length != N)
                throw new ArgumentException(
                    "boolean array argument obj to delete must be one dimensional " +
                    $"and match the axis length of {N}",
                    nameof(obj));

            // keep = ~obj; route through np.compress which has the fused gather kernel.
            var keep = new bool[obj.Length];
            for (int i = 0; i < obj.Length; i++) keep[i] = !obj[i];
            // keepArr is released by the caller's scope (exceptions included).
            var keepArr = np.array(keep);
            return WithSourceOrder(arr, np.compress(keepArr, ctx.work, ctx.axis));
        }

        /// <summary>
        ///     <c>delete(arr, obj, axis)</c> with an index array or a boolean mask array, after np.delete's class boundaries
        ///     and null checks: the axis context, the boolean branch, NumPy's refusal of any other non-integer index array,
        ///     then the integer positions (one element: the scalar-index path; more: the index-array path) in the source
        ///     order.
        /// </summary>
        /// <param name="arr">Input array (not null).</param>
        /// <param name="obj">The positions or mask (not null): an integer array of any rank (read in C order) or a 1-D
        ///     boolean mask as long as the axis.</param>
        /// <param name="axis">The axis to delete along; <c>null</c> deletes from the flattened array.</param>
        /// <returns>A fresh array without the given positions.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="ArgumentException">A boolean mask that is not 1-D or not as long as the axis.</exception>
        /// <exception cref="IndexError">An index array that is neither integer nor boolean (<c>arrays used as indices must
        ///     be of integer (or boolean) type</c>).</exception>
        /// <exception cref="IndexOutOfRangeException">A position outside <c>[-N, N)</c>.</exception>
        /// <remarks>Covered by <c>[NDScopedCovered]</c>: the ravel of a multi-dimensional index array and the keep mask
        ///     are released by the caller's scope.</remarks>
        [NDScopedCovered]
        internal static NDArray Delete(NDArray arr, NDArray obj, int? axis)
        {
            var ctx = PrepareAxisContext(arr, axis);

            if (obj.GetTypeCode == NPTypeCode.Boolean)
            {
                long N = ctx.work.Shape.dimensions[ctx.axis];
                if (obj.size != N || obj.ndim != 1)
                    throw new ArgumentException(
                        "boolean array argument obj to delete must be one dimensional " +
                        $"and match the axis length of {N}",
                        nameof(obj));
                // keep = ~obj. Build a contig bool keep-mask without leaning on
                // bitwise-not over a possibly-strided NDArray.
                bool[] keep = new bool[N];
                for (long i = 0; i < N; i++) keep[i] = !obj.GetBoolean((int)i);
                // keepArr is released by the caller's scope (exceptions included).
                var keepArr = np.array(keep);
                return WithSourceOrder(arr, np.compress(keepArr, ctx.work, ctx.axis));
            }

            // Not an integer index array: NumPy's `keep[obj,] = False` refuses it, whatever its size (text, floats, complex).
            if (!IsIntegerIndexDtype(obj.typecode))
                throw new IndexError("arrays used as indices must be of integer (or boolean) type");

            // Integer obj of any rank: its elements in C order (NumPy's fancy keep[obj,] reads every one).
            var flatObj = obj.ndim > 1 ? np.ravel(obj) : obj;

            // Integer obj. Size-1 ⇒ collapse to scalar path (NumPy parity).
            if (flatObj.size == 1)
            {
                long idx = ToInt64Scalar(flatObj, "delete");
                return WithSourceOrder(arr, DeleteSingleIndex(ctx.work, idx, ctx.axis));
            }

            // Materialise indices into a managed long[] for the multi-index path.
            long[] indices = ToInt64Vector(flatObj, "delete");
            return WithSourceOrder(arr, DeleteIndexArray(ctx.work, indices, ctx.axis));
        }

        /// <summary>
        ///     Scalar-index path: two concatenated views (pre + post) along
        ///     <paramref name="axis"/>. Bounds-checks first; -N…N-1 are valid.
        ///     Single-chunk fall-throughs route through <see cref="np.copy"/>
        ///     to produce an owning C-contig duplicate.
        /// </summary>
        /// <param name="arr">The working array (the input, or its ravel for <c>axis=None</c>).</param>
        /// <param name="obj">The index to remove; negative values count from the end.</param>
        /// <param name="axis">The normalized axis.</param>
        /// <returns>A fresh array without the sub-array at <paramref name="obj"/>.</returns>
        /// <exception cref="IndexOutOfRangeException"><paramref name="obj"/> is outside <c>[-N, N)</c>.</exception>
        /// <remarks>Covered by <c>[NDScopedCovered]</c>: its callers are the <c>[NDScoped]</c> <c>delete</c>
        ///     overloads, so the pre/post views are released by the caller's scope, exceptions included.</remarks>
        [NDScopedCovered]
        private static NDArray DeleteSingleIndex(NDArray arr, long obj, int axis)
        {
            long N = arr.Shape.dimensions[axis];
            if (obj < -N || obj >= N)
                throw new IndexOutOfRangeException(
                    $"index {obj} is out of bounds for axis {axis} with size {N}");
            if (obj < 0) obj += N;

            var pre = SliceAlongAxis(arr, axis, 0, obj);
            var post = SliceAlongAxis(arr, axis, obj + 1, N);

            if (pre.Shape.dimensions[axis] == 0) return np.copy(post);
            if (post.Shape.dimensions[axis] == 0) return np.copy(pre);
            return np.concatenate(new[] { pre, post }, axis);
        }

        /// <summary>
        ///     Slice-obj path. Mirrors NumPy's implementation: positive-step slice
        ///     concatenates pre + middle-kept + post; negative step is normalised
        ///     by inverting the slice direction. step!=1 falls back to the
        ///     keep-mask path which is identical to the array-obj path with the
        ///     slice expanded.
        /// </summary>
        /// <param name="arr">The working array (the input, or its ravel for <c>axis=None</c>).</param>
        /// <param name="obj">The slice of positions to remove, read as Python's <c>slice.indices(N)</c>.</param>
        /// <param name="axis">The normalized axis.</param>
        /// <returns>A fresh array without the sliced sub-arrays.</returns>
        /// <remarks>Covered by <c>[NDScopedCovered]</c>: its only caller is the <c>[NDScoped]</c> slice overload of
        ///     <c>delete</c>, so the pre/post views of the step-1 path are released by the caller's scope, exceptions
        ///     included.</remarks>
        [NDScopedCovered]
        private static NDArray DeleteSlice(NDArray arr, Slice obj, int axis)
        {
            long N = arr.Shape.dimensions[axis];
            var (start, stop, step) = PythonSliceIndices(obj, N);

            // Count elements that will actually be removed (matches NumPy's `xr`).
            long numToDelete;
            if (step > 0)
            {
                if (stop <= start) numToDelete = 0;
                else numToDelete = (stop - start + step - 1) / step;
            }
            else
            {
                if (stop >= start) numToDelete = 0;
                else
                {
                    long mag = -step;
                    numToDelete = (start - stop + mag - 1) / mag;
                }
            }

            if (numToDelete <= 0)
                return DuplicateArray(arr);

            // Normalise negative-step slices to forward iteration over the same
            // covered range (NumPy: start = xr[-1]; stop = xr[0] + 1; step = -step).
            if (step < 0)
            {
                long absStep = -step;
                long lastIncluded = start + (numToDelete - 1) * step; // smallest index removed
                long firstIncluded = start; // largest index removed
                start = lastIncluded;
                stop = firstIncluded + 1;
                step = absStep;
            }

            // step == 1 is a single contiguous block — pre + post concatenate.
            if (step == 1)
            {
                var pre = SliceAlongAxis(arr, axis, 0, start);
                var post = SliceAlongAxis(arr, axis, stop, N);
                if (pre.Shape.dimensions[axis] == 0) return np.copy(post);
                if (post.Shape.dimensions[axis] == 0) return np.copy(pre);
                return np.concatenate(new[] { pre, post }, axis);
            }

            // Strided slice: expand to indices array and route through the mask path.
            var indices = new long[numToDelete];
            for (long i = 0; i < numToDelete; i++)
                indices[i] = start + i * step;
            return DeleteIndexArray(arr, indices, axis);
        }

        /// <summary>
        ///     Multi-index path with two execution branches based on the ratio
        ///     of deletions to axis length:
        ///     <list type="bullet">
        ///       <item><b>Sparse</b> (numDelete &lt; <see cref="DeleteChunkConcatThreshold"/>):
        ///             sort + dedupe, then concat the kept chunks between deleted
        ///             positions. Skips the O(N) bool-mask scan that compress's
        ///             popcount+gather kernel needs. Wins ~2-3x on workloads like
        ///             "delete 5 of 1M".</item>
        ///       <item><b>Dense</b> (everything else): bool keep-mask + compress
        ///             — popcount + fused-gather kernel reads each axis position
        ///             exactly once, beats chunk-concat when chunks would be tiny.</item>
        ///     </list>
        /// </summary>
        /// <param name="arr">The working array (the input, or its ravel for <c>axis=None</c>).</param>
        /// <param name="indices">The positions to remove; normalized in place (negative values count from the end).
        ///     The caller passes an array it owns (a copy of the user's).</param>
        /// <param name="axis">The normalized axis.</param>
        /// <returns>A fresh array without the indexed sub-arrays.</returns>
        /// <exception cref="IndexOutOfRangeException">An index is outside <c>[-N, N)</c>.</exception>
        /// <remarks>Covered by <c>[NDScopedCovered]</c>: its callers are the <c>[NDScoped]</c> <c>delete</c>
        ///     overloads and the covered <see cref="DeleteSlice"/>, so the dense path's keep mask is released by the
        ///     caller's scope, exceptions included.</remarks>
        [NDScopedCovered]
        private static NDArray DeleteIndexArray(NDArray arr, long[] indices, int axis)
        {
            long N = arr.Shape.dimensions[axis];

            if (indices.Length == 0)
                return DuplicateArray(arr);

            // Normalize + bounds-check.
            for (int k = 0; k < indices.Length; k++)
            {
                long idx = indices[k];
                if (idx < -N || idx >= N)
                    throw new IndexOutOfRangeException(
                        $"index {indices[k]} is out of bounds for axis {axis} with size {N}");
                if (idx < 0) idx += N;
                indices[k] = idx;
            }

            // Sparse fast path: a small handful of deletions => chunk-concat is
            // far cheaper than scanning the whole bool-mask. The constant is
            // tuned against the popcount+gather kernel's per-byte throughput.
            if (indices.Length < DeleteChunkConcatThreshold)
                return DeleteChunkConcat(arr, indices, axis);

            // Dense path: compress over a bool keep mask.
            var keep = new bool[N];
            for (long i = 0; i < N; i++) keep[i] = true;
            for (int k = 0; k < indices.Length; k++) keep[indices[k]] = false;

            // keepArr is released by the caller's scope (exceptions included).
            var keepArr = np.array(keep);
            return np.compress(keepArr, arr, axis);
        }

        /// <summary>
        ///     Number of axis-deletions below which <see cref="DeleteIndexArray"/>
        ///     prefers chunk-concat over compress. 256 is the rough crossover
        ///     point on a 1M-element axis where compress's amortized cost catches
        ///     up with the chunk-concat's per-chunk overhead.
        /// </summary>
        private const int DeleteChunkConcatThreshold = 256;

        /// <summary>
        ///     Sparse-delete fast path. Sorts and dedupes the (already-normalized)
        ///     <paramref name="indices"/>, then concatenates the kept chunks
        ///     between deleted positions. Equivalent to the bool-mask + compress
        ///     path but skips the O(N) mask scan.
        /// </summary>
        /// <param name="arr">The working array.</param>
        /// <param name="indices">The normalized, in-bounds positions to remove; sorted and deduplicated in place.</param>
        /// <param name="axis">The normalized axis.</param>
        /// <returns>The concatenation of the kept chunks, or an empty array when every position is removed.</returns>
        /// <remarks>Its chunk views are released by the caller's scope; they share <paramref name="arr"/>'s buffer,
        ///     so holding them until that scope closes keeps no extra memory alive.</remarks>
        [NDScopedCovered] // only caller: DeleteIndexArray ← the [NDScoped] delete overloads
        private static NDArray DeleteChunkConcat(NDArray arr, long[] indices, int axis)
        {
            long N = arr.Shape.dimensions[axis];

            // Sort + dedupe in place. The chunks must be derived from monotonic
            // axis positions; duplicates count as a single deletion.
            System.Array.Sort(indices);
            int unique = 0;
            for (int k = 0; k < indices.Length; k++)
            {
                if (k == 0 || indices[k] != indices[k - 1])
                    indices[unique++] = indices[k];
            }

            // Build chunks: arr[prev:idx_0], arr[idx_0+1:idx_1], ..., arr[idx_{n-1}+1:N].
            // The first chunk uses prev=0; subsequent chunks use prev = idx_{k-1}+1.
            // Skip zero-length chunks so concatenate sees only real data. The chunk views are
            // released by the caller's scope (they share arr's buffer, so holding them until the
            // scope closes keeps no extra memory alive).
            var chunks = new System.Collections.Generic.List<NDArray>(unique + 1);
            long prev = 0;
            for (int k = 0; k < unique; k++)
            {
                long idx = indices[k];
                if (idx > prev)
                {
                    var chunk = SliceAlongAxis(arr, axis, prev, idx);
                    chunks.Add(chunk);
                }
                prev = idx + 1;
            }
            if (prev < N)
            {
                var tail = SliceAlongAxis(arr, axis, prev, N);
                chunks.Add(tail);
            }

            if (chunks.Count == 0)
            {
                // Every axis position was deleted ⇒ allocate an empty
                // result with axis dim 0 (NumPy parity).
                var emptyDims = new long[arr.ndim];
                for (int d = 0; d < arr.ndim; d++) emptyDims[d] = arr.Shape.dimensions[d];
                emptyDims[axis] = 0;
                return new NDArray(arr.dtype, Shape.Adopt(emptyDims), false);
            }

            return np.concatenate(chunks.ToArray(), axis);
        }

        // ----------------------------------------------------------------------------------------------------------------
        // np.extract and np.asarray_chkfinite (P3's batch 7): NumPy's statements after the facades' class boundaries and
        // null checks. The kernel namespace is named in full.
        // ----------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     NumPy's <c>numpy.extract</c> (<c>lib/_function_base_impl.py</c>) after the facade's class boundaries and null
        ///     checks: <c>take(ravel(arr), nonzero(ravel(condition))[0])</c>, the elements of <paramref name="arr"/> whose
        ///     flat position holds a truthy <paramref name="condition"/> element.
        /// </summary>
        /// <param name="condition">The mask, any dtype (NumPy's nonzero truth) and any shape (raveled).</param>
        /// <param name="arr">The source, any shape (raveled).</param>
        /// <returns>A new 1-D array of <paramref name="arr"/>'s dtype.</returns>
        /// <exception cref="IndexOutOfRangeException">A truthy <paramref name="condition"/> element lies past the end of
        ///     <paramref name="arr"/> (NumPy's IndexError from take's raise mode).</exception>
        /// <remarks>
        ///     Two routes: the fused IL kernel for a contiguous bool mask over a contiguous source of fixed-size elements
        ///     (<see cref="TryExtractFast"/>), NumPy's own chain (flatnonzero, then take) for everything else.
        /// </remarks>
        [NDScoped] // reclaims the ravel view/copy handed to take (and the flatnonzero indices on the throw path)
        internal static NDArray Extract(NDArray condition, NDArray arr)
        {
            // Fast path: bool cond + contig source/cond + cond.size <= arr.size.
            // The kernel walks the mask and gathers from src in one pass with no
            // intermediate indices materialisation.
            if (TryExtractFast(condition, arr, out var fast))
                return fast;

            // Generic path: flatnonzero handles non-bool / 0-d / strided cond
            // correctly (includes ravel internally); take handles non-contig arr
            // and OOB-True via its RAISE mode.
            var indices = np.flatnonzero(condition);
            try
            {
                var flatArr = np.ravel(arr);
                return np.take(flatArr, indices);
            }
            finally
            {
                indices.Dispose();
            }
        }

        /// <summary>
        ///     Fused fast path for <c>np.extract</c>. Skips index materialisation
        ///     and runs a single IL kernel (popcount + SIMD bit-scan + cpblk) on
        ///     the bool mask + contig source. Returns <c>false</c> when the
        ///     preconditions aren't met so the caller falls back to the generic
        ///     path (flatnonzero → take).
        /// </summary>
        /// <param name="condition">The mask; only a C-contiguous bool array qualifies.</param>
        /// <param name="arr">The source; only a C-contiguous array of fixed-size elements qualifies (a StringDType source
        ///     holds pointers and takes the generic path).</param>
        /// <param name="result">The extracted elements when the fast path ran, otherwise <c>null</c>.</param>
        /// <returns>Whether the fast path ran.</returns>
        /// <exception cref="IndexOutOfRangeException">The mask is longer than <paramref name="arr"/> and holds a true
        ///     element past its end (NumPy's IndexError text from take).</exception>
        /// <exception cref="InvalidOperationException">The gather wrote another count than the popcount (a kernel defect).</exception>
        private static unsafe bool TryExtractFast(NDArray condition, NDArray arr, out NDArray result)
        {
            result = null;
            // A source whose elements hold references (StringDType's cells; NumPy's PyDataType_REFCHK) takes the generic
            // path (ravel + np.take, whose StringDType gather duplicates the cells); this kernel copies elements as raw
            // bytes. The descriptor answers, as the item-selection family's one HasReferences decision does.
            if (arr.dtype.Meta.HasReferences) return false;
            if (condition.GetTypeCode != NPTypeCode.Boolean) return false;

            // Materialise to a contig 1-D view of the bool mask and source. The
            // raveled views share storage with their parents when contig; otherwise
            // they allocate a fresh contig copy (which we Dispose at the end).
            // For non-contig parents the generic path is already correct and not
            // markedly slower than copying, so we skip the fast path there.
            if (!condition.Shape.IsContiguous) return false;
            if (!arr.Shape.IsContiguous) return false;

            long maskSize = condition.size;
            long arrSize = arr.size;

            // OOB-True detection: if cond is longer than arr, any True at index
            // >= arr.size triggers an out-of-range read. We mirror NumPy's
            // IndexError. Cheap check: popcount the tail. If it's >0, raise.
            long effectiveScan = Math.Min(maskSize, arrSize);

            var countKernel = Backends.Kernels.DirectILKernelGenerator.GetArgwhereCountKernel(typeof(bool));
            var filterKernel = Backends.Kernels.DirectILKernelGenerator.GetFilterAxisKernel(arr.dtypesize);
            if (countKernel == null || filterKernel == null) return false;

            byte* maskPtr = (byte*)condition.Storage.Address + condition.Shape.offset;

            if (maskSize > arrSize)
            {
                long tailTrues = countKernel(maskPtr + arrSize, maskSize - arrSize);
                if (tailTrues > 0)
                    throw new IndexOutOfRangeException(
                        $"index {arrSize} is out of bounds for axis 0 with size {arrSize}");
            }

            // Pass 1: popcount mask over the effective scan range.
            long count = countKernel(maskPtr, effectiveScan);

            // Allocate result: shape (count,) with arr's dtype, C-contig.
            result = new NDArray(arr.dtype, new Shape(count), false); // descriptor: a string's width

            if (count == 0)
                return true;

            // Pass 2: fused gather.
            byte* srcPtr = (byte*)arr.Storage.Address + arr.Shape.offset * arr.dtypesize;
            byte* dstPtr = (byte*)result.Storage.Address;
            long written = filterKernel(
                srcPtr, maskPtr, effectiveScan,
                outerSize: 1, srcOuterStride: 0, dstOuterStride: 0,
                innerSize: arr.dtypesize,
                dstPtr);

            // Sanity: the popcount and the kernel walk the same mask, so written
            // must equal count. A mismatch indicates kernel-corruption.
            if (written != count)
                throw new InvalidOperationException(
                    $"extract: filter kernel wrote {written}, expected {count}");

            return true;
        }

        /// <summary>
        ///     NumPy's <c>numpy.asarray_chkfinite</c> (<c>lib/_function_base_impl.py</c>) after the facade's class
        ///     boundaries and null check: <c>a = asarray(a, dtype=dtype, order=order)</c>, then a ValueError when a
        ///     floating or complex result holds an infinity or a NaN.
        /// </summary>
        /// <param name="a">The array.</param>
        /// <param name="dtype">The requested dtype, or <c>null</c> to keep <paramref name="a"/>'s.</param>
        /// <param name="order">'C', 'F', 'A' or 'K', as np.asarray takes it.</param>
        /// <returns>np.asarray's result: <paramref name="a"/> itself when it already has the dtype and layout asked for,
        ///     otherwise a converted copy.</returns>
        /// <exception cref="ValueError">The result holds an infinity or a NaN ("array must not contain infs or NaNs").</exception>
        /// <remarks>
        ///     The finiteness scan is one fused SIMD pass (<c>FiniteScan.IsAllFinite</c>) and runs only for NumPy's
        ///     <c>typecodes['AllFloat']</c> (<see cref="IsFloatFamily"/>): integers, bool, char and decimal cannot hold a
        ///     non-finite value. np.asarray's own errors (a dtype it cannot cast to, an order letter) come first.
        /// </remarks>
        internal static NDArray AsarrayChkfinite(NDArray a, DType dtype, char order)
        {
            var arr = np.asarray(a, dtype, order);

            if (IsFloatFamily(arr.typecode) && !Backends.Kernels.FiniteScan.IsAllFinite(arr))
                throw new ValueError("array must not contain infs or NaNs");

            return arr;
        }

        /// <summary>
        ///     True for the dtypes NumPy checks for finiteness (its <c>typecodes['AllFloat']</c>: the real
        ///     and complex floating kinds). Decimal is excluded — it has no NumPy analog and, being exact,
        ///     can never represent inf/NaN.
        /// </summary>
        /// <param name="tc">The array's storage lane.</param>
        /// <returns>True for float16, float32, float64 and complex128.</returns>
        private static bool IsFloatFamily(NPTypeCode tc)
            => tc == NPTypeCode.Half || tc == NPTypeCode.Single || tc == NPTypeCode.Double || tc == NPTypeCode.Complex;
    }
}
