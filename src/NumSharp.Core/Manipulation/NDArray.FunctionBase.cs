using System;
using NumSharp.Backends;
// NDExprTypeRules: NEP 50's check of a Python int against an integer dtype (the multi-index branch's `indices < 0` add).
using NumSharp.Backends.Iteration;
// np.piecewise's bool conditions (NDArray<bool>).
using NumSharp.Generic;
// The statements call np's public functions unqualified in places, as they did inside np; importing np's statics keeps
// them verbatim. The insert / delete helpers they share (PrepareAxisContext, WithSourceOrder, SliceAlongAxis,
// DuplicateArray, ToInt64Scalar, ToInt64Vector, PythonSliceIndices) are this class's own members since P3's batch 7
// moved np.delete's rest here, and a class member is found before an imported static.
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
        // ---------------------------- np.flip ----------------------------

        /// <summary>
        ///     The reversing slice (::-1) placed on every flipped axis of <see cref="FlipAxes"/>' cold-path indexer.
        /// </summary>
        private static readonly Slice ReversedSlice = new Slice(null, null, -1);

        /// <summary>
        ///     Builds the flipped view: reversing an axis is stride negation plus moving the base
        ///     offset to the last element along that axis — O(ndim), no data movement, exactly the
        ///     view NumPy's m[..., ::-1, ...] basic indexing produces.
        /// </summary>
        /// <param name="m">Input array (any layout: contiguous, sliced, transposed, broadcast).</param>
        /// <param name="axes">Normalized (non-negative, unique) axes to reverse; null reverses all axes.</param>
        /// <returns>A new view over <paramref name="m"/>'s buffer (read-only when <paramref name="m"/> is); for a
        ///     0-d or empty <paramref name="m"/> the general slice machinery's result.</returns>
        /// <remarks>The axes are not validated: np.flip normalizes them (AxisError, repeated axis) before it calls
        ///     this, and an engine caller passes axes it computed.</remarks>
        internal static NDArray FlipAxes(NDArray m, int[] axes)
        {
            var shape = m.Shape;
            int ndim = m.ndim;

            // Cold path — 0-d (the empty indexer m[()] yields the scalar back) and size-0 arrays
            // (nothing to reverse) go through the general slice machinery.
            if (ndim == 0 || shape.size == 0)
            {
                var indexer = new Slice[ndim];
                for (int i = 0; i < ndim; i++)
                    indexer[i] = Slice.All;
                if (axes == null)
                    for (int i = 0; i < ndim; i++)
                        indexer[i] = ReversedSlice;
                else
                    for (int i = 0; i < axes.Length; i++)
                        indexer[axes[i]] = ReversedSlice;
                return m[indexer];
            }

            var srcDims = shape.dimensions;
            var srcStrides = shape.strides;

            var dims = new long[ndim];
            var strides = new long[ndim];
            for (int i = 0; i < ndim; i++)
            {
                dims[i] = srcDims[i];
                strides[i] = srcStrides[i];
            }

            // size > 0 guarantees every dim >= 1, so (dim - 1) never underflows the offset.
            long offset = shape.offset;
            if (axes == null)
            {
                for (int i = 0; i < ndim; i++)
                {
                    offset += srcStrides[i] * (srcDims[i] - 1);
                    strides[i] = -srcStrides[i];
                }
            }
            else
            {
                for (int i = 0; i < axes.Length; i++)
                {
                    int ax = axes[i];
                    offset += srcStrides[ax] * (srcDims[ax] - 1);
                    strides[ax] = -srcStrides[ax];
                }
            }

            long bufferSize = shape.bufferSize > 0 ? shape.bufferSize : shape.size;
            var flipped = new Shape(dims, strides, offset, bufferSize);

            // Alias() inherits writeability from the source, so flipping a read-only array
            // (broadcast / 'r' memmap) stays read-only — same pattern as Transpose.
            return new NDArray(m.Storage.Alias(flipped)) {TensorEngine = m.TensorEngine};
        }

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
        internal static NDArray insert(NDArray arr, long obj, NDArray values, int? axis)
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
        /// <remarks>Covered by <c>[NDScopedCovered]</c>, as <see cref="insert(NDArray, long, NDArray, int?)"/>.</remarks>
        [NDScopedCovered]
        internal static NDArray insert(NDArray arr, Slice obj, NDArray values, int? axis)
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
        /// <remarks>Covered by <c>[NDScopedCovered]</c>, as <see cref="insert(NDArray, long, NDArray, int?)"/>.</remarks>
        [NDScopedCovered]
        internal static NDArray insert(NDArray arr, int[] obj, NDArray values, int? axis)
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
        /// <remarks>Covered by <c>[NDScopedCovered]</c>, as <see cref="insert(NDArray, long, NDArray, int?)"/>.</remarks>
        [NDScopedCovered]
        internal static NDArray insert(NDArray arr, long[] obj, NDArray values, int? axis)
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
        /// <remarks>Covered by <c>[NDScopedCovered]</c>, as <see cref="insert(NDArray, long, NDArray, int?)"/>; the
        ///     boolean mask's flatnonzero temp is released by the same scope.</remarks>
        [NDScopedCovered]
        internal static NDArray insert(NDArray arr, NDArray obj, NDArray values, int? axis)
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
        ///     <see cref="insert(NDArray, NDArray, NDArray, int?)"/>, under the <c>[NDScoped]</c> np.insert overloads, so the
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
        internal static NDArray delete(NDArray arr, long obj, int? axis)
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
        internal static NDArray delete(NDArray arr, Slice obj, int? axis)
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
        /// <exception cref="IndexOutOfRangeException">A position outside <c>[-N, N)</c>: NumPy's IndexError, naming the
        ///     deletion axis for a one-element list and the keep mask's axis 0 for a longer one.</exception>
        /// <remarks>Covered by <c>[NDScopedCovered]</c> (see the number overload).</remarks>
        [NDScopedCovered]
        internal static NDArray delete(NDArray arr, int[] obj, int? axis)
        {
            var ctx = PrepareAxisContext(arr, axis);
            // NumPy's np.asarray(obj) of one integer is its single-value path (`obj.size == 1 and obj.dtype.kind in "ui"`):
            // the scalar deletion, whose bounds error names the deletion axis.
            if (obj.Length == 1)
                return WithSourceOrder(arr, DeleteSingleIndex(ctx.work, obj[0], ctx.axis));
            var longs = new long[obj.Length];
            for (int i = 0; i < obj.Length; i++) longs[i] = obj[i];
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
        /// <exception cref="IndexOutOfRangeException">A position outside <c>[-N, N)</c>: NumPy's IndexError, naming the
        ///     deletion axis for a one-element list and the keep mask's axis 0 for a longer one.</exception>
        /// <remarks>Covered by <c>[NDScopedCovered]</c> (see the number overload).</remarks>
        [NDScopedCovered]
        internal static NDArray delete(NDArray arr, long[] obj, int? axis)
        {
            var ctx = PrepareAxisContext(arr, axis);
            // One integer is NumPy's single-value path (see the int[] overload).
            if (obj.Length == 1)
                return WithSourceOrder(arr, DeleteSingleIndex(ctx.work, obj[0], ctx.axis));
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
        internal static NDArray delete(NDArray arr, bool[] obj, int? axis)
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
        internal static NDArray delete(NDArray arr, NDArray obj, int? axis)
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
        /// <exception cref="IndexOutOfRangeException">An index is outside <c>[-N, N)</c>: NumPy's text of its
        ///     <c>keep[obj,] = False</c>, the first such index as given, on the keep mask's axis 0 whatever
        ///     <paramref name="axis"/> is (<c>index 3 is out of bounds for axis 0 with size 3</c>).</exception>
        /// <remarks>Covered by <c>[NDScopedCovered]</c>: its callers are the <c>[NDScoped]</c> <c>delete</c>
        ///     overloads and the covered <see cref="DeleteSlice"/>, so the dense path's keep mask is released by the
        ///     caller's scope, exceptions included.</remarks>
        [NDScopedCovered]
        private static NDArray DeleteIndexArray(NDArray arr, long[] indices, int axis)
        {
            long N = arr.Shape.dimensions[axis];

            if (indices.Length == 0)
                return DuplicateArray(arr);

            // Normalize + bounds-check. NumPy checks these indices as the fancy index of `keep[obj,] = False`, a 1-D mask
            // of the deletion axis's length, so its error names that mask's axis 0, not the deletion axis.
            for (int k = 0; k < indices.Length; k++)
            {
                long idx = indices[k];
                if (idx < -N || idx >= N)
                    throw new IndexOutOfRangeException(
                        $"index {indices[k]} is out of bounds for axis 0 with size {N}");
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
        internal static NDArray extract(NDArray condition, NDArray arr)
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
        internal static NDArray asarray_chkfinite(NDArray a, DType dtype, char order)
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

        // ----------------------------------------------------------------------------------------------------------------
        // np.delete's rest (P3's batch 7): the helpers np.insert's and np.delete's statements above share, moved from
        // np.delete.cs.
        // ----------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     NumPy's <c>delete</c>/<c>insert</c> allocate their output in ONE of two layouts —
        ///     <c>arrorder = 'F' if arr.flags.fnc else 'C'</c> (<c>_function_base_impl.py</c>): F-contiguous for a
        ///     strictly F-contiguous source (probed: num 1286), C-contiguous for every other source, a transposed or
        ///     broadcast one included. The composition kernels here may build another layout (concatenate keeps its
        ///     inputs' axis order), so the result is copied into the NumPy layout when it is not already in it.
        ///     IDEMPOTENT (a result already in the layout passes through), so nested overload delegation is safe.
        /// </summary>
        /// <param name="arr">The source array (its flags decide the layout).</param>
        /// <param name="result">The composed result.</param>
        /// <returns><paramref name="result"/>, or a C / F copy of it.</returns>
        internal static NDArray WithSourceOrder(NDArray arr, NDArray result)
        {
            if (result.ndim <= 1)
                return result;
            bool wantF = arr.Shape.IsFContiguous && !arr.Shape.IsContiguous;
            if (wantF)
                return result.Shape.IsFContiguous ? result : asfortranarray(result);
            return result.Shape.IsContiguous ? result : np.ascontiguousarray(result);
        }

        /// <summary>
        ///     Resolves <c>axis=None</c> to "ravel + axis=0" and normalises a
        ///     non-null axis against <c>arr.ndim</c>. The returned <c>work</c>
        ///     array is either <c>arr</c> itself or a freshly-allocated ravel
        ///     (which the caller can treat as owning since insert/delete always
        ///     produces a new array).
        /// </summary>
        /// <param name="arr">The input of insert / delete.</param>
        /// <param name="axis">The caller's axis, or null for the flattened array.</param>
        /// <returns>The array to work on and the non-negative axis.</returns>
        /// <exception cref="AxisError">The axis is out of bounds (NumPy's <c>normalize_axis_index</c> text).</exception>
        /// <remarks>Internal for np.insert's and np.delete's statements (NDArray.FunctionBase.cs), which share it.</remarks>
        internal static (NDArray work, int axis) PrepareAxisContext(NDArray arr, int? axis)
        {
            if (axis is null)
            {
                NDArray work = arr.ndim == 1 ? arr : np.ravel(arr);
                return (work, 0);
            }

            int ax = axis.Value;
            if (ax < 0) ax += arr.ndim;
            // NumPy's normalize_axis_index raises AxisError (a subclass of the ArgumentOutOfRangeException this used to be).
            if (ax < 0 || ax >= arr.ndim)
                throw new AxisError(axis.Value, arr.ndim);
            return (arr, ax);
        }

        /// <summary>
        ///     Casts <paramref name="obj"/> to <see cref="long"/> with NumPy-style
        ///     bounds error on out-of-range. Used by the scalar-index fast paths.
        /// </summary>
        /// <param name="obj">An integer index array; its first element is read (a uint64 above long's range wraps).</param>
        /// <param name="fn">The NumPy function's name for the error text (<c>"insert"</c> / <c>"delete"</c>).</param>
        /// <returns>The index as a <see cref="long"/>.</returns>
        /// <exception cref="ArgumentException"><paramref name="obj"/> is not an integer array (the callers test that
        ///     first, so this is not reached from them).</exception>
        /// <remarks>Internal for np.insert's and np.delete's statements (NDArray.FunctionBase.cs), which share it.</remarks>
        internal static long ToInt64Scalar(NDArray obj, string fn)
        {
            // Accept any integer dtype (NumPy: kind in "ui"). Bool is handled by
            // the caller before this point.
            switch (obj.GetTypeCode)
            {
                case NPTypeCode.Byte:   return obj.GetByte(0);
                case NPTypeCode.SByte:  return obj.GetSByte(0);
                case NPTypeCode.Char:   return obj.GetChar(0);
                case NPTypeCode.Int16:  return obj.GetInt16(0);
                case NPTypeCode.UInt16: return obj.GetUInt16(0);
                case NPTypeCode.Int32:  return obj.GetInt32(0);
                case NPTypeCode.UInt32: return obj.GetUInt32(0);
                case NPTypeCode.Int64:  return obj.GetInt64(0);
                case NPTypeCode.UInt64: return (long)obj.GetUInt64(0);
                default:
                    throw new ArgumentException(
                        $"np.{fn}: obj must be int / slice / integer array / boolean mask, got {obj.dtype.type.Name}",
                        "obj");
            }
        }

        /// <summary>
        ///     Materialises <paramref name="obj"/> as a managed <c>long[]</c>.
        ///     Any integer dtype is accepted; size-0 input returns an empty array
        ///     so the caller's "no-op" branch kicks in.
        /// </summary>
        /// <param name="obj">A 0-d or 1-D integer index array (a uint64 above long's range wraps).</param>
        /// <param name="fn">The NumPy function's name for the error texts (<c>"insert"</c> / <c>"delete"</c>).</param>
        /// <returns>A fresh array of the indices in order.</returns>
        /// <exception cref="ArgumentException"><paramref name="obj"/> has more than one dimension or is not an integer
        ///     array (the callers test both first, so neither is reached from them).</exception>
        /// <remarks>Internal for np.insert's and np.delete's statements (NDArray.FunctionBase.cs), which share it.</remarks>
        internal static long[] ToInt64Vector(NDArray obj, string fn)
        {
            if (obj.ndim > 1)
                throw new ArgumentException(
                    $"np.{fn}: index array argument obj must be one dimensional or scalar",
                    "obj");

            long n = obj.size;
            var result = new long[n];

            switch (obj.GetTypeCode)
            {
                case NPTypeCode.Byte:
                    for (long i = 0; i < n; i++) result[i] = obj.GetByte((int)i);
                    break;
                case NPTypeCode.SByte:
                    for (long i = 0; i < n; i++) result[i] = obj.GetSByte((int)i);
                    break;
                case NPTypeCode.Char:
                    for (long i = 0; i < n; i++) result[i] = obj.GetChar((int)i);
                    break;
                case NPTypeCode.Int16:
                    for (long i = 0; i < n; i++) result[i] = obj.GetInt16((int)i);
                    break;
                case NPTypeCode.UInt16:
                    for (long i = 0; i < n; i++) result[i] = obj.GetUInt16((int)i);
                    break;
                case NPTypeCode.Int32:
                    for (long i = 0; i < n; i++) result[i] = obj.GetInt32((int)i);
                    break;
                case NPTypeCode.UInt32:
                    for (long i = 0; i < n; i++) result[i] = obj.GetUInt32((int)i);
                    break;
                case NPTypeCode.Int64:
                    for (long i = 0; i < n; i++) result[i] = obj.GetInt64((int)i);
                    break;
                case NPTypeCode.UInt64:
                    for (long i = 0; i < n; i++) result[i] = (long)obj.GetUInt64((int)i);
                    break;
                default:
                    throw new ArgumentException(
                        $"np.{fn}: obj must be int / slice / integer array / boolean mask, got {obj.dtype.type.Name}",
                        "obj");
            }

            return result;
        }

        /// <summary>
        ///     Allocates a fresh, owning copy of <paramref name="arr"/>. Used by
        ///     delete/insert no-op paths to honour NumPy's "delete does not occur
        ///     in-place" contract.
        /// </summary>
        internal static NDArray DuplicateArray(NDArray arr) => np.copy(arr);

        /// <summary>
        ///     Returns an axis-aligned slice view <c>arr[..., start:stop, ...]</c>
        ///     by building a <see cref="Slice"/> array. The result shares storage
        ///     with <paramref name="arr"/> when <c>arr.Shape.IsContiguous</c> /
        ///     stride-friendly; the caller is responsible for disposing the wrapper.
        /// </summary>
        internal static NDArray SliceAlongAxis(NDArray arr, int axis, long start, long stop)
        {
            int ndim = arr.ndim;
            var slices = new Slice[ndim];
            for (int i = 0; i < ndim; i++)
                slices[i] = i == axis ? new Slice(start, stop) : Slice.All;
            return arr[slices];
        }

        /// <summary>
        ///     Python-compatible <c>slice.indices(N)</c> implementation. Resolves
        ///     None/negative bounds to absolute positions and clamps to a legal
        ///     iteration range. Used by both <see cref="DeleteSlice"/> and
        ///     <see cref="NDArray.InsertExpandSliceObj"/>.
        /// </summary>
        internal static (long start, long stop, long step) PythonSliceIndices(Slice s, long N)
        {
            long step = s.Step == 0 ? 1 : s.Step;
            if (step == 0)
                throw new ArgumentException("slice step cannot be zero");

            long start, stop;
            long defaultStart = step > 0 ? 0 : N - 1;
            long defaultStop = step > 0 ? N : -1; // exclusive; -1 sentinel for backwards

            if (s.Start.HasValue)
            {
                start = s.Start.Value;
                if (start < 0) start += N;
                if (step > 0)
                {
                    if (start < 0) start = 0;
                    if (start > N) start = N;
                }
                else
                {
                    if (start < 0) start = -1;
                    if (start >= N) start = N - 1;
                }
            }
            else start = defaultStart;

            if (s.Stop.HasValue)
            {
                stop = s.Stop.Value;
                if (stop < 0) stop += N;
                if (step > 0)
                {
                    if (stop < 0) stop = 0;
                    if (stop > N) stop = N;
                }
                else
                {
                    if (stop < 0) stop = -1;
                    if (stop >= N) stop = N - 1;
                }
            }
            else stop = defaultStop;

            return (start, stop, step);
        }

        // ----------------------------------------------------------------------------------------------------------------
        // np.select (P3's batch 7): NumPy's statements after the facade's numpy.matrix region and null checks, with the
        // helpers that resolve its dtype and run its fused kernel. The kernel classes are named in full.
        // ----------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     NumPy's <c>numpy.select</c> (<c>lib/_function_base_impl.py</c>) after the facade's numpy.matrix region and null
        ///     checks: the length checks, the choices' and the default's NEP 50 result dtype (a C# number literal is a weak
        ///     Python scalar; text resolves by descriptor), the two broadcast groups (the conditions, then the choices with the
        ///     default), the bool check of each condition, then the default written everywhere and every choice copied in
        ///     under its condition from the last to the first, so the first true condition wins.
        /// </summary>
        /// <param name="condlist">The conditions (bool arrays), one per choice.</param>
        /// <param name="choicelist">The choices: arrays (strong) or boxed C# values (a number literal is weak).</param>
        /// <param name="default">The value where every condition is false; null is NumPy's <c>0</c>, a weak Python int.</param>
        /// <returns>A new C-contiguous array of the broadcast shape in the result dtype.</returns>
        /// <exception cref="ValueError">The two lists differ in length ("list of cases must be same length as list of
        ///     conditions"), or are empty ("select with an empty condition list is not possible").</exception>
        /// <exception cref="TypeError">A condition is not bool ("invalid entry {i} in condlist: should be boolean
        ///     ndarray"), or the choices and the default have no common dtype (NumPy's "Choicelist and default value do not
        ///     have a common dtype" text around the promotion error).</exception>
        /// <exception cref="IncorrectShapeException">A group does not broadcast, or the two groups against each other: NumPy's
        ///     np.broadcast_arrays text naming the two operands of the group that disagree (<c>shape mismatch: objects cannot
        ///     be broadcast to a single shape.  Mismatch is between arg 0 with shape (7,) and arg 1 with shape (3, 4).</c>), the
        ///     two groups being arguments 0 and 1 of the last broadcast.</exception>
        /// <exception cref="InvalidCastException">A choice whose dtype does not cast to the result dtype under
        ///     <c>same_kind</c> (np.copyto's TypeError text: a C# number literal is NumPy's int64 / float64 / complex128 array,
        ///     so an int choice beside a uint8 array raises <c>Cannot cast array data from dtype('int64') to dtype('uint8')
        ///     according to the rule 'same_kind'</c>, <c>Cannot cast scalar ...</c> when every choice and the default are
        ///     0-d).</exception>
        /// <exception cref="ArgumentNullException">A null choice (NumPy's object array; NumSharp has no object dtype).</exception>
        /// <remarks>
        ///     The fused single-pass IL kernel serves the contiguous, no-cast, full-size-array case
        ///     (<see cref="TrySelectFused"/>); every other shape takes NumPy's own composition through np.copyto. Scoped: the
        ///     arrays a C# number becomes and the broadcast choices handed to np.copyto are this call's and are released on
        ///     the way out.
        /// </remarks>
        [NDScoped]
        internal static NDArray select(NDArray[] condlist, object[] choicelist, object @default)
        {
            // NumPy's order: the length check precedes the empty check, so select([], [x])
            // is a length error while select([], []) is the empty-condition error.
            if (condlist.Length != choicelist.Length)
                throw new ValueError("list of cases must be same length as list of conditions");
            if (condlist.Length == 0)
                throw new ValueError("select with an empty condition list is not possible");

            int n = condlist.Length;

            // Classify each choice plus the default, resolving the NEP50 result dtype while
            // materialising every entry to an NDArray for the broadcast/copy below. The default
            // occupies the final slot [n] — NumPy appends it to choicelist before promoting so it
            // both fixes the dtype and joins the choice broadcast shape.
            var mats = new NDArray[n + 1];
            bool anyStrong = false;
            bool anyBeyondInt64 = false;
            bool anyString = false; // a fixed-width string choice or default: the dtype is resolved by DESCRIPTOR below
            int weakRank = 0; // 0 none, 1 int, 2 float, 3 complex (ordered by NEP50 dominance)
            NPTypeCode strong = NPTypeCode.Empty;

            for (int i = 0; i <= n; i++)
            {
                object item = i < n ? choicelist[i] : @default;
                NDArray mat;
                switch (item)
                {
                    case null:
                        // A null CHOICE is a caller bug (NumPy would build an object array, a dtype
                        // NumSharp lacks). A null DEFAULT is NumPy's default=0 — a weak python int.
                        if (i < n)
                            throw new ArgumentNullException($"choicelist[{i}]",
                                "choicelist entries must not be null.");
                        // np.broadcast_arrays makes np.asarray(0) of it: int64.
                        mat = Scalar(0L);
                        weakRank = Math.Max(weakRank, 1);
                        break;

                    case NDArray lit when lit.WasPythonLiteral:
                        // A C# number literal that reached select as its implicit 0-d array (an NDArray default, a literal
                        // inside an NDArray[] choicelist): NumPy keeps a Python int / float / complex as the weak scalar
                        // (`choice if type(choice) in (int, float, complex) else np.asarray(choice)`), exactly as the boxed
                        // literal below — an int8 choice beside the default 1000 stays int8.
                        // The dtype reads the literal; the copies read the array np.broadcast_arrays makes of the Python
                        // scalar (int64 whatever the C# width), as NumPy's np.copyto receives it.
                        mat = SelectPythonScalarArray(lit.Storage.GetAtIndex(0));
                        weakRank = Math.Max(weakRank, lit.typecode == NPTypeCode.Complex ? 3
                            : lit.typecode is NPTypeCode.Single or NPTypeCode.Double or NPTypeCode.Half ? 2 : 1);
                        if (lit.typecode == NPTypeCode.UInt64 && lit.GetAtIndex<ulong>(0) > long.MaxValue) anyBeyondInt64 = true;
                        break;

                    case NDArray a:
                        mat = a;
                        if (a.typecode == NPTypeCode.String)
                            anyString = true;
                        else if (!anyString)
                            strong = anyStrong ? NDExprTypeRules.PromoteStrong(strong, a.GetTypeCode) : a.GetTypeCode;
                        anyStrong = true;
                        break;

                    // Weak (NEP50 "python literal") scalars — the eight C# integer primitives, the two
                    // binary floats and Complex. These adopt the other operands' dtype instead of
                    // forcing promotion by their own width/value (int8 array + 1000 stays int8).
                    // Each becomes the array np.broadcast_arrays makes of the Python scalar (np.asarray: int64, uint64 past
                    // int64, float64, complex128), which is what np.copyto receives: a byte 5 beside a uint8 choice is an int64
                    // array there, refused under same_kind as in NumPy.
                    case sbyte or byte or short or ushort or int or uint or long or ulong:
                        mat = SelectPythonScalarArray(item);
                        weakRank = Math.Max(weakRank, 1);
                        if (item is ulong u && u > long.MaxValue) anyBeyondInt64 = true;
                        break;

                    case float or double:
                        mat = SelectPythonScalarArray(item);
                        weakRank = Math.Max(weakRank, 2);
                        break;

                    case System.Numerics.Complex:
                        mat = SelectPythonScalarArray(item);
                        weakRank = Math.Max(weakRank, 3);
                        break;

                    default:
                        // Strong by NumPy's rule `type(x) in (int, float, complex) else asarray(x)`:
                        // bool, char, Half, decimal, C# arrays and collections all take the asarray
                        // branch and contribute their concrete dtype.
                        mat = np.asanyarray(item);
                        if (mat.typecode == NPTypeCode.String)
                            anyString = true;
                        else if (!anyString)
                            strong = anyStrong ? NDExprTypeRules.PromoteStrong(strong, mat.GetTypeCode) : mat.GetTypeCode;
                        anyStrong = true;
                        break;
                }

                mats[i] = mat;
            }

            // Fixed-width strings: NumPy's two result_type calls over DESCRIPTORS (the widest choice's width; S + U → U),
            // each refusal wrapped in select's own TypeError text. A Python number default next to strings has no common
            // dtype — select(…, default=0) over string choices raises, as in NumPy.
            DType stringDescr = anyString ? ResolveSelectStringDescr(choicelist, @default, mats, n) : null;

            NPTypeCode dtype = anyString ? NPTypeCode.String : ResolveSelectDtype(anyStrong, strong, weakRank, anyBeyondInt64);

            // The conditions and the choices are broadcast in two SEPARATE groups (NumPy makes two
            // independent broadcast_arrays calls). We only need each GROUP'S shape here — to size the
            // result and to validate mutual compatibility exactly where NumPy's two calls do — because
            // copyto broadcasts each operand to the result shape itself below. Materialising the
            // broadcast VIEWS is therefore skipped: it would allocate needlessly AND, for a group of
            // one, route through broadcast_arrays's single-array path (Shape.Clean(), which resets a
            // non-contiguous operand's strides to contiguous and would read a transposed/strided/
            // reversed condition through the wrong elements). Each group's shape is NumPy's
            // _broadcast_shape, so a mismatch names the two operands of that group as NumPy does.
            Shape scShape = Shape.BroadcastShape(GroupShapes(condlist));   // validates conditions mutually
            Shape schShape = Shape.BroadcastShape(GroupShapes(mats));       // validates choices + default mutually

            // Conditions must be boolean — checked after the group broadcasts, reporting the index.
            for (int i = 0; i < n; i++)
                if (condlist[i].GetTypeCode != NPTypeCode.Boolean)
                    throw new TypeError($"invalid entry {i} in condlist: should be boolean ndarray");

            // Result shape: the condition shape when every choice is 0-d (NumPy skips the call), else
            // np.broadcast_arrays(condlist[0], choicelist[0]) - the two GROUP shapes are its arguments 0 and 1.
            long[] resultDims = schShape.NDim == 0 ? scShape.dimensions : Shape.BroadcastShape(scShape, schShape).dimensions;

            // Fused single-pass fast path for the contiguous, no-cast, full-size-array-choice case —
            // the one NumPy leaves memory-bound (its (n+1) masked copytos each re-read AND re-write the
            // whole result buffer). One IL kernel reads each condition and choice once and writes the
            // result once. Every other shape (broadcast / strided / cast / scalar-choice) declines the
            // gate and takes the composition below, which already outruns NumPy on those.
            if (!anyString && TrySelectFused(condlist, mats, n, dtype, resultDims, out var fused))
                return fused;

            var result = new NDArray(stringDescr ?? (DType)dtype, Shape.Adopt((long[])resultDims.Clone()), false);

            // Composition fallback (NumPy's own structure): burn the default across the whole result
            // (unsafe cast, matching np.full's copyto), then overlay each choice where its condition is
            // true — LAST choice first, so the FIRST matching condition ends up winning each position.
            // copyto broadcasts each operand (and mask) to the result shape, preserving its own strides,
            // so it correctly serves the broadcast/strided/cast/scalar-choice shapes the fused kernel
            // declines; every masked write rides NumSharp's SIMD masked-cast kernel.
            np.copyto(result, mats[n], casting: "unsafe");
            for (int i = n - 1; i >= 0; i--)
            {
                // NumPy copies the choice np.broadcast_arrays gave the choice group's shape: a 0-d choice in a group with
                // axes reaches np.copyto as an ARRAY, so a refused cast reads "Cannot cast array data", not "Cannot cast
                // scalar" (the values are the same either way).
                NDArray choice = mats[i].ndim == 0 && schShape.NDim > 0 ? np.broadcast_to(mats[i], schShape) : mats[i];
                np.copyto(result, choice, casting: "same_kind", @where: condlist[i]);
            }

            return result;
        }

        /// <summary>
        ///     NumPy's <c>np.asarray</c> of a Python number: the array <c>np.broadcast_arrays</c> makes of a weak choice or default
        ///     of np.select, and so what its np.copyto calls receive. A Python int is int64 (uint64 past int64, as NumPy discovers
        ///     it), a float float64 and a complex complex128, whatever the C# width of the literal that stands for it.
        /// </summary>
        /// <param name="value">A boxed C# integer primitive, <c>float</c>, <c>double</c> or
        ///     <see cref="System.Numerics.Complex"/> (the values the select loop classifies as weak).</param>
        /// <returns>A new 0-d array; created inside <see cref="select"/>'s scope, which releases it.</returns>
        /// <exception cref="ArgumentException"><paramref name="value"/> is none of those (a caller defect).</exception>
        private static NDArray SelectPythonScalarArray(object value) => value switch
        {
            ulong u when u > long.MaxValue => Scalar(u),
            sbyte or byte or short or ushort or int or uint or long or ulong => Scalar(Convert.ToInt64(value)),
            float or double => Scalar(Convert.ToDouble(value)),
            System.Numerics.Complex c => Scalar(c),
            _ => throw new ArgumentException($"{value?.GetType().Name ?? "null"} is not a Python number.", nameof(value)),
        };

        /// <summary>The shapes of a group of operands, in order, for <see cref="Shape.BroadcastShape"/>.</summary>
        /// <param name="arrays">The operands (none null).</param>
        /// <returns>A new array of their shapes.</returns>
        private static Shape[] GroupShapes(NDArray[] arrays)
        {
            var shapes = new Shape[arrays.Length];
            for (int i = 0; i < arrays.Length; i++)
                shapes[i] = arrays[i].Shape;
            return shapes;
        }

        /// <summary>
        ///     <c>np.select</c>'s result dtype when a string is involved — NumPy 2.4.2's ONE
        ///     <c>dtype = result_type(*choicelist, default)</c> over descriptors, with C# integer / float / complex literals
        ///     (boxed, or the 0-d array an implicit conversion marked <see cref="NDArray.WasPythonLiteral"/>) and the null
        ///     default (NumPy's <c>0</c>) taking part as NEP 50 weak Python scalars.
        /// </summary>
        /// <param name="choicelist">The caller's choices.</param>
        /// <param name="default">The caller's default (null means NumPy's <c>0</c>).</param>
        /// <param name="mats">The materialised choices and default (slot <paramref name="n"/>).</param>
        /// <param name="n">The number of choices.</param>
        /// <returns>The result descriptor.</returns>
        /// <exception cref="TypeError"><c>Choicelist and default value do not have a common dtype: …</c> — NumPy's text
        ///     around the promotion error, whose DType list names every choice and the default in order.</exception>
        /// <remarks>
        ///     NumPy 2.4.2 promotes the choices and the default in a single <c>result_type</c> call (NumPy 1.x made two, the
        ///     first wrapped in <c>Choicelist elements do not have a common dtype</c>; this followed that before). The single
        ///     call decides the error text too: a string choice beside a number choice reports <c>Choicelist and default value
        ///     …</c> with the default's <c>_PyLongDType</c> at the end of the list.
        /// </remarks>
        private static DType ResolveSelectStringDescr(object[] choicelist, object @default, NDArray[] mats, int n)
        {
            static object Operand(object item, NDArray mat)
                => item is sbyte or byte or short or ushort or int or uint or long or ulong or float or double or System.Numerics.Complex ? item
                    : item is NDArray nd ? nd.Nep50Operand
                    : mat;

            var operands = new object[n + 1];
            for (int i = 0; i < n; i++)
                operands[i] = Operand(choicelist[i], mats[i]);
            operands[n] = @default is null ? 0 : Operand(@default, mats[n]);
            try
            {
                return DTypePromotion.ResultType(operands);
            }
            catch (DTypePromotionError e)
            {
                throw new TypeError($"Choicelist and default value do not have a common dtype: {e.Message}");
            }
        }

        /// <summary>
        ///     The fused single-pass fast path for <see cref="np.select"/>. Fires only when every
        ///     condition and choice is C-contiguous at the full result shape and the result dtype
        ///     (no broadcast, no cast) — the case NumPy's copyto composition leaves memory-bound.
        ///     Delegates the whole select to one IL kernel
        ///     (<see cref="Backends.Kernels.DirectILKernelGenerator.GetSelectKernel"/>) that seeds each element with
        ///     the default then walks the conditions last-to-first, so the first true condition wins.
        ///     Returns false (and leaves <paramref name="result"/> null) for any other shape, dtype,
        ///     or layout, so the caller falls back to the composition.
        /// </summary>
        private static unsafe bool TrySelectFused(NDArray[] condlist, NDArray[] mats, int n, NPTypeCode dtype, long[] resultDims, out NDArray result)
        {
            result = null;

            if (!Backends.Kernels.DirectILKernelGenerator.Enabled)
                return false;
            if (!Backends.Kernels.DirectILKernelGenerator.SelectKernelSupportsDtype(dtype))
                return false;

            long resultSize = 1;
            for (int d = 0; d < resultDims.Length; d++)
                resultSize *= resultDims[d];
            if (resultSize == 0)
                return false; // empty result — the composition handles it trivially.

            // Conditions: boolean, C-contiguous, exactly the result shape (no broadcast stretch).
            for (int i = 0; i < n; i++)
            {
                var c = condlist[i];
                if (c.GetTypeCode != NPTypeCode.Boolean) return false;
                if (!c.Shape.IsContiguous) return false;
                if (!SelectDimsEqual(c.Shape, resultDims)) return false;
            }

            // Choices: already the result dtype (no cast), C-contiguous, exactly the result shape.
            for (int i = 0; i < n; i++)
            {
                var ch = mats[i];
                if (ch.GetTypeCode != dtype) return false;
                if (!ch.Shape.IsContiguous) return false;
                if (!SelectDimsEqual(ch.Shape, resultDims)) return false;
            }

            // Default: a scalar (broadcast, cast to dtype for one element) or a full-size, same-dtype,
            // C-contiguous array. Anything else declines.
            var def = mats[n];
            bool defScalar = def.size == 1;
            NDArray defArr = def;
            if (defScalar)
            {
                if (def.GetTypeCode != dtype)
                    defArr = def.astype(dtype);
            }
            else
            {
                if (def.GetTypeCode != dtype) return false;
                if (!def.Shape.IsContiguous) return false;
                if (!SelectDimsEqual(def.Shape, resultDims)) return false;
            }

            var kernel = Backends.Kernels.DirectILKernelGenerator.GetSelectKernel(dtype, n, defScalar);
            if (kernel == null)
                return false;

            var res = new NDArray(dtype, Shape.Adopt((long[])resultDims.Clone()), false);
            SelectFusedInvoke(kernel, condlist, mats, defArr, defScalar, res, n, dtype);
            result = res;
            return true;
        }

        /// <summary>True when <paramref name="shape"/>'s dimensions equal <paramref name="dims"/> exactly.</summary>
        private static bool SelectDimsEqual(Shape shape, long[] dims)
        {
            var sd = shape.dimensions;
            if (sd.Length != dims.Length) return false;
            for (int i = 0; i < dims.Length; i++)
                if (sd[i] != dims[i]) return false;
            return true;
        }

        /// <summary>
        ///     Gather the operand base pointers (base + offset, always the logical element 0 for a
        ///     C-contiguous operand) and run the fused kernel. The <c>stackalloc</c> pointer arrays
        ///     stay alive for the whole synchronous kernel call.
        /// </summary>
        private static unsafe void SelectFusedInvoke(Backends.Kernels.SelectFusedKernel kernel, NDArray[] condlist, NDArray[] mats, NDArray defArr, bool defScalar, NDArray res, int n, NPTypeCode dtype)
        {
            int elemSize = Utilities.InfoOf.GetSize(dtype);

            bool** conds = stackalloc bool*[n];
            void** choices = stackalloc void*[n];
            for (int i = 0; i < n; i++)
            {
                conds[i] = (bool*)condlist[i].Storage.Address + condlist[i].Shape.offset;
                choices[i] = (byte*)mats[i].Storage.Address + mats[i].Shape.offset * elemSize;
            }

            void* defPtr = (byte*)defArr.Storage.Address + defArr.Shape.offset * elemSize;
            void* resPtr = (byte*)res.Storage.Address + res.Shape.offset * elemSize;

            kernel(conds, choices, defPtr, resPtr, res.size);
        }

        /// <summary>
        ///     NumPy's <c>result_type(*choices, default)</c> folded with NEP50 weak-scalar rules —
        ///     the same resolution <see cref="AxisConcatenator"/> uses, minus the weak-bool kind
        ///     (select treats python bool as a strong asarray) and the weak-int range check (select
        ///     wraps an out-of-range default via <see cref="copyto"/> rather than raising).
        /// </summary>
        private static NPTypeCode ResolveSelectDtype(bool anyStrong, NPTypeCode strong, int weakRank, bool anyBeyondInt64)
        {
            if (!anyStrong)
            {
                // All weak literals — the NEP50 defaults. A weak int past int64 lifts the default to
                // uint64 (NumPy: result_type(2**63) and result_type(2**64-1) are both uint64).
                return weakRank switch
                {
                    3 => NPTypeCode.Complex,
                    2 => NPTypeCode.Double,
                    _ => anyBeyondInt64 ? NPTypeCode.UInt64 : NPTypeCode.Int64,
                };
            }

            return weakRank switch
            {
                // No literal.
                0 => strong,
                // A weak int adopts the strong dtype, except bool which it lifts to the default int.
                1 => strong == NPTypeCode.Boolean ? NPTypeCode.Int64 : strong,
                // A weak float keeps a float/decimal/complex width, else forces the default float.
                2 => NDExprTypeRules.IsFloatKind(strong) || strong == NPTypeCode.Decimal || strong == NPTypeCode.Complex
                    ? strong
                    : NPTypeCode.Double,
                // NumSharp has a single complex width, so complex64 collapses onto Complex.
                _ => NPTypeCode.Complex,
            };
        }

        // ----------------------------------------------------------------------------------------------------------------
        // np.piecewise (P3's batch 7): NumPy's statements after the facades' numpy.matrix rule and null checks, one member
        // per C# form of condlist, with the engine and its helpers. The kernel classes are named in full.
        // ----------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     NumPy's <c>numpy.piecewise</c> (<c>lib/_function_base_impl.py</c>) after the facade's numpy.matrix rule and null
        ///     checks, for a list of conditions: NumPy's <c>condlist[0]</c> inspection (an empty list is its IndexError),
        ///     then the shared engine (<see cref="PiecewiseCore"/>): the default condition when one extra function is given,
        ///     <c>zeros_like(x)</c>, and each condition's function or constant written forward, so the last true one wins.
        /// </summary>
        /// <param name="condlist">The conditions (a non-bool one reads by its truth), one per function.</param>
        /// <param name="x">The input domain; the result takes its shape and dtype (NumPy's <c>zeros_like(x)</c>).</param>
        /// <param name="funclist">One entry per condition, or one more (the default): callables
        ///     (<c>Func&lt;NDArray, NDArray&gt;</c>, <c>Func&lt;NDArray, object[], NDArray&gt;</c>) or constants.</param>
        /// <param name="args">Extra arguments for the <c>Func&lt;NDArray, object[], NDArray&gt;</c> entries.</param>
        /// <returns>A new array of <paramref name="x"/>'s shape and dtype.</returns>
        /// <exception cref="ValueError">The function count is neither the condition count nor one more.</exception>
        /// <exception cref="OverflowException">A Python int constant outside <paramref name="x"/>'s integer dtype ("Python
        ///     integer 1000 out of bounds for int8"), a float infinity into an integer dtype.</exception>
        /// <exception cref="TypeError">A complex constant into an integer or float <paramref name="x"/> (Python's
        ///     <c>int()</c> / <c>float()</c> texts), a None constant into an integer one.</exception>
        /// <exception cref="ValueError">A float NaN or a text that is not a number into a number dtype.</exception>
        /// <exception cref="IndexError">An empty condition list ("list index out of range").</exception>
        internal static NDArray piecewise(NDArray x, NDArray[] condlist, object[] funclist, object[] args)
        {
            // NumPy's promotion inspects condlist[0]; an empty list raises there, before the
            // function-count check ever runs.
            if (condlist.Length == 0)
                throw new IndexError("list index out of range");

            return PiecewiseCore(x, condlist, funclist, args);
        }

        /// <summary>
        ///     <c>numpy.piecewise</c> for a bare condition array (NumPy's promotion of a non-list condlist,
        ///     <see cref="PromoteBareCondlist"/>): a 2-D+ array is one condition per row, a 1-D one a single condition for a
        ///     non-0-d x, its elements for a 0-d x.
        /// </summary>
        /// <param name="condlist">The bare condition array (the caller's; never disposed here).</param>
        /// <param name="x">The input domain; the result takes its shape and dtype (NumPy's <c>zeros_like(x)</c>).</param>
        /// <param name="funclist">One entry per condition, or one more (the default): callables
        ///     (<c>Func&lt;NDArray, NDArray&gt;</c>, <c>Func&lt;NDArray, object[], NDArray&gt;</c>) or constants.</param>
        /// <param name="args">Extra arguments for the <c>Func&lt;NDArray, object[], NDArray&gt;</c> entries.</param>
        /// <returns>A new array of <paramref name="x"/>'s shape and dtype.</returns>
        /// <exception cref="ValueError">The function count is neither the condition count nor one more.</exception>
        /// <exception cref="OverflowException">A Python int constant outside <paramref name="x"/>'s integer dtype ("Python
        ///     integer 1000 out of bounds for int8"), a float infinity into an integer dtype.</exception>
        /// <exception cref="TypeError">A complex constant into an integer or float <paramref name="x"/> (Python's
        ///     <c>int()</c> / <c>float()</c> texts), a None constant into an integer one.</exception>
        /// <exception cref="ValueError">A float NaN or a text that is not a number into a number dtype.</exception>
        /// <exception cref="IndexError">A 0-d condlist ("too many indices for array") or an empty first axis ("index 0 is out
        ///     of bounds for axis 0 with size 0").</exception>
        /// <remarks>The row views the promotion indexes out of the caller's array are this call's own and are disposed on
        ///     the way out; the caller's array is not.</remarks>
        internal static NDArray piecewise(NDArray x, NDArray condlist, object[] funclist, object[] args)
        {
            NDArray[] promoted = PromoteBareCondlist(x, condlist);
            try
            {
                return PiecewiseCore(x, promoted, funclist, args);
            }
            finally
            {
                // PromoteBareCondlist hands back either the caller's own array (one whole condition) or
                // row views it indexed out of it. The views belong to this call: each holds a reference
                // on the caller's buffer, which would otherwise stay pinned until the finalizer ran.
                // The caller's own instance is never disposed here.
                foreach (var c in promoted)
                    if (!ReferenceEquals(c, condlist))
                        c.Dispose();
            }
        }

        /// <summary>
        ///     <c>numpy.piecewise</c> for a scalar condition (NumPy's <c>isscalar(condlist)</c> branch): the one function over
        ///     all of <paramref name="x"/> when it is true, nowhere when it is false.
        /// </summary>
        /// <param name="condlist">The condition.</param>
        /// <param name="x">The input domain; the result takes its shape and dtype (NumPy's <c>zeros_like(x)</c>).</param>
        /// <param name="funclist">One entry per condition, or one more (the default): callables
        ///     (<c>Func&lt;NDArray, NDArray&gt;</c>, <c>Func&lt;NDArray, object[], NDArray&gt;</c>) or constants.</param>
        /// <param name="args">Extra arguments for the <c>Func&lt;NDArray, object[], NDArray&gt;</c> entries.</param>
        /// <returns>A new array of <paramref name="x"/>'s shape and dtype.</returns>
        /// <exception cref="ValueError">The function count is neither the condition count nor one more.</exception>
        /// <exception cref="OverflowException">A Python int constant outside <paramref name="x"/>'s integer dtype ("Python
        ///     integer 1000 out of bounds for int8"), a float infinity into an integer dtype.</exception>
        /// <exception cref="TypeError">A complex constant into an integer or float <paramref name="x"/> (Python's
        ///     <c>int()</c> / <c>float()</c> texts), a None constant into an integer one.</exception>
        /// <exception cref="ValueError">A float NaN or a text that is not a number into a number dtype.</exception>
        internal static NDArray piecewise(NDArray x, bool condlist, object[] funclist, object[] args)
        {
            // isscalar(condlist) => condlist = [condlist] => a single 0-d bool condition (all / nothing).
            // The 0-d wrapper is this call's own temporary (the result never refers to it), so release it
            // on the way out instead of leaving its buffer to the finalizer.
            using var cond = Scalar<bool>(condlist);
            return PiecewiseCore(x, new[] { cond }, funclist, args);
        }

        /// <summary>
        ///     Reproduce NumPy's condlist promotion for a bare (non-list) ndarray:
        ///     <code>
        ///     if isscalar(condlist) or (not isinstance(condlist[0], (list, ndarray)) and x.ndim != 0):
        ///         condlist = [condlist]
        ///     </code>
        ///     For a bare ndarray, <c>condlist[0]</c> is a scalar iff ndim == 1 and an ndarray iff
        ///     ndim &gt;= 2; a 0-d array cannot be indexed at all. So: ndim == 0 raises; ndim == 1 with a
        ///     non-0-d x is ONE condition; otherwise (ndim &gt;= 2, or ndim == 1 with a 0-d x) axis 0 is
        ///     the condition axis and each row is a condition.
        /// </summary>
        private static NDArray[] PromoteBareCondlist(NDArray x, NDArray condlist)
        {
            if (condlist.ndim == 0)
                throw new IndexError("too many indices for array: array is 0-dimensional, but 1 were indexed");

            // A 1-D condition over a non-scalar x is the single whole-array condition (condlist[0] is a
            // scalar element and x.ndim != 0, so NumPy wraps it in a one-element list).
            if (condlist.ndim == 1 && x.ndim != 0)
                return new[] { condlist };

            // Otherwise axis 0 is the condition axis. NumPy's condlist[0] access raises on an empty
            // first axis before any wrap decision is made.
            long count = condlist.Shape.dimensions[0];
            if (count == 0)
                throw new IndexError("index 0 is out of bounds for axis 0 with size 0");

            var conds = new NDArray[count];
            for (int i = 0; i < count; i++)
                conds[i] = condlist[i];   // row i — a 0-d scalar view for a 1-D condlist, else a sub-array
            return conds;
        }

        /// <summary>
        ///     The shared piecewise engine: validate the function count, append the default condition if
        ///     one extra function was given, then <see cref="zeros_like"/> the output and overwrite it
        ///     condition-by-condition in forward order (so the last true condition wins).
        /// </summary>
        /// <param name="x">The input domain; the result takes its shape and dtype.</param>
        /// <param name="condlist">The conditions (non-empty). Read only — never disposed here; the caller owns them.</param>
        /// <param name="funclist">One function per condition, or one more (the default).</param>
        /// <param name="args">Extra arguments for <c>Func&lt;NDArray, object[], NDArray&gt;</c> entries.</param>
        /// <returns>A fresh array of <paramref name="x"/>'s shape and dtype, owned by the caller.</returns>
        /// <exception cref="ValueError">The function count is neither <c>condlist.Length</c> nor one more; or a constant
        ///     NumPy's store refuses (a float NaN, a text that is not a number, into a number dtype).</exception>
        /// <exception cref="OverflowException">A constant does not fit <paramref name="x"/>'s integer dtype.</exception>
        /// <exception cref="TypeError">A complex constant into an integer or float <paramref name="x"/>, a None constant into
        ///     an integer one.</exception>
        /// <exception cref="IndexError">A condition that does not fit <paramref name="x"/> (NumPy's boolean-index text),
        ///     raised before its constant is converted.</exception>
        /// <remarks>
        /// Every temporary this creates (the bool condition aliases, the default's "otherwise" mask, the
        /// packed scalar constants, and the result itself when a later step throws) is disposed before
        /// it returns, so a call leaves no buffer for the finalizer to reclaim.
        /// </remarks>
        private static NDArray PiecewiseCore(NDArray x, NDArray[] condlist, object[] funclist, object[] args)
        {
            int n = condlist.Length;    // >= 1 — every caller guarantees a non-empty condition list
            int n2 = funclist.Length;

            // The function count must equal the condition count, or exceed it by exactly one (the extra
            // function is the default). NumPy's message reports the condition count.
            bool hasDefault = n == n2 - 1;
            if (!hasDefault && n != n2)
                throw new ValueError(
                    $"with {n} condition(s), either {n} or {n + 1} functions are expected");

            // Ownership: every conds[i] and condElse is created HERE (a fresh MakeGeneric alias or a
            // converted copy, never the caller's instance), and the finally releases them all. Before,
            // they were dropped live, keeping a reference on their buffers until the finalizer ran;
            // the FuzzMatrix scope audit (Corpus_AllOps_LeaveNoUndisposedIntermediates) counted 1-3
            // escaped buffers per call. `y` is released only if we throw before handing it out. The
            // result of a user callable is NOT ours and is never disposed (see ApplyCallable).
            var conds = new NDArray<bool>[n];
            NDArray<bool> condElse = null;
            NDArray y = null;
            bool handedOut = false;
            try
            {
                // asarray(condlist, dtype=bool): comparison masks are already bool (a cheap alias); an
                // int/float condition converts by nonzero.
                for (int i = 0; i < n; i++)
                    conds[i] = ToBoolCondition(condlist[i]);

                // Fused single-pass fast path for the SCALAR-funclist contiguous case (the common signum
                // pattern). One IL kernel reads each condition once and writes the result once, avoiding the
                // zeros_like fill, the per-condition BooleanMaskSet (popcount + materialize + scatter), AND —
                // for the default case — the ~any(condlist) computation (the kernel seeds each element with
                // the default scalar and overlays the real conditions forward instead). Declines for
                // callables, NDArray funcs, non-contiguous x/conditions, or a non-SIMD dtype — those take
                // the composition below, which already outruns NumPy at scale.
                if (TryPiecewiseScalarFused(x, conds, funclist, hasDefault, out var fused))
                {
                    handedOut = true;
                    return fused;
                }

                // Composition (NumPy's own structure). One extra function => it is the default, evaluated
                // where NO condition is true: condelse = ~np.any(stack(condlist), axis=0), i.e.
                // ~(c0 | c1 | … | c_{n-1}).
                NDArray<bool>[] effConds;
                if (hasDefault)
                {
                    effConds = new NDArray<bool>[n + 1];
                    System.Array.Copy(conds, effConds, n);
                    effConds[n] = condElse = ComputeElseCondition(conds, n);
                }
                else
                {
                    effConds = conds;
                }

                // The output shares x's shape AND dtype (order 'K' matches x's layout), then starts at 0.
                y = np.zeros_like(x);

                for (int i = 0; i < effConds.Length; i++)
                {
                    NDArray<bool> cond = effConds[i];
                    object func = funclist[i];

                    switch (func)
                    {
                        // A callable f(x[cond]) — evaluated only over the selected values, exactly as NumPy
                        // does, and (matching its `vals.size > 0` guard) NOT called on an empty selection.
                        case Func<NDArray, NDArray> f1:
                            ApplyCallable(x, y, cond, f1, null, args);
                            break;
                        case Func<NDArray, object[], NDArray> f2:
                            ApplyCallable(x, y, cond, null, f2, args);
                            break;

                        // Anything else is a constant "function" (a scalar or an array), assigned into the
                        // selected slots — the `not callable(func)` branch of NumPy's loop.
                        default:
                            AssignConstant(y, cond, func);
                            break;
                    }
                }

                handedOut = true;
                return y;
            }
            finally
            {
                foreach (var c in conds)
                    c?.Dispose();
                condElse?.Dispose();
                // A validation error (e.g. an out-of-range weak scalar) or a throwing callable abandons
                // the half-written result; the caller never sees it, so it is ours to release.
                if (!handedOut)
                    y?.Dispose();
            }
        }

        /// <summary>
        ///     Evaluate a callable over the selected values and scatter the result back:
        ///     <c>vals = x[cond]; if vals.size &gt; 0: y[cond] = func(vals[, args])</c>. Reads
        ///     <paramref name="x"/> through its own strides in logical C-order (any layout) and writes
        ///     <paramref name="y"/> in the same C-order, so the k-th produced value lands at the k-th
        ///     selected coordinate. The result broadcasts / places into the selection and unsafe-casts to
        ///     <paramref name="y"/>'s dtype (a wrong-length result raises the NumPy boolean-assignment
        ///     ValueError). Exactly one of <paramref name="f1"/>/<paramref name="f2"/> is non-null.
        /// </summary>
        private static void ApplyCallable(NDArray x, NDArray y, NDArray<bool> cond,
            Func<NDArray, NDArray> f1, Func<NDArray, object[], NDArray> f2, object[] args)
        {
            NDArray vals = x[cond];
            if (vals.size == 0)
                return;   // NumPy skips calling func on an empty slice.

            NDArray result = f1 != null ? f1(vals) : f2(vals, args);
            y[cond] = result;
        }

        /// <summary>
        ///     Assign a constant "function" (the <c>not callable(func)</c> branch of NumPy's loop) into the selected slots:
        ///     NumPy's <c>y[cond] = func</c>. The boolean index is prepared first, so a condition that does not fit
        ///     <paramref name="y"/> raises its IndexError before the value is looked at; then the value is converted into
        ///     <paramref name="y"/>'s dtype by <c>PyArray_FromAny(func, y.dtype, FORCECAST)</c>.
        /// </summary>
        /// <param name="y">The result being filled (writeable, fresh).</param>
        /// <param name="cond">The condition's bool mask.</param>
        /// <param name="func">The constant: an <see cref="NDArray"/> (strong: broadcast or placed into the selection and
        ///     cast), or any other value, which goes in as the Python value it stands for (<see cref="FromPythonOperand"/>):
        ///     a C# number is a Python int / float / complex (a Python int range-checked, a float truncated and checked into
        ///     an integer dtype, a complex refused by <c>int()</c> / <c>float()</c> and stored by its truth into bool, by its
        ///     <c>str()</c> into text), a <c>bool</c> a Python bool, a string a Python str (parsed into a number dtype),
        ///     <see cref="Half"/> / <c>char</c> NumPy scalars (cast, as the FromAny route casts them: <c>np.float16(300)</c>
        ///     into int8 is 44), null Python's None (NaN, <c>nan+nanj</c>, False, <c>'None'</c>, or int()'s TypeError), a C#
        ///     array or collection a Python list (one value per selected slot, or one for all).</param>
        /// <exception cref="IndexError">The mask does not fit <paramref name="y"/>.</exception>
        /// <exception cref="OverflowException">A Python int outside an integer dtype, or a float infinity into one.</exception>
        /// <exception cref="TypeError">A complex value into an integer or float dtype, None into an integer one.</exception>
        /// <exception cref="ValueError">A float NaN or a non-number text into a number dtype, or a sequence whose length is
        ///     neither 1 nor the number of selected slots.</exception>
        /// <remarks>
        ///     NumSharp's mask setter (<see cref="SetMasked"/>) keeps NumPy's order: it checks the mask, then packs a marked
        ///     Python scalar with <c>PyArray_Pack</c>'s rules (<see cref="PackForStore(NDArray, NPTypeCode, bool)"/>), so the
        ///     constant only has to arrive as that mark. The value array <see cref="FromPythonOperand"/> makes is this call's
        ///     and is released once its value is stored.
        /// </remarks>
        private static void AssignConstant(NDArray y, NDArray<bool> cond, object func)
        {
            switch (func)
            {
                case NDArray arr:
                    // Strong: broadcast (size 1) or place (exact length) into the selection, unsafe-cast
                    // to y's dtype - exactly NumPy's `y[cond] = ndarray`.
                    y[cond] = arr;
                    return;

                default:
                    // A Python value (None included: the setter stores None's setitem value). FromPythonOperand mints a fresh
                    // array for every input but an NDArray (handled above), so disposing it never touches a caller's array.
                    using (var value = func is null ? null : FromPythonOperand(func))
                        y[cond] = value;
                    return;
            }
        }

        /// <summary>
        ///     The fused single-pass fast path for the SCALAR-funclist case. Fires only when every
        ///     function is a C# scalar constant, <paramref name="x"/> and every effective condition are
        ///     C-contiguous offset-0 at <paramref name="x"/>'s shape, and <paramref name="x"/>'s dtype is
        ///     SIMD-eligible (the 1/2/4/8-byte numerics). Packs the scalars into <paramref name="x"/>'s
        ///     dtype (the conversion the composition's store applies - a float truncates, an
        ///     out-of-range integer raises, a complex-into-int raises) and delegates to one IL kernel
        ///     (<see cref="Backends.Kernels.DirectILKernelGenerator.GetPiecewiseScalarKernel"/>) that seeds each element
        ///     with 0 and overlays each condition's scalar forward, so the last true condition wins.
        ///     Returns false (leaving <paramref name="result"/> null) for any callable / NDArray func,
        ///     non-contiguous or broadcast operand, or non-SIMD dtype, so the caller uses the composition.
        /// </summary>
        private static unsafe bool TryPiecewiseScalarFused(NDArray x, NDArray<bool>[] conds, object[] funclist, bool hasDefault, out NDArray result)
        {
            result = null;

            if (!Backends.Kernels.DirectILKernelGenerator.Enabled)
                return false;

            NPTypeCode dtype = x.typecode;
            if (!Backends.Kernels.DirectILKernelGenerator.PiecewiseScalarKernelSupportsDtype(dtype))
                return false;

            long size = x.size;
            if (size == 0)
                return false;   // empty — the composition handles it trivially.

            // The kernel writes a fresh C-contiguous result and reads each condition as a full element
            // mask, so x and every condition must be C-contiguous, offset 0, at exactly x's shape.
            if (!(x.Shape.IsContiguous && x.Shape.offset == 0))
                return false;

            int n = conds.Length;   // number of real conditions (the default, if any, is the seed)
            for (int k = 0; k < n; k++)
            {
                var c = conds[k];
                if (!(c.Shape.IsContiguous && c.Shape.offset == 0)) return false;
                if (!PiecewiseShapeEquals(c.Shape, x.Shape)) return false;
            }

            // Every function must be ONE C# number or a text (a Python str, parsed into x's number dtype). A
            // callable, an NDArray func, a null (Python's None) or a C# array / collection (a Python list, one value per
            // selected slot) declines to the composition, which stores each through the mask setter.
            for (int i = 0; i < funclist.Length; i++)
            {
                object f = funclist[i];
                if (!(IsStoreScalar(f) || f is string))
                    return false;
            }

            var kernel = Backends.Kernels.DirectILKernelGenerator.GetPiecewiseScalarKernel(dtype, n);
            if (kernel == null)
                return false;

            int elemSize = Utilities.InfoOf.GetSize(dtype);

            // Pack the n condition scalars PLUS the seed into a small STACK buffer of x's dtype (n+1
            // slots; funcVals[n] is the seed). The seed is the default function where one was given, else
            // 0 (piecewise's zeros_like default). This is the SAME conversion the composition's store
            // applies (NumPy's PyArray_FromAny of each constant, in funclist order, the default last) - an
            // out-of-range integer / complex-into-int raises here, before any kernel work - and it folds
            // the default in WITHOUT the ~any(condlist) pass. Every mask fits x here, so NumPy's
            // index-before-value order cannot show.
            byte* fvals = stackalloc byte[checked((n + 1) * elemSize)];
            for (int k = 0; k < n; k++)
                WriteCoercedScalar(fvals + k * elemSize, funclist[k], dtype, elemSize);
            object seed = hasDefault ? funclist[n] : (object)0;   // funclist[n] is the default
            WriteCoercedScalar(fvals + n * elemSize, seed, dtype, elemSize);

            var res = new NDArray(dtype, Shape.Adopt((long[])x.Shape.dimensions.Clone()), false);

            bool** condPtrs = stackalloc bool*[n];
            for (int k = 0; k < n; k++)
                condPtrs[k] = (bool*)conds[k].Storage.Address + conds[k].Shape.offset;
            void* resPtr = (byte*)res.Storage.Address + res.Shape.offset * elemSize;

            kernel(condPtrs, fvals, resPtr, size);

            GC.KeepAlive(conds);
            result = res;
            return true;
        }

        /// <summary>
        ///     Convert one constant into <paramref name="dtype"/> the way NumPy's <c>y[cond] = func</c> does
        ///     (<c>PyArray_FromAny(func, y.dtype, FORCECAST)</c>, <see cref="AsStoreValue"/>) and copy its bytes into
        ///     <paramref name="dst"/>. Shared by the fused kernel's condition-scalar and seed writes.
        /// </summary>
        /// <param name="dst">Destination for exactly one <paramref name="dtype"/> element.</param>
        /// <param name="value">One C# number or a text (the fused path admits nothing else).</param>
        /// <param name="dtype">The target dtype (x's dtype, a number dtype).</param>
        /// <param name="elemSize">The byte width of <paramref name="dtype"/>.</param>
        /// <exception cref="OverflowException">A Python int outside an integer dtype, or a float infinity into one.</exception>
        /// <exception cref="TypeError">A complex value into an integer or float dtype.</exception>
        /// <exception cref="ValueError">A float NaN or a non-number text into a number dtype.</exception>
        private static unsafe void WriteCoercedScalar(byte* dst, object value, NPTypeCode dtype, int elemSize)
        {
            // AsStoreValue packs a number or a text into a NEW 0-d array of exactly the destination dtype (a Python scalar by
            // PyArray_Pack's rules, a NumPy scalar - Half, char, decimal - by a cast); once its bytes are copied out nothing
            // refers to it.
            using var packed = AsStoreValue(value, dtype);
            System.Diagnostics.Debug.Assert(packed.typecode == dtype, "AsStoreValue packs a number into its destination dtype");
            Buffer.MemoryCopy((byte*)packed.Storage.Address + packed.Shape.offset * elemSize,
                dst, elemSize, elemSize);
        }

        /// <summary>True when two shapes have identical dimensions.</summary>
        private static bool PiecewiseShapeEquals(Shape a, Shape b)
        {
            var da = a.dimensions;
            var db = b.dimensions;
            if (da.Length != db.Length) return false;
            for (int i = 0; i < da.Length; i++)
                if (da[i] != db[i]) return false;
            return true;
        }

        /// <summary>Convert a condition to <see cref="NDArray{Boolean}"/> (nonzero for a non-bool array).</summary>
        /// <param name="cond">A caller-owned condition of any dtype; never disposed or mutated here.</param>
        /// <returns>
        /// A NEW object the caller owns and must dispose: a <see cref="NDArray.MakeGeneric{T}"/> alias
        /// (its own reference on <paramref name="cond"/>'s buffer) for a bool condition, or the sole
        /// owner of a fresh nonzero-converted buffer otherwise. Never <paramref name="cond"/> itself,
        /// so disposing the result can never free the caller's array.
        /// </returns>
        private static NDArray<bool> ToBoolCondition(NDArray cond)
        {
            if (cond.typecode == NPTypeCode.Boolean)
                return cond.MakeGeneric<bool>();

            // The conversion's own wrapper is released as soon as the alias exists: the alias holds its
            // own reference, so it becomes the buffer's only owner (previously the wrapper was dropped
            // live and pinned the buffer until the finalizer ran).
            using var converted = cond.astype(NPTypeCode.Boolean);
            return converted.MakeGeneric<bool>();
        }

        /// <summary>
        ///     The "otherwise" condition — true wherever none of the <paramref name="n"/> conditions is
        ///     (NumPy's <c>~np.any(condlist, axis=0)</c>, i.e. <c>np.logical_not(c0 | c1 | … | c_{n-1})</c>).
        ///     Uses <see cref="logical_not"/> (the ufunc) rather than the <c>!</c> operator: the ufunc is
        ///     layout-correct for every operand, whereas the operator mis-maps a non-C-contiguous input.
        /// </summary>
        /// <param name="conds">The bool conditions; read only (the caller owns and disposes them).</param>
        /// <param name="n">How many leading entries of <paramref name="conds"/> to fold (≥ 1).</param>
        /// <returns>A fresh bool mask the caller owns and must dispose.</returns>
        private static NDArray<bool> ComputeElseCondition(NDArray<bool>[] conds, int n)
        {
            // OR-fold. Each partial OR is a fresh array this method made, so it is released once the next
            // one exists. conds[0] seeds the fold but belongs to the caller, so it is never disposed here.
            NDArray anyTrue = conds[0];
            for (int i = 1; i < n; i++)
            {
                NDArray next = anyTrue | conds[i];
                if (!ReferenceEquals(anyTrue, conds[0]))
                    anyTrue.Dispose();
                anyTrue = next;
            }

            // The alias outlives the `using`: it holds its own reference, so disposing notTrue (and the
            // last partial OR) leaves the returned mask the sole owner of its buffer.
            using var notTrue = np.logical_not(anyTrue);
            if (!ReferenceEquals(anyTrue, conds[0]))
                anyTrue.Dispose();
            return notTrue.MakeGeneric<bool>();
        }
    }
}
