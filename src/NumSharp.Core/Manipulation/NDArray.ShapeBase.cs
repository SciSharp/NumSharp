using System;
using System.Collections.Generic;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

namespace NumSharp
{
    // ============================== lib/_shape_base_impl.py ==============================
    // NumPy's take_along_axis, put_along_axis and apply_along_axis as members of the array object (P3's batch 7):
    // TakeAlongAxis, PutAlongAxis with its scatter core, and ApplyAlongAxis. The facades keep their numpy.matrix /
    // numpy.char rules and their null checks; np.apply_along_axis's *args overload wraps its delegate and calls the
    // one-argument facade.
    public partial class NDArray
    {
        // ---------------------------- np.expand_dims ----------------------------

        /// <summary>
        ///     NumPy's <c>numpy.expand_dims(a, axis)</c> for one axis after the facade's numpy.char.chararray rule and
        ///     matrix region: a VIEW of <paramref name="a"/> with a length-1 axis inserted at <paramref name="axis"/>,
        ///     the axis normalized against the OUTPUT's dimensions (<c>a.ndim + 1</c>) and reported as given when it is
        ///     out of range.
        /// </summary>
        /// <param name="a">The array (not null).</param>
        /// <param name="axis">Where the new axis goes in the result: <c>-(a.ndim + 1) .. a.ndim</c>.</param>
        /// <returns>A new view over <paramref name="a"/>'s buffer (it keeps the buffer alive and shares its
        ///     writeability), or <paramref name="a"/> itself for the uninitialized-shape sentinel (a default
        ///     <see cref="Shape"/>), which NumPy has no analog of.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is outside the output's dimensions (NumPy's text,
        ///     <c>axis 5 is out of bounds for array of dimension 2</c>, naming the axis as given).</exception>
        /// <remarks>No class boundary runs here: an engine caller sits inside its public function's boundary, and the
        ///     facade applies its rules before it calls this.</remarks>
        internal static NDArray ExpandDims(NDArray a, int axis)
        {
            // Only the uninitialized-shape sentinel is passed through. A genuine empty array
            // (size == 0 but with real dimensions, e.g. (0, 3)) MUST still gain the new axis —
            // NumPy: expand_dims((0, 3), 0) -> (1, 0, 3). Guarding on a.size == 0 here made
            // expand_dims a no-op for every empty array (and, transitively, broke np.stack and
            // np.atleast_2d / np.vstack, which build on it).
            if (a.Shape.IsEmpty)
                return a;

            // NumPy normalizes the axis against the OUTPUT ndim (a.ndim + 1) and reports the axis
            // AS GIVEN. Shape.ExpandDimension guards only the negative side, and does it against
            // the already-normalized value, so all three of these were wrong:
            //   expand_dims(arange(6), 5)  -> ArgumentOutOfRangeException "... (Parameter 'index')"
            //                                 leaking out of Arrays.Insert, three frames down
            //   expand_dims(arange(6), -3) -> "Effective axis -1 is less than 0" (the NORMALIZED
            //                                 axis, not the -3 the caller passed)
            //   expand_dims(scalar, 5)     -> silently returned shape (1); the 0-d branch never
            //                                 looked at the axis at all
            // NumPy answers all three with: axis N is out of bounds for array of dimension {outNdim}.
            //
            // The guard lives here rather than in Shape.ExpandDimension because that method has
            // internal callers (keepdims reductions, fancy-index shaping, asmatrix) that are not
            // bound by NumPy's axis contract. The sibling int[] overload already validates this way,
            // and this reproduces its text exactly so both overloads agree.
            // normalize_axis_tuple((axis,), out_ndim) raises NumPy's AxisError; the range test runs on the axis as given.
            int outNdim = a.ndim + 1;
            Shape.check_and_adjust_axis(outNdim, axis);

            return new NDArray(a.Storage.Alias(a.Shape.ExpandDimension(axis))) { TensorEngine = a.TensorEngine };
        }

        // ---------------------------- np.take_along_axis ----------------------------

        /// <summary>
        ///     NumPy's <c>numpy.take_along_axis</c> after the facade's numpy.matrix / numpy.char rules and null checks: a
        ///     <c>StringDType</c> source through its cells' gather, then the axis (<c>axis=None</c> flattens the source in C
        ///     order), the index dtype and rank checks, the result shape (the axis from the indices, every other dimension
        ///     broadcast) and the strided gather kernel.
        /// </summary>
        /// <param name="arr">The source (not null).</param>
        /// <param name="indices">The integer index array (not null).</param>
        /// <param name="axis">The axis to take along, or null for the source flattened in C order.</param>
        /// <returns>A new C-contiguous array of <paramref name="arr"/>'s dtype, shaped <c>(Ni..., J, Nk...)</c>.</returns>
        /// <exception cref="ValueError"><c>axis=None</c> with indices that are not 1-D; indices of another rank than
        ///     <paramref name="arr"/>.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> out of range (the original axis reported).</exception>
        /// <exception cref="IndexError">A non-integer index array, a non-axis dimension that does not broadcast (NumPy's
        ///     fancy-index text), or an index out of bounds for the axis.</exception>
        /// <exception cref="NotSupportedException">The IL kernel is unavailable.</exception>
        /// <remarks>Scoped: the <c>axis=None</c> re-lay, its reshape view and the int64 index cast are temporaries of this
        ///     call.</remarks>
        [NDScoped]
        internal static unsafe NDArray TakeAlongAxis(NDArray arr, NDArray indices, int? axis)
        {
            // A StringDType source: the same gather runs on its cell view (NumPy's checks, this array's shapes) and the cells
            // are duplicated (NDVStringMove) — never the raw byte kernel below. The descriptor answers which sources those
            // are: the ones whose elements hold references (NumPy's PyDataType_REFCHK).
            if (arr.dtype.Meta.HasReferences)
                return Backends.Strings.NDVStringMove.Gather(arr, cells => np.take_along_axis(cells, indices, axis));

            // Boundary scope: reclaims the axis=None contiguous re-lay + its reshape view and the
            // int64 index cast at exit. The scope's tracked list also strong-roots those temps
            // through the raw-pointer kernel call below (the ARC/JIT dead-local hazard NonZero
            // documents cannot arise under a scope).

            // ── normalize inputs (NumPy _shape_base_impl.take_along_axis) ─────────────
            int ax;
            if (axis is null)
            {
                if (indices.ndim != 1)
                    throw new ValueError("when axis=None, `indices` must have a single dimension.");
                // arr = np.array(arr.flat): a C-order 1-D image of the source. A non-C-contiguous
                // view must be re-laid to C-logical order first (ravel semantics), then reshaped
                // to 1-D; a 0-d source flattens to shape (1,).
                var flatSource = arr.Shape.IsContiguous ? arr : np.ascontiguousarray(arr);
                arr = flatSource.reshape(flatSource.size);
                ax = 0;
            }
            else
            {
                ax = axis.Value;
                if (ax < 0) ax += arr.ndim;
                if (ax < 0 || ax >= arr.ndim)
                    throw new AxisError(axis.Value, arr.ndim);
            }

            // _make_along_axis_idx: integer-dtype check BEFORE the ndim-match check.
            if (!IsIntegerIndexType(indices.typecode))
                throw new IndexError("`indices` must be an integer array");

            if (arr.ndim != indices.ndim)
                throw new ValueError("`indices` and `arr` must have the same number of dimensions");

            int ndim = arr.ndim;
            long[] arrShape = arr.Shape.dimensions;
            long[] idxShape = indices.Shape.dimensions;

            // Result shape: the axis dimension comes from `indices` (J, which need not equal M);
            // every other dimension is the NumPy broadcast of arr's and indices' extents. NumPy's
            // fancy index is (arange(arr.shape[d]) grids) + indices, so a broadcast conflict on any
            // non-axis dim raises IndexError with all the fancy shapes listed (mapping.c:2617).
            var resultDims = new long[ndim];
            for (int d = 0; d < ndim; d++)
            {
                if (d == ax)
                {
                    resultDims[d] = idxShape[d];
                    continue;
                }

                long a = arrShape[d], i = idxShape[d];
                if (a == i) resultDims[d] = a;
                else if (a == 1) resultDims[d] = i;
                else if (i == 1) resultDims[d] = a;
                else throw new IndexError(BuildBroadcastMismatchMessage(arrShape, idxShape, ax));
            }

            var result = new NDArray(arr.dtype, Shape.Adopt(resultDims), false); // descriptor: a string's width

            long totalSize = result.size;
            if (totalSize == 0)
                return result;   // empty result — nothing to gather (arr may even be empty)

            // int64 view of the indices. When already int64 we keep the ORIGINAL layout (any
            // strides, offset, or broadcast dims) and read through it; otherwise astype yields a
            // C-contiguous int64 copy at the index shape. Either way the odometer reads it with
            // per-dimension strides, so no contiguity is required.
            var idxC = indices.typecode == NPTypeCode.Int64 ? indices : indices.astype(NPTypeCode.Int64);

            long elemBytes = arr.dtypesize;
            long axisStrideBytes = arr.Shape.strides[ax] * elemBytes;
            long axisLen = arrShape[ax];

            var arrStrides = new long[ndim];
            var idxStrides = new long[ndim];
            for (int d = 0; d < ndim; d++)
            {
                // arr contribution along `axis` rides the resolved index (axisStrideBytes), so the
                // odometer stride there is 0; a broadcast (size-1) source dim also contributes 0.
                arrStrides[d] = (d == ax || arrShape[d] == 1) ? 0 : arr.Shape.strides[d] * elemBytes;
                // idx broadcast (size-1) dim contributes 0; the axis dim keeps its real stride (J ≥ 1).
                idxStrides[d] = idxShape[d] == 1 ? 0 : idxC.Shape.strides[d];
            }

            var kernel = DirectILKernelGenerator.GetTakeAlongAxisKernel((int)elemBytes);
            if (kernel is null)
                throw new NotSupportedException("np.take_along_axis: IL kernel unavailable");

            long badIdx = 0;
            long status;
            fixed (long* pArrStrides = arrStrides)
            fixed (long* pIdxStrides = idxStrides)
            fixed (long* pShape = resultDims)
            {
                byte* arrBase = (byte*)arr.Storage.Address + arr.Shape.offset * elemBytes;
                long* idxBase = (long*)idxC.Storage.Address + idxC.Shape.offset;
                byte* dstBase = (byte*)result.Storage.Address;

                status = kernel(arrBase, pArrStrides, axisStrideBytes, axisLen,
                                idxBase, pIdxStrides, dstBase, pShape, ndim, totalSize, &badIdx);
            }

            if (status < totalSize)
                // The kernel reports the FIRST out-of-bounds index in the result's C-order
                // traversal. For a C-contiguous `indices` array (the norm — argsort/argmax output)
                // this is byte-identical to NumPy, whose fancy-index bounds check
                // (PyArray_MapIterCheckIndices) also visits it first. For a deliberately
                // NON-contiguous `indices` array with MULTIPLE out-of-range values, NumPy reports
                // whichever its NpyIter visits first (memory/axis order with negative-stride
                // flipping — see MisalignedRegistry) which need not be the C-order-first; the
                // error TYPE, axis and size still match, only the offending index VALUE may differ.
                throw new IndexError(
                    $"index {badIdx} is out of bounds for axis {ax} with size {axisLen}");

            return result;
        }

        /// <summary>True for the eight integer NPTypeCodes NumPy accepts as index arrays (<c>np.issubdtype(dt, np.integer)</c>).</summary>
        private static bool IsIntegerIndexType(NPTypeCode tc)
        {
            switch (tc)
            {
                case NPTypeCode.Byte:
                case NPTypeCode.SByte:
                case NPTypeCode.Int16:
                case NPTypeCode.UInt16:
                case NPTypeCode.Int32:
                case NPTypeCode.UInt32:
                case NPTypeCode.Int64:
                case NPTypeCode.UInt64:
                    return true;
                default:
                    return false;
            }
        }

        /// <summary>
        ///     Reproduce NumPy's fancy-index broadcast-mismatch message verbatim: it lists every
        ///     fancy index shape in dimension order — an <c>arange(arr.shape[d])</c> grid
        ///     <c>(1,..,arr.shape[d],..,1)</c> for each non-axis dim and <c>indices.shape</c> at
        ///     the axis position — followed by a trailing space.
        /// </summary>
        private static string BuildBroadcastMismatchMessage(long[] arrShape, long[] idxShape, int ax)
        {
            int nd = arrShape.Length;
            var parts = new string[nd];
            for (int d = 0; d < nd; d++)
            {
                long[] shp;
                if (d == ax)
                {
                    shp = idxShape;
                }
                else
                {
                    shp = new long[nd];
                    for (int k = 0; k < nd; k++) shp[k] = 1;
                    shp[d] = arrShape[d];
                }
                parts[d] = "(" + string.Join(",", shp) + ")";
            }
            return "shape mismatch: indexing arrays could not be broadcast together with shapes "
                   + string.Join(" ", parts) + " ";
        }

        // ---------------------------- np.put_along_axis ----------------------------

        /// <summary>
        ///     NumPy's <c>numpy.put_along_axis</c> after the facade's numpy.matrix region and null checks: the axis
        ///     (<c>axis=None</c> writes through the flat view of a C-contiguous array and refuses any other as NumPy's
        ///     read-only flat copy does), then the scatter core.
        /// </summary>
        /// <param name="arr">The destination, modified in place (not null).</param>
        /// <param name="indices">The integer index array (not null).</param>
        /// <param name="values">The values (null is Python's None).</param>
        /// <param name="axis">The axis to put along, or null for the destination flattened in C order.</param>
        /// <exception cref="ValueError"><c>axis=None</c> with indices that are not 1-D; a rank mismatch; values that do not
        ///     broadcast; a read-only destination (a non-contiguous one under <c>axis=None</c>).</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> out of range.</exception>
        /// <exception cref="IndexError">A non-integer index array, a non-axis dimension that does not broadcast, or an
        ///     index out of bounds (nothing is written then).</exception>
        /// <exception cref="TypeError">None into an integer array, or a complex literal into a real one.</exception>
        /// <exception cref="OverflowException">A C# literal out of the destination's integer range.</exception>
        /// <exception cref="NotSupportedException">The IL kernels are unavailable.</exception>
        /// <remarks>Scoped: the <c>axis=None</c> reshape view, the index cast and the value cast or overlap snapshot are
        ///     temporaries of this call.</remarks>
        [NDScoped]
        internal static unsafe void PutAlongAxis(NDArray arr, NDArray indices, NDArray values, int? axis)
        {
            // Boundary scope: reclaims the axis=null contiguous reshape view, the int64 index cast,
            // the values astype/overlap-snapshot, and the stride arrays at exit — and strong-roots the
            // index/value temps THROUGH the raw-pointer kernel call in the core (the ARC/JIT dead-local
            // hazard cannot arise under a scope). `arr` is a parameter, so the scope never disposes it;
            // the contiguous reshape is a view that aliases `arr`'s storage, so its disposal is a no-op
            // on the buffer the writes land in.

            int ax;
            if (axis is null)
            {
                // NumPy: `arr = np.array(arr.flat); axis = 0`. `indices` must be 1-D FIRST.
                if (indices.ndim != 1)
                    throw new ValueError("when axis=None, `indices` must have a single dimension.");

                if (arr.Shape.IsContiguous)
                {
                    // np.array(arr.flat) is a WRITEABLE 1-D view sharing memory for a C-contiguous
                    // source, so the scatter propagates back to `arr` in C-order. A contiguous reshape
                    // IS that view; a read-only source stays read-only, so the core's writeable check
                    // raises exactly as NumPy does. (0-d flattens to (1,); the write lands the element.)
                    arr = arr.reshape(arr.size);
                    ax = 0;
                }
                else
                {
                    // np.array(arr.flat) is a READ-ONLY 1-D COPY for a non-contiguous source, so NumPy's
                    // assignment raises read-only — but _make_along_axis_idx checks the index dtype
                    // FIRST (probed 2.4.2: non-contiguous + float indices -> IndexError, not read-only;
                    // the ndim check is vacuous here since both operands are 1-D).
                    if (!IsIntegerIndexType(indices.typecode))
                        throw new IndexError("`indices` must be an integer array");
                    NumSharpException.ThrowReadOnly();   // "assignment destination is read-only"
                    return;                              // unreachable (ThrowReadOnly is [DoesNotReturn])
                }
            }
            else
            {
                // normalize_axis_index(axis, arr.ndim) — AxisError on out-of-range, reporting the
                // ORIGINAL axis. Runs BEFORE the dtype/ndim checks (NumPy's normalize precedes
                // _make_along_axis_idx).
                ax = axis.Value;
                if (ax < 0) ax += arr.ndim;
                if (ax < 0 || ax >= arr.ndim)
                    throw new AxisError(axis.Value, arr.ndim);
            }

            PutAlongAxisCore(arr, indices, values, ax);
        }

        /// <summary>
        ///     The shared scatter core: validate, resolve the iteration/value broadcast, then run the
        ///     atomic (validate-all-then-write) IL scatter. Called by <see cref="PutAlongAxis"/>
        ///     with <paramref name="arr"/> already normalized to the working axis (the flat 1-D view
        ///     under <c>axis=null</c>). All validation ordering here mirrors NumPy 2.4.2's
        ///     <c>arr[_make_along_axis_idx(...)] = values</c> (probed): dtype -> ndim -> writeable ->
        ///     non-axis fancy broadcast -> value broadcast -> per-index bounds.
        /// </summary>
        /// <param name="arr">Destination, normalized to <paramref name="ax"/> (writeable check happens here).</param>
        /// <param name="indices">Integer index array (matching <paramref name="arr"/>'s rank).</param>
        /// <param name="values">Values to place (cast + broadcast to the iteration shape).</param>
        /// <param name="ax">The resolved (non-negative) axis.</param>
        /// <exception cref="IndexError"><paramref name="indices"/> non-integer / non-axis broadcast conflict / out-of-bounds.</exception>
        /// <exception cref="ValueError">rank mismatch, or <paramref name="values"/> not broadcastable to the result shape.</exception>
        /// <exception cref="ValueError"><paramref name="arr"/> is read-only — NumPy's "assignment destination is read-only".</exception>
        /// <exception cref="NotSupportedException">the IL kernels are unavailable (e.g. codegen disabled).</exception>
        /// <exception cref="OverflowException">A C# literal out of <paramref name="arr"/>'s integer range.</exception>
        /// <exception cref="TypeError">A complex literal into a real <paramref name="arr"/>.</exception>
        private static unsafe void PutAlongAxisCore(NDArray arr, NDArray indices, NDArray values, int ax)
        {
            // _make_along_axis_idx: dtype check BEFORE the ndim-match check.
            if (!IsIntegerIndexType(indices.typecode))
                throw new IndexError("`indices` must be an integer array");
            if (arr.ndim != indices.ndim)
                throw new ValueError("`indices` and `arr` must have the same number of dimensions");

            // Assignment writeable guard — fires before the fancy/value broadcasts and the bounds pass
            // (probed 2.4.2: read-only reported ahead of a bad value shape, a non-axis conflict, and an
            // out-of-bounds index; only the dtype check precedes it).
            NumSharpException.ThrowIfNotWriteable(arr.Shape);

            // A C# literal (NDArray.WasPythonLiteral) is NumPy's Python scalar: the fancy assignment converts the value
            // with PyArray_FromAny(values, arr.dtype, FORCECAST), which packs it with PyArray_Pack's range checks
            // (put_along_axis(int8, idx, 1000, 0) raises "Python integer 1000 out of bounds for int8") after the read-only
            // check and BEFORE the index arrays are broadcast or bounds-checked. The packed 0-d is the caller's scope's. A null
            // is Python's None, converted at the same point into arr's descriptor (NaN, nan+nanj, False, 'None' truncated, a
            // StringDType NA object; int(None)'s TypeError for an integer array, before an out-of-bounds index is seen).
            values = NDArray.PackForStore(values, arr.dtype, pyArrayPackRoute: false);

            int ndim = arr.ndim;
            long[] arrShape = arr.Shape.dimensions;
            long[] idxShape = indices.Shape.dimensions;

            // Iteration (indexing result) shape: the axis dim comes from `indices` (J, which need not
            // equal M); every other dim is the NumPy broadcast of arr's and indices' extents. Identical
            // to take_along_axis; a non-axis broadcast conflict raises the same fancy-index IndexError,
            // listing each arange-grid shape.
            var resultDims = new long[ndim];
            for (int d = 0; d < ndim; d++)
            {
                if (d == ax) { resultDims[d] = idxShape[d]; continue; }

                long a = arrShape[d], i = idxShape[d];
                if (a == i) resultDims[d] = a;
                else if (a == 1) resultDims[d] = i;
                else if (i == 1) resultDims[d] = a;
                else throw new IndexError(BuildBroadcastMismatchMessage(arrShape, idxShape, ax));
            }

            long totalSize = 1;
            for (int d = 0; d < ndim; d++) totalSize *= resultDims[d];

            if (arr.dtype.Meta.HasReferences)
            {
                // A StringDType target (its elements hold references, NumPy's PyDataType_REFCHK): the values in its
                // descriptor, then the same scatter on marks (the values' element numbers: the value broadcast, the overlap
                // snapshot and the bounds pass are the numeric route's) and the marked elements receive the strings.
                var tvals = Backends.Strings.NDVStringMove.StoreValue(values, arr.dtype);
                Backends.Strings.NDVStringMove.ScatterVia(arr, tvals, (marks, index) => PutAlongAxisCore(marks, indices, index, ax));
                return;
            }

            long elemBytes = arr.dtypesize;

            // Cast values to arr's dtype (assignment cast — floats truncate toward zero; over/underflow
            // wraps modularly, matching NumPy's ndarray-value path and the put/place/putmask siblings).
            // A same-dtype source is used AS-IS: any layout / broadcast dims are read through strides.
            // Convert unless the element type already matches (for strings: class AND width).
            NDArray valArr = values.HasSameElementType(arr) ? values : values.astype(arr.dtype);

            // COPY_IF_OVERLAP: if the values source may share memory with `arr`, snapshot it so the
            // scatter reads the ORIGINAL values rather than progressively-overwritten ones (probed
            // 2.4.2: put_along_axis(a, reversing_idx, a, ...) reverses a slice via a copy). An astype'd
            // source is already a fresh buffer that cannot alias `arr`, so this only bites same-dtype
            // aliases (values IS arr, or a view of it). NPY_MAY_SHARE_BOUNDS == maxWork:0.
            if (NDMemOverlap.SolveMayShareMemory(arr, valArr, maxWork: 0) != MemOverlap.No)
                valArr = valArr.copy();

            // Value broadcast strides to the iteration shape (assignment broadcast: right-aligned, extra
            // LEADING size-1 value dims stripped). Raises the verbatim value-broadcast ValueError on
            // mismatch — validated BEFORE the bounds pass (probed: bad value shape reported ahead of an
            // out-of-bounds index).
            var valStrides = BuildValueBroadcastStrides(valArr, resultDims, elemBytes);

            // Every shape/dtype validation is now done; an empty iteration writes nothing (NumPy's
            // zero-trip loop). Reached e.g. by J==0 indices or a size-0 non-axis dimension.
            if (totalSize == 0)
                return;

            // int64 view of the indices — read through per-dimension strides, so no contiguity needed.
            var idxC = indices.typecode == NPTypeCode.Int64 ? indices : indices.astype(NPTypeCode.Int64);

            long axisStrideBytes = arr.Shape.strides[ax] * elemBytes;
            long axisLen = arrShape[ax];

            var arrStrides = new long[ndim];
            var idxStrides = new long[ndim];
            for (int d = 0; d < ndim; d++)
            {
                // arr's axis contribution rides the resolved index (axisStrideBytes), so its odometer
                // stride is 0 there; a broadcast (size-1) arr dim also contributes 0 — several iteration
                // positions collapse onto arr coordinate 0 along that dim (last write wins).
                arrStrides[d] = (d == ax || arrShape[d] == 1) ? 0 : arr.Shape.strides[d] * elemBytes;
                // idx broadcast (size-1) dim contributes 0; else its real element stride (J >= 1 at ax).
                idxStrides[d] = idxShape[d] == 1 ? 0 : idxC.Shape.strides[d];
            }

            var validate = DirectILKernelGenerator.GetPutAlongAxisValidateKernel();
            var scatter = DirectILKernelGenerator.GetPutAlongAxisScatterKernel((int)elemBytes);
            if (validate is null || scatter is null)
                throw new NotSupportedException("np.put_along_axis: IL kernel unavailable");

            long badIdx = 0;
            fixed (long* pArrStrides = arrStrides)
            fixed (long* pIdxStrides = idxStrides)
            fixed (long* pValStrides = valStrides)
            fixed (long* pShape = resultDims)
            {
                byte* arrBase = (byte*)arr.Storage.Address + arr.Shape.offset * elemBytes;
                long* idxBase = (long*)idxC.Storage.Address + idxC.Shape.offset;
                byte* valBase = (byte*)valArr.Storage.Address + valArr.Shape.offset * elemBytes;

                // Pass 1: validate ALL indices before writing anything (NumPy's all-or-nothing
                // assignment). On the first out-of-bounds index, `arr` is left completely untouched.
                long status = validate(idxBase, pIdxStrides, axisLen, pShape, ndim, totalSize, &badIdx);
                if (status < totalSize)
                    throw new IndexError(
                        $"index {badIdx} is out of bounds for axis {ax} with size {axisLen}");

                // Pass 2: the bounds-check-free scatter (indices now known in range).
                scatter(arrBase, pArrStrides, axisStrideBytes, axisLen,
                        idxBase, pIdxStrides, valBase, pValStrides, pShape, ndim, totalSize);
            }
        }

        /// <summary>
        ///     Build the per-iteration-dimension BYTE strides for reading <paramref name="valArr"/>
        ///     broadcast to <paramref name="resultDims"/>, using NumPy's assignment-broadcast rule
        ///     (which is more lenient than <c>broadcast_to</c>): right-aligned, and extra LEADING
        ///     size-1 value dimensions are stripped. A stretched (size-1) or absent value dimension
        ///     contributes stride 0.
        /// </summary>
        /// <param name="valArr">The values array (already cast to the destination dtype).</param>
        /// <param name="resultDims">The indexing result (iteration) shape.</param>
        /// <param name="elemBytes">Element size in bytes (to convert element strides to byte strides).</param>
        /// <returns>A BYTE-stride array of length <c>resultDims.Length</c> (0 at broadcast dimensions).</returns>
        /// <exception cref="ValueError">
        ///     <paramref name="valArr"/> cannot broadcast to <paramref name="resultDims"/> — a leading
        ///     extra dimension is not 1, or an aligned dimension is neither 1 nor equal to the target.
        ///     The message reads verbatim like NumPy's assignment broadcast error.
        /// </exception>
        private static long[] BuildValueBroadcastStrides(NDArray valArr, long[] resultDims, long elemBytes)
        {
            long[] valDims = valArr.Shape.dimensions;
            int vn = valDims.Length;
            int rn = resultDims.Length;
            int extra = vn - rn;   // leading value dims with no result counterpart (may be negative)

            // Extra leading value dimensions (assignment strips them) must all be 1.
            for (int k = 0; k < extra; k++)
                if (valDims[k] != 1)
                    throw new ValueError(ValueBroadcastMessage(valArr.Shape, resultDims));

            var valStrides = new long[rn];
            for (int rd = 0; rd < rn; rd++)
            {
                int vd = extra + rd;                     // = vn - rn + rd (right-aligned value dim)
                if (vd < 0) { valStrides[rd] = 0; continue; }   // value shorter here -> broadcast

                long vDim = valDims[vd], rDim = resultDims[rd];
                if (vDim == 1) valStrides[rd] = 0;                                    // broadcast this dim
                else if (vDim == rDim) valStrides[rd] = valArr.Shape.strides[vd] * elemBytes;
                else throw new ValueError(ValueBroadcastMessage(valArr.Shape, resultDims));
            }
            return valStrides;
        }

        /// <summary>NumPy's verbatim assignment value-broadcast error (Python tuple spelling, e.g. <c>(2,)</c>).</summary>
        private static string ValueBroadcastMessage(Shape valShape, long[] resultDims)
            => "shape mismatch: value array of shape " + valShape.ToPythonTuple()
               + " could not be broadcast to indexing result of shape " + PyShapeTuple(resultDims);

        // ---------------------------- np.apply_along_axis ----------------------------

        // =====================================================================
        //  np.apply_along_axis — apply a function to 1-D slices along an axis.
        //
        //  Port of NumPy 2.x numpy.apply_along_axis (numpy/lib/_shape_base_impl.py). NumPy's own
        //  structure, followed here move for move:
        //
        //      axis = normalize_axis_index(axis, arr.ndim)
        //      inarr_view = transpose(arr, [..axis-1, axis+1.., axis])   # iteration axis last
        //      inds = ndindex(inarr_view.shape[:-1])                     # the non-axis dims
        //      res  = asanyarray(func1d(inarr_view[ind0], ...))          # first slice fixes shape+dtype
        //      buff = zeros(inarr_view.shape[:-1] + res.shape, res.dtype)
        //      for ind: buff[ind] = func1d(inarr_view[ind], ...)         # each write is contiguous
        //      out  = transpose(buff, buff_permute)                      # move the new dims into `axis`
        //
        //  There is no data-parallel kernel to emit here: func1d is arbitrary user code invoked once
        //  per 1-D slice, so this is a composition over the existing view/iterator machinery
        //  (transpose views, ndindex, GetData slices, copyto writes). The C# delegate call is orders
        //  of magnitude cheaper than NumPy's per-slice Python call, so even this straight composition
        //  outruns NumPy comfortably — the callback dispatch, not the plumbing, is the cost.
        // =====================================================================

        /// <summary>
        ///     NumPy's <c>numpy.apply_along_axis</c> after the facade's numpy.matrix rule and null checks: the axis moved
        ///     last, the first slice's result fixing the buffer's dtype and trailing shape, every slice's result written into
        ///     its slot, and the buffer transposed so the result's dimensions replace the axis.
        /// </summary>
        /// <param name="func1d">The function of a 1-D slice (not null).</param>
        /// <param name="axis">The axis to slice along; negative counts from the end.</param>
        /// <param name="arr">The input (not null).</param>
        /// <returns>A transposed view of a fresh buffer holding the results, of the first result's dtype (a text scalar's
        ///     as wide as its value, <see cref="TextScalarAsArray"/>).</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> out of range (the original axis reported).</exception>
        /// <exception cref="ValueError">An iteration dimension is 0, or <paramref name="func1d"/> returned null.</exception>
        /// <remarks>Not scoped: every array <paramref name="func1d"/> returns, and every slice handed to it, stays the
        ///     caller's; the buffer's wrapper and the transposed input are released here by hand.</remarks>
        internal static NDArray ApplyAlongAxis(Func<NDArray, NDArray> func1d, int axis, NDArray arr)
        {
            int nd = arr.ndim;

            // normalize_axis_index: AxisError reports the ORIGINAL axis value (before wrapping).
            int ax = axis >= 0 ? axis : nd + axis;
            if (ax < 0 || ax >= nd)
                throw new AxisError(axis, nd);

            // Move the iteration axis to the end so each 1-D slice is the trailing dimension:
            // perm = [0..ax-1, ax+1..nd-1, ax]. transpose returns a view (no data movement).
            var perm = new int[nd];
            int p = 0;
            for (int i = 0; i < nd; i++)
                if (i != ax)
                    perm[p++] = i;
            perm[nd - 1] = ax;
            NDArray inarr = np.transpose(arr, perm);

            // The iteration space is every dimension except the last (the sliced axis). For a 1-D
            // input this is the empty shape, over which ndindex yields exactly one (empty) coordinate.
            long[] inDims = inarr.Shape.dimensions;
            int leadingLen = nd - 1;
            var leading = new long[leadingLen];
            System.Array.Copy(inDims, 0, leading, 0, leadingLen);

            // Drive ndindex manually: the FIRST coordinate is consumed before the buffer exists (its
            // func result fixes the buffer's shape+dtype), the rest fill the buffer. When a leading
            // dimension is 0 the sequence is empty — there is no first slice — which is exactly the
            // condition NumPy rejects.
            IEnumerator<long[]> inds = np.ndindex(leading).GetEnumerator();
            if (!inds.MoveNext())
                throw new ValueError("Cannot apply_along_axis when any iteration dimensions are 0");

            long[] ind0 = inds.Current;
            NDArray res0 = InvokeFunc1d(func1d, inarr, ind0);
            // NumPy: res = asanyarray(func1d(inarr_view[ind0])). A text SCALAR result becomes an array as wide as its value
            // (np.asanyarray(np.str_('3')) is <U1), so the buffer takes that width and every later result is cast into it
            // (truncated), as in NumPy; the array made here is this method's, released once its slot is written.
            NDArray firstResult = res0;
            res0 = TextScalarAsArray(res0);
            bool res0Made = !ReferenceEquals(res0, firstResult);

            int resNd = res0.ndim;
            long[] resDims = res0.Shape.dimensions ?? System.Array.Empty<long>();

            // Buffer laid out so each write is contiguous: leading dims first, result dims trailing.
            // Allocated at the FIRST result's dtype (a DType — NumPy's zeros_like(res, shape=...)).
            var bufDims = new long[leadingLen + resNd];
            System.Array.Copy(leading, 0, bufDims, 0, leadingLen);
            System.Array.Copy(resDims, 0, bufDims, leadingLen, resNd);
            NDArray buff = np.zeros(Shape.Adopt(bufDims), res0.dtype);

            // Because ndindex walks the leading dims in C-order and `buff` is C-contiguous, the k-th
            // slice's slot is exactly the k-th contiguous block of `buff`'s bytes — NumPy's "each
            // write is contiguous" layout. When a result matches that slot verbatim (same dtype,
            // same shape, contiguous) we blit its bytes straight into the block, skipping the
            // per-slice destination view AND copyto's broadcast/cast setup (the dominant cost for a
            // tiny slice with a fast func); anything else falls back to the view+copyto path, which
            // reproduces NumPy's broadcast-on-assign and cast for a differing result. The span form
            // needs a 32-bit byte count, so it is gated on the whole buffer fitting int.MaxValue.
            // A StringDType cell is a pointer or an arena offset into ITS array's allocator: copying the raw bytes into the
            // buffer would leave the buffer's cells pointing at the result's storage (and its element size is not a byte
            // count a route may use). Those go through copyto, which writes each text into the buffer's own allocator; the
            // test comes first because the byte counts below refuse a StringDType array. The descriptor answers which buffers
            // those are: the ones whose elements hold references (NumPy's PyDataType_REFCHK).
            bool useSpanWrite = !buff.dtype.Meta.HasReferences && buff.nbytes <= int.MaxValue;
            long resBytes = useSpanWrite ? res0.size * buff.dtypesize : 0;
            Span<byte> buffBytes = useSpanWrite ? buff.Unsafe.Bytes() : default;

            try
            {
                WriteSlot(buff, buffBytes, useSpanWrite, 0, ind0, res0, resDims, resBytes);
            }
            finally
            {
                // The re-discovered text scalar is this method's own array (func1d's result stays the caller's): released
                // once its slot is written, or when the write fails. Only its dims (resDims) and byte count are used later.
                if (res0Made)
                    res0.Dispose();
            }
            long k = 1;
            while (inds.MoveNext())
            {
                long[] ind = inds.Current;
                WriteSlot(buff, buffBytes, useSpanWrite, k, ind, InvokeFunc1d(func1d, inarr, ind), resDims, resBytes);
                k++;
            }

            // Permute the buffer so the result dims land back at `axis`:
            //   buff_permute = [0:axis] + [buffNd-resNd : buffNd] + [axis : buffNd-resNd]
            // yielding shape arr.shape[:axis] + res.shape + arr.shape[axis+1:].
            int buffNd = leadingLen + resNd;
            var buffPermute = new int[buffNd];
            int q = 0;
            for (int i = 0; i < ax; i++)
                buffPermute[q++] = i;
            for (int i = buffNd - resNd; i < buffNd; i++)
                buffPermute[q++] = i;
            for (int i = ax; i < buffNd - resNd; i++)
                buffPermute[q++] = i;

            // Ownership: this method may release ONLY what it created and never handed to func1d — the
            // `buff` wrapper (the returned transpose view holds its own reference, so the buffer lives on)
            // and the `inarr` view (each slice was re-wrapped with its own reference). It must NOT scope or
            // dispose func1d's results or the slices passed to it: a callback may return or retain arrays
            // the caller still owns (a held array, the slice itself), so their lifetime stays the caller's.
            // Without the release, every call stranded `buff`'s reference — one pooled buffer until a GC.
            var result = np.transpose(buff, buffPermute);
            buff.Dispose();
            inarr.Dispose();
            return result;
        }

        /// <summary>
        ///     Invokes <paramref name="func1d"/> on the 1-D slice at leading coordinate
        ///     <paramref name="ind"/> and guards a null result (NumPy would build a 0-d object array
        ///     from <c>None</c>, a dtype NumSharp has no analog for — a null here is a caller bug, so
        ///     it is surfaced rather than allowed to NRE deep in the buffer allocation).
        /// </summary>
        /// <param name="func1d">The per-slice function.</param>
        /// <param name="inarr">The axis-last transposed view of the input.</param>
        /// <param name="ind">Leading coordinate (over the non-axis dims); empty for a 1-D input.</param>
        /// <returns>The func's result as an <see cref="NDArray"/>.</returns>
        /// <exception cref="ValueError"><paramref name="func1d"/> returned null.</exception>
        private static NDArray InvokeFunc1d(Func<NDArray, NDArray> func1d, NDArray inarr, long[] ind)
        {
            // GetData(ind) with (ndim-1) coordinates returns the trailing 1-D slice (a writeable view
            // sharing storage — read-only use here); an empty `ind` (1-D input) returns the whole 1-D
            // array. This is NumPy's inarr_view[ind + (Ellipsis,)] without the 0-d-decay hazard, since
            // GetData never collapses to a scalar.
            NDArray slice = inarr.GetData(ind);
            NDArray res = func1d(slice);
            if (res is null)
                throw new ValueError("apply_along_axis: func1d returned null");
            return res;
        }

        /// <summary>
        ///     NumPy's <c>asanyarray</c> of a func1d result when the result is a TEXT SCALAR: an unmarked 0-d array of a
        ///     string dtype stands for NumPy's <c>np.str_</c> / <c>np.bytes_</c> (an element of a fixed-width array, as
        ///     <c>lambda v: v[0]</c> returns) or for StringDType's scalar, a Python <c>str</c> (what <c>np.sum</c> of a
        ///     StringDType slice returns), and <c>np.asanyarray</c> of a text scalar is an array as wide as its value:
        ///     <c>np.asanyarray(np.str_('3'))</c> is <c>&lt;U1</c> whatever array the element came from, an empty value
        ///     <c>&lt;U1</c> / <c>|S1</c>. Every other result (an array of any rank, a number, an explicit 0-d array) is
        ///     returned as it is.
        /// </summary>
        /// <param name="res">A func1d result (not null).</param>
        /// <returns><paramref name="res"/> itself, or a NEW 0-d <c>U</c> / <c>S</c> array the caller disposes.</returns>
        /// <remarks>
        ///     This decides the buffer's dtype in <see cref="ApplyAlongAxis"/>: NumPy allocates <c>zeros_like(res)</c>, so a
        ///     <c>&lt;U5</c> source whose first element is one character gives a <c>&lt;U1</c> result, and later, longer
        ///     results are truncated to it (probed 2.4.2). A StringDType NA element is no str (NumPy's <c>na_object</c>): it
        ///     keeps its array, as no NumSharp dtype holds NumPy's object scalar.
        /// </remarks>
        private static NDArray TextScalarAsArray(NDArray res)
        {
            // An explicit 0-d array (np.array<T>'s mark) is an ndarray in NumPy too: kept whole. A scalar stand-in of a number
            // is its own dtype's array already.
            if (res.ndim != 0 || res.IsExplicitZeroDimArray || res.dtype.kind is not ('U' or 'S' or 'T'))
                return res;
            // The value as NumPy's scalar holds it (trailing NULs stripped): a str for U and StringDType, bytes for S; the
            // coercion discovers its class and width as np.asanyarray does.
            object value = res.Storage.GetAtIndex(0);
            return value is string or byte[] ? Backends.UnmanagedStorage.FromAnyScalar(value, null) : res;
        }

        /// <summary>
        ///     Writes one func result into the buffer's k-th contiguous slot. FAST PATH (when
        ///     <paramref name="useSpanWrite"/> and the result matches the slot verbatim — same dtype
        ///     as the buffer, same shape as the first result, C-contiguous): blit the result's raw
        ///     bytes into the block at <c>k * resBytes</c>, no destination view, no cast/broadcast
        ///     setup. FALLBACK (any other result — a differing shape, dtype, or non-contiguous layout,
        ///     or a buffer too large for a 32-bit span): the trailing result-shaped sub-view
        ///     <c>buff.GetData(ind)</c> plus <see cref="np.copyto(NDArray, NDArray, string, NDArray)"/>, which reproduces NumPy's
        ///     <c>buff[ind] = res</c> broadcast-on-assign and unsafe cast exactly.
        /// </summary>
        /// <param name="buff">The contiguous output buffer (leading dims + result dims).</param>
        /// <param name="buffBytes">The whole buffer as bytes (valid only when <paramref name="useSpanWrite"/>).</param>
        /// <param name="useSpanWrite">True if the buffer fits a 32-bit span, enabling the blit fast path.</param>
        /// <param name="k">The linear slot index (C-order over the leading dims) — the fast-path block offset.</param>
        /// <param name="ind">Leading coordinate; empty targets the whole buffer (1-D input). Used only by the fallback.</param>
        /// <param name="res">The func result to store.</param>
        /// <param name="resDims">The first result's dimensions — the slot shape the fast path requires a verbatim match to.</param>
        /// <param name="resBytes">Bytes per slot (<c>res0.size * buff.dtypesize</c>).</param>
        private static void WriteSlot(NDArray buff, Span<byte> buffBytes, bool useSpanWrite, long k,
            long[] ind, NDArray res, long[] resDims, long resBytes)
        {
            // Blit only when the result IS the slot: matching the first result's dtype and shape means
            // no cast and no broadcast, and C-contiguity means its bytes are already in slot order.
            if (useSpanWrite && res.dtype == buff.dtype && res.Shape.IsContiguous && SameDims(res.Shape.dimensions, resDims))
            {
                res.Unsafe.ReadOnlyBytes().CopyTo(buffBytes.Slice((int)(k * resBytes), (int)resBytes));
                return;
            }

            NDArray dst = buff.GetData(ind);
            np.copyto(dst, res, casting: "unsafe");
        }

        /// <summary>
        ///     Compares two dimension arrays for exact equality (rank and every extent). Used to
        ///     decide whether a func result matches the buffer slot verbatim — a mismatch routes to
        ///     the broadcast/cast fallback so a differing result behaves like NumPy's item assignment.
        /// </summary>
        /// <param name="a">First dimension array (a result's dimensions; may be null for a 0-d result).</param>
        /// <param name="b">Second dimension array (the first result's dimensions).</param>
        /// <returns>True if both are the same rank with identical extents (two 0-d/empty are equal).</returns>
        private static bool SameDims(long[] a, long[] b)
        {
            int an = a?.Length ?? 0;
            int bn = b?.Length ?? 0;
            if (an != bn)
                return false;
            for (int i = 0; i < an; i++)
                if (a[i] != b[i])
                    return false;
            return true;
        }
    }
}
