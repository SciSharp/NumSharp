using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

namespace NumSharp
{
    // ============================== lib/_shape_base_impl.py ==============================
    // NumPy's numpy/lib/_shape_base_impl.py array-object functions as members of the array object: expand_dims,
    // take_along_axis, put_along_axis with its scatter core and apply_along_axis (P3's batch 7), and the split family
    // (split, array_split, hsplit, vsplit, dsplit, with SplitContext's view builder), tile, column_stack and dstack_ (P8).
    // The facades keep their numpy.matrix / numpy.char rules and their null checks; np.apply_along_axis's *args overload
    // wraps its delegate and calls the one-argument facade. np.apply_over_axes stays a facade: its np.expand_dims of the
    // callback's result applies the chararray rule (NumPy's asanyarray + reshape keep a chararray result's class), which
    // these plain statements would not. The statements call the members the np.* functions dispatch to (the plan's phase
    // P6), but for the ndindex iterator apply_along_axis walks (np.NDIndex, NumPy's ndindex: a public type nested in np,
    // which ratchet R2 pins), and the file sits under Backends/, beside the array object's class (plan C3).
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
        internal static NDArray expand_dims(NDArray a, int axis)
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
        internal static unsafe NDArray take_along_axis(NDArray arr, NDArray indices, int? axis)
        {
            // A StringDType source: the same gather runs on its cell view (NumPy's checks, this array's shapes) and the cells
            // are duplicated (NDVStringMove) — never the raw byte kernel below. The descriptor answers which sources those
            // are: the ones whose elements hold references (NumPy's PyDataType_REFCHK).
            if (arr.dtype.Meta.HasReferences)
                return Backends.Strings.NDVStringMove.Gather(arr, cells => take_along_axis(cells, indices, axis));

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
                var flatSource = arr.Shape.IsContiguous ? arr : ascontiguousarray(arr, null);
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
        internal static unsafe void put_along_axis(NDArray arr, NDArray indices, NDArray values, int? axis)
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
        ///     atomic (validate-all-then-write) IL scatter. Called by <see cref="put_along_axis"/>
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
        internal static NDArray apply_along_axis(Func<NDArray, NDArray> func1d, int axis, NDArray arr)
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
            // np.transpose(arr, perm)'s statement, the engine's Transpose view: a numpy.matrix operand runs inside its
            // facade's rule, and a chararray one is sliced plain by GetData whatever class the view has.
            NDArray inarr = arr.TensorEngine.Transpose(arr, perm);

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
            // np.ndindex(leading)'s statement: NumPy's ndindex iterator (np.NDIndex, validated at construction).
            IEnumerator<long[]> inds = new np.NDIndex(leading).GetEnumerator();
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
            NDArray buff = new NDArray(res0.dtype, Shape.Adopt(bufDims), true);

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
            var result = buff.TensorEngine.Transpose(buff, buffPermute);
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
        ///     This decides the buffer's dtype in <see cref="apply_along_axis"/>: NumPy allocates <c>zeros_like(res)</c>, so a
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
        ///     <c>buff.GetData(ind)</c> plus the assignment <see cref="UnmanagedStorage.Copyto"/> at <c>unsafe</c> (np.copyto's
        ///     statement), which reproduces NumPy's <c>buff[ind] = res</c> broadcast-on-assign and unsafe cast exactly.
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
            UnmanagedStorage.Copyto(dst, res, NPY_CASTING.NPY_UNSAFE_CASTING, null);
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

        // ---------------------------- np.column_stack ----------------------------

        /// <summary>
        ///     NumPy's <c>numpy.column_stack(tup)</c> after the facade's class rules and null check: an array below two
        ///     dimensions becomes a column (NumPy's <c>array(arr, copy=None, subok=True, ndmin=2).T</c>, one view here), an
        ///     array of two dimensions or more passes through, and the list joins along axis 1. The column views are this
        ///     call's own and are released once the join has copied them, on the join's error path too.
        /// </summary>
        /// <param name="tup">The arrays (the array itself not null; a null entry is refused before anything joins).</param>
        /// <returns>A new array: concatenate's result along axis 1, in its promoted dtype and NumPy's layout vote (the column
        ///     of a contiguous 1-d array is both C- and F-contiguous, so columns alone vote C).</returns>
        /// <exception cref="ArgumentNullException">A null entry (the parameter is named <c>tup[i]</c> with the first null
        ///     entry's index). The columns made for the entries before it are left to the finalizer.</exception>
        /// <exception cref="ValueError">No array: concatenate's <c>need at least one array to concatenate</c>, the text NumPy's
        ///     column_stack([]) raises.</exception>
        /// <exception cref="IncorrectShapeException">Arrays whose ranks or first dimensions differ once turned into columns
        ///     (concatenate's NumPy texts).</exception>
        /// <exception cref="InvalidCastException">An array that does not cast to the result dtype (concatenate's
        ///     refusal).</exception>
        /// <remarks>No class boundary runs here: the facade applies its rules (a matrix result wrapped, a chararray operand
        ///     computed plain) before it calls this. Every error but a null entry comes from concatenate, exactly as in NumPy,
        ///     whose column_stack ends with <c>_nx.concatenate(arrays, 1)</c>.</remarks>
        internal static NDArray column_stack(NDArray[] tup)
        {
            // Port of numpy/lib/_shape_base_impl.py::column_stack —
            //   arr = array(arr, copy=None, subok=True, ndmin=2).T  when ndim < 2
            // ndmin=2 prepends a leading 1 as a VIEW: (N,) → (1,N), () → (1,1);
            // the transpose then turns the row into a column: (N,1) / (1,1).
            // Inputs of ndim >= 2 pass through untouched. Errors (empty tup,
            // ndim/shape mismatches, promotion) all flow from concatenate,
            // exactly as in NumPy.
            var arrays = new NDArray[tup.Length];
            NDArray[] columnViews = null; // views created here; released post-concat
            for (int i = 0; i < tup.Length; i++)
            {
                var arr = tup[i];
                if (arr is null)
                    throw new ArgumentNullException($"{nameof(tup)}[{i}]");

                if (arr.ndim < 2)
                {
                    arr = AsColumnView(arr);
                    (columnViews ??= new NDArray[tup.Length])[i] = arr;
                }

                arrays[i] = arr;
            }

            try
            {
                return concatenate(arrays, 1, null, null, "same_kind");
            }
            finally
            {
                // The (N,1) wrappers alias caller storage and are dead after the
                // copy — release them eagerly (same pattern as concatenate's
                // axis=null ravel intermediates).
                if (columnViews != null)
                    for (int i = 0; i < columnViews.Length; i++)
                        columnViews[i]?.Dispose();
            }
        }

        /// <summary>
        ///     <c>array(arr, copy=None, ndmin=2).T</c> as ONE view: (N,) → (N,1)
        ///     and () → (1,1), preserving stride/offset (a strided 1-d input stays
        ///     a view, like NumPy's ndmin path). The dim-1 column stride mirrors
        ///     what the (1,N)-transpose produces (N*stride), keeping the view
        ///     both C- and F-contiguous for contiguous inputs — which is what the
        ///     concatenate layout vote (ambiguous → C order) keys off, per NumPy.
        /// </summary>
        /// <param name="arr">An array of rank 0 or 1 (not null); a higher rank would be read as its first axis only.</param>
        /// <returns>A new (N,1) or (1,1) view over <paramref name="arr"/>'s buffer, in <paramref name="arr"/>'s engine: it
        ///     holds its own reference on the buffer, so the caller must dispose it (column_stack does once the join has
        ///     copied it).</returns>
        private static NDArray AsColumnView(NDArray arr)
        {
            var sh = arr.Shape;
            long n, stride0;
            if (arr.ndim == 0)
            {
                n = 1;
                stride0 = 1;
            }
            else
            {
                n = sh.dimensions[0];
                stride0 = sh.strides[0];
            }

            var column = new Shape(new[] { n, 1L }, new[] { stride0, n * stride0 }, sh.offset, sh.bufferSize);
            return new NDArray(arr.Storage.Alias(column), arr.TensorEngine, skipEngineResolve: true);
        }

        // ---------------------------- np.dstack ----------------------------

        /// <summary>
        ///     NumPy's <c>numpy.dstack(tup)</c> after the facade's class rules and null check: <c>arrs = atleast_3d(*tup);
        ///     return concatenate(arrs, 2)</c> - a 1-d array of length N becomes (1, N, 1), a 2-d array (M, N) becomes
        ///     (M, N, 1), and the arrays join along the third axis.
        /// </summary>
        /// <param name="tup">The arrays (the array itself not null; a null entry is NumPy's None, refused).</param>
        /// <returns>A new array of three dimensions or more: concatenate's result, in its promoted dtype and NumPy's layout
        ///     vote.</returns>
        /// <exception cref="ValueError">No array: <c>atleast_3d(*tup)</c> of nothing is the empty tuple, and concatenating it
        ///     is concatenate's <c>need at least one array to concatenate</c>, the text NumPy's dstack([]) raises.</exception>
        /// <exception cref="NullReferenceException">A null entry (read by atleast_3d).</exception>
        /// <exception cref="IncorrectShapeException">Arrays whose ranks, or dimensions other than the third, differ
        ///     (concatenate's NumPy texts).</exception>
        /// <exception cref="InvalidCastException">An array that does not cast to the result dtype (concatenate's
        ///     refusal).</exception>
        /// <remarks>No class boundary runs here: the facade applies its rules (a matrix result wrapped, a chararray operand
        ///     computed plain) before it calls this, so the inner atleast_3d and concatenate are the members. The atleast_3d
        ///     views are not released here, as <see cref="vstack_"/>'s atleast_2d views are not: each holds its input's buffer
        ///     until it is collected. The trailing underscore is NumPy's convention for a name already taken (D18):
        ///     <see cref="dstack(NDArray[])"/> is the array object's public instance method that stacks the array with
        ///     others.</remarks>
        internal static NDArray dstack_(NDArray[] tup)
        {
            // NumPy's atleast_3d(*tup) of no arrays is the empty tuple, and concatenating it is np.concatenate's ValueError —
            // the text NumPy's dstack([]) raises.
            if (tup.Length == 0)
                throw new ValueError("need at least one array to concatenate");

            var arrs = new NDArray[tup.Length];
            for (int i = 0; i < tup.Length; i++)
                arrs[i] = atleast_3d(tup[i]);
            return concatenate(arrs, 2, null, null, "same_kind");
        }

        // ---------------------------- np.array_split / np.split ----------------------------

        /// <summary>
        ///     NumPy's <c>numpy.array_split(ary, indices_or_sections, axis)</c> for a section count, after the facade's class
        ///     rules and null check: <c>Ntotal = ary.shape[axis]</c> first, then the count's check, then
        ///     <paramref name="indices_or_sections"/> views along the axis - the first <c>Ntotal % sections</c> of them one
        ///     element longer than the rest, so the count need not divide the axis.
        /// </summary>
        /// <param name="ary">The array to divide (not null).</param>
        /// <param name="indices_or_sections">The number of sections; their lengths differ by at most one.</param>
        /// <param name="axis">The axis to split along; negative counts from the end.</param>
        /// <returns>The sections, each a VIEW of <paramref name="ary"/>'s buffer (it writes through and keeps the buffer
        ///     alive; a read-only or broadcast operand gives read-only views), plain arrays whatever <paramref name="ary"/>'s
        ///     class. A count larger than the axis gives empty views at the end.</returns>
        /// <exception cref="IndexError"><paramref name="axis"/> is out of range ("tuple index out of range": NumPy reads
        ///     <c>ary.shape[axis]</c>, a Python tuple index, before anything else; a 0-d array refuses every axis).</exception>
        /// <exception cref="ValueError"><paramref name="indices_or_sections"/> is not positive ("number sections must be
        ///     larger than 0.").</exception>
        /// <remarks>No class boundary runs here: np.array_split applies its rules before it calls this.</remarks>
        internal static NDArray[] array_split(NDArray ary, int indices_or_sections, int axis)
        {
            // `Ntotal = ary.shape[axis]` first, then `Nsections = int(…); if Nsections <= 0: raise`.
            int ax = NormalizeSplitAxis(axis, ary.ndim);
            if (indices_or_sections <= 0)
                throw new ValueError("number sections must be larger than 0.");
            return SplitContext.FromParent(ary, ax).SplitBySections(indices_or_sections);
        }

        /// <summary>
        ///     NumPy's <c>numpy.array_split(ary, indices_or_sections, axis)</c> for split points, after the facade's class
        ///     rules and null check: one view per span of <c>[0, indices..., Ntotal]</c> along the axis, each point clamped as
        ///     a Python slice bound (a negative point counts from the end, a point past the axis is its length), and a span
        ///     that runs backwards empty.
        /// </summary>
        /// <param name="ary">The array to divide (not null).</param>
        /// <param name="indices">The split points along the axis, in the caller's order: nothing is sorted, so unsorted
        ///     points give empty views, as NumPy's slices do.</param>
        /// <param name="axis">The axis to split along; negative counts from the end.</param>
        /// <returns><c>indices.Length + 1</c> views of <paramref name="ary"/>'s buffer (writing through, keeping the buffer
        ///     alive, read-only when the operand is), plain arrays whatever <paramref name="ary"/>'s class.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="indices"/> is null.</exception>
        /// <exception cref="IndexError"><paramref name="axis"/> is out of range ("tuple index out of range"; a 0-d array
        ///     refuses every axis).</exception>
        /// <remarks>No class boundary runs here: np.array_split, and np.split / np.hsplit / np.vsplit / np.dsplit through
        ///     <see cref="split(NDArray, long[], int)"/>, apply their rules before they reach this.</remarks>
        internal static NDArray[] array_split(NDArray ary, long[] indices, int axis)
        {
            // NumSharp's own guard - NumPy's single indices_or_sections argument reads a None as a count (int(None)'s
            // TypeError). It lives here because every split-points route reaches this member (np.split and the h/v/d
            // splits after their rank check), after the facade's `ary` check and before the axis: the order the
            // np.array_split facade has always checked them in.
            if (indices is null) throw new ArgumentNullException(nameof(indices));

            int ax = NormalizeSplitAxis(axis, ary.ndim);
            return SplitContext.FromParent(ary, ax).SplitIndices(indices);
        }

        /// <summary>
        ///     <see cref="array_split(NDArray, long[], int)"/> for <see cref="int"/> split points: the same spans and clamping,
        ///     each point widened as it is read (no <see cref="long"/> copy of the list is made).
        /// </summary>
        /// <param name="ary">The array to divide (not null).</param>
        /// <param name="indices">The split points along the axis, in the caller's order (unsorted points give empty
        ///     views).</param>
        /// <param name="axis">The axis to split along; negative counts from the end.</param>
        /// <returns><c>indices.Length + 1</c> views of <paramref name="ary"/>'s buffer (writing through, keeping the buffer
        ///     alive, read-only when the operand is), plain arrays whatever <paramref name="ary"/>'s class.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="indices"/> is null.</exception>
        /// <exception cref="IndexError"><paramref name="axis"/> is out of range ("tuple index out of range"; a 0-d array
        ///     refuses every axis).</exception>
        /// <remarks>No class boundary runs here: np.array_split, and np.split / np.hsplit / np.vsplit / np.dsplit through
        ///     <see cref="split(NDArray, int[], int)"/>, apply their rules before they reach this.</remarks>
        internal static NDArray[] array_split(NDArray ary, int[] indices, int axis)
        {
            // NumSharp's own guard, before the axis: see the long[] overload.
            if (indices is null) throw new ArgumentNullException(nameof(indices));

            int ax = NormalizeSplitAxis(axis, ary.ndim);
            return SplitContext.FromParent(ary, ax).SplitIndices(indices);
        }

        /// <summary>
        ///     NumPy's <c>numpy.split(ary, indices_or_sections, axis)</c> for a section count, after the facade's class rules
        ///     and null check: <c>N = ary.shape[axis]</c> first, then the count must divide the axis, then
        ///     <paramref name="indices_or_sections"/> equal views.
        /// </summary>
        /// <param name="ary">The array to divide (not null).</param>
        /// <param name="indices_or_sections">The number of equal sections; it must divide the axis' length.</param>
        /// <param name="axis">The axis to split along; negative counts from the end.</param>
        /// <returns>The equal sections, each a view of <paramref name="ary"/>'s buffer (writing through, keeping the buffer
        ///     alive, read-only when the operand is), plain arrays whatever <paramref name="ary"/>'s class.</returns>
        /// <exception cref="IndexError"><paramref name="axis"/> is out of range ("tuple index out of range"; a 0-d array
        ///     refuses every axis).</exception>
        /// <exception cref="ValueError"><paramref name="indices_or_sections"/> is not positive ("number sections must be
        ///     larger than 0.", checked before the division), or does not divide the axis ("array split does not result
        ///     in an equal division").</exception>
        /// <remarks>No class boundary runs here: the facades (np.split, np.hsplit, np.vsplit, np.dsplit) apply their rules
        ///     before they reach this.</remarks>
        internal static NDArray[] split(NDArray ary, int indices_or_sections, int axis)
        {
            // `N = ary.shape[axis]` is NumPy's first statement: the axis is checked before the sections.
            int ax = NormalizeSplitAxis(axis, ary.ndim);

            // array_split's argument validation handles section<=0; we validate here
            // so the equal-division check below doesn't divide by zero (NumPy: raw
            // int % 0 throws ZeroDivisionError, but array_split's ValueError is clearer
            // and is what NumPy raises for the same count one call later).
            if (indices_or_sections <= 0)
                throw new ValueError("number sections must be larger than 0.");

            long N = ary.Shape.dimensions[ax];
            if (N % indices_or_sections != 0)
                throw new ValueError("array split does not result in an equal division");

            return SplitContext.FromParent(ary, ax).SplitBySections(indices_or_sections);
        }

        /// <summary>
        ///     NumPy's <c>numpy.split(ary, indices_or_sections, axis)</c> for split points: there is no division to check, so
        ///     it is <see cref="array_split(NDArray, long[], int)"/>, as NumPy's split ends in array_split.
        /// </summary>
        /// <param name="ary">The array to divide (not null).</param>
        /// <param name="indices">The split points along the axis (unsorted points give empty views).</param>
        /// <param name="axis">The axis to split along; negative counts from the end.</param>
        /// <returns><c>indices.Length + 1</c> views of <paramref name="ary"/>'s buffer, plain arrays.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="indices"/> is null (array_split's guard).</exception>
        /// <exception cref="IndexError"><paramref name="axis"/> is out of range ("tuple index out of range").</exception>
        /// <remarks>No class boundary runs here: the facades apply their rules before they reach this.</remarks>
        internal static NDArray[] split(NDArray ary, long[] indices, int axis) => array_split(ary, indices, axis);

        /// <summary>
        ///     NumPy's <c>numpy.split(ary, indices_or_sections, axis)</c> for <see cref="int"/> split points: it is
        ///     <see cref="array_split(NDArray, int[], int)"/>, as NumPy's split ends in array_split.
        /// </summary>
        /// <param name="ary">The array to divide (not null).</param>
        /// <param name="indices">The split points along the axis (unsorted points give empty views).</param>
        /// <param name="axis">The axis to split along; negative counts from the end.</param>
        /// <returns><c>indices.Length + 1</c> views of <paramref name="ary"/>'s buffer, plain arrays.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="indices"/> is null (array_split's guard).</exception>
        /// <exception cref="IndexError"><paramref name="axis"/> is out of range ("tuple index out of range").</exception>
        /// <remarks>No class boundary runs here: the facades apply their rules before they reach this.</remarks>
        internal static NDArray[] split(NDArray ary, int[] indices, int axis) => array_split(ary, indices, axis);

        /// <summary>
        ///     The split family's axis: NumPy reads <c>ary.shape[axis]</c> before anything else, a Python tuple index, so an
        ///     axis outside <c>[-ndim, ndim)</c> is the IndexError Python raises there — not an AxisError (NumPy never
        ///     normalizes it first). A 0-d array has the empty shape, which no index fits.
        /// </summary>
        /// <param name="axis">The axis; negative values count from the end.</param>
        /// <param name="ndim">The array's rank.</param>
        /// <returns>The non-negative axis.</returns>
        /// <exception cref="IndexError"><paramref name="axis"/> is out of range ("tuple index out of range").</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static int NormalizeSplitAxis(int axis, int ndim)
        {
            if (axis < -ndim || axis >= ndim)
                throw new IndexError("tuple index out of range");
            return axis < 0 ? axis + ndim : axis;
        }

        /// <summary>
        ///     Shared state for one split call. Snapshots the parent's shape /
        ///     storage / engine / per-axis derivation once and exposes
        ///     <see cref="SplitBySections"/> for the integer-section paths
        ///     (<c>split(a, N)</c> / <c>array_split(a, N)</c>) and
        ///     <see cref="SplitIndices(long[])"/> / <see cref="SplitIndices(int[])"/>
        ///     for the explicit-indices paths
        ///     (<c>split(a, [3,5,6])</c> / <c>array_split(a, [3,5,6])</c>).
        ///     Naming note: "Sections" refers to NumPy's
        ///     <c>indices_or_sections</c> integer mode — NOT a dtype. Split is
        ///     dtype-agnostic; only views are produced, no element loop runs.
        ///     Each per-sub-array call:
        ///     <list type="bullet">
        ///         <item>Reuses a shared dims[] when the previous sub had the same length on the split axis (typical case for int sections — at most 2 distinct lengths)</item>
        ///         <item>Derives sub flags once per distinct sub-length via <see cref="DeriveSubFlags"/> — an O(ndim) recompute with the canonical Shape walks (parent-flag algebra cannot express NumPy's size-1 contiguity relaxation)</item>
        ///         <item>Derives sub size in O(1) from <c>parent.size * subLen / parent.dim[axis]</c></item>
        ///         <item>Constructs Shape through the no-walk ctor</item>
        ///     </list>
        ///     This pulls the per-sub-array cost from ~640ns (Shape + Alias + NDArray)
        ///     to ~540ns on a 1-D arange(1000) → 4 split benchmark.
        /// </summary>
        /// <remarks>Sub-arrays sharing a length share one dims[] array: safe only because nothing writes into a Shape's
        ///     dimensions after construction (the Shape sharing rule), so a caller that did would change every sibling.</remarks>
        [NDBorrowed /* the array being split belongs to the caller; the pieces are views yielded to them */]
        private readonly struct SplitContext
        {
            /// <summary>The array being split: its storage is aliased by every piece.</summary>
            private readonly NDArray _ary;
            /// <summary>The parent's dimensions, copied into each distinct piece length's dims[].</summary>
            private readonly long[] _srcDims;
            /// <summary>The parent's strides, shared by every piece (a split keeps the layout).</summary>
            private readonly long[] _srcStrides;
            /// <summary>The parent's stride along the split axis: a piece's offset advances by it per element.</summary>
            private readonly long _axisStride;
            /// <summary>The parent's offset into its buffer: the first piece starts here.</summary>
            private readonly long _baseOffset;
            /// <summary>The parent's buffer size (its element count when the shape reports none), given to every piece.</summary>
            private readonly long _bufSize;
            /// <summary>The parent's length along the split axis (parent.dimensions[axis]).</summary>
            private readonly long _axisDim;
            /// <summary>One slab's element count along the axis (parent.size / axisDim; 0 for an empty axis).</summary>
            private readonly long _otherDimsProduct;
            /// <summary>The parent's engine, handed to every piece without re-resolving it.</summary>
            private readonly TensorEngine _engine;
            /// <summary>The parent's rank.</summary>
            private readonly int _ndim;
            /// <summary>The normalized split axis.</summary>
            private readonly int _axis;
            /// <summary>The parent's cached <see cref="ArrayFlags"/>, read for its WRITEABLE bit.</summary>
            private readonly int _parentFlags;

            /// <summary>Snapshots <paramref name="ary"/>'s layout for splitting along <paramref name="axis"/>.</summary>
            /// <param name="ary">The array to split (not null).</param>
            /// <param name="axis">The normalized split axis (<c>0 .. ndim - 1</c>; not checked here).</param>
            private SplitContext(NDArray ary, int axis)
            {
                _ary = ary;
                _engine = ary.TensorEngine;
                var shp = ary.Shape;
                _srcDims = shp.dimensions;
                _srcStrides = shp.strides;
                _axisStride = _srcStrides[axis];
                _baseOffset = shp.offset;
                _bufSize = shp.bufferSize > 0 ? shp.bufferSize : shp.size;
                _ndim = _srcDims.Length;
                _axis = axis;
                _axisDim = _srcDims[axis];
                _parentFlags = shp._flags;
                // Per-slab size: parent.size / axisDim. For axisDim==0 the parent is
                // empty; we keep otherDimsProduct=0 so sub-size collapses to 0.
                _otherDimsProduct = _axisDim == 0 ? 0 : shp.size / _axisDim;
            }

            /// <summary>The split state of <paramref name="ary"/> along <paramref name="axis"/>.</summary>
            /// <param name="ary">The array to split (not null).</param>
            /// <param name="axis">The normalized split axis (<c>0 .. ndim - 1</c>; the callers normalize it with
            ///     <see cref="NormalizeSplitAxis"/>, nothing is checked here).</param>
            /// <returns>The snapshot the split methods read.</returns>
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            internal static SplitContext FromParent(NDArray ary, int axis) => new SplitContext(ary, axis);

            /// <summary>
            ///     Equal-section split (NumPy <c>split(a, N)</c> /
            ///     <c>array_split(a, N)</c> when <c>indices_or_sections</c> is an
            ///     integer). <c>extras = N % Nsections</c> sub-arrays get size
            ///     <c>N/Nsections + 1</c>, the rest get <c>N/Nsections</c>.
            ///     Only TWO distinct sub-lengths exist — we materialise dims[]
            ///     for each and share across all subs with the matching length.
            /// </summary>
            /// <param name="Nsections">Number of sections to split into. Must be &gt; 0 (the callers check; zero would
            ///     divide by zero here).</param>
            /// <returns><paramref name="Nsections"/> views, the longer ones first.</returns>
            internal NDArray[] SplitBySections(int Nsections)
            {
                long Neach = _axisDim / Nsections;
                long extras = _axisDim % Nsections;

                long[] dimsLarge = null;   // for subs of size Neach+1
                long[] dimsSmall = null;   // for subs of size Neach
                // Pre-compute the (flags, size, hash) tuple for both sub-lengths once.
                int flagsLarge = 0, flagsSmall = 0;
                long sizeLarge = 0, sizeSmall = 0;
                int hashLarge = 0, hashSmall = 0;
                if (extras > 0)
                {
                    dimsLarge = BuildSubDims(Neach + 1);
                    flagsLarge = DeriveSubFlags(dimsLarge);
                    sizeLarge = _otherDimsProduct * (Neach + 1);
                    hashLarge = ComputeHashFromDims(dimsLarge, sizeLarge);
                }
                if (Nsections > extras)
                {
                    dimsSmall = BuildSubDims(Neach);
                    flagsSmall = DeriveSubFlags(dimsSmall);
                    sizeSmall = _otherDimsProduct * Neach;
                    hashSmall = ComputeHashFromDims(dimsSmall, sizeSmall);
                }

                var sub_arys = new NDArray[Nsections];
                long cursor = 0;
                for (int i = 0; i < Nsections; i++)
                {
                    bool isLarge = i < extras;
                    long size = isLarge ? Neach + 1 : Neach;
                    sub_arys[i] = BuildView(
                        isLarge ? dimsLarge : dimsSmall,
                        isLarge ? flagsLarge : flagsSmall,
                        isLarge ? sizeLarge : sizeSmall,
                        isLarge ? hashLarge : hashSmall,
                        cursor);
                    cursor += size;
                }

                return sub_arys;
            }

            /// <summary>
            ///     <see cref="SplitIndices(int[])"/> for <see cref="long"/> split points: the same two-cursor walk over
            ///     <c>0, indices[0..^1], Ntotal</c> with the same per-length cache.
            /// </summary>
            /// <param name="indices">The split points (not null), clamped as Python slice bounds.</param>
            /// <returns><c>indices.Length + 1</c> views; a span that runs backwards is empty.</returns>
            internal NDArray[] SplitIndices(long[] indices)
            {
                int Nsections = indices.Length + 1;
                var sub_arys = new NDArray[Nsections];
                long prev = 0;
                long lastSubLen = -1;
                long[] cachedDims = null;
                int cachedFlags = 0;
                long cachedSize = 0;
                int cachedHash = 0;

                for (int i = 0; i < Nsections; i++)
                {
                    long raw = (i == indices.Length) ? _axisDim : indices[i];
                    long cur = ClampSlicePoint(raw, _axisDim);
                    long st = prev;
                    long end = cur < st ? st : cur;
                    long subLen = end - st;

                    // Cache the dims/flags/size/hash quadruple keyed on subLen so
                    // adjacent same-length sub-arrays share allocations (common for
                    // even indices like [3,6,9] or repeated indices).
                    if (subLen != lastSubLen)
                    {
                        cachedDims = BuildSubDims(subLen);
                        cachedFlags = DeriveSubFlags(cachedDims);
                        cachedSize = _otherDimsProduct * subLen;
                        cachedHash = ComputeHashFromDims(cachedDims, cachedSize);
                        lastSubLen = subLen;
                    }
                    sub_arys[i] = BuildView(cachedDims, cachedFlags, cachedSize, cachedHash, st);
                    prev = cur;
                }
                return sub_arys;
            }

            /// <summary>
            ///     Indices-mode split that walks the indices array directly without
            ///     allocating a div_points scratch buffer. Walks the boundary list
            ///     <c>0, indices[0..^1], Ntotal</c> with two cursors (prev, cur).
            ///     Caches the most recently used (dims, flags, size, hash) tuple so
            ///     repeated same-length sub-arrays don't realloc.
            /// </summary>
            /// <param name="indices">The split points (not null), clamped as Python slice bounds.</param>
            /// <returns><c>indices.Length + 1</c> views; a span that runs backwards is empty.</returns>
            internal NDArray[] SplitIndices(int[] indices)
            {
                int Nsections = indices.Length + 1;
                var sub_arys = new NDArray[Nsections];
                long prev = 0;
                long lastSubLen = -1;
                long[] cachedDims = null;
                int cachedFlags = 0;
                long cachedSize = 0;
                int cachedHash = 0;

                for (int i = 0; i < Nsections; i++)
                {
                    long raw = (i == indices.Length) ? _axisDim : indices[i];
                    long cur = ClampSlicePoint(raw, _axisDim);
                    long st = prev;
                    long end = cur < st ? st : cur;
                    long subLen = end - st;

                    if (subLen != lastSubLen)
                    {
                        cachedDims = BuildSubDims(subLen);
                        cachedFlags = DeriveSubFlags(cachedDims);
                        cachedSize = _otherDimsProduct * subLen;
                        cachedHash = ComputeHashFromDims(cachedDims, cachedSize);
                        lastSubLen = subLen;
                    }
                    sub_arys[i] = BuildView(cachedDims, cachedFlags, cachedSize, cachedHash, st);
                    prev = cur;
                }
                return sub_arys;
            }

            /// <summary>
            ///     Build the sub-array's dims[] by cloning parent's dims and patching
            ///     <c>dims[axis] = subLen</c>. Shape stores dims by reference so
            ///     callers can share this array across same-length sub-arrays.
            /// </summary>
            /// <param name="subLen">The piece's length along the split axis.</param>
            /// <returns>A new dims[] (never the parent's own array).</returns>
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            private long[] BuildSubDims(long subLen)
            {
                var dims = new long[_ndim];
                // System.Array: inside NDArray the bare name binds the array object's Array property.
                System.Array.Copy(_srcDims, dims, _ndim);
                dims[_axis] = subLen;
                return dims;
            }

            /// <summary>
            ///     Build a sub-array NDArray view from a pre-computed (dims, flags,
            ///     size, hash) tuple and a start offset (in elements) along the
            ///     split axis. The Shape uses the no-walk ctor.
            /// </summary>
            /// <param name="dims">The piece's dimensions (possibly shared with same-length siblings).</param>
            /// <param name="flags">The piece's <see cref="ArrayFlags"/> from <see cref="DeriveSubFlags"/>.</param>
            /// <param name="size">The piece's element count.</param>
            /// <param name="hash">The piece's Shape hash from <see cref="ComputeHashFromDims"/>.</param>
            /// <param name="startElems">The piece's first index along the split axis.</param>
            /// <returns>A new view over the parent's storage (it holds its own reference on the buffer).</returns>
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            private NDArray BuildView(long[] dims, int flags, long size, int hash, long startElems)
            {
                long newOffset = _baseOffset + startElems * _axisStride;
                var newShape = new Shape(dims, _srcStrides, newOffset, _bufSize, flags, size, hash);
                // Hot-path ctor: pre-resolved engine, no `?? BackendFactory.GetEngine()`
                // guard, no `Storage.Engine = tensorEngine` writeback.
                return new NDArray(_ary.Storage.Alias(newShape), _engine, skipEngineResolve: true);
            }

            /// <summary>
            ///     Derive sub-array <see cref="ArrayFlags"/> from parent flags in O(1).
            ///     <list type="bullet">
            ///       <item>Empty sub (any dim==0): NumPy convention → both C and F contig, WRITEABLE, ALIGNED, no BROADCASTED.</item>
            ///       <item>Parent broadcasted: sub inherits BROADCASTED + !WRITEABLE (its stride-0 axes are untouched).</item>
            ///       <item>Parent C-contig: sub stays C-contig iff <c>axis==0</c> or <c>subLen == parent.dim[axis]</c>; otherwise the <c>strides[i] == dims[i+1]*strides[i+1]</c> invariant breaks at <c>i = axis-1</c>.</item>
            ///       <item>Parent F-contig: symmetric — stays F-contig iff <c>axis==ndim-1</c> or <c>subLen == parent.dim[axis]</c>.</item>
            ///       <item>WRITEABLE inherits from parent: read-only views (e.g. <c>np.diagonal</c> output) propagate their non-writeable status to sub-arrays.</item>
            ///     </list>
            /// </summary>
            /// <param name="subDims">The piece's dimensions.</param>
            /// <returns>The piece's flag bits (ALIGNED always; WRITEABLE from the parent unless broadcast).</returns>
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            private int DeriveSubFlags(long[] subDims)
            {
                // NumPy convention: any 0-dim → both C and F contig (vacuously),
                // ALIGNED. WRITEABLE inherits from parent (so a read-only view's
                // empty sub stays read-only).
                int parentWriteable = _parentFlags & (int)ArrayFlags.WRITEABLE;
                for (int i = 0; i < subDims.Length; i++)
                {
                    if (subDims[i] == 0)
                        return (int)(ArrayFlags.C_CONTIGUOUS | ArrayFlags.F_CONTIGUOUS
                                    | ArrayFlags.ALIGNED) | parentWriteable;
                }

                // Broadcastness is RECOMPUTED for the child's dims (same rule as the walking Shape
                // ctor): collapsing the split axis to length 1 ENDS a broadcast running along it —
                // a stride-0 axis of size 1 is not broadcast — after which contiguity is judged
                // normally. BROADCASTED implies non-writeable per NumPy, and a child of a read-only
                // broadcast stays read-only on the non-broadcast path too (parentWriteable is 0).
                if (Shape.ComputeIsBroadcastedStatic(subDims, _srcStrides))
                    return (int)(ArrayFlags.BROADCASTED | ArrayFlags.ALIGNED);

                // Contiguity is RECOMPUTED with NumPy's _UpdateContiguousFlags walk over the
                // child's dims (size-1 dims skipped) — parent-flag algebra cannot express the
                // relaxation. Probed against NumPy 2.4.2: np.split of a C-contiguous (3,4) into
                // three (1,4) children gives C1 F1 num=259 (the collapsed axis stops mattering,
                // making the child BOTH-contiguous), and a (1,6) parent split on axis 1 keeps its
                // (1,3) children C-contiguous — both were mis-derived by the old
                // parent-C/parent-F × axis-position inference.
                var (isC, isF) = Shape.ComputeContiguousFlagsStatic(subDims, _srcStrides);
                int flags = (int)ArrayFlags.ALIGNED | parentWriteable;
                if (isC)
                    flags |= (int)ArrayFlags.C_CONTIGUOUS;
                if (isF)
                    flags |= (int)ArrayFlags.F_CONTIGUOUS;
                return flags;
            }

            /// <summary>
            ///     Reconstruct Shape's standard hash for a sub-dims[] array. Mirrors
            ///     <c>ComputeSizeAndHash</c> exactly — same seed (<c>'C' * 397</c>),
            ///     same XOR formula — so Shape.GetHashCode stays consistent and
            ///     IsEmpty (which checks <c>_hashCode == 0</c>) never misfires.
            /// </summary>
            /// <param name="dims">The piece's dimensions.</param>
            /// <param name="finalSize">The piece's element count (unused by the fold, which recomputes the running
            ///     product).</param>
            /// <returns>The hash the walking Shape constructor would compute; <see cref="int.MinValue"/> for no
            ///     dimensions.</returns>
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            private static int ComputeHashFromDims(long[] dims, long finalSize)
            {
                if (dims == null || dims.Length == 0)
                    return int.MinValue;
                // Seed must match Shape.ComputeSizeAndHash: `layout * 397` where
                // layout == 'C' (67). Without the non-zero seed, XOR-fold can land
                // on 0 (e.g. dims=[4,2]) and Shape.IsEmpty reads as true.
                int hash = unchecked('C' * 397);
                long size = 1;
                unchecked
                {
                    foreach (var v in dims)
                    {
                        size *= v;
                        hash ^= ((int)(size & 0x7FFFFFFF) * 397) * ((int)(v & 0x7FFFFFFF) * 397);
                    }
                }
                return hash;
            }
        }

        /// <summary>
        ///     NumPy slice clamping: <c>n &lt; 0 ? max(0, n + N) : min(n, N)</c>.
        /// </summary>
        /// <param name="n">The split point as given (negative counts from the end).</param>
        /// <param name="N">The axis' length.</param>
        /// <returns>The point clamped into <c>[0, N]</c>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static long ClampSlicePoint(long n, long N)
        {
            if (n < 0)
            {
                long wrapped = n + N;
                return wrapped < 0 ? 0 : wrapped;
            }
            return n > N ? N : n;
        }

        // ---------------------------- np.hsplit / np.vsplit / np.dsplit ----------------------------

        /// <summary>
        ///     NumPy's <c>numpy.hsplit(ary, indices_or_sections)</c> for a section count, after the facade's class rules: a
        ///     0-d array is refused, then the array splits along axis 1 when it has two dimensions or more and along axis 0
        ///     when it is 1-d (its only, "horizontal" axis).
        /// </summary>
        /// <param name="ary">The array to divide; its rank is read first (np.hsplit passes a null through).</param>
        /// <param name="indices_or_sections">The number of equal sections; it must divide the axis' length.</param>
        /// <returns>The equal sections, views of <paramref name="ary"/>'s buffer (see
        ///     <see cref="split(NDArray, int, int)"/>).</returns>
        /// <exception cref="ArgumentException"><paramref name="ary"/> is 0-d ("hsplit only works on arrays of 1 or more
        ///     dimensions"; NumPy raises the text as a ValueError).</exception>
        /// <exception cref="ValueError">The count is not positive, or does not divide the axis (split's texts).</exception>
        /// <exception cref="NullReferenceException"><paramref name="ary"/> is null.</exception>
        /// <remarks>No class boundary runs here: np.hsplit applies its rules before it calls this.</remarks>
        internal static NDArray[] hsplit(NDArray ary, int indices_or_sections)
        {
            if (ary.ndim == 0)
                throw new ArgumentException("hsplit only works on arrays of 1 or more dimensions");

            int axis = ary.ndim > 1 ? 1 : 0;
            return split(ary, indices_or_sections, axis);
        }

        /// <summary>
        ///     NumPy's <c>numpy.hsplit(ary, indices_or_sections)</c> for split points, after the facade's class rules: a 0-d
        ///     array is refused, then the array splits at <paramref name="indices"/> along axis 1 (two dimensions or more) or
        ///     axis 0 (1-d).
        /// </summary>
        /// <param name="ary">The array to divide; its rank is read first (np.hsplit passes a null through).</param>
        /// <param name="indices">The split points along the axis: <c>[2, 3]</c> gives <c>ary[:, :2]</c>,
        ///     <c>ary[:, 2:3]</c>, <c>ary[:, 3:]</c> (unsorted points give empty views).</param>
        /// <returns><c>indices.Length + 1</c> views of <paramref name="ary"/>'s buffer, plain arrays.</returns>
        /// <exception cref="ArgumentException"><paramref name="ary"/> is 0-d ("hsplit only works on arrays of 1 or more
        ///     dimensions"; NumPy raises the text as a ValueError).</exception>
        /// <exception cref="ArgumentNullException"><paramref name="indices"/> is null (array_split's guard, after the rank
        ///     check).</exception>
        /// <exception cref="NullReferenceException"><paramref name="ary"/> is null.</exception>
        /// <remarks>No class boundary runs here: np.hsplit applies its rules before it calls this.</remarks>
        internal static NDArray[] hsplit(NDArray ary, int[] indices)
        {
            if (ary.ndim == 0)
                throw new ArgumentException("hsplit only works on arrays of 1 or more dimensions");

            int axis = ary.ndim > 1 ? 1 : 0;
            return split(ary, indices, axis);
        }

        /// <summary>
        ///     NumPy's <c>numpy.vsplit(ary, indices_or_sections)</c> for a section count, after the facade's class rules: an
        ///     array below two dimensions is refused, then the array splits along axis 0 (row-wise).
        /// </summary>
        /// <param name="ary">The array to divide; its rank is read first (np.vsplit passes a null through).</param>
        /// <param name="indices_or_sections">The number of equal sections; it must divide the first axis' length.</param>
        /// <returns>The equal sections, views of <paramref name="ary"/>'s buffer, plain arrays.</returns>
        /// <exception cref="ArgumentException"><paramref name="ary"/> has fewer than two dimensions ("vsplit only works on
        ///     arrays of 2 or more dimensions"; NumPy raises the text as a ValueError).</exception>
        /// <exception cref="ValueError">The count is not positive, or does not divide the axis (split's texts).</exception>
        /// <exception cref="NullReferenceException"><paramref name="ary"/> is null.</exception>
        /// <remarks>No class boundary runs here: np.vsplit applies its rules before it calls this.</remarks>
        internal static NDArray[] vsplit(NDArray ary, int indices_or_sections)
        {
            if (ary.ndim < 2)
                throw new ArgumentException("vsplit only works on arrays of 2 or more dimensions");

            return split(ary, indices_or_sections, axis: 0);
        }

        /// <summary>
        ///     NumPy's <c>numpy.vsplit(ary, indices_or_sections)</c> for split points, after the facade's class rules: an array
        ///     below two dimensions is refused, then the array splits at <paramref name="indices"/> along axis 0.
        /// </summary>
        /// <param name="ary">The array to divide; its rank is read first (np.vsplit passes a null through).</param>
        /// <param name="indices">The split points along axis 0: <c>[2, 3]</c> gives <c>ary[:2]</c>, <c>ary[2:3]</c>,
        ///     <c>ary[3:]</c> (unsorted points give empty views).</param>
        /// <returns><c>indices.Length + 1</c> views of <paramref name="ary"/>'s buffer, plain arrays.</returns>
        /// <exception cref="ArgumentException"><paramref name="ary"/> has fewer than two dimensions ("vsplit only works on
        ///     arrays of 2 or more dimensions"; NumPy raises the text as a ValueError).</exception>
        /// <exception cref="ArgumentNullException"><paramref name="indices"/> is null (array_split's guard, after the rank
        ///     check).</exception>
        /// <exception cref="NullReferenceException"><paramref name="ary"/> is null.</exception>
        /// <remarks>No class boundary runs here: np.vsplit applies its rules before it calls this.</remarks>
        internal static NDArray[] vsplit(NDArray ary, int[] indices)
        {
            if (ary.ndim < 2)
                throw new ArgumentException("vsplit only works on arrays of 2 or more dimensions");

            return split(ary, indices, axis: 0);
        }

        /// <summary>
        ///     NumPy's <c>numpy.dsplit(ary, indices_or_sections)</c> for a section count, after the facade's class rules: an
        ///     array below three dimensions is refused, then the array splits along axis 2 (depth-wise).
        /// </summary>
        /// <param name="ary">The array to divide; its rank is read first (np.dsplit passes a null through).</param>
        /// <param name="indices_or_sections">The number of equal sections; it must divide the third axis' length.</param>
        /// <returns>The equal sections, views of <paramref name="ary"/>'s buffer, plain arrays.</returns>
        /// <exception cref="ArgumentException"><paramref name="ary"/> has fewer than three dimensions ("dsplit only works on
        ///     arrays of 3 or more dimensions"; NumPy raises the text as a ValueError).</exception>
        /// <exception cref="ValueError">The count is not positive, or does not divide the axis (split's texts).</exception>
        /// <exception cref="NullReferenceException"><paramref name="ary"/> is null.</exception>
        /// <remarks>No class boundary runs here: np.dsplit applies its rules before it calls this.</remarks>
        internal static NDArray[] dsplit(NDArray ary, int indices_or_sections)
        {
            if (ary.ndim < 3)
                throw new ArgumentException("dsplit only works on arrays of 3 or more dimensions");

            return split(ary, indices_or_sections, axis: 2);
        }

        /// <summary>
        ///     NumPy's <c>numpy.dsplit(ary, indices_or_sections)</c> for split points, after the facade's class rules: an array
        ///     below three dimensions is refused, then the array splits at <paramref name="indices"/> along axis 2.
        /// </summary>
        /// <param name="ary">The array to divide; its rank is read first (np.dsplit passes a null through).</param>
        /// <param name="indices">The split points along axis 2: <c>[2, 3]</c> gives <c>ary[:, :, :2]</c>,
        ///     <c>ary[:, :, 2:3]</c>, <c>ary[:, :, 3:]</c> (unsorted points give empty views).</param>
        /// <returns><c>indices.Length + 1</c> views of <paramref name="ary"/>'s buffer, plain arrays.</returns>
        /// <exception cref="ArgumentException"><paramref name="ary"/> has fewer than three dimensions ("dsplit only works on
        ///     arrays of 3 or more dimensions"; NumPy raises the text as a ValueError).</exception>
        /// <exception cref="ArgumentNullException"><paramref name="indices"/> is null (array_split's guard, after the rank
        ///     check).</exception>
        /// <exception cref="NullReferenceException"><paramref name="ary"/> is null.</exception>
        /// <remarks>No class boundary runs here: np.dsplit applies its rules before it calls this.</remarks>
        internal static NDArray[] dsplit(NDArray ary, int[] indices)
        {
            if (ary.ndim < 3)
                throw new ArgumentException("dsplit only works on arrays of 3 or more dimensions");

            return split(ary, indices, axis: 2);
        }

        // ---------------------------- np.tile ----------------------------

        /// <summary>
        ///     NumPy's <c>numpy.tile(A, reps)</c> after the facade's class rules and null checks: <paramref name="A"/>'s shape
        ///     and <paramref name="reps"/> are padded with leading 1s to a common rank, all-ones repetitions give a keep-order
        ///     copy, an empty result a zero-element array, and anything else the interleaved reshape - broadcast - copy -
        ///     reshape that lays the repetitions out in C order.
        /// </summary>
        /// <param name="A">The array to repeat (not null).</param>
        /// <param name="reps">The repetitions along each axis (not null, every entry non-negative): fewer entries than
        ///     <paramref name="A"/>'s rank repeat its trailing axes, more promote <paramref name="A"/> by leading axes.</param>
        /// <returns>The tiled array, of rank <c>max(reps.Length, A.ndim)</c> and <paramref name="A"/>'s dtype, plain whatever
        ///     <paramref name="A"/>'s class: a fresh C-contiguous array for a real repetition; for all-ones repetitions a
        ///     keep-order copy that owns its data (an F-contiguous <paramref name="A"/> stays F-contiguous); for an empty
        ///     result a view - of <paramref name="A"/>'s own buffer when <paramref name="A"/> is empty, of a fresh
        ///     zero-filled buffer otherwise (owndata false either way, as NumPy's final reshape gives).</returns>
        /// <exception cref="ArgumentException">An entry of the padded repetitions is negative (<c>reps[i] must be
        ///     non-negative, got r.</c>, parameter <c>reps</c>). The index counts the PADDED list: a <paramref name="reps"/>
        ///     shorter than <paramref name="A"/>'s rank reports an index past its own length.</exception>
        /// <remarks>
        ///     Scoped by <c>[NDScoped]</c>: the interleaved reshape of <paramref name="A"/>, its broadcast view and the
        ///     materialized copy are released on return, and the final reshape (a view that holds its own reference on the
        ///     copy's buffer) is yielded; the all-ones path releases the reshape of a lower-rank <paramref name="A"/>, and the
        ///     empty path the zero-filled array its returned view aliases. No class boundary runs here: both np.tile
        ///     overloads apply their rules before they call this.
        /// </remarks>
        [NDScoped /* the interleaved reshape, its broadcast view, the materialized copy and the empty path's zeros are released; the result is yielded */]
        internal static NDArray tile(NDArray A, long[] reps)
        {
            int d = reps.Length;
            int aDim = A.ndim;
            int outDim = Math.Max(d, aDim);

            // Pad A's shape with leading 1s when reps has more entries than A.ndim.
            // Pad reps with leading 1s when A.ndim is larger than reps' length.
            // Both yield a common ndim = max(d, aDim) where in[i] aligns with rep[i].
            var aShape = new long[outDim];
            var tup = new long[outDim];
            for (int i = 0; i < outDim - aDim; i++) aShape[i] = 1;
            for (int i = 0; i < aDim; i++) aShape[outDim - aDim + i] = A.Shape.dimensions[i];
            for (int i = 0; i < outDim - d; i++) tup[i] = 1;
            for (int i = 0; i < d; i++) tup[outDim - d + i] = reps[i];

            for (int i = 0; i < outDim; i++)
                if (tup[i] < 0)
                    throw new ArgumentException($"reps[{i}] must be non-negative, got {tup[i]}.", nameof(reps));

            // Compute output shape.
            var outShape = new long[outDim];
            long outSize = 1;
            for (int i = 0; i < outDim; i++)
            {
                outShape[i] = aShape[i] * tup[i];
                outSize *= outShape[i];
            }

            // Trivial case: all reps are 1 → return a keep-order copy preserving the
            // (possibly promoted) shape. Matches NumPy's array(A, copy=True, ndmin=d)
            // shortcut: F-contiguous inputs stay F-contiguous, other views materialize as C.
            // NumPy tests the all-ones reps FIRST, so an EMPTY input with all-ones reps is this owning copy too (probed
            // 2.4.2: np.tile(np.zeros((0, 3)), (1, 1)).flags.owndata is True), not the view the empty branch below makes.
            bool allOnes = true;
            for (int i = 0; i < outDim; i++) if (tup[i] != 1) { allOnes = false; break; }
            if (allOnes)
            {
                var c = aDim == outDim ? A.copy('K') : A.reshape(Shape.Adopt(aShape)).copy('K');
                return c;
            }

            // Empty result: any rep==0 or any aShape[i]==0 → a zero-element array of the correct shape and dtype.
            // NumPy ends tile with `c.reshape(shape_out)`: for an EMPTY input c is A itself (or its ndmin view), so the
            // result is a view of A; for a zero repetition of a non-empty input c is the fresh repeat, so the result is a
            // view of a fresh buffer — owndata false either way.
            if (outSize == 0)
            {
                if (A.size == 0)
                {
                    var c = aDim == outDim ? A : A.reshape(Shape.Adopt(aShape));
                    return c.reshape(Shape.Adopt(outShape));
                }
                // np.zeros(shape, dtype)'s statement: a zero-filled allocation of A's dtype.
                var fresh = new NDArray(A.dtype, Shape.Adopt(outShape), true);
                return new NDArray(fresh.Storage.Alias(fresh.Shape)) { TensorEngine = A.TensorEngine };
            }

            // General case: insert size-1 axes between A's axes to create a tile axis next to each
            // input axis, then broadcast and copy to materialize, then collapse.
            //
            //   A.shape (a0, a1, ..., a_{n-1})
            //     ↓ reshape to interleaved (1, a0, 1, a1, ..., 1, a_{n-1})
            //     ↓ broadcast_to (r0, a0, r1, a1, ..., r_{n-1}, a_{n-1})  — each leading 1 expands
            //     ↓ copy() → contiguous (size = product of all)
            //     ↓ reshape to (r0*a0, r1*a1, ..., r_{n-1}*a_{n-1})
            //
            // This composes broadcast + copy + reshape (all O(N)) and produces NumPy-aligned output.
            var interleaved = new long[2 * outDim];
            var broadcastTarget = new long[2 * outDim];
            for (int i = 0; i < outDim; i++)
            {
                interleaved[2 * i] = 1;
                interleaved[2 * i + 1] = aShape[i];
                broadcastTarget[2 * i] = tup[i];
                broadcastTarget[2 * i + 1] = aShape[i];
            }

            // promoted is a new wrapper (view or copy depending on A's contiguity);
            // broadcasted is a stride-0 broadcast view of promoted; contiguous is a
            // fresh materialized copy. All three are intermediates that the final
            // reshape doesn't keep, released by this method's scope — the returned
            // NDArray shares contiguous's storage but holds its own ARC ref via
            // InitializeArc, so releasing contiguous does not free that buffer.
            var promoted = A.reshape(Shape.Adopt(interleaved));
            var broadcasted = broadcast_to(promoted, Shape.Adopt(broadcastTarget));
            var contiguous = broadcasted.copy();
            return contiguous.reshape(Shape.Adopt(outShape));
        }
    }
}
