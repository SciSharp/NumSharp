using System;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Put values into the destination array by matching 1-D index and data slices — the
        ///     setter twin of <see cref="take_along_axis(NDArray,NDArray,int?)"/>. Iterates over
        ///     matching 1-D slices oriented along <paramref name="axis"/> in the index and data
        ///     arrays and uses the former to place <paramref name="values"/> into the latter.
        ///     In-place: <paramref name="arr"/> is modified and nothing is returned. Functions
        ///     returning an index along an axis, like <c>argsort</c> and <c>argmax</c>/<c>argmin</c>
        ///     with <c>keepdims</c>, produce suitable indices.
        /// </summary>
        /// <param name="arr">
        ///     Destination array <c>(Ni..., M, Nk...)</c>, modified in place. Must be writeable.
        /// </param>
        /// <param name="indices">
        ///     Integer index array <c>(Ni..., J, Nk...)</c>. Must match the dimension count of
        ///     <paramref name="arr"/>; the non-axis dimensions <c>Ni</c>/<c>Nk</c> need only
        ///     broadcast against <paramref name="arr"/>. <c>J</c> need not equal <c>M</c>.
        /// </param>
        /// <param name="values">
        ///     Values to write, cast to <paramref name="arr"/>'s dtype (assignment cast: floats
        ///     truncate toward zero). Broadcast — NOT cycled — to the indexing result shape
        ///     <c>(Ni..., J, Nk...)</c>: right-aligned, with extra leading size-1 dimensions
        ///     stripped. Where several index positions collapse onto one <paramref name="arr"/>
        ///     element (a size-1 <paramref name="arr"/> dimension, or duplicate indices in a slice),
        ///     the last write in C-order wins.
        /// </param>
        /// <param name="axis">
        ///     The axis to place 1-D slices along (no default — required, as in NumPy). When
        ///     <c>null</c> the destination is treated as its C-order flattening: a <b>C-contiguous</b>
        ///     <paramref name="arr"/> is written back through that flat view (matching NumPy's
        ///     <c>np.array(arr.flat)</c>, which aliases contiguous storage), while a
        ///     <b>non-contiguous</b> <paramref name="arr"/> raises read-only — NumPy's
        ///     <c>np.array(arr.flat)</c> yields a read-only copy there, so the assignment fails
        ///     (probed 2.4.2). <paramref name="indices"/> must then be 1-D.
        /// </param>
        /// <exception cref="ArgumentNullException"><paramref name="arr"/>, <paramref name="indices"/> or <paramref name="values"/> is null.</exception>
        /// <exception cref="ValueError">
        ///     <paramref name="axis"/> is <c>null</c> and <paramref name="indices"/> is not 1-D; the
        ///     dimension counts of <paramref name="arr"/> and <paramref name="indices"/> differ; or
        ///     <paramref name="values"/> cannot broadcast to the indexing result shape.
        /// </exception>
        /// <exception cref="IndexError">
        ///     <paramref name="indices"/> is not an integer array; a non-axis dimension of
        ///     <paramref name="indices"/> conflicts with <paramref name="arr"/>; or an index is out
        ///     of bounds for the axis (after a single negative wrap). On out-of-bounds NOTHING is
        ///     written — every index is validated before the first store (NumPy's atomicity).
        /// </exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range for <paramref name="arr"/>'s rank.</exception>
        /// <exception cref="ValueError"><paramref name="arr"/> (or the non-contiguous <c>axis=null</c> flat copy) is read-only — NumPy's "assignment destination is read-only".</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.put_along_axis.html</remarks>
        [NDScoped]
        public static unsafe void put_along_axis(NDArray arr, NDArray indices, NDArray values, int? axis)
        {
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            if (indices is null) throw new ArgumentNullException(nameof(indices));
            if (values is null) throw new ArgumentNullException(nameof(values));

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

            long elemBytes = arr.dtypesize;

            // Cast values to arr's dtype (assignment cast — floats truncate toward zero; over/underflow
            // wraps modularly, matching NumPy's ndarray-value path and the put/place/putmask siblings).
            // A same-dtype source is used AS-IS: any layout / broadcast dims are read through strides.
            NDArray valArr = values.typecode == arr.typecode ? values : values.astype(arr.typecode);

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

        /// <summary>Format a dimension array as a Python tuple: <c>()</c>, <c>(N,)</c>, <c>(A,B)</c> — matching <see cref="Shape.ToPythonTuple"/>.</summary>
        private static string PyShapeTuple(long[] dims)
        {
            if (dims is null || dims.Length == 0) return "()";
            if (dims.Length == 1) return "(" + dims[0] + ",)";
            return "(" + string.Join(",", dims) + ")";
        }
    }
}
