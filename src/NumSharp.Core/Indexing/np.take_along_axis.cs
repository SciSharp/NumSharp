using System;
using NumSharp.Backends;
using NumSharp.Backends.Kernels;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Take values from the input array by matching 1-D index and data slices.
        ///     Iterates over matching 1-D slices oriented along <paramref name="axis"/> in the
        ///     index and data arrays, using the former to look up values in the latter. These
        ///     slices can be different lengths. Functions returning an index along an axis, like
        ///     <see cref="argsort(NDArray,int?,string)"/> and <c>argmax</c>/<c>argmin</c> with
        ///     <c>keepdims</c>, produce suitable indices.
        /// </summary>
        /// <param name="arr">Source array <c>(Ni..., M, Nk...)</c>.</param>
        /// <param name="indices">
        ///     Integer index array <c>(Ni..., J, Nk...)</c>. Must match the dimension count of
        ///     <paramref name="arr"/>; the non-axis dimensions <c>Ni</c>/<c>Nk</c> only need to
        ///     broadcast against <paramref name="arr"/>.
        /// </param>
        /// <param name="axis">
        ///     The axis to take 1-D slices along (default <c>-1</c>, matching NumPy 2.3+). When
        ///     <c>null</c> the source is treated as if first flattened to 1-D in C-order, for
        ///     consistency with <c>sort</c>/<c>argsort</c>; then <paramref name="indices"/> must
        ///     be 1-D.
        /// </param>
        /// <returns>
        ///     A fresh C-contiguous array of shape <c>(Ni..., J, Nk...)</c> (the broadcast of the
        ///     non-axis dimensions, with <c>J = indices.shape[axis]</c>) and dtype of
        ///     <paramref name="arr"/>.
        /// </returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.take_along_axis.html</remarks>
        public static unsafe NDArray take_along_axis(NDArray arr, NDArray indices, int? axis = -1)
        {
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            if (indices is null) throw new ArgumentNullException(nameof(indices));

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

            var result = new NDArray(arr.typecode, new Shape(resultDims), false);

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
    }
}
