using System;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's <c>PyArray_Diagonal</c> (item_selection.c): the two strides combine into one,
        ///     <c>stride[axis1] + stride[axis2]</c>, so one step walks the diagonal, and the offset shifts the start along
        ///     <paramref name="axis2"/> (positive) or <paramref name="axis1"/> (negative). No element is read or copied: the
        ///     result aliases <paramref name="a"/>'s storage, READ-ONLY by NumPy's contract. Moved from the public
        ///     <c>np.diagonal</c> by the np-layer dtype plan's P1b (np.trace's engine member reads its diagonal through it,
        ///     which a call up into np.* could not do); the public function keeps the class boundaries.
        /// </remarks>
        public override NDArray Diagonal(NDArray a, int offset = 0, int axis1 = 0, int axis2 = 1)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));

            int ndim = a.ndim;
            // PyArray_Diagonal's ValueError, before either axis is read.
            if (ndim < 2)
                throw new ValueError(
                    "diag requires an array of at least two dimensions",
                    nameof(a));

            // check_and_adjust_axis_msg: the argument name leads the AxisError's text ("axis1: axis ...").
            int ax1 = check_and_adjust_axis(ndim, axis1, nameof(axis1));
            int ax2 = check_and_adjust_axis(ndim, axis2, nameof(axis2));
            // PyArray_Diagonal's ValueError, after both axes are in range.
            if (ax1 == ax2)
                throw new ValueError(
                    "axis1 and axis2 cannot be the same",
                    nameof(axis2));

            var srcShape = a.Shape;
            long dim1 = srcShape.dimensions[ax1];
            long dim2 = srcShape.dimensions[ax2];
            long stride1 = srcShape.strides[ax1];
            long stride2 = srcShape.strides[ax2];

            // NumPy formula: positive offset shifts along axis2 (column direction),
            // negative along axis1 (row direction). We adjust the corresponding
            // dimension and the data offset.
            long offsetStride;
            long offAbs;
            if (offset >= 0)
            {
                offsetStride = stride2;
                dim2 -= offset;
                offAbs = offset;
            }
            else
            {
                offsetStride = stride1;
                dim1 -= -(long)offset;
                offAbs = -(long)offset;
            }

            long diagSize = Math.Min(dim1, dim2);
            if (diagSize < 0)
                diagSize = 0;

            // Element size for byte→element offset conversion. NumSharp strides are
            // in elements (not bytes); the source offset added below stays in
            // elements.
            long newOffset = srcShape.offset;
            if (diagSize > 0)
                newOffset += offAbs * offsetStride;

            // Build new shape and strides: remove axis1 and axis2 then append
            // (diagSize, stride1 + stride2) as the last axis.
            int outNdim = ndim - 1;
            var outDims = new long[outNdim];
            var outStrides = new long[outNdim];
            int w = 0;
            for (int d = 0; d < ndim; d++)
            {
                if (d == ax1 || d == ax2) continue;
                outDims[w] = srcShape.dimensions[d];
                outStrides[w] = srcShape.strides[d];
                w++;
            }
            outDims[outNdim - 1] = diagSize;
            outStrides[outNdim - 1] = stride1 + stride2;

            long bufSize = srcShape.bufferSize > 0 ? srcShape.bufferSize : srcShape.size;
            var newShape = new Shape(outDims, outStrides, newOffset, bufSize);
            // NumPy contract: diagonal view is READ-ONLY.
            newShape = newShape.WithFlags(flagsToClear: ArrayFlags.WRITEABLE);

            return new NDArray(a.Storage.Alias(newShape)) { TensorEngine = a.TensorEngine };
        }
    }
}
