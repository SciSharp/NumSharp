using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return specified diagonals of <paramref name="a"/>. For a 2-D
        ///     array, returns the diagonal as a 1-D array. For an N-D array,
        ///     the last two axes (default <paramref name="axis1"/>=0,
        ///     <paramref name="axis2"/>=1) define the 2-D sub-arrays from which
        ///     diagonals are taken; the diagonal is appended as the last axis
        ///     of the returned array.
        /// </summary>
        /// <param name="a">Source array. Must have at least 2 dimensions.</param>
        /// <param name="offset">
        ///     Offset of the diagonal from the main diagonal. Positive values
        ///     refer to diagonals above the main, negative below. Default 0.
        /// </param>
        /// <param name="axis1">
        ///     First axis of the 2-D sub-array. Default 0.
        /// </param>
        /// <param name="axis2">
        ///     Second axis of the 2-D sub-array. Default 1.
        /// </param>
        /// <returns>
        ///     A <strong>read-only view</strong> sharing storage with
        ///     <paramref name="a"/>. Shape: <c>a.shape</c> with
        ///     <paramref name="axis1"/> and <paramref name="axis2"/> removed
        ///     and the diagonal appended as the last axis.
        /// </returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.diagonal.html
        ///     <para>
        ///     Mirrors NumPy's <c>PyArray_Diagonal</c> (item_selection.c). The
        ///     view trick: combining the two strides into one
        ///     <c>stride[axis1] + stride[axis2]</c> walks the diagonal in one
        ///     step. Read-only by NumPy contract (the writeable-by-default
        ///     change pencilled in for NumPy 1.10 hasn't shipped in 2.x).
        ///     </para>
        /// </remarks>
        public static NDArray diagonal(NDArray a, int offset = 0, int axis1 = 0, int axis2 = 1)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));

            int ndim = a.ndim;
            if (ndim < 2)
                throw new ArgumentException(
                    "diag requires an array of at least two dimensions",
                    nameof(a));

            int ax1 = NormalizeAxis(axis1, ndim, nameof(axis1));
            int ax2 = NormalizeAxis(axis2, ndim, nameof(axis2));
            if (ax1 == ax2)
                throw new ArgumentException(
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

        /// <summary>
        ///     Normalises a possibly-negative axis to the [0, ndim) range, matching
        ///     NumPy's <c>check_and_adjust_axis_msg</c>. Throws with the parameter
        ///     name embedded in the message.
        /// </summary>
        private static int NormalizeAxis(int axis, int ndim, string argName)
        {
            int adjusted = axis < 0 ? axis + ndim : axis;
            if (adjusted < 0 || adjusted >= ndim)
                throw new ArgumentOutOfRangeException(argName,
                    $"{argName}: axis {axis} is out of bounds for array of dimension {ndim}");
            return adjusted;
        }
    }
}
