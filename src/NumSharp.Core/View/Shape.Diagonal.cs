using System;
using NumSharp.Backends;

namespace NumSharp
{
    public readonly partial struct Shape
    {
        /// <summary>
        ///     The layout of the diagonal view NumPy's <c>np.diagonal</c> returns for an array of this shape: the shape
        ///     without <paramref name="axis1"/> and <paramref name="axis2"/>, the diagonal appended as the last axis, over
        ///     the SAME buffer. Use it to build that view (<see cref="NDArray.Diagonal(NDArray, int, int, int)"/> installs
        ///     it over the array's storage); it reads and copies no element.
        /// </summary>
        /// <param name="offset">The diagonal's offset from the main one: positive above it (the start moves along
        ///     <paramref name="axis2"/>), negative below it (along <paramref name="axis1"/>). An offset past the edge gives a
        ///     diagonal of length 0, never an error.</param>
        /// <param name="axis1">The first axis of the 2-D sub-arrays (negative counts from the end).</param>
        /// <param name="axis2">The second axis of the 2-D sub-arrays (negative counts from the end).</param>
        /// <returns>
        ///     A new shape of rank <c>ndim - 1</c> with this shape's offset moved to the diagonal's first element (left
        ///     where it is when the diagonal is empty), the diagonal's stride <c>stride[axis1] + stride[axis2]</c>, this
        ///     shape's buffer size, and the WRITEABLE bit CLEARED: NumPy's diagonal is a read-only view.
        /// </returns>
        /// <exception cref="ValueError">This shape has fewer than two axes ("diag requires an array of at least two
        ///     dimensions"), or the two axes are one axis after wrapping ("axis1 and axis2 cannot be the same"). The
        ///     exceptions name np.diagonal's arguments (<c>a</c> and <c>axis2</c>), so their text is np.diagonal's.</exception>
        /// <exception cref="AxisError">An axis is out of range: "axis1: axis …" / "axis2: axis …", axis1 checked
        ///     first.</exception>
        /// <remarks>
        ///     <para>
        ///     NumPy's <c>PyArray_Diagonal</c> (<c>item_selection.c</c>) statement for statement, up to the view's
        ///     creation: the rank check, <c>check_and_adjust_axis_msg</c> on each axis, the same-axis check, then the
        ///     offset folded into the start and the two strides combined into one, so one step walks the diagonal.
        ///     Moved here from <c>DefaultEngine.Diagonal</c> by the np-layer dtype plan's phase PE (E7): a view is the
        ///     array object's, so it is not an engine member.
        ///     </para>
        ///     <para>
        ///     The layout flags are computed again from the new strides; the WRITEABLE bit is then cleared, and the
        ///     OWNDATA mirror is set by the storage that installs the shape (<see cref="UnmanagedStorage.Alias(Shape)"/>
        ///     marks a view).
        ///     </para>
        /// </remarks>
        internal readonly Shape Diagonal(int offset, int axis1, int axis2)
        {
            int ndim = NDim;
            // PyArray_Diagonal's ValueError, before either axis is read. It names np.diagonal's array argument "a" (an
            // ArgumentException appends the name to its message), so the text matches the function the caller called.
            if (ndim < 2)
                throw new ValueError(
                    "diag requires an array of at least two dimensions",
                    "a");

            // check_and_adjust_axis_msg: the argument name leads the AxisError's text ("axis1: axis ...").
            int ax1 = DefaultEngine.check_and_adjust_axis(ndim, axis1, nameof(axis1));
            int ax2 = DefaultEngine.check_and_adjust_axis(ndim, axis2, nameof(axis2));
            // PyArray_Diagonal's ValueError, after both axes are in range.
            if (ax1 == ax2)
                throw new ValueError(
                    "axis1 and axis2 cannot be the same",
                    nameof(axis2));

            long dim1 = dimensions[ax1];
            long dim2 = dimensions[ax2];
            long stride1 = strides[ax1];
            long stride2 = strides[ax2];

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

            // NumSharp strides are in elements (not bytes), so the start moves by offAbs elements along the offset's axis;
            // an empty diagonal keeps the start where it is, as NumPy only moves its data pointer for a non-empty one.
            long newOffset = this.offset;
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
                outDims[w] = dimensions[d];
                outStrides[w] = strides[d];
                w++;
            }
            outDims[outNdim - 1] = diagSize;
            outStrides[outNdim - 1] = stride1 + stride2;

            // The view addresses the same buffer, so it keeps the buffer's size (a shape that never recorded one reports
            // its element count).
            long bufSize = bufferSize > 0 ? bufferSize : size;
            var newShape = new Shape(outDims, outStrides, newOffset, bufSize);
            // NumPy contract: diagonal view is READ-ONLY.
            return newShape.WithFlags(flagsToClear: ArrayFlags.WRITEABLE);
        }
    }
}
