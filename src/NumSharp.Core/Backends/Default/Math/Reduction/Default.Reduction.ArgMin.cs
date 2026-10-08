using System;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     NumPy's <c>np.argmin</c>: the int64 index of the smallest element along
        ///     <paramref name="axis_"/> (first occurrence), or of the flattened array.
        /// </summary>
        /// <param name="arr">The array.</param>
        /// <param name="axis_">The reduced axis, or null for the flattened array (a 0-d input ravels first).</param>
        /// <param name="keepdims">Whether the reduced axis stays with length one.</param>
        /// <returns>The int64 indices (0-d for the flattened form).</returns>
        /// <exception cref="ArgumentException">An empty line (<c>attempt to get argmin of an empty sequence</c>; NumPy's
        ///     ValueError, which a string array raises as <see cref="ValueError"/>).</exception>
        /// <exception cref="AxisError">A string array with an out-of-bounds axis.</exception>
        /// <exception cref="StringComputationNotSupportedException">A string array whose lines hold two or more elements:
        ///     NumPy compares the text (the next phase of NumSharp's string support). An empty result or lines of one
        ///     element give their zero indices.</exception>
        public override NDArray ReduceArgMin(NDArray arr, int? axis_, bool keepdims = false)
        {
            // Text: NumPy's argmin checks the axis (a 0-d input ravels first) and refuses an empty line, then compares the
            // text (the next phase) — except where no comparison happens: an empty result, or lines of one element, whose
            // indices are all 0. The bool stand-in of the same shape computes those through the numeric path.
            if (Strings.NDStringUfuncs.ReduceStandIn("argmin", arr, axis_) is { } standIn)
            {
                using (standIn)
                    return ReduceArgMin(standIn, axis_, keepdims);
            }
            //in order to iterate an axis:
            //consider arange shaped (1,2,3,4) when we want to summarize axis 1 (2nd dimension which its value is 2)
            //the size of the array is [1, 2, n, m] all shapes after 2nd multiplied gives size
            //the size of what we need to reduce is the size of the shape of the given axis (shape[axis])
            var shape = arr.Shape;

            if (shape.IsEmpty)
                throw new ValueError("attempt to get argmin of an empty sequence"); // NumPy's ValueError (an ArgumentException)

            // Handle empty arrays (size == 0) with axis reduction
            // NumPy: np.argmin(np.zeros((0,3)), axis=0) raises ValueError (reducing along zero-size axis)
            // NumPy: np.argmin(np.zeros((0,3)), axis=1) returns array([], dtype=int64) with shape (0,)
            if (shape.size == 0)
            {
                if (axis_ == null)
                {
                    // No axis specified - raise error
                    throw new ValueError("attempt to get argmin of an empty sequence"); // NumPy's ValueError (an ArgumentException)
                }

                // Axis specified - check if reducing along zero-size axis
                var emptyAxis = axis_.Value;
                while (emptyAxis < 0)
                    emptyAxis = arr.ndim + emptyAxis;
                if (emptyAxis >= arr.ndim)
                    throw new ArgumentOutOfRangeException(nameof(axis_));

                if (shape[emptyAxis] == 0)
                {
                    // Reducing along a zero-size axis - raise error
                    throw new ValueError("attempt to get argmin of an empty sequence"); // NumPy's ValueError (an ArgumentException)
                }

                // Reducing along non-zero axis - return empty Int64 array with reduced shape
                var resultShape = Shape.GetAxis(shape, emptyAxis);
                var result = np.empty(Shape.Adopt(resultShape), NPTypeCode.Int64);

                if (keepdims)
                {
                    var keepdimsShape = new long[arr.ndim];
                    for (int d = 0, sd = 0; d < arr.ndim; d++)
                    {
                        if (d == emptyAxis)
                            keepdimsShape[d] = 1;
                        else
                            keepdimsShape[d] = resultShape[sd++];
                    }
                    result.Storage.Reshape(Shape.Adopt(keepdimsShape));
                }

                return result;
            }

            if (shape.IsScalar || (shape.size == 1 && shape.NDim == 1))
            {
                var r = NDArray.Scalar(0L);  // Int64 for NumPy 2.x alignment
                if (keepdims)
                {
                    var keepdimsShape = new long[arr.ndim];
                    for (int i = 0; i < arr.ndim; i++)
                        keepdimsShape[i] = 1;
                    r.Storage.Reshape(Shape.Adopt(keepdimsShape));
                }
                return r.MarkReductionScalar();
            }

            if (axis_ == null)
            {
                // Use IL-generated kernels for element-wise reduction
                var r = NDArray.Scalar(argmin_elementwise_il(arr));
                if (keepdims)
                {
                    // NumPy: keepdims preserves the number of dimensions, all set to 1
                    var keepdimsShape = new long[arr.ndim];
                    for (int i = 0; i < arr.ndim; i++)
                        keepdimsShape[i] = 1;
                    r.Storage.Reshape(Shape.Adopt(keepdimsShape));
                }
                return r.MarkReductionScalar();
            }

            var axis = axis_.Value;
            while (axis < 0)
                axis = arr.ndim + axis; //handle negative axis

            if (axis >= arr.ndim)
                throw new ArgumentOutOfRangeException(nameof(axis));
            if (shape[axis] == 1)
            {
                //if the given div axis is 1, result is all zeros
                if (keepdims)
                {
                    // Keep the axis but reduce to size 1 (it's already 1)
                    return np.zeros(Shape.Adopt(shape.dimensions), NPTypeCode.Int64);
                }
                // The reduced shape is allocated directly, so the result owns its data as NumPy's does (a squeeze view
                // of a full-shape zeros array reported OWNDATA False). A 1-D input whose only axis has size 1 reduces to
                // a fresh 0-d — a numpy SCALAR (read-only) at the boundary.
                var reducedDims = Shape.GetAxis(shape, axis);
                var reduced = np.zeros(reducedDims.Length == 0 ? Shape.Scalar : Shape.Adopt(reducedDims), NPTypeCode.Int64);
                return reduced.MarkReductionScalar();
            }

            //handle keepdims - prepare output shape
            Shape axisedShape = Shape.Adopt(Shape.GetAxis(shape, axis));
            Shape outputShape = axisedShape;
            if (keepdims)
            {
                // Insert a 1 at the axis position
                var keepdimsShapeDims = new long[arr.ndim];
                int srcIdx = 0;
                for (int i = 0; i < arr.ndim; i++)
                {
                    if (i == axis)
                        keepdimsShapeDims[i] = 1;
                    else
                        keepdimsShapeDims[i] = axisedShape.dimensions[srcIdx++];
                }
                outputShape = Shape.Adopt(keepdimsShapeDims);
            }

            // Use IL kernel for axis reduction (reuse the ArgMax method which handles both ArgMax and ArgMin)
            return ExecuteAxisArgReduction(arr, axis, keepdims, outputShape, axisedShape, ReductionOp.ArgMin);
        }
    }
}
