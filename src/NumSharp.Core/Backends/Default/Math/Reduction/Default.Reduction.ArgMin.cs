using System;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <inheritdoc />
        /// <remarks>
        ///     DefaultEngine computes the indices (<see cref="ArgMinIndices"/>) and then writes them into
        ///     <paramref name="out"/> (<see cref="WriteArgOut"/>). A numeric computation raises only what NumPy raises
        ///     before it looks at the out (the axis, the empty length), so the out's checks after it report what NumPy
        ///     reports; a text comparison can raise (a <c>StringDType</c> None-like NA), so a text operand's out is checked
        ///     first (<see cref="ArgTextChecks"/>), as NumPy checks it before it compares.
        /// </remarks>
        public override NDArray ReduceArgMin(NDArray arr, int? axis_, bool keepdims = false, NDArray @out = null)
        {
            // NumPy's _PyArray_ArgMinMaxCommon: the axis (a 0-d input ravels first), the empty length and the out, all
            // before its arg_func compares an element. Only a text comparison can raise, so only text needs them first.
            if (@out is not null && (Strings.NDStringUfuncs.IsVString(arr) || Strings.NDStringUfuncs.IsString(arr)))
            {
                Strings.NDStringUfuncs.ValidateReduceAxis("argmin", arr.ndim, axis_);
                ArgTextChecks(arr, axis_, @out, keepdims, "argmin");
            }
            return WriteArgOut(ArgMinIndices(arr, axis_, keepdims), @out, "argmin");
        }

        /// <summary>
        ///     The indices of NumPy's <c>np.argmin</c>, before <c>out=</c>: the int64 index of the smallest element along
        ///     <paramref name="axis_"/> (first occurrence), or of the flattened array.
        /// </summary>
        /// <param name="arr">The array.</param>
        /// <param name="axis_">The reduced axis, or null for the flattened array (a 0-d input ravels first).</param>
        /// <param name="keepdims">Whether the reduced axis stays with length one.</param>
        /// <returns>The int64 indices (0-d for the flattened form).</returns>
        /// <exception cref="ArgumentException">An empty line (<c>attempt to get argmin of an empty sequence</c>; NumPy's
        ///     ValueError, which a string array raises as <see cref="ValueError"/>).</exception>
        /// <exception cref="AxisError"><paramref name="axis_"/> is outside the input's rank (rank 1 for a 0-d input, which
        ///     NumPy ravels first) — checked before anything else, for every dtype.</exception>
        /// <exception cref="ValueError">An empty reduced line (<c>attempt to get argmin of an empty sequence</c>), or a
        ///     <c>StringDType</c> comparison that meets a <c>None</c>-like NA (<c>Cannot compare null that is not a nan-like
        ///     value</c>).</exception>
        /// <remarks>
        ///     A <c>StringDType</c> array is compared by its dtype's ArgFunc (<see cref="Strings.NDVStringReduction.ArgMaxMin"/>:
        ///     NumPy's <c>compare</c>, a NaN-like NA greater than any text). An <c>S</c> / <c>U</c> array is compared as
        ///     NumPy's <c>STRING_argmin</c> / <c>UNICODE_argmin</c> compare it
        ///     (unsigned code units over the item width; the first minimum of each line
        ///     wins), by <see cref="Sorting.TextArgReduce"/>.
        /// </remarks>
        private NDArray ArgMinIndices(NDArray arr, int? axis_, bool keepdims)
        {
            // StringDType first (the S / U test below refuses it): the dtype's ArgFunc over the cells (NumPy's `compare`, a
            // NaN-like NA greater than any text) once NumPy's checks passed (the axis, the empty-sequence error).
            if (Strings.NDStringUfuncs.VStringReduces("argmin", arr, axis_))
                return Strings.NDVStringReduction.ArgMaxMin(false, arr, axis_, keepdims);
            // Fixed-width text: NumPy's argmin checks the axis (a 0-d input ravels first) and refuses an empty line; where
            // it compares nothing (an empty result, lines of one element) the indices are all 0 and the bool stand-in of the
            // same shape computes them through the numeric path; otherwise the text is compared (TextArgReduce).
            if (Strings.NDStringUfuncs.IsString(arr))
            {
                Strings.NDStringUfuncs.ValidateReduceAxis("argmin", arr.ndim, axis_);
                if (Strings.NDStringUfuncs.EmptyArgReductionError("argmin", arr, axis_) is { } empty)
                    throw empty;
                if (!Strings.NDStringUfuncs.ShapeDecides("argmin", arr, axis_))
                    return Sorting.TextArgReduce.ArgReduce(arr, axis_, keepdims, max: false);
                using var textStandIn = Strings.NDStringUfuncs.BoolStandIn(arr);
                return ArgMinIndices(textStandIn, axis_, keepdims);
            }
            //in order to iterate an axis:
            //consider arange shaped (1,2,3,4) when we want to summarize axis 1 (2nd dimension which its value is 2)
            //the size of the array is [1, 2, n, m] all shapes after 2nd multiplied gives size
            //the size of what we need to reduce is the size of the shape of the given axis (shape[axis])
            var shape = arr.Shape;

            // PyArray_CheckAxis runs before anything reads the data: the axis is checked against the input's rank — rank 1
            // for a 0-d input, which NumPy ravels first — so an empty, a one-element and a 0-d input all raise NumPy's
            // AxisError for an axis out of range (a wrap-then-test loop here used to accept -ndim - 1, return 0 for any
            // axis of a 0-d or one-element input, and never end for a negative axis of a 0-d input).
            int? axisChecked = axis_ is null ? (int?)null : check_and_adjust_axis(Math.Max(arr.ndim, 1), axis_.Value);

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
                var emptyAxis = axisChecked.Value;

                if (shape[emptyAxis] == 0)
                {
                    // Reducing along a zero-size axis - raise error
                    throw new ValueError("attempt to get argmin of an empty sequence"); // NumPy's ValueError (an ArgumentException)
                }

                // Reducing along non-zero axis - return empty Int64 array with reduced shape
                var resultShape = Shape.GetAxis(shape, emptyAxis);
                var result = new NDArray(NPTypeCode.Int64, Shape.Adopt(resultShape), false);

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

            var axis = axisChecked.Value;
            if (shape[axis] == 1)
            {
                //if the given div axis is 1, result is all zeros
                if (keepdims)
                {
                    // Keep the axis but reduce to size 1 (it's already 1)
                    return new NDArray(NPTypeCode.Int64, Shape.Adopt(shape.dimensions), true);
                }
                // The reduced shape is allocated directly, so the result owns its data as NumPy's does (a squeeze view
                // of a full-shape zeros array reported OWNDATA False). A 1-D input whose only axis has size 1 reduces to
                // a fresh 0-d — a numpy SCALAR (read-only) at the boundary.
                var reducedDims = Shape.GetAxis(shape, axis);
                var reduced = new NDArray(NPTypeCode.Int64, reducedDims.Length == 0 ? Shape.Scalar : Shape.Adopt(reducedDims), true);
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
