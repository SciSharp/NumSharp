using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return the cumulative sum of the elements along a given axis.
        /// </summary>
        /// <param name="arr">Input array.</param>
        /// <param name="axis">Axis along which the cumulative sum is computed. The default (None) is to compute the cumsum over the flattened array.</param>
        /// <param name="typeCode">Type of the returned array and of the accumulator in which the elements are summed. If dtype is not specified, it defaults to the dtype of a, unless a has an integer dtype with a precision less than that of the default platform integer. In that case, the default platform integer is used.</param>
        /// <param name="out">Alternate output array in which to place the result. It must have the same shape and buffer length as the expected output, but its dtype may differ (the result is cast into it with NumPy's unsafe casting) and a reference to <paramref name="out"/> is returned.</param>
        /// <returns>A new array holding the result is returned unless out is specified, in which case a reference to out is returned. The result has the same size as a, and the same shape as a if axis is not None or a is a 1-d array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html</remarks>
        public static NDArray cumsum(NDArray arr, int? axis = null, NPTypeCode? typeCode = null, NDArray @out = null)
        {
            // np.cumsum == np.add.accumulate. With an axis argument it preserves the source
            // memory layout (KEEPORDER); ReduceCumAdd now allocates the output in that layout
            // directly (no post-hoc copy needed). axis=None ravels in C-order to a 1-D result.
            NDArray result = arr.TensorEngine.ReduceCumAdd(arr, axis, typeCode);
            return WriteScanToOut(result, @out);
        }

        // Shared out= writeback for np.cumsum/np.cumprod. NumPy's add.accumulate/multiply.accumulate
        // require out to have the natural output shape (any layout) and store the result into it with
        // UNSAFE casting (float->int truncates, complex->real drops the imaginary part), returning out.
        internal static NDArray WriteScanToOut(NDArray result, NDArray @out)
        {
            if (@out is null)
                return result;
            // A read-only out refuses FIRST — probed 2.4.2: a wrong-sized read-only out reports
            // "output array is read-only", not the size error (accumulate validates writeability
            // ahead of the accumulation-shape check). Without the guard the copyto below raised
            // the generic assignment text instead of NumPy's ufunc-out wording.
            NumSharpException.ThrowIfNotWriteable(@out.Shape, "output array");
            if (!@out.Shape.Equals(result.Shape))
                // NumPy 2.4.2 verbatim (PyUFunc_Accumulate); house IncorrectShapeException stands
                // in for NumPy's ValueError per the long-standing shape-error convention.
                throw new IncorrectShapeException("provided out is the wrong size for the accumulation.");
            copyto(@out, result, casting: "unsafe");
            return @out;
        }
    }
}
