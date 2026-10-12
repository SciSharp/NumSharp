using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return the cumulative product of the elements along a given axis.
        /// </summary>
        /// <param name="arr">Input array.</param>
        /// <param name="axis">Axis along which the cumulative product is computed. The default (None) is to compute the cumprod over the flattened array.</param>
        /// <param name="typeCode">Type of the returned array and of the accumulator in which the elements are multiplied. If dtype is not specified, it defaults to the dtype of a, unless a has an integer dtype with a precision less than that of the default platform integer. In that case, the default platform integer is used.</param>
        /// <param name="out">Alternate output array in which to place the result. It must have the same shape and buffer length as the expected output, but its dtype may differ (the result is cast into it with NumPy's unsafe casting) and a reference to <paramref name="out"/> is returned.</param>
        /// <returns>A new array holding the result is returned unless out is specified, in which case a reference to out is returned. The result has the same size as a, and the same shape as a if axis is not None or a is a 1-d array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.cumprod.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        [NDScoped]
        public static NDArray cumprod(NDArray arr, int? axis = null, DType typeCode = null, NDArray @out = null)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(arr))
                return NDMatrixBoundary.Wrapped((arr, axis, typeCode, @out), static s => cumprod(s.arr, s.axis, s.typeCode, s.@out), @out);
            // Text (input or out=): multiply.accumulate's loop on (out, arr) answers after the axis (ReduceOutError); without
            // out= the engine's guard answers (ReduceError).
            // A numeric dtype= over text: NumPy casts the text to that number type first (a parse); the scope reclaims it.
            if (Backends.Strings.NDStringUfuncs.ParseForNumericReduce("cumprod", arr, axis, typeCode) is { } parsed)
                arr = parsed;
            Backends.Strings.NDStringUfuncs.ThrowIfReduce("cumprod", arr, axis, @out, typeCode);
            NDArray result = arr.TensorEngine.ReduceCumMul(arr, axis, typeCode);
            return WriteScanToOut(result, @out);
        }
    }
}
