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
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>); a read-only <paramref name="out"/> (<c>output array is read-only</c>); text a numeric loop must parse that does not parse.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range for the input (a 0-d input counts as 1-D).</exception>
        /// <exception cref="TypeError">Fixed-width text, or a string <paramref name="typeCode"/>, where NumPy's loop resolution refuses (its verbatim text); a <c>StringDType</c> input concatenates.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="out"/> does not have the result's shape (<c>provided out is the wrong size for the accumulation.</c>).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html. The facade holds only the matrix boundary; NumPy's whole call is <see cref="Backends.TensorEngine.ReduceCumAdd"/>.</remarks>
        [NDScoped]
        public static NDArray cumsum(NDArray arr, int? axis = null, DType typeCode = null, NDArray @out = null)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(arr))
                return NDMatrixBoundary.Wrapped((arr, axis, typeCode, @out), static s => cumsum(s.arr, s.axis, s.typeCode, s.@out), @out);
            // np.cumsum IS np.add.accumulate, and NumPy's whole call (the text routes, the loop resolution, out=) is the
            // engine's: DefaultEngine.ReduceCumAdd.
            return arr.TensorEngine.ReduceCumAdd(arr, axis, typeCode, @out);
        }
    }
}
