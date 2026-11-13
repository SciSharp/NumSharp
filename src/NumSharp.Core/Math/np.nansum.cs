namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Return the sum of array elements over a given axis treating Not a Numbers (NaNs) as zero.
        /// </summary>
        /// <param name="a">Array containing numbers whose sum is desired. If a is not an array, a conversion is attempted.</param>
        /// <param name="axis">Axis or axes along which the sum is computed. The default is to compute the sum of the
        ///     flattened array. A tuple (<c>(0, 2)</c>) reduces every axis it names together, in one pass (NumPy's
        ///     tuple axis; the empty tuple reduces none, and a tuple item gets no 0-d allowance).</param>
        /// <param name="dtype">np.sum's <c>dtype=</c>: the type of the accumulator and the result, or null for np.sum's
        ///     default (the input's dtype; int64 for a narrower integer or bool input).</param>
        /// <param name="out">np.sum's <c>out=</c>: the destination, of the expected output shape (returned), or null.</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one.</param>
        /// <param name="initial">np.sum's <c>initial=</c>: the starting value of the sum (a Python scalar), or null.</param>
        /// <param name="where">np.sum's <c>where=</c>: the bool mask of the elements to include, or null for all.</param>
        /// <returns>A new array containing the sum, with NaN values treated as zero
        ///     (<paramref name="out"/> itself when given).</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is outside <paramref name="a"/>'s dimensions.</exception>
        /// <exception cref="ValueError">A tuple that names an axis twice (<c>duplicate value in 'axis'</c>).</exception>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.nansum.html
        /// Only applicable to float and double arrays.
        /// For integer arrays, this is equivalent to np.sum (no NaN values possible).
        /// NumPy's statements: <c>a, mask = _replace_nan(a, 0); return np.sum(a, axis, dtype, out, keepdims, initial,
        /// where)</c>.
        ///     <para>One signature in NumPy's order (API-parity batch 3, 2026-11-11): it replaces <c>nansum(a, axis, keepdims)</c>, so a positional <c>keepdims</c>
        ///     is written <c>keepdims: true</c> (a bool in the dtype slot does not compile). A tuple axis waits for the
        ///     owner's axis type (OD1, decided: <c>NDAxis</c>).</para>
        /// </remarks>
        /// <exception cref="TypeError">A <c>numpy.matrix</c> operand with <c>keepdims: true</c>, <c>initial=</c> or
        ///     <c>where=</c>: NumPy calls the matrix method, which has no such keyword (<c>matrix.sum() got an unexpected
        ///     keyword argument 'where'</c>).</exception>
        public static NDArray nansum(NDArray a, NDAxis? axis = null, DType dtype = null, NDArray @out = null, bool keepdims = false,
            object initial = null, NDArray @where = null)
            => a.TensorEngine.NanSum(a, axis, keepdims, dtype, @out, initial, @where);

        /// <summary>
        ///     <see cref="nansum(NDArray, NDAxis?, DType, NDArray, bool, object, NDArray)"/> with the axis as one
        ///     <c>int</c>: the overload a C# integer binds (its identity conversion beats NDAxis' implicit one),
        ///     forwarding to the NDAxis form, so every answer, error, layout and subclass result is that overload's.
        /// </summary>
        /// <param name="axis">The axis (NumPy's <c>axis=k</c>; a negative value counts from the end).</param>
        /// <inheritdoc cref="nansum(NDArray, NDAxis?, DType, NDArray, bool, object, NDArray)"/>
        public static NDArray nansum(NDArray a, int axis, DType dtype = null, NDArray @out = null, bool keepdims = false, object initial = null, NDArray @where = null)
            => nansum(a, (NDAxis?)axis, dtype, @out, keepdims, initial, @where);

        /// <summary>
        ///     <see cref="nansum(NDArray, NDAxis?, DType, NDArray, bool, object, NDArray)"/> with the axes as an
        ///     <c>int[]</c>: the overload an <c>int[]</c> binds, and a bare <c>null</c> too (null converts to
        ///     <c>int[]</c> better than to NDAxis), forwarding to the NDAxis form: a null array is NumPy's None, any
        ///     other array NumPy's tuple (<c>new int[0]</c> is <c>()</c>).
        /// </summary>
        /// <param name="axis">The axes as NumPy's tuple, or <c>null</c> for None.</param>
        /// <inheritdoc cref="nansum(NDArray, NDAxis?, DType, NDArray, bool, object, NDArray)"/>
        public static NDArray nansum(NDArray a, int[] axis, DType dtype = null, NDArray @out = null, bool keepdims = false, object initial = null, NDArray @where = null)
            => nansum(a, (NDAxis?)axis, dtype, @out, keepdims, initial, @where);
    }
}
