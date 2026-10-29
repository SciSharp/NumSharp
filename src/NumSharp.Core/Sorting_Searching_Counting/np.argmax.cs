namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Returns the indices of the maximum values along an axis.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="axis">By default, the index is into the flattened array, otherwise along the specified axis.</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one.</param>
        /// <returns>Array of indices into the array. It has the same shape as a.shape with the dimension along axis removed (unless keepdims is True).</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.argmax.html</remarks>
        /// <exception cref="TypeError">A <c>numpy.matrix</c> operand with <c>keepdims: true</c>: NumPy calls the matrix method, which has no such keyword. The text names that method, as in <c>matrix.sum() got an unexpected keyword argument 'keepdims'</c> for <c>np.sum</c>.</exception>
        /// <exception cref="ValueError">A <c>numpy.char.chararray</c> operand: NumPy creates
        ///     the int64 index result as the subclass (<c>Can only create a chararray from string
        ///     data.</c>), after the argument checks.</exception>
        public static NDArray argmax(NDArray a, int axis, bool keepdims = false)
            => NDMatrixBoundary.Sees(a)   // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            ? NDMatrixBoundary.Reduced((a, axis, keepdims), static s => argmax(s.a, s.axis, s.keepdims), axis, keepdims, null)
            : NDCharArrayBoundary.Sees(a)   // a numpy.char.chararray operand: NumPy creates the index as the subclass, so the finalize ValueError follows the plain checks (NDCharArrayBoundary)
            ? NDCharArrayBoundary.IndexResult((a, axis, keepdims), static s => argmax(s.a, s.axis, s.keepdims), null)
            : a.TensorEngine.ArgMax(a, axis: axis, keepdims: keepdims);

        /// <summary>
        ///     NumPy's <c>np.argmax(a, axis=None, out=None, *, keepdims=False)</c>: the int64 indices of the
        ///     maximum values along <paramref name="axis"/> (the first occurrence), or of the C-order flattened array.
        /// </summary>
        /// <param name="a">Input array (any dtype, fixed-width text included: compared as NumPy's
        ///     <c>STRING_argmax</c> / <c>UNICODE_argmax</c> compare it).</param>
        /// <param name="axis">The axis along which to operate, or null (the default) for the flattened array.</param>
        /// <param name="out">If provided, the indices are written into this array and the SAME instance is returned: its
        ///     shape must equal the result's exactly, and its dtype must be bool or an integer that casts safely to int64
        ///     (bool through uint32); the int64 indices are then cast into it unsafely, as NumPy writes them.</param>
        /// <param name="keepdims">If true, the reduced axis is left in the result with length one (with
        ///     <paramref name="axis"/> null, every axis: the shape <c>(1,) * a.ndim</c>).</param>
        /// <returns>The int64 indices (a 0-d array for the flattened form without <paramref name="keepdims"/>), or
        ///     <paramref name="out"/>.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is outside the input's rank (rank 1 for a 0-d input, which
        ///     NumPy ravels first).</exception>
        /// <exception cref="ValueError">The reduced length is 0 (<c>attempt to get argmax of an empty sequence</c>), or
        ///     <paramref name="out"/> does not have the result's shape (<c>output array does not match result of
        ///     np.argmax.</c>).</exception>
        /// <exception cref="TypeError"><paramref name="out"/>'s dtype does not cast safely to int64 (uint64, floats,
        ///     complex, text).</exception>
        /// <exception cref="ValueError">A <c>StringDType</c> line of two or more elements that holds an NA of a descriptor
        ///     whose NA object is neither a string nor NaN-like (NumPy's <c>Cannot compare null that is not a nan-like
        ///     value</c>).</exception>
        /// <exception cref="OutOfMemoryException">A corrupt <c>StringDType</c> cell (NumPy's <c>MemoryError</c> text).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
        ///     <para>
        ///     The order of NumPy's checks (<c>_PyArray_ArgMinMaxCommon</c>): the axis, the empty length, then the out —
        ///     the engine raises the first two before it reads an element, and nothing it computes can raise, so checking the
        ///     out afterwards reports the same error NumPy reports.
        ///     </para>
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.char.chararray</c> operand without <paramref name="out"/>: NumPy creates
        ///     the int64 index result as the subclass (<c>Can only create a chararray from string
        ///     data.</c>), after the argument checks; with <paramref name="out"/> the indices are
        ///     written.</exception>
        [NDScoped]
        public static NDArray argmax(NDArray a, int? axis = null, NDArray @out = null, bool keepdims = false)
        {
            // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Reduced((a, axis, @out, keepdims), static s => argmax(s.a, s.axis, s.@out, s.keepdims), axis, keepdims, null);
            // a numpy.char.chararray operand: NumPy creates the index as the subclass, so the finalize ValueError follows the plain checks (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(a))
                return NDCharArrayBoundary.IndexResult((a, axis, @out, keepdims), static s => argmax(s.a, s.axis, s.@out, s.keepdims), @out);
            return np.EngineOf(a).ReduceArgMax(a, axis, keepdims, @out);
        }

        /// <summary>
        ///     Returns the index of the maximum value.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <returns>Index of the maximum value in the flattened array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.argmax.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.char.chararray</c> operand: NumPy creates the int64 index result as the
        ///     subclass (<c>Can only create a chararray from string data.</c>), after the argument
        ///     checks.</exception>
        public static long argmax(NDArray a)
            => NDCharArrayBoundary.Sees(a)   // a numpy.char.chararray operand: NumPy creates the index as the subclass, so the finalize ValueError follows the plain checks (NDCharArrayBoundary)
            ? NDCharArrayBoundary.IndexScalar(a, static s => argmax(s))
            : (long)a.TensorEngine.ArgMax(a);

        /// <summary>
        ///     Returns the indices of the minimum values along an axis.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="axis">By default, the index is into the flattened array, otherwise along the specified axis.</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one.</param>
        /// <returns>Array of indices into the array. It has the same shape as a.shape with the dimension along axis removed (unless keepdims is True).</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.argmin.html</remarks>
        /// <exception cref="TypeError">A <c>numpy.matrix</c> operand with <c>keepdims: true</c>: NumPy calls the matrix method, which has no such keyword. The text names that method, as in <c>matrix.sum() got an unexpected keyword argument 'keepdims'</c> for <c>np.sum</c>.</exception>
        /// <exception cref="ValueError">A <c>numpy.char.chararray</c> operand: NumPy creates
        ///     the int64 index result as the subclass (<c>Can only create a chararray from string
        ///     data.</c>), after the argument checks.</exception>
        public static NDArray argmin(NDArray a, int axis, bool keepdims = false)
            => NDMatrixBoundary.Sees(a)   // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            ? NDMatrixBoundary.Reduced((a, axis, keepdims), static s => argmin(s.a, s.axis, s.keepdims), axis, keepdims, null)
            : NDCharArrayBoundary.Sees(a)   // a numpy.char.chararray operand: NumPy creates the index as the subclass, so the finalize ValueError follows the plain checks (NDCharArrayBoundary)
            ? NDCharArrayBoundary.IndexResult((a, axis, keepdims), static s => argmin(s.a, s.axis, s.keepdims), null)
            : a.TensorEngine.ArgMin(a, axis: axis, keepdims: keepdims);

        /// <summary>
        ///     NumPy's <c>np.argmin(a, axis=None, out=None, *, keepdims=False)</c>: the int64 indices of the
        ///     minimum values along <paramref name="axis"/> (the first occurrence), or of the C-order flattened array.
        /// </summary>
        /// <param name="a">Input array (any dtype, fixed-width text included: compared as NumPy's
        ///     <c>STRING_argmin</c> / <c>UNICODE_argmin</c> compare it).</param>
        /// <param name="axis">The axis along which to operate, or null (the default) for the flattened array.</param>
        /// <param name="out">If provided, the indices are written into this array and the SAME instance is returned: its
        ///     shape must equal the result's exactly, and its dtype must be bool or an integer that casts safely to int64
        ///     (bool through uint32); the int64 indices are then cast into it unsafely, as NumPy writes them.</param>
        /// <param name="keepdims">If true, the reduced axis is left in the result with length one (with
        ///     <paramref name="axis"/> null, every axis: the shape <c>(1,) * a.ndim</c>).</param>
        /// <returns>The int64 indices (a 0-d array for the flattened form without <paramref name="keepdims"/>), or
        ///     <paramref name="out"/>.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is outside the input's rank (rank 1 for a 0-d input, which
        ///     NumPy ravels first).</exception>
        /// <exception cref="ValueError">The reduced length is 0 (<c>attempt to get argmin of an empty sequence</c>), or
        ///     <paramref name="out"/> does not have the result's shape (<c>output array does not match result of
        ///     np.argmin.</c>).</exception>
        /// <exception cref="TypeError"><paramref name="out"/>'s dtype does not cast safely to int64 (uint64, floats,
        ///     complex, text).</exception>
        /// <exception cref="ValueError">A <c>StringDType</c> line of two or more elements that holds an NA of a descriptor
        ///     whose NA object is neither a string nor NaN-like (NumPy's <c>Cannot compare null that is not a nan-like
        ///     value</c>).</exception>
        /// <exception cref="OutOfMemoryException">A corrupt <c>StringDType</c> cell (NumPy's <c>MemoryError</c> text).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.argmin.html
        ///     <para>
        ///     The order of NumPy's checks (<c>_PyArray_ArgMinMaxCommon</c>): the axis, the empty length, then the out —
        ///     the engine raises the first two before it reads an element, and nothing it computes can raise, so checking the
        ///     out afterwards reports the same error NumPy reports.
        ///     </para>
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.char.chararray</c> operand without <paramref name="out"/>: NumPy creates
        ///     the int64 index result as the subclass (<c>Can only create a chararray from string
        ///     data.</c>), after the argument checks; with <paramref name="out"/> the indices are
        ///     written.</exception>
        [NDScoped]
        public static NDArray argmin(NDArray a, int? axis = null, NDArray @out = null, bool keepdims = false)
        {
            // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Reduced((a, axis, @out, keepdims), static s => argmin(s.a, s.axis, s.@out, s.keepdims), axis, keepdims, null);
            // a numpy.char.chararray operand: NumPy creates the index as the subclass, so the finalize ValueError follows the plain checks (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(a))
                return NDCharArrayBoundary.IndexResult((a, axis, @out, keepdims), static s => argmin(s.a, s.axis, s.@out, s.keepdims), @out);
            return np.EngineOf(a).ReduceArgMin(a, axis, keepdims, @out);
        }

        /// <summary>
        ///     Returns the index of the minimum value.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <returns>Index of the minimum value in the flattened array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.argmin.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.char.chararray</c> operand: NumPy creates the int64 index result as the
        ///     subclass (<c>Can only create a chararray from string data.</c>), after the argument
        ///     checks.</exception>
        public static long argmin(NDArray a)
            => NDCharArrayBoundary.Sees(a)   // a numpy.char.chararray operand: NumPy creates the index as the subclass, so the finalize ValueError follows the plain checks (NDCharArrayBoundary)
            ? NDCharArrayBoundary.IndexScalar(a, static s => argmin(s))
            : (long)a.TensorEngine.ArgMin(a);
    }
}
