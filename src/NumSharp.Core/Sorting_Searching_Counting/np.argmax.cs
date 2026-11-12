using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     NumPy's <c>np.argmax(a, axis=None, out=None, *, keepdims=False)</c>: the int64 indices of the
        ///     maximum values along <paramref name="axis"/> (the first occurrence), or of the C-order flattened array.
        /// </summary>
        /// <param name="a">Input array (any dtype, fixed-width text included: compared as NumPy's
        ///     <c>STRING_argmax</c> / <c>UNICODE_argmax</c> compare it).</param>
        /// <param name="axis">The axis along which to operate, or null (the default) for the flattened array. NumPy's axis
        ///     type, but ONE axis: a tuple is NumPy's TypeError (np.argmax takes no axis set).</param>
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
        /// <exception cref="MemoryError">A corrupt <c>StringDType</c> cell (NumPy's <c>MemoryError</c> text).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
        ///     <para>
        ///     The order of NumPy's checks (<c>_PyArray_ArgMinMaxCommon</c>): the axis, the empty length, then the out, all
        ///     before an element is compared — so an out error comes before a <c>StringDType</c> comparison's refusal
        ///     (until 0.80.0 NumSharp compared the text first).
        ///     </para>
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.char.chararray</c> operand without <paramref name="out"/>: NumPy creates
        ///     the int64 index result as the subclass (<c>Can only create a chararray from string
        ///     data.</c>), after the argument checks; with <paramref name="out"/> the indices are
        ///     written.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="TypeError"><paramref name="axis"/> is a tuple (<c>'tuple' object cannot be interpreted as an
        ///     integer</c>, the empty tuple and a one-axis tuple included), raised before anything else is looked at; or
        ///     <paramref name="out"/> is a C# scalar converted to a 0-d array (<c>output must be an array</c>): the third
        ///     positional parameter is out=, so <c>np.argmax(a, 0, true)</c> raises - write <c>keepdims: true</c> by name.</exception>
        [NDScoped]
        public static NDArray argmax(NDArray a, NDAxis? axis = null, NDArray @out = null, bool keepdims = false)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            // ndarray.argmax's argument parser converts the axis to ONE integer (PyArray_AxisConverter): a tuple is its
            // TypeError, before the out's check and the axis' bounds.
            if (axis is { IsTuple: true })
                throw new TypeError("'tuple' object cannot be interpreted as an integer");
            return a.TensorEngine.ReduceArgMax(a, Backends.ReductionArguments.IntegerAxis(axis), keepdims, @out);
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
            => (long)a.TensorEngine.ArgMax(a);

        /// <summary>
        ///     NumPy's <c>np.argmin(a, axis=None, out=None, *, keepdims=False)</c>: the int64 indices of the
        ///     minimum values along <paramref name="axis"/> (the first occurrence), or of the C-order flattened array.
        /// </summary>
        /// <param name="a">Input array (any dtype, fixed-width text included: compared as NumPy's
        ///     <c>STRING_argmin</c> / <c>UNICODE_argmin</c> compare it).</param>
        /// <param name="axis">The axis along which to operate, or null (the default) for the flattened array. NumPy's axis
        ///     type, but ONE axis: a tuple is NumPy's TypeError (np.argmin takes no axis set).</param>
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
        /// <exception cref="MemoryError">A corrupt <c>StringDType</c> cell (NumPy's <c>MemoryError</c> text).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.argmin.html
        ///     <para>
        ///     The order of NumPy's checks (<c>_PyArray_ArgMinMaxCommon</c>): the axis, the empty length, then the out, all
        ///     before an element is compared — so an out error comes before a <c>StringDType</c> comparison's refusal
        ///     (until 0.80.0 NumSharp compared the text first).
        ///     </para>
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.char.chararray</c> operand without <paramref name="out"/>: NumPy creates
        ///     the int64 index result as the subclass (<c>Can only create a chararray from string
        ///     data.</c>), after the argument checks; with <paramref name="out"/> the indices are
        ///     written.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="TypeError"><paramref name="axis"/> is a tuple (<c>'tuple' object cannot be interpreted as an
        ///     integer</c>, the empty tuple and a one-axis tuple included), raised before anything else is looked at; or
        ///     <paramref name="out"/> is a C# scalar converted to a 0-d array (<c>output must be an array</c>): the third
        ///     positional parameter is out=, so <c>np.argmin(a, 0, true)</c> raises - write <c>keepdims: true</c> by name.</exception>
        [NDScoped]
        public static NDArray argmin(NDArray a, NDAxis? axis = null, NDArray @out = null, bool keepdims = false)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            // ndarray.argmin's argument parser converts the axis to ONE integer (PyArray_AxisConverter): a tuple is its
            // TypeError, before the out's check and the axis' bounds.
            if (axis is { IsTuple: true })
                throw new TypeError("'tuple' object cannot be interpreted as an integer");
            return a.TensorEngine.ReduceArgMin(a, Backends.ReductionArguments.IntegerAxis(axis), keepdims, @out);
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
            => (long)a.TensorEngine.ArgMin(a);
    }
}
