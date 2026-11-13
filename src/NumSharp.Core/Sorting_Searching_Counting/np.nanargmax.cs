using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return the index of the maximum value in the flattened array, ignoring NaNs.
        ///     For an all-NaN array a <see cref="ValueError"/> is raised. Warning: the result
        ///     cannot be trusted if the array contains only NaNs and -Infs (NumPy caveat).
        /// </summary>
        /// <param name="a">Input data.</param>
        /// <returns>Index of the maximum value in the flattened array (NumPy intp = int64).</returns>
        /// <exception cref="ValueError">All-NaN slice encountered; or a <c>StringDType</c> line of two or more elements that
        ///     holds an NA whose NA object is neither a string nor NaN-like (<c>np.argmax</c>'s <c>Cannot compare null that is
        ///     not a nan-like value</c>).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.nanargmax.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.char.chararray</c> operand: NumPy creates the int64 index result as the
        ///     subclass (<c>Can only create a chararray from string data.</c>), after the argument
        ///     checks.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        [NDScoped]
        public static long nanargmax(NDArray a)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            // Scalar return: the flattened form's 0-d index is this scope's own.
            return (long)a.TensorEngine.NanArgMax(a);
        }

        /// <summary>
        ///     Return the indices of the maximum values in the specified axis, ignoring NaNs.
        ///     For all-NaN slices a <see cref="ValueError"/> is raised. Warning: the results
        ///     cannot be trusted if a slice contains only NaNs and -Infs (NumPy caveat).
        /// </summary>
        /// <param name="a">Input data.</param>
        /// <param name="axis">Axis along which to operate. If <c>null</c>, the flattened input is used. NumPy's axis here is
        ///     ONE integer: a tuple is refused with NumPy's TypeError, after the NaN mask has been reduced over it.</param>
        /// <param name="out">If provided, the result is inserted into this array and the SAME instance is
        ///     returned (NumPy semantics): the shape must equal the result's exactly, and the dtype must be
        ///     an integer family safe-castable to int64 (bool through uint32 — uint64/floats raise NumPy's
        ///     verbatim cast TypeError); the int64 indices then cast UNSAFELY into it (wrap/truncate).</param>
        /// <param name="keepdims">If true, the reduced axes are left in the result as dimensions with size one
        ///     (with <paramref name="axis"/> <c>null</c> the result has shape <c>(1,) * a.ndim</c>, like NumPy).</param>
        /// <returns>Array of int64 indices (or a 0-d scalar for the flattened form); <paramref name="out"/> when given.</returns>
        /// <exception cref="ValueError">All-NaN slice encountered; or a <c>StringDType</c> line of two or more elements that
        ///     holds an NA whose NA object is neither a string nor NaN-like (<c>np.argmax</c>'s <c>Cannot compare null that is
        ///     not a nan-like value</c>).</exception>
        /// <exception cref="AxisError">Axis out of bounds (reports the original axis, like NumPy); for a tuple, an item out of
        ///     bounds of an inexact input (the NaN mask's np.all reads it first).</exception>
        /// <exception cref="TypeError">A tuple <paramref name="axis"/> (<c>'tuple' object cannot be interpreted as an
        ///     integer</c>), after a repeated item (<c>duplicate value in 'axis'</c>) or an all-NaN slice over the tuple of an
        ///     inexact input has raised NumPy's ValueError first.</exception>
        /// <exception cref="MemoryError">A corrupt <c>StringDType</c> cell (NumPy's <c>MemoryError</c> text).</exception>
        /// <exception cref="ValueError">A <c>numpy.char.chararray</c> operand without <paramref name="out"/>: NumPy creates
        ///     the int64 index result as the subclass (<c>Can only create a chararray from string
        ///     data.</c>), after the argument checks; with <paramref name="out"/> the indices are
        ///     written.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.nanargmax.html
        ///     <para>Text of every class (<c>S</c>, <c>U</c>, NEP 55 <c>StringDType</c>) is <c>np.argmax</c> /
        ///     <c>np.argmin</c> of the text: NumPy's <c>_replace_nan</c> leaves an array that is not inexact alone, so a
        ///     <c>StringDType</c> NA is ordered by the argmax rules (a NaN-like NA is the greatest value, a None-like NA in a
        ///     line of two or more elements raises their <see cref="ValueError"/>), never skipped as a NaN.</para>
        /// </remarks>
        [NDScoped]
        public static NDArray nanargmax(NDArray a, NDAxis? axis = null, NDArray @out = null, bool keepdims = false)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            // NumPy's axis here is ONE integer: a tuple reaches np.argmax's axis converter, which refuses it, after
            // nanargmax's own mask statements (NanArgTupleRefusal).
            if (axis is { IsTuple: true } tuple)
                throw Backends.DefaultEngine.NanArgTupleRefusal(a, tuple);
            return a.TensorEngine.NanArgMax(a, Backends.ReductionArguments.IntegerAxis(axis), @out, keepdims);
        }

        /// <summary>
        ///     <see cref="nanargmax(NDArray, NDAxis?, NDArray, bool)"/> with the axis as an <c>int?</c>: the overload a C# integer,
        ///     an <c>int?</c>, a bare <c>null</c> and <c>default</c> bind (NumPy's axis here is ONE integer, so there is no
        ///     <c>int[]</c> form, and <c>null</c> / <c>default</c> stay None), forwarding to the NDAxis? form.
        /// </summary>
        /// <param name="a">Input data.</param>
        /// <param name="axis">The axis, or <c>null</c> for the flattened array.</param>
        /// <param name="out">The output array, or null (see the NDAxis? form).</param>
        /// <param name="keepdims">Keep the reduced axis with length one.</param>
        /// <returns>The int64 indices; <paramref name="out"/> when given.</returns>
        /// <inheritdoc cref="nanargmax(NDArray, NDAxis?, NDArray, bool)"/>
        public static NDArray nanargmax(NDArray a, int? axis, NDArray @out = null, bool keepdims = false)
            => nanargmax(a, NDAxis.FromNullable(axis), @out, keepdims);

        /// <summary>
        ///     Return the index of the minimum value in the flattened array, ignoring NaNs.
        ///     For an all-NaN array a <see cref="ValueError"/> is raised. Warning: the result
        ///     cannot be trusted if the array contains only NaNs and Infs (NumPy caveat).
        /// </summary>
        /// <param name="a">Input data.</param>
        /// <returns>Index of the minimum value in the flattened array (NumPy intp = int64).</returns>
        /// <exception cref="ValueError">All-NaN slice encountered; or a <c>StringDType</c> line of two or more elements that
        ///     holds an NA whose NA object is neither a string nor NaN-like (<c>np.argmax</c>'s <c>Cannot compare null that is
        ///     not a nan-like value</c>).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.nanargmin.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.char.chararray</c> operand: NumPy creates the int64 index result as the
        ///     subclass (<c>Can only create a chararray from string data.</c>), after the argument
        ///     checks.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        [NDScoped]
        public static long nanargmin(NDArray a)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            // Scalar return: the flattened form's 0-d index is this scope's own.
            return (long)a.TensorEngine.NanArgMin(a);
        }

        /// <summary>
        ///     Return the indices of the minimum values in the specified axis, ignoring NaNs.
        ///     For all-NaN slices a <see cref="ValueError"/> is raised. Warning: the results
        ///     cannot be trusted if a slice contains only NaNs and Infs (NumPy caveat).
        /// </summary>
        /// <param name="a">Input data.</param>
        /// <param name="axis">Axis along which to operate. If <c>null</c>, the flattened input is used. NumPy's axis here is
        ///     ONE integer: a tuple is refused with NumPy's TypeError, after the NaN mask has been reduced over it.</param>
        /// <param name="out">If provided, the result is inserted into this array and the SAME instance is
        ///     returned (NumPy semantics — see <see cref="nanargmax(NDArray, int?, NDArray, bool)"/>).</param>
        /// <param name="keepdims">If true, the reduced axes are left in the result as dimensions with size one
        ///     (with <paramref name="axis"/> <c>null</c> the result has shape <c>(1,) * a.ndim</c>, like NumPy).</param>
        /// <returns>Array of int64 indices (or a 0-d scalar for the flattened form); <paramref name="out"/> when given.</returns>
        /// <exception cref="ValueError">All-NaN slice encountered; or a <c>StringDType</c> line of two or more elements that
        ///     holds an NA whose NA object is neither a string nor NaN-like (<c>np.argmax</c>'s <c>Cannot compare null that is
        ///     not a nan-like value</c>).</exception>
        /// <exception cref="AxisError">Axis out of bounds (reports the original axis, like NumPy); for a tuple, an item out of
        ///     bounds of an inexact input (the NaN mask's np.all reads it first).</exception>
        /// <exception cref="TypeError">A tuple <paramref name="axis"/> (<c>'tuple' object cannot be interpreted as an
        ///     integer</c>), after a repeated item (<c>duplicate value in 'axis'</c>) or an all-NaN slice over the tuple of an
        ///     inexact input has raised NumPy's ValueError first.</exception>
        /// <exception cref="MemoryError">A corrupt <c>StringDType</c> cell (NumPy's <c>MemoryError</c> text).</exception>
        /// <exception cref="ValueError">A <c>numpy.char.chararray</c> operand without <paramref name="out"/>: NumPy creates
        ///     the int64 index result as the subclass (<c>Can only create a chararray from string
        ///     data.</c>), after the argument checks; with <paramref name="out"/> the indices are
        ///     written.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.nanargmin.html
        ///     <para>Text of every class (<c>S</c>, <c>U</c>, NEP 55 <c>StringDType</c>) is <c>np.argmax</c> /
        ///     <c>np.argmin</c> of the text: NumPy's <c>_replace_nan</c> leaves an array that is not inexact alone, so a
        ///     <c>StringDType</c> NA is ordered by the argmax rules (a NaN-like NA is the greatest value, a None-like NA in a
        ///     line of two or more elements raises their <see cref="ValueError"/>), never skipped as a NaN.</para>
        /// </remarks>
        [NDScoped]
        public static NDArray nanargmin(NDArray a, NDAxis? axis = null, NDArray @out = null, bool keepdims = false)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            // NumPy's axis here is ONE integer: a tuple reaches np.argmin's axis converter, which refuses it, after
            // nanargmin's own mask statements (NanArgTupleRefusal).
            if (axis is { IsTuple: true } tuple)
                throw Backends.DefaultEngine.NanArgTupleRefusal(a, tuple);
            return a.TensorEngine.NanArgMin(a, Backends.ReductionArguments.IntegerAxis(axis), @out, keepdims);
        }

        /// <summary>
        ///     <see cref="nanargmin(NDArray, NDAxis?, NDArray, bool)"/> with the axis as an <c>int?</c>: the overload a C# integer,
        ///     an <c>int?</c>, a bare <c>null</c> and <c>default</c> bind (NumPy's axis here is ONE integer, so there is no
        ///     <c>int[]</c> form, and <c>null</c> / <c>default</c> stay None), forwarding to the NDAxis? form.
        /// </summary>
        /// <param name="a">Input data.</param>
        /// <param name="axis">The axis, or <c>null</c> for the flattened array.</param>
        /// <param name="out">The output array, or null (see the NDAxis? form).</param>
        /// <param name="keepdims">Keep the reduced axis with length one.</param>
        /// <returns>The int64 indices; <paramref name="out"/> when given.</returns>
        /// <inheritdoc cref="nanargmin(NDArray, NDAxis?, NDArray, bool)"/>
        public static NDArray nanargmin(NDArray a, int? axis, NDArray @out = null, bool keepdims = false)
            => nanargmin(a, NDAxis.FromNullable(axis), @out, keepdims);
    }
}
