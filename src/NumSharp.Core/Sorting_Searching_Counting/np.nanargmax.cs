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
            // a numpy.char.chararray operand: NumPy creates the index as the subclass, so the finalize ValueError follows the plain checks (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(a))
                return NDCharArrayBoundary.IndexScalar(a, static s => nanargmax(s));
            // Scalar return: the flattened form's 0-d index is this scope's own.
            return (long)a.TensorEngine.NanArgMax(a);
        }

        /// <summary>
        ///     Return the indices of the maximum values in the specified axis, ignoring NaNs.
        ///     For all-NaN slices a <see cref="ValueError"/> is raised. Warning: the results
        ///     cannot be trusted if a slice contains only NaNs and -Infs (NumPy caveat).
        /// </summary>
        /// <param name="a">Input data.</param>
        /// <param name="axis">Axis along which to operate. If <c>null</c>, the flattened input is used.</param>
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
        /// <exception cref="AxisError">Axis out of bounds (reports the original axis, like NumPy).</exception>
        /// <exception cref="OutOfMemoryException">A corrupt <c>StringDType</c> cell (NumPy's <c>MemoryError</c> text).</exception>
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
        public static NDArray nanargmax(NDArray a, int? axis = null, NDArray @out = null, bool keepdims = false)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Reduced((a, axis, @out, keepdims), static s => nanargmax(s.a, s.axis, s.@out, s.keepdims), axis, keepdims, null);
            // a numpy.char.chararray operand: NumPy creates the index as the subclass, so the finalize ValueError follows the plain checks (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(a))
                return NDCharArrayBoundary.IndexResult((a, axis, @out, keepdims), static s => nanargmax(s.a, s.axis, s.@out, s.keepdims), @out);
            return a.TensorEngine.NanArgMax(a, axis, @out, keepdims);
        }

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
            // a numpy.char.chararray operand: NumPy creates the index as the subclass, so the finalize ValueError follows the plain checks (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(a))
                return NDCharArrayBoundary.IndexScalar(a, static s => nanargmin(s));
            // Scalar return: the flattened form's 0-d index is this scope's own.
            return (long)a.TensorEngine.NanArgMin(a);
        }

        /// <summary>
        ///     Return the indices of the minimum values in the specified axis, ignoring NaNs.
        ///     For all-NaN slices a <see cref="ValueError"/> is raised. Warning: the results
        ///     cannot be trusted if a slice contains only NaNs and Infs (NumPy caveat).
        /// </summary>
        /// <param name="a">Input data.</param>
        /// <param name="axis">Axis along which to operate. If <c>null</c>, the flattened input is used.</param>
        /// <param name="out">If provided, the result is inserted into this array and the SAME instance is
        ///     returned (NumPy semantics — see <see cref="nanargmax(NDArray, int?, NDArray, bool)"/>).</param>
        /// <param name="keepdims">If true, the reduced axes are left in the result as dimensions with size one
        ///     (with <paramref name="axis"/> <c>null</c> the result has shape <c>(1,) * a.ndim</c>, like NumPy).</param>
        /// <returns>Array of int64 indices (or a 0-d scalar for the flattened form); <paramref name="out"/> when given.</returns>
        /// <exception cref="ValueError">All-NaN slice encountered; or a <c>StringDType</c> line of two or more elements that
        ///     holds an NA whose NA object is neither a string nor NaN-like (<c>np.argmax</c>'s <c>Cannot compare null that is
        ///     not a nan-like value</c>).</exception>
        /// <exception cref="AxisError">Axis out of bounds (reports the original axis, like NumPy).</exception>
        /// <exception cref="OutOfMemoryException">A corrupt <c>StringDType</c> cell (NumPy's <c>MemoryError</c> text).</exception>
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
        public static NDArray nanargmin(NDArray a, int? axis = null, NDArray @out = null, bool keepdims = false)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Reduced((a, axis, @out, keepdims), static s => nanargmin(s.a, s.axis, s.@out, s.keepdims), axis, keepdims, null);
            // a numpy.char.chararray operand: NumPy creates the index as the subclass, so the finalize ValueError follows the plain checks (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(a))
                return NDCharArrayBoundary.IndexResult((a, axis, @out, keepdims), static s => nanargmin(s.a, s.axis, s.@out, s.keepdims), @out);
            return a.TensorEngine.NanArgMin(a, axis, @out, keepdims);
        }
    }
}
