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
        [NDScoped]
        public static long nanargmax(NDArray a)
        {
            // a numpy.char.chararray operand: NumPy creates the index as the subclass, so the finalize ValueError follows the plain checks (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(a))
                return NDCharArrayBoundary.IndexScalar(a, static s => nanargmax(s));
            // Scalar return: the flattened form's 0-d index is this scope's own.
            return (long)np.EngineOf(a).NanArgMax(a);
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
            // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Reduced((a, axis, @out, keepdims), static s => nanargmax(s.a, s.axis, s.@out, s.keepdims), axis, keepdims, null);
            // a numpy.char.chararray operand: NumPy creates the index as the subclass, so the finalize ValueError follows the plain checks (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(a))
                return NDCharArrayBoundary.IndexResult((a, axis, @out, keepdims), static s => nanargmax(s.a, s.axis, s.@out, s.keepdims), @out);
            return np.EngineOf(a).NanArgMax(a, axis, @out, keepdims);
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
        [NDScoped]
        public static long nanargmin(NDArray a)
        {
            // a numpy.char.chararray operand: NumPy creates the index as the subclass, so the finalize ValueError follows the plain checks (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(a))
                return NDCharArrayBoundary.IndexScalar(a, static s => nanargmin(s));
            // Scalar return: the flattened form's 0-d index is this scope's own.
            return (long)np.EngineOf(a).NanArgMin(a);
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
            // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Reduced((a, axis, @out, keepdims), static s => nanargmin(s.a, s.axis, s.@out, s.keepdims), axis, keepdims, null);
            // a numpy.char.chararray operand: NumPy creates the index as the subclass, so the finalize ValueError follows the plain checks (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(a))
                return NDCharArrayBoundary.IndexResult((a, axis, @out, keepdims), static s => nanargmin(s.a, s.axis, s.@out, s.keepdims), @out);
            return np.EngineOf(a).NanArgMin(a, axis, @out, keepdims);
        }

        /// <summary>
        ///     NumPy's argmax/argmin <c>out=</c> contract (probed 2.4.2): the out array's dtype must be
        ///     safe-castable TO intp — every bool/int dtype through uint32 (uint64 is not: verbatim
        ///     <c>Cannot cast array data from dtype('uint64') to dtype('int64') according to the rule
        ///     'safe'</c>, a TypeError) — its shape must EQUAL the (keepdims-adjusted) result shape
        ///     (<c>output array does not match result of np.argmax.</c> — nanargmax leaks "argmax",
        ///     nanargmin "argmin"), and the int64 indices then cast UNSAFELY into it (an int8 out wraps
        ///     index 200 to -56; a bool out reads index != 0 — probed). Returns the SAME instance.
        ///     Char (no NumPy dtype) is integer-family and safe to int64, so it is allowed (house call). np.argmax /
        ///     np.argmin call it, and so do DefaultEngine's NanArgMax / NanArgMin (internal for them).
        /// </summary>
        /// <param name="result">The int64 indices.</param>
        /// <param name="out">The out array, or null.</param>
        /// <param name="name"><c>argmax</c> or <c>argmin</c> (the nan forms leak the plain name, as NumPy does).</param>
        /// <returns><paramref name="result"/> without an out; otherwise <paramref name="out"/>, the indices written.</returns>
        /// <exception cref="ValueError">The out's shape differs from the result's.</exception>
        /// <exception cref="TypeError">The out's dtype does not cast safely to int64.</exception>
        internal static NDArray NanArgWriteOut(NDArray result, NDArray @out, string name)
        {
            if (@out is null)
                return result;

            ValidateNanArgOut(@out, result.Shape.dimensions, name);
            np.copyto(@out, result, casting: "unsafe");
            return @out;
        }

        /// <summary>
        ///     <c>PyArray_ArgMaxWithKeepdims</c>'s two <c>out=</c> checks, in its order: the out must have the result's
        ///     shape exactly (<c>output array does not match result of np.argmax.</c>), then it must reach intp by a SAFE cast
        ///     — bool through uint32 (and NumSharp's char) do; uint64, floats, complex and text do not (NumPy's cast TypeError,
        ///     which words a 0-d out as a <c>scalar</c>).
        /// </summary>
        /// <param name="out">The out array.</param>
        /// <param name="dims">The result's dimensions.</param>
        /// <param name="name"><c>argmax</c> or <c>argmin</c> (the nan forms leak the plain name, as NumPy does).</param>
        /// <exception cref="ValueError">The shape differs from the result's.</exception>
        /// <exception cref="TypeError">The out's dtype does not cast safely to int64.</exception>
        /// <remarks>
        ///     The shape is checked FIRST: NumPy compares the out's dimensions before it converts the out to intp (an out of
        ///     the wrong shape AND the wrong dtype reports the shape). The numeric path used to check the dtype first.
        /// </remarks>
        internal static void ValidateNanArgOut(NDArray @out, long[] dims, string name)
        {
            if (@out.ndim != dims.Length)
                throw new ValueError($"output array does not match result of np.{name}.");
            for (int d = 0; d < dims.Length; d++)
                if (@out.Shape.dimensions[d] != dims[d])
                    throw new ValueError($"output array does not match result of np.{name}.");

            switch (@out.typecode)
            {
                case NPTypeCode.Boolean:
                case NPTypeCode.Byte:
                case NPTypeCode.SByte:
                case NPTypeCode.Int16:
                case NPTypeCode.UInt16:
                case NPTypeCode.Char:
                case NPTypeCode.Int32:
                case NPTypeCode.UInt32:
                case NPTypeCode.Int64:
                    return;
                default:
                    throw new TypeError(
                        $"Cannot cast {(@out.ndim == 0 ? "scalar" : "array data")} from {@out.dtype.ToString(true)} to " +
                        "dtype('int64') according to the rule 'safe'");
            }
        }
    }
}
