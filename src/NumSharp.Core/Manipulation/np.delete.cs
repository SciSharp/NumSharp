using System;
using NumSharp.Backends;
using NumSharp.Generic;

namespace NumSharp
{
    public static partial class np
    {
        // ============================== np.delete ==============================
        // Return a copy of an array with sub-arrays along an axis deleted.
        //
        // NumPy 2.4.2 reference: numpy/lib/_function_base_impl.py::delete
        // NumPy's obj is a slice, an int or an array of ints / bools: NumPy's two typed forms here (API-parity batch 13),
        // Slice and NDArray - a C# int[], long[] or bool[] converts to NDArray implicitly - plus one long form, because a C#
        // int converts to Slice and NDArray alike (CS0121 without it). The obj kinds drive the paths:
        //   1. scalar int       — drop one position along axis (a 0-d or one-element integer array: NumPy's single value)
        //   2. slice            — drop a stride-defined range along axis
        //   3. integer ndarray  — drop multiple (possibly duplicate / unordered) positions
        //   4. boolean ndarray  — keep where mask is False (length must match axis dim)
        //
        // axis=None ravels the input to 1-D before deletion.
        //
        // For 1D contiguous arrays, the scalar/slice paths reuse np.concatenate of
        // pre/post views (zero-copy slicing + Buffer.MemoryCopy on the dst). The
        // array/bool paths build a keep-mask and delegate to np.compress, which
        // owns the fused popcount + axis-gather kernel.

        /// <summary>
        ///     NumPy's <c>delete(arr, obj, axis=None)</c> with a slice: the positions <paramref name="obj"/> selects along
        ///     <paramref name="axis"/> are removed, <paramref name="obj"/> read through Python's <c>slice.indices(N)</c>
        ///     against the axis length.
        /// </summary>
        /// <param name="arr">Input array.</param>
        /// <param name="obj">The positions to remove, as a slice (<c>new Slice(1, 4)</c> is NumPy's <c>slice(1, 4)</c>).</param>
        /// <param name="axis">The axis to delete along; <c>null</c> (default) flattens <paramref name="arr"/> first and
        ///     returns a 1-D result.</param>
        /// <returns>A fresh array without the sliced positions (F-ordered when <paramref name="arr"/> is). A
        ///     <c>numpy.char.chararray</c> operand gives a chararray VIEW of a text result (NumPy's <c>__array_wrap__</c>:
        ///     OWNDATA False); a result of another dtype stays plain.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="arr"/> or <paramref name="obj"/> is null.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="ArgumentException">A slice step of zero.</exception>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.delete.html
        ///     <para>One of np.delete's forms (API-parity batch 13): a slice has no NDArray form, so it keeps its own; one
        ///     integer is <see cref="delete(NDArray, long, int?)"/>, every other obj NumPy takes the
        ///     <see cref="delete(NDArray, NDArray, int?)"/> form.</para>
        /// </remarks>
        [NDScoped]
        public static NDArray delete(NDArray arr, Slice obj, int? axis = null)
        {
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            if (obj is null) throw new ArgumentNullException(nameof(obj));

            // NumPy's statements after the class boundaries and the null checks: NDArray.delete (NDArray.FunctionBase.cs).
            return NDArray.delete(arr, obj, axis);
        }

        /// <summary>
        ///     NumPy's <c>delete(arr, obj, axis=None)</c> with one integer: the position <paramref name="obj"/> along
        ///     <paramref name="axis"/> is removed - NumPy's single-value path for a Python int.
        /// </summary>
        /// <param name="arr">Input array.</param>
        /// <param name="obj">The position to remove; negative counts from the end. Every C# integer type binds here.</param>
        /// <param name="axis">The axis to delete along; <c>null</c> (default) flattens <paramref name="arr"/> first and
        ///     returns a 1-D result.</param>
        /// <returns>A fresh array without the position (F-ordered when <paramref name="arr"/> is). A
        ///     <c>numpy.char.chararray</c> operand gives a chararray VIEW of a text result; a result of another dtype stays
        ///     plain.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="arr"/> is null.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="IndexOutOfRangeException"><paramref name="obj"/> is outside <c>[-N, N)</c> (NumPy's
        ///     <c>index 7 is out of bounds for axis 0 with size 6</c>).</exception>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.delete.html
        ///     <para>The one integer form C# needs beside NumPy's two (API-parity batch 13): a C# <c>int</c> converts to
        ///     both <see cref="Slice"/> and <see cref="NDArray"/> by user-defined conversions, so without it
        ///     <c>np.delete(a, 1)</c> would not compile (CS0121). It is the <see cref="delete(NDArray, NDArray, int?)"/>
        ///     call with the Python int <paramref name="obj"/> stands for (NumPy's <c>obj</c> as given, a 0-d integer).</para>
        /// </remarks>
        [NDScoped /* the literal array of obj is this call's own */]
        public static NDArray delete(NDArray arr, long obj, int? axis = null)
        {
            if (arr is null) throw new ArgumentNullException(nameof(arr));

            // NumPy's statements after the class boundaries: NDArray.delete's NDArray path, whose 0-d integer is NumPy's
            // single-value branch (the Python int itself, range-checked against the axis).
            return NDArray.delete(arr, NDArray.PythonLiteral(obj), axis);
        }

        /// <summary>
        ///     NumPy's <c>delete(arr, obj, axis=None)</c> with an index or a mask: the positions <paramref name="obj"/> names
        ///     along <paramref name="axis"/> are removed. Selects the integer-array path or the boolean-mask path based on
        ///     <c>obj.GetTypeCode</c>; a 0-d or 1-element integer array collapses to the scalar-index path (NumPy's
        ///     <c>obj.size == 1 and obj.dtype.kind in "ui": obj = obj.item()</c>).
        /// </summary>
        /// <param name="arr">Input array.</param>
        /// <param name="obj">A boolean mask (removed where true) or an integer index array (a 0-d one is one position,
        ///     negative from the end); a C# <c>int[]</c> / <c>long[]</c> (positions) or <c>bool[]</c> (a mask) converts
        ///     implicitly, and a C# integer binds <see cref="delete(NDArray, long, int?)"/>, which calls this form.</param>
        /// <param name="axis">The axis to delete along; <c>null</c> deletes from the flattened array.</param>
        /// <returns>A fresh array without the selected sub-arrays (F-ordered when <paramref name="arr"/> is).
        ///     A <c>numpy.char.chararray</c> operand gives a chararray VIEW of a text result (NumPy's
        ///     <c>__array_wrap__</c>: OWNDATA False); a result of another dtype stays plain.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="arr"/> or <paramref name="obj"/> is null.</exception>
        /// <exception cref="ValueError">A boolean <paramref name="obj"/> is not 1-D or its length differs from the axis
        ///     length (NumPy's <c>boolean array argument obj to delete must be one dimensional and match the axis length of
        ///     6</c>; a C# <c>true</c> is a 0-d mask).</exception>
        /// <exception cref="IndexError"><paramref name="obj"/>'s dtype is neither boolean nor integer — text, a float,
        ///     complex, decimal — whatever its size (NumPy's <c>keep[obj,] = False</c>: <c>arrays used as indices must be of
        ///     integer (or boolean) type</c>).</exception>
        /// <exception cref="IndexOutOfRangeException">An index is outside <c>[-N, N)</c>.</exception>
        /// <remarks>
        ///     An integer <paramref name="obj"/> of any rank is accepted, as NumPy's fancy <c>keep[obj,]</c> accepts it (its
        ///     elements are the indices; one element takes the single-index path).
        ///     <para>API-parity batch 13: one of np.delete's forms - the int, int[], long[] and bool[] overloads went (their
        ///     arguments convert to this one or to the long form, which answer alike).</para>
        ///     <para>Scoped by <c>[NDScoped]</c>: the ravel work array, the keep mask, the helpers' chunk views and
        ///     the C-order result that an F-ordered relabel supersedes are released on return.</para>
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        [NDScoped]
        public static NDArray delete(NDArray arr, NDArray obj, int? axis = null)
        {
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            if (obj is null) throw new ArgumentNullException(nameof(obj));

            // NumPy's statements after the class boundaries and the null checks: NDArray.delete (NDArray.FunctionBase.cs).
            return NDArray.delete(arr, obj, axis);
        }
    }
}
