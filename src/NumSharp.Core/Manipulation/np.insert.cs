using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        // ============================== np.insert ==============================
        // Insert values along the given axis before the given indices.
        //
        // The body is NDArray.insert (Manipulation/NDArray.FunctionBase.cs, beside this facade), which describes
        // the two branches; the facades keep their signatures, documentation, class boundaries and null checks.
        //
        // NumPy's obj is a slice, an int or a sequence of ints: NumPy's two typed forms here (API-parity batch 13), Slice
        // and NDArray - a C# int[] or long[] converts to NDArray implicitly - plus one long form (a C# int converts to Slice
        // and NDArray alike, CS0121 without it), which passes the Python int as a 0-d obj, NumPy's scalar-obj semantics.
        // Every value a C# caller passes converts to NDArray (a number is NumPy's
        // Python scalar: the range check, None as NaN, a str parsed). The twelve forms of batch 9 (int / long / Slice /
        // int[] / long[] / NDArray obj, each with NDArray and object values) answered alike; a C# List is passed as
        // np.array(list).

        /// <summary>
        ///     NumPy's <c>insert(arr, obj, values, axis=None)</c> with a slice: <paramref name="values"/> is inserted along
        ///     <paramref name="axis"/> before each position <paramref name="obj"/> selects (expanded via Python's
        ///     <c>slice.indices(N)</c> into an index array, then the multi-index branch runs).
        /// </summary>
        /// <param name="arr">Input array.</param>
        /// <param name="obj">The insertion positions, as a slice (<c>new Slice(1, 3)</c> is NumPy's <c>slice(1, 3)</c>).</param>
        /// <param name="values">The values to insert, cast to <paramref name="arr"/>'s dtype and broadcast to the insertion
        ///     slots; a C# number, string or array converts implicitly (null is Python's None, converted after the index
        ///     checks).</param>
        /// <param name="axis">The axis to insert along; <c>null</c> inserts into the flattened array.</param>
        /// <returns>A fresh array with the values inserted (F-ordered when <paramref name="arr"/> is). A
        ///     <c>numpy.char.chararray</c> operand gives a chararray VIEW of a text result (NumPy's <c>__array_wrap__</c>:
        ///     OWNDATA False); a result of another dtype stays plain.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="arr"/> or <paramref name="obj"/> is null.</exception>
        /// <exception cref="ValueError">The values do not fit the insertion slots (MapIter's <c>shape mismatch: ...</c>);
        ///     text values that do not parse as a numeric array's number type; a <c>numpy.matrix</c> operand whose result
        ///     cannot be a matrix (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="TypeError">None into an integer array (<c>int()</c>'s TypeError).</exception>
        /// <exception cref="OverflowError">A C# number literal outside the array dtype's range (NumPy's
        ///     OverflowError).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.insert.html
        ///     <para>One of np.insert's forms (API-parity batch 13): a slice has no NDArray form, so it keeps its own; one
        ///     integer is <see cref="insert(NDArray, long, NDArray, int?)"/>, every other obj NumPy takes the
        ///     <see cref="insert(NDArray, NDArray, NDArray, int?)"/> form.</para>
        /// </remarks>
        [NDScoped /* reclaims PrepareAxisContext's ravel work array and the pre-order intermediate WithSourceOrder supersedes (np.delete's exact pattern) */]
        public static NDArray insert(NDArray arr, Slice obj, NDArray values, int? axis = null)
        {
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            if (obj is null) throw new ArgumentNullException(nameof(obj));
            // A null `values` is Python's None, converted where NumPy converts the values: after the index checks
            // (NDArray.FunctionBase.cs's CoerceValuesForSingleIndex / CoerceValuesForMultiIndex).
            return NDArray.insert(arr, obj, values, axis);
        }

        /// <summary>
        ///     NumPy's <c>insert(arr, obj, values, axis=None)</c>: <paramref name="values"/> inserted along
        ///     <paramref name="axis"/> before the position(s) <paramref name="obj"/> names. Dispatches:
        ///     <list type="bullet">
        ///       <item>0-D integer (what <see cref="insert(NDArray, long, NDArray, int?)"/> passes for a C# integer) ⇒ the
        ///             scalar single-index path, with NumPy's scalar-obj value rule (the values' first axis moved to
        ///             <paramref name="axis"/>).</item>
        ///       <item>1-element 1-D integer (<c>new[] { 1 }</c>) ⇒ the single-index path without that move.</item>
        ///       <item>1-D bool ⇒ <see cref="np.flatnonzero"/> ⇒ the multi-index path.</item>
        ///       <item>1-D integer ⇒ the multi-index path.</item>
        ///       <item>&gt; 1-D ⇒ NumPy's ValueError "index array argument obj to insert must be one dimensional or
        ///             scalar".</item>
        ///     </list>
        /// </summary>
        /// <param name="arr">Input array.</param>
        /// <param name="obj">A 0-d or 1-D integer index array (negative from the end, <c>N</c> appends), or a 1-D boolean
        ///     mask of insertion positions; a C# <c>int[]</c> / <c>long[]</c> converts implicitly, and a C# integer binds
        ///     <see cref="insert(NDArray, long, NDArray, int?)"/>, which calls this form.</param>
        /// <param name="values">The values to insert, cast to <paramref name="arr"/>'s dtype and broadcast to the
        ///     insertion slot; a C# number is NumPy's Python scalar (range-checked: <c>300</c> into int8 is NumPy's
        ///     OverflowError), a string is parsed by a numeric array, null is Python's None (NaN, <c>nan+nanj</c>,
        ///     <c>False</c>, <c>'None'</c> or a <c>StringDType</c> NA object, converted after the index checks).</param>
        /// <param name="axis">The axis to insert along; <c>null</c> inserts into the flattened array.</param>
        /// <returns>A fresh array with the values inserted (F-ordered when <paramref name="arr"/> is).
        ///     A <c>numpy.char.chararray</c> operand gives a chararray VIEW of a text result (NumPy's
        ///     <c>__array_wrap__</c>: OWNDATA False); a result of another dtype stays plain.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="arr"/> or <paramref name="obj"/> is null.</exception>
        /// <exception cref="ValueError"><paramref name="obj"/> is a boolean mask that is not 1-D, or an index array of more
        ///     than one dimension (NumPy's two texts); text values that do not parse as the array's number type (NumPy
        ///     converts the values with <c>array(values, dtype=arr.dtype)</c>: <c>could not convert string to float</c>); a
        ///     <c>numpy.matrix</c> operand whose result cannot be a matrix (<c>shape too large to be a
        ///     matrix.</c>).</exception>
        /// <exception cref="IndexOutOfRangeException">An insertion index is outside <c>[-N, N]</c>.</exception>
        /// <exception cref="TypeError">A one-element text or complex <paramref name="obj"/> (<c>'&lt;' not supported between
        ///     instances of 'str' and 'int'</c>), a text <paramref name="obj"/> of any other size (<c>ufunc 'less' did not
        ///     contain a loop …</c>), or a one-element float within bounds (<c>slice indices must be integers …</c>); None
        ///     into an integer array (<c>int()</c>'s TypeError).</exception>
        /// <exception cref="IndexError">A float / complex <paramref name="obj"/> of any size but one (<c>arrays used as indices
        ///     must be of integer (or boolean) type</c>), or a one-element float out of bounds.</exception>
        /// <exception cref="OverflowError">A C# number literal outside the array dtype's range (NumPy's
        ///     OverflowError).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.insert.html
        ///     <para>Scoped by <c>[NDScoped]</c>: the ravel work array, the pre-order intermediate and the bool-mask
        ///     flatnonzero temp are released on return; an array passed as <paramref name="values"/> was built before the
        ///     scope opened, so it is never disposed.</para>
        ///     <para>API-parity batch 13: one of np.insert's forms - the int / int[] / long[] obj forms and every
        ///     <c>object values</c> form went (their arguments convert to this one or the long form, which answer alike; a
        ///     C# <c>List</c> is passed as <c>np.array(list)</c>).</para>
        /// </remarks>
        [NDScoped /* the same transient set as the slice form, plus the bool-mask flatnonzero temp */]
        public static NDArray insert(NDArray arr, NDArray obj, NDArray values, int? axis = null)
        {
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            if (obj is null) throw new ArgumentNullException(nameof(obj));
            // A null `values` is Python's None, converted where NumPy converts the values: after the index checks
            // (NDArray.FunctionBase.cs's CoerceValuesForSingleIndex / CoerceValuesForMultiIndex).
            return NDArray.insert(arr, obj, values, axis);
        }

        /// <summary>
        ///     NumPy's <c>insert(arr, obj, values, axis=None)</c> with one integer: <paramref name="values"/> inserted along
        ///     <paramref name="axis"/> before position <paramref name="obj"/> - NumPy's scalar-obj path for a Python int (the
        ///     values' first axis moved to <paramref name="axis"/>).
        /// </summary>
        /// <param name="arr">Input array.</param>
        /// <param name="obj">The insertion position; negative counts from the end, <c>N</c> appends. Every C# integer type
        ///     binds here.</param>
        /// <param name="values">The values to insert, cast to <paramref name="arr"/>'s dtype and broadcast to the insertion
        ///     slot; a C# number is NumPy's Python scalar (range-checked), a string is parsed by a numeric array, null is
        ///     Python's None (converted after the index check).</param>
        /// <param name="axis">The axis to insert along; <c>null</c> inserts into the flattened array.</param>
        /// <returns>A fresh array with the values inserted (F-ordered when <paramref name="arr"/> is). A
        ///     <c>numpy.char.chararray</c> operand gives a chararray VIEW of a text result; a result of another dtype stays
        ///     plain.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="arr"/> is null.</exception>
        /// <exception cref="IndexOutOfRangeException"><paramref name="obj"/> is outside <c>[-N, N]</c>.</exception>
        /// <exception cref="ValueError">The values do not fit the insertion slot; text values that do not parse as the
        ///     array's number type; a <c>numpy.matrix</c> operand whose result cannot be a matrix.</exception>
        /// <exception cref="TypeError">None into an integer array (<c>int()</c>'s TypeError).</exception>
        /// <exception cref="OverflowError">A C# number value outside the array dtype's range (NumPy's
        ///     OverflowError).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.insert.html
        ///     <para>The one integer form C# needs beside NumPy's two (API-parity batch 13): a C# <c>int</c> converts to both
        ///     <see cref="Slice"/> and <see cref="NDArray"/> by user-defined conversions, so without it
        ///     <c>np.insert(a, 1, 9)</c> would not compile (CS0121). It is the
        ///     <see cref="insert(NDArray, NDArray, NDArray, int?)"/> call with the Python int <paramref name="obj"/> stands
        ///     for (NumPy's <c>np.array(obj)</c>, a 0-d integer).</para>
        /// </remarks>
        [NDScoped /* the literal array of obj and the same transient set as the array form */]
        public static NDArray insert(NDArray arr, long obj, NDArray values, int? axis = null)
        {
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            // NumPy's `indices = np.array(obj)` of a Python int: a 0-d integer, the scalar-obj branch of NDArray.insert.
            return NDArray.insert(arr, NDArray.PythonLiteral(obj), values, axis);
        }
    }
}
