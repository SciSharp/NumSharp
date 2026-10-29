using System;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Change elements of <paramref name="a"/> based on a boolean mask and
        ///     input values. Where <paramref name="mask"/> is true (walked in
        ///     C-order), writes <c>a.flat[i] = values.flat[i % values.size]</c>.
        ///     In-place.
        /// </summary>
        /// <param name="a">Target array (modified in place).</param>
        /// <param name="mask">
        ///     Boolean mask. Must have the same total <em>size</em> as
        ///     <paramref name="a"/> — NumPy allows shape mismatch as long as element
        ///     counts match. Non-bool masks are cast to bool (<c>!= 0</c>; NaN/inf → true).
        /// </param>
        /// <param name="values">
        ///     Values to write, cast to <paramref name="a"/>'s dtype. Cycled by
        ///     <em>position</em>: the value written at flat index <c>i</c> is
        ///     <c>values.flat[i % values.size]</c> (the cursor advances on every
        ///     element, unlike <see cref="place"/> which advances only on True
        ///     positions). An empty <paramref name="values"/> is a no-op — NumPy
        ///     returns without touching <paramref name="a"/> (unlike
        ///     <see cref="place"/>, which raises). <c>null</c> is Python's <c>None</c>: one value, converted into
        ///     <paramref name="a"/>'s dtype after the mask's size check (<see cref="NDArray.NoneForStore"/>: NaN,
        ///     <c>nan+nanj</c>, <c>False</c>, <c>'None'</c>, a <c>StringDType</c> NA object; int()'s TypeError for an
        ///     integer array, even under an all-false mask).
        /// </param>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.putmask.html
        ///     <para>
        ///     Differs from <see cref="place"/> in two probed ways (NumPy 2.4.2):
        ///     the values cursor advances by position (every element) rather than per
        ///     True, and an empty <paramref name="values"/> is a silent no-op rather
        ///     than a <c>ValueError</c>.
        ///     </para>
        ///     <para>Scoped by <c>[NDScoped]</c>: the scratch copy (a non-contiguous target, or one that may overlap
        ///     <paramref name="values"/>/<paramref name="mask"/>) and the bool/dtype-converted copies of
        ///     <paramref name="mask"/> and <paramref name="values"/> are released on return, exceptions included. A
        ///     mask or values array used as-is is the caller's (built before the scope opened), so the scope never
        ///     disposes it.</para>
        /// </remarks>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> or <paramref name="mask"/> is null.</exception>
        /// <exception cref="TypeError"><paramref name="values"/> is null (<c>None</c>) and <paramref name="a"/> is an
        ///     integer (or <c>decimal</c>) array.</exception>
        /// <exception cref="ValueError"><paramref name="a"/> is read-only (checked first, as NumPy does), or
        ///     <paramref name="mask"/>'s size differs from <paramref name="a"/>'s (<c>putmask: mask and data must be the
        ///     same size</c>).</exception>
        /// <exception cref="InvalidCastException">An array <paramref name="values"/> does not cast 'safely' to
        ///     <paramref name="a"/>'s dtype (NumPy's TypeError: <c>Cannot cast array data from dtype('float64') to
        ///     dtype('int32') according to the rule 'safe'</c>) — even when it is empty. A 0-d value (NumSharp's stand-in
        ///     for a Python scalar) is converted unconditionally.</exception>
        /// <exception cref="UnicodeEncodeError">Text values that are not ASCII into an <c>S</c> array.</exception>
        /// <exception cref="UnicodeDecodeError">Byte values that are not ASCII into a <c>U</c> array.</exception>
        /// <exception cref="ValueError">A text value that does not parse into a numeric array (NumPy's <c>int()</c> /
        ///     <c>float()</c> / <c>complex()</c> texts); numbers into a text array are written as their <c>str()</c>.</exception>
        /// <exception cref="SystemError">A <c>U</c> mask holds a unit above U+10FFFF: NumPy's string-to-bool cast of the mask
        ///     raises it, before the size check. (A text mask is otherwise read by its truth values: a non-empty string is
        ///     True.)</exception>
        public static void putmask(NDArray a, NDArray mask, NDArray values)
            => UnmanagedStorage.PutMaskCore(a, mask, values, sequence: false);

        /// <summary>
        ///     <see cref="putmask(NDArray, NDArray, NDArray)"/> with the values given as a C# array — NumPy's Python LIST of
        ///     values: <c>np.putmask(a, mask, [1.9, 2.9])</c> is <c>np.putmask(a, mask, new[] { 1.9, 2.9 })</c>.
        /// </summary>
        /// <param name="a">Target array (modified in place).</param>
        /// <param name="mask">Boolean mask of <paramref name="a"/>'s size.</param>
        /// <param name="values">
        ///     The values, cycled by position. NumPy converts a list STRAIGHT INTO <paramref name="a"/>'s dtype
        ///     (<c>PyArray_FromAny(list, a.dtype)</c>), so there is no <c>'safe'</c> rule: <c>[1.9, 2.9]</c> into an int32
        ///     array truncates — where <c>np.array([1.9, 2.9])</c> raises the cast TypeError.
        /// </param>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> or <paramref name="mask"/> is null.</exception>
        /// <exception cref="TypeError"><paramref name="values"/> is null (<c>None</c>) and <paramref name="a"/> is an
        ///     integer (or <c>decimal</c>) array.</exception>
        /// <exception cref="ValueError">The checks of <see cref="putmask(NDArray, NDArray, NDArray)"/>.</exception>
        /// <exception cref="UnicodeEncodeError">Text values that are not ASCII into an <c>S</c> array.</exception>
        /// <exception cref="UnicodeDecodeError">Byte values that are not ASCII into a <c>U</c> array.</exception>
        /// <exception cref="ValueError">A text value that does not parse into a numeric array.</exception>
        /// <remarks>C# binds a typed array argument here rather than to the <see cref="NDArray"/> overload (a reference
        ///     conversion beats NumSharp's implicit <c>Array → NDArray</c> one): pass <c>np.array(...)</c> for NumPy's ndarray
        ///     semantics. A bare <c>null</c> binds here too (<c>Array</c> is the better conversion target), and is Python's
        ///     <c>None</c> exactly as in the <see cref="NDArray"/> overload.</remarks>
        [NDScoped]
        public static void putmask(NDArray a, NDArray mask, Array values)
        {
            // None: PutMaskCore converts it after the mask checks (a null values array is None there).
            if (values is null)
            {
                UnmanagedStorage.PutMaskCore(a, mask, null, sequence: false);
                return;
            }
            // The list's own dtype is discovered first, then converted into a's dtype by PutMaskCore after the size check.
            // A StringDType target converts the list with its own descriptor (every element by setitem, as NumPy's
            // PyArray_FromAny(values, dtype) does); the discovered U / number array would be cast instead.
            UnmanagedStorage.PutMaskCore(a, mask, a is not null && a.Storage.IsVString ? Backends.Strings.NDVStringMove.StoreValue(values, a.dtype) : np.array(values), sequence: true);
        }
    }
}
