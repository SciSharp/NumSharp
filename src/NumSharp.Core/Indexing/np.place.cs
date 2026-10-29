using System;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Change elements of <paramref name="arr"/> based on a boolean mask.
        ///     Where <paramref name="mask"/> is true (walked in C-order), the
        ///     next value from <paramref name="vals"/> (cycling) is written into
        ///     <paramref name="arr"/>. In-place.
        /// </summary>
        /// <param name="arr">Target array (modified in place).</param>
        /// <param name="mask">
        ///     Boolean mask. Must have the same total <em>size</em> as
        ///     <paramref name="arr"/> — NumPy allows shape mismatch as long as
        ///     element counts match.
        /// </param>
        /// <param name="vals">
        ///     Values to write, converted to <paramref name="arr"/>'s dtype and cycled to fill the True positions. An
        ///     ARRAY of values must convert under NumPy's <c>'safe'</c> rule (a 0-d value — NumSharp's stand-in for a
        ///     Python scalar — is converted unconditionally). Must be non-empty when at least one mask entry is True.
        ///     <c>null</c> is Python's <c>None</c>: one value, converted into <paramref name="arr"/>'s dtype after the mask's
        ///     size check (<see cref="NDArray.NoneForStore"/>: NaN, <c>nan+nanj</c>, <c>False</c>, <c>'None'</c>; int()'s
        ///     TypeError for an integer array, even under an all-false mask).
        /// </param>
        /// <exception cref="ArgumentNullException"><paramref name="arr"/> or <paramref name="mask"/> is null.</exception>
        /// <exception cref="TypeError"><paramref name="vals"/> is null (<c>None</c>) and <paramref name="arr"/> is an
        ///     integer (or <c>decimal</c>) array.</exception>
        /// <exception cref="ValueError"><paramref name="arr"/> is read-only (checked first, as NumPy does), or
        ///     <paramref name="mask"/>'s size differs from <paramref name="arr"/>'s (<c>place: mask and data must be the
        ///     same size</c>), or <paramref name="vals"/> is empty while some mask entry is True (<c>Cannot insert from an
        ///     empty array!</c>).</exception>
        /// <exception cref="InvalidCastException">An array <paramref name="vals"/> does not cast 'safely' to
        ///     <paramref name="arr"/>'s dtype (NumPy's TypeError: <c>Cannot cast array data from dtype('float64') to
        ///     dtype('int64') according to the rule 'safe'</c>) — even when <paramref name="arr"/> is empty.</exception>
        /// <exception cref="UnicodeEncodeError">Text values that are not ASCII into an <c>S</c> array.</exception>
        /// <exception cref="UnicodeDecodeError">Byte values that are not ASCII into a <c>U</c> array.</exception>
        /// <exception cref="ValueError">A text value that does not parse into a numeric array (NumPy's <c>int()</c> /
        ///     <c>float()</c> / <c>complex()</c> texts); numbers into a text array are written as their <c>str()</c>.</exception>
        /// <exception cref="SystemError">A <c>U</c> mask holds a unit above U+10FFFF: NumPy's string-to-bool cast of the mask
        ///     raises it, before the size check. (A text mask is otherwise read by its truth values: a non-empty string is
        ///     True.)</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.place.html
        ///     <para><b>Order (NumPy's <c>arr_place</c>).</b> The target's writeability; the mask's conversion to bool and its
        ///     size; the values' conversion (<c>PyArray_FromAny(vals, arr.dtype, CARRAY)</c>: the 'safe' rule for an array,
        ///     every value converted — so a non-ASCII byte value raises even under an all-false mask); then the empty-values
        ///     check and the loop. A NUMBER formatted into a text array cannot fail, so it is refused only where it is
        ///     written; a TEXT mask is refused where its truth is read (the empty-values check, or the loop).</para>
        ///     <para>Scoped by <c>[NDScoped]</c>: the contiguous scratch of a non-contiguous target and the
        ///     bool/dtype-converted copies of <paramref name="mask"/> and <paramref name="vals"/> are released on
        ///     return, exceptions included. A mask or values array used as-is is the caller's (built before the scope
        ///     opened), so the scope never disposes it.</para>
        /// </remarks>
        public static void place(NDArray arr, NDArray mask, NDArray vals)
            => UnmanagedStorage.PlaceCore(arr, mask, vals, sequence: false);

        /// <summary>
        ///     <see cref="place(NDArray, NDArray, NDArray)"/> with the values given as a C# array — NumPy's Python LIST of
        ///     values: <c>np.place(a, mask, ['P', 'QQ', 'RRRR'])</c> is <c>np.place(a, mask, new[] { "P", "QQ", "RRRR" })</c>.
        /// </summary>
        /// <param name="arr">Target array (modified in place).</param>
        /// <param name="mask">Boolean mask of <paramref name="arr"/>'s size.</param>
        /// <param name="vals">
        ///     The values, cycled over the True positions. NumPy converts a list STRAIGHT INTO <paramref name="arr"/>'s dtype
        ///     (<c>PyArray_FromAny(list, arr.dtype)</c>, as <c>np.array(vals, dtype=arr.dtype)</c> does), so there is no
        ///     <c>'safe'</c> rule: <c>'RRRR'</c> into a <c>&lt;U3</c> array truncates to <c>'RRR'</c>, <c>1.9</c> into an int
        ///     array truncates to 1 — where an ndarray of the same values raises the cast TypeError.
        /// </param>
        /// <exception cref="ArgumentNullException"><paramref name="arr"/> or <paramref name="mask"/> is null.</exception>
        /// <exception cref="TypeError"><paramref name="vals"/> is null (<c>None</c>) and <paramref name="arr"/> is an
        ///     integer (or <c>decimal</c>) array.</exception>
        /// <exception cref="ValueError">The checks of <see cref="place(NDArray, NDArray, NDArray)"/>.</exception>
        /// <exception cref="UnicodeEncodeError">Text values that are not ASCII into an <c>S</c> array.</exception>
        /// <exception cref="UnicodeDecodeError">Byte values that are not ASCII into a <c>U</c> array.</exception>
        /// <exception cref="ValueError">A text value that does not parse into a numeric array.</exception>
        /// <remarks>
        ///     C# binds a typed array argument (<c>double[]</c>, <c>string[]</c>) here rather than to the
        ///     <see cref="NDArray"/> overload: an identity/reference conversion beats NumSharp's implicit
        ///     <c>Array → NDArray</c> conversion. Pass <c>np.array(...)</c> to get NumPy's ndarray semantics. A bare
        ///     <c>null</c> binds here too, and is Python's <c>None</c> exactly as in the <see cref="NDArray"/> overload.
        /// </remarks>
        [NDScoped]
        public static void place(NDArray arr, NDArray mask, Array vals)
        {
            // None: PlaceCore converts it after the mask checks (a null values array is None there).
            if (vals is null)
            {
                UnmanagedStorage.PlaceCore(arr, mask, null, sequence: false);
                return;
            }
            // The list's own dtype is discovered first (np.array), then converted into arr's dtype by PlaceCore after the
            // size check, as NumPy's conversion follows its mask checks. The discovered array is released with the scope.
            // A StringDType target converts the list with its own descriptor (every element by setitem, as NumPy's
            // PyArray_FromAny(vals, dtype) does); the discovered U / number array would be cast instead.
            UnmanagedStorage.PlaceCore(arr, mask, arr is not null && arr.Storage.IsVString ? Backends.Strings.NDVStringMove.StoreValue(vals, arr.dtype) : np.array(vals), sequence: true);
        }
    }
}
