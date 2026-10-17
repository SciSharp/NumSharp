using System;
using NumSharp.Backends.Strings;

// =============================================================================
// np.char.transform.cs — numpy.char's partition wrappers and the padding functions it re-exports from numpy.strings
// =============================================================================
//
// numpy/_core/defchararray.py (NumPy 2.4.2) imports center, ljust, rjust, zfill and expandtabs from numpy.strings, so
// np.char.center IS np.strings.center (the same function object). numpy.char defines its own partition and rpartition:
//
//     def partition(a, sep):  return np.stack(strings_partition(a, sep), axis=-1)
//     def rpartition(a, sep): return np.stack(strings_rpartition(a, sep), axis=-1)
//
// (np.strings returns three arrays, numpy.char one array with a last axis of length 3). The re-exports below convert
// their array_likes with the module's text rule (CharModule.TextOperand: a byte[] is Python bytes) and call the
// np.strings function, so every answer — the result's class for a chararray operand, its layout, every error and its
// order — is np.strings', as it is in NumPy.
//
// =============================================================================

namespace NumSharp
{
    public sealed partial class CharModule
    {
        // =============================================================================================================
        // partition / rpartition (numpy.char's own wrappers: the three parts stacked on a new last axis)
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>np.char.partition(a, sep)</c>: each element split around the FIRST occurrence of
        ///     <paramref name="sep"/> into the part before it, the separator and the part after it (without a match: the
        ///     element and two empty strings), stacked as ONE array whose last axis holds the three parts —
        ///     <c>np.stack(np.strings.partition(a, sep), axis=-1)</c>. <see cref="np.strings.partition"/> returns the three
        ///     parts as three arrays instead.
        /// </summary>
        /// <param name="a">The strings: an <c>S</c> / <c>U</c> / <c>StringDType</c> array or a text array_like (see the class
        ///     remarks). A number array is cast to text the way NumPy casts it (<c>np.char.partition(5, ' ')</c> raises
        ///     NumPy's <c>invalid literal for int()</c>).</param>
        /// <param name="sep">The separators, broadcast against <paramref name="a"/>: cast to <paramref name="a"/>'s dtype for
        ///     <c>S</c> / <c>U</c> (a number is its text, <c>1</c> is <c>'1'</c>); a <c>StringDType</c>
        ///     <paramref name="a"/> needs a <c>StringDType</c> separator (NumPy 2.4.2 has no <c>(StringDType, str)</c>
        ///     loop).</param>
        /// <returns>A plain array of the broadcast shape plus a last axis of length 3, as wide as its widest part
        ///     (<c>S</c> / <c>U</c>) or <c>StringDType</c>; laid out as NumPy's <c>np.stack</c> lays out the parts (C order
        ///     for a C-ordered input). A chararray input still gives a plain array: <c>np.stack</c> keeps no
        ///     subclass.</returns>
        /// <exception cref="ArgumentNullException">An operand is null (see the class remarks).</exception>
        /// <exception cref="ValueError">An empty separator (<c>empty separator</c>); an empty broadcast (<c>zero-size array
        ///     to reduction operation maximum which has no identity</c>, NumPy's sizing); a number operand that cannot be
        ///     cast to the other's text (<c>invalid literal for int() with base 10: np.str_(' ')</c>); a
        ///     <c>StringDType</c> NA whose NA object is not a string.</exception>
        /// <exception cref="TypeError">No common dtype or no loop for the classes (NumPy's texts: a <c>StringDType</c>
        ///     array with a <c>U</c> separator, a bytes separator beside <c>StringDType</c>); two different
        ///     <c>StringDType</c> NA objects.</exception>
        /// <exception cref="UnicodeEncodeError">A non-ASCII separator cast to <c>S</c>.</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        /// <remarks>
        ///     For <c>S</c> / <c>U</c> NumPy computes the three parts as field views of one structured array and stacks
        ///     them; NumSharp computes three arrays and stacks them the same way. The stacked result is a fresh copy in
        ///     both, with NumPy's dtype, values and strides (an F-ordered input gives NumPy's permuted layout, neither C nor
        ///     F).
        /// </remarks>
        [NDScoped]
        public NDArray partition(object a, object sep)
        {
            // NumPy's np.stack keeps no subclass of equal priority, so a chararray input gives a plain result whatever the
            // parts' class: the whole body computes plain (the parts are never wrapped as chararray views, and np.stack
            // meets no chararray rule).
            using var plain = NDCharArrayBoundary.Inside();
            var (before, middle, after) = np.strings.partition(TextOperand(a, nameof(a)), TextOperand(sep, nameof(sep)));
            return np.stack(new NDArray[] { before, middle, after }, -1);
        }

        /// <summary>
        ///     NumPy's <c>np.char.rpartition(a, sep)</c>: each element split around the LAST occurrence of
        ///     <paramref name="sep"/> into the part before it, the separator and the part after it (without a match: two
        ///     empty strings and the element), stacked as one array whose last axis holds the three parts —
        ///     <c>np.stack(np.strings.rpartition(a, sep), axis=-1)</c>.
        /// </summary>
        /// <param name="a">The strings (see <see cref="partition"/>).</param>
        /// <param name="sep">The separators, broadcast and cast as <see cref="partition"/> casts them.</param>
        /// <returns>A plain array of the broadcast shape plus a last axis of length 3 (see <see cref="partition"/>).</returns>
        /// <exception cref="ArgumentNullException">An operand is null.</exception>
        /// <exception cref="ValueError">An empty separator; an empty broadcast; a number operand that cannot be cast to the
        ///     other's text; a <c>StringDType</c> NA whose NA object is not a string.</exception>
        /// <exception cref="TypeError">No common dtype or no loop for the classes; two different <c>StringDType</c> NA
        ///     objects.</exception>
        /// <exception cref="UnicodeEncodeError">A non-ASCII separator cast to <c>S</c>.</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        [NDScoped]
        public NDArray rpartition(object a, object sep)
        {
            // A plain result for a chararray input too (see partition): the body computes plain.
            using var plain = NDCharArrayBoundary.Inside();
            var (before, middle, after) = np.strings.rpartition(TextOperand(a, nameof(a)), TextOperand(sep, nameof(sep)));
            return np.stack(new NDArray[] { before, middle, after }, -1);
        }

        // =============================================================================================================
        // center / ljust / rjust / zfill / expandtabs (numpy.strings' functions, re-exported)
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>np.char.center(a, width, fillchar=' ')</c>, which IS <see cref="np.strings.center"/>: each element
        ///     centered in a string of <paramref name="width"/> characters, padded with <paramref name="fillchar"/>. An
        ///     element longer than the width is unchanged.
        /// </summary>
        /// <param name="a">The strings: an <c>S</c> / <c>U</c> / <c>StringDType</c> array or a text array_like.</param>
        /// <param name="width">The widths: an integer array or a C# integer, broadcast. A float, a bool or text is NumPy's
        ///     TypeError.</param>
        /// <param name="fillchar">The fill characters, each exactly one character long, broadcast; null is NumPy's default
        ///     <c>' '</c>. A <c>str</c> fill beside an <c>S</c> operand is cast to bytes (ASCII only), a <c>bytes</c> fill
        ///     beside a <c>U</c> operand to text.</param>
        /// <returns>A fresh array: <paramref name="a"/>'s class at the widest result width for <c>S</c> / <c>U</c>, laid out
        ///     like <paramref name="a"/>; a <c>StringDType</c> array for <c>StringDType</c>. A chararray <paramref name="a"/>
        ///     gives a chararray view of the result (NumPy's <c>chararray.__array_wrap__</c>).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> or <paramref name="width"/> is null.</exception>
        /// <exception cref="TypeError"><paramref name="width"/> is not an integer array (<c>unsupported type float64 for
        ///     operand 'width'</c>); a fill that is not one character (<c>The fill character must be exactly one
        ///     character long</c>); a non-text operand; no common dtype or no loop for the classes; a uint64 width (NumPy's
        ///     <c>data type 'U6.0' not understood</c>).</exception>
        /// <exception cref="ValueError">An empty broadcast (<c>zero-size array to reduction operation maximum which has no
        ///     identity</c>); a <c>StringDType</c> NA its rule refuses.</exception>
        /// <exception cref="UnicodeEncodeError">A non-ASCII fill for an <c>S</c> operand.</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        /// <exception cref="OverflowException">A <c>StringDType</c> result past <c>sys.maxsize</c> bytes.</exception>
        [NDScoped]
        public NDArray center(object a, object width, object fillchar = null)
            => np.strings.center(TextOperand(a, nameof(a)), TextOperand(width, nameof(width)), OptionalOperand(fillchar));

        /// <summary>
        ///     NumPy's <c>np.char.ljust(a, width, fillchar=' ')</c>, which IS <see cref="np.strings.ljust"/>: each element
        ///     left-justified in a string of <paramref name="width"/> characters, padded on the right with
        ///     <paramref name="fillchar"/>.
        /// </summary>
        /// <param name="a">The strings: an <c>S</c> / <c>U</c> / <c>StringDType</c> array or a text array_like.</param>
        /// <param name="width">The integer widths, broadcast.</param>
        /// <param name="fillchar">The one-character fills, broadcast; null is NumPy's default <c>' '</c>.</param>
        /// <returns>A fresh array (see <see cref="center"/>).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> or <paramref name="width"/> is null.</exception>
        /// <exception cref="TypeError">See <see cref="center"/>.</exception>
        /// <exception cref="ValueError">An empty broadcast; a <c>StringDType</c> NA its rule refuses.</exception>
        /// <exception cref="UnicodeEncodeError">A non-ASCII fill for an <c>S</c> operand.</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        /// <exception cref="OverflowException">A <c>StringDType</c> result past <c>sys.maxsize</c> bytes.</exception>
        [NDScoped]
        public NDArray ljust(object a, object width, object fillchar = null)
            => np.strings.ljust(TextOperand(a, nameof(a)), TextOperand(width, nameof(width)), OptionalOperand(fillchar));

        /// <summary>
        ///     NumPy's <c>np.char.rjust(a, width, fillchar=' ')</c>, which IS <see cref="np.strings.rjust"/>: each element
        ///     right-justified in a string of <paramref name="width"/> characters, padded on the left with
        ///     <paramref name="fillchar"/>.
        /// </summary>
        /// <param name="a">The strings: an <c>S</c> / <c>U</c> / <c>StringDType</c> array or a text array_like.</param>
        /// <param name="width">The integer widths, broadcast.</param>
        /// <param name="fillchar">The one-character fills, broadcast; null is NumPy's default <c>' '</c>.</param>
        /// <returns>A fresh array (see <see cref="center"/>).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> or <paramref name="width"/> is null.</exception>
        /// <exception cref="TypeError">See <see cref="center"/>.</exception>
        /// <exception cref="ValueError">An empty broadcast; a <c>StringDType</c> NA its rule refuses.</exception>
        /// <exception cref="UnicodeEncodeError">A non-ASCII fill for an <c>S</c> operand.</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        /// <exception cref="OverflowException">A <c>StringDType</c> result past <c>sys.maxsize</c> bytes.</exception>
        [NDScoped]
        public NDArray rjust(object a, object width, object fillchar = null)
            => np.strings.rjust(TextOperand(a, nameof(a)), TextOperand(width, nameof(width)), OptionalOperand(fillchar));

        /// <summary>
        ///     NumPy's <c>np.char.zfill(a, width)</c>, which IS <see cref="np.strings.zfill"/>: each element left-filled with
        ///     <c>'0'</c> to <paramref name="width"/> characters, a leading <c>'+'</c> / <c>'-'</c> moved in front of the
        ///     zeros (<c>zfill('-42', 5)</c> is <c>'-0042'</c>).
        /// </summary>
        /// <param name="a">The strings: an <c>S</c> / <c>U</c> / <c>StringDType</c> array or a text array_like.</param>
        /// <param name="width">The integer widths, broadcast.</param>
        /// <returns>A fresh array (for <c>S</c> / <c>U</c> at the widest result width, laid out like <paramref name="a"/>;
        ///     a chararray view of it for a chararray <paramref name="a"/>).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> or <paramref name="width"/> is null.</exception>
        /// <exception cref="TypeError"><paramref name="width"/> is not an integer array (<c>unsupported type &lt;U1 for
        ///     operand 'width'</c>); a non-text operand; a uint64 width; a <c>StringDType</c> NA that is neither a string
        ///     nor NaN-like.</exception>
        /// <exception cref="ValueError">An empty broadcast.</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        [NDScoped]
        public NDArray zfill(object a, object width)
            => np.strings.zfill(TextOperand(a, nameof(a)), TextOperand(width, nameof(width)));

        /// <summary>
        ///     NumPy's <c>np.char.expandtabs(a, tabsize=8)</c>, which IS <see cref="np.strings.expandtabs"/>: each tab
        ///     replaced by spaces up to the next multiple of <paramref name="tabsize"/> columns (the column restarts after
        ///     <c>'\n'</c> and <c>'\r'</c>; a tab size of 0 or less removes the tabs).
        /// </summary>
        /// <param name="a">The strings: an <c>S</c> / <c>U</c> / <c>StringDType</c> array or a text array_like.</param>
        /// <param name="tabsize">The tab sizes (any integer dtype), broadcast; null is NumPy's default 8.</param>
        /// <returns>A fresh array (for <c>S</c> / <c>U</c> at the widest result width as NumPy sizes it: a tab's columns
        ///     plus every other character's BYTES, so a <c>U</c> result is wider than its text).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="TypeError">A non-text operand (NumPy's no-loop text).</exception>
        /// <exception cref="ArgumentException">A tab size that cannot reach int64 by a same_kind cast (a float: <c>Cannot
        ///     cast ufunc '_expandtabs_length' input 1 from dtype('float64') to dtype('int64') with casting rule
        ///     'same_kind'</c>).</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        /// <exception cref="OverflowException">An expanded string longer than 2^31 - 1 (<c>new string is too
        ///     long</c>).</exception>
        /// <exception cref="ValueError">An empty broadcast; a <c>StringDType</c> NA whose NA object is not a string.</exception>
        [NDScoped]
        public NDArray expandtabs(object a, object tabsize = null)
            => np.strings.expandtabs(TextOperand(a, nameof(a)), OptionalOperand(tabsize));

        // =============================================================================================================
        // Shared pieces
        // =============================================================================================================

        /// <summary>
        ///     An optional array_like argument whose C# <c>null</c> is NumPy's default (a fill character, a tab size): null
        ///     stays null, so the <see cref="np.strings"/> function applies NumPy's default; anything else is converted by
        ///     <see cref="TextOperand"/>.
        /// </summary>
        /// <param name="value">The argument, or null for NumPy's default.</param>
        /// <returns>The array, or null.</returns>
        /// <exception cref="NotSupportedException"><paramref name="value"/> has no NumSharp array form.</exception>
        /// <remarks>C# cannot tell an omitted optional <c>object</c> argument from an explicit <c>null</c>, so NumPy's
        ///     explicit <c>None</c> (an object array, NumPy's own TypeError in these functions) reads as the default
        ///     here.</remarks>
        internal static NDArray OptionalOperand(object value) => value is null ? null : TextOperand(value, nameof(value));
    }
}
