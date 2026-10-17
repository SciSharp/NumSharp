using System;
using NumSharp.Backends.Strings;

// =============================================================================
// np.strings.Translate.cs — numpy.strings.translate (strings phase 2, P2.6 milestone 3)
// =============================================================================
//
// NumPy's translate(a, table, deletechars=None) is `a_arr = np.asarray(a)`, then _vec_string over str.translate (U,
// StringDType) or bytes.translate (S) with the broadcast arguments: not a ufunc (no out= / where= / dtype=), a fresh
// C-ordered result of a's own dtype over the broadcast shape. np.asarray drops a chararray's class, so a chararray operand
// gives a plain array. The engine is NDCharObjectPath.Translate (the arguments, the one-table kernel route and the C-order
// walk) over NDTranslate.cs's compiled tables.
// =============================================================================

namespace NumSharp
{
    public static partial class np
    {
        public static partial class strings
        {
            /// <summary>
            ///     NumPy's <c>np.strings.translate</c>: every character of each element mapped through
            ///     <paramref name="table"/> — CPython's <c>str.translate</c> for <c>U</c> and <c>StringDType</c> (a dict keyed by
            ///     code point: a str value replaces the character, an int names a code point, null deletes it, a missing key
            ///     keeps it), <c>bytes.translate</c> for <c>S</c> (a 256-byte table, or null; then the bytes of
            ///     <paramref name="deletechars"/> are deleted).
            /// </summary>
            /// <param name="a">The strings: <c>S</c>, <c>U</c> or <c>StringDType</c> (a C# <c>string</c> converts to a 0-d
            ///     <c>U</c> array).</param>
            /// <param name="table">
            ///     The table, read as the Python object it stands for (NumPy converts it with <c>np.asarray</c> and broadcasts
            ///     it against <paramref name="a"/>, one table per element): an <see cref="System.Collections.IDictionary"/> is a
            ///     dict (keys equal to an integer under Python's <c>==</c> — <c>97</c>, <c>97.0</c>, <c>true</c> for 1 — map
            ///     that code point; a <see cref="string"/> key never does); a <see cref="string"/> is a str and a <c>byte[]</c>
            ///     bytes (a str table maps code point <c>c</c> to <c>table[c]</c>; an <c>S</c> table must be exactly 256 bytes);
            ///     null is <c>None</c>; a C# array or list of tables is an array of tables; an <see cref="NDArray"/> is used as
            ///     it is.
            /// </param>
            /// <param name="deletechars">The bytes to delete (an <c>S</c> operand only): a <c>byte[]</c> or
            ///     <see cref="string"/> (its UCS-4 bytes, as NumPy reads an <c>np.str_</c>). Null (Python's <c>None</c>) passes
            ///     nothing. NumPy ignores it for <c>U</c>, and passes it to <c>str.translate</c> for <c>StringDType</c>, which
            ///     refuses it.</param>
            /// <returns>A fresh C-ordered array of <paramref name="a"/>'s dtype over the broadcast shape. An <c>S</c> /
            ///     <c>U</c> result keeps the element width, so a translation that grows is cut (<c>{97: "XYZ"}</c> over a
            ///     <c>U3</c> <c>"abc"</c> is <c>"XYZ"</c>). A <c>StringDType</c> result has <paramref name="a"/>'s NA object and
            ///     coercion.</returns>
            /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
            /// <exception cref="TypeError">
            ///     <paramref name="a"/> is not text (<c>string operation on non-string array</c>); a <c>U</c> table that cannot
            ///     be indexed (<c>'NoneType' object is not subscriptable</c>, raised only when a character is looked up); a
            ///     dict value that is not an int, None or str (<c>character mapping must return integer, None or str</c>); an
            ///     <c>S</c> table or delete set that is not bytes-like (<c>a bytes-like object is required, not 'dict'</c>); a
            ///     <c>StringDType</c> NA whose NA object is not a str (<c>descriptor 'translate' for 'str' objects doesn't
            ///     apply to a 'NoneType' object</c>); <paramref name="deletechars"/> with a <c>StringDType</c> operand
            ///     (<c>str.translate() takes exactly one argument (2 given)</c>); a <c>StringDType</c> result holding a lone
            ///     surrogate (<c>result array type does not match underlying function</c>).
            /// </exception>
            /// <exception cref="ValueError">A mapping outside <c>range(0x110000)</c> (<c>character mapping must be in
            ///     range(0x110000)</c>); an <c>S</c> table that is not 256 bytes long (<c>translation table must be 256
            ///     characters long</c>, raised for any non-empty array, even of empty strings).</exception>
            /// <exception cref="SystemError">A <c>U</c> unit past U+10FFFF that the call would keep in a str (CPython's
            ///     <c>invalid maximum character passed to PyUnicode_New</c>).</exception>
            /// <exception cref="IncorrectShapeException">The arguments do not broadcast (<c>shape mismatch: objects cannot be
            ///     broadcast to a single shape.  Mismatch is between arg 0 with shape (3,) and arg 1 with shape (2,).</c>).</exception>
            /// <exception cref="NotSupportedException">An argument with no NumSharp form (an object array of lists).</exception>
            /// <remarks>
            ///     <para>The element is the text without its trailing NULs for <c>S</c> / <c>U</c> (NumPy's scalar), the whole
            ///     text for <c>StringDType</c>. An error ends the call at the first element (C order) that raises, as in
            ///     NumPy.</para>
            ///     <para><c>np.strings.translate(np.array(new[] { "abc", "cab" }), new Dictionary&lt;int, string&gt; { [97] =
            ///     "x", [98] = null })</c> is <c>["xc", "cx"]</c> (<c>U3</c>). A C# <see cref="char"/> argument is NumSharp's
            ///     <c>Char</c> scalar, a 16-bit number to NumPy: as a <c>U</c> table it keeps every character.</para>
            /// </remarks>
            public static NDArray translate(NDArray a, object table, object deletechars = null)
            {
                if (NDCharArrayBoundary.Sees(a))
                    return NDCharArrayBoundary.Plain((a, table, deletechars), static s => translate(s.a, s.table, s.deletechars));
                if (a is null)
                    throw new ArgumentNullException(nameof(a));
                return NDCharObjectPath.Translate(a, table, deletechars);
            }
        }
    }
}
