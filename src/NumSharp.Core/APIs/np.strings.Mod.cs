using System;
using NumSharp.Backends.Strings;

// =============================================================================
// np.strings.Mod.cs — numpy.strings.mod (strings phase 2, P2.6 milestone 3)
// =============================================================================
//
// NumPy's mod(a, values) is _to_bytes_or_str_array(_vec_string(a, np.object_, '__mod__', (values,)), a): CPython's %
// formatting per element (bytes.__mod__ for S, str.__mod__ for U and StringDType) over the broadcast of a and np.asarray(
// values), then the results as a U / S array as wide as the longest one (StringDType: a default StringDType array). Not a
// ufunc (no out= / where= / dtype=). np.asarray drops a chararray's class, so a chararray operand gives a plain array. The
// engine is NDCharObjectPath.Mod (the C-order walk and the per-element formatting) over NDStringMod.cs's compiled formats.
// =============================================================================

namespace NumSharp
{
    public static partial class np
    {
        public static partial class strings
        {
            /// <summary>
            ///     NumPy's <c>np.strings.mod</c>: <c>a % values</c> element-wise — CPython's <c>%</c> formatting of each element of
            ///     <paramref name="a"/> with the broadcast element of <paramref name="values"/> (<c>bytes.__mod__</c> for
            ///     <c>S</c>, <c>str.__mod__</c> for <c>U</c> and <c>StringDType</c>).
            /// </summary>
            /// <param name="a">The format strings: <c>S</c>, <c>U</c> or <c>StringDType</c> (a C# <c>string</c> converts to a
            ///     0-d <c>U</c> array).</param>
            /// <param name="values">
            ///     The values, converted as NumPy's <c>np.asarray</c> converts them and broadcast against <paramref name="a"/>,
            ///     ONE value per element: an <see cref="NDArray"/> as it is (each element reaches <c>%</c> as its NumPy scalar —
            ///     <c>%r</c> of an int64 element is <c>np.int64(5)</c>); a C# number as a Python number (an integer is int64, a
            ///     <c>float</c> / <c>double</c> float64); a <see cref="string"/> a str; a <c>byte[]</c> bytes; a <c>bool</c> a
            ///     bool; a C# array, list or tuple of values an array of them — a tuple is a SEQUENCE, as NumPy's
            ///     <c>np.asarray</c> reads one, never one tuple of arguments per element (that takes an object array of tuples,
            ///     which no C# value spells); null Python's <c>None</c>; an <see cref="System.Collections.IDictionary"/> a dict,
            ///     which feeds <c>%(key)s</c> specs; a list holding a null, a dict or a Python int past uint64 NumPy's object
            ///     array, whose elements are those objects.
            /// </param>
            /// <returns>A fresh C-ordered array over the broadcast shape: <c>U</c> / <c>S</c> (by <paramref name="a"/>'s kind) as
            ///     wide as the longest result, at least 1; a default <c>StringDType</c> array (no NA object, whatever
            ///     <paramref name="a"/>'s descriptor holds) for a <c>StringDType</c> operand; an empty array of
            ///     <paramref name="a"/>'s own dtype when the broadcast is empty.</returns>
            /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
            /// <exception cref="TypeError">
            ///     <paramref name="a"/> is not text (<c>string operation on non-string array</c>); the value count does not
            ///     fit the format (<c>not enough arguments for format string</c>, <c>not all arguments converted during string
            ///     formatting</c> — a NumPy numeric scalar, a bytes value, a list or a dict is a mapping and never raises the
            ///     second); a conversion the value's type refuses (<c>%x format: an integer is required, not numpy.float64</c>,
            ///     <c>%c requires int or char</c>, <c>%b requires a bytes-like object, or an object that implements __bytes__,
            ///     not 'str'</c>, <c>float argument required, not str</c>); <c>%(key)s</c> without a mapping (<c>format requires
            ///     a mapping</c>); a <c>*</c> width that is not a Python int (<c>* wants int</c>); a <c>StringDType</c> NA that is
            ///     not a str (<c>descriptor '__mod__' requires a 'str' object but received a 'NoneType'</c>); a
            ///     <c>StringDType</c> result holding a surrogate (<c>Invalid unicode code point found</c>).
            /// </exception>
            /// <exception cref="ValueError">A malformed format (<c>incomplete format</c>, <c>incomplete format key</c>,
            ///     <c>unsupported format character 'y' (0x79) at index 1</c>, <c>width too big</c>, <c>precision too big</c>);
            ///     a NaN under an integer conversion (<c>cannot convert float NaN to integer</c>); a ragged
            ///     <paramref name="values"/> list.</exception>
            /// <exception cref="OverflowException">An infinity under an integer conversion; a <c>%c</c> argument out of range
            ///     (<c>%c arg not in range(0x110000)</c>, <c>%c arg not in range(256)</c>); an int too large for a float;
            ///     <c>precision too large</c>; an <c>S</c> format's unsupported conversion byte past 0x7F (<c>character
            ///     argument not in range(0x110000)</c>).</exception>
            /// <exception cref="IndexError">A <c>%(key)s</c> lookup on a NumPy scalar (<c>invalid index to scalar
            ///     variable.</c>).</exception>
            /// <exception cref="KeyError">A <c>%(key)s</c> missing from a dict (the text is the key's repr).</exception>
            /// <exception cref="SystemError">A <c>U</c> element of one unit past U+10FFFF (CPython's <c>invalid maximum character
            ///     passed to PyUnicode_New</c>).</exception>
            /// <exception cref="IncorrectShapeException">The operands do not broadcast (<c>shape mismatch: objects cannot be
            ///     broadcast to a single shape.  Mismatch is between arg 0 with shape (2,) and arg 1 with shape (3,).</c>).</exception>
            /// <exception cref="NotSupportedException">A value with no NumSharp form (an object array of lists).</exception>
            /// <remarks>
            ///     <para>The format element is the text without its trailing NULs for <c>S</c> / <c>U</c> (NumPy's scalar), the
            ///     whole text for <c>StringDType</c>. An error ends the call at the first element (C order) that raises, as in
            ///     NumPy; CPython parses as it formats, so <c>'%s %y'</c> over one value is the argument-count error and
            ///     <c>'%y %s'</c> the unsupported character.</para>
            ///     <para><c>np.strings.mod(np.array(new[] { "%d bytes", "%.2f%%" }), np.array(new[] { 8.0, 99.5 }))</c> is
            ///     <c>["8 bytes", "99.50%"]</c> (<c>U7</c>). Over <c>S</c>, <c>%s</c> writes a NumPy scalar's raw bytes (the buffer
            ///     protocol): <c>b"%s"</c> of an int64 5 is <c>b"\x05\x00\x00\x00\x00\x00\x00\x00"</c>, read back as
            ///     <c>b"\x05"</c> in an <c>S8</c> result.</para>
            /// </remarks>
            public static NDArray mod(NDArray a, object values)
            {
                if (NDCharArrayBoundary.Sees(a))
                    return NDCharArrayBoundary.Plain((a, values), static s => mod(s.a, s.values));
                if (a is null)
                    throw new ArgumentNullException(nameof(a));
                return NDCharObjectPath.Mod(a, values);
            }
        }
    }
}
