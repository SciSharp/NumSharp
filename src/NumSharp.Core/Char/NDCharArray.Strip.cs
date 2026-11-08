using System;
using System.Runtime.CompilerServices;

// =============================================================================
// NDCharArray.Strip.cs — chararray.__getitem__'s rstrip(): Python's whitespace for str, ASCII's for bytes
// =============================================================================
//
// NumPy's chararray.__getitem__ returns `val.rstrip()` for a scalar: the NumPy scalar's own rstrip, i.e. Python's
// str.rstrip() / bytes.rstrip() with no argument. The two classes strip DIFFERENT sets:
//
//   str    Py_UNICODE_ISSPACE: 29 code points (CPython 3.12, Unicode 15.0.0's White_Space property):
//          U+0009..U+000D, U+001C..U+001F, U+0020, U+0085, U+00A0, U+1680, U+2000..U+200A, U+2028, U+2029, U+202F,
//          U+205F, U+3000
//   bytes  Py_ISSPACE: the 6 ASCII bytes \t \n \v \f \r and the space
//
// Trailing NULs are gone before rstrip runs: NumPy's scalar of a fixed-width element drops them. A NUL is not
// whitespace, so 'ab\0 ' keeps its NUL ('ab\0'), and the strip stops there.
//
// These sets are NOT the np.char comparisons' set: those strip NUL and the ASCII whitespace for both classes
// (Buffer::rstrip, ILKernelGenerator.Text.Strip.cs), so '\x1c' and U+3000 are kept by np.char.equal and stripped
// here. The loops below are per ELEMENT (one element access, one printed or iterated value) and touch only its own
// code units.
//
// =============================================================================

namespace NumSharp
{
    public partial class NDCharArray
    {
        /// <summary>
        ///     CPython's <c>Py_UNICODE_ISSPACE</c> for one code point: the 29 whitespace code points <c>str.rstrip()</c>
        ///     removes. A code point above the BMP is never whitespace.
        /// </summary>
        /// <param name="codePoint">The code point (a <c>U</c> code unit).</param>
        /// <returns>True for whitespace.</returns>
        /// <remarks>A unit test compares this predicate with .NET's <see cref="char.IsWhiteSpace(char)"/> plus the four
        ///     information separators <c>U+001C</c>..<c>U+001F</c> over the whole BMP (the identity the loadtxt tokenizer
        ///     relies on, verified there exhaustively).</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool IsPyWhitespace(uint codePoint)
        {
            if (codePoint <= 0x20)
                return codePoint == 0x20 || codePoint - 0x09 <= 0x0D - 0x09 || codePoint - 0x1C <= 0x1F - 0x1C;
            if (codePoint < 0x85)
                return false;
            return codePoint == 0x85 || codePoint == 0xA0 || codePoint == 0x1680 || codePoint - 0x2000 <= 0x200A - 0x2000
                   || codePoint == 0x2028 || codePoint == 0x2029 || codePoint == 0x202F || codePoint == 0x205F
                   || codePoint == 0x3000;
        }

        /// <summary>CPython's <c>Py_ISSPACE</c> for one byte: the 6 ASCII whitespace bytes <c>bytes.rstrip()</c> removes.</summary>
        /// <param name="b">The byte.</param>
        /// <returns>True for <c>\t \n \v \f \r</c> and the space.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool IsBytesWhitespace(byte b) => b == 0x20 || (uint)(b - 0x09) <= 0x0D - 0x09;

        /// <summary>
        ///     The number of code units NumPy's stripped scalar keeps from one element: trailing NULs dropped (the scalar's
        ///     padding rule), then the trailing whitespace of the element's class (<see cref="IsPyWhitespace"/> /
        ///     <see cref="IsBytesWhitespace"/>).
        /// </summary>
        /// <param name="element">The element's first byte.</param>
        /// <param name="chars">The element width in code units.</param>
        /// <param name="unicode">True for <c>U</c> (4-byte code units), false for <c>S</c>.</param>
        /// <returns>The kept length in code units (0 for an element that is all NULs and whitespace).</returns>
        internal static unsafe int KeptUnits(byte* element, int chars, bool unicode)
        {
            int n = TrimmedUnits(element, chars, unicode);
            if (unicode)
            {
                uint* units = (uint*)element;
                while (n > 0 && IsPyWhitespace(units[n - 1]))
                    n--;
            }
            else
            {
                while (n > 0 && IsBytesWhitespace(element[n - 1]))
                    n--;
            }

            return n;
        }

        /// <summary>The number of code units left once the trailing NULs (the fixed-width padding) are dropped.</summary>
        /// <param name="element">The element's first byte.</param>
        /// <param name="chars">The element width in code units.</param>
        /// <param name="unicode">True for <c>U</c>, false for <c>S</c>.</param>
        /// <returns>The length of the NumPy scalar's value in code units.</returns>
        /// <remarks>
        ///     Aggressively inlined because np.strings.encode / decode call it once per element from fully optimized runs,
        ///     which decline to inline a method with a loop on their own: the call then reached a tier-0 copy of this method
        ///     for a short-lived process's first calls (a cp1252 encode of 100K accented words: 58 ns per element on calls
        ///     2-12, 37 once tier 1 is reached).
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static unsafe int TrimmedUnits(byte* element, int chars, bool unicode)
        {
            int n = chars;
            if (unicode)
            {
                uint* units = (uint*)element;
                while (n > 0 && units[n - 1] == 0)
                    n--;
            }
            else
            {
                while (n > 0 && element[n - 1] == 0)
                    n--;
            }

            return n;
        }

        /// <summary>
        ///     Python's <c>str.rstrip()</c>: <paramref name="text"/> without its trailing whitespace
        ///     (<see cref="IsPyWhitespace"/>). A surrogate is never whitespace, so an astral character stops the strip.
        /// </summary>
        /// <param name="text">The decoded element (trailing NULs already dropped).</param>
        /// <returns>The stripped text; <paramref name="text"/> itself when nothing is stripped.</returns>
        internal static string PyRStrip(string text)
        {
            int n = text.Length;
            while (n > 0 && IsPyWhitespace(text[n - 1]))
                n--;
            return n == text.Length ? text : text.Substring(0, n);
        }

        /// <summary>Python's <c>bytes.rstrip()</c>: <paramref name="bytes"/> without its trailing ASCII whitespace.</summary>
        /// <param name="bytes">The decoded element (trailing NULs already dropped).</param>
        /// <returns>The stripped bytes; <paramref name="bytes"/> itself when nothing is stripped.</returns>
        internal static byte[] BytesRStrip(byte[] bytes)
        {
            int n = bytes.Length;
            while (n > 0 && IsBytesWhitespace(bytes[n - 1]))
                n--;
            return n == bytes.Length ? bytes : bytes.AsSpan(0, n).ToArray();
        }

        /// <summary>
        ///     <c>chararray.__getitem__</c>'s strip of a decoded element value: a <see cref="string"/> by
        ///     <see cref="PyRStrip"/>, a <c>byte[]</c> by <see cref="BytesRStrip"/>, any other value (a number of a chararray
        ///     over int8 data) unchanged.
        /// </summary>
        /// <param name="value">The element as the storage decodes it.</param>
        /// <returns>The value NumPy's element access returns.</returns>
        internal static object StripValue(object value) => value switch
        {
            string text => PyRStrip(text),
            byte[] bytes => BytesRStrip(bytes),
            _ => value,
        };
    }
}
