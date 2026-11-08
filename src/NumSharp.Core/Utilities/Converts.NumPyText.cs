using System;
using System.Globalization;
using System.Numerics;
using System.Runtime.CompilerServices;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Printing;
using NumSharp.Backends.Strings;

namespace NumSharp.Utilities
{
    /// <summary>
    ///     The text side of <see cref="Converts"/>: every conversion between a C# <see cref="string"/> (NumPy's Python
    ///     <c>str</c>) and a value of one of NumSharp's dtypes, written as NumPy writes it.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Number → text</b> is the NumPy scalar's <c>str()</c> — the text <c>astype('U')</c> writes for the element
    ///     (<see cref="NDTextFormat"/>): <c>1.0</c> → <c>'1.0'</c>, <c>1e20</c> → <c>'1e+20'</c>, NaN → <c>'nan'</c>, a
    ///     float32 or float16 value with its own shortest digits (<c>np.float32(0.1)</c> → <c>'0.1'</c>), a complex value
    ///     as <c>'(1+2j)'</c>, a bool as <c>'True'</c> / <c>'False'</c>. NumSharp's two dtypes without a NumPy analog
    ///     follow their documented text rules: a <c>decimal</c> is written as the float64 it converts to, and a
    ///     <c>char</c> (NumSharp's 16-bit code unit, a uint16 in every type rule) as its number (<c>'a'</c> →
    ///     <c>'97'</c>), as <c>np.array(new[] { 'a' }).astype("U")</c> gives.
    ///     </para>
    ///     <para>
    ///     <b>Text → number</b> is NumPy's store of a Python <c>str</c> into the dtype (<c>a[0] = '1.5'</c>):
    ///     CPython 3.12's <c>int()</c> / <c>float()</c> / <c>complex()</c> grammar (whitespace, underscores between digits,
    ///     non-ASCII decimal digits, <c>inf</c> / <c>nan</c> in any case), NumPy's packing check for an integer target, and
    ///     NumPy's error texts quoting the str (<see cref="NDTextParse.ParseObject(string, NPTypeCode, byte*)"/>). A bool
    ///     target takes the str's truth value (any non-empty str is True — <c>'False'</c> included — as NumPy 2.x's
    ///     <c>str → bool</c> cast and <c>BOOL_setitem</c> give).
    ///     </para>
    ///     <para>
    ///     Both directions are culture-invariant: NumPy's text has no culture. These replace the .NET forms the class
    ///     carried before (<c>double.ToString(CultureInfo.CurrentCulture)</c>, <c>int.Parse(s, CurrentCulture)</c>), whose
    ///     result depended on the machine's culture (under de-DE <c>"1.5"</c> parsed as 15) and differed from every NumPy
    ///     route NumSharp already runs (<c>astype</c>, setitem, <c>np.array</c>, <c>fill</c>).
    ///     </para>
    /// </remarks>
    public static partial class Converts
    {
        /// <summary>
        ///     NumPy's <c>str()</c> of a value of one of NumSharp's 15 dtypes — the text <c>astype('U')</c> writes for it; a
        ///     <see cref="string"/> is returned as it is.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns>The text, or null when <paramref name="value"/> is not a value of a NumSharp dtype or a string (a
        ///     <see cref="DateTime"/>, an array, null): the caller keeps its own rule for those.</returns>
        /// <remarks>See the class remarks for the rule per dtype.</remarks>
        internal static string NumPyStr(object value) => value switch
        {
            string s => s,
            bool b => NumPyStrBool(b),
            sbyte v => NumPyStrSigned(v),
            short v => NumPyStrSigned(v),
            int v => NumPyStrSigned(v),
            long v => NumPyStrSigned(v),
            byte v => NumPyStrUnsigned(v),
            ushort v => NumPyStrUnsigned(v),
            uint v => NumPyStrUnsigned(v),
            ulong v => NumPyStrUnsigned(v),
            // NumSharp's char dtype crosses to text as its uint16 number (NDTextFormat.KindOf(Char) = Unsigned).
            char v => NumPyStrUnsigned(v),
            Half v => NumPyStrFloat((double)v, FloatKind.Half),
            float v => NumPyStrFloat(v, FloatKind.Single),
            double v => NumPyStrFloat(v, FloatKind.Double),
            // The decimal → float64 conversion every NumSharp text route uses (NDTextFormat.KindOf(Decimal) = Double).
            decimal v => NumPyStrFloat(ToDouble(v), FloatKind.Double),
            Complex v => NumPyStrComplex(v),
            _ => null,
        };

        /// <summary><c>str(np.bool_)</c>: <c>"True"</c> or <c>"False"</c>.</summary>
        /// <param name="value">The value.</param>
        /// <returns>The text.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static string NumPyStrBool(bool value) => value ? "True" : "False";

        /// <summary><c>str()</c> of a signed integer: its decimal digits, a leading <c>-</c> when negative (no culture).</summary>
        /// <param name="value">The value (any signed width, widened).</param>
        /// <returns>The text.</returns>
        internal static string NumPyStrSigned(long value)
        {
            Span<char> buf = stackalloc char[24];
            int n = NDTextFormat.FormatInt64(value, buf);
            return new string(buf.Slice(0, n));
        }

        /// <summary><c>str()</c> of an unsigned integer (or NumSharp's char as its uint16): its decimal digits.</summary>
        /// <param name="value">The value (any unsigned width, widened).</param>
        /// <returns>The text.</returns>
        internal static string NumPyStrUnsigned(ulong value)
        {
            Span<char> buf = stackalloc char[24];
            int n = NDTextFormat.FormatUInt64(value, buf);
            return new string(buf.Slice(0, n));
        }

        /// <summary>
        ///     <c>str()</c> of a NumPy float scalar of the given precision: Dragon4's shortest round-trip digits of that
        ///     precision, positional inside NumPy's window and scientific outside it (<c>1e+20</c>), <c>nan</c> / <c>inf</c>
        ///     / <c>-inf</c>, <c>-0.0</c>.
        /// </summary>
        /// <param name="value">The value, exactly representable in <paramref name="kind"/> (a float16 / float32 value widened
        ///     to double).</param>
        /// <param name="kind">The precision whose digits and window apply.</param>
        /// <returns>The text.</returns>
        internal static string NumPyStrFloat(double value, FloatKind kind)
        {
            Span<char> buf = stackalloc char[NDTextFormat.MaxTextLength];
            int n = NDTextFormat.FormatFloat(value, kind, buf);
            return new string(buf.Slice(0, n));
        }

        /// <summary><c>str(np.complex128)</c>: <c>(1+2j)</c>, a pure imaginary value as <c>2j</c>, each part like a float64.</summary>
        /// <param name="value">The value.</param>
        /// <returns>The text.</returns>
        internal static string NumPyStrComplex(Complex value)
        {
            Span<char> buf = stackalloc char[NDTextFormat.MaxTextLength];
            int n = NDTextFormat.FormatComplex(value, buf);
            return new string(buf.Slice(0, n));
        }

        /// <summary>
        ///     The <see cref="string"/> target of a conversion: NumPy's <c>str()</c> of a dtype value, else the culture-invariant
        ///     .NET text of anything else (a <see cref="DateTime"/>, a <see cref="TimeSpan"/>, another object), else null for a
        ///     null value.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns>The text (null for a null value).</returns>
        internal static string ToStringTarget(object value)
        {
            if (value == null)
                return null;
            return NumPyStr(value)
                   ?? (value is IFormattable f ? f.ToString(null, CultureInfo.InvariantCulture) : value.ToString());
        }

        /// <summary>
        ///     Converts a Python <c>str</c> into a value of the numeric dtype <paramref name="target"/> as NumPy's store
        ///     <c>a[0] = text</c> converts it: parsed into the canonical value of the target's family (int64, uint64, float64
        ///     or complex128; an integer range-checked for the target), then cast into the target (one more rounding for
        ///     float32 / float16, the float64 → decimal conversion for decimal, exact for an integer).
        /// </summary>
        /// <typeparam name="TOut">The C# type of <paramref name="target"/>.</typeparam>
        /// <param name="text">The text (not null).</param>
        /// <param name="target">The target dtype: an integer, char, float, decimal or complex type code (not bool).</param>
        /// <returns>The converted value.</returns>
        /// <exception cref="ValueError">The text is not a literal of the target's kind (CPython's texts, quoting
        ///     <c>'…'</c>), or an integer literal has more than 4300 digits.</exception>
        /// <exception cref="OverflowException">An integer literal outside the target (NumPy's <c>Python integer 300 out of
        ///     bounds for uint8</c>, or <c>Python int too large to convert to C long</c>).</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static unsafe TOut ParseNumPyText<TOut>(string text, NPTypeCode target) where TOut : unmanaged
        {
            // The canonical value is at most 16 bytes (a Complex); the parse writes it, the cast reads it into the target.
            byte* canon = stackalloc byte[16];
            NDTextParse.ParseObject(text, target, canon);
            TOut result = default;
            NDIterCasting.ConvertValue(canon, &result, NDTextParse.CanonicalType(NDTextParse.FamilyOf(target)), target);
            return result;
        }

        /// <summary>
        ///     The truth value of a Python <c>str</c>, NumPy's <c>str → bool</c> conversion: True for any non-empty text.
        /// </summary>
        /// <param name="text">The text, or null (False, C#'s "no value").</param>
        /// <returns>True when the text has at least one character.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool NumPyTextTruth(string text) => !string.IsNullOrEmpty(text);

        /// <summary>
        ///     Converts a Python <c>str</c> into any of NumSharp's dtypes by NumPy's store rules (see
        ///     <see cref="ParseNumPyText{TOut}"/> and <see cref="NumPyTextTruth"/>); a <see cref="NPTypeCode.String"/> target
        ///     returns the text.
        /// </summary>
        /// <param name="text">The text (not null).</param>
        /// <param name="target">The target dtype.</param>
        /// <returns>The boxed value of <paramref name="target"/>'s C# type.</returns>
        /// <exception cref="ValueError">The text is not a literal of the target's kind.</exception>
        /// <exception cref="OverflowException">An integer literal outside the target.</exception>
        /// <exception cref="InvalidCastException"><paramref name="target"/> is <see cref="NPTypeCode.Empty"/>.</exception>
        internal static object ParseNumPyTextBoxed(string text, NPTypeCode target) => target switch
        {
            NPTypeCode.Boolean => NumPyTextTruth(text),
            NPTypeCode.Byte => ParseNumPyText<byte>(text, target),
            NPTypeCode.SByte => ParseNumPyText<sbyte>(text, target),
            NPTypeCode.Int16 => ParseNumPyText<short>(text, target),
            NPTypeCode.UInt16 => ParseNumPyText<ushort>(text, target),
            NPTypeCode.Int32 => ParseNumPyText<int>(text, target),
            NPTypeCode.UInt32 => ParseNumPyText<uint>(text, target),
            NPTypeCode.Int64 => ParseNumPyText<long>(text, target),
            NPTypeCode.UInt64 => ParseNumPyText<ulong>(text, target),
            NPTypeCode.Char => ParseNumPyText<char>(text, target),
            NPTypeCode.Half => ParseNumPyText<Half>(text, target),
            NPTypeCode.Single => ParseNumPyText<float>(text, target),
            NPTypeCode.Double => ParseNumPyText<double>(text, target),
            NPTypeCode.Decimal => ParseNumPyText<decimal>(text, target),
            NPTypeCode.Complex => ParseNumPyText<Complex>(text, target),
            NPTypeCode.String => text,
            _ => throw new InvalidCastException("InvalidCast_Empty"),
        };
    }
}
