using System;
using System.Globalization;
using System.Numerics;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Printing;

namespace NumSharp.Backends.Strings
{
    /// <summary>
    ///     The value family a number crosses the text boundary in: every numeric dtype is read into (number → text) or
    ///     written from (text → number) ONE canonical C# type per family, so the formatter and the parser have one routine
    ///     per family instead of one per dtype. A block of elements is converted to / from its canonical type by the house
    ///     cast kernels (<see cref="NumSharp.Backends.Kernels.DirectILKernelGenerator.TryGetInnerCastKernel"/>).
    /// </summary>
    internal enum TextNumberKind : byte
    {
        /// <summary><c>bool</c>; canonical <see cref="bool"/> (read as "byte != 0", NumPy's <c>BOOL_getitem</c>).</summary>
        Bool,

        /// <summary>The signed integers int8..int64; canonical <see cref="long"/>.</summary>
        Signed,

        /// <summary>The unsigned integers uint8..uint64 and NumSharp's <c>Char</c> (a 16-bit code unit read as uint16);
        ///     canonical <see cref="ulong"/>.</summary>
        Unsigned,

        /// <summary>float16; canonical <see cref="Half"/> (its own shortest digits: <c>str(np.float16(0.1))</c> is
        ///     <c>0.1</c>).</summary>
        Half,

        /// <summary>float32; canonical <see cref="float"/>.</summary>
        Single,

        /// <summary>float64 and NumSharp's <c>Decimal</c> (formatted and parsed as float64, the way NumSharp prints a
        ///     decimal scalar); canonical <see cref="double"/>.</summary>
        Double,

        /// <summary>complex128; canonical <see cref="System.Numerics.Complex"/>.</summary>
        Complex,
    }

    /// <summary>
    ///     Number → text: NumPy's legacy cast loop <c>@FROMTYPE@_to_UNICODE</c> / <c>_to_STRING</c>
    ///     (<c>arraytypes.c.src</c>), which builds the NumPy scalar of each element (<c>PyArray_Scalar</c>), takes its
    ///     <c>str()</c> and stores the text truncated to the element width (<c>UNICODE_setitem</c> / <c>STRING_setitem</c>).
    ///     This class produces exactly that <c>str()</c> text without allocating: <c>True</c> / <c>False</c>, decimal
    ///     integers, the shortest repr of a float in ITS OWN precision with NumPy's per-dtype positional window, and
    ///     NumPy's complex form <c>(1+2j)</c>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Byte-identical to the printing path.</b> <see cref="ArrayFormatter.ScalarStr"/> (the <c>str()</c> of a 0-d
    ///     array, gated by the printing oracle) builds the same text through strings; the routines here lay the same digits
    ///     out into a span. The unit tests compare the two over every dtype's specials and a random sweep.
    ///     </para>
    ///     <para>
    ///     <b>The float rule</b> (<c>scalartypes.c.src</c>, <c>@name@type_@kind@_either</c>): shortest round-trip digits of
    ///     the value in its own precision; positional when <c>1e-4 &lt;= |v| &lt; max</c> with <c>max</c> 1e3 for float16,
    ///     1e6 for float32 and 1e16 for float64 — tested on the VALUE (float32 1e-4 is 9.99999974738e-05 and prints
    ///     <c>1e-04</c>) — else scientific with at least two exponent digits; <c>nan</c>, <c>inf</c>, <c>-inf</c>,
    ///     <c>-0.0</c>.
    ///     </para>
    /// </remarks>
    internal static unsafe class NDTextFormat
    {
        /// <summary>
        ///     Room for the longest <c>str()</c> of any numeric value: a complex with two 24-character float64 parts in
        ///     parentheses (<c>(-1.7976931348623157e+308-1.7976931348623157e+308j)</c>, 51 characters). A float64 alone is
        ///     at most 24 characters, an int64 20, a uint64 20.
        /// </summary>
        internal const int MaxTextLength = 64;

        private static readonly CultureInfo CI = CultureInfo.InvariantCulture;

        /// <summary>The canonical family of a numeric dtype (see <see cref="TextNumberKind"/>).</summary>
        /// <param name="tc">The dtype's type code.</param>
        /// <returns>The family its values are formatted from and parsed into.</returns>
        /// <exception cref="NotSupportedException"><paramref name="tc"/> is not a numeric dtype (string, empty).</exception>
        internal static TextNumberKind KindOf(NPTypeCode tc) => tc switch
        {
            NPTypeCode.Boolean => TextNumberKind.Bool,
            NPTypeCode.SByte or NPTypeCode.Int16 or NPTypeCode.Int32 or NPTypeCode.Int64 => TextNumberKind.Signed,
            NPTypeCode.Byte or NPTypeCode.UInt16 or NPTypeCode.UInt32 or NPTypeCode.UInt64 or NPTypeCode.Char => TextNumberKind.Unsigned,
            NPTypeCode.Half => TextNumberKind.Half,
            NPTypeCode.Single => TextNumberKind.Single,
            NPTypeCode.Double or NPTypeCode.Decimal => TextNumberKind.Double,
            NPTypeCode.Complex => TextNumberKind.Complex,
            _ => throw new NotSupportedException($"{tc} has no number <-> text conversion."),
        };

        /// <summary>The type code of a family's canonical C# type (the scratch block a cast kernel fills or drains).</summary>
        /// <param name="kind">The family.</param>
        /// <returns><see cref="NPTypeCode.Boolean"/>, <see cref="NPTypeCode.Int64"/>, <see cref="NPTypeCode.UInt64"/>,
        ///     <see cref="NPTypeCode.Half"/>, <see cref="NPTypeCode.Single"/>, <see cref="NPTypeCode.Double"/> or
        ///     <see cref="NPTypeCode.Complex"/>.</returns>
        internal static NPTypeCode CanonicalType(TextNumberKind kind) => kind switch
        {
            TextNumberKind.Bool => NPTypeCode.Boolean,
            TextNumberKind.Signed => NPTypeCode.Int64,
            TextNumberKind.Unsigned => NPTypeCode.UInt64,
            TextNumberKind.Half => NPTypeCode.Half,
            TextNumberKind.Single => NPTypeCode.Single,
            TextNumberKind.Double => NPTypeCode.Double,
            _ => NPTypeCode.Complex,
        };

        /// <summary>
        ///     NumPy's <c>REQUIRED_STR_LEN</c> width of a numeric dtype — the width an UNSIZED <c>U</c> / <c>S</c> cast
        ///     target takes (<c>np.zeros(3, 'i8').astype('U')</c> is <c>&lt;U21</c>), and the width a number contributes when
        ///     it promotes with text (<c>np.array(['x', 1])</c> is <c>&lt;U21</c>).
        /// </summary>
        /// <param name="tc">The numeric dtype.</param>
        /// <returns>bool 5; int8 4, uint8 3, int16 6, uint16 / char 5, int32 11, uint32 10, int64 21, uint64 20; every
        ///     float (and decimal) 32; complex 64.</returns>
        /// <remarks>Delegates to <see cref="ToStringCastingImpl.RequiredLength"/>, the table the descriptor resolution
        ///     uses, so the cast width and the promotion width cannot drift apart.</remarks>
        internal static int RequiredWidth(NPTypeCode tc) => (int)ToStringCastingImpl.RequiredLength(DType.From(tc));

        // ---------------------------------------------------------------------------------------------
        // Formatting one canonical value
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     The <c>str()</c> of one canonical value of a family, written into <paramref name="buf"/>.
        /// </summary>
        /// <param name="kind">The value's family.</param>
        /// <param name="value">The canonical value (a <see cref="bool"/> read as its byte, a <see cref="long"/>, ...).</param>
        /// <param name="buf">Receives the text; room for <see cref="MaxTextLength"/> characters.</param>
        /// <returns>The text's length.</returns>
        internal static int Format(TextNumberKind kind, byte* value, Span<char> buf)
        {
            switch (kind)
            {
                case TextNumberKind.Bool:
                    // NumPy's BOOL_getitem tests the byte against zero: a bool array holding 2 is True.
                    return FormatBool(*value != 0, buf);
                case TextNumberKind.Signed:
                    return FormatInt64(*(long*)value, buf);
                case TextNumberKind.Unsigned:
                    return FormatUInt64(*(ulong*)value, buf);
                case TextNumberKind.Half:
                    return FormatFloat((double)*(Half*)value, FloatKind.Half, buf);
                case TextNumberKind.Single:
                    return FormatFloat(*(float*)value, FloatKind.Single, buf);
                case TextNumberKind.Double:
                    return FormatFloat(*(double*)value, FloatKind.Double, buf);
                default:
                    return FormatComplex(*(Complex*)value, buf);
            }
        }

        /// <summary><c>str(np.bool_)</c>: <c>True</c> or <c>False</c>.</summary>
        /// <param name="v">The value.</param>
        /// <param name="buf">Receives the text.</param>
        /// <returns>4 or 5.</returns>
        internal static int FormatBool(bool v, Span<char> buf)
        {
            if (v)
            {
                "True".AsSpan().CopyTo(buf);
                return 4;
            }
            "False".AsSpan().CopyTo(buf);
            return 5;
        }

        /// <summary><c>str()</c> of a signed integer: its decimal digits, a leading <c>-</c> when negative.</summary>
        /// <param name="v">The value.</param>
        /// <param name="buf">Receives the text.</param>
        /// <returns>The text's length (1 to 20).</returns>
        internal static int FormatInt64(long v, Span<char> buf)
        {
            v.TryFormat(buf, out int n, default, CI);
            return n;
        }

        /// <summary><c>str()</c> of an unsigned integer: its decimal digits.</summary>
        /// <param name="v">The value.</param>
        /// <param name="buf">Receives the text.</param>
        /// <returns>The text's length (1 to 20).</returns>
        internal static int FormatUInt64(ulong v, Span<char> buf)
        {
            v.TryFormat(buf, out int n, default, CI);
            return n;
        }

        /// <summary>
        ///     <c>str()</c> of a NumPy float scalar of the given precision (see the class remarks for the rule).
        /// </summary>
        /// <param name="value">The value, exactly representable in <paramref name="kind"/> (a float16 / float32 value
        ///     widened to double).</param>
        /// <param name="kind">The precision whose shortest digits and positional window apply.</param>
        /// <param name="buf">Receives the text.</param>
        /// <returns>The text's length.</returns>
        internal static int FormatFloat(double value, FloatKind kind, Span<char> buf)
        {
            if (double.IsNaN(value))
                return Put(buf, 0, "nan");
            if (double.IsInfinity(value))
                return Put(buf, 0, value > 0 ? "inf" : "-inf");

            Span<byte> digits = stackalloc byte[32];
            int nd = ShortestDigits(value, kind, digits, out int decExp, out bool negative);
            int n = 0;
            if (negative)
                buf[n++] = '-';
            if (nd == 1 && digits[0] == 0)
                return Put(buf, n, "0.0");

            double maxPositional = kind switch
            {
                FloatKind.Half => 1e3,
                FloatKind.Single => 1e6,
                _ => 1e16,
            };
            double abs = Math.Abs(value);
            if (abs >= 1e-4 && abs < maxPositional)
            {
                if (decExp >= 0)
                {
                    // Integer part: the first decExp + 1 digits, zero-filled when the shortest form is shorter.
                    int intLen = decExp + 1;
                    for (int i = 0; i < intLen; i++)
                        buf[n++] = (char)('0' + (i < nd ? digits[i] : 0));
                    buf[n++] = '.';
                    if (nd > intLen)
                        for (int i = intLen; i < nd; i++)
                            buf[n++] = (char)('0' + digits[i]);
                    else
                        buf[n++] = '0';
                }
                else
                {
                    buf[n++] = '0';
                    buf[n++] = '.';
                    for (int i = 0; i < -decExp - 1; i++)
                        buf[n++] = '0';
                    for (int i = 0; i < nd; i++)
                        buf[n++] = (char)('0' + digits[i]);
                }
                return n;
            }

            // Scientific: d[.ddd]e±XX, two exponent digits at least.
            buf[n++] = (char)('0' + digits[0]);
            if (nd > 1)
            {
                buf[n++] = '.';
                for (int i = 1; i < nd; i++)
                    buf[n++] = (char)('0' + digits[i]);
            }
            buf[n++] = 'e';
            buf[n++] = decExp >= 0 ? '+' : '-';
            int e = Math.Abs(decExp);
            if (e < 10)
                buf[n++] = '0';
            e.TryFormat(buf.Slice(n), out int en, default, CI);
            return n + en;
        }

        /// <summary>
        ///     <c>str()</c> of a NumPy complex128 scalar (<c>scalartypes.c.src</c>, <c>c@name@type_@kind@</c>, trim mode
        ///     <c>DptZeros</c>): <c>{imag}j</c> when the real part is +0.0 (not -0.0), else <c>({real}{±imag}j)</c>; each
        ///     finite part in the float64 rule with a trailing <c>.0</c> dropped (<c>1</c>, <c>-0</c>, <c>1.5</c>,
        ///     <c>1e+20</c>); a NaN imaginary part prints <c>+nan</c> whatever its sign, an infinite one <c>+inf</c> /
        ///     <c>-inf</c>.
        /// </summary>
        /// <param name="v">The value.</param>
        /// <param name="buf">Receives the text.</param>
        /// <returns>The text's length.</returns>
        internal static int FormatComplex(Complex v, Span<char> buf)
        {
            double re = v.Real, im = v.Imaginary;
            int n;
            if (re == 0.0 && !double.IsNegative(re))
            {
                n = FormatComplexPart(im, buf);
                buf[n++] = 'j';
                return n;
            }

            n = 0;
            buf[n++] = '(';
            n += FormatComplexPart(re, buf.Slice(n));
            int imStart = n;
            int imLen = FormatComplexPart(im, buf.Slice(n));
            // The imaginary part always carries a sign: a part that does not start with one gets '+'.
            if (buf[imStart] != '-' && buf[imStart] != '+')
            {
                buf.Slice(imStart, imLen).CopyTo(buf.Slice(imStart + 1));
                buf[imStart] = '+';
                imLen++;
            }
            n += imLen;
            buf[n++] = 'j';
            buf[n++] = ')';
            return n;
        }

        /// <summary>One part of a complex's text: the float64 <c>str()</c> with a trailing <c>.0</c> removed, or
        ///     <c>nan</c> / <c>inf</c> / <c>-inf</c>.</summary>
        /// <param name="d">The part.</param>
        /// <param name="buf">Receives the text.</param>
        /// <returns>The text's length.</returns>
        private static int FormatComplexPart(double d, Span<char> buf)
        {
            int n = FormatFloat(d, FloatKind.Double, buf);
            if (n >= 2 && buf[n - 2] == '.' && buf[n - 1] == '0')
                n -= 2;
            return n;
        }

        /// <summary>
        ///     The shortest round-trip decimal digits of a value in its own precision (.NET's <c>"R"</c> format is the
        ///     shortest form for <see cref="double"/>, <see cref="float"/> and <see cref="System.Half"/>), as digit VALUES
        ///     with leading and trailing zeros removed, plus the decimal exponent of the first digit.
        /// </summary>
        /// <param name="value">The value (finite).</param>
        /// <param name="kind">Its precision.</param>
        /// <param name="digits">Receives the digit values 0..9 (zero is the single digit 0).</param>
        /// <param name="decExp">Receives the exponent of the first digit (value = 0.d1d2... × 10^(decExp+1)).</param>
        /// <param name="negative">Receives the sign (true for -0.0 too).</param>
        /// <returns>The number of digits.</returns>
        /// <remarks>The same extraction <see cref="Dragon4.ExtractShortest"/> runs on strings, kept span-based here so a
        ///     cast of a million floats allocates nothing.</remarks>
        private static int ShortestDigits(double value, FloatKind kind, Span<byte> digits, out int decExp, out bool negative)
        {
            Span<char> r = stackalloc char[40];
            int len;
            switch (kind)
            {
                case FloatKind.Half:
                    ((Half)value).TryFormat(r, out len, "R", CI);
                    break;
                case FloatKind.Single:
                    ((float)value).TryFormat(r, out len, "R", CI);
                    break;
                default:
                    value.TryFormat(r, out len, "R", CI);
                    break;
            }

            int p = 0;
            negative = r[0] == '-';
            if (negative)
                p++;
            int exp = 0;
            int e = r.Slice(0, len).IndexOfAny('E', 'e');
            int mantEnd = e < 0 ? len : e;
            if (e >= 0)
            {
                // "R" writes E+dd / E-dd.
                int sign = 1, q = e + 1;
                if (r[q] == '+') q++;
                else if (r[q] == '-') { sign = -1; q++; }
                for (; q < len; q++)
                    exp = exp * 10 + (r[q] - '0');
                exp *= sign;
            }

            // Collect the mantissa digits (point removed), noting how many came before the point.
            int raw = 0, intDigits = -1;
            for (int i = p; i < mantEnd; i++)
            {
                char c = r[i];
                if (c == '.')
                {
                    intDigits = raw;
                    continue;
                }
                digits[raw++] = (byte)(c - '0');
            }
            if (intDigits < 0)
                intDigits = raw;

            int e0 = intDigits - 1 + exp;
            int lead = 0;
            while (lead < raw - 1 && digits[lead] == 0)
            {
                lead++;
                e0--;
            }
            int end = raw;
            while (end > lead + 1 && digits[end - 1] == 0)
                end--;
            int nd = end - lead;
            if (lead > 0)
                digits.Slice(lead, nd).CopyTo(digits);
            if (nd == 1 && digits[0] == 0)
            {
                decExp = 0;
                return 1;
            }
            decExp = e0;
            return nd;
        }

        /// <summary>Writes a literal at <paramref name="at"/> and returns the new length.</summary>
        /// <param name="buf">The output.</param>
        /// <param name="at">The position to write at.</param>
        /// <param name="s">The literal.</param>
        /// <returns><paramref name="at"/> + the literal's length.</returns>
        private static int Put(Span<char> buf, int at, string s)
        {
            s.AsSpan().CopyTo(buf.Slice(at));
            return at + s.Length;
        }

        // ---------------------------------------------------------------------------------------------
        // Storing the text into an element
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     Stores ASCII text into one <c>S</c> / <c>U</c> element: the first <paramref name="chars"/> characters (a
        ///     longer text is TRUNCATED, as <c>UNICODE_setitem</c> / <c>STRING_setitem</c> truncate), the rest NUL-padded.
        /// </summary>
        /// <param name="text">The text (every number's <c>str()</c> is ASCII).</param>
        /// <param name="dst">The element's first byte.</param>
        /// <param name="unicode">True for <c>U</c> (one UCS-4 unit per character), false for <c>S</c> (one byte).</param>
        /// <param name="chars">The element's width in characters.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void Store(ReadOnlySpan<char> text, byte* dst, bool unicode, int chars)
        {
            int n = Math.Min(text.Length, chars);
            if (unicode)
            {
                uint* u = (uint*)dst;
                for (int i = 0; i < n; i++)
                    u[i] = text[i];
                for (int i = n; i < chars; i++)
                    u[i] = 0;
            }
            else
            {
                for (int i = 0; i < n; i++)
                    dst[i] = (byte)text[i];
                for (int i = n; i < chars; i++)
                    dst[i] = 0;
            }
        }

        /// <summary>
        ///     The length of a value's <c>str()</c> — how wide a <c>U</c> / <c>S</c> descriptor DISCOVERED from the value
        ///     is (<c>np.array([1, 22], dtype='U')</c> is <c>&lt;U2</c>, <c>np.array(1.5, dtype='U')</c> is <c>&lt;U3</c>).
        /// </summary>
        /// <param name="kind">The value's family.</param>
        /// <param name="value">The canonical value.</param>
        /// <returns>The number of characters <see cref="Format"/> writes.</returns>
        internal static int StrLength(TextNumberKind kind, byte* value)
        {
            Span<char> buf = stackalloc char[MaxTextLength];
            return Format(kind, value, buf);
        }
    }
}
