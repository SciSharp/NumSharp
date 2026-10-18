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
        ///     CPython's <c>str()</c> of a Python int of any size (a C# <see cref="BigInteger"/>, the boundary map's Python
        ///     int): its decimal digits, a leading <c>-</c> when negative — refused past <c>sys.get_int_max_str_digits()</c>
        ///     digits, as CPython's <c>long_to_decimal_string</c> refuses it. Every store and every <c>str()</c> /
        ///     <c>repr()</c> port that turns a Python int into text goes through here.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns>The text.</returns>
        /// <exception cref="ValueError">The value has more than 4300 decimal digits: CPython's <c>Exceeds the limit (4300
        ///     digits) for integer string conversion; use sys.set_int_max_str_digits() to increase the limit</c>. NumPy raises
        ///     it wherever it takes such an int's <c>str()</c> or <c>repr()</c> — a store into a <c>U</c> / <c>S</c> array or
        ///     a coercing <c>StringDType</c> array, a cast that writes a missing-data object as text, the repr of a
        ///     <c>StringDType</c> whose missing-data object it is.</exception>
        /// <remarks>
        ///     The limit counts the digits without the sign: <c>-(10**4300 - 1)</c> prints, <c>10**4300</c> does not. A value
        ///     whose bit length alone proves more than 4300 digits is refused before any digit is computed (CPython makes the
        ///     same early test), so a huge value costs nothing; the decimal conversion of .NET grows faster than linearly.
        /// </remarks>
        internal static string FormatPythonInt(BigInteger value)
        {
            // 2^(L-1) <= |v| < 2^L for a bit length L, and 2^14285 > 10^4300 (14285 * log10(2) = 4300.2): a bit length of
            // 14286 or more cannot have 4300 digits or fewer. Below that the digits are counted.
            if (BigInteger.Abs(value).GetBitLength() >= 14286)
                throw MaxStrDigitsExceeded();
            string text = value.ToString(CI);
            if (text.Length - (value.Sign < 0 ? 1 : 0) > NDTextParse.MaxStrDigits)
                throw MaxStrDigitsExceeded();
            return text;
        }

        /// <summary>CPython's refusal of an int → text conversion past <c>sys.get_int_max_str_digits()</c> digits.</summary>
        /// <returns>The exception (<c>_MAX_STR_DIGITS_ERROR_FMT_TO_STR</c> in <c>longobject.c</c>; its parsing twin, which
        ///     also names the digit count, is <see cref="NDTextParse"/>'s).</returns>
        private static ValueError MaxStrDigitsExceeded() => new ValueError(
            $"Exceeds the limit ({NDTextParse.MaxStrDigits} digits) for integer string conversion; use sys.set_int_max_str_digits() to increase the limit");

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
            // The ASCII core writes the text; widening it is cheaper than laying the digits out twice.
            byte* ascii = stackalloc byte[MaxTextLength];
            int n = FormatFloatAscii(value, kind, ascii);
            for (int i = 0; i < n; i++)
                buf[i] = (char)ascii[i];
            return n;
        }

        /// <summary>
        ///     <see cref="FormatFloat"/>'s text as ASCII bytes: the core both the char form and <c>np.strings.mod</c>'s
        ///     <c>%s</c> / <c>%r</c> fast path write through. The shortest digits come from Ryu as one integer and are laid
        ///     out eight at a time (<see cref="FloatDigits.WriteDecimalBackwards(ulong, byte*)"/>), not one division per
        ///     digit.
        /// </summary>
        /// <param name="value">The value, exactly representable in <paramref name="kind"/>.</param>
        /// <param name="kind">The precision whose shortest digits and positional window apply.</param>
        /// <param name="dst">Receives the text; room for <see cref="MaxTextLength"/> bytes.</param>
        /// <returns>The text's length (at most 24).</returns>
        /// <remarks>
        ///     Ryu's digits are CPython's and NumPy's (the shortest digits that read back, the closest one on a tie). They
        ///     used to come from .NET's <c>"R"</c> format, which gives the same digits for every float32 and float16 (both
        ///     checked exhaustively) and for every float64 but two on the .NET 10 runtime: 10.0.1 writes 16 digits for 2^-25
        ///     and 2^-958 that read back as the neighbouring double, where NumPy (and .NET 8) prints 17
        ///     (<c>str(np.float64(2**-25))</c> is <c>2.9802322387695312e-08</c>; measured over every power of two, every
        ///     float16 and 2,000,000 random doubles). Ryu is also several times faster, which the float → text casts and
        ///     <c>np.strings.mod</c>'s <c>%s</c> / <c>%r</c> pay per element.
        /// </remarks>
        [SkipLocalsInit]
        internal static int FormatFloatAscii(double value, FloatKind kind, byte* dst)
        {
            int n = 0;
            if (double.IsNaN(value))
                return PutAscii(dst, 0, "nan"u8);
            if (double.IsInfinity(value))
                return PutAscii(dst, 0, value > 0 ? "inf"u8 : "-inf"u8);

            if (double.IsNegative(value))
                dst[n++] = (byte)'-';
            ulong mantissa;
            int exponent;
            switch (kind)
            {
                case FloatKind.Half:
                    mantissa = FloatDigits.Shortest((Half)value, out exponent);
                    break;
                case FloatKind.Single:
                    mantissa = FloatDigits.Shortest((float)value, out exponent);
                    break;
                default:
                    mantissa = FloatDigits.Shortest(value, out exponent);
                    break;
            }
            if (mantissa == 0)
                return PutAscii(dst, n, "0.0"u8);

            // The digits (no leading or trailing zero by Shortest's contract) and the exponent of the first one. They end at
            // offset 24 of a 48-byte buffer: at most 17 digits, and every 16-byte copy below reads inside the buffer.
            byte* digitBuf = stackalloc byte[48];
            int nd = FloatDigits.WriteDecimalBackwards(mantissa, digitBuf + 24);
            byte* digits = digitBuf + 24 - nd;
            int decExp = exponent + nd - 1;

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
                    // The positional window keeps decExp below 16, so the integer part is at most 16 digits.
                    int intLen = decExp + 1;
                    if (nd > intLen)
                    {
                        // d…d . d…d: two runs of at most 16 digits.
                        CopySmall(digits, dst + n, intLen);
                        n += intLen;
                        dst[n++] = (byte)'.';
                        CopySmall(digits + intLen, dst + n, nd - intLen);
                        n += nd - intLen;
                    }
                    else
                    {
                        // An integral value: its digits, zeros to the integer part's length, then ".0".
                        CopySmall(digits, dst + n, nd);
                        n += nd;
                        for (int i = nd; i < intLen; i++)
                            dst[n++] = (byte)'0';
                        dst[n++] = (byte)'.';
                        dst[n++] = (byte)'0';
                    }
                }
                else
                {
                    // 0.000d…d: at most three zeros after the point (the window starts at 1e-4).
                    dst[n++] = (byte)'0';
                    dst[n++] = (byte)'.';
                    for (int i = 0; i < -decExp - 1; i++)
                        dst[n++] = (byte)'0';
                    CopySmall(digits, dst + n, nd);
                    n += nd;
                }
                return n;
            }

            // Scientific: d[.ddd]e±XX, two exponent digits at least.
            dst[n++] = digits[0];
            if (nd > 1)
            {
                dst[n++] = (byte)'.';
                CopySmall(digits + 1, dst + n, nd - 1);
                n += nd - 1;
            }
            dst[n++] = (byte)'e';
            dst[n++] = decExp >= 0 ? (byte)'+' : (byte)'-';
            int e = Math.Abs(decExp);
            if (e >= 100)
            {
                dst[n++] = (byte)('0' + e / 100);
                e %= 100;
            }
            dst[n++] = (byte)('0' + e / 10);
            dst[n++] = (byte)('0' + e % 10);
            return n;
        }

        /// <summary>
        ///     Copies at most 32 bytes as one or two (overlapping) 16-byte moves: the copy may write up to 16 bytes past
        ///     <paramref name="length"/> at <paramref name="dst"/> and read up to 16 past it at <paramref name="src"/> — the
        ///     callers' buffers have that room, and the bytes past the text are never read back.
        /// </summary>
        /// <param name="src">The bytes.</param>
        /// <param name="dst">The destination.</param>
        /// <param name="length">The number of bytes that matter (at most 32).</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void CopySmall(byte* src, byte* dst, int length)
        {
            System.Runtime.Intrinsics.Vector128.Store(System.Runtime.Intrinsics.Vector128.Load(src), dst);
            if (length > 16)
                System.Runtime.Intrinsics.Vector128.Store(System.Runtime.Intrinsics.Vector128.Load(src + length - 16), dst + length - 16);
        }

        /// <summary>Writes an ASCII literal at <paramref name="at"/> and returns the new length.</summary>
        /// <param name="dst">The output.</param>
        /// <param name="at">The position to write at.</param>
        /// <param name="text">The literal.</param>
        /// <returns><paramref name="at"/> + the literal's length.</returns>
        private static int PutAscii(byte* dst, int at, ReadOnlySpan<byte> text)
        {
            text.CopyTo(new Span<byte>(dst + at, text.Length));
            return at + text.Length;
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
