using System;
using System.Globalization;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text;
using NumSharp.Backends.Printing;
using NumSharp.Utilities;

namespace NumSharp.Backends.Strings
{
    /// <summary>The outcome of parsing one text value into a number (see <see cref="NDTextParse"/>).</summary>
    internal enum TextParseStatus : byte
    {
        /// <summary>Parsed; the canonical value is written.</summary>
        Ok,

        /// <summary><c>int()</c> refused the text: <c>invalid literal for int() with base 10: …</c> (ValueError).</summary>
        InvalidInt,

        /// <summary>A valid integer literal with more than 4300 digits (CPython's <c>sys.get_int_max_str_digits()</c>).</summary>
        DigitLimit,

        /// <summary>The integer fits the C conversion type but not the target: <c>Python integer {v} out of bounds for
        ///     {dtype}</c> (OverflowError).</summary>
        OutOfBounds,

        /// <summary>The integer does not fit the C conversion type (LP64 <c>long</c>): <c>Python int too large to convert to
        ///     C long</c> (OverflowError).</summary>
        TooLargeForCLong,

        /// <summary><c>float()</c> refused the text: <c>could not convert string to float: …</c> (ValueError).</summary>
        InvalidFloat,

        /// <summary><c>complex()</c> refused the text: <c>complex() arg is a malformed string</c> (ValueError).</summary>
        InvalidComplex,

        /// <summary><c>complex()</c> refused the text's underscores: <c>could not convert string to complex: …</c>
        ///     (ValueError).</summary>
        ComplexUnderscore,

        /// <summary>A <c>U</c> element that is ONE unit above U+10FFFF: NumPy cannot even build its <c>np.str_</c>
        ///     (<c>SystemError: invalid maximum character passed to PyUnicode_New</c>).</summary>
        InvalidCodePoint,

        /// <summary>An <c>S</c> value that is not strict UTF-8, met where NumPy decodes it (<c>S → complex</c>):
        ///     <c>UnicodeDecodeError: 'utf-8' codec can't decode …</c>.</summary>
        DecodeError,
    }

    /// <summary>
    ///     How a parse error quotes the refused text — NumPy quotes the OBJECT it handed to <c>int()</c> / <c>float()</c> /
    ///     <c>complex()</c>, and which object that is depends on where the text came from.
    /// </summary>
    internal enum TextReprStyle : byte
    {
        /// <summary>
        ///     An element of an <c>S</c> / <c>U</c> array (a cast, <c>astype</c>, <c>copyto</c>): NumPy's cast loop builds
        ///     the NumPy scalar (<c>PyArray_Scalar</c>), so <c>int()</c> and <c>float()</c> quote <c>np.str_('…')</c> and
        ///     <c>float()</c> of bytes quotes <c>np.bytes_(b'…')</c>.
        /// </summary>
        ArrayElement,

        /// <summary>
        ///     A Python <c>str</c> / <c>bytes</c> object stored into a numeric element (<c>a[0] = '1.5'</c>,
        ///     <c>np.array(['x'], dtype=float)</c>, <c>a.fill('7')</c>): NumPy's <c>setitem</c> gets the object itself, so
        ///     the errors quote <c>'…'</c> / <c>b'…'</c>.
        /// </summary>
        PythonObject,
    }

    /// <summary>
    ///     Text → number: what NumPy's legacy cast loop <c>STRING_to_@TYPE@</c> / <c>UNICODE_to_@TYPE@</c>
    ///     (<c>arraytypes.c.src</c>) does per element — build the NumPy scalar (<c>np.str_</c> / <c>np.bytes_</c>, trailing
    ///     NULs removed) and hand it to the target's <c>setitem</c>, which runs CPython's <c>int()</c>, <c>float()</c> or
    ///     <c>complex()</c> on it and packs the result into the element. This class is that per-element parse, exact to
    ///     CPython 3.12 and NumPy 2.4.2: the grammar, the Unicode rewrite, the packing checks and every error text.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>The grammars.</b> Every parse first rewrites the text to ASCII (<see cref="PyUnicode.TransformDecimalAndSpaceToAscii(ReadOnlySpan{uint}, Span{byte})"/>:
    ///     Unicode whitespace becomes a space, Unicode decimal digits become ASCII digits). <c>S</c> bytes go to <c>int()</c>
    ///     and <c>float()</c> as they are (a byte above 0x7F is an invalid character there) but are DECODED as strict UTF-8
    ///     for <c>complex()</c> (<c>CDOUBLE_setitem</c> calls <c>PyUnicode_FromEncodedObject(op, NULL, NULL)</c>), so
    ///     <c>np.array([b'\xd9\xa1j']).astype(complex)</c> is <c>1j</c> (an Arabic-Indic one) and <c>b'\xff'</c> raises
    ///     <c>UnicodeDecodeError</c>.
    ///     </para>
    ///     <list type="bullet">
    ///     <item><c>int()</c> (<c>PyLong_FromString</c>, base 10): surrounding ASCII whitespace, one sign, digits with single
    ///     underscores BETWEEN digits, leading zeros allowed; no point, no exponent, no base prefix. More than 4300 digits is
    ///     its own ValueError, raised only for an otherwise valid literal.</item>
    ///     <item><c>float()</c> (<c>_Py_string_to_number_with_underscores</c> + <c>float_from_string_inner</c> +
    ///     <c>_PyOS_ascii_strtod</c>): underscores validated over the whole text first (each between two digits), then
    ///     surrounding whitespace, then the strtod grammar <c>[sign](digits[.digits]|.digits)[(e|E)[sign]digits]</c> or
    ///     <c>[sign](inf|infinity|nan)</c> in any case; an exponent with no digit is not consumed (<c>1e</c> fails); the
    ///     value is correctly rounded, overflow is <c>±inf</c> and underflow <c>±0</c>; <c>-nan</c> is a NaN with the sign
    ///     bit set.</item>
    ///     <item><c>complex()</c> (<c>complex_from_string_inner</c>): optional whitespace and parentheses around
    ///     <c>&lt;float&gt;</c>, <c>&lt;float&gt;j</c>, <c>&lt;float&gt;&lt;signed-float&gt;j</c>, <c>&lt;float&gt;&lt;sign&gt;j</c>,
    ///     <c>&lt;sign&gt;j</c> or <c>j</c> (either case), no whitespace inside (<c>1 + 2j</c> fails).</item>
    ///     </list>
    ///     <para>
    ///     <b>Packing (LP64).</b> An integer target converts the Python int with NumPy's <c>@TYPE@_safe_pyint_setitem</c>:
    ///     through C <c>long</c> for int8..int64, uint8 and uint16 (and NumSharp's char, a uint16), through C <c>unsigned
    ///     long</c> with a wrap flag for uint32 and uint64. Past the C type the text is CPython's <c>Python int too large to
    ///     convert to C long</c>; inside it but outside the target (or negative into uint32 / uint64) NumPy's <c>Python
    ///     integer -1 out of bounds for uint64</c>. These are the Linux / macOS texts (C <c>long</c> is 64-bit there), which
    ///     NumSharp pins; Windows NumPy reports int64 overflow as <c>int too big to convert</c>. A float target rounds the
    ///     parsed double once more into float32 (C cast) or float16 (<c>npy_double_to_half</c>, one rounding from the
    ///     double).
    ///     </para>
    /// </remarks>
    internal static unsafe class NDTextParse
    {
        /// <summary>CPython's default <c>sys.get_int_max_str_digits()</c>.</summary>
        internal const int MaxStrDigits = 4300;

        /// <summary>CPython's <c>Py_NAN</c> (positive quiet NaN), built from its bits at run time (the JIT folds NaN
        ///     constants into whatever bit pattern it likes).</summary>
        private static readonly double PositiveNaN = BitConverter.UInt64BitsToDouble(0x7FF8_0000_0000_0000UL);

        /// <summary><c>-fabs(Py_NAN)</c>: what <c>float('-nan')</c> returns, the sign bit set.</summary>
        private static readonly double NegativeNaN = BitConverter.UInt64BitsToDouble(0xFFF8_0000_0000_0000UL);

        /// <summary>The parse family of a numeric target, which decides the grammar and the canonical type.</summary>
        internal enum Family : byte
        {
            /// <summary>Signed integers; <c>int()</c>, canonical <see cref="long"/>.</summary>
            Signed,

            /// <summary>Unsigned integers and char; <c>int()</c>, canonical <see cref="ulong"/>.</summary>
            Unsigned,

            /// <summary>float16 / float32 / float64 / decimal; <c>float()</c>, canonical <see cref="double"/>.</summary>
            Float,

            /// <summary>complex128; <c>complex()</c>, canonical <see cref="Complex"/>.</summary>
            Complex,
        }

        /// <summary>The parse family of a numeric target dtype.</summary>
        /// <param name="target">The target dtype (not bool — a text → bool cast is a truth test, not a parse).</param>
        /// <returns>The family.</returns>
        /// <exception cref="NotSupportedException"><paramref name="target"/> is bool, string or empty.</exception>
        internal static Family FamilyOf(NPTypeCode target) => target switch
        {
            NPTypeCode.SByte or NPTypeCode.Int16 or NPTypeCode.Int32 or NPTypeCode.Int64 => Family.Signed,
            NPTypeCode.Byte or NPTypeCode.UInt16 or NPTypeCode.UInt32 or NPTypeCode.UInt64 or NPTypeCode.Char => Family.Unsigned,
            NPTypeCode.Half or NPTypeCode.Single or NPTypeCode.Double or NPTypeCode.Decimal => Family.Float,
            NPTypeCode.Complex => Family.Complex,
            _ => throw new NotSupportedException($"{target} is not a parse target."),
        };

        /// <summary>The type code of a family's canonical value (what the parse writes before the cast into the target).</summary>
        /// <param name="family">The family.</param>
        /// <returns><see cref="NPTypeCode.Int64"/>, <see cref="NPTypeCode.UInt64"/>, <see cref="NPTypeCode.Double"/> or
        ///     <see cref="NPTypeCode.Complex"/>.</returns>
        internal static NPTypeCode CanonicalType(Family family) => family switch
        {
            Family.Signed => NPTypeCode.Int64,
            Family.Unsigned => NPTypeCode.UInt64,
            Family.Float => NPTypeCode.Double,
            _ => NPTypeCode.Complex,
        };

        /// <summary>
        ///     The scratch bytes <see cref="ParseElement"/> needs for an element of <paramref name="chars"/> characters: the
        ///     ASCII rewrite, its underscore-free copy, and (for <c>S → complex</c>) the decoded code points.
        /// </summary>
        /// <param name="chars">The element width in characters.</param>
        /// <returns>The size in bytes.</returns>
        internal static int ScratchBytes(int chars) => checked(6 * chars + 16);

        /// <summary>The details of a failed parse that its error text needs (filled only on the failing path).</summary>
        private struct Detail
        {
            /// <summary>The digit count of a <see cref="TextParseStatus.DigitLimit"/> literal.</summary>
            public int Digits;

            /// <summary>The sign of an <see cref="TextParseStatus.OutOfBounds"/> value.</summary>
            public bool Negative;

            /// <summary>The magnitude of an <see cref="TextParseStatus.OutOfBounds"/> value.</summary>
            public ulong Magnitude;

            /// <summary>The first byte of a <see cref="TextParseStatus.DecodeError"/>'s range.</summary>
            public int DecodeStart;

            /// <summary>The end (exclusive) of a <see cref="TextParseStatus.DecodeError"/>'s range.</summary>
            public int DecodeEnd;

            /// <summary>The UTF-8 decoder's reason (<c>invalid start byte</c>, ...).</summary>
            public string DecodeReason;
        }

        // ---------------------------------------------------------------------------------------------
        // Elements
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     Parses ONE <c>S</c> / <c>U</c> element into the canonical value of <paramref name="target"/>'s family
        ///     (range-checked for an integer target, so the cast into the target that follows is exact).
        /// </summary>
        /// <param name="src">The element's first byte.</param>
        /// <param name="unicode">True for <c>U</c> (UCS-4 units), false for <c>S</c> (bytes).</param>
        /// <param name="chars">The element's width in characters.</param>
        /// <param name="target">The numeric target dtype (decides grammar, family and the packing check).</param>
        /// <param name="canon">Receives the canonical value (<see cref="CanonicalType"/> of the family) on success.</param>
        /// <param name="scratch">Work space of at least <see cref="ScratchBytes"/>(<paramref name="chars"/>) bytes.</param>
        /// <returns><see cref="TextParseStatus.Ok"/>, or the reason it failed (build the exception with
        ///     <see cref="ElementError"/>).</returns>
        internal static TextParseStatus ParseElement(byte* src, bool unicode, int chars, NPTypeCode target, byte* canon, Span<byte> scratch)
        {
            Detail d = default;
            return ParseElementCore(src, unicode, chars, target, canon, scratch, ref d);
        }

        /// <summary>
        ///     The exception NumPy raises for an element <see cref="ParseElement"/> refused: the element is parsed again with
        ///     diagnostics and the text quoted the way <paramref name="style"/> says.
        /// </summary>
        /// <param name="src">The element's first byte.</param>
        /// <param name="unicode">True for <c>U</c>, false for <c>S</c>.</param>
        /// <param name="chars">The element's width in characters.</param>
        /// <param name="target">The numeric target dtype.</param>
        /// <param name="style">How the text is quoted (an array element's NumPy scalar, or a stored Python object).</param>
        /// <returns>The exception (not thrown), or null when the element parses after all.</returns>
        internal static Exception ElementError(byte* src, bool unicode, int chars, NPTypeCode target, TextReprStyle style)
        {
            byte* canon = stackalloc byte[16];
            int need = ScratchBytes(chars);
            // An error path: a wide element's scratch comes from the heap rather than a huge stack frame.
            Span<byte> scratch = need <= 1024 ? stackalloc byte[1024] : new byte[need];
            Detail d = default;
            TextParseStatus status = ParseElementCore(src, unicode, chars, target, canon, scratch, ref d);
            if (status == TextParseStatus.Ok)
                return null;
            if (unicode)
            {
                var units = new ReadOnlySpan<uint>(src, TrimmedUnits((uint*)src, chars));
                string repr = PyRepr.StrCodePoints(units);
                return Build(status, d, target, style == TextReprStyle.ArrayElement ? "np.str_(" + repr + ")" : repr, null, style);
            }
            return Build(status, d, target, null, new ReadOnlySpan<byte>(src, TrimmedBytes(src, chars)).ToArray(), style);
        }

        /// <summary>
        ///     Parses a C# <see cref="string"/> (a Python <c>str</c>) into <paramref name="target"/>'s canonical value — the
        ///     scalar store <c>a[0] = '1.5'</c>, <c>np.array(['12'], dtype='i8')</c>, <c>a.fill('7')</c>. A valid surrogate pair
        ///     is one code point, as everywhere in NumSharp's text model.
        /// </summary>
        /// <param name="s">The text.</param>
        /// <param name="target">The numeric target dtype.</param>
        /// <param name="canon">Receives the canonical value (16 bytes).</param>
        /// <exception cref="ValueError">The text is not a number of the target's kind (NumPy's text, quoting
        ///     <c>'…'</c>).</exception>
        /// <exception cref="OverflowException">An integer outside the target (NumPy's OverflowError texts).</exception>
        internal static void ParseObject(string s, NPTypeCode target, byte* canon)
        {
            uint[] cps = ToCodePoints(s);
            fixed (uint* p = cps)
            {
                byte[] scratch = new byte[ScratchBytes(cps.Length)];
                Detail d = default;
                TextParseStatus status = ParseUnitsCore(new ReadOnlySpan<uint>(p, cps.Length), target, canon, scratch, ref d);
                if (status != TextParseStatus.Ok)
                    throw Build(status, d, target, PyRepr.Str(s), null, TextReprStyle.PythonObject);
            }
        }

        /// <summary>
        ///     Parses a C# <c>byte[]</c> (a Python <c>bytes</c>) into <paramref name="target"/>'s canonical value — the scalar
        ///     store <c>a[0] = b'12'</c>.
        /// </summary>
        /// <param name="b">The bytes (trailing NULs are part of a Python bytes value and are kept).</param>
        /// <param name="target">The numeric target dtype.</param>
        /// <param name="canon">Receives the canonical value (16 bytes).</param>
        /// <exception cref="ValueError">Not a number of the target's kind (quoting <c>b'…'</c>).</exception>
        /// <exception cref="OverflowException">An integer outside the target.</exception>
        /// <exception cref="UnicodeDecodeError">A complex target meets bytes that are not UTF-8.</exception>
        internal static void ParseObject(byte[] b, NPTypeCode target, byte* canon)
        {
            byte[] scratch = new byte[ScratchBytes(b.Length)];
            Detail d = default;
            TextParseStatus status = ParseBytesCore(b, target, canon, scratch, ref d);
            if (status != TextParseStatus.Ok)
                throw Build(status, d, target, null, b, TextReprStyle.PythonObject);
        }

        /// <summary>The element's unit count without its trailing NUL units (NumPy's <c>np.str_</c> drops them).</summary>
        /// <param name="u">The element's first unit.</param>
        /// <param name="chars">The element's width.</param>
        /// <returns>The trimmed length.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int TrimmedUnits(uint* u, int chars)
        {
            int n = chars;
            while (n > 0 && u[n - 1] == 0)
                n--;
            return n;
        }

        /// <summary>The element's byte count without its trailing NUL bytes (NumPy's <c>np.bytes_</c> drops them).</summary>
        /// <param name="b">The element's first byte.</param>
        /// <param name="chars">The element's width.</param>
        /// <returns>The trimmed length.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int TrimmedBytes(byte* b, int chars)
        {
            int n = chars;
            while (n > 0 && b[n - 1] == 0)
                n--;
            return n;
        }

        /// <summary>
        ///     The element parse with diagnostics: trims the element's trailing NULs (NumPy's scalar drops them) and
        ///     dispatches on the element kind.
        /// </summary>
        /// <param name="src">The element's first byte.</param>
        /// <param name="unicode">True for <c>U</c>, false for <c>S</c>.</param>
        /// <param name="chars">The element's width in characters.</param>
        /// <param name="target">The numeric target dtype.</param>
        /// <param name="canon">Receives the canonical value on success.</param>
        /// <param name="scratch">Work space of <see cref="ScratchBytes"/>(<paramref name="chars"/>) bytes.</param>
        /// <param name="d">Receives the failure details.</param>
        /// <returns>The outcome.</returns>
        private static TextParseStatus ParseElementCore(byte* src, bool unicode, int chars, NPTypeCode target, byte* canon,
            Span<byte> scratch, ref Detail d)
        {
            if (unicode)
            {
                uint* u = (uint*)src;
                return ParseUnitsCore(new ReadOnlySpan<uint>(u, TrimmedUnits(u, chars)), target, canon, scratch, ref d);
            }
            return ParseBytesCore(new ReadOnlySpan<byte>(src, TrimmedBytes(src, chars)), target, canon, scratch, ref d);
        }

        /// <summary>
        ///     Parses a Python <c>str</c> given as code points (a trimmed <c>U</c> element or a C# string): the
        ///     one-unit-above-U+10FFFF refusal, the Unicode rewrite, then the grammar.
        /// </summary>
        /// <param name="units">The text's code points, trailing NULs removed.</param>
        /// <param name="target">The numeric target dtype.</param>
        /// <param name="canon">Receives the canonical value on success.</param>
        /// <param name="scratch">Work space of <see cref="ScratchBytes"/>(<c>units.Length</c>) bytes.</param>
        /// <param name="d">Receives the failure details.</param>
        /// <returns>The outcome.</returns>
        private static TextParseStatus ParseUnitsCore(ReadOnlySpan<uint> units, NPTypeCode target, byte* canon,
            Span<byte> scratch, ref Detail d)
        {
            // PyArray_Scalar builds the np.str_ first: CPython refuses a ONE-character string above U+10FFFF (a longer one
            // only caps the maximum character it records), whatever the target.
            if (units.Length == 1 && units[0] > NDStringCodec.MaxCodePoint)
                return TextParseStatus.InvalidCodePoint;
            int n = PyUnicode.TransformDecimalAndSpaceToAscii(units, scratch);
            return ParseAscii(scratch.Slice(0, n), target, canon, scratch.Slice(n), ref d);
        }

        /// <summary>
        ///     Parses a Python <c>bytes</c> (a trimmed <c>S</c> element or a C# byte[]): raw for <c>int()</c> /
        ///     <c>float()</c>, strict-UTF-8-decoded then rewritten for <c>complex()</c>.
        /// </summary>
        /// <param name="bytes">The value's bytes (an element's trailing NULs removed).</param>
        /// <param name="target">The numeric target dtype.</param>
        /// <param name="canon">Receives the canonical value on success.</param>
        /// <param name="scratch">Work space of <see cref="ScratchBytes"/>(<c>bytes.Length</c>) bytes.</param>
        /// <param name="d">Receives the failure details (the decode range for a <see cref="TextParseStatus.DecodeError"/>).</param>
        /// <returns>The outcome.</returns>
        private static TextParseStatus ParseBytesCore(ReadOnlySpan<byte> bytes, NPTypeCode target, byte* canon,
            Span<byte> scratch, ref Detail d)
        {
            if (FamilyOf(target) != Family.Complex)
                // int() and float() read bytes as they are: no rewrite, a byte above 0x7F is just an invalid character.
                return ParseAscii(bytes, target, canon, scratch, ref d);

            // complex() needs a str: CDOUBLE_setitem decodes the bytes as strict UTF-8 FIRST (the whole value, so a bad
            // byte anywhere raises), then complex() runs its Unicode rewrite on the decoded text.
            Span<uint> cps = MemoryMarshalCast(scratch.Slice(0, 4 * bytes.Length));
            if (!DecodeUtf8(bytes, cps, out int count, ref d))
                return TextParseStatus.DecodeError;
            Span<byte> ascii = scratch.Slice(4 * bytes.Length);
            int n = PyUnicode.TransformDecimalAndSpaceToAscii(cps.Slice(0, count), ascii);
            return ParseAscii(ascii.Slice(0, n), target, canon, ascii.Slice(n), ref d);
        }

        /// <summary>Views a byte span as UCS-4 units (the scratch is 4-byte sized by <see cref="ScratchBytes"/>).</summary>
        /// <param name="bytes">The bytes (length a multiple of 4).</param>
        /// <returns>The same memory as <see cref="uint"/>s.</returns>
        private static Span<uint> MemoryMarshalCast(Span<byte> bytes)
            => System.Runtime.InteropServices.MemoryMarshal.Cast<byte, uint>(bytes);

        /// <summary>Converts a C# string to code points: a valid surrogate pair becomes one code point.</summary>
        /// <param name="s">The text.</param>
        /// <returns>The code points.</returns>
        private static uint[] ToCodePoints(string s)
        {
            int n = NDStringCodec.CodePointCount(s);
            var cps = new uint[n];
            int k = 0;
            for (int i = 0; i < s.Length; i++)
            {
                char c = s[i];
                if (char.IsHighSurrogate(c) && i + 1 < s.Length && char.IsLowSurrogate(s[i + 1]))
                {
                    cps[k++] = (uint)char.ConvertToUtf32(c, s[i + 1]);
                    i++;
                }
                else
                    cps[k++] = c;
            }
            return cps;
        }

        /// <summary>Runs the target family's grammar over the rewritten ASCII text and packs the value.</summary>
        /// <param name="ascii">The text after the Unicode rewrite (or raw <c>S</c> bytes for int / float).</param>
        /// <param name="target">The target dtype.</param>
        /// <param name="canon">Receives the canonical value.</param>
        /// <param name="work">Work space of at least <c>ascii.Length</c> bytes (the underscore-free copy).</param>
        /// <param name="d">Receives the failure details.</param>
        /// <returns>The outcome.</returns>
        private static TextParseStatus ParseAscii(ReadOnlySpan<byte> ascii, NPTypeCode target, byte* canon, Span<byte> work, ref Detail d)
        {
            switch (FamilyOf(target))
            {
                case Family.Signed:
                case Family.Unsigned:
                {
                    var status = ParseInt(ascii, out bool negative, out ulong magnitude, out bool beyond64, out int digits);
                    if (status == TextParseStatus.DigitLimit)
                        d.Digits = digits;
                    if (status != TextParseStatus.Ok)
                        return status;
                    return Pack(negative, magnitude, beyond64, target, canon, ref d);
                }
                case Family.Float:
                {
                    var status = ParseFloat(ascii, work, out double v);
                    if (status == TextParseStatus.Ok)
                        *(double*)canon = v;
                    return status;
                }
                default:
                {
                    var status = ParseComplex(ascii, work, out double re, out double im);
                    if (status == TextParseStatus.Ok)
                        *(Complex*)canon = new Complex(re, im);
                    return status;
                }
            }
        }

        // ---------------------------------------------------------------------------------------------
        // int()
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     CPython 3.12's <c>PyLong_FromString(s, &amp;end, 10)</c> plus its caller's "parsed to the end" check
        ///     (<c>PyLong_FromUnicodeObject</c> / <c>_PyLong_FromBytes</c>), over ASCII text.
        /// </summary>
        /// <param name="s">The text (may hold NULs and, for raw bytes, characters above 0x7F).</param>
        /// <param name="negative">Receives the sign.</param>
        /// <param name="magnitude">Receives the absolute value when it fits 64 bits.</param>
        /// <param name="beyond64">Receives whether the absolute value is past <see cref="ulong.MaxValue"/>.</param>
        /// <param name="digits">Receives the digit count (underscores excluded, leading zeros included).</param>
        /// <returns><see cref="TextParseStatus.Ok"/>, <see cref="TextParseStatus.InvalidInt"/> or
        ///     <see cref="TextParseStatus.DigitLimit"/>.</returns>
        /// <remarks>
        ///     The C string <c>PyLong_FromString</c> reads ends at the first NUL; the parse up to it is validated (trailing
        ///     junk is a syntax error) and only then is the digit limit checked, so a syntax error wins over the limit; the
        ///     caller then rejects a parse that did not reach the text's true end (a NUL inside) — after the limit, which
        ///     propagates on its own.
        /// </remarks>
        private static TextParseStatus ParseInt(ReadOnlySpan<byte> s, out bool negative, out ulong magnitude, out bool beyond64, out int digits)
        {
            negative = false;
            magnitude = 0;
            beyond64 = false;
            digits = 0;
            int len = s.Length;
            int cend = s.IndexOf((byte)0);
            if (cend < 0)
                cend = len;

            int p = 0;
            while (p < cend && PyUnicode.IsAsciiSpace(s[p]))
                p++;
            if (p < cend && (s[p] == '+' || s[p] == '-'))
            {
                negative = s[p] == '-';
                p++;
            }
            // long_from_string_base: no leading underscore, no doubled one, no trailing one.
            if (p < cend && s[p] == '_')
                return TextParseStatus.InvalidInt;
            int start = p;
            byte prev = 0;
            while (p < cend && ((uint)(s[p] - '0') <= 9 || s[p] == '_'))
            {
                if (s[p] == '_')
                {
                    if (prev == '_')
                        return TextParseStatus.InvalidInt;
                }
                else
                    digits++;
                prev = s[p];
                p++;
            }
            if (prev == '_' || p == start)
                return TextParseStatus.InvalidInt;
            int digitEnd = p;
            while (p < cend && PyUnicode.IsAsciiSpace(s[p]))
                p++;
            if (p < cend)
                return TextParseStatus.InvalidInt;
            if (digits > MaxStrDigits)
                return TextParseStatus.DigitLimit;
            if (cend != len)
                return TextParseStatus.InvalidInt;

            // The value, or just "past 64 bits" (every target rejects that with the same C long text).
            ulong m = 0;
            for (int i = start; i < digitEnd; i++)
            {
                uint dv = (uint)(s[i] - '0');
                if (dv > 9)
                    continue; // an underscore
                if (m > (ulong.MaxValue - dv) / 10)
                {
                    beyond64 = true;
                    break;
                }
                m = m * 10 + dv;
            }
            magnitude = m;
            return TextParseStatus.Ok;
        }

        /// <summary>
        ///     NumPy's <c>@TYPE@_safe_pyint_setitem</c> under LP64 (see the class remarks): packs the Python int into the
        ///     family's canonical value, or names why it does not fit.
        /// </summary>
        /// <param name="negative">The value's sign.</param>
        /// <param name="m">Its magnitude (when <paramref name="beyond64"/> is false).</param>
        /// <param name="beyond64">Whether the magnitude is past 64 bits.</param>
        /// <param name="target">The integer target dtype.</param>
        /// <param name="canon">Receives the <see cref="long"/> / <see cref="ulong"/> value.</param>
        /// <param name="d">Receives the value for an <see cref="TextParseStatus.OutOfBounds"/> text.</param>
        /// <returns>The outcome.</returns>
        private static TextParseStatus Pack(bool negative, ulong m, bool beyond64, NPTypeCode target, byte* canon, ref Detail d)
        {
            if (beyond64)
                return TextParseStatus.TooLargeForCLong;
            d.Negative = negative;
            d.Magnitude = m;
            bool fitsLong = negative ? m <= 1UL << 63 : m <= long.MaxValue;

            if (target == NPTypeCode.UInt32 || target == NPTypeCode.UInt64)
            {
                // MyPyLong_AsUnsignedLongWithWrap: a negative value retries as C long and sets the wrap flag.
                if (negative)
                    return m == 0 ? Store(0UL) : fitsLong ? TextParseStatus.OutOfBounds : TextParseStatus.TooLargeForCLong;
                if (target == NPTypeCode.UInt32 && m > uint.MaxValue)
                    return TextParseStatus.OutOfBounds;
                return Store(m);
            }

            // Every other integer target converts through C long first.
            if (!fitsLong)
                return TextParseStatus.TooLargeForCLong;
            long v = negative ? (long)(0UL - m) : (long)m;
            (long min, long max) = target switch
            {
                NPTypeCode.SByte => (sbyte.MinValue, sbyte.MaxValue),
                NPTypeCode.Int16 => (short.MinValue, short.MaxValue),
                NPTypeCode.Int32 => (int.MinValue, int.MaxValue),
                NPTypeCode.Int64 => (long.MinValue, long.MaxValue),
                NPTypeCode.Byte => (0, byte.MaxValue),
                _ => (0, ushort.MaxValue), // uint16 and char
            };
            if (v < min || v > max)
                return TextParseStatus.OutOfBounds;
            if (FamilyOf(target) == Family.Signed)
                *(long*)canon = v;
            else
                *(ulong*)canon = (ulong)v;
            return TextParseStatus.Ok;

            TextParseStatus Store(ulong value)
            {
                *(ulong*)canon = value;
                return TextParseStatus.Ok;
            }
        }

        // ---------------------------------------------------------------------------------------------
        // float()
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     CPython's <c>float(str)</c> after the Unicode rewrite: <c>_Py_string_to_number_with_underscores</c> then
        ///     <c>float_from_string_inner</c>.
        /// </summary>
        /// <param name="s">The text.</param>
        /// <param name="work">Room for the underscore-free copy.</param>
        /// <param name="value">Receives the value.</param>
        /// <returns><see cref="TextParseStatus.Ok"/> or <see cref="TextParseStatus.InvalidFloat"/>.</returns>
        private static TextParseStatus ParseFloat(ReadOnlySpan<byte> s, Span<byte> work, out double value)
        {
            value = 0;
            switch (StripUnderscores(s, work, out int stripped))
            {
                case UnderscoreResult.Invalid:
                    return TextParseStatus.InvalidFloat;
                case UnderscoreResult.Stripped:
                    s = work.Slice(0, stripped);
                    break;
            }

            // float_from_string_inner: surrounding whitespace, then one strtod prefix that must reach the end.
            int a = 0, last = s.Length;
            while (a < last && PyUnicode.IsAsciiSpace(s[a]))
                a++;
            if (a == last)
                return TextParseStatus.InvalidFloat;
            while (a < last - 1 && PyUnicode.IsAsciiSpace(s[last - 1]))
                last--;
            int end = FloatPrefix(s, a, out value);
            return end == last ? TextParseStatus.Ok : TextParseStatus.InvalidFloat;
        }

        /// <summary>The outcome of <see cref="StripUnderscores"/>.</summary>
        private enum UnderscoreResult : byte
        {
            /// <summary>The C string holds no underscore: parse the text as it is.</summary>
            None,

            /// <summary>The underscores were valid and removed: parse the copy.</summary>
            Stripped,

            /// <summary>An underscore not between two digits, or a NUL after the first underscore.</summary>
            Invalid,
        }

        /// <summary>
        ///     CPython's <c>_Py_string_to_number_with_underscores</c>: when the C string (up to the first NUL) holds an
        ///     underscore, every underscore must sit between two ASCII digits and the text must hold no NUL; the copy without
        ///     underscores is what the number grammar parses.
        /// </summary>
        /// <param name="s">The text.</param>
        /// <param name="dup">Receives the copy without underscores.</param>
        /// <param name="length">Receives the copy's length.</param>
        /// <returns>The outcome.</returns>
        private static UnderscoreResult StripUnderscores(ReadOnlySpan<byte> s, Span<byte> dup, out int length)
        {
            length = 0;
            int cend = s.IndexOf((byte)0);
            ReadOnlySpan<byte> c = cend < 0 ? s : s.Slice(0, cend);
            // strchr(s, '_') stops at the first NUL: an underscore after one is not seen.
            if (c.IndexOf((byte)'_') < 0)
                return UnderscoreResult.None;

            byte prev = 0;
            int p = 0;
            for (; p < c.Length; p++)
            {
                byte ch = c[p];
                if (ch == '_')
                {
                    // Underscores are only allowed after digits.
                    if (!((uint)(prev - '0') <= 9))
                        return UnderscoreResult.Invalid;
                }
                else
                {
                    dup[length++] = ch;
                    // Underscores are only allowed before digits.
                    if (prev == '_' && !((uint)(ch - '0') <= 9))
                        return UnderscoreResult.Invalid;
                }
                prev = ch;
            }
            // Underscores are not allowed at the end, and no embedded NUL is allowed.
            if (prev == '_' || p != s.Length)
                return UnderscoreResult.Invalid;
            return UnderscoreResult.Stripped;
        }

        /// <summary>
        ///     CPython's <c>_PyOS_ascii_strtod</c> as a PREFIX parser: <c>_Py_dg_strtod</c>'s grammar, and when that consumes
        ///     nothing, <c>_Py_parse_inf_or_nan</c>. The text is a C string: reading past its end reads NUL.
        /// </summary>
        /// <param name="s">The text.</param>
        /// <param name="i">Where the number starts.</param>
        /// <param name="value">Receives the value (correctly rounded; ±inf on overflow, ±0 on underflow).</param>
        /// <returns>The index after the number, or <paramref name="i"/> when nothing parsed.</returns>
        private static int FloatPrefix(ReadOnlySpan<byte> s, int i, out double value)
        {
            int end = StrtodEnd(s, i);
            if (end == i)
                return InfNanPrefix(s, i, out value);
            // The validated run [i, end) is [sign]digits[.digits][e[sign]digits]: .NET's IEEE-correct parser rounds it
            // exactly as _Py_dg_strtod does (correctly rounded, exponent overflow to inf, underflow to zero, -0 kept).
            value = double.Parse(s.Slice(i, end - i), NumberStyles.Float, CultureInfo.InvariantCulture);
            return end;
        }

        /// <summary>
        ///     The byte at <paramref name="i"/>, or NUL past the end: CPython's parsers read a NUL-terminated C string, so
        ///     reading one past the text is reading its terminator, which matches no grammar rule.
        /// </summary>
        /// <param name="s">The text.</param>
        /// <param name="i">The index (may be past the end).</param>
        /// <returns>The byte, or 0.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static byte At(ReadOnlySpan<byte> s, int i) => (uint)i < (uint)s.Length ? s[i] : (byte)0;

        /// <summary>
        ///     The end of <c>_Py_dg_strtod</c>'s parse from <paramref name="i"/>: <c>[+-]</c>, digits, an optional point and
        ///     digits (at least one digit or one leading zero overall), an optional exponent that is consumed only when it
        ///     has a digit.
        /// </summary>
        /// <param name="s">The text.</param>
        /// <param name="i">The start.</param>
        /// <returns>The end, or <paramref name="i"/> when the grammar does not match (a parse error).</returns>
        private static int StrtodEnd(ReadOnlySpan<byte> s, int i)
        {
            int p = i;
            byte c = At(s, p);
            if (c == '-' || c == '+')
                p++;
            int z = p;
            while (At(s, p) == '0')
                p++;
            bool lz = p != z;
            int d0 = p;
            while ((uint)(At(s, p) - '0') <= 9)
                p++;
            long ndigits = p - d0;
            if (At(s, p) == '.')
            {
                p++;
                if (ndigits == 0)
                {
                    int z2 = p;
                    while (At(s, p) == '0')
                        p++;
                    lz = lz || p != z2;
                }
                int f = p;
                while ((uint)(At(s, p) - '0') <= 9)
                    p++;
                ndigits += p - f;
            }
            if (ndigits == 0 && !lz)
                return i;
            c = At(s, p);
            if (c == 'e' || c == 'E')
            {
                int eStart = p;
                p++;
                c = At(s, p);
                if (c == '-' || c == '+')
                    p++;
                int ez = p;
                while (At(s, p) == '0')
                    p++;
                bool elz = p != ez;
                int ed = p;
                while ((uint)(At(s, p) - '0') <= 9)
                    p++;
                // A valid exponent must have at least one digit: otherwise the 'e' is not part of the number.
                if (p == ed && !elz)
                    p = eStart;
            }
            return p;
        }

        /// <summary>
        ///     CPython's <c>_Py_parse_inf_or_nan</c>: an optional sign, then <c>inf</c> (optionally <c>inity</c>) or
        ///     <c>nan</c>, case-insensitively.
        /// </summary>
        /// <param name="s">The text.</param>
        /// <param name="i">The start.</param>
        /// <param name="value">Receives ±inf, or NaN with the sign bit set for <c>-nan</c>.</param>
        /// <returns>The end, or <paramref name="i"/> when neither word is there.</returns>
        private static int InfNanPrefix(ReadOnlySpan<byte> s, int i, out double value)
        {
            int p = i;
            bool negate = false;
            if (At(s, p) == '-')
            {
                negate = true;
                p++;
            }
            else if (At(s, p) == '+')
                p++;
            if (MatchesIgnoreCase(s, p, "inf"))
            {
                p += 3;
                if (MatchesIgnoreCase(s, p, "inity"))
                    p += 5;
                value = negate ? double.NegativeInfinity : double.PositiveInfinity;
                return p;
            }
            if (MatchesIgnoreCase(s, p, "nan"))
            {
                value = negate ? NegativeNaN : PositiveNaN;
                return p + 3;
            }
            value = -1.0;
            return i;
        }

        /// <summary>
        ///     CPython's <c>case_insensitive_match</c>: whether <paramref name="word"/> (lowercase ASCII) appears at
        ///     <paramref name="p"/>, in any case; a text that ends early does not match.
        /// </summary>
        /// <param name="s">The text.</param>
        /// <param name="p">Where the word must start.</param>
        /// <param name="word">The lowercase word (<c>inf</c>, <c>inity</c>, <c>nan</c>).</param>
        /// <returns>True when every character matches.</returns>
        private static bool MatchesIgnoreCase(ReadOnlySpan<byte> s, int p, string word)
        {
            for (int k = 0; k < word.Length; k++)
            {
                int c = At(s, p + k);
                if (c >= 'A' && c <= 'Z')
                    c += 32;
                if (c != word[k])
                    return false;
            }
            return true;
        }

        // ---------------------------------------------------------------------------------------------
        // complex()
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     CPython's <c>complex(str)</c> after the Unicode rewrite: <c>_Py_string_to_number_with_underscores</c> (an
        ///     underscore error is its own text) then <c>complex_from_string_inner</c>.
        /// </summary>
        /// <param name="s">The text.</param>
        /// <param name="work">Room for the underscore-free copy.</param>
        /// <param name="re">Receives the real part.</param>
        /// <param name="im">Receives the imaginary part.</param>
        /// <returns><see cref="TextParseStatus.Ok"/>, <see cref="TextParseStatus.ComplexUnderscore"/> or
        ///     <see cref="TextParseStatus.InvalidComplex"/>.</returns>
        private static TextParseStatus ParseComplex(ReadOnlySpan<byte> s, Span<byte> work, out double re, out double im)
        {
            re = 0;
            im = 0;
            switch (StripUnderscores(s, work, out int stripped))
            {
                case UnderscoreResult.Invalid:
                    return TextParseStatus.ComplexUnderscore;
                case UnderscoreResult.Stripped:
                    s = work.Slice(0, stripped);
                    break;
            }

            int len = s.Length;
            int p = 0;
            while (PyUnicode.IsAsciiSpace(At(s, p)))
                p++;
            bool gotBracket = false;
            if (At(s, p) == '(')
            {
                gotBracket = true;
                p++;
                while (PyUnicode.IsAsciiSpace(At(s, p)))
                    p++;
            }

            double x = 0, y = 0;
            int end = FloatPrefix(s, p, out double z);
            if (end != p)
            {
                // All forms starting with <float>.
                p = end;
                byte c = At(s, p);
                if (c == '+' || c == '-')
                {
                    // <float><signed-float>j | <float><sign>j
                    x = z;
                    int end2 = FloatPrefix(s, p, out y);
                    if (end2 != p)
                        p = end2;
                    else
                    {
                        y = c == '+' ? 1.0 : -1.0;
                        p++;
                    }
                    c = At(s, p);
                    if (c != 'j' && c != 'J')
                        return TextParseStatus.InvalidComplex;
                    p++;
                }
                else if (c == 'j' || c == 'J')
                {
                    // <float>j
                    p++;
                    y = z;
                }
                else
                    // <float>
                    x = z;
            }
            else
            {
                // Not starting with <float>: must be <sign>j or j.
                byte c = At(s, p);
                if (c == '+' || c == '-')
                {
                    y = c == '+' ? 1.0 : -1.0;
                    p++;
                }
                else
                    y = 1.0;
                c = At(s, p);
                if (c != 'j' && c != 'J')
                    return TextParseStatus.InvalidComplex;
                p++;
            }

            // Trailing whitespace and the closing bracket.
            while (PyUnicode.IsAsciiSpace(At(s, p)))
                p++;
            if (gotBracket)
            {
                if (At(s, p) != ')')
                    return TextParseStatus.InvalidComplex;
                p++;
                while (PyUnicode.IsAsciiSpace(At(s, p)))
                    p++;
            }
            // We should now be at the end of the string (an embedded NUL stops every step above before it).
            if (p != len)
                return TextParseStatus.InvalidComplex;
            re = x;
            im = y;
            return TextParseStatus.Ok;
        }

        // ---------------------------------------------------------------------------------------------
        // Strict UTF-8 (S -> complex)
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     CPython 3.12's strict UTF-8 decoder (<c>stringlib/codecs.h</c>, <c>utf8_decode</c>, non-incremental): decodes
        ///     the whole value or reports the FIRST error with CPython's byte range and reason.
        /// </summary>
        /// <param name="b">The bytes.</param>
        /// <param name="cps">Receives the code points (room for <c>b.Length</c>).</param>
        /// <param name="count">Receives the number of code points.</param>
        /// <param name="d">Receives the error range and reason on failure.</param>
        /// <returns>True when the value is valid UTF-8.</returns>
        /// <remarks>
        ///     The reasons and ranges, as CPython reports them: a byte that cannot start a sequence (0x80..0xC1, 0xF5..0xFF)
        ///     is <c>invalid start byte</c> over that byte; a continuation that is missing, out of range, or makes an overlong
        ///     form, a surrogate (<c>ED A0..</c>) or a value past U+10FFFF (<c>F4 90..</c>) is <c>invalid continuation byte</c>
        ///     over the bytes BEFORE the bad one; a sequence cut off by the end of the value is <c>unexpected end of data</c>
        ///     over the bytes to the end — unless an early byte already proves it invalid, which is reported as a
        ///     continuation error.
        /// </remarks>
        private static bool DecodeUtf8(ReadOnlySpan<byte> b, Span<uint> cps, out int count, ref Detail d)
        {
            count = 0;
            int i = 0, n = b.Length;
            while (i < n)
            {
                uint ch = b[i];
                if (ch < 0x80)
                {
                    cps[count++] = ch;
                    i++;
                    continue;
                }
                if (ch < 0xE0)
                {
                    if (ch < 0xC2)
                        return Fail(ref d, i, i + 1, "invalid start byte");
                    if (n - i < 2)
                        return Fail(ref d, i, n, "unexpected end of data");
                    uint c2 = b[i + 1];
                    if (!IsContinuation(c2))
                        return Fail(ref d, i, i + 1, "invalid continuation byte");
                    cps[count++] = ((ch & 0x1F) << 6) | (c2 & 0x3F);
                    i += 2;
                    continue;
                }
                if (ch < 0xF0)
                {
                    if (n - i < 3)
                    {
                        if (n - i >= 2)
                        {
                            uint t2 = b[i + 1];
                            if (!IsContinuation(t2) || (t2 < 0xA0 ? ch == 0xE0 : ch == 0xED))
                                return Fail(ref d, i, i + 1, "invalid continuation byte");
                        }
                        return Fail(ref d, i, n, "unexpected end of data");
                    }
                    uint c2 = b[i + 1], c3 = b[i + 2];
                    if (!IsContinuation(c2) || (ch == 0xE0 && c2 < 0xA0) || (ch == 0xED && c2 >= 0xA0))
                        return Fail(ref d, i, i + 1, "invalid continuation byte");
                    if (!IsContinuation(c3))
                        return Fail(ref d, i, i + 2, "invalid continuation byte");
                    cps[count++] = ((ch & 0x0F) << 12) | ((c2 & 0x3F) << 6) | (c3 & 0x3F);
                    i += 3;
                    continue;
                }
                if (ch < 0xF5)
                {
                    if (n - i < 4)
                    {
                        if (n - i >= 2)
                        {
                            uint t2 = b[i + 1];
                            if (!IsContinuation(t2) || (t2 < 0x90 ? ch == 0xF0 : ch == 0xF4))
                                return Fail(ref d, i, i + 1, "invalid continuation byte");
                            if (n - i >= 3 && !IsContinuation(b[i + 2]))
                                return Fail(ref d, i, i + 2, "invalid continuation byte");
                        }
                        return Fail(ref d, i, n, "unexpected end of data");
                    }
                    uint c2 = b[i + 1], c3 = b[i + 2], c4 = b[i + 3];
                    if (!IsContinuation(c2) || (ch == 0xF0 && c2 < 0x90) || (ch == 0xF4 && c2 >= 0x90))
                        return Fail(ref d, i, i + 1, "invalid continuation byte");
                    if (!IsContinuation(c3))
                        return Fail(ref d, i, i + 2, "invalid continuation byte");
                    if (!IsContinuation(c4))
                        return Fail(ref d, i, i + 3, "invalid continuation byte");
                    cps[count++] = ((ch & 0x07) << 18) | ((c2 & 0x3F) << 12) | ((c3 & 0x3F) << 6) | (c4 & 0x3F);
                    i += 4;
                    continue;
                }
                return Fail(ref d, i, i + 1, "invalid start byte");
            }
            return true;

            static bool IsContinuation(uint c) => (c & 0xC0) == 0x80;

            static bool Fail(ref Detail d, int start, int end, string reason)
            {
                d.DecodeStart = start;
                d.DecodeEnd = end;
                d.DecodeReason = reason;
                return false;
            }
        }

        // ---------------------------------------------------------------------------------------------
        // Error texts
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     Builds NumPy's exception for a failed parse.
        /// </summary>
        /// <param name="status">Why it failed.</param>
        /// <param name="d">The failure details.</param>
        /// <param name="target">The target dtype.</param>
        /// <param name="strRepr">The repr of the text when it is a <c>str</c> (already wrapped in <c>np.str_(…)</c> for an
        ///     array element), or null for bytes. Built by the caller on the error path only.</param>
        /// <param name="bytes">The bytes when the text is <c>bytes</c> (trailing NULs already removed for an element), else
        ///     null.</param>
        /// <param name="style">How the text is quoted.</param>
        /// <returns>The exception (not thrown).</returns>
        /// <exception cref="InvalidOperationException"><paramref name="status"/> is <see cref="TextParseStatus.Ok"/> (a
        ///     caller bug: there is no error to build).</exception>
        private static Exception Build(TextParseStatus status, Detail d, NPTypeCode target, string strRepr, byte[] bytes, TextReprStyle style)
        {
            switch (status)
            {
                case TextParseStatus.InvalidInt:
                {
                    // "%.200R": the repr cut at 200 code points. int() of bytes quotes a NEW bytes object of at most 200
                    // bytes (_PyLong_FromBytes), so it is always b'...', never np.bytes_(...).
                    string repr = bytes != null ? PyRepr.Bytes(bytes.AsSpan(0, Math.Min(bytes.Length, 200))) : strRepr;
                    return new ValueError("invalid literal for int() with base 10: " + CutCodePoints(repr, 200));
                }
                case TextParseStatus.DigitLimit:
                    return new ValueError($"Exceeds the limit ({MaxStrDigits} digits) for integer string conversion: value has " +
                                          $"{d.Digits} digits; use sys.set_int_max_str_digits() to increase the limit");
                case TextParseStatus.OutOfBounds:
                {
                    string v = (d.Negative && d.Magnitude != 0 ? "-" : "") + d.Magnitude.ToString(CultureInfo.InvariantCulture);
                    return new OverflowException($"Python integer {v} out of bounds for {target.AsNumpyDtypeName()}");
                }
                case TextParseStatus.TooLargeForCLong:
                    return new OverflowException("Python int too large to convert to C long");
                case TextParseStatus.InvalidFloat:
                {
                    // float() quotes the object it got: np.bytes_(b'...') for an array element, b'...' for a bytes object.
                    string repr = bytes == null
                        ? strRepr
                        : style == TextReprStyle.ArrayElement ? "np.bytes_(" + PyRepr.Bytes(bytes) + ")" : PyRepr.Bytes(bytes);
                    return new ValueError("could not convert string to float: " + repr);
                }
                case TextParseStatus.ComplexUnderscore:
                {
                    // complex() quotes the str it parsed: for bytes that is the DECODED text, a plain str.
                    string repr = bytes == null ? strRepr : PyRepr.StrCodePoints(DecodeForRepr(bytes));
                    return new ValueError("could not convert string to complex: " + repr);
                }
                case TextParseStatus.InvalidComplex:
                    return new ValueError("complex() arg is a malformed string");
                case TextParseStatus.InvalidCodePoint:
                    return new SystemError(NDStringCodec.InvalidMaxCharMessage);
                case TextParseStatus.DecodeError:
                {
                    string text = d.DecodeEnd - d.DecodeStart == 1
                        ? $"'utf-8' codec can't decode byte 0x{bytes[d.DecodeStart]:x2} in position {d.DecodeStart}: {d.DecodeReason}"
                        : $"'utf-8' codec can't decode bytes in position {d.DecodeStart}-{d.DecodeEnd - 1}: {d.DecodeReason}";
                    return new UnicodeDecodeError(text);
                }
                default:
                    throw new InvalidOperationException($"No error for parse status {status}.");
            }
        }

        /// <summary>The code points of bytes already proven to be valid UTF-8 (for a repr).</summary>
        /// <param name="b">The bytes.</param>
        /// <returns>The decoded code points.</returns>
        private static uint[] DecodeForRepr(byte[] b)
        {
            var cps = new uint[b.Length];
            Detail d = default;
            DecodeUtf8(b, cps, out int count, ref d);
            return cps.AsSpan(0, count).ToArray();
        }

        /// <summary>CPython's <c>%.200R</c>: the repr cut at <paramref name="max"/> code points (a surrogate pair is one).</summary>
        /// <param name="repr">The repr.</param>
        /// <param name="max">The code point limit.</param>
        /// <returns>The cut repr.</returns>
        private static string CutCodePoints(string repr, int max)
        {
            if (repr.Length <= max)
                return repr;
            int units = 0;
            for (int cps = 0; cps < max && units < repr.Length; cps++)
                units += char.IsHighSurrogate(repr[units]) && units + 1 < repr.Length && char.IsLowSurrogate(repr[units + 1]) ? 2 : 1;
            return repr.Substring(0, units);
        }
    }
}
