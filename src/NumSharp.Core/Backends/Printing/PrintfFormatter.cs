using System;
using System.Globalization;
using System.Numerics;
using System.Text;
using NumSharp.Backends.Strings;

namespace NumSharp.Backends.Printing
{
    /// <summary>
    ///     A port of CPython 3.12's <c>str.__mod__</c> (the <c>%</c> operator, <c>PyUnicode_Format</c> in
    ///     <c>Objects/unicodeobject.c</c>) as NumPy uses it: <c>np.savetxt</c> writes each row as
    ///     <c>format % tuple(row)</c> and <c>ndarray.tofile(sep, format)</c> each element as <c>format % (item,)</c>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <see cref="PercentFormat"/> is the parse (a format is compiled once per call and run per row or element); this
    ///     class holds the conversions. Both follow CPython exactly, errors included: a conversion the value's type does
    ///     not take raises CPython's <c>TypeError</c> (<see cref="PrintfArgumentException"/>), a NaN under an integer
    ///     conversion its <c>ValueError</c>, an infinity or a <c>%c</c> code point out of range its <c>OverflowError</c>
    ///     (<see cref="OverflowException"/>), and a malformed format <c>incomplete format</c> or <c>unsupported format
    ///     character ...</c>. Nothing is rendered "best effort": the writer raises where NumPy raises.
    ///     </para>
    ///     <para>
    ///     Conversions: <c>s</c> / <c>r</c> / <c>a</c> (<c>str()</c> / <c>repr()</c> / <c>ascii()</c>, cut to the
    ///     precision and padded to the width in code points); <c>d i u</c> (decimal; a float truncates toward zero);
    ///     <c>x X o</c> (an integer only; the <c>#</c> form prefixes <c>0x</c> / <c>0X</c> / <c>0o</c>, even for 0);
    ///     <c>f F</c>, <c>e E</c>, <c>g G</c> (default precision 6, a two-digit-minimum exponent, <c>nan</c> /
    ///     <c>inf</c>); <c>c</c> (a code point, or a one-character <c>str</c>). Flags <c>- + space 0 #</c>, a width and a
    ///     <c>.precision</c> (each may be <c>*</c>, a Python <c>int</c> argument) as CPython applies them.
    ///     </para>
    ///     <para>
    ///     The value of each conversion is either a Python object (<see cref="ApplyPyObject"/>) or a NumPy scalar of a
    ///     dtype (<see cref="ApplyNumPyScalar"/>); see <see cref="PercentFormat"/> for which writer passes which.
    ///     NumSharp's two dtypes without a NumPy scalar follow their established NumPy analogs: a <c>Char</c> element is
    ///     a <c>uint16</c> (<c>%s</c> is its number, <c>%c</c> its character) and a <c>Decimal</c> element a
    ///     <c>float64</c> of its value, as <see cref="ArrayFormatter.ScalarStr"/> prints them.
    ///     </para>
    /// </remarks>
    internal static partial class PrintfFormatter
    {
        private static readonly CultureInfo CI = CultureInfo.InvariantCulture;

        /// <summary>
        ///     The largest precision .NET's <c>F</c> / <c>E</c> formats accept (999,999,999 on .NET 7 and later). A longer
        ///     result could not be held anyway: it passes the longest string .NET allows.
        /// </summary>
        private const int MaxNetPrecision = 999_999_999;

        /// <summary>
        ///     A precision a <c>%g</c> conversion without the <c>#</c> flag never needs more of: a double has at most 767
        ///     significant digits, and without <c>#</c> the trailing zeros past them are stripped, so any larger precision
        ///     prints the same text (CPython prints <c>'%.2147483647g' % 1.0</c> as <c>'1'</c>, in no time).
        /// </summary>
        private const int MaxUsefulGeneralPrecision = 1100;

        /// <summary>Compiles a Python <c>%</c> format once, for a writer that applies it many times.</summary>
        /// <param name="format">The format string.</param>
        /// <returns>The compiled format (<see cref="PercentFormat.Compile"/>).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="format"/> is null.</exception>
        public static PercentFormat Compile(string format) => PercentFormat.Compile(format);

        /// <summary>
        ///     <c>format % (value,)</c> — a one-item tuple, the way <c>ndarray.tofile</c> applies its format to each
        ///     element — compiled and run once.
        /// </summary>
        /// <param name="format">The Python <c>%</c> format string.</param>
        /// <param name="value">The item.</param>
        /// <param name="kind">The item's kind: <see cref="PercentFormat.PyObject"/> for a Python object, else the dtype of
        ///     the NumPy scalar it stands for.</param>
        /// <returns>The formatted text.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="format"/> is null.</exception>
        /// <exception cref="PrintfArgumentException">CPython's <c>TypeError</c>s (see
        ///     <see cref="PercentFormat.FormatInto"/>).</exception>
        /// <exception cref="ValueError">CPython's <c>ValueError</c>s (see <see cref="PercentFormat.FormatInto"/>).</exception>
        /// <exception cref="OverflowException">CPython's <c>OverflowError</c>s (see
        ///     <see cref="PercentFormat.FormatInto"/>).</exception>
        /// <exception cref="OutOfMemoryException">A result longer than .NET's longest string.</exception>
        public static string Format(string format, object value, NPTypeCode kind)
        {
            var sb = new StringBuilder(format?.Length + 16 ?? 16);
            object[] args = { value };
            NPTypeCode[] kinds = { kind };
            Compile(format).FormatInto(sb, args, kinds);
            return sb.ToString();
        }

        /// <summary>
        ///     One CPython <c>%</c> conversion of a NUMPY SCALAR: an element of <c>np.savetxt</c>'s <c>tuple(row)</c> (and
        ///     of its complex branch's <c>(z.real, z.imag)</c> pairs, which are <c>np.float64</c>).
        /// </summary>
        /// <param name="conv">The conversion character, one <see cref="PercentFormat"/> has checked is supported.</param>
        /// <param name="left">The <c>-</c> flag (or a negative <c>*</c> width).</param>
        /// <param name="plus">The <c>+</c> flag.</param>
        /// <param name="space">The <c> </c> flag.</param>
        /// <param name="zero">The <c>0</c> flag.</param>
        /// <param name="alt">The <c>#</c> flag.</param>
        /// <param name="width">The minimum width (0 for none).</param>
        /// <param name="precision">The precision (-1 for none).</param>
        /// <param name="value">The boxed scalar.</param>
        /// <param name="tc">Its dtype (<see cref="NPTypeCode.String"/>: an <c>np.str_</c> / <c>np.bytes_</c>, a C#
        ///     <see cref="string"/> / <c>byte[]</c>).</param>
        /// <returns>The converted text, padded to the width.</returns>
        /// <remarks>
        ///     <para>
        ///     NumPy 2.4.2's scalars under CPython's rules, probed for every dtype: <c>%s</c> is the scalar's
        ///     <c>str()</c> (<c>0.1</c> for an <c>np.float32</c>, not the widened double); <c>%r</c> / <c>%a</c> its
        ///     <c>repr()</c> (<c>np.float64(1.5)</c>, <c>np.True_</c>, <c>np.uint8(200)</c>); <c>%d</c> / <c>%i</c> /
        ///     <c>%u</c> its <c>int()</c> (<c>np.True_</c> is 1, a float truncates, NaN and infinity raise); <c>%x</c> /
        ///     <c>%X</c> / <c>%o</c> and <c>%c</c> only an integer (an <c>np.bool</c> is NOT one: its <c>__index__</c>
        ///     raises); <c>%e</c> / <c>%f</c> / <c>%g</c> its <c>float()</c> (an <c>np.uint64</c> rounded once, a bool
        ///     1.0 / 0.0). An <c>np.complex128</c> converts its real part, as its <c>__int__</c> / <c>__float__</c> do
        ///     (NumPy warns <c>ComplexWarning</c>; <c>savetxt</c> never hands one over, it splits complex rows).
        ///     </para>
        ///     <para>
        ///     The <c>TypeError</c> names the scalar's type as CPython quotes it (<c>Py_TYPE(v)-&gt;tp_name</c>):
        ///     <c>numpy.bool</c>, <c>numpy.float64</c>, <c>numpy.float32</c>, <c>numpy.float16</c>,
        ///     <c>numpy.complex128</c>, <c>numpy.str_</c>, <c>numpy.bytes_</c>.
        ///     </para>
        /// </remarks>
        /// <exception cref="PrintfArgumentException">A conversion the scalar's type does not take: <c>%x format: an
        ///     integer is required, not numpy.float64</c>, <c>%c requires int or char</c>, <c>%d format: a real number is
        ///     required, not numpy.str_</c>, <c>must be real number, not numpy.bytes_</c>.</exception>
        /// <exception cref="ValueError"><c>cannot convert float NaN to integer</c>.</exception>
        /// <exception cref="OverflowException"><c>cannot convert float infinity to integer</c>; <c>%c arg not in
        ///     range(0x110000)</c>; <c>precision too large</c> (an integer conversion past <c>INT_MAX - 3</c>).</exception>
        /// <exception cref="OutOfMemoryException">A result longer than .NET's longest string.</exception>
        internal static string ApplyNumPyScalar(char conv, bool left, bool plus, bool space, bool zero, bool alt,
            int width, int precision, object value, NPTypeCode tc)
        {
            // np.str_ / np.bytes_ subclass str / bytes: the text rules, with the NumPy names in repr and errors.
            if (tc == NPTypeCode.String)
                return ApplyText(conv, left, width, precision, value, numpyScalar: true);

            switch (conv)
            {
                case 's':
                    return PadText(ArrayFormatter.ScalarStr(value, tc), width, precision, left);
                case 'r':
                    return PadText(NumPyScalarRepr(value, tc), width, precision, left);
                case 'a':
                    return PadText(PyRepr.Ascii(NumPyScalarRepr(value, tc)), width, precision, left);
                case 'd':
                case 'i':
                case 'u':
                    // mainformatlong: PyNumber_Long — the integer itself, a float truncated (NaN / infinity refused).
                    return FormatInteger(NumPyScalarToInteger(value, tc), 10, false, left, plus, space, zero, false, width,
                        precision);
                case 'x':
                case 'X':
                case 'o':
                    // _PyNumber_Index: only an integer scalar. np.bool_.__index__ raises in NumPy 2, and every float /
                    // complex scalar has none, so each is "not an integer" even when its value is integral.
                    if (!IsTrueIntegerTc(tc))
                        throw new PrintfArgumentException($"%{conv} format: an integer is required, not {NumPyTypeName(tc)}");
                    return FormatInteger(NumPyScalarToInteger(value, tc), conv == 'o' ? 8 : 16, conv == 'X',
                        left, plus, space, zero, alt, width, precision);
                case 'c':
                    // formatchar: PyLong_AsLongAndOverflow, which goes through __index__ (refused as above).
                    if (!IsTrueIntegerTc(tc))
                        throw new PrintfArgumentException("%c requires int or char");
                    return PadCodePoints(CodePointText(NumPyScalarToInteger(value, tc)), width, left);
                default:
                    // e E f F g G: formatfloat's PyFloat_AsDouble, which every numeric scalar answers.
                    return FormatReal(conv, NumPyScalarToDouble(value, tc), left, plus, space, zero, alt, width, precision);
            }
        }

        /// <summary>
        ///     The integral dtypes whose scalars have an <c>__index__</c> (<c>%x</c> / <c>%X</c> / <c>%o</c> /
        ///     <c>%c</c> take them): every integer dtype and NumSharp's <c>Char</c> (a <c>uint16</c>), but NOT
        ///     <c>Boolean</c> — <c>np.bool_.__index__</c> raises in NumPy 2.
        /// </summary>
        /// <param name="tc">The dtype.</param>
        /// <returns>True for an integer dtype.</returns>
        private static bool IsTrueIntegerTc(NPTypeCode tc)
        {
            switch (tc)
            {
                case NPTypeCode.SByte:
                case NPTypeCode.Byte:
                case NPTypeCode.Int16:
                case NPTypeCode.UInt16:
                case NPTypeCode.Int32:
                case NPTypeCode.UInt32:
                case NPTypeCode.Int64:
                case NPTypeCode.UInt64:
                case NPTypeCode.Char:
                    return true;
                default:
                    return false;
            }
        }

        /// <summary>The type name CPython's <c>TypeError</c> quotes for a NumPy scalar: <c>numpy.</c> and the dtype's
        ///     NumPy name (<c>numpy.bool</c>, <c>numpy.float32</c>; a <c>Decimal</c> is the <c>float64</c> it prints
        ///     as).</summary>
        /// <param name="tc">The dtype.</param>
        /// <returns>The name.</returns>
        private static string NumPyTypeName(NPTypeCode tc) => "numpy." + tc.AsNumpyDtypeName();

        /// <summary>
        ///     NumPy 2's <c>repr()</c> of a scalar: <c>np.True_</c> / <c>np.False_</c>, else <c>np.&lt;dtype&gt;(str)</c>
        ///     (<c>np.int64(-7)</c>, <c>np.float32(0.1)</c>, <c>np.float64(nan)</c>; an <c>np.complex128</c> without its
        ///     str's parentheses, <c>np.complex128(1+2j)</c>).
        /// </summary>
        /// <param name="value">The boxed scalar.</param>
        /// <param name="tc">Its dtype (not <see cref="NPTypeCode.String"/>, which <see cref="TextRepr"/> handles).</param>
        /// <returns>The repr.</returns>
        private static string NumPyScalarRepr(object value, NPTypeCode tc)
        {
            if (tc == NPTypeCode.Boolean)
                return (bool)value ? "np.True_" : "np.False_";
            string str = ArrayFormatter.ScalarStr(value, tc);
            if (tc == NPTypeCode.Complex && str.Length >= 2 && str[0] == '(' && str[^1] == ')')
                str = str.Substring(1, str.Length - 2);
            return "np." + tc.AsNumpyDtypeName() + "(" + str + ")";
        }

        /// <summary>
        ///     A NumPy scalar's <c>int()</c> (<c>PyNumber_Long</c>): an integer or a bool exactly, a float (a decimal
        ///     through its double, a complex through its real part) truncated toward zero.
        /// </summary>
        /// <param name="value">The boxed scalar.</param>
        /// <param name="tc">Its dtype.</param>
        /// <returns>The integer.</returns>
        /// <exception cref="ValueError"><c>cannot convert float NaN to integer</c>.</exception>
        /// <exception cref="OverflowException"><c>cannot convert float infinity to integer</c>.</exception>
        private static BigInteger NumPyScalarToInteger(object value, NPTypeCode tc)
        {
            switch (tc)
            {
                case NPTypeCode.Boolean: return (bool)value ? BigInteger.One : BigInteger.Zero;
                case NPTypeCode.SByte: return (sbyte)value;
                case NPTypeCode.Byte: return (byte)value;
                case NPTypeCode.Int16: return (short)value;
                case NPTypeCode.UInt16: return (ushort)value;
                case NPTypeCode.Int32: return (int)value;
                case NPTypeCode.UInt32: return (uint)value;
                case NPTypeCode.Int64: return (long)value;
                case NPTypeCode.UInt64: return (ulong)value;
                case NPTypeCode.Char: return (char)value;
                default:
                    return TruncateReal(NumPyScalarToDouble(value, tc));
            }
        }

        /// <summary>
        ///     A NumPy scalar's <c>float()</c>: a bool 1.0 / 0.0, an integer rounded once to the nearest double (an
        ///     <c>np.uint64</c> too: .NET's own conversion double-rounds at and above 2^63), a float widened exactly, a
        ///     decimal as <see cref="ArrayFormatter.ScalarStr"/> converts it, a complex's real part.
        /// </summary>
        /// <param name="value">The boxed scalar.</param>
        /// <param name="tc">Its dtype.</param>
        /// <returns>The double.</returns>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="tc"/> is not a numeric dtype.</exception>
        private static double NumPyScalarToDouble(object value, NPTypeCode tc)
        {
            switch (tc)
            {
                case NPTypeCode.Boolean: return (bool)value ? 1.0 : 0.0;
                case NPTypeCode.SByte: return (sbyte)value;
                case NPTypeCode.Byte: return (byte)value;
                case NPTypeCode.Int16: return (short)value;
                case NPTypeCode.UInt16: return (ushort)value;
                case NPTypeCode.Int32: return (int)value;
                case NPTypeCode.UInt32: return (uint)value;
                case NPTypeCode.Int64: return (long)value;
                case NPTypeCode.UInt64:
                {
                    ulong u = (ulong)value;
                    return u <= long.MaxValue ? (long)u : PythonInt.ToDouble(new BigInteger(u));
                }
                case NPTypeCode.Char: return (char)value;
                case NPTypeCode.Half: return (double)(Half)value;
                case NPTypeCode.Single: return (float)value;
                case NPTypeCode.Double: return (double)value;
                case NPTypeCode.Decimal: return (double)(decimal)value;
                case NPTypeCode.Complex: return ((Complex)value).Real;
                default: throw new ArgumentOutOfRangeException(nameof(tc), tc, "not a numeric dtype");
            }
        }

        /// <summary>A float's <c>int()</c>: truncated toward zero, exactly (as a Python int of any size).</summary>
        /// <param name="d">The value.</param>
        /// <returns>The integer.</returns>
        /// <exception cref="ValueError"><c>cannot convert float NaN to integer</c>.</exception>
        /// <exception cref="OverflowException"><c>cannot convert float infinity to integer</c>.</exception>
        private static BigInteger TruncateReal(double d)
        {
            if (double.IsNaN(d))
                throw new ValueError("cannot convert float NaN to integer");
            if (double.IsInfinity(d))
                throw new OverflowException("cannot convert float infinity to integer");
            return new BigInteger(Math.Truncate(d));
        }

        /// <summary>
        ///     <c>%c</c> of an integer: the one-character string of the code point (a lone surrogate code point is a valid
        ///     one-unit Python <c>str</c>, which <see cref="char.ConvertFromUtf32"/> would refuse).
        /// </summary>
        /// <param name="cp">The code point.</param>
        /// <returns>Its character.</returns>
        /// <exception cref="OverflowException"><c>%c arg not in range(0x110000)</c> for a negative value or one past
        ///     U+10FFFF (CPython's test also catches an overflow of the C <c>long</c>).</exception>
        private static string CodePointText(BigInteger cp)
        {
            if (cp.Sign < 0 || cp > 0x10FFFF)
                throw new OverflowException("%c arg not in range(0x110000)");
            int code = (int)cp;
            return code >= 0xD800 && code <= 0xDFFF ? ((char)code).ToString() : char.ConvertFromUtf32(code);
        }

        /// <summary>
        ///     A float conversion (<c>formatfloat</c> over <c>PyOS_double_to_string</c>): <c>e</c> / <c>E</c>, <c>f</c> /
        ///     <c>F</c>, <c>g</c> / <c>G</c> of a double, with CPython's default precision 6.
        /// </summary>
        /// <param name="conv">The conversion character.</param>
        /// <param name="d">The value.</param>
        /// <param name="left">The <c>-</c> flag.</param>
        /// <param name="plus">The <c>+</c> flag.</param>
        /// <param name="space">The <c> </c> flag.</param>
        /// <param name="zero">The <c>0</c> flag.</param>
        /// <param name="alt">The <c>#</c> flag.</param>
        /// <param name="width">The minimum width.</param>
        /// <param name="precision">The precision (-1 for the default 6).</param>
        /// <returns>The text.</returns>
        /// <exception cref="OutOfMemoryException">A result longer than .NET's longest string.</exception>
        private static string FormatReal(char conv, double d, bool left, bool plus, bool space, bool zero, bool alt,
            int width, int precision)
        {
            bool upper = conv == 'E' || conv == 'F' || conv == 'G';
            switch (conv)
            {
                case 'e':
                case 'E':
                    // PyOS_double_to_string asks dtoa for precision + 1 digits. At INT_MAX the sum wraps negative, and
                    // dtoa's mode 2 then makes ONE digit: exactly what precision 0 prints ('%.2147483647e' % 1.5 is
                    // '2e+00' in CPython 3.12, at once).
                    int pe = precision < 0 ? 6 : precision == int.MaxValue ? 0 : precision;
                    return FormatScientific(d, upper, pe, alt, left, plus, space, zero, width);
                case 'f':
                case 'F':
                    return FormatFixed(d, upper, precision < 0 ? 6 : precision, alt, left, plus, space, zero, width);
                default:
                    return FormatGeneral(d, upper, precision, alt, left, plus, space, zero, width);
            }
        }

        // ---- text values (fixed-width string elements) --------------------------------

        /// <summary>
        ///     CPython's <c>str.__mod__</c> conversion of ONE text value — a <c>U</c> element (a C# <see cref="string"/>)
        ///     or an <c>S</c> element (a <c>byte[]</c>): <c>%s</c> is <c>str(value)</c> (the text; a bytes value prints as
        ///     its <c>b'...'</c> repr), <c>%r</c> is <c>repr(value)</c>, <c>%a</c> is <c>ascii(value)</c>; a precision
        ///     truncates and a width pads with spaces, both in code points (the <c>-</c> flag left-aligns, the numeric
        ///     flags do nothing); <c>%c</c> takes a one-character <c>str</c>; every numeric conversion raises CPython's
        ///     <c>TypeError</c>.
        /// </summary>
        /// <param name="conv">The conversion character, one <see cref="PercentFormat"/> has checked is supported.</param>
        /// <param name="left">The <c>-</c> flag.</param>
        /// <param name="width">The minimum width in code points (0 for none).</param>
        /// <param name="precision">The precision (-1 for none): the maximum number of code points kept.</param>
        /// <param name="value">The element: a <see cref="string"/> or a <c>byte[]</c>.</param>
        /// <param name="numpyScalar">True when the value stands for a NumPy scalar (<c>np.str_</c> / <c>np.bytes_</c>:
        ///     <c>np.savetxt</c> formats <c>tuple(row)</c>), false for the Python object <c>getitem</c> returns
        ///     (<c>str</c> / <c>bytes</c>: <c>ndarray.tofile</c>). The two differ in <c>repr</c>
        ///     (<c>np.str_('ab')</c> vs <c>'ab'</c>) and in the type name a <c>TypeError</c> reports.</param>
        /// <returns>The formatted text.</returns>
        /// <exception cref="PrintfArgumentException">CPython's <c>TypeError</c> for a numeric conversion of text
        ///     (<c>%d format: a real number is required, not str</c>, <c>%x format: an integer is required, not
        ///     bytes</c>, <c>must be real number, not numpy.str_</c>) or <c>%c</c> of anything but one character
        ///     (<c>%c requires int or char</c>).</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="conv"/> is not a conversion character (the
        ///     caller checks it first).</exception>
        private static string ApplyText(char conv, bool left, int width, int precision, object value, bool numpyScalar)
        {
            bool isBytes = value is byte[];
            string typeName = isBytes ? (numpyScalar ? "numpy.bytes_" : "bytes") : (numpyScalar ? "numpy.str_" : "str");
            string body;
            switch (conv)
            {
                case 's':
                    // str(bytes) is its repr; str(np.bytes_) too. str(np.str_) / str(str) is the text itself.
                    body = isBytes ? PyRepr.Bytes((byte[])value) : (string)value;
                    break;
                case 'r':
                    body = TextRepr(value, numpyScalar);
                    break;
                case 'a':
                    body = PyRepr.Ascii(TextRepr(value, numpyScalar));
                    break;
                case 'c':
                    // A str of exactly one code point is its own %c; CPython refuses every other value with one text.
                    if (!isBytes && PyRepr.CodePointLength((string)value) == 1)
                        return PadCodePoints((string)value, width, left);
                    throw new PrintfArgumentException("%c requires int or char");
                case 'd':
                case 'i':
                case 'u':
                    throw new PrintfArgumentException($"%{conv} format: a real number is required, not {typeName}");
                case 'x':
                case 'X':
                case 'o':
                    throw new PrintfArgumentException($"%{conv} format: an integer is required, not {typeName}");
                case 'e':
                case 'E':
                case 'f':
                case 'F':
                case 'g':
                case 'G':
                    throw new PrintfArgumentException($"must be real number, not {typeName}");
                default:
                    throw new ArgumentOutOfRangeException(nameof(conv), conv, "not a % conversion character");
            }

            if (precision >= 0)
                body = TruncateCodePoints(body, precision);
            return PadCodePoints(body, width, left);
        }

        /// <summary>
        ///     Python's <c>repr()</c> of a text element: the plain <c>'ab'</c> / <c>b'ab'</c> of the Python object
        ///     (<c>tofile</c>'s <c>getitem</c>), or <c>np.str_('ab')</c> / <c>np.bytes_(b'ab')</c> of a NumPy scalar.
        /// </summary>
        /// <param name="value">A <see cref="string"/> or a <c>byte[]</c>.</param>
        /// <param name="numpyScalar">Whether the value stands for a NumPy scalar.</param>
        /// <returns>The representation.</returns>
        private static string TextRepr(object value, bool numpyScalar)
        {
            string repr = value is byte[] bytes ? PyRepr.Bytes(bytes) : PyRepr.Str((string)value);
            if (!numpyScalar)
                return repr;
            return (value is byte[] ? "np.bytes_(" : "np.str_(") + repr + ")";
        }

        /// <summary>The first <paramref name="count"/> code points of <paramref name="s"/> (Python's <c>s[:count]</c>;
        ///     a surrogate pair is one code point and is never split).</summary>
        /// <param name="s">The text.</param>
        /// <param name="count">The number of code points to keep.</param>
        /// <returns>The truncated text (the text itself when it is not longer).</returns>
        private static string TruncateCodePoints(string s, int count)
        {
            int units = 0;
            for (int kept = 0; kept < count && units < s.Length; kept++)
                units += char.IsHighSurrogate(s[units]) && units + 1 < s.Length && char.IsLowSurrogate(s[units + 1]) ? 2 : 1;
            return units >= s.Length ? s : s.Substring(0, units);
        }

        /// <summary>Pads <paramref name="body"/> with spaces to <paramref name="width"/> CODE POINTS (Python's width
        ///     counts characters, not UTF-16 units), on the right when <paramref name="left"/>.</summary>
        /// <param name="body">The text.</param>
        /// <param name="width">The minimum width.</param>
        /// <param name="left">Left-align (pad on the right).</param>
        /// <returns>The padded text.</returns>
        /// <exception cref="OutOfMemoryException">A width .NET cannot allocate.</exception>
        private static string PadCodePoints(string body, int width, bool left)
        {
            int deficit = width - PyRepr.CodePointLength(body);
            if (deficit <= 0)
                return body;
            return left ? body + new string(' ', deficit) : new string(' ', deficit) + body;
        }

        // ---- numeric helpers ---------------------------------------------------------

        /// <summary>
        ///     An integer conversion's text (<c>_PyUnicode_FormatLong</c> + <c>unicode_format_arg_output</c>): the digits
        ///     in base 10, 16 or 8, a precision as the MINIMUM number of digits, the <c>#</c> prefix, the sign, the width.
        /// </summary>
        /// <param name="big">The integer.</param>
        /// <param name="radix">10, 16 or 8.</param>
        /// <param name="upper">Upper-case hex digits and prefix (<c>%X</c>).</param>
        /// <param name="left">The <c>-</c> flag.</param>
        /// <param name="plus">The <c>+</c> flag.</param>
        /// <param name="space">The <c> </c> flag.</param>
        /// <param name="zero">The <c>0</c> flag.</param>
        /// <param name="alt">The <c>#</c> flag (a prefix for base 16 and 8 only).</param>
        /// <param name="width">The minimum width.</param>
        /// <param name="precision">The minimum number of digits (-1 for none).</param>
        /// <returns>The text.</returns>
        /// <exception cref="OverflowException"><c>precision too large</c>: CPython refuses a precision past
        ///     <c>INT_MAX - 3</c> before it computes a digit.</exception>
        /// <exception cref="ValueError">A decimal conversion of a Python int with more than 4300 digits (CPython's
        ///     <c>sys.get_int_max_str_digits()</c>; base 16 and 8 have no limit).</exception>
        /// <exception cref="OutOfMemoryException">A width or precision .NET cannot allocate.</exception>
        private static string FormatInteger(BigInteger big, int radix, bool upper,
            bool left, bool plus, bool space, bool zero, bool alt, int width, int precision)
        {
            if (precision > int.MaxValue - 3)
                throw new OverflowException("precision too large");

            bool neg = big.Sign < 0;
            BigInteger mag = BigInteger.Abs(big);

            string digits = radix switch
            {
                16 => ToBaseString(mag, 16, upper),
                8 => ToBaseString(mag, 8, upper),
                _ => NDTextFormat.FormatPythonInt(mag)
            };

            // Python's `.precision` on an integer conversion (d/i/u/x/X/o) is the MINIMUM number of digits:
            // the magnitude is left-zero-filled to `precision` chars ("%.5d" % 42 == "00042"). The digit run
            // for 0 is "0", so ".0" never shrinks it ("%.0d" % 0 == "0"). Unlike C, Python still honors the
            // '0' flag together with a precision ("%08.3d" % 7 == "00000007").
            if (precision >= 0 && digits.Length < precision)
                digits = new string('0', precision - digits.Length) + digits;

            string prefix = "";
            if (alt)
            {
                // Python's '#' flag prefixes even a zero value ("%#x" % 0 == "0x0", "%#o" % 0 == "0o0"),
                // unlike C printf which omits the prefix for zero.
                if (radix == 16) prefix = upper ? "0X" : "0x";
                else if (radix == 8) prefix = "0o";
            }

            string sign = neg ? "-" : Sign(false, plus, space);
            return PadInteger(sign, prefix, digits, width, left, zero);
        }

        // Width-pad an integer field. Zero-fill goes BETWEEN the alt prefix (0x/0o) and the digits, after the
        // sign — "%#08x" % 255 == "0x0000ff", "%08.3d" % -7 == "-0000007" — which the generic Pad (which would
        // place the zeros before the prefix) cannot express. Left-justify always space-pads on the right.
        private static string PadInteger(string sign, string prefix, string digits, int width, bool left, bool zero)
        {
            int len = sign.Length + prefix.Length + digits.Length;
            if (len >= width)
                return sign + prefix + digits;

            int deficit = width - len;
            if (left)
                return sign + prefix + digits + new string(' ', deficit);
            if (zero)
                return sign + prefix + new string('0', deficit) + digits;
            return new string(' ', deficit) + sign + prefix + digits;
        }

        private static string ToBaseString(BigInteger mag, int radix, bool upper)
        {
            if (mag.IsZero) return "0";
            const string lo = "0123456789abcdef";
            const string hi = "0123456789ABCDEF";
            string digitset = upper ? hi : lo;
            var sb = new StringBuilder();
            while (mag > 0)
            {
                sb.Insert(0, digitset[(int)(mag % radix)]);
                mag /= radix;
            }
            return sb.ToString();
        }

        /// <summary>
        ///     .NET's <c>F</c> / <c>E</c> format of a non-negative double: the exact decimal digits, rounded half to even
        ///     (the rounding of CPython's dtoa, verified on both target frameworks).
        /// </summary>
        /// <param name="abs">The value (non-negative, finite).</param>
        /// <param name="format">'F' or 'E'.</param>
        /// <param name="precision">The number of digits after the point.</param>
        /// <returns>The text.</returns>
        /// <exception cref="OutOfMemoryException">A precision past <see cref="MaxNetPrecision"/>: the result would be
        ///     longer than any .NET string (CPython builds such strings when memory allows).</exception>
        private static string FormatDigits(double abs, char format, int precision)
        {
            if (precision > MaxNetPrecision)
                throw new OutOfMemoryException();
            return abs.ToString(format == 'F' ? "F" + precision.ToString(CI) : "E" + precision.ToString(CI), CI);
        }

        /// <summary><c>%f</c> / <c>%F</c>: fixed notation with <paramref name="precision"/> digits after the point.</summary>
        /// <param name="d">The value.</param>
        /// <param name="upper"><c>%F</c> (upper-case <c>INF</c> / <c>NAN</c>).</param>
        /// <param name="precision">The number of fraction digits.</param>
        /// <param name="alt">The <c>#</c> flag: a point even with no fraction digits.</param>
        /// <param name="left">The <c>-</c> flag.</param>
        /// <param name="plus">The <c>+</c> flag.</param>
        /// <param name="space">The <c> </c> flag.</param>
        /// <param name="zero">The <c>0</c> flag.</param>
        /// <param name="width">The minimum width.</param>
        /// <returns>The text.</returns>
        /// <exception cref="OutOfMemoryException">A result longer than .NET's longest string.</exception>
        private static string FormatFixed(double d, bool upper, int precision, bool alt,
            bool left, bool plus, bool space, bool zero, int width)
        {
            if (!double.IsFinite(d))
                return NonFinite(d, upper, plus, space, width, left, zero);

            string sign = (d < 0 || IsNegZero(d)) ? "-" : Sign(false, plus, space);
            string digits = FormatDigits(Math.Abs(d), 'F', precision);
            // '#' forces a trailing decimal point even when no fractional digits are emitted ("%#.0f"%3=="3.").
            if (alt && digits.IndexOf('.') < 0) digits += ".";
            return Pad(digits, sign, width, left, zero);
        }

        /// <summary><c>%e</c> / <c>%E</c>: one digit, the point, <paramref name="precision"/> digits, a two-digit-minimum
        ///     exponent.</summary>
        /// <param name="d">The value.</param>
        /// <param name="upper"><c>%E</c>.</param>
        /// <param name="precision">The number of fraction digits.</param>
        /// <param name="alt">The <c>#</c> flag: a point even at precision 0.</param>
        /// <param name="left">The <c>-</c> flag.</param>
        /// <param name="plus">The <c>+</c> flag.</param>
        /// <param name="space">The <c> </c> flag.</param>
        /// <param name="zero">The <c>0</c> flag.</param>
        /// <param name="width">The minimum width.</param>
        /// <returns>The text.</returns>
        /// <exception cref="OutOfMemoryException">A result longer than .NET's longest string.</exception>
        private static string FormatScientific(double d, bool upper, int precision, bool alt,
            bool left, bool plus, bool space, bool zero, int width)
        {
            if (!double.IsFinite(d))
                return NonFinite(d, upper, plus, space, width, left, zero);

            string sign = (d < 0 || IsNegZero(d)) ? "-" : Sign(false, plus, space);
            string body = FormatDigits(Math.Abs(d), 'E', precision); // e.g. "1.500000E+000"
            int ei = body.IndexOf('E');
            string mant = body.Substring(0, ei);
            // '#' keeps the mantissa's decimal point even at precision 0 ("%#.0e"%3=="3.e+00").
            if (alt && mant.IndexOf('.') < 0) mant += ".";
            string digits = mant + ExponentSuffix(body.Substring(ei + 1), upper);
            return Pad(digits, sign, width, left, zero);
        }

        /// <summary><c>%g</c> / <c>%G</c>: <paramref name="precision"/> significant digits (6 by default, 0 means 1),
        ///     fixed notation when the exponent is in <c>[-4, precision)</c>, trailing zeros stripped unless <c>#</c>.</summary>
        /// <param name="d">The value.</param>
        /// <param name="upper"><c>%G</c>.</param>
        /// <param name="precision">The number of significant digits (-1 for the default).</param>
        /// <param name="alt">The <c>#</c> flag: keep the trailing zeros and the point.</param>
        /// <param name="left">The <c>-</c> flag.</param>
        /// <param name="plus">The <c>+</c> flag.</param>
        /// <param name="space">The <c> </c> flag.</param>
        /// <param name="zero">The <c>0</c> flag.</param>
        /// <param name="width">The minimum width.</param>
        /// <returns>The text.</returns>
        /// <exception cref="OutOfMemoryException">A result longer than .NET's longest string.</exception>
        private static string FormatGeneral(double d, bool upper, int precision, bool alt,
            bool left, bool plus, bool space, bool zero, int width)
        {
            if (!double.IsFinite(d))
                return NonFinite(d, upper, plus, space, width, left, zero);

            int P = precision < 0 ? 6 : (precision == 0 ? 1 : precision);
            // Without '#' the digits past a double's 767 significant ones are zeros that get stripped, and the exponent
            // test below is the same for any P past 309: a smaller P prints the same text, within .NET's format limit.
            if (!alt && P > MaxUsefulGeneralPrecision)
                P = MaxUsefulGeneralPrecision;
            string sign = (d < 0 || IsNegZero(d)) ? "-" : Sign(false, plus, space);
            double abs = Math.Abs(d);

            // decimal exponent X of the value written in %e style with P-1 fraction digits.
            string es = FormatDigits(abs, 'E', P - 1);
            int X = int.Parse(es.Substring(es.IndexOf('E') + 1), CI);

            string digits;
            if (X >= -4 && X < P)
            {
                digits = FormatDigits(abs, 'F', Math.Max(0, P - 1 - X));
                // '#' keeps trailing zeros AND forces a decimal point ("%#g"%100000=="100000.");
                // the default form strips both.
                if (!alt) digits = StripTrailingZeros(digits);
                else if (digits.IndexOf('.') < 0) digits += ".";
            }
            else
            {
                string body = es;
                int ei = body.IndexOf('E');
                string mant = body.Substring(0, ei);
                if (!alt) mant = StripTrailingZeros(mant);
                else if (mant.IndexOf('.') < 0) mant += ".";
                digits = mant + ExponentSuffix(body.Substring(ei + 1), upper);
            }

            return Pad(digits, sign, width, left, zero);
        }

        // ---- shared formatting primitives -------------------------------------------

        // .NET "E{p}" emits a 3-digit exponent with an uppercase 'E'; Python uses a 2-digit-minimum
        // exponent and matches the conversion's case. Rebuild the "±NN" tail.
        private static string ExponentSuffix(string netExpDigits, bool upper)
        {
            int exp = int.Parse(netExpDigits, CI);
            string es = Math.Abs(exp).ToString(CI);
            if (es.Length < 2) es = es.PadLeft(2, '0');
            return (upper ? "E" : "e") + (exp < 0 ? "-" : "+") + es;
        }

        private static string StripTrailingZeros(string s)
        {
            if (s.IndexOf('.') < 0) return s;
            s = s.TrimEnd('0');
            if (s.EndsWith(".")) s = s.Substring(0, s.Length - 1);
            return s;
        }

        private static string NonFinite(double d, bool upper, bool plus, bool space,
            int width, bool left, bool zero)
        {
            string sign = double.IsNaN(d) ? Sign(false, plus, space)
                                          : (d < 0 ? "-" : Sign(false, plus, space));
            string t = double.IsNaN(d) ? "nan" : "inf";
            if (upper) t = t.ToUpperInvariant();
            return Pad(t, sign, width, left, zero);
        }

        private static string Sign(bool neg, bool plus, bool space)
        {
            if (neg) return "-";
            if (plus) return "+";
            if (space) return " ";
            return "";
        }

        private static bool IsNegZero(double d) => d == 0.0 && double.IsNegative(d);

        // Combine sign + body into a width-padded field. Numeric zero-padding inserts the zeros
        // between the sign and the body (Python's behavior, including for inf/nan). Left-justify
        // ('-') always pads with spaces on the right and wins over the zero flag.
        private static string Pad(string body, string sign, int width, bool left, bool zero)
        {
            string full = sign + body;
            if (full.Length >= width)
                return full;

            int deficit = width - full.Length;
            if (left)
                return full + new string(' ', deficit);
            if (zero)
                return sign + new string('0', deficit) + body;
            return new string(' ', deficit) + full;
        }
    }
}
