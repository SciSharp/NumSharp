using System;
using System.Numerics;
using NumSharp.Backends.Strings;
using NumSharp.Utilities;

namespace NumSharp.Backends.Printing
{
    internal static partial class PrintfFormatter
    {
        /// <summary>
        ///     CPython's <c>format % (value,)</c> for ONE Python object (NumPy's <c>PyUnicode_Format(format, (obj,))</c>,
        ///     the call <c>ndarray.tofile</c> makes for every element <c>getitem</c> returned), compiled and run once.
        /// </summary>
        /// <param name="format">The Python <c>%</c> format string.</param>
        /// <param name="value">The Python object under the C# boundary map (<see cref="NDPyObject"/>): a
        ///     <see cref="string"/>, <c>null</c> for <c>None</c>, a double NaN, an int, …</param>
        /// <returns>The formatted text.</returns>
        /// <remarks>
        ///     A format with more than one conversion spec raises <c>not enough arguments for format string</c>, one with
        ///     none <c>not all arguments converted during string formatting</c>, as for any one-item tuple. A writer that
        ///     formats many values compiles the format once instead (<see cref="Compile"/>).
        /// </remarks>
        /// <exception cref="ArgumentNullException"><paramref name="format"/> is null.</exception>
        /// <exception cref="PrintfArgumentException">CPython's <c>TypeError</c> (the argument count, or a conversion the
        ///     object's type does not support: <c>%d format: a real number is required, not str</c>).</exception>
        /// <exception cref="ValueError"><c>cannot convert float NaN to integer</c>; a malformed format.</exception>
        /// <exception cref="OverflowException">An infinity under an integer conversion, a <c>%c</c> argument outside
        ///     <c>range(0x110000)</c>, or an int too large for a float conversion.</exception>
        /// <exception cref="OutOfMemoryException">A result longer than .NET's longest string.</exception>
        public static string FormatPyObject(string format, object value) => Format(format, value, PercentFormat.PyObject);

        /// <summary>
        ///     One CPython <c>str.__mod__</c> conversion of a Python OBJECT (the C# boundary map of <see cref="NDPyObject"/>):
        ///     the conversions of <c>unicode_format_arg_format</c> for <c>None</c>, <c>bool</c> / <c>int</c> (any C# integer,
        ///     <see cref="BigInteger"/>), <c>float</c> (<see cref="double"/> / <see cref="float"/>; NumSharp's
        ///     <see cref="decimal"/> as the float it prints like), <c>complex</c>, <c>np.float16</c> (<see cref="Half"/>),
        ///     <c>str</c> (<see cref="string"/> / <see cref="char"/>) and <c>bytes</c> (<c>byte[]</c>), and the containers.
        /// </summary>
        /// <param name="conv">The conversion character, one <see cref="PercentFormat"/> has checked is supported.</param>
        /// <param name="left">The <c>-</c> flag (or a negative <c>*</c> width).</param>
        /// <param name="plus">The <c>+</c> flag.</param>
        /// <param name="space">The <c> </c> flag.</param>
        /// <param name="zero">The <c>0</c> flag.</param>
        /// <param name="alt">The <c>#</c> flag.</param>
        /// <param name="width">The minimum width (0 for none).</param>
        /// <param name="precision">The precision (-1 for none).</param>
        /// <param name="value">The object.</param>
        /// <returns>The converted text.</returns>
        /// <remarks>
        ///     <para>
        ///     CPython's rules, probed on 3.12 for every kind here: <c>%s</c> / <c>%r</c> / <c>%a</c> are <c>str()</c> /
        ///     <c>repr()</c> / <c>ascii()</c>, cut to the precision and space-padded to the width in code points (the numeric
        ///     flags do nothing). <c>%d</c> / <c>%i</c> / <c>%u</c> take any real number (<c>PyNumber_Long</c>: a float
        ///     truncates toward zero, <c>True</c> is 1); <c>%x</c> / <c>%X</c> / <c>%o</c> only an integer
        ///     (<c>__index__</c>); <c>%e</c> / <c>%f</c> / <c>%g</c> any real number (<c>PyFloat_AsDouble</c>); <c>%c</c> an
        ///     integer code point or a one-character <c>str</c>. Everything else raises the <c>TypeError</c> naming the
        ///     object's Python type (<c>not NoneType</c>, <c>not complex</c>, <c>not numpy.float16</c>).
        ///     </para>
        ///     <para>
        ///     A Python <c>bool</c> is an <c>int</c> here (<c>'%x' % True</c> is <c>'1'</c>), unlike a NumPy <c>bool</c>
        ///     scalar, which <see cref="FormatRowInto"/>'s NumPy-scalar mode refuses for <c>%x</c> / <c>%c</c>.
        ///     </para>
        /// </remarks>
        /// <exception cref="PrintfArgumentException">The conversion does not apply to the object's type.</exception>
        /// <exception cref="ValueError">A NaN under an integer conversion; a decimal conversion of an int with more than
        ///     4300 digits (and its <c>%s</c> / <c>%r</c>).</exception>
        /// <exception cref="OverflowException">An infinity under an integer conversion; a <c>%c</c> code point outside
        ///     <c>range(0x110000)</c>; an int whose float conversion overflows (<c>int too large to convert to float</c>);
        ///     <c>precision too large</c>.</exception>
        /// <exception cref="OutOfMemoryException">A result longer than .NET's longest string.</exception>
        internal static string ApplyPyObject(char conv, bool left, bool plus, bool space, bool zero, bool alt,
            int width, int precision, object value)
        {
            // A str / bytes object: the text rules of ApplyText, with the Python type names (getitem's objects are plain
            // str / bytes, never NumPy scalars). A C# char is a one-character str under the boundary map.
            if (value is string || value is byte[])
                return ApplyText(conv, left, width, precision, value, numpyScalar: false);
            if (value is char ch)
                return ApplyText(conv, left, width, precision, ch.ToString(), numpyScalar: false);

            switch (conv)
            {
                case 's':
                    return PadText(NDPyObject.Str(value), width, precision, left);
                case 'r':
                    return PadText(NDPyObject.Repr(value), width, precision, left);
                case 'a':
                    return PadText(PyRepr.Ascii(NDPyObject.Repr(value)), width, precision, left);
                case 'c':
                    // %c takes an int code point (a bool is an int) in range(0x110000).
                    if (!TryPyInt(value, out BigInteger cp))
                        throw new PrintfArgumentException("%c requires int or char");
                    return PadCodePoints(CodePointText(cp), width, left);
                case 'd':
                case 'i':
                case 'u':
                {
                    // PyNumber_Long: an int as itself, a real number truncated toward zero (NaN / infinity refused the way
                    // int() refuses them), anything else is "not a real number".
                    if (!TryPyInt(value, out BigInteger big))
                    {
                        if (!TryPyReal(value, out double d))
                            throw new PrintfArgumentException($"%{conv} format: a real number is required, not {PyTypeName(value)}");
                        big = TruncateReal(d);
                    }
                    return FormatInteger(big, 10, false, left, plus, space, zero, false, width, precision);
                }
                case 'x':
                case 'X':
                case 'o':
                {
                    // _PyNumber_Index: only an integer (a bool included); a float is "not an integer" even when integral.
                    if (!TryPyInt(value, out BigInteger big))
                        throw new PrintfArgumentException($"%{conv} format: an integer is required, not {PyTypeName(value)}");
                    return FormatInteger(big, conv == 'o' ? 8 : 16, conv == 'X', left, plus, space, zero, alt, width,
                        precision);
                }
                case 'e':
                case 'E':
                case 'f':
                case 'F':
                case 'g':
                case 'G':
                {
                    // PyFloat_AsDouble: an int is converted (correctly rounded; OverflowError past the float range), a real
                    // number is itself, anything else "must be real number".
                    double d;
                    if (TryPyInt(value, out BigInteger big))
                    {
                        d = NumSharp.PythonInt.ToDouble(big);
                        if (double.IsInfinity(d))
                            throw new OverflowException("int too large to convert to float");
                    }
                    else if (!TryPyReal(value, out d))
                        throw new PrintfArgumentException($"must be real number, not {PyTypeName(value)}");
                    return FormatReal(conv, d, left, plus, space, zero, alt, width, precision);
                }
                default:
                    throw new ArgumentOutOfRangeException(nameof(conv), conv, "not a % conversion character");
            }
        }

        /// <summary>A text conversion's precision cut and width padding, both in code points (<c>'%-5.1s' % None</c> is
        ///     <c>'N    '</c>).</summary>
        /// <param name="body">The <c>str()</c> / <c>repr()</c> text.</param>
        /// <param name="width">The minimum width.</param>
        /// <param name="precision">The maximum number of code points kept (-1 for all).</param>
        /// <param name="left">Left-align.</param>
        /// <returns>The field text.</returns>
        private static string PadText(string body, int width, int precision, bool left)
        {
            if (precision >= 0)
                body = TruncateCodePoints(body, precision);
            return PadCodePoints(body, width, left);
        }

        /// <summary>The value as a Python <c>int</c>: any C# integer, a <see cref="BigInteger"/>, or a <see cref="bool"/>
        ///     (Python's <c>bool</c> is an <c>int</c> subclass). This is CPython's <c>PyLong_Check</c> under the boundary
        ///     map, which a <c>*</c> width or precision also asks.</summary>
        /// <param name="value">The object.</param>
        /// <param name="big">The integer value.</param>
        /// <returns>True for an integer object.</returns>
        internal static bool TryPyInt(object value, out BigInteger big)
        {
            switch (value)
            {
                case bool b: big = b ? BigInteger.One : BigInteger.Zero; return true;
                case BigInteger bi: big = bi; return true;
                case sbyte v: big = v; return true;
                case byte v: big = v; return true;
                case short v: big = v; return true;
                case ushort v: big = v; return true;
                case int v: big = v; return true;
                case uint v: big = v; return true;
                case long v: big = v; return true;
                case ulong v: big = v; return true;
                default: big = BigInteger.Zero; return false;
            }
        }

        /// <summary>The value as a real number that is not an int: a Python <c>float</c> (<see cref="double"/>,
        ///     <see cref="float"/>; NumSharp's <see cref="decimal"/> converts to one), or an <c>np.float16</c>
        ///     (<see cref="Half"/>, whose <c>__int__</c> / <c>__float__</c> CPython's integer and float conversions call).</summary>
        /// <param name="value">The object.</param>
        /// <param name="d">The value as a double.</param>
        /// <returns>True for a real number that is not an int.</returns>
        private static bool TryPyReal(object value, out double d)
        {
            switch (value)
            {
                case double v: d = v; return true;
                case float v: d = v; return true;
                case Half v: d = (double)v; return true;
                case decimal v: d = Converts.ToDouble(v); return true;
                default: d = 0; return false;
            }
        }

        /// <summary>
        ///     The Python type name a CPython <c>TypeError</c> of <c>%</c> reports for an object (<c>Py_TYPE(v)-&gt;tp_name</c>)
        ///     under the boundary map: <c>NoneType</c>, <c>complex</c>, <c>numpy.float16</c>, <c>tuple</c>, <c>list</c>,
        ///     <c>dict</c>, <c>numpy.ndarray</c>; an opaque C# object reports its .NET type name.
        /// </summary>
        /// <param name="value">An object that is not an int, a real number, a str or bytes.</param>
        /// <returns>The type name (<see cref="NDPyObject.TypeName"/>, the one mapping every string-module message uses).</returns>
        private static string PyTypeName(object value) => NDPyObject.TypeName(value);
    }
}
