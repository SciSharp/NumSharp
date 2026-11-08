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
            // An ndarray argument ('%d' % a, the right operand of str % chararray): CPython reaches it through the
            // ndarray's number slots, whose rules differ from a scalar's (see ApplyArray).
            if (value is NDArray array)
                return ApplyArray(conv, left, plus, space, zero, alt, width, precision, array);

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

        /// <summary>
        ///     One <c>str.__mod__</c> conversion of an ndarray ARGUMENT, through the slots CPython calls on it: <c>%s</c> /
        ///     <c>%r</c> / <c>%a</c> are <c>str()</c> / <c>repr()</c> / <c>ascii()</c> (NumPy's printing); <c>%d</c> /
        ///     <c>%i</c> / <c>%u</c> are <c>int(a)</c>; <c>%x</c> / <c>%X</c> / <c>%o</c> and <c>%c</c> its <c>__index__</c>;
        ///     the float conversions <c>float(a)</c>.
        /// </summary>
        /// <param name="conv">The conversion character.</param>
        /// <param name="left">The <c>-</c> flag.</param>
        /// <param name="plus">The <c>+</c> flag.</param>
        /// <param name="space">The <c> </c> flag.</param>
        /// <param name="zero">The <c>0</c> flag.</param>
        /// <param name="alt">The <c>#</c> flag.</param>
        /// <param name="width">The minimum width.</param>
        /// <param name="precision">The precision (-1 for none).</param>
        /// <param name="array">The argument.</param>
        /// <returns>The converted text.</returns>
        /// <exception cref="PrintfArgumentException">CPython's TypeErrors: <c>%d format: a real number is required, not
        ///     numpy.ndarray</c> (any array but a 0-d one whose element <c>int()</c> takes — NumPy 2.4's <c>array_int</c>
        ///     refuses an array of one or more dimensions, and CPython reports every TypeError of <c>int(a)</c> with its own
        ///     text), <c>%x format: an integer is required, not chararray</c> (only a 0-d INTEGER array has an
        ///     <c>__index__</c>), <c>%c requires int or char</c>, and the float conversions' errors as <c>float(a)</c> raises
        ///     them (<c>only 0-dimensional arrays can be converted to Python scalars</c>, <c>float() argument must be a string
        ///     or a real number, not 'complex'</c>: CPython's <c>formatfloat</c> passes them through).</exception>
        /// <exception cref="ValueError"><c>int()</c> / <c>float()</c> of a 0-d text element that spells no number
        ///     (<c>invalid literal for int() with base 10: 'x'</c>, <c>could not convert string to float: b'x'</c>); a NaN
        ///     under an integer conversion.</exception>
        /// <exception cref="OverflowException">An infinity under an integer conversion; a <c>%c</c> code point outside
        ///     <c>range(0x110000)</c>.</exception>
        /// <remarks>
        ///     The element of a 0-d array is NumPy's <c>getitem</c> object (<c>PyArray_GETITEM</c>, not the NumPy scalar): a
        ///     Python int / float / complex / bool for a number, a Python str for <c>U</c> / <c>StringDType</c>, bytes for
        ///     <c>S</c> — so <c>'%d' % np.array(' 7 ')</c> is <c>'7'</c> (<c>int()</c> strips whitespace) and the errors quote
        ///     <c>'x'</c>, not <c>np.str_('x')</c>.
        /// </remarks>
        private static string ApplyArray(char conv, bool left, bool plus, bool space, bool zero, bool alt,
            int width, int precision, NDArray array)
        {
            switch (conv)
            {
                case 's':
                    return PadText(NDPyObject.Str(array), width, precision, left);
                case 'r':
                    return PadText(NDPyObject.Repr(array), width, precision, left);
                case 'a':
                    return PadText(PyRepr.Ascii(NDPyObject.Repr(array)), width, precision, left);
                case 'c':
                    // formatchar: PyLong_AsLongAndOverflow, i.e. __index__; its TypeError is "%c requires int or char".
                    if (!TryArrayIndex(array, out BigInteger cp))
                        throw new PrintfArgumentException("%c requires int or char");
                    return PadCodePoints(CodePointText(cp), width, left);
                case 'd':
                case 'i':
                case 'u':
                {
                    // mainformatlong: PyNumber_Long(a); a TypeError (an array of one or more dimensions, a complex or None
                    // element) becomes "a real number is required", a ValueError / OverflowError passes through.
                    if (!TryArrayInt(array, out BigInteger big))
                        throw new PrintfArgumentException($"%{conv} format: a real number is required, not {PyTypeName(array)}");
                    return FormatInteger(big, 10, false, left, plus, space, zero, false, width, precision);
                }
                case 'x':
                case 'X':
                case 'o':
                {
                    if (!TryArrayIndex(array, out BigInteger big))
                        throw new PrintfArgumentException($"%{conv} format: an integer is required, not {PyTypeName(array)}");
                    return FormatInteger(big, conv == 'o' ? 8 : 16, conv == 'X', left, plus, space, zero, alt, width,
                        precision);
                }
                case 'e':
                case 'E':
                case 'f':
                case 'F':
                case 'g':
                case 'G':
                    return FormatReal(conv, ArrayFloat(array), left, plus, space, zero, alt, width, precision);
                default:
                    throw new ArgumentOutOfRangeException(nameof(conv), conv, "not a % conversion character");
            }
        }

        /// <summary>
        ///     NumPy's <c>array_index</c> (<c>a.__index__()</c>): the value of a 0-d array of an INTEGER dtype (NumSharp's
        ///     <c>Char</c>, a <c>uint16</c>, included; bool is not an integer there).
        /// </summary>
        /// <param name="array">The array.</param>
        /// <param name="value">Receives the integer.</param>
        /// <returns>False where NumPy raises its TypeError (<c>only integer scalar arrays can be converted to a scalar
        ///     index</c>): another dtype, or an array of one or more dimensions.</returns>
        internal static bool TryArrayIndex(NDArray array, out BigInteger value)
        {
            value = BigInteger.Zero;
            if (array.ndim != 0)
                return false;
            object item = array.Storage.GetAtIndex(0);
            switch (item)
            {
                case bool:
                    return false;
                case char c:
                    value = c;
                    return true;
                default:
                    return TryPyInt(item, out value);
            }
        }

        /// <summary>
        ///     NumPy 2.4's <c>array_int</c> (<c>int(a)</c>) as CPython's <c>%d</c> sees it: a 0-d array's element through
        ///     <c>int()</c> — a number truncated toward zero, text parsed as a Python int literal (whitespace, underscores and
        ///     Unicode digits as <c>int()</c> takes them).
        /// </summary>
        /// <param name="array">The array.</param>
        /// <param name="value">Receives the integer.</param>
        /// <returns>False where <c>int(a)</c> raises a TypeError: an array of one or more dimensions (<c>only 0-dimensional
        ///     arrays can be converted to Python scalars</c>), a complex element, a <c>StringDType</c> NA object that is not a
        ///     number or text.</returns>
        /// <exception cref="ValueError">A NaN element; text that is no int literal (<c>invalid literal for int() with base
        ///     10: 'x'</c>, quoting the Python str / bytes); a literal of more than 4300 digits.</exception>
        /// <exception cref="OverflowException">An infinite element.</exception>
        private static bool TryArrayInt(NDArray array, out BigInteger value)
        {
            value = BigInteger.Zero;
            if (array.ndim != 0)
                return false;
            object item = ArrayItem(array);
            switch (item)
            {
                case string s:
                    value = NumSharp.PythonInt.Parse(s, 10);
                    return true;
                case byte[] b:
                    value = NumSharp.PythonInt.ParseBytes(b, 10);
                    return true;
            }
            if (TryPyInt(item, out value))
                return true;
            if (TryPyReal(item, out double d))
            {
                value = TruncateReal(d);
                return true;
            }
            return false;
        }

        /// <summary>
        ///     NumPy 2.4's <c>array_float</c> (<c>float(a)</c>): a 0-d array's element through <c>float()</c> — a number as a
        ///     double, text parsed as a Python float literal.
        /// </summary>
        /// <param name="array">The array.</param>
        /// <returns>The value.</returns>
        /// <exception cref="PrintfArgumentException">NumPy's and Python's TypeErrors, which CPython's float conversions pass
        ///     through: <c>only 0-dimensional arrays can be converted to Python scalars</c>, <c>float() argument must be a
        ///     string or a real number, not 'complex'</c> (or <c>'NoneType'</c> for a <c>StringDType</c> NA None).</exception>
        /// <exception cref="ValueError">Text that is no float literal (<c>could not convert string to float: 'x'</c>, quoting
        ///     the Python str / bytes).</exception>
        private static double ArrayFloat(NDArray array)
        {
            if (array.ndim != 0)
                throw new PrintfArgumentException("only 0-dimensional arrays can be converted to Python scalars");
            object item = ArrayItem(array);
            if (TryItemFloat(item, out double d))
                return d;
            switch (item)
            {
                case string s:
                    return ParseFloatObject(s, null);
                case byte[] b:
                    return ParseFloatObject(null, b);
                default:
                    throw new PrintfArgumentException(
                        $"float() argument must be a string or a real number, not '{NDPyObject.TypeName(item)}'");
            }
        }

        /// <summary>Whether <c>float(a)</c> succeeds for an ndarray (bytes formatting's float conversions turn every failure
        ///     into one TypeError, so they only ask).</summary>
        /// <param name="array">The array.</param>
        /// <param name="value">Receives the value.</param>
        /// <returns>True when <see cref="ArrayFloat"/> would return.</returns>
        internal static bool TryArrayFloat(NDArray array, out double value)
        {
            value = 0;
            if (array.ndim != 0)
                return false;
            object item = ArrayItem(array);
            if (TryItemFloat(item, out value))
                return true;
            try
            {
                switch (item)
                {
                    case string s:
                        value = ParseFloatObject(s, null);
                        return true;
                    case byte[] b:
                        value = ParseFloatObject(null, b);
                        return true;
                    default:
                        return false;
                }
            }
            catch (ValueError)
            {
                // float('x'): bytes formatting reports it as its own "float argument required" TypeError.
                return false;
            }
        }

        /// <summary>A getitem object's float value when it is a number: a Python int (correctly rounded), bool, float, an
        ///     np.float16 or a decimal.</summary>
        /// <param name="item">The element object.</param>
        /// <param name="value">Receives the value.</param>
        /// <returns>True for a real number.</returns>
        /// <exception cref="OverflowException">A Python int too large for a float (<c>int too large to convert to
        ///     float</c>).</exception>
        private static bool TryItemFloat(object item, out double value)
        {
            if (TryPyInt(item, out BigInteger big))
            {
                value = NumSharp.PythonInt.ToDouble(big);
                if (double.IsInfinity(value))
                    throw new OverflowException("int too large to convert to float");
                return true;
            }
            return TryPyReal(item, out value);
        }

        /// <summary>Python's <c>float()</c> of a str or bytes object (<c>PyFloat_FromString</c>: whitespace, underscores, the
        ///     special values; its ValueError quotes the object).</summary>
        /// <param name="s">The str, or null.</param>
        /// <param name="b">The bytes, when <paramref name="s"/> is null.</param>
        /// <returns>The value.</returns>
        /// <exception cref="ValueError"><c>could not convert string to float: 'x'</c> / <c>b'x'</c>.</exception>
        private static unsafe double ParseFloatObject(string s, byte[] b)
        {
            // NDTextParse writes the target's canonical value (16 bytes; a double at the start for float64).
            byte* canon = stackalloc byte[16];
            if (s is not null)
                NDTextParse.ParseObject(s, NPTypeCode.Double, canon);
            else
                NDTextParse.ParseObject(b, NPTypeCode.Double, canon);
            return *(double*)canon;
        }

        /// <summary>
        ///     NumPy's <c>PyArray_GETITEM</c> of a 0-d array: the element as the Python object <c>getitem</c> returns — a C#
        ///     number for a number (NumSharp's <c>Char</c>, a <c>uint16</c>, as an int), a string for <c>U</c> /
        ///     <c>StringDType</c> (a <c>StringDType</c> NA as its NA object), a <c>byte[]</c> for <c>S</c>.
        /// </summary>
        /// <param name="array">A 0-d array.</param>
        /// <returns>The element.</returns>
        private static object ArrayItem(NDArray array)
        {
            object item = array.Storage.GetAtIndex(0);
            return item is char c ? (object)(int)c : item;
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
