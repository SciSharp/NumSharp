using System;
using System.Collections.Generic;
using System.Numerics;
using System.Text;

namespace NumSharp.Backends.Printing
{
    /// <summary>
    ///     A Python <c>%</c> format string compiled once: the parse half of CPython 3.12's <c>PyUnicode_Format</c>
    ///     (<c>unicode_format_arg_parse</c> in <c>Objects/unicodeobject.c</c>) done before any argument is seen, so
    ///     <c>np.savetxt</c> (<c>format % tuple(row)</c>, once per row) and <c>ndarray.tofile</c>
    ///     (<c>format % (item,)</c>, once per element) run it without parsing the format again each time.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     CPython parses AS it formats, so which error a bad format raises depends on the arguments: in
    ///     <c>'%d %y' % (1,)</c> the first spec converts the <c>1</c> and the second finds no argument left
    ///     (<c>not enough arguments for format string</c>) before <c>%y</c> is found unsupported. A compiled program keeps
    ///     that order. A spec records where its parse stopped (a mapping key, a width or precision too big, the end of the
    ///     string), and a run raises the error when it reaches that spec, after taking the arguments CPython takes before
    ///     it (a <c>*</c> width, a <c>*</c> precision). Compiling stops at the first parse error, as CPython's parse does.
    ///     </para>
    ///     <para>
    ///     The arguments are always a tuple, never a mapping: NumPy passes <c>tuple(row)</c> or <c>(item,)</c>. So a
    ///     <c>%(key)</c> spec is CPython's <c>TypeError: format requires a mapping</c>, and an argument left over at the
    ///     end is <c>TypeError: not all arguments converted during string formatting</c>.
    ///     </para>
    ///     <para>
    ///     Each argument comes with its KIND: <see cref="PyObject"/> for a Python object under the boundary map of
    ///     <see cref="Strings.NDPyObject"/> (what <c>getitem</c> returns: <c>tofile</c>'s elements, a <c>StringDType</c>
    ///     array's rows), else the dtype of the NumPy scalar the value stands for (<c>savetxt</c>'s <c>tuple(row)</c> of a
    ///     numeric, <c>U</c> or <c>S</c> array). The two differ wherever Python's <c>%</c> asks for the object's type: a
    ///     Python <c>bool</c> is an <c>int</c> (<c>'%x' % True</c> is <c>'1'</c>) and a NumPy <c>bool</c> is not; a
    ///     Python <c>int</c> can be a <c>*</c> width and no NumPy scalar can; a NumPy scalar's <c>repr</c> names its type
    ///     (<c>np.float64(1.5)</c>).
    ///     </para>
    ///     <para>
    ///     An instance is immutable and safe to share between threads; a run keeps its state on the stack.
    ///     </para>
    /// </remarks>
    internal sealed class PercentFormat
    {
        /// <summary>
        ///     The kind of an argument that is a Python object (<c>getitem</c>'s result) rather than a NumPy scalar. It is
        ///     <see cref="NPTypeCode.Empty"/>, which no NumPy scalar can be.
        /// </summary>
        internal const NPTypeCode PyObject = NPTypeCode.Empty;

        /// <summary>What one compiled step does.</summary>
        private enum OpKind : byte
        {
            /// <summary>Copies a run of literal text.</summary>
            Text,

            /// <summary>Writes the <c>%</c> of a <c>%%</c> (no argument).</summary>
            Percent,

            /// <summary>Converts one argument (a conversion spec), or raises the parse error the spec ends with.</summary>
            Spec,
        }

        /// <summary>The parse error a spec ends with: CPython raises it when its parse reaches that point.</summary>
        private enum ParseFailure : byte
        {
            /// <summary>The spec parsed completely.</summary>
            None,

            /// <summary><c>%(</c>: <c>TypeError: format requires a mapping</c> (the arguments are a tuple).</summary>
            Mapping,

            /// <summary>The width's digits exceed <c>PY_SSIZE_T_MAX</c>: <c>ValueError: width too big</c>.</summary>
            WidthTooBig,

            /// <summary>The precision's digits exceed <c>INT_MAX</c>: <c>ValueError: precision too big</c>.</summary>
            PrecisionTooBig,

            /// <summary>The string ended before the conversion character: <c>ValueError: incomplete format</c>.</summary>
            Incomplete,
        }

        /// <summary>One step of a compiled format.</summary>
        private readonly struct Op
        {
            /// <summary>What the step does.</summary>
            internal readonly OpKind Kind;

            /// <summary>A text step's first UTF-16 index in the format.</summary>
            internal readonly int Start;

            /// <summary>A text step's length in UTF-16 units.</summary>
            internal readonly int Length;

            /// <summary>The <c>-</c> flag (left-justify).</summary>
            internal readonly bool Left;

            /// <summary>The <c>+</c> flag (a sign on every number).</summary>
            internal readonly bool Plus;

            /// <summary>The <c> </c> flag (a space before a positive number).</summary>
            internal readonly bool Space;

            /// <summary>The <c>0</c> flag (zero padding for numbers).</summary>
            internal readonly bool Zero;

            /// <summary>The <c>#</c> flag (the alternate form).</summary>
            internal readonly bool Alt;

            /// <summary>The width is a <c>*</c>: the next argument, which must be a Python <c>int</c>.</summary>
            internal readonly bool WidthStar;

            /// <summary>The precision is a <c>*</c>: the next argument after the width, a Python <c>int</c>.</summary>
            internal readonly bool PrecisionStar;

            /// <summary>The written width, or -1 for none (also when it is a <c>*</c>).</summary>
            internal readonly long Width;

            /// <summary>The written precision (0 for a bare <c>.</c>), or -1 for none (also when it is a <c>*</c>).</summary>
            internal readonly int Precision;

            /// <summary>The conversion character as a code point (CPython reads code points, so a supplementary character
            ///     is one), or -1 when the spec ends in a parse error first.</summary>
            internal readonly int Conversion;

            /// <summary>The conversion character's index in CODE POINTS, which the unsupported-character error
            ///     reports.</summary>
            internal readonly int ConversionIndex;

            /// <summary>The parse error the spec ends with.</summary>
            internal readonly ParseFailure Failure;

            /// <summary>Builds a step; see the fields.</summary>
            /// <param name="kind">What the step does.</param>
            /// <param name="start">A text step's first index.</param>
            /// <param name="length">A text step's length.</param>
            /// <param name="left">The <c>-</c> flag.</param>
            /// <param name="plus">The <c>+</c> flag.</param>
            /// <param name="space">The <c> </c> flag.</param>
            /// <param name="zero">The <c>0</c> flag.</param>
            /// <param name="alt">The <c>#</c> flag.</param>
            /// <param name="widthStar">A <c>*</c> width.</param>
            /// <param name="width">The written width (-1 for none).</param>
            /// <param name="precisionStar">A <c>*</c> precision.</param>
            /// <param name="precision">The written precision (-1 for none).</param>
            /// <param name="conversion">The conversion code point (-1 for none).</param>
            /// <param name="conversionIndex">Its code-point index.</param>
            /// <param name="failure">The parse error the spec ends with.</param>
            internal Op(OpKind kind, int start, int length, bool left, bool plus, bool space, bool zero, bool alt,
                bool widthStar, long width, bool precisionStar, int precision, int conversion, int conversionIndex,
                ParseFailure failure)
            {
                Kind = kind;
                Start = start;
                Length = length;
                Left = left;
                Plus = plus;
                Space = space;
                Zero = zero;
                Alt = alt;
                WidthStar = widthStar;
                Width = width;
                PrecisionStar = precisionStar;
                Precision = precision;
                Conversion = conversion;
                ConversionIndex = conversionIndex;
                Failure = failure;
            }

            /// <summary>A run of literal text.</summary>
            /// <param name="start">Its first UTF-16 index.</param>
            /// <param name="length">Its length in UTF-16 units.</param>
            /// <returns>The step.</returns>
            internal static Op Text(int start, int length)
                => new Op(OpKind.Text, start, length, false, false, false, false, false, false, -1, false, -1, -1, -1,
                    ParseFailure.None);

            /// <summary>The <c>%</c> of a <c>%%</c>.</summary>
            internal static Op Percent
                => new Op(OpKind.Percent, 0, 0, false, false, false, false, false, false, -1, false, -1, -1, -1,
                    ParseFailure.None);
        }

        /// <summary>The format string the text steps slice.</summary>
        private readonly string _format;

        /// <summary>The compiled steps, in order; a step with a parse failure is the last.</summary>
        private readonly Op[] _ops;

        /// <summary>Wraps compiled steps.</summary>
        /// <param name="format">The format string.</param>
        /// <param name="ops">Its steps.</param>
        private PercentFormat(string format, Op[] ops)
        {
            _format = format;
            _ops = ops;
        }

        /// <summary>
        ///     Compiles a Python <c>%</c> format string: CPython's parse of each conversion spec (flags <c>-+ #0</c>, a
        ///     width of digits or <c>*</c>, a precision of <c>.</c> then digits or <c>*</c>, ONE length modifier
        ///     <c>h</c> / <c>l</c> / <c>L</c> that is skipped, then the conversion character), recorded without looking at
        ///     an argument.
        /// </summary>
        /// <param name="format">The format string (a Python <c>str</c>; any character, a NUL included, is allowed).</param>
        /// <returns>The compiled format.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="format"/> is null.</exception>
        /// <remarks>
        ///     Compiling never raises a Python error: a spec that CPython cannot parse is recorded with its failure and
        ///     raises when a run reaches it (see the class remarks). A width of digits must not exceed
        ///     <c>PY_SSIZE_T_MAX</c> and a precision <c>INT_MAX</c>, checked digit by digit as CPython checks them, so
        ///     <c>'%9223372036854775808d'</c> fails as soon as its last digit is read even if the string then ends.
        /// </remarks>
        internal static PercentFormat Compile(string format)
        {
            if (format is null)
                throw new ArgumentNullException(nameof(format));

            var ops = new List<Op>();
            int n = format.Length, i = 0, cp = 0;
            while (i < n)
            {
                if (format[i] != '%')
                {
                    // A literal run up to the next '%'. Its code points are counted for the index of a later error: a
                    // surrogate pair is ONE character of the Python string.
                    int start = i;
                    while (i < n && format[i] != '%')
                    {
                        if (char.IsHighSurrogate(format[i]) && i + 1 < n && char.IsLowSurrogate(format[i + 1]))
                            i++;
                        i++;
                        cp++;
                    }
                    ops.Add(Op.Text(start, i - start));
                    continue;
                }

                // unicode_format_arg: the character after the '%'. A second '%' is a literal percent sign that takes no
                // argument; a '%' at the very end reads CPython's NUL terminator, which is not one.
                i++;
                cp++;
                if (i < n && format[i] == '%')
                {
                    ops.Add(Op.Percent);
                    i++;
                    cp++;
                    continue;
                }

                if (!TryParseSpec(format, ref i, ref cp, out Op spec))
                {
                    // The spec ends in a parse error: CPython raises it there and never reads the rest of the string.
                    ops.Add(spec);
                    break;
                }
                ops.Add(spec);
            }
            return new PercentFormat(format, ops.ToArray());
        }

        /// <summary>
        ///     Parses one conversion spec after its <c>%</c> — <c>unicode_format_arg_parse</c>, step for step.
        /// </summary>
        /// <param name="format">The format string.</param>
        /// <param name="i">In: the index just after the <c>%</c>. Out: the index after the conversion character (or where
        ///     the parse failed).</param>
        /// <param name="cp">In / out: the code-point index matching <paramref name="i"/>.</param>
        /// <param name="spec">The parsed spec, or the spec carrying its parse failure.</param>
        /// <returns>False when the spec ends in a parse error (compiling stops after it).</returns>
        private static bool TryParseSpec(string format, ref int i, ref int cp, out Op spec)
        {
            int n = format.Length;

            // "%(key)": the arguments are a tuple, so CPython's `if (ctx->dict == NULL)` refuses the key before reading it.
            if (i < n && format[i] == '(')
            {
                spec = Failed(false, false, false, false, false, false, -1, false, -1, ParseFailure.Mapping);
                return false;
            }

            // Flags, in any order and any number ('0' is always a flag, so a width of digits never starts with 0).
            bool left = false, plus = false, space = false, zero = false, alt = false;
            for (; i < n; i++, cp++)
            {
                char c = format[i];
                if (c == '-') left = true;
                else if (c == '+') plus = true;
                else if (c == ' ') space = true;
                else if (c == '#') alt = true;
                else if (c == '0') zero = true;
                else break;
            }

            // The width: '*' (an argument, taken at run time) or digits, each one checked against PY_SSIZE_T_MAX before
            // it is accumulated — exactly CPython's overflow test.
            bool widthStar = false;
            long width = -1;
            if (i < n && format[i] == '*')
            {
                widthStar = true;
                i++;
                cp++;
            }
            else if (i < n && IsDigit(format[i]))
            {
                width = format[i] - '0';
                i++;
                cp++;
                while (i < n && IsDigit(format[i]))
                {
                    int d = format[i] - '0';
                    if (width > (long.MaxValue - d) / 10)
                    {
                        spec = Failed(left, plus, space, zero, alt, false, -1, false, -1, ParseFailure.WidthTooBig);
                        return false;
                    }
                    width = width * 10 + d;
                    i++;
                    cp++;
                }
            }

            // The precision: '.' then '*' or digits (none is precision 0), the digits checked against INT_MAX.
            bool precisionStar = false;
            int precision = -1;
            if (i < n && format[i] == '.')
            {
                precision = 0;
                i++;
                cp++;
                if (i < n && format[i] == '*')
                {
                    precisionStar = true;
                    i++;
                    cp++;
                }
                else if (i < n && IsDigit(format[i]))
                {
                    precision = format[i] - '0';
                    i++;
                    cp++;
                    while (i < n && IsDigit(format[i]))
                    {
                        int d = format[i] - '0';
                        if (precision > (int.MaxValue - d) / 10)
                        {
                            spec = Failed(left, plus, space, zero, alt, widthStar, width, false, -1, ParseFailure.PrecisionTooBig);
                            return false;
                        }
                        precision = precision * 10 + d;
                        i++;
                        cp++;
                    }
                }
            }

            // ONE length modifier is skipped ("%ld" is "%d"); a second one is the conversion character ("%lld" is the
            // unsupported 'l').
            if (i < n && (format[i] == 'h' || format[i] == 'l' || format[i] == 'L'))
            {
                i++;
                cp++;
            }

            if (i >= n)
            {
                spec = Failed(left, plus, space, zero, alt, widthStar, width, precisionStar, precision, ParseFailure.Incomplete);
                return false;
            }

            // The conversion character, read as a code point. Whether it is one CPython supports is decided at run time,
            // after its argument is taken (unicode_format_arg_format calls getnextarg before its switch).
            int conversion, units;
            if (char.IsHighSurrogate(format[i]) && i + 1 < n && char.IsLowSurrogate(format[i + 1]))
            {
                conversion = char.ConvertToUtf32(format[i], format[i + 1]);
                units = 2;
            }
            else
            {
                conversion = format[i];
                units = 1;
            }
            spec = new Op(OpKind.Spec, 0, 0, left, plus, space, zero, alt, widthStar, width, precisionStar, precision,
                conversion, cp, ParseFailure.None);
            i += units;
            cp++;
            return true;
        }

        /// <summary>A spec step that ends in a parse error.</summary>
        /// <param name="left">The <c>-</c> flag.</param>
        /// <param name="plus">The <c>+</c> flag.</param>
        /// <param name="space">The <c> </c> flag.</param>
        /// <param name="zero">The <c>0</c> flag.</param>
        /// <param name="alt">The <c>#</c> flag.</param>
        /// <param name="widthStar">Whether a <c>*</c> width was read before the error (its argument is taken first).</param>
        /// <param name="width">The written width.</param>
        /// <param name="precisionStar">Whether a <c>*</c> precision was read before the error.</param>
        /// <param name="precision">The written precision.</param>
        /// <param name="failure">The error.</param>
        /// <returns>The step.</returns>
        private static Op Failed(bool left, bool plus, bool space, bool zero, bool alt, bool widthStar, long width,
            bool precisionStar, int precision, ParseFailure failure)
            => new Op(OpKind.Spec, 0, 0, left, plus, space, zero, alt, widthStar, width, precisionStar, precision, -1, -1,
                failure);

        /// <summary>An ASCII decimal digit (CPython compares the code point with <c>'0'</c> and <c>'9'</c>).</summary>
        /// <param name="c">The character.</param>
        /// <returns>True for <c>0</c>–<c>9</c>.</returns>
        private static bool IsDigit(char c) => c >= '0' && c <= '9';

        /// <summary>
        ///     Runs the format over a tuple of arguments, appending the result: CPython's <c>format % args</c>.
        /// </summary>
        /// <param name="sb">The buffer the text is appended to. When the run raises, the text of the steps before the
        ///     error is left in it: a caller that must not keep a partial result truncates back (CPython discards the
        ///     whole string).</param>
        /// <param name="args">The tuple's items.</param>
        /// <param name="kinds">Each item's kind, as long as <paramref name="args"/>: <see cref="PyObject"/>, or the dtype
        ///     of the NumPy scalar the item stands for.</param>
        /// <exception cref="ArgumentException"><paramref name="kinds"/> is not as long as <paramref name="args"/>.</exception>
        /// <exception cref="PrintfArgumentException">CPython's <c>TypeError</c>s: <c>not enough arguments for format
        ///     string</c>, <c>not all arguments converted during string formatting</c>, <c>format requires a mapping</c>,
        ///     <c>* wants int</c>, or a conversion the item's type does not support (<c>%x format: an integer is required,
        ///     not numpy.float64</c>, <c>%c requires int or char</c>, <c>must be real number, not str</c>).</exception>
        /// <exception cref="ValueError"><c>incomplete format</c>, <c>width too big</c>, <c>precision too big</c>,
        ///     <c>unsupported format character 'y' (0x79) at index 1</c>, <c>cannot convert float NaN to integer</c>, or a
        ///     Python <c>int</c> of more than 4300 digits under a decimal conversion.</exception>
        /// <exception cref="OverflowException">CPython's <c>OverflowError</c>s: <c>cannot convert float infinity to
        ///     integer</c>, <c>%c arg not in range(0x110000)</c>, <c>int too large to convert to float</c>, <c>precision too
        ///     large</c> (an integer conversion with a precision past <c>INT_MAX - 3</c>), <c>Python int too large to
        ///     convert to C ssize_t</c> / <c>C int</c> (a <c>*</c> argument).</exception>
        /// <exception cref="OutOfMemoryException">The result would exceed the longest string .NET can hold (CPython's
        ///     <c>MemoryError</c> for a huge width; CPython itself can build somewhat longer strings).</exception>
        internal void FormatInto(StringBuilder sb, ReadOnlySpan<object> args, ReadOnlySpan<NPTypeCode> kinds)
        {
            if (kinds.Length != args.Length)
                throw new ArgumentException("Every argument needs its kind.", nameof(kinds));

            int next = 0;
            foreach (ref readonly Op op in _ops.AsSpan())
            {
                switch (op.Kind)
                {
                    case OpKind.Text:
                        sb.Append(_format, op.Start, op.Length);
                        break;
                    case OpKind.Percent:
                        sb.Append('%');
                        break;
                    default:
                        RunSpec(sb, in op, args, kinds, ref next);
                        break;
                }
            }

            // PyUnicode_Format's last check: a tuple item no spec took.
            if (next < args.Length)
                throw new PrintfArgumentException("not all arguments converted during string formatting");
        }

        /// <summary>
        ///     Runs one conversion spec: its parse error (if any) at CPython's point, the <c>*</c> arguments, the value,
        ///     then the conversion (<c>unicode_format_arg_format</c> + <c>unicode_format_arg_output</c>).
        /// </summary>
        /// <param name="sb">The output buffer.</param>
        /// <param name="op">The spec step.</param>
        /// <param name="args">The tuple's items.</param>
        /// <param name="kinds">Their kinds.</param>
        /// <param name="next">In / out: the index of the next item to take.</param>
        /// <exception cref="PrintfArgumentException">See <see cref="FormatInto"/>.</exception>
        /// <exception cref="ValueError">See <see cref="FormatInto"/>.</exception>
        /// <exception cref="OverflowException">See <see cref="FormatInto"/>.</exception>
        /// <exception cref="OutOfMemoryException">See <see cref="FormatInto"/>.</exception>
        private static void RunSpec(StringBuilder sb, in Op op, ReadOnlySpan<object> args, ReadOnlySpan<NPTypeCode> kinds,
            ref int next)
        {
            // The parse errors raise in the order CPython's parse meets them: the mapping key and a width of digits come
            // before any argument is taken; a precision of digits after a '*' width's argument; the end of the string
            // after both '*' arguments.
            if (op.Failure == ParseFailure.Mapping)
                throw new PrintfArgumentException("format requires a mapping");
            if (op.Failure == ParseFailure.WidthTooBig)
                throw new ValueError("width too big");

            bool left = op.Left;
            long width = op.Width;
            if (op.WidthStar)
            {
                object w = Next(args, kinds, ref next, out NPTypeCode wKind);
                width = StarWidth(w, wKind);
                // A negative '*' width left-justifies. CPython negates it as a Py_ssize_t, so the most negative value
                // stays negative — and a negative width pads nothing.
                if (width < 0)
                {
                    left = true;
                    width = unchecked(-width);
                }
            }

            if (op.Failure == ParseFailure.PrecisionTooBig)
                throw new ValueError("precision too big");

            int precision = op.Precision;
            if (op.PrecisionStar)
            {
                object p = Next(args, kinds, ref next, out NPTypeCode pKind);
                precision = StarPrecision(p, pKind);
                if (precision < 0)
                    precision = 0; // a negative '*' precision is 0, not "none"
            }

            if (op.Failure == ParseFailure.Incomplete)
                throw new ValueError("incomplete format");

            // The value is taken BEFORE the conversion character is checked: '%y' % () reports the missing argument.
            object value = Next(args, kinds, ref next, out NPTypeCode kind);
            int conversion = op.Conversion;
            if (!IsConversion(conversion))
                throw Unsupported(conversion, op.ConversionIndex);

            // A width past int.MaxValue cannot be met by a .NET string: it pads to int.MaxValue, whose allocation raises
            // OutOfMemoryException — after the conversion's own checks, where CPython's MemoryError comes too. A negative
            // width (the most negative '*' width) pads nothing.
            int w32 = width < 0 ? 0 : width > int.MaxValue ? int.MaxValue : (int)width;
            char c = (char)conversion; // every supported conversion is ASCII
            sb.Append(kind == PyObject
                ? PrintfFormatter.ApplyPyObject(c, left, op.Plus, op.Space, op.Zero, op.Alt, w32, precision, value)
                : PrintfFormatter.ApplyNumPyScalar(c, left, op.Plus, op.Space, op.Zero, op.Alt, w32, precision, value, kind));
        }

        /// <summary>CPython's <c>unicode_format_getnextarg</c>: the next tuple item, or its <c>TypeError</c>.</summary>
        /// <param name="args">The tuple's items.</param>
        /// <param name="kinds">Their kinds.</param>
        /// <param name="next">In / out: the index of the next item.</param>
        /// <param name="kind">The item's kind.</param>
        /// <returns>The item.</returns>
        /// <exception cref="PrintfArgumentException"><c>not enough arguments for format string</c>.</exception>
        private static object Next(ReadOnlySpan<object> args, ReadOnlySpan<NPTypeCode> kinds, ref int next, out NPTypeCode kind)
        {
            if (next >= args.Length)
                throw new PrintfArgumentException("not enough arguments for format string");
            kind = kinds[next];
            return args[next++];
        }

        /// <summary>
        ///     A <c>*</c> width: <c>PyLong_Check</c> then <c>PyLong_AsSsize_t</c>. Only a Python <c>int</c> qualifies (a
        ///     <c>bool</c> is one); no NumPy scalar does, not even <c>np.int64</c>, which is not an <c>int</c> subclass.
        /// </summary>
        /// <param name="value">The argument.</param>
        /// <param name="kind">Its kind.</param>
        /// <returns>The width (negative left-justifies).</returns>
        /// <exception cref="PrintfArgumentException"><c>* wants int</c>.</exception>
        /// <exception cref="OverflowException"><c>Python int too large to convert to C ssize_t</c>.</exception>
        private static long StarWidth(object value, NPTypeCode kind)
        {
            if (kind != PyObject || !PrintfFormatter.TryPyInt(value, out BigInteger big))
                throw new PrintfArgumentException("* wants int");
            if (big < long.MinValue || big > long.MaxValue)
                throw new OverflowException("Python int too large to convert to C ssize_t");
            return (long)big;
        }

        /// <summary>A <c>*</c> precision: <c>PyLong_Check</c> then <c>_PyLong_AsInt</c> (a C <c>int</c>).</summary>
        /// <param name="value">The argument.</param>
        /// <param name="kind">Its kind.</param>
        /// <returns>The precision (negative means 0).</returns>
        /// <exception cref="PrintfArgumentException"><c>* wants int</c>.</exception>
        /// <exception cref="OverflowException"><c>Python int too large to convert to C int</c>.</exception>
        private static int StarPrecision(object value, NPTypeCode kind)
        {
            if (kind != PyObject || !PrintfFormatter.TryPyInt(value, out BigInteger big))
                throw new PrintfArgumentException("* wants int");
            if (big < int.MinValue || big > int.MaxValue)
                throw new OverflowException("Python int too large to convert to C int");
            return (int)big;
        }

        /// <summary>The conversion characters <c>unicode_format_arg_format</c> handles.</summary>
        /// <param name="c">A code point.</param>
        /// <returns>True for <c>s r a i d u o x X e E f F g G c</c>.</returns>
        private static bool IsConversion(int c)
            => c is 's' or 'r' or 'a' or 'i' or 'd' or 'u' or 'o' or 'x' or 'X' or 'e' or 'E' or 'f' or 'F' or 'g' or 'G'
                or 'c';

        /// <summary>CPython's error for any other conversion character: the character itself when printable ASCII
        ///     (31–126), else <c>?</c>, then its code point in lowercase hex and its index in code points.</summary>
        /// <param name="c">The conversion code point.</param>
        /// <param name="index">Its code-point index in the format.</param>
        /// <returns><c>ValueError: unsupported format character 'y' (0x79) at index 1</c>.</returns>
        private static ValueError Unsupported(int c, int index)
        {
            char shown = c >= 31 && c <= 126 ? (char)c : '?';
            return new ValueError($"unsupported format character '{shown}' (0x{c:x}) at index {index}");
        }
    }
}
