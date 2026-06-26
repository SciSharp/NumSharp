using System;
using System.Collections.Generic;
using System.Globalization;
using System.Numerics;
using System.Text;

namespace NumSharp.Backends.Printing
{
    /// <summary>
    ///     C# port of NumPy's array-to-string machinery in <c>numpy/_core/arrayprint.py</c>:
    ///     the recursive layout engine (<c>_formatArray</c>), the per-dtype formatter selection
    ///     (<c>_get_format_function</c>), and the <c>array2string</c> / <c>array_str</c> /
    ///     <c>array_repr</c> entry points.
    /// </summary>
    internal static class ArrayFormatter
    {
        private static readonly CultureInfo CI = CultureInfo.InvariantCulture;

        // NumPy's _typelessdata: dtypes whose repr does not need an explicit "dtype=" suffix.
        private static bool DtypeIsImplied(NPTypeCode tc) =>
            tc == NPTypeCode.Int64 || tc == NPTypeCode.Double ||
            tc == NPTypeCode.Complex || tc == NPTypeCode.Boolean;

        #region public entry points

        /// <summary>Port of <c>array_str</c> — the data of an array as a single string (NumPy <c>str</c>).</summary>
        public static string ArrayStr(NDArray a, PrintOptions opts)
        {
            // 0d arrays print like a scalar: floats are not truncated by precision and strings are
            // not quoted, so we return the scalar's str.
            if (a.ndim == 0)
                return ScalarStr(a.GetAtIndex(0), a.typecode);

            return Array2String(a, opts, " ", "", "");
        }

        /// <summary>Port of <c>array_repr</c> — the array with class/dtype/shape info (NumPy <c>repr</c>).</summary>
        public static string ArrayRepr(NDArray a, PrintOptions opts)
        {
            const string prefix = "array(";
            string lst = Array2String(a, opts, ", ", prefix, ")");

            var extras = new List<string>();
            bool shapeShown = (a.size == 0 && !(a.ndim == 1 && a.shape[0] == 0))
                              || (opts.legacy > 210 && a.size > opts.threshold);
            if (shapeShown)
                extras.Add("shape=" + ShapeTuple(a.shape));
            if (!DtypeIsImplied(a.typecode) || a.size == 0)
                extras.Add("dtype=" + DtypeShortRepr(a.typecode));

            if (extras.Count == 0)
                return prefix + lst + ")";

            string arrStr = prefix + lst + ",";
            string extraStr = string.Join(", ", extras) + ")";

            int lastNewline = arrStr.LastIndexOf('\n');
            int lastLineLen = arrStr.Length - (lastNewline + 1);
            string spacer = " ";
            if (lastLineLen + extraStr.Length + 1 > opts.linewidth)
                spacer = "\n" + new string(' ', prefix.Length);

            return arrStr + spacer + extraStr;
        }

        /// <summary>Port of <c>array2string</c>.</summary>
        public static string Array2String(NDArray a, PrintOptions opts, string separator, string prefix, string suffix)
        {
            int linewidth = opts.linewidth;
            if (opts.legacy > 113)
                linewidth -= suffix.Length;

            // treat as a null array if any of the shape elements == 0
            if (a.size == 0)
                return "[]";

            bool summarize = a.size > opts.threshold;

            var reduced = CollectReducedValues(a, opts.edgeitems, summarize);
            IElementFormatter formatFn = BuildFormatFunction(reduced, a.typecode, a.ndim, opts);

            string nextLinePrefix = " " + new string(' ', prefix.Length);
            string summaryInsert = summarize ? "..." : "";

            return FormatArray(a, formatFn, linewidth, nextLinePrefix, separator,
                opts.edgeitems, summaryInsert, opts.legacy);
        }

        #endregion

        #region recursive layout engine (_formatArray)

        private static string FormatArray(NDArray a, IElementFormatter formatFn, int lineWidth,
            string nextLinePrefix, string separator, int edgeItems, string summaryInsert, long legacy)
        {
            return Recurse(a, formatFn, lineWidth, separator, edgeItems, summaryInsert, legacy,
                nextLinePrefix, lineWidth);
        }

        private static string Recurse(NDArray cur, IElementFormatter formatFn, int lineWidth,
            string separator, int edgeItems, string summaryInsert, long legacy,
            string hangingIndent, int currWidth)
        {
            int axesLeft = cur.ndim;
            if (axesLeft == 0)
                return formatFn.Format(cur.GetAtIndex(0));

            string nextHangingIndent = hangingIndent + " ";
            int nextWidth = legacy <= 113 ? currWidth : currWidth - 1; // - len(']')

            long aLen = cur.shape[0];
            bool showSummary = summaryInsert.Length != 0 && 2L * edgeItems < aLen;
            long leadingItems = showSummary ? edgeItems : 0L;
            long trailingItems = showSummary ? edgeItems : aLen;

            var s = new StringBuilder();

            if (axesLeft == 1)
            {
                int elemWidth = currWidth - Math.Max(RStripLen(separator), 1); // max(sep.rstrip, ']')
                string line = hangingIndent;

                for (long i = 0; i < leadingItems; i++)
                {
                    string word = formatFn.Format(cur.GetAtIndex(i));
                    line = ExtendLine(s, line, word, elemWidth, hangingIndent, legacy);
                    line += separator;
                }

                if (showSummary)
                {
                    line = ExtendLine(s, line, summaryInsert, elemWidth, hangingIndent, legacy);
                    line += separator;
                }

                for (long i = trailingItems; i > 1; i--)
                {
                    string word = formatFn.Format(cur.GetAtIndex(aLen - i));
                    line = ExtendLine(s, line, word, elemWidth, hangingIndent, legacy);
                    line += separator;
                }

                string lastWord = formatFn.Format(cur.GetAtIndex(aLen - 1));
                line = ExtendLine(s, line, lastWord, elemWidth, hangingIndent, legacy);

                s.Append(line);
            }
            else
            {
                string lineSep = RStrip(separator) + new string('\n', axesLeft - 1);

                for (long i = 0; i < leadingItems; i++)
                {
                    string nested = Recurse(cur[i], formatFn, lineWidth, separator, edgeItems,
                        summaryInsert, legacy, nextHangingIndent, nextWidth);
                    s.Append(hangingIndent).Append(nested).Append(lineSep);
                }

                if (showSummary)
                    s.Append(hangingIndent).Append(summaryInsert).Append(lineSep);

                for (long i = trailingItems; i > 1; i--)
                {
                    string nested = Recurse(cur[aLen - i], formatFn, lineWidth, separator, edgeItems,
                        summaryInsert, legacy, nextHangingIndent, nextWidth);
                    s.Append(hangingIndent).Append(nested).Append(lineSep);
                }

                string lastNested = Recurse(cur[aLen - 1], formatFn, lineWidth, separator, edgeItems,
                    summaryInsert, legacy, nextHangingIndent, nextWidth);
                s.Append(hangingIndent).Append(lastNested);
            }

            // remove the hanging indent, and wrap in []
            string body = s.ToString();
            return "[" + body.Substring(hangingIndent.Length) + "]";
        }

        // Port of _extendLine. Appends a wrapped line to `s` and returns the new `line`.
        private static string ExtendLine(StringBuilder s, string line, string word, int lineWidth,
            string nextLinePrefix, long legacy)
        {
            bool needsWrap = line.Length + word.Length > lineWidth;
            if (legacy > 113 && line.Length <= nextLinePrefix.Length)
                needsWrap = false;

            if (needsWrap)
            {
                s.Append(RStrip(line)).Append('\n');
                line = nextLinePrefix;
            }

            return line + word;
        }

        private static string RStrip(string s) => s.TrimEnd(' ');
        private static int RStripLen(string s) => RStrip(s).Length;

        #endregion

        #region formatter selection (_get_format_function)

        private static IElementFormatter BuildFormatFunction(IReadOnlyList<object> reduced,
            NPTypeCode tc, int rank, PrintOptions opts)
        {
            switch (tc)
            {
                case NPTypeCode.Boolean:
                    return new BoolFormat(rank);
                case NPTypeCode.SByte:
                case NPTypeCode.Byte:
                case NPTypeCode.Int16:
                case NPTypeCode.UInt16:
                case NPTypeCode.Int32:
                case NPTypeCode.UInt32:
                case NPTypeCode.Int64:
                case NPTypeCode.UInt64:
                case NPTypeCode.Char:
                    return new IntegerFormat(reduced, opts.sign);
                case NPTypeCode.Half:
                    return new FloatingFormat(reduced, FloatKind.Half, 3, opts, opts.sign);
                case NPTypeCode.Single:
                    return new FloatingFormat(reduced, FloatKind.Single, 6, opts, opts.sign);
                case NPTypeCode.Double:
                    return new FloatingFormat(reduced, FloatKind.Double, 15, opts, opts.sign);
                case NPTypeCode.Decimal:
                    return new FloatingFormat(reduced, FloatKind.Double, 15, opts, opts.sign);
                case NPTypeCode.Complex:
                    return new ComplexFloatingFormat(reduced, opts, opts.sign);
                default:
                    throw new NotSupportedException($"Cannot format dtype {tc}.");
            }
        }

        // Collect the leading/trailing "corner" values that NumPy's _leading_trailing feeds to
        // _get_format_function (only these participate in column-width / exp-format decisions).
        private static List<object> CollectReducedValues(NDArray a, int edgeitems, bool summarize)
        {
            var result = new List<object>();
            if (a.ndim == 0)
            {
                result.Add(a.GetAtIndex(0));
                return result;
            }

            Walk(a, summarize, edgeitems, result);
            return result;
        }

        private static void Walk(NDArray cur, bool summarize, int edge, List<object> outp)
        {
            long n = cur.shape[0];
            if (cur.ndim == 1)
            {
                ForEachIndex(n, summarize, edge, i => outp.Add(cur.GetAtIndex(i)));
            }
            else
            {
                ForEachIndex(n, summarize, edge, i => Walk(cur[i], summarize, edge, outp));
            }
        }

        private static void ForEachIndex(long n, bool summarize, int edge, Action<long> action)
        {
            if (summarize && n > 2L * edge)
            {
                for (long i = 0; i < edge; i++) action(i);
                for (long i = n - edge; i < n; i++) action(i);
            }
            else
            {
                for (long i = 0; i < n; i++) action(i);
            }
        }

        #endregion

        #region scalar str / dtype-shape helpers

        // str() of a 0d array element (Python scalar str semantics).
        internal static string ScalarStr(object value, NPTypeCode tc)
        {
            switch (tc)
            {
                case NPTypeCode.Boolean:
                    return Convert.ToBoolean(value) ? "True" : "False";
                case NPTypeCode.SByte:
                case NPTypeCode.Byte:
                case NPTypeCode.Int16:
                case NPTypeCode.UInt16:
                case NPTypeCode.Int32:
                case NPTypeCode.UInt32:
                case NPTypeCode.Int64:
                case NPTypeCode.UInt64:
                case NPTypeCode.Char:
                    return IntegerFormat.DecimalString(value);
                case NPTypeCode.Half:
                    return PythonFloatRepr(value, FloatKind.Half);
                case NPTypeCode.Single:
                    return PythonFloatRepr(value, FloatKind.Single);
                case NPTypeCode.Double:
                    return PythonFloatRepr(value, FloatKind.Double);
                case NPTypeCode.Decimal:
                    return PythonFloatRepr((double)(decimal)value, FloatKind.Double);
                case NPTypeCode.Complex:
                    return PythonComplexRepr((Complex)value);
                default:
                    return value?.ToString() ?? "";
            }
        }

        // Python's float repr: shortest digits, scientific iff decExp < -4 or decExp >= 16.
        private static string PythonFloatRepr(object v, FloatKind kind)
        {
            double d = kind switch
            {
                FloatKind.Half => (double)(Half)v,
                FloatKind.Single => (float)v,
                _ => Convert.ToDouble(v, CI)
            };

            if (double.IsNaN(d)) return "nan";
            if (double.IsPositiveInfinity(d)) return "inf";
            if (double.IsNegativeInfinity(d)) return "-inf";

            var (digits, decExp, neg) = Dragon4.ExtractShortest(v, kind);
            string sign = neg ? "-" : "";
            if (digits == "0")
                return sign + "0.0";

            if (decExp >= -4 && decExp < 16)
            {
                var (intStr, fracStr) = Dragon4.SplitPositional(digits, decExp);
                return fracStr.Length == 0 ? sign + intStr + ".0" : sign + intStr + "." + fracStr;
            }

            string lead = digits.Substring(0, 1);
            string rest = digits.Length > 1 ? "." + digits.Substring(1) : "";
            string es = (decExp >= 0 ? "+" : "-") + Math.Abs(decExp).ToString(CI).PadLeft(2, '0');
            return sign + lead + rest + "e" + es;
        }

        // str() of a 0d complex array, matching Python's complex repr "(a+bj)".
        private static string PythonComplexRepr(Complex c)
        {
            string im = PythonFloatPart(c.Imaginary);
            // imaginary always carries an explicit sign
            if (!(im.StartsWith("-") || im.StartsWith("+")))
                im = "+" + im;

            if (c.Real == 0.0 && !double.IsNegative(c.Real))
                return im + "j";

            return "(" + PythonFloatPart(c.Real) + im + "j)";
        }

        // float repr used inside a complex: like Python's, integers drop the ".0".
        private static string PythonFloatPart(double d)
        {
            if (double.IsNaN(d)) return "nan";
            if (double.IsPositiveInfinity(d)) return "inf";
            if (double.IsNegativeInfinity(d)) return "-inf";
            string s = PythonFloatRepr(d, FloatKind.Double);
            if (s.EndsWith(".0"))
                s = s.Substring(0, s.Length - 2);
            return s;
        }

        private static string DtypeShortRepr(NPTypeCode tc) => tc.AsNumpyDtypeName();

        private static string ShapeTuple(long[] shape)
        {
            if (shape.Length == 1)
                return "(" + shape[0].ToString(CI) + ",)";
            var sb = new StringBuilder("(");
            for (int i = 0; i < shape.Length; i++)
            {
                if (i > 0) sb.Append(", ");
                sb.Append(shape[i].ToString(CI));
            }
            sb.Append(')');
            return sb.ToString();
        }

        #endregion
    }
}
