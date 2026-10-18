using System;
using System.Numerics;
using System.Text;
using NumSharp.Backends;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Printing;
using NumSharp.Backends.Strings;

// =============================================================================
// NDPolyBase.Printing.cs — __str__ / __format__ / __repr__ / _repr_latex_ of the series classes
// =============================================================================
//
// NumPy 2.4.2's _polybase.py printers, statement for statement:
//
//   __str__        _generate_string with the unicode or the ASCII term printer (np.polynomial.set_default_printstyle;
//                  ASCII on Windows, NumPy's `os.name != 'nt'`).
//   __format__     '' is __str__; 'ascii' / 'unicode' pick the printer; anything else is NumPy's ValueError.
//   _generate_string  pu.format_float(coef[0]); then for each further coefficient " + "/" - " and its
//                  |value| (format_float with parens: a scientific value is parenthesized), the term printer
//                  with the power and the scaled symbol (the domain map `off + scl x`, parenthesized unless the
//                  identity), and NumPy's line breaking at np.get_printoptions()['linewidth'].
//   __repr__       repr(self.coef)[6:-1] (the array repr without "array(" and ")"), the same for the domain and
//                  the window, and symbol='x'.
//   _repr_latex_   "$x \mapsto ...$" with \text{...} scalars, \color{LightGray}{...} around zero terms, and a
//                  non-real coefficient parenthesized.
//
// The classes differ only in the term printers: every basis but the power series writes `basis_name` with
// the power as a subscript (`T_2(x)` / `·T₂(x)` / `{T}_{2}(x)`); Polynomial writes powers of the symbol
// (`x**2` / `·x²` / `x^{2}`, with \left( \right) around a scaled symbol).
//
// A coefficient prints through polyutils.format_float (NDPolyPrinting.FormatFloat): its dtype's own digits for a
// float, Python's str() for anything else (a complex prints as '(1+2j)'). The comparisons are NumPy's scalar ones:
// a complex orders lexicographically (`re > 0 or (re == 0 and im >= 0)`), a NaN is never >= 0.
//
// =============================================================================

namespace NumSharp
{
    public abstract partial class NDPolyBase
    {
        /// <summary>NumPy's <c>_superscript_mapping</c>: the digits 0-9 as Unicode superscripts.</summary>
        private protected const string Superscripts = "⁰¹²³⁴⁵⁶⁷⁸⁹";

        /// <summary>NumPy's <c>_subscript_mapping</c>: the digits 0-9 as Unicode subscripts.</summary>
        private const string Subscripts = "₀₁₂₃₄₅₆₇₈₉";

        /// <summary>
        ///     NumPy's <c>str(p)</c> (<c>__str__</c>): the series written out with the current print style — the ASCII printer
        ///     (<c>1.0 + 2.0 x + 3.0 x**2</c>) on Windows unless <c>np.polynomial.set_default_printstyle("unicode")</c> was
        ///     called, the Unicode one (<c>1.0 + 2.0·x + 3.0·x²</c>) elsewhere.
        /// </summary>
        /// <returns>The text.</returns>
        /// <exception cref="ValueError">An assigned N-D coefficient array (NumPy's truth-value error of <c>coef &gt;= 0</c>).</exception>
        public string __str__() => GenerateString(NDPolyPrinting.UseUnicode);

        /// <summary>NumPy's <c>str(p)</c> (see <see cref="__str__"/>).</summary>
        /// <returns>The text.</returns>
        public override string ToString() => __str__();

        /// <summary>
        ///     NumPy's <c>format(p, fmt)</c> (<c>__format__</c>): <c>''</c> is <see cref="__str__"/>, <c>'ascii'</c> and
        ///     <c>'unicode'</c> force a printer.
        /// </summary>
        /// <param name="fmt_str">The format: <c>""</c>, <c>"ascii"</c> or <c>"unicode"</c>.</param>
        /// <returns>The text.</returns>
        /// <exception cref="ValueError">Any other format: <c>Unsupported format string 'x' passed to &lt;class
        ///     'numpy.polynomial.polynomial.Polynomial'&gt;.__format__. Valid options are 'ascii' and 'unicode'</c>.</exception>
        public string __format__(string fmt_str)
        {
            if (string.IsNullOrEmpty(fmt_str))
                return __str__();
            if (fmt_str != "ascii" && fmt_str != "unicode")
                throw new ValueError($"Unsupported format string '{fmt_str}' passed to {_kind}.__format__. Valid options are 'ascii' and 'unicode'");
            return GenerateString(fmt_str == "unicode");
        }

        /// <summary>
        ///     C#'s formatting (<c>$"{p:unicode}"</c>, <c>string.Format("{0:ascii}", p)</c>): NumPy's <c>__format__</c>; a null
        ///     format is <c>''</c>.
        /// </summary>
        /// <param name="format">The format (see <see cref="__format__"/>).</param>
        /// <param name="formatProvider">Ignored: NumPy's printers do not localize.</param>
        /// <returns>The text.</returns>
        /// <exception cref="ValueError">An unsupported format.</exception>
        public string ToString(string format, IFormatProvider formatProvider) => __format__(format ?? string.Empty);

        /// <summary>
        ///     NumPy's <c>repr(p)</c> (<c>__repr__</c>): <c>Polynomial([1., 2., 3.], domain=[-1.,  1.], window=[-1.,  1.],
        ///     symbol='x')</c> — each array's repr with <c>array(</c> and <c>)</c> cut off, so a dtype and NumPy's line wrapping
        ///     and summarization come through as the array printer makes them.
        /// </summary>
        /// <returns>The text.</returns>
        public string __repr__()
            => $"{_kind.__name__}({ReprBody(Coef)}, domain={ReprBody(domain)}, window={ReprBody(window)}, symbol='{symbol}')";

        /// <summary>Python's <c>repr(a)[6:-1]</c> of an array.</summary>
        /// <param name="a">The array.</param>
        /// <returns>The repr without its first six and its last character (empty when shorter, as Python's slice).</returns>
        private static string ReprBody(NDArray a)
        {
            string r = a.ToString(true);
            return r.Length <= 7 ? string.Empty : r.Substring(6, r.Length - 7);
        }

        /// <summary>
        ///     NumPy's <c>_generate_string(term_method)</c>: the first coefficient, then <c>" + "</c> / <c>" - "</c> with each
        ///     further coefficient's magnitude and its term, breaking the line (the term's first space becomes a newline) when
        ///     it would reach <c>linewidth</c>.
        /// </summary>
        /// <param name="unicode">Use the Unicode term printer (else the ASCII one).</param>
        /// <returns>The text.</returns>
        /// <exception cref="ValueError">An assigned N-D coefficient array (NumPy's truth-value error).</exception>
        private string GenerateString(bool unicode)
        {
            using var scope = NDScope.Open();
            int linewidth = PrintOptions.Current.linewidth;
            if (linewidth < 1)
                linewidth = 1;
            NDArray c = Coef;
            long n = PyLen(c);
            var sb = new StringBuilder();
            sb.Append(NDPolyPrinting.FormatFloat(Item(c, 0), parens: false));

            var (off, scale) = MapParmsOf(domain, window);
            var (scaledSymbol, needsParens) = FormatTerm(v => NDPolyPrinting.FormatFloat(v, parens: false), off, scale);
            if (needsParens)
                scaledSymbol = "(" + scaledSymbol + ")";

            for (long i = 0; i < n - 1; i++)
            {
                sb.Append(' ');
                string power = (i + 1).ToString(System.Globalization.CultureInfo.InvariantCulture);
                object coef = Item(c, i + 1);
                string nextTerm;
                // `if coef >= 0` — a value with no ordering against 0 (a str coefficient) takes NumPy's TypeError branch.
                if (IsUnordered(coef))
                    nextTerm = "+ " + NDPyObject.Str(coef);
                else if (GreaterEqualZero(coef))
                    nextTerm = "+ " + NDPolyPrinting.FormatFloat(coef, parens: true);
                else
                    nextTerm = "- " + NDPolyPrinting.FormatFloat(Negated(coef), parens: true);
                nextTerm += unicode ? StrTermUnicode(power, scaledSymbol) : StrTermAscii(power, scaledSymbol);

                // len(out.split('\n')[-1]) + len(next_term), two more for the "+ " of a term still to come.
                int lastBreak = LastIndexOfNewline(sb);
                long lineLen = sb.Length - (lastBreak + 1) + nextTerm.Length;
                if (i < n - 2)
                    lineLen += 2;
                if (lineLen >= linewidth)
                {
                    int sp = nextTerm.IndexOf(' ');
                    if (sp >= 0)
                        nextTerm = nextTerm.Substring(0, sp) + "\n" + nextTerm.Substring(sp + 1);
                }
                sb.Append(nextTerm);
            }
            return sb.ToString();
        }

        /// <summary>The index of the last newline in the builder, or -1.</summary>
        /// <param name="sb">The text so far.</param>
        /// <returns>The index.</returns>
        private static int LastIndexOfNewline(StringBuilder sb)
        {
            for (int i = sb.Length - 1; i >= 0; i--)
                if (sb[i] == '\n')
                    return i;
            return -1;
        }

        /// <summary>
        ///     NumPy's <c>_format_term(scalar_format, off, scale)</c>: the argument of the basis functions — the bare symbol for
        ///     the identity map (<c>off == 0 and scale == 1</c>), else <c>"{off} + {scale}{symbol}"</c> with the zero offset or
        ///     the unit scale left out, and whether it needs parentheses.
        /// </summary>
        /// <param name="scalarFormat">The scalar printer (format_float, or the LaTeX \text{} one).</param>
        /// <param name="off">The map's offset.</param><param name="scale">The map's scale.</param>
        /// <returns>The term and whether it is parenthesized by the caller.</returns>
        private (string term, bool needsParens) FormatTerm(Func<object, string> scalarFormat, in PolyNumber off, in PolyNumber scale)
        {
            bool offZero = ScalarEqualsInt(off, 0), scaleOne = ScalarEqualsInt(scale, 1);
            if (offZero && scaleOne)
                return (symbol, false);
            if (scaleOne)
                return ($"{scalarFormat(ValueOf(off))} + {symbol}", true);
            if (offZero)
                return ($"{scalarFormat(ValueOf(scale))}{symbol}", true);
            return ($"{scalarFormat(ValueOf(off))} + {scalarFormat(ValueOf(scale))}{symbol}", true);
        }

        /// <summary>
        ///     The term printers' Unicode form: <c>·T₂(x)</c> (the basis name with a subscript power); the power series
        ///     overrides it.
        /// </summary>
        /// <param name="i">The power, as text.</param><param name="argStr">The (scaled) symbol.</param>
        /// <returns>The term.</returns>
        private protected virtual string StrTermUnicode(string i, string argStr)
            => "·" + basis_name + Translate(i, Subscripts) + "(" + argStr + ")";

        /// <summary>The term printers' ASCII form: <c> T_2(x)</c>; the power series overrides it.</summary>
        /// <param name="i">The power, as text.</param><param name="argStr">The (scaled) symbol.</param>
        /// <returns>The term.</returns>
        private protected virtual string StrTermAscii(string i, string argStr)
            => " " + basis_name + "_" + i + "(" + argStr + ")";

        /// <summary>The LaTeX term: <c>{T}_{2}(x)</c> (the argument is always parenthesized); the power series overrides it.</summary>
        /// <param name="i">The power.</param><param name="argStr">The (scaled) symbol.</param>
        /// <param name="needsParens">Whether the argument is a scaled symbol (unused here: NumPy always adds parentheses).</param>
        /// <returns>The term.</returns>
        private protected virtual string ReprLatexTerm(long i, string argStr, bool needsParens)
            => "{" + basis_name + "}_{" + i.ToString(System.Globalization.CultureInfo.InvariantCulture) + "}(" + argStr + ")";

        /// <summary>Python's <c>str.translate</c> of the decimal digits through a ten-character table.</summary>
        /// <param name="digits">The digits.</param><param name="table">The characters for 0-9.</param>
        /// <returns>The translated text.</returns>
        private protected static string Translate(string digits, string table)
        {
            var chars = new char[digits.Length];
            for (int k = 0; k < digits.Length; k++)
            {
                char d = digits[k];
                chars[k] = d >= '0' && d <= '9' ? table[d - '0'] : d;
            }
            return new string(chars);
        }

        /// <summary>
        ///     NumPy's <c>_repr_latex_()</c>: <c>$x \mapsto \text{1.0} + \text{2.0}\,x$</c> — each coefficient a LaTeX
        ///     <c>\text{}</c> scalar, a non-real one in parentheses, a zero term greyed out with
        ///     <c>\color{LightGray}{...}</c>.
        /// </summary>
        /// <returns>The LaTeX text.</returns>
        /// <exception cref="ValueError">An assigned N-D coefficient array (NumPy's truth-value error).</exception>
        public string _repr_latex_()
        {
            using var scope = NDScope.Open();
            var (off, scale) = MapParmsOf(domain, window);
            var (term, needsParens) = FormatTerm(v => LatexScalar(v, false), off, scale);
            NDArray c = Coef;
            long n = PyLen(c);
            var body = new StringBuilder();
            for (long i = 0; i < n; i++)
            {
                object coef = Item(c, i);
                string coefStr;
                if (i == 0)
                    coefStr = LatexScalar(coef, false);
                else if (!IsReal(coef))
                    coefStr = " + (" + LatexScalar(coef, false) + ")";
                else if (GreaterEqualZero(coef))
                    coefStr = " + " + LatexScalar(coef, true);
                else
                    coefStr = " - " + LatexScalar(Negated(coef), true);
                string termStr = ReprLatexTerm(i, term, needsParens);
                string part = termStr == "1" ? coefStr : coefStr + "\\," + termStr;
                if (EqualsZero(coef))
                    part = "\\color{LightGray}{" + part + "}";
                body.Append(part);
            }
            string b = body.Length > 0 ? body.ToString() : "0";
            return "$" + symbol + " \\mapsto " + b + "$";
        }

        /// <summary>NumPy's <c>_repr_latex_scalar(x, parens)</c>: <c>\text{...}</c> around format_float's text.</summary>
        /// <param name="x">The scalar.</param><param name="parens">Parenthesize a scientific value.</param>
        /// <returns>The LaTeX text.</returns>
        private static string LatexScalar(object x, bool parens) => "\\text{" + NDPolyPrinting.FormatFloat(x, parens) + "}";

        // ---------------------------------------------------------------------------------------------
        //  Scalar helpers (NumPy scalar semantics of the coefficient elements)
        // ---------------------------------------------------------------------------------------------

        /// <summary><c>self.coef[i]</c>: the element as its boxed C# value (a NumPy scalar), or the row of an N-D array.</summary>
        /// <param name="c">The coefficients.</param><param name="i">The index.</param>
        /// <returns>The element or row.</returns>
        private static object Item(NDArray c, long i) => c.ndim == 1 ? c.GetAtIndex(i) : c[i];

        /// <summary>A NumPy scalar's boxed C# value (what format_float receives for it).</summary>
        /// <param name="v">The scalar.</param>
        /// <returns>The value (an NDArray for an array).</returns>
        private static object ValueOf(in PolyNumber v) => v.ToObject();

        /// <summary>Whether <c>value &gt;= 0</c> raises TypeError in NumPy (a str has no ordering against an int).</summary>
        /// <param name="value">The element.</param>
        /// <returns>Whether NumPy's TypeError branch runs.</returns>
        private static bool IsUnordered(object value) => value is string;

        /// <summary>
        ///     NumPy's scalar <c>value &gt;= 0</c>: IEEE for a float (NaN false), lexicographic for a complex
        ///     (<c>re &gt; 0 or (re == 0 and im &gt;= 0)</c>, probed), plain for an integer, a bool or a decimal; an N-D
        ///     coefficient row has no single truth value.
        /// </summary>
        /// <param name="value">The element.</param>
        /// <returns>The comparison.</returns>
        /// <exception cref="ValueError">A row of more than one element (NumPy's truth-value text).</exception>
        private static bool GreaterEqualZero(object value) => value switch
        {
            double d => d >= 0,
            float f => f >= 0,
            Half h => h >= Half.Zero,
            Complex z => z.Real > 0 || (z.Real == 0 && z.Imaginary >= 0),
            decimal m => m >= 0,
            bool => true,
            char => true,
            NDArray row => RowTruth(row, r => GreaterEqualZero(r)),
            _ => Convert.ToDouble(value, System.Globalization.CultureInfo.InvariantCulture) >= 0,
        };

        /// <summary>NumPy's scalar <c>value == 0</c> (NaN false; a complex needs both parts zero).</summary>
        /// <param name="value">The element.</param>
        /// <returns>The comparison.</returns>
        /// <exception cref="ValueError">A row of more than one element.</exception>
        private static bool EqualsZero(object value) => value switch
        {
            double d => d == 0,
            float f => f == 0,
            Half h => h == Half.Zero,
            Complex z => z.Real == 0 && z.Imaginary == 0,
            decimal m => m == 0,
            bool b => !b,
            char ch => ch == 0,
            string => false,
            NDArray row => RowTruth(row, r => EqualsZero(r)),
            _ => Convert.ToDouble(value, System.Globalization.CultureInfo.InvariantCulture) == 0,
        };

        /// <summary><c>isinstance(c, numbers.Real)</c>: the real NumPy scalar types (floats, integers, NumSharp's decimal) —
        ///     not a complex, a bool or a str.</summary>
        /// <param name="value">The element.</param>
        /// <returns>Whether it is a Real.</returns>
        private static bool IsReal(object value) => value is not (Complex or bool or string or NDArray);

        /// <summary>The truth value NumPy takes of an elementwise comparison of a row: its one element, or an error.</summary>
        /// <param name="row">The row.</param><param name="test">The scalar comparison.</param>
        /// <returns>The single element's comparison.</returns>
        /// <exception cref="ValueError">NumPy's truth-value texts (no element, or more than one).</exception>
        private static bool RowTruth(NDArray row, Func<object, bool> test)
        {
            if (row.size == 0)
                throw new ValueError("The truth value of an empty array is ambiguous. Use `array.size > 0` to check that an array is not empty.");
            if (row.size > 1)
                throw new ValueError("The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()");
            return test(row.GetAtIndex(0));
        }

        /// <summary>
        ///     NumPy's scalar <c>-value</c> of a coefficient that compared below zero: the sign flipped (a complex negates both
        ///     parts, an integer wraps in its own width, as NumPy's scalar negative does).
        /// </summary>
        /// <param name="value">The element.</param>
        /// <returns>The negated value, boxed as its C# type (what format_float receives).</returns>
        /// <exception cref="NotSupportedException">An element type no coefficient dtype has.</exception>
        private static object Negated(object value) => value switch
        {
            // Every arm is cast to object: without it the switch takes its arms' best common type — NDArray, through the
            // implicit number → NDArray conversions — and format_float would print the 0-d array's str() (no sign='+').
            double d => (object)(-d),
            float f => (object)(-f),
            Half h => (object)(-h),
            Complex z => (object)new Complex(-z.Real, -z.Imaginary),
            decimal m => (object)(-m),
            long l => (object)unchecked(-l),
            int k => (object)unchecked(-k),
            short s => (object)unchecked((short)-s),
            sbyte sb => (object)unchecked((sbyte)-sb),
            NDArray row => np.negative(row),
            _ => throw new NotSupportedException($"a {value.GetType().Name} coefficient has no NumPy negative"),
        };

        /// <summary>Python's <c>value == k</c> of a NumPy scalar and a Python int (IEEE; a complex by both parts).</summary>
        /// <param name="v">The scalar.</param><param name="k">The int.</param>
        /// <returns>The comparison.</returns>
        private static bool ScalarEqualsInt(in PolyNumber v, int k) => v.ToObject() switch
        {
            double d => d == k,
            float f => f == k,
            Half h => (float)h == k,
            Complex z => z.Real == k && z.Imaginary == 0,
            decimal m => m == k,
            long l => l == k,
            BigInteger b => b == k,
            bool b => (b ? 1 : 0) == k,
            NDArray => false,
            var o => Convert.ToDouble(o, System.Globalization.CultureInfo.InvariantCulture) == k,
        };
    }
}
