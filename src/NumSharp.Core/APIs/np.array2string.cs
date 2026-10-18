using System;
using System.Collections.Generic;
using NumSharp.Backends.Printing;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return a string representation of an array (NumPy's <c>np.array2string</c>).
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="max_line_width">Inserts newlines if text is longer than this. Defaults to the current linewidth.</param>
        /// <param name="precision">Floating point precision. Defaults to the current precision.</param>
        /// <param name="suppress_small">Represent numbers very close to zero as zero. Defaults to the current option.</param>
        /// <param name="separator">Inserted between elements (default " ").</param>
        /// <param name="prefix">Used to align/wrap the output; its content is not included.</param>
        /// <param name="suffix">Used to wrap the output; its content is not included.</param>
        /// <param name="threshold">Total number of elements which trigger summarization.</param>
        /// <param name="edgeitems">Number of items at the beginning and end of each dimension in summary.</param>
        /// <param name="sign">'-', '+', or ' '.</param>
        /// <param name="floatmode">One of "fixed", "unique", "maxprec", "maxprec_equal".</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.array2string.html</remarks>
        public static string array2string(NDArray a,
            int? max_line_width = null, int? precision = null, bool? suppress_small = null,
            string separator = " ", string prefix = "",
            int? threshold = null, int? edgeitems = null, char? sign = null,
            string floatmode = null, string suffix = "")
        {
            if (a is null) throw new ArgumentNullException(nameof(a));

            var opts = BuildOptions(max_line_width, precision, suppress_small, threshold,
                edgeitems, sign, floatmode);

            // NumPy: for legacy <= 113 a 0d non-structured array returns repr(item) — not applicable
            // for the modern default. Empty arrays short-circuit to "[]".
            if (a.size == 0)
                return "[]";

            return ArrayFormatter.Array2String(a, opts, separator, prefix, suffix);
        }

        /// <summary>
        ///     Return a string representation of the data in an array (NumPy's <c>np.array_str</c>).
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.array_str.html</remarks>
        public static string array_str(NDArray a,
            int? max_line_width = null, int? precision = null, bool? suppress_small = null)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            var opts = BuildOptions(max_line_width, precision, suppress_small, null, null, null, null);
            return ArrayFormatter.ArrayStr(a, opts);
        }

        /// <summary>
        ///     Return the string representation of an array (NumPy's <c>np.array_repr</c>).
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.array_repr.html</remarks>
        public static string array_repr(NDArray a,
            int? max_line_width = null, int? precision = null, bool? suppress_small = null)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            var opts = BuildOptions(max_line_width, precision, suppress_small, null, null, null, null);
            return ArrayFormatter.ArrayRepr(a, opts);
        }

        /// <summary>
        ///     Set printing options (NumPy's <c>np.set_printoptions</c>). These options determine the
        ///     way floating point numbers, arrays and other NumPy objects are displayed.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.set_printoptions.html</remarks>
        public static void set_printoptions(int? precision = null, int? threshold = null,
            int? edgeitems = null, int? linewidth = null, bool? suppress = null,
            string nanstr = null, string infstr = null, char? sign = null, string floatmode = null)
        {
            PrintOptions.ValidateFloatmode(floatmode);
            PrintOptions.ValidateSign(sign);

            var opts = PrintOptions.Current.Clone();
            if (precision != null)
            {
                if (precision.Value < 0) throw new ArgumentException("precision must be >= 0");
                opts.precision = precision.Value;
            }
            if (threshold != null) opts.threshold = threshold.Value;
            if (edgeitems != null) opts.edgeitems = edgeitems.Value;
            if (linewidth != null) opts.linewidth = linewidth.Value;
            if (suppress != null) opts.suppress = suppress.Value;
            if (nanstr != null) opts.nanstr = nanstr;
            if (infstr != null) opts.infstr = infstr;
            if (sign != null) opts.sign = sign.Value;
            if (floatmode != null) opts.floatmode = floatmode;

            PrintOptions.Current = opts;
        }

        /// <summary>
        ///     Return the current print options as a dictionary (NumPy's <c>np.get_printoptions</c>).
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.get_printoptions.html</remarks>
        public static IReadOnlyDictionary<string, object> get_printoptions()
        {
            var o = PrintOptions.Current;
            return new Dictionary<string, object>
            {
                ["precision"] = o.precision,
                ["threshold"] = o.threshold,
                ["edgeitems"] = o.edgeitems,
                ["linewidth"] = o.linewidth,
                ["suppress"] = o.suppress,
                ["nanstr"] = o.nanstr,
                ["infstr"] = o.infstr,
                ["sign"] = o.sign.ToString(),
                ["floatmode"] = o.floatmode,
                ["legacy"] = o.legacy == long.MaxValue ? (object)false : o.legacy,
            };
        }

        /// <summary>
        ///     Context manager for setting print options (NumPy's <c>np.printoptions</c>). Restores the
        ///     previous options when disposed.
        /// </summary>
        /// <example><code>using (np.printoptions(precision: 2, suppress: true)) { Console.WriteLine(arr); }</code></example>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.printoptions.html</remarks>
        public static IDisposable printoptions(int? precision = null, int? threshold = null,
            int? edgeitems = null, int? linewidth = null, bool? suppress = null,
            string nanstr = null, string infstr = null, char? sign = null, string floatmode = null)
        {
            var previous = PrintOptions.Current.Clone();
            set_printoptions(precision, threshold, edgeitems, linewidth, suppress, nanstr, infstr, sign, floatmode);
            return new PrintOptionsScope(previous);
        }

        /// <summary>
        ///     Formats a floating-point scalar in positional notation with NumPy's Dragon4 (NumPy's
        ///     <c>np.format_float_positional</c>): the shortest digits that identify the value by default, every digit
        ///     under control of the options.
        /// </summary>
        /// <param name="x">The value. A <see cref="Half"/> formats as <c>np.float16</c> and a <see cref="float"/> as
        ///     <c>np.float32</c>, each with its OWN shortest digits (<c>0.1f</c> prints <c>'0.1'</c>, not float64's
        ///     <c>'0.10000000149011612'</c>); a <see cref="double"/> as <c>np.float64</c>; anything else converts like
        ///     Python's <c>float(x)</c> to a float64 — a C# integer or <see cref="System.Numerics.BigInteger"/> (a Python
        ///     int, correctly rounded), a <see cref="bool"/>, a <see cref="char"/> (NumSharp's Char dtype, NumPy's
        ///     <c>np.uint16</c>: its number), a <see cref="decimal"/>, a 0-d <see cref="NDArray"/> (its element; a 0-d
        ///     text array parsed as a float literal).</param>
        /// <param name="precision">The most digits to print after the point (with <paramref name="fractional"/> false:
        ///     significant digits); null prints the digits <paramref name="unique"/> asks for. The last digit is rounded
        ///     (half to even on the exact value), and in unique mode a rounding that carries through 9s prints FEWER
        ///     digits (<c>1.99999</c> at precision 2 is <c>'2.'</c>).</param>
        /// <param name="unique">True: the shortest digits that read back as the value, cut at
        ///     <paramref name="precision"/>. False: the value's exact digits to <paramref name="precision"/>, which is then
        ///     required.</param>
        /// <param name="fractional">True: <paramref name="precision"/> and <paramref name="min_digits"/> count digits
        ///     after the point; false: all significant digits.</param>
        /// <param name="trim">Trailing zeros and point: <c>"k"</c> keeps them (and pads exact mode to the precision),
        ///     <c>"."</c> trims the zeros, <c>"0"</c> trims all but one zero after the point, <c>"-"</c> trims the zeros and
        ///     the point.</param>
        /// <param name="sign">Print '+' before a non-negative value.</param>
        /// <param name="pad_left">Pad with spaces on the left until the integer part (with its sign) has this many
        ///     characters.</param>
        /// <param name="pad_right">Pad with spaces on the right until the fraction has this many characters.</param>
        /// <param name="min_digits">The fewest digits to print in unique mode (zeros pad it); counted like
        ///     <paramref name="precision"/>.</param>
        /// <returns>The text; an infinity is <c>'inf'</c> / <c>'-inf'</c> (<c>'+inf'</c> with <paramref name="sign"/>),
        ///     a NaN <c>'nan'</c> (never signed), neither padded.</returns>
        /// <exception cref="ValueError">A negative <paramref name="precision"/> / <paramref name="pad_left"/> /
        ///     <paramref name="pad_right"/> / <paramref name="min_digits"/> (<c>precision must be &gt;= 0</c>, checked in
        ///     that order); <paramref name="fractional"/> false with precision 0 (<c>precision must be greater than 0 if
        ///     fractional=False</c>); <paramref name="min_digits"/> above a positive precision (<c>min_digits must be less
        ///     than or equal to precision</c>); a 0-d text array that spells no float.</exception>
        /// <exception cref="TypeError">A trim other than one of the four characters (NumPy's text, <c>str(trim)</c>
        ///     right-aligned in 100 characters); <paramref name="unique"/> false without a precision (<c>in non-unique
        ///     mode `precision` must be supplied</c>); an <paramref name="x"/> that is not a real number (<c>must be real
        ///     number, not str</c>; an array of one or more dimensions: <c>only 0-dimensional arrays can be converted to
        ///     Python scalars</c>) — raised in that order, after the ValueErrors.</exception>
        /// <exception cref="OverflowException">A Python int beyond the float range (<c>int too large to convert to
        ///     float</c>).</exception>
        /// <exception cref="RuntimeError">Zeros or padding beyond NumPy's 16384-character buffer (<c>Float formatting
        ///     result too large</c>).</exception>
        /// <remarks>
        ///     NumPy's <c>arrayprint.format_float_positional</c> and its C <c>dragon4_positional</c>, statement for statement
        ///     (<see cref="Dragon4"/>'s port of <c>dragon4.c</c>).
        ///     https://numpy.org/doc/stable/reference/generated/numpy.format_float_positional.html
        /// </remarks>
        public static string format_float_positional(object x, int? precision = null, bool unique = true,
            bool fractional = true, string trim = "k", bool sign = false, int? pad_left = null, int? pad_right = null,
            int? min_digits = null)
        {
            // arrayprint's Python checks, in its order.
            int p = NoneOrPositive(precision, "precision");
            int padLeft = NoneOrPositive(pad_left, "pad_left");
            int padRight = NoneOrPositive(pad_right, "pad_right");
            int minDigits = NoneOrPositive(min_digits, "min_digits");
            if (!fractional && p == 0)
                throw new ValueError("precision must be greater than 0 if fractional=False");
            if (minDigits > 0 && p > 0 && minDigits > p)
                throw new ValueError("min_digits must be less than or equal to precision");
            // dragon4_positional: the argument parser converts trim, then the non-unique check, then x.
            TrimMode trimMode = Dragon4.ParseTrimArgument(trim);
            if (!unique && p < 0)
                throw new TypeError("in non-unique mode `precision` must be supplied");
            Dragon4.ScalarArgument(x, out ulong bits, out FloatKind kind);
            return Dragon4.Print(bits, kind, new Dragon4Options(false, unique ? DigitMode.Unique : DigitMode.Exact,
                fractional ? CutoffMode.FractionLength : CutoffMode.TotalLength, p, minDigits, sign, trimMode,
                padLeft, padRight, -1));
        }

        /// <summary>
        ///     Formats a floating-point scalar in scientific notation with NumPy's Dragon4 (NumPy's
        ///     <c>np.format_float_scientific</c>): one digit before the point, the shortest digits that identify the value
        ///     after it by default, and an exponent of at least two digits.
        /// </summary>
        /// <param name="x">The value, as for <see cref="format_float_positional"/> (a <see cref="Half"/> /
        ///     <see cref="float"/> formats with its own width's digits).</param>
        /// <param name="precision">The most digits to print after the point; null prints the digits
        ///     <paramref name="unique"/> asks for. In unique mode a rounding that carries through 9s prints fewer digits
        ///     (<c>0.0995</c> at precision 1 is <c>'1.e-01'</c>).</param>
        /// <param name="unique">True: the shortest digits, cut at <paramref name="precision"/>; false: the exact digits to
        ///     <paramref name="precision"/>, which is then required.</param>
        /// <param name="trim">As for <see cref="format_float_positional"/>, except that <c>"-"</c> keeps the point when a
        ///     fraction digit was generated and trimmed away (<c>'1.e-01'</c>), dropping it only when none was
        ///     (<c>'1e+00'</c>).</param>
        /// <param name="sign">Print '+' before a non-negative value.</param>
        /// <param name="pad_left">Pad with spaces on the left until the integer part (one digit and the sign) has this
        ///     many characters.</param>
        /// <param name="exp_digits">The fewest exponent digits (default 2; at most 5 are honoured; 0 prints no exponent
        ///     digit for exponent 0: <c>'1.5e+'</c>).</param>
        /// <param name="min_digits">The fewest digits after the point in unique mode (zeros pad it).</param>
        /// <returns>The text; the special values as for <see cref="format_float_positional"/>.</returns>
        /// <exception cref="ValueError">A negative <paramref name="precision"/> / <paramref name="pad_left"/> /
        ///     <paramref name="exp_digits"/> / <paramref name="min_digits"/> (checked in that order);
        ///     <paramref name="min_digits"/> above a positive precision; a 0-d text array that spells no float.</exception>
        /// <exception cref="TypeError">A bad trim; <paramref name="unique"/> false without a precision; an
        ///     <paramref name="x"/> that is not a real number — in that order, after the ValueErrors.</exception>
        /// <exception cref="OverflowException">A Python int beyond the float range.</exception>
        /// <remarks>
        ///     NumPy's <c>arrayprint.format_float_scientific</c> and its C <c>dragon4_scientific</c>, statement for
        ///     statement. NumPy's zero padding does not count down its buffer, so a huge precision in exact mode returns a
        ///     text a few characters longer than NumPy's 16383-character limit — NumPy's own result, reproduced.
        ///     https://numpy.org/doc/stable/reference/generated/numpy.format_float_scientific.html
        /// </remarks>
        public static string format_float_scientific(object x, int? precision = null, bool unique = true,
            string trim = "k", bool sign = false, int? pad_left = null, int? exp_digits = null, int? min_digits = null)
        {
            int p = NoneOrPositive(precision, "precision");
            int padLeft = NoneOrPositive(pad_left, "pad_left");
            int expDigits = NoneOrPositive(exp_digits, "exp_digits");
            int minDigits = NoneOrPositive(min_digits, "min_digits");
            if (minDigits > 0 && p > 0 && minDigits > p)
                throw new ValueError("min_digits must be less than or equal to precision");
            TrimMode trimMode = Dragon4.ParseTrimArgument(trim);
            if (!unique && p < 0)
                throw new TypeError("in non-unique mode `precision` must be supplied");
            Dragon4.ScalarArgument(x, out ulong bits, out FloatKind kind);
            return Dragon4.Print(bits, kind, new Dragon4Options(true, unique ? DigitMode.Unique : DigitMode.Exact,
                CutoffMode.TotalLength, p, minDigits, sign, trimMode, padLeft, -1, expDigits));
        }

        /// <summary>
        ///     NumPy's <c>arrayprint._none_or_positive_arg</c>: a missing count is -1 (Dragon4's "none"), a negative one an
        ///     error.
        /// </summary>
        /// <param name="x">The count.</param>
        /// <param name="name">The parameter name the error names.</param>
        /// <returns>The count, or -1 when null.</returns>
        /// <exception cref="ValueError">A negative count (<c>{name} must be &gt;= 0</c>).</exception>
        private static int NoneOrPositive(int? x, string name)
        {
            if (x is null)
                return -1;
            if (x.Value < 0)
                throw new ValueError($"{name} must be >= 0");
            return x.Value;
        }

        private static PrintOptions BuildOptions(int? maxLineWidth, int? precision, bool? suppressSmall,
            int? threshold, int? edgeitems, char? sign, string floatmode)
        {
            PrintOptions.ValidateFloatmode(floatmode);
            PrintOptions.ValidateSign(sign);

            var opts = PrintOptions.Current.Clone();
            if (maxLineWidth != null) opts.linewidth = maxLineWidth.Value;
            if (precision != null) opts.precision = precision.Value;
            if (suppressSmall != null) opts.suppress = suppressSmall.Value;
            if (threshold != null) opts.threshold = threshold.Value;
            if (edgeitems != null) opts.edgeitems = edgeitems.Value;
            if (sign != null) opts.sign = sign.Value;
            if (floatmode != null) opts.floatmode = floatmode;
            return opts;
        }

        private sealed class PrintOptionsScope : IDisposable
        {
            private readonly PrintOptions _previous;
            public PrintOptionsScope(PrintOptions previous) => _previous = previous;
            public void Dispose() => PrintOptions.Current = _previous;
        }
    }
}
