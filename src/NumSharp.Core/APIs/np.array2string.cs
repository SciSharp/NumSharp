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
        ///     Format a floating-point scalar as a decimal string in positional notation
        ///     (NumPy's <c>np.format_float_positional</c>).
        /// </summary>
        public static string format_float_positional(double x, int? precision = null, bool unique = true,
            bool fractional = true, char trim = 'k', bool sign = false, int? pad_left = null, int? pad_right = null,
            int? min_digits = null)
        {
            int p = precision ?? -1;
            int md = min_digits ?? -1;
            if (!fractional && p == 0)
                throw new ArgumentException("precision must be greater than 0 if fractional=False");
            return Dragon4.FormatPositional(x, FloatKind.Double, p, md < 0 ? 0 : md, unique,
                Dragon4.ParseTrim(trim), sign, pad_left ?? -1, pad_right ?? -1);
        }

        /// <summary>
        ///     Format a floating-point scalar as a decimal string in scientific notation
        ///     (NumPy's <c>np.format_float_scientific</c>).
        /// </summary>
        public static string format_float_scientific(double x, int? precision = null, bool unique = true,
            char trim = 'k', bool sign = false, int? pad_left = null, int? exp_digits = null, int? min_digits = null)
        {
            int p = precision ?? -1;
            int md = min_digits ?? -1;
            return Dragon4.FormatScientific(x, FloatKind.Double, p, md < 0 ? 0 : md, unique,
                Dragon4.ParseTrim(trim), sign, pad_left ?? -1, exp_digits ?? -1);
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
