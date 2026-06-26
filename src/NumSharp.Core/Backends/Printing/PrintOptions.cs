using System;
using System.Threading;

namespace NumSharp.Backends.Printing
{
    /// <summary>
    ///     Mirror of NumPy's print options (the <c>format_options</c> context in
    ///     <c>numpy/_core/printoptions.py</c>). Controls how an <see cref="NDArray"/>
    ///     is rendered by <c>array2string</c> / <c>array_str</c> / <c>array_repr</c>
    ///     and therefore by <see cref="NDArray.ToString()"/>.
    /// </summary>
    /// <remarks>
    ///     The defaults match NumPy 2.4.2 exactly. Note that the runtime default
    ///     <see cref="floatmode"/> is <c>"maxprec"</c> — the value documented as
    ///     <c>"maxprec_equal"</c> in <c>set_printoptions</c> is outdated.
    /// </remarks>
    public sealed class PrintOptions
    {
        /// <summary>Number of digits of precision for floating point output (default 8).</summary>
        public int precision = 8;

        /// <summary>Total number of array elements which trigger summarization (default 1000).</summary>
        public int threshold = 1000;

        /// <summary>Number of array items in summary at beginning and end of each dimension (default 3).</summary>
        public int edgeitems = 3;

        /// <summary>Number of characters per line for inserting line breaks (default 75).</summary>
        public int linewidth = 75;

        /// <summary>If true, always print floating point numbers using fixed point notation (default false).</summary>
        public bool suppress = false;

        /// <summary>String representation of floating point not-a-number (default "nan").</summary>
        public string nanstr = "nan";

        /// <summary>String representation of floating point infinity (default "inf").</summary>
        public string infstr = "inf";

        /// <summary>Controls printing of the sign of floating-point types: '-', '+', or ' ' (default '-').</summary>
        public char sign = '-';

        /// <summary>Controls the interpretation of <see cref="precision"/>: fixed, unique, maxprec, maxprec_equal (default maxprec).</summary>
        public string floatmode = "maxprec";

        /// <summary>Legacy print mode as an int. <see cref="long.MaxValue"/> (sys.maxsize) means no legacy mode.</summary>
        public long legacy = long.MaxValue;

        internal PrintOptions Clone() => (PrintOptions)MemberwiseClone();

        // NumPy stores print options in a ContextVar (per-thread/async-local). We mirror that
        // with AsyncLocal so set_printoptions has the same scoping semantics.
        private static readonly AsyncLocal<PrintOptions> _current = new AsyncLocal<PrintOptions>();

        internal static PrintOptions Current
        {
            get => _current.Value ??= new PrintOptions();
            set => _current.Value = value;
        }

        /// <summary>
        ///     Validate the floatmode/sign values exactly as NumPy's <c>_make_options_dict</c> does.
        /// </summary>
        internal static void ValidateFloatmode(string floatmode)
        {
            if (floatmode != null &&
                floatmode != "fixed" && floatmode != "unique" &&
                floatmode != "maxprec" && floatmode != "maxprec_equal")
            {
                throw new ArgumentException(
                    "floatmode option must be one of \"fixed\", \"unique\", \"maxprec\", \"maxprec_equal\"");
            }
        }

        internal static void ValidateSign(char? sign)
        {
            if (sign != null && sign != '-' && sign != '+' && sign != ' ')
                throw new ArgumentException("sign option must be one of ' ', '+', or '-'");
        }
    }
}
