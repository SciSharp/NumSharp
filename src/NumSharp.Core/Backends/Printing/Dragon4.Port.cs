using System;
using System.Numerics;
using NumSharp.Backends.Strings;
using NumSharp.Utilities;

// =============================================================================
// Dragon4.Port.cs — NumPy's Dragon4 printing, ported statement for statement
// (numpy/_core/src/multiarray/dragon4.c, NumPy 2.4.2)
// =============================================================================
//
// WHY A SECOND PATH. Dragon4.cs EMULATES NumPy's digits: Ryu's shortest digits for unique mode, and .NET's "F" / "E"
// formatting (correctly rounded, half to even) wherever a precision cuts them. That is exact for the parameter space the
// array printer uses (FloatingFormat: trim '.' / '0', or trim 'k' with unique=False), and it is fast, so the array
// printer keeps it. The scalar formatters (np.format_float_positional / np.format_float_scientific and
// numpy.polynomial.polyutils.format_float) take NumPy's WHOLE option set, where the emulation cannot know what NumPy's
// digit buffer holds: a precision cut that rounds up through 9s leaves FEWER digits in NumPy's buffer ('2.' for
// 1.99999 at precision 2, never '2.00'); fractional=False counts significant digits; min_digits beyond a precision of 0
// pads zeros after rounding; the scientific DptZeros mode keeps its point; and inf / nan have their own printer. Those
// answers follow from NumPy's code, so this file runs NumPy's code.
//
// WHAT IS PORTED, and how:
//   * Dragon4() — Burger & Dybvig's digit generation with NumPy's changes (the 0.69 exponent estimate, the clamp of the
//     estimate to a fractional cutoff, the post-loop carry, IEEE unbiased rounding through isEven) — on NumPy's own
//     in-place BigInt arithmetic with 64-bit blocks (Dragon4.BigInt.cs: GenerateDigitsBlocks, allocation-free). The same
//     loop on System.Numerics.BigInteger (GenerateDigitsReference) is kept as the independent reference the tests hold
//     the block arithmetic to.
//   * FormatPositional / FormatScientific / PrintInfNan — verbatim over a char buffer of NumPy's size (16384, its
//     Dragon4_Scratch.repr), including the buffer-limit truncations and the RuntimeError "Float formatting result too
//     large". One NumPy defect is reproduced on purpose: FormatScientific's trailing-zero loop does not count down its
//     buffer size, so a huge precision writes up to 6 characters past the 16384-character buffer (NumPy's text for
//     format_float_scientific(1.5, precision=20000, unique=False) is 16387 characters long). The scratch therefore
//     carries a slack beyond NumPy's size, so the port returns NumPy's text instead of failing.
//   * The IEEE binary16 / binary32 / binary64 decompositions (sign, special values, mantissa, exponent, the index of the
//     highest mantissa bit, and whether the margins are unequal: a power of two above the smallest normal).
//
// THE FAST PATH. Unique mode whose cutoffs do not bind — the value's shortest round-trip digits end at or before the
// cutoff and min_digits asks for no more — is Ryu's shortest digits (FloatDigits.Shortest, the digits NumPy's unique
// Dragon4 produces: the fact Dragon4.cs rests on); every other request runs the block loop. The test Dragon4PortTests
// holds the fast path, the block loop and the BigInteger reference to the same bytes over random and edge values of every
// width.
//
// =============================================================================

namespace NumSharp.Backends.Printing
{
    /// <summary>
    ///     NumPy's <c>DigitMode</c>: which digits Dragon4 generates.
    /// </summary>
    internal enum DigitMode
    {
        /// <summary>The shortest digits that identify the value uniquely (cut at the cutoff when it comes first) — <c>unique=True</c>.</summary>
        Unique,

        /// <summary>The value's exact digits, up to the cutoff (or until they run out) — <c>unique=False</c>.</summary>
        Exact,
    }

    /// <summary>
    ///     NumPy's <c>CutoffMode</c>: what a precision counts.
    /// </summary>
    internal enum CutoffMode
    {
        /// <summary>Significant digits (scientific notation, and positional notation with <c>fractional=False</c>).</summary>
        TotalLength,

        /// <summary>Digits past the decimal point (positional notation with <c>fractional=True</c>, NumPy's default).</summary>
        FractionLength,
    }

    /// <summary>
    ///     NumPy's <c>Dragon4_Options</c>: one formatting request. Every count uses -1 for "none", as NumPy's C layer does
    ///     (its Python wrappers map <c>None</c> to -1 and refuse other negatives before they get here).
    /// </summary>
    internal readonly struct Dragon4Options
    {
        /// <summary>Scientific notation (<c>d.ddde+XX</c>) instead of positional notation.</summary>
        public readonly bool Scientific;

        /// <summary>Unique (shortest) or exact digits.</summary>
        public readonly DigitMode DigitMode;

        /// <summary>What <see cref="Precision"/> and <see cref="MinDigits"/> count (scientific notation always counts
        ///     significant digits: NumPy passes <see cref="CutoffMode.TotalLength"/> there).</summary>
        public readonly CutoffMode CutoffMode;

        /// <summary>The most digits to print (-1: as many as the digit mode gives). Exact mode needs one.</summary>
        public readonly int Precision;

        /// <summary>The fewest digits to print (-1: no minimum); unique mode pads to it with zeros.</summary>
        public readonly int MinDigits;

        /// <summary>Print a '+' before a non-negative value (a negative one always gets '-').</summary>
        public readonly bool Sign;

        /// <summary>What to trim from the end: trailing zeros and / or the decimal point.</summary>
        public readonly TrimMode TrimMode;

        /// <summary>Pad with spaces on the left until the integer part (with its sign) is this wide (-1: no padding).</summary>
        public readonly int DigitsLeft;

        /// <summary>Pad with spaces on the right until the fraction is this wide (-1: no padding; positional only).</summary>
        public readonly int DigitsRight;

        /// <summary>The fewest exponent digits (-1: two; at most five; scientific only).</summary>
        public readonly int ExpDigits;

        /// <summary>Creates one request; every argument is the field of the same name.</summary>
        /// <param name="scientific">Scientific notation.</param>
        /// <param name="digitMode">Unique or exact digits.</param>
        /// <param name="cutoffMode">What the counts count.</param>
        /// <param name="precision">The most digits (-1 for none).</param>
        /// <param name="minDigits">The fewest digits (-1 for none).</param>
        /// <param name="sign">Print '+' for non-negative values.</param>
        /// <param name="trimMode">The trim mode.</param>
        /// <param name="digitsLeft">Left padding width (-1 for none).</param>
        /// <param name="digitsRight">Right padding width (-1 for none).</param>
        /// <param name="expDigits">The fewest exponent digits (-1 for two).</param>
        public Dragon4Options(bool scientific, DigitMode digitMode, CutoffMode cutoffMode, int precision, int minDigits,
            bool sign, TrimMode trimMode, int digitsLeft, int digitsRight, int expDigits)
        {
            Scientific = scientific;
            DigitMode = digitMode;
            CutoffMode = cutoffMode;
            Precision = precision;
            MinDigits = minDigits;
            Sign = sign;
            TrimMode = trimMode;
            DigitsLeft = digitsLeft;
            DigitsRight = digitsRight;
            ExpDigits = expDigits;
        }
    }

    internal static partial class Dragon4
    {
        /// <summary>
        ///     The size of NumPy's output buffer (<c>Dragon4_Scratch.repr</c>, 16384 characters with the terminating NUL):
        ///     it bounds the digits Dragon4 generates and the padding the formatters add, and decides where NumPy raises
        ///     <c>RuntimeError: Float formatting result too large</c>.
        /// </summary>
        internal const int ReprBufferSize = 16384;

        /// <summary>
        ///     Extra characters behind <see cref="ReprBufferSize"/> in the scratch: FormatScientific's zero padding does not
        ///     count down NumPy's buffer size, so its exponent can land up to 7 characters past the end ('e', a sign, five
        ///     digits) plus the NUL. NumPy writes them (an overrun of its static buffer); the slack lets the port return the
        ///     same text instead of failing.
        /// </summary>
        private const int ScratchSlack = 16;

        /// <summary>One character buffer per thread, reused by every call (NumPy's thread-local <c>_bigint_static</c>).</summary>
        [ThreadStatic]
        private static char[] t_repr;

        /// <summary>
        ///     Test hook: when true, every request on THIS thread runs the general digit loop (<see cref="GenerateDigitsBlocks"/>,
        ///     or <see cref="GenerateDigitsReference"/> under <see cref="UseReferenceDigits"/>), never the shortest-digits
        ///     fast path. <c>Dragon4PortTests</c> sets it to hold the paths to the same bytes; production code never sets
        ///     it. Thread-static so a test cannot change another thread's formatting.
        /// </summary>
        [ThreadStatic]
        internal static bool DisableShortestDigits;

        /// <summary>
        ///     Test counter: how many requests on THIS thread the shortest-digits fast path served — what proves the fast-path
        ///     equivalence test is not vacuous (a fast path that never ran would agree with the loop trivially).
        /// </summary>
        [ThreadStatic]
        internal static long ShortestDigitsRuns;

        /// <summary>
        ///     NumPy's <c>trimmode_converter</c>: the one-character trim argument of <c>dragon4_positional</c> /
        ///     <c>dragon4_scientific</c> as a <see cref="TrimMode"/>.
        /// </summary>
        /// <param name="trim">The argument: exactly one of <c>"k"</c> (keep), <c>"."</c>, <c>"0"</c>, <c>"-"</c>.</param>
        /// <returns>The trim mode.</returns>
        /// <exception cref="TypeError">Anything else — null (<c>None</c>), the empty string, a longer string, another
        ///     character — with NumPy's text, which right-aligns <c>str(trim)</c> in a 100-character field (its
        ///     <c>%100S</c>): <c>if supplied, trim must be 'k', '.', '0' or '-' found `          …x`</c>.</exception>
        internal static TrimMode ParseTrimArgument(string trim)
        {
            // NumPy tests the length in code points (PyUnicode_GetLength), then the first character.
            if (trim is not null && PyRepr.CodePointLength(trim) == 1)
            {
                switch (trim[0])
                {
                    case 'k': return TrimMode.Keep;
                    case '.': return TrimMode.Zeros;
                    case '0': return TrimMode.LeaveOneZero;
                    case '-': return TrimMode.DptZeros;
                }
            }
            string shown = trim ?? "None";
            int pad = 100 - PyRepr.CodePointLength(shown);
            throw new TypeError("if supplied, trim must be 'k', '.', '0' or '-' found `"
                                + (pad > 0 ? new string(' ', pad) : "") + shown + "`");
        }

        /// <summary>
        ///     The value <c>Dragon4_Positional(obj)</c> / <c>Dragon4_Scientific(obj)</c> format: a NumPy float scalar in
        ///     its OWN width (<c>PyArray_IsScalar</c>: Half, Float, Double), anything else through <c>PyFloat_AsDouble</c>
        ///     as a float64.
        /// </summary>
        /// <param name="x">The argument under the C# boundary map: a <see cref="Half"/> is <c>np.float16</c>, a
        ///     <see cref="float"/> <c>np.float32</c> (the only float32 scalar a C# caller can spell, so it formats with
        ///     float32's digits: <c>0.1f</c> prints <c>'0.1'</c>), a <see cref="double"/> <c>np.float64</c> or a Python
        ///     float (one format); a <see cref="decimal"/> (no NumPy analog) converts to float64; a <see cref="char"/>
        ///     (NumSharp's Char dtype, NumPy's <c>np.uint16</c>) converts as its number; a Python int (any C#
        ///     integer, <see cref="BigInteger"/>, a <see cref="bool"/>) converts correctly rounded; a 0-d
        ///     <see cref="NDArray"/> converts as <c>float(a)</c> does (a number element, or text parsed as a float
        ///     literal).</param>
        /// <param name="bits">Receives the bit pattern to format.</param>
        /// <param name="kind">Receives its width.</param>
        /// <exception cref="TypeError">Not a real number (CPython's <c>must be real number, not str</c> / <c>not
        ///     NoneType</c> / <c>not complex</c> / <c>not list</c> / <c>not tuple</c> / <c>not bytes</c>); an array of one
        ///     or more dimensions (<c>only 0-dimensional arrays can be converted to Python scalars</c>); a 0-d array whose
        ///     element <c>float()</c> refuses (<c>float() argument must be a string or a real number, not
        ///     'complex'</c>).</exception>
        /// <exception cref="OverflowException">A Python int beyond the float range (<c>int too large to convert to
        ///     float</c>).</exception>
        /// <exception cref="ValueError">A 0-d text array that spells no float (<c>could not convert string to float:
        ///     'x'</c>).</exception>
        internal static void ScalarArgument(object x, out ulong bits, out FloatKind kind)
        {
            switch (x)
            {
                case Half h:
                    bits = BitConverter.HalfToUInt16Bits(h);
                    kind = FloatKind.Half;
                    return;
                case float f:
                    bits = BitConverter.SingleToUInt32Bits(f);
                    kind = FloatKind.Single;
                    return;
                case double d:
                    bits = BitConverter.DoubleToUInt64Bits(d);
                    kind = FloatKind.Double;
                    return;
                case NDArray array:
                    bits = BitConverter.DoubleToUInt64Bits(ArrayFloat(array));
                    kind = FloatKind.Double;
                    return;
            }
            bits = BitConverter.DoubleToUInt64Bits(PyFloatAsDouble(x));
            kind = FloatKind.Double;
        }

        /// <summary>
        ///     CPython's <c>PyFloat_AsDouble</c> for a non-array argument: a real number as a double, a Python int correctly
        ///     rounded.
        /// </summary>
        /// <param name="x">The argument.</param>
        /// <returns>The value.</returns>
        /// <exception cref="TypeError"><c>must be real number, not {type}</c>.</exception>
        /// <exception cref="OverflowException">A Python int beyond the float range.</exception>
        private static double PyFloatAsDouble(object x)
        {
            switch (x)
            {
                case double d: return d;
                case float f: return f;
                case Half h: return (double)h;
                case decimal m: return Converts.ToDouble(m);
                // NumSharp's Char dtype is NumPy's uint16: a char is an np.uint16 scalar, whose __float__ is exact.
                case char c: return c;
            }
            if (PrintfFormatter.TryPyInt(x, out BigInteger big))
            {
                double d = PythonInt.ToDouble(big);
                if (double.IsInfinity(d))
                    throw new OverflowException("int too large to convert to float");
                return d;
            }
            throw new TypeError($"must be real number, not {NDPyObject.TypeName(x)}");
        }

        /// <summary>
        ///     NumPy 2.4's <c>array_float</c> (<c>float(a)</c>), which <c>PyFloat_AsDouble</c> calls for an ndarray: only a
        ///     0-d array converts, through its element's <c>float()</c>.
        /// </summary>
        /// <param name="array">The array.</param>
        /// <returns>The value.</returns>
        /// <exception cref="TypeError">An array of one or more dimensions; an element <c>float()</c> refuses.</exception>
        /// <exception cref="ValueError">Text that is no float literal.</exception>
        /// <exception cref="OverflowException">A Python int element beyond the float range.</exception>
        private static unsafe double ArrayFloat(NDArray array)
        {
            if (array.ndim != 0)
                throw new TypeError("only 0-dimensional arrays can be converted to Python scalars");
            // The element as NumPy's getitem object: a number, a str (U / StringDType) or bytes (S); NumSharp's Char is a
            // uint16, an int.
            object item = array.Storage.GetAtIndex(0);
            if (item is char c)
                item = (int)c;
            switch (item)
            {
                case double or float or Half or decimal:
                    return PyFloatAsDouble(item);
                case string s:
                {
                    byte* canon = stackalloc byte[16];
                    NDTextParse.ParseObject(s, NPTypeCode.Double, canon);
                    return *(double*)canon;
                }
                case byte[] b:
                {
                    byte* canon = stackalloc byte[16];
                    NDTextParse.ParseObject(b, NPTypeCode.Double, canon);
                    return *(double*)canon;
                }
            }
            if (PrintfFormatter.TryPyInt(item, out _))
                return PyFloatAsDouble(item);
            throw new TypeError($"float() argument must be a string or a real number, not '{NDPyObject.TypeName(item)}'");
        }

        /// <summary>
        ///     Formats one float with NumPy's Dragon4 — the text <c>Dragon4_Positional_*</c> / <c>Dragon4_Scientific_*</c>
        ///     return for these options.
        /// </summary>
        /// <param name="bits">The value's bit pattern, in the low 16 / 32 / 64 bits for <paramref name="kind"/>.</param>
        /// <param name="kind">The float width the bits encode: NumPy formats a float16 / float32 value with its OWN
        ///     digits (np.float16(0.1) prints '0.1', not float64's '0.0999755859375').</param>
        /// <param name="opt">The request.</param>
        /// <returns>The text: <c>inf</c> / <c>-inf</c> / <c>+inf</c> / <c>nan</c> for the special values (no padding:
        ///     NumPy's PrintInfNan), otherwise the formatted number.</returns>
        /// <exception cref="RuntimeError">The padding or the zeros NumPy adds do not fit its 16384-character buffer
        ///     (<c>Float formatting result too large</c>).</exception>
        internal static string Print(ulong bits, FloatKind kind, in Dragon4Options opt)
        {
            char[] buffer = t_repr ??= new char[ReprBufferSize + ScratchSlack];
            int length = kind switch
            {
                FloatKind.Half => PrintBinary16((ushort)bits, buffer, opt),
                FloatKind.Single => PrintBinary32((uint)bits, buffer, opt),
                _ => PrintBinary64(bits, buffer, opt),
            };
            return new string(buffer, 0, length);
        }

        /// <summary>NumPy's <c>Dragon4_PrintFloat_IEEE_binary16</c>: 1 sign bit, 5 exponent bits, 10 mantissa bits.</summary>
        /// <param name="val">The float16 bits.</param>
        /// <param name="buffer">The output scratch.</param>
        /// <param name="opt">The request.</param>
        /// <returns>The text length.</returns>
        private static int PrintBinary16(ushort val, char[] buffer, in Dragon4Options opt)
        {
            uint floatMantissa = val & 0x3FFu;
            uint floatExponent = (uint)(val >> 10) & 0x1Fu;
            uint floatSign = (uint)val >> 15;
            char signbit = floatSign != 0 ? '-' : opt.Sign ? '+' : '\0';
            if (floatExponent == 0x1Fu)
                return PrintInfNan(buffer, ReprBufferSize, floatMantissa, signbit);

            // normalised: (2^10 + mantissa) * 2^(exponent - 15 - 10); denormalised: mantissa * 2^(1 - 15 - 10).
            ulong mantissa;
            int exponent, mantissaBit;
            bool hasUnequalMargins;
            if (floatExponent != 0)
            {
                mantissa = (1u << 10) | floatMantissa;
                exponent = (int)floatExponent - 15 - 10;
                mantissaBit = 10;
                hasUnequalMargins = floatExponent != 1 && floatMantissa == 0;
            }
            else
            {
                mantissa = floatMantissa;
                exponent = 1 - 15 - 10;
                mantissaBit = LogBase2(mantissa);
                hasUnequalMargins = false;
            }
            return FormatFloatBits(buffer, mantissa, exponent, signbit, mantissaBit, hasUnequalMargins, FloatKind.Half, val, opt);
        }

        /// <summary>NumPy's <c>Dragon4_PrintFloat_IEEE_binary32</c>: 1 sign bit, 8 exponent bits, 23 mantissa bits.</summary>
        /// <param name="val">The float32 bits.</param>
        /// <param name="buffer">The output scratch.</param>
        /// <param name="opt">The request.</param>
        /// <returns>The text length.</returns>
        private static int PrintBinary32(uint val, char[] buffer, in Dragon4Options opt)
        {
            uint floatMantissa = val & 0x7FFFFFu;
            uint floatExponent = (val >> 23) & 0xFFu;
            uint floatSign = val >> 31;
            char signbit = floatSign != 0 ? '-' : opt.Sign ? '+' : '\0';
            if (floatExponent == 0xFFu)
                return PrintInfNan(buffer, ReprBufferSize, floatMantissa, signbit);

            ulong mantissa;
            int exponent, mantissaBit;
            bool hasUnequalMargins;
            if (floatExponent != 0)
            {
                mantissa = (1u << 23) | floatMantissa;
                exponent = (int)floatExponent - 127 - 23;
                mantissaBit = 23;
                hasUnequalMargins = floatExponent != 1 && floatMantissa == 0;
            }
            else
            {
                mantissa = floatMantissa;
                exponent = 1 - 127 - 23;
                mantissaBit = LogBase2(mantissa);
                hasUnequalMargins = false;
            }
            return FormatFloatBits(buffer, mantissa, exponent, signbit, mantissaBit, hasUnequalMargins, FloatKind.Single, val, opt);
        }

        /// <summary>NumPy's <c>Dragon4_PrintFloat_IEEE_binary64</c>: 1 sign bit, 11 exponent bits, 52 mantissa bits.</summary>
        /// <param name="val">The float64 bits.</param>
        /// <param name="buffer">The output scratch.</param>
        /// <param name="opt">The request.</param>
        /// <returns>The text length.</returns>
        private static int PrintBinary64(ulong val, char[] buffer, in Dragon4Options opt)
        {
            ulong floatMantissa = val & 0xFFFFFFFFFFFFFul;
            uint floatExponent = (uint)(val >> 52) & 0x7FFu;
            ulong floatSign = val >> 63;
            char signbit = floatSign != 0 ? '-' : opt.Sign ? '+' : '\0';
            if (floatExponent == 0x7FFu)
                return PrintInfNan(buffer, ReprBufferSize, floatMantissa, signbit);

            ulong mantissa;
            int exponent, mantissaBit;
            bool hasUnequalMargins;
            if (floatExponent != 0)
            {
                mantissa = (1ul << 52) | floatMantissa;
                exponent = (int)floatExponent - 1023 - 52;
                mantissaBit = 52;
                hasUnequalMargins = floatExponent != 1 && floatMantissa == 0;
            }
            else
            {
                mantissa = floatMantissa;
                exponent = 1 - 1023 - 52;
                mantissaBit = LogBase2(mantissa);
                hasUnequalMargins = false;
            }
            return FormatFloatBits(buffer, mantissa, exponent, signbit, mantissaBit, hasUnequalMargins, FloatKind.Double, val, opt);
        }

        /// <summary>
        ///     The index of the highest set bit (NumPy's <c>LogBase2_64</c>); 0 for 0 (NumPy's table answers 0 there, and a
        ///     zero mantissa never reaches the digit loop).
        /// </summary>
        /// <param name="val">The mantissa.</param>
        /// <returns>floor(log2(val)), or 0.</returns>
        private static int LogBase2(ulong val) => val == 0 ? 0 : 63 - BitOperations.LeadingZeroCount(val);

        /// <summary>
        ///     NumPy's <c>Format_floatbits</c>: hands a decomposed value to <see cref="FormatScientificPort"/> or
        ///     <see cref="FormatPositionalPort"/>.
        /// </summary>
        /// <param name="buffer">The output scratch.</param>
        /// <param name="mantissa">The integer significand.</param>
        /// <param name="exponent">The base-2 exponent: value = mantissa · 2^exponent.</param>
        /// <param name="signbit">'+', '-' or '\0'.</param>
        /// <param name="mantissaBit">The index of the highest set mantissa bit.</param>
        /// <param name="hasUnequalMargins">Whether the upper neighbour is twice as far as the lower one.</param>
        /// <param name="kind">The width (for the shortest-digits fast path).</param>
        /// <param name="bits">The original bits (for the shortest-digits fast path).</param>
        /// <param name="opt">The request.</param>
        /// <returns>The text length.</returns>
        private static int FormatFloatBits(char[] buffer, ulong mantissa, int exponent, char signbit, int mantissaBit,
            bool hasUnequalMargins, FloatKind kind, ulong bits, in Dragon4Options opt)
        {
            var value = new Dragon4Value(mantissa, exponent, mantissaBit, hasUnequalMargins, kind, bits);
            return opt.Scientific
                ? FormatScientificPort(buffer, ReprBufferSize, value, signbit, opt.DigitMode, opt.Precision, opt.MinDigits,
                    opt.TrimMode, opt.DigitsLeft, opt.ExpDigits)
                : FormatPositionalPort(buffer, ReprBufferSize, value, signbit, opt.DigitMode, opt.CutoffMode, opt.Precision,
                    opt.MinDigits, opt.TrimMode, opt.DigitsLeft, opt.DigitsRight);
        }

        /// <summary>
        ///     A decomposed finite value: <c>mantissa · 2^exponent</c>, with the facts Dragon4 needs about its neighbours,
        ///     plus the original width and bits for the shortest-digits fast path.
        /// </summary>
        private readonly struct Dragon4Value
        {
            /// <summary>The integer significand (0 for a zero).</summary>
            public readonly ulong Mantissa;

            /// <summary>The base-2 exponent.</summary>
            public readonly int Exponent;

            /// <summary>The index of the highest set bit of <see cref="Mantissa"/>.</summary>
            public readonly int MantissaBit;

            /// <summary>Whether the gap to the next value above is twice the gap below (a normal power of two).</summary>
            public readonly bool HasUnequalMargins;

            /// <summary>The float width.</summary>
            public readonly FloatKind Kind;

            /// <summary>The original bit pattern.</summary>
            public readonly ulong Bits;

            /// <summary>Stores the parts.</summary>
            /// <param name="mantissa">The significand.</param>
            /// <param name="exponent">The exponent.</param>
            /// <param name="mantissaBit">The highest mantissa bit.</param>
            /// <param name="hasUnequalMargins">Unequal margins.</param>
            /// <param name="kind">The width.</param>
            /// <param name="bits">The bits.</param>
            public Dragon4Value(ulong mantissa, int exponent, int mantissaBit, bool hasUnequalMargins, FloatKind kind, ulong bits)
            {
                Mantissa = mantissa;
                Exponent = exponent;
                MantissaBit = mantissaBit;
                HasUnequalMargins = hasUnequalMargins;
                Kind = kind;
                Bits = bits;
            }
        }

        /// <summary>
        ///     NumPy's <c>PrintInfNan</c>: <c>inf</c> with the sign character in front (a '+' only when the caller asked
        ///     for signs), or <c>nan</c> — never signed, whatever its sign bit and payload. No padding is applied.
        /// </summary>
        /// <param name="buffer">The output scratch.</param>
        /// <param name="bufferSize">NumPy's buffer size.</param>
        /// <param name="mantissa">The mantissa field (0 for an infinity).</param>
        /// <param name="signbit">'+', '-' or '\0'.</param>
        /// <returns>The text length.</returns>
        private static int PrintInfNan(char[] buffer, int bufferSize, ulong mantissa, char signbit)
        {
            int maxPrintLen = bufferSize - 1;
            int pos = 0;
            if (mantissa == 0)
            {
                if ((signbit == '+' || signbit == '-') && pos < maxPrintLen - 1)
                    buffer[pos++] = signbit;
                int printLen = Math.Min(3, maxPrintLen - pos);
                "inf".AsSpan(0, printLen).CopyTo(buffer.AsSpan(pos));
                return pos + printLen;
            }
            else
            {
                int printLen = Math.Min(3, maxPrintLen - pos);
                "nan".AsSpan(0, printLen).CopyTo(buffer.AsSpan(pos));
                return pos + printLen;
            }
        }

        /// <summary>
        ///     NumPy's <c>Dragon4</c>: writes the digits of the value into <paramref name="buffer"/> at
        ///     <paramref name="start"/> (no sign, no point) and returns how many; <paramref name="printExponent"/> receives
        ///     the base-10 exponent of the first digit.
        /// </summary>
        /// <param name="value">The decomposed value.</param>
        /// <param name="digitMode">Unique or exact digits.</param>
        /// <param name="cutoffMode">What the cutoffs count.</param>
        /// <param name="cutoffMax">Stop after this many digits (-1: no cutoff).</param>
        /// <param name="cutoffMin">Print at least this many digits (-1: no minimum).</param>
        /// <param name="buffer">The output scratch.</param>
        /// <param name="start">Where the digits start.</param>
        /// <param name="bufferSize">The most digits that may be written (NumPy's remaining buffer).</param>
        /// <param name="printExponent">Receives the exponent of the first digit.</param>
        /// <returns>The number of digits written (at least 1). A zero value is the single digit '0' with exponent 0.</returns>
        /// <remarks>
        ///     <para>
        ///     The digits are those of NumPy's loop exactly: unique mode stops at the first digit that identifies the value
        ///     (once <paramref name="cutoffMin"/> is reached) or at the cutoff, exact mode when the remainder is zero or at
        ///     the cutoff; the last digit is then rounded (towards the closer neighbour, a tie to the even digit — IEEE
        ///     unbiased rounding) and a carry through 9s REMOVES them instead of writing zeros (1.99 at 1 digit is "2", one
        ///     digit). That last fact is what the formatters' trim modes and padding then see.
        ///     </para>
        ///     <para>
        ///     The fast path returns Ryu's shortest digits when the request is unique mode and neither cutoff binds; it is the
        ///     same output (the property Dragon4.cs documents), measured against this loop by <c>Dragon4PortTests</c>.
        ///     Every other request runs NumPy's loop on NumPy's in-place block arithmetic (<see cref="GenerateDigitsBlocks"/>).
        ///     </para>
        /// </remarks>
        private static int GenerateDigits(in Dragon4Value value, DigitMode digitMode, CutoffMode cutoffMode,
            int cutoffMax, int cutoffMin, char[] buffer, int start, int bufferSize, out int printExponent)
        {
            // A zero mantissa is zero whatever the exponent.
            if (value.Mantissa == 0)
            {
                buffer[start] = '0';
                printExponent = 0;
                return 1;
            }

            if (digitMode == DigitMode.Unique && !DisableShortestDigits && TryShortestDigits(value, cutoffMode, cutoffMax,
                    cutoffMin, buffer, start, bufferSize, out int shortestLength, out printExponent))
            {
                ShortestDigitsRuns++;
                return shortestLength;
            }

            // NumPy's loop on its in-place block arithmetic (Dragon4.BigInt.cs); the System.Numerics reference only under
            // the test hook.
            return UseReferenceDigits
                ? GenerateDigitsReference(value, digitMode, cutoffMode, cutoffMax, cutoffMin, buffer, start, bufferSize, out printExponent)
                : GenerateDigitsBlocks(value, digitMode, cutoffMode, cutoffMax, cutoffMin, buffer, start, bufferSize, out printExponent);
        }

        /// <summary>
        ///     Ryu's shortest digits when they ARE Dragon4's unique digits for this request: the cutoff does not cut them
        ///     and the minimum does not ask for more. Computes NumPy's two cutoff exponents from the true first-digit
        ///     exponent and compares them with the last shortest digit's.
        /// </summary>
        /// <param name="value">The decomposed value (with its width and bits).</param>
        /// <param name="cutoffMode">What the cutoffs count.</param>
        /// <param name="cutoffMax">The cutoff (-1 for none).</param>
        /// <param name="cutoffMin">The minimum (-1 for none).</param>
        /// <param name="buffer">The output scratch.</param>
        /// <param name="start">Where the digits start.</param>
        /// <param name="bufferSize">The most digits that may be written.</param>
        /// <param name="length">Receives the digit count.</param>
        /// <param name="printExponent">Receives the first digit's exponent.</param>
        /// <returns>True when the shortest digits were written; false to run the exact algorithm.</returns>
        private static bool TryShortestDigits(in Dragon4Value value, CutoffMode cutoffMode, int cutoffMax, int cutoffMin,
            char[] buffer, int start, int bufferSize, out int length, out int printExponent)
        {
            length = 0;
            printExponent = 0;
            ulong mantissa;
            int exponent;
            switch (value.Kind)
            {
                case FloatKind.Half:
                    mantissa = FloatDigits.Shortest(BitConverter.UInt16BitsToHalf((ushort)(value.Bits & 0x7FFF)), out exponent);
                    break;
                case FloatKind.Single:
                    mantissa = FloatDigits.Shortest(BitConverter.UInt32BitsToSingle((uint)(value.Bits & 0x7FFFFFFF)), out exponent);
                    break;
                default:
                    mantissa = FloatDigits.Shortest(BitConverter.UInt64BitsToDouble(value.Bits & 0x7FFFFFFFFFFFFFFFul), out exponent);
                    break;
            }
            // Shortest's digits carry no leading or trailing zeros: the decimal value is mantissa · 10^exponent.
            Span<char> digits = stackalloc char[20];
            if (!mantissa.TryFormat(digits, out int count, default, System.Globalization.CultureInfo.InvariantCulture))
                return false;
            int firstExponent = exponent + count - 1;   // the exponent of the first digit
            int lastExponent = exponent;                 // the exponent of the last digit
            int digitExponent = firstExponent + 1;       // NumPy's digitExponent after its correction

            // NumPy clamps its estimate to a fractional cutoff; a clamped run starts at the cutoff and is not the shortest.
            if (cutoffMax >= 0 && cutoffMode == CutoffMode.FractionLength && digitExponent <= -cutoffMax)
                return false;

            int cutoffMaxExponent = digitExponent - bufferSize;
            if (cutoffMax >= 0)
            {
                int desired = cutoffMode == CutoffMode.TotalLength ? digitExponent - cutoffMax : -cutoffMax;
                if (desired > cutoffMaxExponent)
                    cutoffMaxExponent = desired;
            }
            int cutoffMinExponent = digitExponent;
            if (cutoffMin >= 0)
            {
                int desired = cutoffMode == CutoffMode.TotalLength ? digitExponent - cutoffMin : -cutoffMin;
                if (desired < cutoffMinExponent)
                    cutoffMinExponent = desired;
            }
            // The cutoff cuts the shortest digits (its exponent is above the last one), or the minimum asks for digits
            // below the last one: NumPy's loop generates other digits than the shortest — run it.
            if (cutoffMaxExponent > lastExponent || cutoffMinExponent < lastExponent)
                return false;

            digits.Slice(0, count).CopyTo(buffer.AsSpan(start));
            length = count;
            printExponent = firstExponent;
            return true;
        }

        /// <summary>
        ///     NumPy's <c>Dragon4</c> loop on <see cref="BigInteger"/>, statement for statement — the REFERENCE the block
        ///     arithmetic of <see cref="GenerateDigitsBlocks"/> is tested against (<see cref="UseReferenceDigits"/>):
        ///     an independent exact computation of the same digits. Not a production path: BigInteger allocates a new number
        ///     per operation, which made it 1.1-3x slower than NumPy's C on long digit runs.
        /// </summary>
        /// <remarks>
        ///     NumPy's own BigInt also normalizes the scale before each division (BigInt_DivideWithRemainder_MaxQuotient9
        ///     needs its top block in range); <see cref="BigInteger.DivRem(BigInteger, BigInteger, out BigInteger)"/> needs no
        ///     such step, and the shift scales every number by the same power of two, so the digits are the same without it.
        /// </remarks>
        /// <param name="value">The decomposed value (non-zero mantissa).</param>
        /// <param name="digitMode">Unique or exact digits.</param>
        /// <param name="cutoffMode">What the cutoffs count.</param>
        /// <param name="cutoffMax">The cutoff (-1 for none).</param>
        /// <param name="cutoffMin">The minimum (-1 for none).</param>
        /// <param name="buffer">The output scratch.</param>
        /// <param name="start">Where the digits start.</param>
        /// <param name="bufferSize">The most digits that may be written.</param>
        /// <param name="printExponent">Receives the first digit's exponent.</param>
        /// <returns>The digit count.</returns>
        private static int GenerateDigitsReference(in Dragon4Value value, DigitMode digitMode, CutoffMode cutoffMode, int cutoffMax,
            int cutoffMin, char[] buffer, int start, int bufferSize, out int printExponent)
        {
            int cur = start;
            int exponent = value.Exponent;
            BigInteger mantissa = value.Mantissa;
            bool isEven = (value.Mantissa & 1) == 0;

            // value = scaledValue / scale; the margins are half the distance to the neighbouring floats, on the same scale.
            // With unequal margins (a power of two) the upper one is twice the lower one, so everything is scaled by 4.
            BigInteger scaledValue, scale, scaledMarginLow;
            bool unequal = value.HasUnequalMargins;
            if (unequal)
            {
                if (exponent > 0)
                {
                    scaledValue = mantissa << (exponent + 2);
                    scale = 4;
                    scaledMarginLow = BigInteger.One << exponent;
                }
                else
                {
                    scaledValue = mantissa << 2;
                    scale = BigInteger.One << (-exponent + 2);
                    scaledMarginLow = 1;
                }
            }
            else
            {
                if (exponent > 0)
                {
                    scaledValue = mantissa << (exponent + 1);
                    scale = 2;
                    scaledMarginLow = BigInteger.One << exponent;
                }
                else
                {
                    scaledValue = mantissa << 1;
                    scale = BigInteger.One << (-exponent + 1);
                    scaledMarginLow = 1;
                }
            }
            // scaledMarginHigh is scaledMarginLow, or twice it with unequal margins (NumPy recomputes it after every change).
            BigInteger scaledMarginHigh = unequal ? scaledMarginLow << 1 : scaledMarginLow;

            // NumPy's estimate of floor(log10(v)) + 1, low by one at most (0.69 makes the cheaper correction more likely).
            const double log10_2 = 0.30102999566398119521373889472449;
            int digitExponent = (int)Math.Ceiling((double)(value.MantissaBit + exponent) * log10_2 - 0.69);

            // A fractional cutoff above the first digit: start at the cutoff (the digit there is then rounded).
            if (cutoffMax >= 0 && cutoffMode == CutoffMode.FractionLength && digitExponent <= -cutoffMax)
                digitExponent = -cutoffMax + 1;

            // Divide the value by 10^digitExponent (a positive exponent multiplies the scale, a negative one the value and
            // the margins).
            if (digitExponent > 0)
            {
                scale *= BigInteger.Pow(10, digitExponent);
            }
            else if (digitExponent < 0)
            {
                BigInteger pow10 = BigInteger.Pow(10, -digitExponent);
                scaledValue *= pow10;
                scaledMarginLow *= pow10;
                scaledMarginHigh = unequal ? scaledMarginLow << 1 : scaledMarginLow;
            }

            // value >= 1: the estimate was low — take the next exponent without the first multiplication.
            if (scaledValue >= scale)
            {
                digitExponent++;
            }
            else
            {
                scaledValue *= 10;
                scaledMarginLow *= 10;
                scaledMarginHigh = unequal ? scaledMarginLow << 1 : scaledMarginLow;
            }

            // The exponent of the last digit to print (the buffer bounds it), and of the last forced digit.
            int cutoffMaxExponent = digitExponent - bufferSize;
            if (cutoffMax >= 0)
            {
                int desired = cutoffMode == CutoffMode.TotalLength ? digitExponent - cutoffMax : -cutoffMax;
                if (desired > cutoffMaxExponent)
                    cutoffMaxExponent = desired;
            }
            int cutoffMinExponent = digitExponent;
            if (cutoffMin >= 0)
            {
                int desired = cutoffMode == CutoffMode.TotalLength ? digitExponent - cutoffMin : -cutoffMin;
                if (desired < cutoffMinExponent)
                    cutoffMinExponent = desired;
            }

            printExponent = digitExponent - 1;

            bool low, high;
            int outputDigit;
            if (digitMode == DigitMode.Unique)
            {
                // Print until the digits distinguish the value from its neighbours (and the minimum is reached), or until
                // the cutoff.
                for (;;)
                {
                    digitExponent--;
                    outputDigit = (int)BigInteger.DivRem(scaledValue, scale, out scaledValue);
                    BigInteger scaledValueHigh = scaledValue + scaledMarginHigh;
                    int cmp = scaledValue.CompareTo(scaledMarginLow);
                    low = isEven ? cmp <= 0 : cmp < 0;
                    cmp = scaledValueHigh.CompareTo(scale);
                    high = isEven ? cmp >= 0 : cmp > 0;
                    if (((low | high) & (digitExponent <= cutoffMinExponent)) | (digitExponent == cutoffMaxExponent))
                        break;
                    buffer[cur++] = (char)('0' + outputDigit);
                    scaledValue *= 10;
                    scaledMarginLow *= 10;
                    scaledMarginHigh = unequal ? scaledMarginLow << 1 : scaledMarginLow;
                }
            }
            else
            {
                // Print until the remaining value is zero or until the cutoff.
                low = false;
                high = false;
                for (;;)
                {
                    digitExponent--;
                    outputDigit = (int)BigInteger.DivRem(scaledValue, scale, out scaledValue);
                    if (scaledValue.IsZero | (digitExponent == cutoffMaxExponent))
                        break;
                    buffer[cur++] = (char)('0' + outputDigit);
                    scaledValue *= 10;
                }
            }

            // Round the last digit: down when only the lower neighbour is close, else by comparing the remainder with half
            // a digit (2 * remainder against the scale), a tie to the even digit.
            bool roundDown = low;
            if (low == high)
            {
                scaledValue <<= 1;
                int compare = scaledValue.CompareTo(scale);
                roundDown = compare < 0;
                if (compare == 0)
                    roundDown = (outputDigit & 1) == 0;
            }

            if (roundDown)
            {
                buffer[cur++] = (char)('0' + outputDigit);
            }
            else if (outputDigit == 9)
            {
                // The carry: drop trailing 9s and increment the first other digit, or print a 1 one exponent higher.
                for (;;)
                {
                    if (cur == start)
                    {
                        buffer[cur++] = '1';
                        printExponent += 1;
                        break;
                    }
                    --cur;
                    if (buffer[cur] != '9')
                    {
                        buffer[cur] += (char)1;
                        ++cur;
                        break;
                    }
                }
            }
            else
            {
                buffer[cur++] = (char)('0' + outputDigit + 1);
            }

            return cur - start;
        }

        /// <summary>
        ///     NumPy's <c>FormatPositional</c>: <c>[sign]ddd.ddd</c> into the buffer, with NumPy's trimming and padding.
        /// </summary>
        /// <param name="buffer">The output scratch (at least <paramref name="bufferSize"/> characters).</param>
        /// <param name="bufferSize">NumPy's buffer size (the text is at most one less).</param>
        /// <param name="value">The decomposed value.</param>
        /// <param name="signbit">'+', '-' or '\0'.</param>
        /// <param name="digitMode">Unique or exact digits.</param>
        /// <param name="cutoffMode">What precision counts.</param>
        /// <param name="precision">The precision (-1 for none).</param>
        /// <param name="minDigits">The minimum digit count (-1 for none).</param>
        /// <param name="trimMode">The trim mode.</param>
        /// <param name="digitsLeft">Left padding width (-1 for none).</param>
        /// <param name="digitsRight">Right padding width (-1 for none).</param>
        /// <returns>The text length.</returns>
        /// <exception cref="RuntimeError">Zeros or padding beyond the buffer (<c>Float formatting result too large</c>).</exception>
        private static int FormatPositionalPort(char[] buffer, int bufferSize, in Dragon4Value value, char signbit,
            DigitMode digitMode, CutoffMode cutoffMode, int precision, int minDigits, TrimMode trimMode, int digitsLeft,
            int digitsRight)
        {
            int numWholeDigits = 0, hasSign = 0;
            int maxPrintLen = bufferSize - 1, pos = 0;
            int numFractionDigits = 0;

            if (signbit == '+' && pos < maxPrintLen)
            {
                buffer[pos++] = '+';
                hasSign = 1;
            }
            else if (signbit == '-' && pos < maxPrintLen)
            {
                buffer[pos++] = '-';
                hasSign = 1;
            }

            int numDigits = GenerateDigits(value, digitMode, cutoffMode, precision, minDigits, buffer, hasSign,
                maxPrintLen - hasSign, out int printExponent);

            if (printExponent >= 0)
            {
                // A whole part: the digits stay at the start of the buffer.
                numWholeDigits = printExponent + 1;
                if (numDigits <= numWholeDigits)
                {
                    // Zeros up to the decimal point.
                    int count = numWholeDigits - numDigits;
                    pos += numDigits;
                    if (count > maxPrintLen - pos)
                        throw new RuntimeError("Float formatting result too large");
                    numDigits += count;
                    for (; count > 0; count--)
                        buffer[pos++] = '0';
                }
                else
                {
                    // Insert the decimal point before the fraction.
                    numFractionDigits = numDigits - numWholeDigits;
                    int maxFractionDigits = maxPrintLen - numWholeDigits - 1 - pos;
                    if (numFractionDigits > maxFractionDigits)
                        numFractionDigits = maxFractionDigits;
                    Array.Copy(buffer, pos + numWholeDigits, buffer, pos + numWholeDigits + 1, numFractionDigits);
                    pos += numWholeDigits;
                    buffer[pos] = '.';
                    numDigits = numWholeDigits + 1 + numFractionDigits;
                    pos += 1 + numFractionDigits;
                }
            }
            else
            {
                // No whole part: "0." and the leading zeros of the fraction go before the digits.
                int numFractionZeros;
                if (pos + 2 < maxPrintLen)
                {
                    int maxFractionZeros = maxPrintLen - 2 - pos;
                    numFractionZeros = -(printExponent + 1);
                    if (numFractionZeros > maxFractionZeros)
                        numFractionZeros = maxFractionZeros;
                    int digitsStartIdx = 2 + numFractionZeros;
                    numFractionDigits = numDigits;
                    int maxFractionDigits = maxPrintLen - digitsStartIdx - pos;
                    if (numFractionDigits > maxFractionDigits)
                        numFractionDigits = maxFractionDigits;
                    Array.Copy(buffer, pos, buffer, pos + digitsStartIdx, numFractionDigits);
                    for (int i = 2; i < digitsStartIdx; ++i)
                        buffer[pos + i] = '0';
                    numFractionDigits += numFractionZeros;
                    numDigits = numFractionDigits;
                }
                if (pos + 1 < maxPrintLen)
                    buffer[pos + 1] = '.';
                if (pos < maxPrintLen)
                {
                    buffer[pos] = '0';
                    numDigits += 1;
                }
                numWholeDigits = 1;
                pos += 2 + numFractionDigits;
            }

            // Always a decimal point, except in DptZeros mode.
            if (trimMode != TrimMode.DptZeros && numFractionDigits == 0 && pos < maxPrintLen)
                buffer[pos++] = '.';

            int addDigits = digitMode == DigitMode.Unique ? minDigits : precision;
            int desiredFractionalDigits = addDigits < 0 ? 0 : addDigits;
            if (cutoffMode == CutoffMode.TotalLength)
                desiredFractionalDigits = addDigits - numWholeDigits;

            if (trimMode == TrimMode.LeaveOneZero)
            {
                // No fraction digit printed: add one zero.
                if (numFractionDigits == 0 && pos < maxPrintLen)
                {
                    buffer[pos++] = '0';
                    numFractionDigits++;
                }
            }
            else if (trimMode == TrimMode.Keep && desiredFractionalDigits > numFractionDigits && pos < maxPrintLen)
            {
                // Trailing zeros up to the requested digit count.
                int count = desiredFractionalDigits - numFractionDigits;
                if (count > maxPrintLen - pos)
                    throw new RuntimeError("Float formatting result too large");
                numFractionDigits += count;
                for (; count > 0; count--)
                    buffer[pos++] = '0';
            }

            // Rounding can still leave trailing zeros: remove them unless trimming is off.
            if (trimMode != TrimMode.Keep && numFractionDigits > 0)
            {
                while (buffer[pos - 1] == '0')
                {
                    pos--;
                    numFractionDigits--;
                }
                if (buffer[pos - 1] == '.')
                {
                    if (trimMode == TrimMode.LeaveOneZero)
                    {
                        buffer[pos++] = '0';
                        numFractionDigits++;
                    }
                    else if (trimMode == TrimMode.DptZeros)
                    {
                        pos--;
                    }
                }
            }

            // Right padding (a space stands in for a trimmed point).
            if (digitsRight >= numFractionDigits)
            {
                int count = digitsRight - numFractionDigits;
                if (trimMode == TrimMode.DptZeros && numFractionDigits == 0 && pos < maxPrintLen)
                    buffer[pos++] = ' ';
                if (count > maxPrintLen - pos)
                    throw new RuntimeError("Float formatting result too large");
                for (; count > 0; count--)
                    buffer[pos++] = ' ';
            }

            // Left padding: shift the text right.
            if (digitsLeft > numWholeDigits + hasSign)
            {
                int shift = digitsLeft - (numWholeDigits + hasSign);
                int count = pos;
                if (count > maxPrintLen - shift)
                    throw new RuntimeError("Float formatting result too large");
                if (count > 0)
                    Array.Copy(buffer, 0, buffer, shift, count);
                pos = shift + count;
                for (; shift > 0; shift--)
                    buffer[shift - 1] = ' ';
            }

            return pos;
        }

        /// <summary>
        ///     NumPy's <c>FormatScientific</c>: <c>[sign]d.ddde[sign]dd</c> into the buffer, with NumPy's trimming and
        ///     padding (including its uncounted zero padding — see <see cref="ScratchSlack"/>).
        /// </summary>
        /// <param name="buffer">The output scratch.</param>
        /// <param name="bufferSize">NumPy's buffer size.</param>
        /// <param name="value">The decomposed value.</param>
        /// <param name="signbit">'+', '-' or '\0'.</param>
        /// <param name="digitMode">Unique or exact digits.</param>
        /// <param name="precision">Digits after the point (-1 for none).</param>
        /// <param name="minDigits">The minimum digits after the point (-1 for none).</param>
        /// <param name="trimMode">The trim mode.</param>
        /// <param name="digitsLeft">Left padding width (-1 for none).</param>
        /// <param name="expDigits">The fewest exponent digits (-1 for two; capped at five).</param>
        /// <returns>The text length.</returns>
        private static int FormatScientificPort(char[] buffer, int bufferSize, in Dragon4Value value, char signbit,
            DigitMode digitMode, int precision, int minDigits, TrimMode trimMode, int digitsLeft, int expDigits)
        {
            int cur = 0;

            // Left padding first (the integer part is one digit plus the sign).
            int leftchars = 1 + (signbit == '-' || signbit == '+' ? 1 : 0);
            if (digitsLeft > leftchars)
            {
                for (int i = 0; i < digitsLeft - leftchars && bufferSize > 1; i++)
                {
                    buffer[cur++] = ' ';
                    --bufferSize;
                }
            }

            if (signbit == '+' && bufferSize > 1)
            {
                buffer[cur++] = '+';
                --bufferSize;
            }
            else if (signbit == '-' && bufferSize > 1)
            {
                buffer[cur++] = '-';
                --bufferSize;
            }

            int numDigits = GenerateDigits(value, digitMode, CutoffMode.TotalLength,
                precision < 0 ? -1 : precision + 1, minDigits < 0 ? -1 : minDigits + 1,
                buffer, cur, bufferSize, out int printExponent);

            // Keep the first digit as the whole number.
            if (bufferSize > 1)
            {
                cur += 1;
                bufferSize -= 1;
            }

            // Insert the decimal point before the fraction.
            int numFractionDigits = numDigits - 1;
            if (numFractionDigits > 0 && bufferSize > 1)
            {
                int maxFractionDigits = bufferSize - 2;
                if (numFractionDigits > maxFractionDigits)
                    numFractionDigits = maxFractionDigits;
                Array.Copy(buffer, cur, buffer, cur + 1, numFractionDigits);
                buffer[cur] = '.';
                cur += 1 + numFractionDigits;
                bufferSize -= 1 + numFractionDigits;
            }

            // Always a decimal point, except in DptZeros mode.
            if (trimMode != TrimMode.DptZeros && numFractionDigits == 0 && bufferSize > 1)
            {
                buffer[cur++] = '.';
                --bufferSize;
            }

            int addDigits = digitMode == DigitMode.Unique ? minDigits : precision;
            addDigits = addDigits < 0 ? 0 : addDigits;
            if (trimMode == TrimMode.LeaveOneZero)
            {
                if (numFractionDigits == 0 && bufferSize > 1)
                {
                    buffer[cur++] = '0';
                    --bufferSize;
                    ++numFractionDigits;
                }
            }
            else if (trimMode == TrimMode.Keep)
            {
                if (addDigits > numFractionDigits)
                {
                    // NumPy caps the zeros at the buffer but does not count them down from it (its defect, kept: the
                    // exponent then lands behind them, which ScratchSlack absorbs).
                    int numZeros = addDigits - numFractionDigits;
                    if (numZeros > bufferSize - 1)
                        numZeros = bufferSize - 1;
                    for (int end = cur + numZeros; cur < end; ++cur)
                    {
                        buffer[cur] = '0';
                        ++numFractionDigits;
                    }
                }
            }

            // Rounding can still leave trailing zeros: remove them unless trimming is off (the point stays, even in
            // DptZeros mode: NumPy's scientific branch only removes it when no fraction digit was generated, above).
            if (trimMode != TrimMode.Keep && numFractionDigits > 0)
            {
                --cur;
                while (buffer[cur] == '0')
                {
                    --cur;
                    ++bufferSize;
                    --numFractionDigits;
                }
                if (trimMode == TrimMode.LeaveOneZero && buffer[cur] == '.')
                {
                    ++cur;
                    buffer[cur] = '0';
                    --bufferSize;
                    ++numFractionDigits;
                }
                ++cur;
            }

            // The exponent: 'e', its sign and at least expDigits digits (two by default, five at most).
            if (bufferSize > 1)
            {
                if (expDigits > 5)
                    expDigits = 5;
                if (expDigits < 0)
                    expDigits = 2;
                Span<char> exponentBuffer = stackalloc char[7];
                Span<int> digits = stackalloc int[5];
                exponentBuffer[0] = 'e';
                if (printExponent >= 0)
                {
                    exponentBuffer[1] = '+';
                }
                else
                {
                    exponentBuffer[1] = '-';
                    printExponent = -printExponent;
                }
                for (int i = 0; i < 5; i++)
                {
                    digits[i] = printExponent % 10;
                    printExponent /= 10;
                }
                // Count back over leading zeros, keeping at least expDigits digits.
                int expSize = 5;
                while (expSize > expDigits && digits[expSize - 1] == 0)
                    expSize--;
                for (int i = expSize; i > 0; i--)
                    exponentBuffer[2 + (expSize - i)] = (char)('0' + digits[i - 1]);
                int count = expSize + 2;
                if (count > bufferSize - 1)
                    count = bufferSize - 1;
                exponentBuffer.Slice(0, count).CopyTo(buffer.AsSpan(cur));
                cur += count;
            }

            return cur;
        }
    }
}
