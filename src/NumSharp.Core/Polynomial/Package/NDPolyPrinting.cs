using System;
using System.Globalization;
using NumSharp.Backends.Printing;
using NumSharp.Backends.Strings;

// =============================================================================
// NDPolyPrinting.cs — numpy.polynomial's print style and polyutils.format_float (plan U11 of numpy-polynomial.md,
// numpy-polynomial-remaining-compute.md §3.8: unit 8)
// =============================================================================
//
// NumPy 2.4.2 (numpy/polynomial/__init__.py, polyutils.py):
//
//   def set_default_printstyle(style):
//       if style not in ('unicode', 'ascii'):
//           raise ValueError(f"Unsupported format string '{style}'. Valid options are 'ascii' and 'unicode'")
//       _use_unicode = True
//       if style == 'ascii':
//           _use_unicode = False
//       ABCPolyBase._use_unicode = _use_unicode
//
//   def format_float(x, parens=False):
//       if not np.issubdtype(type(x), np.floating):
//           return str(x)
//       opts = np.get_printoptions()
//       if np.isnan(x):
//           return opts['nanstr']
//       elif np.isinf(x):
//           return opts['infstr']
//       exp_format = False
//       if x != 0:
//           a = absolute(x)
//           if a >= 1.e8 or a < 10**min(0, -(opts['precision'] - 1) // 2):
//               exp_format = True
//       trim, unique = '0', True
//       if opts['floatmode'] == 'fixed':
//           trim, unique = 'k', False
//       if exp_format:
//           s = dragon4_scientific(x, precision=opts['precision'], unique=unique, trim=trim, sign=opts['sign'] == '+')
//           if parens:
//               s = '(' + s + ')'
//       else:
//           s = dragon4_positional(x, precision=opts['precision'], fractional=True, unique=unique, trim=trim,
//                                  sign=opts['sign'] == '+')
//       return s
//
// The print style is the flag the six series classes (plan U10) read when they render; the classes do not exist yet,
// so set_default_printstyle only records it.
//
// =============================================================================

namespace NumSharp
{
    /// <summary>
    ///     numpy.polynomial's printing helpers: the package-wide print style (<c>set_default_printstyle</c>) and
    ///     <c>polyutils.format_float</c>, the coefficient formatter the series classes print with.
    /// </summary>
    internal static class NDPolyPrinting
    {
        /// <summary>
        ///     <c>ABCPolyBase._use_unicode</c>: whether the series classes print superscripts and subscripts (<c>·x²</c>,
        ///     <c>·T₁(x)</c>) or ASCII (<c>x**2</c>, <c>T_1(x)</c>). NumPy's default is <c>os.name != 'nt'</c>: ASCII on
        ///     Windows, Unicode elsewhere. One process-wide value, as NumPy's class attribute is (no per-thread scope).
        /// </summary>
        private static volatile bool s_useUnicode = !OperatingSystem.IsWindows();

        /// <summary>
        ///     The current print style of the series classes (true: Unicode). Read by the class printers; changed only by
        ///     <see cref="SetDefaultPrintStyle"/>.
        /// </summary>
        internal static bool UseUnicode => s_useUnicode;

        /// <summary>
        ///     NumPy's <c>numpy.polynomial.set_default_printstyle</c>: <c>style not in ('unicode', 'ascii')</c> under
        ///     Python's <c>==</c>, then the flag.
        /// </summary>
        /// <param name="style"><c>"unicode"</c> or <c>"ascii"</c> (case-sensitive), or an array holding one such text
        ///     (a <c>U</c> / <c>StringDType</c> array compares element by element, and its one-element result decides).</param>
        /// <exception cref="ValueError">Any other value, naming <c>str(style)</c>: <c>Unsupported format string 'ASCII'.
        ///     Valid options are 'ascii' and 'unicode'</c> (also for a number, None, bytes, a list — Python's <c>==</c>
        ///     with a str is False for them — and for an array whose elements are not those texts); an empty array or
        ///     one of two or more elements raises NumPy's truth-value ValueError first (<c>The truth value of an empty
        ///     array is ambiguous. …</c> / <c>The truth value of an array with more than one element is ambiguous.
        ///     …</c>).</exception>
        internal static void SetDefaultPrintStyle(object style)
        {
            // `style in ('unicode', 'ascii')`: Python's == with each tuple item in order, each result's truth value.
            if (!(PyEqualsText(style, "unicode") || PyEqualsText(style, "ascii")))
                throw new ValueError($"Unsupported format string '{PyStr(style)}'. Valid options are 'ascii' and 'unicode'");
            // `_use_unicode = True; if style == 'ascii': _use_unicode = False` — the second comparison is NumPy's too.
            s_useUnicode = !PyEqualsText(style, "ascii");
        }

        /// <summary>
        ///     The truth value of Python's <c>style == text</c> for a C# value standing in for a Python object: a
        ///     <see cref="string"/> compares as a str; an <see cref="NDArray"/> compares element by element (a <c>U</c> or
        ///     <c>StringDType</c> element as its text, every other element — numbers, bytes — as unequal) and the result's
        ///     truth value is taken; anything else is never equal to a str.
        /// </summary>
        /// <param name="style">The value.</param>
        /// <param name="text">The str it is compared with.</param>
        /// <returns>Whether Python would read the comparison as true.</returns>
        /// <exception cref="ValueError">An array result that has no truth value: empty, or of two or more elements.</exception>
        private static bool PyEqualsText(object style, string text)
        {
            switch (style)
            {
                case string s:
                    return string.Equals(s, text, StringComparison.Ordinal);
                case NDArray array:
                {
                    // bool(array == text): NumPy's truth-value rules for the elementwise result.
                    long size = array.size;
                    if (size == 0)
                        throw new ValueError("The truth value of an empty array is ambiguous. Use `array.size > 0` to check that an array is not empty.");
                    if (size > 1)
                        throw new ValueError("The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()");
                    // A text dtype compares its element's text; bytes (S) and numbers are never equal to a str.
                    char kind = array.dtype.kind;
                    if (kind != 'U' && kind != 'T')
                        return false;
                    return array.Storage.GetAtIndex(0) is string element && string.Equals(element, text, StringComparison.Ordinal);
                }
                default:
                    return false;
            }
        }

        /// <summary>
        ///     NumPy's <c>numpy.polynomial.polyutils.format_float</c> (see the file header): a non-floating value is
        ///     <c>str(x)</c>; a floating one prints under the current print options — <c>nanstr</c> / <c>infstr</c> for the
        ///     special values (an infinity's sign is DROPPED, as in NumPy), scientific notation from 1e8 up or below
        ///     <c>10**min(0, -(precision - 1) // 2)</c> (1e-4 at the default precision 8), else positional; the shortest
        ///     digits cut at <c>precision</c> with one zero kept after the point, or with <c>floatmode='fixed'</c> exactly
        ///     <c>precision</c> digits; a '+' with <c>sign='+'</c>.
        /// </summary>
        /// <param name="x">The value: a <see cref="double"/> (<c>np.float64</c> / a Python float), a <see cref="float"/>
        ///     (<c>np.float32</c>, printed with float32's digits), a <see cref="Half"/> (<c>np.float16</c>) or a
        ///     <see cref="decimal"/> (NumSharp's floating decimal, printed as the float64 it converts to); anything else
        ///     prints as Python's <c>str()</c> of the object it stands for (<c>'3'</c>, <c>'True'</c>, <c>'None'</c>,
        ///     <c>'(1+2j)'</c>, an array's <c>str()</c> — a 0-d array included, as NumPy's <c>type(x)</c> is
        ///     <c>ndarray</c> there).</param>
        /// <param name="parens">Wrap a scientific result in parentheses (<c>'(1.0e-05)'</c>); a positional one never.</param>
        /// <returns>The text.</returns>
        /// <exception cref="RuntimeError">A precision whose zeros do not fit NumPy's 16384-character buffer (fixed mode
        ///     only: <c>Float formatting result too large</c>).</exception>
        internal static string FormatFloat(object x, bool parens)
        {
            ulong bits;
            FloatKind kind;
            switch (x)
            {
                case double d:
                    bits = BitConverter.DoubleToUInt64Bits(d);
                    kind = FloatKind.Double;
                    break;
                case float f:
                    bits = BitConverter.SingleToUInt32Bits(f);
                    kind = FloatKind.Single;
                    break;
                case Half h:
                    bits = BitConverter.HalfToUInt16Bits(h);
                    kind = FloatKind.Half;
                    break;
                case decimal m:
                    // NumSharp's decimal is a floating scalar class (np.issubdtype(decimal, floating)) with no Dragon4
                    // width: it prints as its float64, like the array printer prints it.
                    bits = BitConverter.DoubleToUInt64Bits((double)m);
                    kind = FloatKind.Double;
                    break;
                default:
                    return PyStr(x);
            }

            PrintOptions opts = PrintOptions.Current;
            if (IsNaN(bits, kind))
                return opts.nanstr;
            if (IsInfinity(bits, kind))
                return opts.infstr;

            // x != 0, then |x| against 1.e8 and 10**k: Python literals, so NEP 50 compares them in x's own dtype (a
            // float16 sees 1.e8 as inf). k = min(0, -(precision - 1) // 2) with Python's floor division; 10**k is the
            // int 1 at k = 0, else the float pow(10, k).
            bool expFormat = false;
            if (!IsZero(bits, kind))
            {
                int k = Math.Min(0, FloorDiv(-(opts.precision - 1), 2));
                double threshold = k == 0 ? 1.0 : Math.Pow(10.0, k);
                switch (kind)
                {
                    case FloatKind.Double:
                    {
                        double a = Math.Abs(BitConverter.UInt64BitsToDouble(bits));
                        expFormat = a >= 1e8 || a < threshold;
                        break;
                    }
                    case FloatKind.Single:
                    {
                        float a = Math.Abs(BitConverter.UInt32BitsToSingle((uint)bits));
                        expFormat = a >= (float)1e8 || a < (float)threshold;
                        break;
                    }
                    default:
                    {
                        Half a = Half.Abs(BitConverter.UInt16BitsToHalf((ushort)bits));
                        expFormat = a >= (Half)1e8 || a < (Half)threshold;
                        break;
                    }
                }
            }

            TrimMode trim = TrimMode.LeaveOneZero;
            DigitMode digitMode = DigitMode.Unique;
            if (opts.floatmode == "fixed")
            {
                trim = TrimMode.Keep;
                digitMode = DigitMode.Exact;
            }
            bool sign = opts.sign == '+';

            if (expFormat)
            {
                string s = Dragon4.Print(bits, kind, new Dragon4Options(true, digitMode, CutoffMode.TotalLength,
                    opts.precision, -1, sign, trim, -1, -1, -1));
                return parens ? "(" + s + ")" : s;
            }
            return Dragon4.Print(bits, kind, new Dragon4Options(false, digitMode, CutoffMode.FractionLength,
                opts.precision, -1, sign, trim, -1, -1, -1));
        }

        /// <summary>
        ///     Python's <c>str()</c> of the object a C# value stands for, under the polynomial package's boundary map: a
        ///     <see cref="char"/> is <c>np.uint16</c> there (NumSharp's Char dtype is NumPy's uint16), so it prints as its
        ///     number (<c>'3'</c>); every other value prints as <see cref="NDPyObject.Str"/> prints it. (NDPyObject's own map
        ///     reads a char as a one-character str, the string stores' rule; a char nested in a list still prints that way.)
        /// </summary>
        /// <param name="x">The value.</param>
        /// <returns>The text.</returns>
        private static string PyStr(object x)
            => x is char c ? ((int)c).ToString(CultureInfo.InvariantCulture) : NDPyObject.Str(x);

        /// <summary>Python's integer floor division <c>a // b</c> (rounds toward negative infinity).</summary>
        /// <param name="a">The dividend.</param>
        /// <param name="b">The divisor (positive here).</param>
        /// <returns>floor(a / b).</returns>
        private static int FloorDiv(int a, int b)
        {
            int q = a / b;
            return (a % b != 0 && (a < 0) != (b < 0)) ? q - 1 : q;
        }

        /// <summary>Whether the bits encode a NaN of the width.</summary>
        /// <param name="bits">The bits.</param>
        /// <param name="kind">The width.</param>
        /// <returns>True for a NaN.</returns>
        private static bool IsNaN(ulong bits, FloatKind kind) => kind switch
        {
            FloatKind.Half => Half.IsNaN(BitConverter.UInt16BitsToHalf((ushort)bits)),
            FloatKind.Single => float.IsNaN(BitConverter.UInt32BitsToSingle((uint)bits)),
            _ => double.IsNaN(BitConverter.UInt64BitsToDouble(bits)),
        };

        /// <summary>Whether the bits encode an infinity of the width.</summary>
        /// <param name="bits">The bits.</param>
        /// <param name="kind">The width.</param>
        /// <returns>True for either infinity.</returns>
        private static bool IsInfinity(ulong bits, FloatKind kind) => kind switch
        {
            FloatKind.Half => Half.IsInfinity(BitConverter.UInt16BitsToHalf((ushort)bits)),
            FloatKind.Single => float.IsInfinity(BitConverter.UInt32BitsToSingle((uint)bits)),
            _ => double.IsInfinity(BitConverter.UInt64BitsToDouble(bits)),
        };

        /// <summary>Whether the bits encode a zero of either sign (<c>x != 0</c> is false for -0.0 too).</summary>
        /// <param name="bits">The bits.</param>
        /// <param name="kind">The width.</param>
        /// <returns>True for ±0.</returns>
        private static bool IsZero(ulong bits, FloatKind kind) => kind switch
        {
            FloatKind.Half => (bits & 0x7FFF) == 0,
            FloatKind.Single => (bits & 0x7FFFFFFF) == 0,
            _ => (bits & 0x7FFFFFFFFFFFFFFFul) == 0,
        };
    }
}
