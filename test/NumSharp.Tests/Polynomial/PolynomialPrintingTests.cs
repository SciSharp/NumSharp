using System;
using System.Collections.Generic;
using System.Numerics;

namespace NumSharp.Tests.Polynomial
{
    /// <summary>
    ///     numpy.polynomial's printing helpers (plan <c>docs/plans/numpy-polynomial-remaining-compute.md</c> §3.8, unit 8):
    ///     <c>polyutils.format_float</c> — the coefficient formatter of the series classes — over every argument kind with a
    ///     C# spelling and every print option it reads, and <c>set_default_printstyle</c>'s acceptance rule (Python's
    ///     <c>==</c> against <c>'unicode'</c> / <c>'ascii'</c>, NumPy's ValueError texts) and its flag.
    /// </summary>
    /// <remarks>
    ///     Every expected text was printed by NumPy 2.4.2 for the same call (scratchpad <c>ff_probe1.py</c>; the
    ///     <c>polyprint</c> oracle tier holds the full matrix). <c>[DoNotParallelize]</c>: <c>set_default_printstyle</c>
    ///     changes one process-wide flag (NumPy's <c>ABCPolyBase._use_unicode</c> class attribute); each test that changes
    ///     it puts NumPy's default back. The print options themselves are scoped per async flow (<c>np.printoptions</c>).
    /// </remarks>
    [TestClass]
    [DoNotParallelize]
    public class PolynomialPrintingTests
    {
        /// <summary>NumPy's default print style: ASCII on Windows (<c>os.name == 'nt'</c>), Unicode elsewhere.</summary>
        private static readonly bool DefaultUnicode = !OperatingSystem.IsWindows();

        /// <summary>Formats a call's outcome as the probe scripts print NumPy's: the text, or the exception.</summary>
        /// <param name="call">The call.</param>
        /// <returns>The text, or <c>raise {NumPy's exception name}: {message}</c>.</returns>
        private static string Outcome(Func<string> call)
        {
            try
            {
                return call();
            }
            catch (Exception e)
            {
                return "raise " + (e is ValueError ? "ValueError" : e is TypeError ? "TypeError" : e.GetType().Name) + ": " + e.Message;
            }
        }

        /// <summary>
        ///     Every argument kind at the default print options, with and without <c>parens</c>: a C# double / float /
        ///     Half is np.float64 / np.float32 / np.float16 (each printed with its own digits); a Python int, bool, complex,
        ///     str, None, list or tuple prints as <c>str()</c>; an array — a 0-d one included, as NumPy's <c>type(x)</c> is
        ///     <c>ndarray</c> there — prints as its <c>str()</c>.
        /// </summary>
        [TestMethod]
        public void FormatFloat_ArgumentKinds_AreNumPys()
        {
            var u = np.polynomial.polyutils;
            var cases = new (string Label, object X, string Plain, string Parens)[]
            {
                ("0.1", 0.1, "0.1", "0.1"),
                ("-0.0", -0.0, "-0.0", "-0.0"),
                ("1e8", 1e8, "1.0e+08", "(1.0e+08)"),
                ("99999999.5", 99999999.5, "99999999.5", "99999999.5"),
                ("1e-4", 1e-4, "0.0001", "0.0001"),
                ("9.9999e-5", 9.9999e-5, "9.9999e-05", "(9.9999e-05)"),
                ("1e-5", 1e-5, "1.0e-05", "(1.0e-05)"),
                ("1.23456789e-3", 1.23456789e-3, "0.00123457", "0.00123457"),
                ("123456.123456789", 123456.123456789, "123456.12345679", "123456.12345679"),
                ("nan", double.NaN, "nan", "nan"),
                ("-nan", -double.NaN, "nan", "nan"),
                ("inf", double.PositiveInfinity, "inf", "inf"),
                // format_float returns infstr itself: the sign of an infinity is dropped.
                ("-inf", double.NegativeInfinity, "inf", "inf"),
                ("1.999999999", 1.999999999, "2.0", "2.0"),
                ("0.999999999", 0.999999999, "1.0", "1.0"),
                ("5e-324", 5e-324, "5.0e-324", "(5.0e-324)"),
                ("max", double.MaxValue, "1.79769313e+308", "(1.79769313e+308)"),
                ("1e300", 1e300, "1.0e+300", "(1.0e+300)"),
                ("1e-300", 1e-300, "1.0e-300", "(1.0e-300)"),
                ("12345678.9", 12345678.9, "12345678.9", "12345678.9"),
                ("f32 0.1", 0.1f, "0.1", "0.1"),
                ("f32 1e-5", 1e-5f, "1.0e-05", "(1.0e-05)"),
                ("f32 -inf", float.NegativeInfinity, "inf", "inf"),
                ("f16 0.1", (Half)0.1, "0.1", "0.1"),
                ("f16 nan", Half.NaN, "nan", "nan"),
                ("int 1", 1, "1", "1"),
                ("int 2**70", BigInteger.Pow(2, 70), "1180591620717411303424", "1180591620717411303424"),
                ("bool", true, "True", "True"),
                // A char is np.uint16 in the polynomial package's boundary map: str(np.uint16(3)).
                ("char", (char)3, "3", "3"),
                ("complex", new Complex(1, 2), "(1+2j)", "(1+2j)"),
                ("complex real", new Complex(1.5, 0), "(1.5+0j)", "(1.5+0j)"),
                ("str", "1.5", "1.5", "1.5"),
                ("None", null, "None", "None"),
                ("list", new object[] { 1.5 }, "[1.5]", "[1.5]"),
                ("tuple", ValueTuple.Create(1.5), "(1.5,)", "(1.5,)"),
                ("0-d float array", np.array(1.5), "1.5", "1.5"),
                ("0-d float array 0.30000000000000004", np.array(0.30000000000000004), "0.30000000000000004", "0.30000000000000004"),
                ("0-d int array", np.array(3L), "3", "3"),
                ("1-d float array", np.array(new[] { 1.5 }), "[1.5]", "[1.5]"),
            };
            var wrong = new List<string>();
            foreach (var (label, x, plain, parens) in cases)
            {
                string got = Outcome(() => u.format_float(x));
                if (got != plain)
                    wrong.Add($"{label}: NumPy '{plain}', NumSharp '{got}'");
                got = Outcome(() => u.format_float(x, parens: true));
                if (got != parens)
                    wrong.Add($"{label} parens: NumPy '{parens}', NumSharp '{got}'");
            }
            wrong.Should().BeEmpty();
        }

        /// <summary>
        ///     The print options format_float reads, as NumPy reads them: <c>precision</c> (the cutoff, and the threshold
        ///     <c>10**min(0, -(precision - 1) // 2)</c> below which it switches to scientific notation),
        ///     <c>floatmode='fixed'</c> (exact digits, nothing trimmed; every other mode prints the shortest digits),
        ///     <c>sign='+'</c> (a ' ' sign is ignored), <c>nanstr</c> / <c>infstr</c>; <c>suppress</c> is ignored.
        /// </summary>
        [TestMethod]
        public void FormatFloat_PrintOptions_AreNumPys()
        {
            var u = np.polynomial.polyutils;
            var cases = new (string Label, Func<IDisposable> Options, object X, bool Parens, string NumPy)[]
            {
                ("p3 1.23456789123", () => np.printoptions(precision: 3), 1.23456789123, false, "1.235"),
                ("p3 0.000123456", () => np.printoptions(precision: 3), 0.000123456, false, "1.235e-04"),
                ("p3 0.000123456 parens", () => np.printoptions(precision: 3), 0.000123456, true, "(1.235e-04)"),
                ("p3 0.0099999", () => np.printoptions(precision: 3), 0.0099999, false, "1.0e-02"),
                ("p3 1.999999", () => np.printoptions(precision: 3), 1.999999, false, "2.0"),
                ("p3 f32 0.1", () => np.printoptions(precision: 3), 0.1f, false, "0.1"),
                ("p0 0.1", () => np.printoptions(precision: 0), 0.1, false, "1.0e-01"),
                ("p0 -2.5", () => np.printoptions(precision: 0), -2.5, false, "-2.0"),
                ("p0 1.23456789123", () => np.printoptions(precision: 0), 1.23456789123, false, "1.0"),
                ("p0 12.0", () => np.printoptions(precision: 0), 12.0, false, "12.0"),
                ("p0 0.0", () => np.printoptions(precision: 0), 0.0, false, "0.0"),
                ("p1 0.1", () => np.printoptions(precision: 1), 0.1, true, "(1.0e-01)"),
                ("p2 0.1", () => np.printoptions(precision: 2), 0.1, false, "0.1"),
                ("p16 1e-5", () => np.printoptions(precision: 16), 1e-5, false, "0.00001"),
                ("p16 1.23456789123", () => np.printoptions(precision: 16), 1.23456789123, false, "1.23456789123"),
                ("p20 0.0099999", () => np.printoptions(precision: 20), 0.0099999, false, "0.0099999"),
                ("fixed 0.1", () => np.printoptions(floatmode: "fixed"), 0.1, false, "0.10000000"),
                ("fixed 1e-5", () => np.printoptions(floatmode: "fixed"), 1e-5, true, "(1.00000000e-05)"),
                ("fixed -0.0", () => np.printoptions(floatmode: "fixed"), -0.0, false, "-0.00000000"),
                ("fixed 1.999999", () => np.printoptions(floatmode: "fixed"), 1.999999, false, "1.99999900"),
                ("fixed 12.0", () => np.printoptions(floatmode: "fixed"), 12.0, false, "12.00000000"),
                ("fixed p3 1.999999", () => np.printoptions(floatmode: "fixed", precision: 3), 1.999999, false, "2.000"),
                ("fixed p3 0.0099999", () => np.printoptions(floatmode: "fixed", precision: 3), 0.0099999, true, "(1.000e-02)"),
                ("fixed p3 f32 0.1", () => np.printoptions(floatmode: "fixed", precision: 3), 0.1f, false, "0.100"),
                ("unique 1.23456789123", () => np.printoptions(floatmode: "unique"), 1.23456789123, false, "1.23456789"),
                ("maxprec_equal 0.000123456", () => np.printoptions(floatmode: "maxprec_equal"), 0.000123456, false, "0.00012346"),
                ("sign + 0.1", () => np.printoptions(sign: '+'), 0.1, false, "+0.1"),
                ("sign + 1e-5 parens", () => np.printoptions(sign: '+'), 1e-5, true, "(+1.0e-05)"),
                ("sign + 0.0", () => np.printoptions(sign: '+'), 0.0, false, "+0.0"),
                ("sign + -0.0", () => np.printoptions(sign: '+'), -0.0, false, "-0.0"),
                ("sign + nan", () => np.printoptions(sign: '+'), double.NaN, false, "nan"),
                ("sign space 0.1", () => np.printoptions(sign: ' '), 0.1, false, "0.1"),
                ("nanstr", () => np.printoptions(nanstr: "NaN", infstr: "INF"), double.NaN, false, "NaN"),
                ("infstr -inf", () => np.printoptions(nanstr: "NaN", infstr: "INF"), double.NegativeInfinity, true, "INF"),
                ("suppress 1e-5", () => np.printoptions(suppress: true), 1e-5, false, "1.0e-05"),
            };
            var wrong = new List<string>();
            foreach (var (label, options, x, parens, want) in cases)
            {
                string got;
                using (options())
                    got = Outcome(() => u.format_float(x, parens));
                if (got != want)
                    wrong.Add($"{label}: NumPy '{want}', NumSharp '{got}'");
            }
            wrong.Should().BeEmpty();
        }

        /// <summary>
        ///     The scientific threshold is compared in x's OWN dtype (NEP 50: the Python literals are weak): a float16 sees
        ///     <c>1.e8</c> as inf and never switches to scientific notation from above, and its 1e-4 threshold is the
        ///     float16 nearest 1e-4.
        /// </summary>
        [TestMethod]
        public void FormatFloat_Threshold_IsComparedInTheValuesDtype()
        {
            var u = np.polynomial.polyutils;
            u.format_float((Half)65504).Should().Be("65500.0");
            u.format_float(BitConverter.UInt16BitsToHalf(0x068e)).Should().Be("0.0001");     // float16(1e-4)
            u.format_float(BitConverter.UInt16BitsToHalf(0x068d)).Should().Be("9.996e-05");  // the float16 below it
            u.format_float(1e-4f).Should().Be("0.0001");
            u.format_float(MathF.BitDecrement(1e-4f)).Should().Be("9.999999e-05");
            // float64's value below 1e-4 has the shortest digits 9.999999999999999e-05, cut at 8 digits: a carry to 1.
            u.format_float(System.Math.BitDecrement(1e-4)).Should().Be("1.0e-04");
            u.format_float(1e8f).Should().Be("1.0e+08");
            u.format_float(MathF.BitDecrement(1e8f)).Should().Be("99999990.0");
        }

        /// <summary>
        ///     set_default_printstyle accepts exactly <c>'unicode'</c> and <c>'ascii'</c> under Python's <c>==</c> — a text
        ///     array of one element compares by its text — and raises NumPy's ValueError naming <c>str(style)</c> otherwise;
        ///     an empty array or one of two or more elements raises NumPy's truth-value ValueError.
        /// </summary>
        [TestMethod]
        public void SetDefaultPrintstyle_AcceptsWhatNumPyAccepts()
        {
            const string Valid = ". Valid options are 'ascii' and 'unicode'";
            var cases = new (string Label, Func<object> Style, string NumPy)[]
            {
                ("ascii", () => "ascii", "ok False"),
                ("unicode", () => "unicode", "ok True"),
                ("ASCII", () => "ASCII", "raise ValueError: Unsupported format string 'ASCII'" + Valid),
                ("empty str", () => "", "raise ValueError: Unsupported format string ''" + Valid),
                ("None", () => null, "raise ValueError: Unsupported format string 'None'" + Valid),
                ("int 1", () => 1, "raise ValueError: Unsupported format string '1'" + Valid),
                ("bytes", () => new byte[] { 97, 115, 99, 105, 105 }, "raise ValueError: Unsupported format string 'b'ascii''" + Valid),
                ("0-d U ascii", () => np.array("ascii"), "ok False"),
                ("1-d U ascii", () => np.array(new[] { "ascii" }), "ok False"),
                ("2-d U unicode", () => np.array(new[,] { { "unicode" } }), "ok True"),
                ("U of two", () => np.array(new[] { "ascii", "unicode" }), "raise ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()"),
                ("list", () => new object[] { "ascii" }, "raise ValueError: Unsupported format string '['ascii']'" + Valid),
                ("tuple", () => ValueTuple.Create("ascii"), "raise ValueError: Unsupported format string '('ascii',)'" + Valid),
                ("float", () => 1.5, "raise ValueError: Unsupported format string '1.5'" + Valid),
                ("bool", () => true, "raise ValueError: Unsupported format string 'True'" + Valid),
                ("char (np.uint16)", () => (char)3, "raise ValueError: Unsupported format string '3'" + Valid),
                ("empty array", () => np.array(new double[0]), "raise ValueError: The truth value of an empty array is ambiguous. Use `array.size > 0` to check that an array is not empty."),
                ("0-d int array", () => np.array(1L), "raise ValueError: Unsupported format string '1'" + Valid),
                ("int array of two", () => np.array(new long[] { 1, 2 }), "raise ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()"),
            };
            var wrong = new List<string>();
            try
            {
                foreach (var (label, style, want) in cases)
                {
                    // A refused style must leave the flag as it was: set it to the other value first.
                    np.polynomial.set_default_printstyle(want == "ok False" ? "unicode" : "ascii");
                    bool before = NDPolyPrinting.UseUnicode;
                    string got = Outcome(() =>
                    {
                        np.polynomial.set_default_printstyle(style());
                        return "ok " + (NDPolyPrinting.UseUnicode ? "True" : "False");
                    });
                    if (got != want)
                        wrong.Add($"{label}: NumPy '{want}', NumSharp '{got}'");
                    if (got.StartsWith("raise", StringComparison.Ordinal) && NDPolyPrinting.UseUnicode != before)
                        wrong.Add($"{label}: a refused style changed the flag");
                }
            }
            finally
            {
                np.polynomial.set_default_printstyle(DefaultUnicode ? "unicode" : "ascii");
            }
            wrong.Should().BeEmpty();
        }

        /// <summary>NumPy's default print style is the platform's: ASCII on Windows, Unicode elsewhere.</summary>
        [TestMethod]
        public void PrintStyle_DefaultsToThePlatforms()
        {
            NDPolyPrinting.UseUnicode.Should().Be(DefaultUnicode);
        }
    }
}
