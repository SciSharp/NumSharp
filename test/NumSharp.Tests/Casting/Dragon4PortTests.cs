using System;
using System.Collections.Generic;
using System.Numerics;
using NumSharp.Backends.Printing;

namespace NumSharp.Tests.Casting;

/// <summary>
///     <c>np.format_float_positional</c> / <c>np.format_float_scientific</c> as NumPy 2.4.2's <c>dragon4.c</c>
///     (<c>Backends/Printing/Dragon4.Port.cs</c>, <c>Dragon4.BigInt.cs</c>; plan
///     <c>docs/plans/numpy-polynomial-remaining-compute.md</c> §3.8): the three digit generators (the shortest-digits fast
///     path, NumPy's loop on its in-place block arithmetic, the System.Numerics reference) held to each other, NumPy's text
///     for the values and options that discriminate between digit generators (rounding at the cutoff, carries through 9s,
///     fractional=False, the fractional clamp, min_digits, every trim mode, padding, exponent digits, the three float
///     widths), and NumPy's argument handling (every argument kind with a C# spelling, the validation order, the buffer
///     limits).
/// </summary>
/// <remarks>
///     Every expected text was printed by NumPy 2.4.2 for the same call (scratchpad probes <c>ff_pins.py</c>,
///     <c>ff_probe2.py</c>, <c>ff_long.py</c> and <c>ff_zerostop.py</c>; the power-of-two pins are the cases
///     <c>test/oracle/gen_surface_oracle.py</c> finds by simulating NumPy's loop); the tests that pinned the old emulation's
///     defects (surface gaps SU6, SU8, SU9, SU18, SU19 of <c>OpenBugs.Surface.cs</c>) are among them. A mutation pass of 25
///     planted defects across the port, the argument checks and the polynomial helpers is killed by these tests and the
///     two oracle tiers together (the last three tests were added for the defects it found only there or nowhere).
/// </remarks>
[TestClass]
public class Dragon4PortTests
{
    /// <summary>Formats a call's outcome the way the probe scripts print NumPy's: the text, or the exception.</summary>
    /// <param name="call">The call.</param>
    /// <returns>The text, or <c>raise {NumPy's exception name}: {message}</c>.</returns>
    private static string Outcome(Func<object> call)
    {
        try
        {
            object r = call();
            return r is string s ? s : r?.ToString();
        }
        catch (Exception e)
        {
            string name = e switch
            {
                ValueError => "ValueError",
                TypeError => "TypeError",
                RuntimeError => "RuntimeError",
                OverflowException => "OverflowError",
                _ => e.GetType().Name,
            };
            return "raise " + name + ": " + e.Message;
        }
    }

    /// <summary>
    ///     The values and options that discriminate between digit generators, with NumPy's text (<c>ff_pins.py</c>).
    /// </summary>
    private static readonly (string Label, Func<object> Call, string NumPy)[] Pins =
    {
        // Each width formats with its OWN digits.
        ("f16 0.1", () => np.format_float_positional((Half)0.1), "0.1"),
        ("f16 65504", () => np.format_float_positional((Half)65504), "65500."),
        ("f16 min subnormal", () => np.format_float_positional(BitConverter.UInt16BitsToHalf(1)), "0.00000006"),
        ("f16 min subnormal sci", () => np.format_float_scientific(BitConverter.UInt16BitsToHalf(1)), "6.e-08"),
        ("f16 max subnormal sci", () => np.format_float_scientific(BitConverter.UInt16BitsToHalf(0x3ff)), "6.1e-05"),
        ("f16 min normal sci exact", () => np.format_float_scientific(BitConverter.UInt16BitsToHalf(0x400), precision: 20, unique: false), "6.10351562500000000000e-05"),
        ("f32 0.1", () => np.format_float_positional(0.1f), "0.1"),
        ("f32 0.1 exact", () => np.format_float_positional(0.1f, precision: 30, unique: false), "0.100000001490116119384765625000"),
        ("f32 16777217", () => np.format_float_positional(16777217f), "16777216."),
        ("f32 max sci", () => np.format_float_scientific(float.MaxValue), "3.4028235e+38"),
        ("f32 min subnormal sci", () => np.format_float_scientific(BitConverter.UInt32BitsToSingle(1)), "1.e-45"),
        ("f32 min normal pos", () => np.format_float_positional(BitConverter.UInt32BitsToSingle(0x00800000)), "0.000000000000000000000000000000000000011754944"),
        ("f64 0.1", () => np.format_float_positional(0.1), "0.1"),
        ("f64 0.1 exact", () => np.format_float_positional(0.1, precision: 60, unique: false), "0.100000000000000005551115123125782702118158340454101562500000"),
        ("f64 5e-324 sci", () => np.format_float_scientific(double.Epsilon), "5.e-324"),
        ("f64 5e-324 sci exact p0", () => np.format_float_scientific(double.Epsilon, precision: 0, unique: false), "5.e-324"),
        ("f64 max sci", () => np.format_float_scientific(double.MaxValue), "1.7976931348623157e+308"),
        ("f64 2**-1022 sci", () => np.format_float_scientific(BitConverter.UInt64BitsToDouble(0x0010000000000000)), "2.2250738585072014e-308"),
        ("f64 1e23", () => np.format_float_positional(1e23), "100000000000000000000000."),
        ("f64 9007199254740993", () => np.format_float_positional(9007199254740992.0), "9007199254740992."),
        // The cutoff: the last digit rounded (a tie to the even digit), and a carry through 9s leaves FEWER digits.
        ("p2 1.99999", () => np.format_float_positional(1.99999, precision: 2), "2."),
        ("p2 9.995", () => np.format_float_positional(9.995, precision: 2), "9.99"),
        ("p0 0.5", () => np.format_float_positional(0.5, precision: 0), "0."),
        ("p0 1.5", () => np.format_float_positional(1.5, precision: 0), "2."),
        ("p0 2.5", () => np.format_float_positional(2.5, precision: 0), "2."),
        ("p1 0.25", () => np.format_float_positional(0.25, precision: 1), "0.2"),
        ("p1 0.35", () => np.format_float_positional(0.35, precision: 1), "0.3"),
        ("p3 2.675", () => np.format_float_positional(2.675, precision: 2), "2.67"),
        ("p2 0.0995 exact", () => np.format_float_positional(0.0995, precision: 2, unique: false), "0.10"),
        ("p3 999.9996", () => np.format_float_positional(999.9996, precision: 3), "1000."),
        ("sci p1 0.0995", () => np.format_float_scientific(0.0995, precision: 1), "1.e-01"),
        ("sci p2 9.9999", () => np.format_float_scientific(9.9999, precision: 2), "1.e+01"),
        // Scientific DptZeros keeps the point when the cutoff left only zero fraction digits.
        ("sci p2 0.1001 trim -", () => np.format_float_scientific(0.1001, precision: 2, trim: "-"), "1.e-01"),
        ("sci p2 0.1001 trim 0", () => np.format_float_scientific(0.1001, precision: 2, trim: "0"), "1.0e-01"),
        ("sci p2 0.1001 trim .", () => np.format_float_scientific(0.1001, precision: 2, trim: "."), "1.e-01"),
        // fractional=False: the precision counts significant digits.
        ("frac False p3 123456789.125", () => np.format_float_positional(123456789.125, precision: 3, fractional: false), "123000000."),
        ("frac False p3 0.000123456", () => np.format_float_positional(0.000123456, precision: 3, fractional: false), "0.000123"),
        ("frac False p1 9.5", () => np.format_float_positional(9.5, precision: 1, fractional: false), "10."),
        ("frac False p2 exact 1e22", () => np.format_float_positional(1e22, precision: 2, fractional: false, unique: false), "10000000000000000000000."),
        // The fractional clamp: a first digit below the cutoff starts the loop AT the cutoff.
        ("p3 0.0004", () => np.format_float_positional(0.0004, precision: 3), "0.000"),
        ("p3 0.0006", () => np.format_float_positional(0.0006, precision: 3), "0.001"),
        ("p3 0.0005", () => np.format_float_positional(0.0005, precision: 3), "0.001"),
        ("p3 exact 0.0006", () => np.format_float_positional(0.0006, precision: 3, unique: false), "0.001"),
        ("p0 exact 1e-300", () => np.format_float_positional(1e-300, precision: 0, unique: false), "0."),
        // min_digits: unique mode generates digits past the shortest, then zeros.
        ("min5 1.5", () => np.format_float_positional(1.5, min_digits: 5), "1.50000"),
        ("min5 p0 1.5", () => np.format_float_positional(1.5, min_digits: 5, precision: 0), "2.00000"),
        ("min3 p5 0.1", () => np.format_float_positional(0.1, min_digits: 3, precision: 5), "0.100"),
        ("min20 0.1", () => np.format_float_positional(0.1, min_digits: 20), "0.10000000000000000555"),
        ("sci min4 2.0", () => np.format_float_scientific(2.0, min_digits: 4), "2.0000e+00"),
        ("min3 frac False", () => np.format_float_positional(12345.0, min_digits: 8, fractional: false), "12345.000"),
        // The trim modes, sign and padding.
        ("trim k 2.0", () => np.format_float_positional(2.0, trim: "k"), "2."),
        ("trim . 2.0", () => np.format_float_positional(2.0, trim: "."), "2."),
        ("trim 0 2.0", () => np.format_float_positional(2.0, trim: "0"), "2.0"),
        ("trim - 2.0", () => np.format_float_positional(2.0, trim: "-"), "2"),
        ("exact p4 trim k 2.5", () => np.format_float_positional(2.5, precision: 4, unique: false, trim: "k"), "2.5000"),
        ("exact p4 trim . 2.5", () => np.format_float_positional(2.5, precision: 4, unique: false, trim: "."), "2.5"),
        ("exact p4 trim 0 2.0", () => np.format_float_positional(2.0, precision: 4, unique: false, trim: "0"), "2.0"),
        ("exact p4 trim - 2.0", () => np.format_float_positional(2.0, precision: 4, unique: false, trim: "-"), "2"),
        ("sign 1.5", () => np.format_float_positional(1.5, sign: true), "+1.5"),
        ("sign -1.5", () => np.format_float_positional(-1.5, sign: true), "-1.5"),
        ("pad 1.5", () => np.format_float_positional(1.5, pad_left: 4, pad_right: 4), "   1.5   "),
        ("pad -12.25", () => np.format_float_positional(-12.25, pad_left: 2, pad_right: 1), "-12.25"),
        ("pad trim -", () => np.format_float_positional(3.0, pad_left: 3, pad_right: 3, trim: "-"), "  3    "),
        ("sci pad", () => np.format_float_scientific(-1.5, pad_left: 4), "  -1.5e+00"),
        ("sci exp3", () => np.format_float_scientific(1e100, exp_digits: 3), "1.e+100"),
        ("sci exp1", () => np.format_float_scientific(1e100, exp_digits: 1), "1.e+100"),
        ("sci exp5", () => np.format_float_scientific(1e-100, exp_digits: 5), "1.e-00100"),
        ("sci exp9", () => np.format_float_scientific(1.5, exp_digits: 9), "1.5e+00000"),
        ("sci sign", () => np.format_float_scientific(0.25, precision: 3, unique: false, sign: true), "+2.500e-01"),
    };

    /// <summary>
    ///     NumPy's text for every discriminating value and option (rounding at the cutoff, carries, fractional=False, the
    ///     fractional clamp, min_digits, trim, padding, exponent digits, the three widths).
    /// </summary>
    [TestMethod]
    public void Pins_AreNumPysText()
    {
        var wrong = new List<string>();
        foreach (var (label, call, want) in Pins)
        {
            string got = Outcome(call);
            if (got != want)
                wrong.Add($"{label}: NumPy '{want}', NumSharp '{got}'");
        }
        wrong.Should().BeEmpty();
    }

    /// <summary>
    ///     inf and nan are NumPy's PrintInfNan: <c>inf</c> / <c>-inf</c> / <c>+inf</c> (a '+' only when signs are asked
    ///     for), <c>nan</c> never signed, and no padding (the old emulation printed .NET's <c>Infinity.</c> /
    ///     <c>NaN.</c> / <c>I.nfinitye+07</c>: surface gap SU6).
    /// </summary>
    [TestMethod]
    public void NonFinite_PrintsInfAndNan()
    {
        np.format_float_positional(double.PositiveInfinity).Should().Be("inf");
        np.format_float_positional(double.NaN).Should().Be("nan");
        np.format_float_scientific(double.NegativeInfinity).Should().Be("-inf");
        np.format_float_positional(double.PositiveInfinity, sign: true).Should().Be("+inf");
        np.format_float_scientific(double.PositiveInfinity, sign: true).Should().Be("+inf");
        np.format_float_positional(double.NegativeInfinity, pad_left: 6, pad_right: 4).Should().Be("-inf");
        np.format_float_scientific(-double.NaN, sign: true, pad_left: 6).Should().Be("nan");
        np.format_float_positional(BitConverter.UInt32BitsToSingle(0xffc00001), sign: true).Should().Be("nan");
        np.format_float_scientific(BitConverter.UInt16BitsToHalf(0xfc00)).Should().Be("-inf");
    }

    /// <summary>
    ///     A carry through 9s at the cutoff REMOVES the 9s: <c>1.99999</c> at precision 2 is <c>2.</c> and
    ///     <c>0.0995</c> at scientific precision 1 is <c>1.e-01</c> (the old emulation printed <c>2.00</c> /
    ///     <c>1.0e-01</c>: surface gap SU18), and scientific notation keeps its point when the cutoff left only zero
    ///     fraction digits under trim '-' (it printed <c>1e-01</c>: gap SU19).
    /// </summary>
    [TestMethod]
    public void CarryAtTheCutoff_LeavesFewerDigits()
    {
        np.format_float_positional(1.99999, precision: 2).Should().Be("2.");
        np.format_float_scientific(0.0995, precision: 1).Should().Be("1.e-01");
        np.format_float_scientific(0.1001, precision: 2, trim: "-").Should().Be("1.e-01");
        np.format_float_positional(1.99999, precision: 2, unique: false).Should().Be("2.00");
    }

    /// <summary>
    ///     <c>fractional=False</c> counts significant digits (<c>123000000.</c>; the old emulation ignored it: surface gap
    ///     SU9), and a negative count is NumPy's ValueError before anything is formatted (gap SU8).
    /// </summary>
    [TestMethod]
    public void FractionalFalse_AndNegativeCounts_AreNumPys()
    {
        np.format_float_positional(123456789.125, precision: 3, fractional: false).Should().Be("123000000.");
        Action act = () => np.format_float_positional(1.5, precision: -1);
        act.Should().Throw<ValueError>().WithMessage("precision must be >= 0");
    }

    /// <summary>
    ///     NumPy's argument handling, call for call (<c>ff_probe2.py</c>): the kinds of x a C# caller can spell (a C# integer
    ///     / bool / <see cref="BigInteger"/> is a Python int, converted correctly rounded; <see cref="Half"/> /
    ///     <see cref="float"/> / <see cref="double"/> are np.float16 / np.float32 / np.float64 scalars, each with its own
    ///     digits; a 0-d <see cref="NDArray"/> converts through <c>float(a)</c>; a decimal converts to float64), the
    ///     ValueErrors and TypeErrors in NumPy's order, and the 16384-character buffer.
    /// </summary>
    [TestMethod]
    public void ArgumentHandling_IsNumPys()
    {
        static string TrimError(string shown)
            => "raise TypeError: if supplied, trim must be 'k', '.', '0' or '-' found `" + shown.PadLeft(100) + "`";
        var cases = new (string Label, Func<object> Call, string NumPy)[]
        {
            ("pos(int 3)", () => np.format_float_positional(3), "3."),
            ("pos(int 2**70)", () => np.format_float_positional(BigInteger.Pow(2, 70)), "1180591620717411300000."),
            ("pos(int 2**1100)", () => np.format_float_positional(BigInteger.Pow(2, 1100)), "raise OverflowError: int too large to convert to float"),
            ("pos(bool)", () => np.format_float_positional(true), "1."),
            // A char is NumSharp's Char dtype, NumPy's uint16: np.format_float_positional(np.uint16(3)).
            ("pos(char)", () => np.format_float_positional((char)3), "3."),
            ("pos(str)", () => np.format_float_positional("1.5"), "raise TypeError: must be real number, not str"),
            ("pos(None)", () => np.format_float_positional(null), "raise TypeError: must be real number, not NoneType"),
            ("pos(complex)", () => np.format_float_positional(new Complex(1, 2)), "raise TypeError: must be real number, not complex"),
            ("pos(0-d f8)", () => np.format_float_positional(np.array(1.5)), "1.5"),
            ("pos(0-d f4)", () => np.format_float_positional(NDArray.Scalar(0.1f)), "0.10000000149011612"),
            ("pos(0-d f2)", () => np.format_float_positional(NDArray.Scalar((Half)0.1)), "0.0999755859375"),
            ("pos(0-d i8)", () => np.format_float_positional(np.array(3L)), "3."),
            ("pos(0-d c16)", () => np.format_float_positional(NDArray.Scalar(new Complex(1, 2))), "raise TypeError: float() argument must be a string or a real number, not 'complex'"),
            ("pos(0-d U)", () => np.format_float_positional(np.array("2.5")), "2.5"),
            ("pos(0-d U bad)", () => np.format_float_positional(np.array("x")), "raise ValueError: could not convert string to float: 'x'"),
            ("pos(1-d)", () => np.format_float_positional(np.array(new[] { 1.5 })), "raise TypeError: only 0-dimensional arrays can be converted to Python scalars"),
            ("pos(list)", () => np.format_float_positional(new object[] { 1.5 }), "raise TypeError: must be real number, not list"),
            ("pos(tuple)", () => np.format_float_positional(ValueTuple.Create(1.5)), "raise TypeError: must be real number, not tuple"),
            ("pos(bytes)", () => np.format_float_positional(new byte[] { 49 }), "raise TypeError: must be real number, not bytes"),
            ("pos(decimal)", () => np.format_float_positional(1.5m), "1.5"),
            ("pos precision=-1", () => np.format_float_positional(1.5, precision: -1), "raise ValueError: precision must be >= 0"),
            ("pos pad_left=-1", () => np.format_float_positional(1.5, pad_left: -1), "raise ValueError: pad_left must be >= 0"),
            ("pos pad_right=-1", () => np.format_float_positional(1.5, pad_right: -1), "raise ValueError: pad_right must be >= 0"),
            ("pos min_digits=-1", () => np.format_float_positional(1.5, min_digits: -1), "raise ValueError: min_digits must be >= 0"),
            ("pos min_digits=5 precision=3", () => np.format_float_positional(1.5, min_digits: 5, precision: 3), "raise ValueError: min_digits must be less than or equal to precision"),
            ("pos min_digits=5 precision=0", () => np.format_float_positional(1.5, min_digits: 5, precision: 0), "2.00000"),
            ("pos unique=False", () => np.format_float_positional(1.5, unique: false), "raise TypeError: in non-unique mode `precision` must be supplied"),
            ("pos unique=False precision=0", () => np.format_float_positional(1.5, unique: false, precision: 0), "2."),
            ("pos trim='x'", () => np.format_float_positional(1.5, trim: "x"), TrimError("x")),
            ("pos trim='kk'", () => np.format_float_positional(1.5, trim: "kk"), TrimError("kk")),
            ("pos trim=None", () => np.format_float_positional(1.5, trim: null), TrimError("None")),
            ("pos trim=''", () => np.format_float_positional(1.5, trim: ""), TrimError("")),
            ("pos trim long", () => np.format_float_positional(1.5, trim: new string('x', 150)), "raise TypeError: if supplied, trim must be 'k', '.', '0' or '-' found `" + new string('x', 150) + "`"),
            // The trim check comes before x is converted, the precision check before the trim.
            ("pos trim='x' x='s'", () => np.format_float_positional("s", trim: "x"), TrimError("x")),
            ("pos unique=False x='s'", () => np.format_float_positional("s", unique: false), "raise TypeError: in non-unique mode `precision` must be supplied"),
            ("pos precision=-1 trim='x'", () => np.format_float_positional(1.5, precision: -1, trim: "x"), "raise ValueError: precision must be >= 0"),
            ("pos fractional=False precision=0", () => np.format_float_positional(1.5, fractional: false, precision: 0), "raise ValueError: precision must be greater than 0 if fractional=False"),
            ("pos fractional=False precision=None", () => np.format_float_positional(1.5, fractional: false), "1.5"),
            // The 16384-character buffer.
            ("pos big precision", () => np.format_float_positional(1.5, precision: 20000, unique: false), "raise RuntimeError: Float formatting result too large"),
            ("pos big pad_left", () => np.format_float_positional(1.5, pad_left: 20000), "raise RuntimeError: Float formatting result too large"),
            ("pos big pad_right", () => np.format_float_positional(1.5, pad_right: 20000), "raise RuntimeError: Float formatting result too large"),
            ("pos pad_left=16380 len", () => np.format_float_positional(1.5, pad_left: 16380).Length, "16382"),
            ("pos pad_left=16381 len", () => np.format_float_positional(1.5, pad_left: 16381).Length, "16383"),
            ("pos pad_left=16382", () => np.format_float_positional(1.5, pad_left: 16382).Length, "raise RuntimeError: Float formatting result too large"),
            ("pos 1e300 exact p0 len", () => np.format_float_positional(1e300, unique: false, precision: 0).Length, "302"),
            ("sci exp_digits=-1", () => np.format_float_scientific(1.5, exp_digits: -1), "raise ValueError: exp_digits must be >= 0"),
            ("sci exp_digits=7", () => np.format_float_scientific(1.5, exp_digits: 7), "1.5e+00000"),
            ("sci exp_digits=0", () => np.format_float_scientific(1.5, exp_digits: 0), "1.5e+"),
            ("sci big pad_left len", () => np.format_float_scientific(1.5, pad_left: 20000).Length, "16383"),
            // NumPy's FormatScientific writes its exponent past the 16384-character buffer (its zero loop does not count
            // down the buffer size): 16387 characters.
            ("sci big precision len", () => np.format_float_scientific(1.5, precision: 20000, unique: false).Length, "16387"),
            ("sci(int 2**70)", () => np.format_float_scientific(BigInteger.Pow(2, 70)), "1.1805916207174113e+21"),
            ("sci(0-d f4)", () => np.format_float_scientific(NDArray.Scalar(0.1f)), "1.0000000149011612e-01"),
            ("sci(np.float32)", () => np.format_float_scientific(0.1f), "1.e-01"),
            ("sci(0.0, trim -)", () => np.format_float_scientific(0.0, trim: "-"), "0e+00"),
            ("pos(-0.0, pad)", () => np.format_float_positional(-0.0, pad_left: 3, pad_right: 3), " -0.   "),
            ("sci(-0.0, pad)", () => np.format_float_scientific(-0.0, pad_left: 3), " -0.e+00"),
            ("pos(0.0, trim 0)", () => np.format_float_positional(0.0, trim: "0"), "0.0"),
            ("sci(0.0, min3)", () => np.format_float_scientific(0.0, min_digits: 3), "0.000e+00"),
            ("pos(0.0, exact p3)", () => np.format_float_positional(0.0, unique: false, precision: 3), "0.000"),
        };
        var wrong = new List<string>();
        foreach (var (label, call, want) in cases)
        {
            string got = Outcome(call);
            if (got != want)
                wrong.Add($"{label}: NumPy '{want}', NumSharp '{got}'");
        }
        wrong.Should().BeEmpty();
    }

    /// <summary>
    ///     The three digit generators give the same text: the shortest-digits fast path (Ryu's digits where no cutoff
    ///     binds), the block loop on NumPy's in-place BigInt arithmetic (<see cref="Dragon4.DisableShortestDigits"/>),
    ///     and the System.Numerics reference loop (<see cref="Dragon4.UseReferenceDigits"/>) — over every finite float16
    ///     pattern, and random plus edge patterns of float32 / float64, under unique-mode requests whose cutoffs sometimes
    ///     bind and sometimes do not, and exact-mode requests (no fast path: the block loop against the reference).
    ///     min_digits past the shortest digits reaches the block loop's margin saturation; the long exact cuts its
    ///     zero remainders. The fast path must have served a large share of the requests, or its agreement proves nothing.
    /// </summary>
    [TestMethod]
    public void DigitGenerators_GiveTheSameText()
    {
        // Unique-mode requests: the fast path's decision depends on the cutoff mode, the precision and min_digits; trim,
        // sign and padding act after the digits and ride along. Exact-mode requests never take the fast path.
        var requests = new (string Label, Func<object, string> Format)[]
        {
            ("pos", x => np.format_float_positional(x)),
            ("sci", x => np.format_float_scientific(x)),
            ("pos p3", x => np.format_float_positional(x, precision: 3)),
            ("pos p12 trim -", x => np.format_float_positional(x, precision: 12, trim: "-")),
            ("pos frac False p5", x => np.format_float_positional(x, precision: 5, fractional: false)),
            ("pos min6", x => np.format_float_positional(x, min_digits: 6)),
            ("pos min20", x => np.format_float_positional(x, min_digits: 20)),
            ("pos p20 min2 sign", x => np.format_float_positional(x, precision: 20, min_digits: 2, sign: true)),
            ("sci p4", x => np.format_float_scientific(x, precision: 4)),
            ("sci p17 trim 0 pad", x => np.format_float_scientific(x, precision: 17, trim: "0", pad_left: 3)),
            ("sci min9 exp3", x => np.format_float_scientific(x, min_digits: 9, exp_digits: 3)),
            ("sci min30", x => np.format_float_scientific(x, min_digits: 30)),
            ("pos exact p0", x => np.format_float_positional(x, precision: 0, unique: false)),
            ("pos exact p17", x => np.format_float_positional(x, precision: 17, unique: false)),
            ("pos exact p60 trim .", x => np.format_float_positional(x, precision: 60, unique: false, trim: ".")),
            ("pos exact frac False p9", x => np.format_float_positional(x, precision: 9, unique: false, fractional: false)),
            ("sci exact p30", x => np.format_float_scientific(x, precision: 30, unique: false)),
        };

        var values = new List<object>();
        // Every finite float16 pattern (the two signs of every magnitude).
        for (int b = 0; b < 0x10000; b++)
            if ((b & 0x7c00) != 0x7c00)
                values.Add(BitConverter.UInt16BitsToHalf((ushort)b));
        var rng = new System.Random(20261018);
        // float32 / float64: random patterns and the edges around every power of two.
        for (int i = 0; i < 4000; i++)
        {
            uint u = (uint)rng.Next() ^ ((uint)rng.Next() << 16);
            if ((u & 0x7f800000) != 0x7f800000)
                values.Add(BitConverter.UInt32BitsToSingle(u));
            ulong l = ((ulong)(uint)rng.Next() << 33) ^ ((ulong)(uint)rng.Next() << 11) ^ (uint)rng.Next();
            if ((l & 0x7ff0000000000000) != 0x7ff0000000000000)
                values.Add(BitConverter.UInt64BitsToDouble(l));
        }
        for (int e = 1; e < 254; e += 7)
            foreach (int d in new[] { -1, 0, 1 })
                values.Add(BitConverter.UInt32BitsToSingle((uint)((e << 23) + d)));
        for (int e = 1; e < 2046; e += 31)
            foreach (int d in new[] { -1, 0, 1 })
                values.Add(BitConverter.UInt64BitsToDouble((ulong)(((long)e << 52) + d)));
        values.AddRange(new object[]
        {
            0.1, 1.0 / 3, 2.675, 9.995, 123456789.125, 1e23, 1e300, double.Epsilon, double.MaxValue, 5e-324 * 3,
            BitConverter.UInt64BitsToDouble(0x000fffffffffffff), BitConverter.UInt64BitsToDouble(0x0010000000000000),
            0.1f, float.Epsilon, float.MaxValue, 16777216f, 1e-7f, BitConverter.UInt32BitsToSingle(0x007fffff),
        });

        var wrong = new List<string>();
        long total = 0, unique = 0;
        long before = Dragon4.ShortestDigitsRuns;
        foreach (var (label, format) in requests)
        {
            // float16's every pattern only for the two plain requests (the 60K-pattern set at every request costs seconds).
            bool every = label is "pos" or "sci";
            bool exact = label.Contains("exact", StringComparison.Ordinal);
            foreach (object x in values)
            {
                if (!every && x is Half h && (BitConverter.HalfToUInt16Bits(h) % 61) != 0)
                    continue;
                string production, blocks, reference;
                try
                {
                    production = format(x);
                    Dragon4.DisableShortestDigits = true;
                    blocks = format(x);
                    Dragon4.UseReferenceDigits = true;
                    reference = format(x);
                }
                finally
                {
                    Dragon4.DisableShortestDigits = false;
                    Dragon4.UseReferenceDigits = false;
                }
                total++;
                if (!exact)
                    unique++;
                if ((production != blocks || blocks != reference) && wrong.Count < 20)
                    wrong.Add($"{label} {x.GetType().Name} {x}: production '{production}', blocks '{blocks}', reference '{reference}'");
            }
        }
        long served = Dragon4.ShortestDigitsRuns - before;

        wrong.Should().BeEmpty();
        total.Should().BeGreaterThan(200_000);
        // The fast path must have served a large share of the unique-mode requests (and the loop the rest).
        served.Should().BeGreaterThan(unique / 2);
        served.Should().BeLessThan(unique);
    }

    /// <summary>
    ///     The chunk step's division (<see cref="Dragon4.BigInt.DivideNormalized"/>, Knuth's algorithm D for one 64-bit
    ///     quotient word) is BigInteger's division: random normalized divisors of one to five blocks, random quotients
    ///     and remainders, and the dividend (q + 1) * d - 1 — whose top blocks make the estimate one too big, so the
    ///     add-back branch runs (random operands almost never reach it).
    /// </summary>
    [TestMethod]
    public void BlockDivision_IsBigIntegersDivision()
    {
        var rng = new System.Random(20261018);
        ulong Next() => ((ulong)(uint)rng.Next() << 34) ^ ((ulong)(uint)rng.Next() << 17) ^ (uint)rng.Next();
        static BigInteger FromBlocks(ulong[] b)
        {
            BigInteger r = BigInteger.Zero;
            for (int i = b.Length - 1; i >= 0; i--)
                r = (r << 64) + b[i];
            return r;
        }
        static ulong[] ToBlocks(BigInteger v)
        {
            var list = new List<ulong>();
            while (v > 0)
            {
                list.Add((ulong)(v & ulong.MaxValue));
                v >>= 64;
            }
            return list.ToArray();
        }

        var n = new Dragon4.BigInt();
        var d = new Dragon4.BigInt();
        var wrong = new List<string>();
        for (int trial = 0; trial < 30000; trial++)
        {
            int length = 1 + trial % 5;
            var db = new ulong[length];
            for (int i = 0; i < length; i++)
                db[i] = Next();
            db[length - 1] |= 1UL << 63;
            BigInteger divisor = FromBlocks(db);
            ulong q = trial % 3 == 0 ? Next() >> rng.Next(64) : Next() | (1UL << 63);
            BigInteger dividend, expectedRemainder;
            if (trial % 2 == 0)
            {
                // A random remainder below the divisor.
                var rb = new ulong[length];
                for (int i = 0; i < length; i++)
                    rb[i] = Next();
                expectedRemainder = FromBlocks(rb) % divisor;
                dividend = q * divisor + expectedRemainder;
            }
            else
            {
                // (q + 1) * d - 1: the quotient is q (q + 1 still fits a word), the remainder d - 1, and the top blocks
                // estimate q + 1.
                if (q == ulong.MaxValue)
                    q--;
                dividend = (new BigInteger(q) + 1) * divisor - 1;
                expectedRemainder = divisor - 1;
            }
            n.SetBlocks(ToBlocks(dividend));
            d.SetBlocks(db);
            ulong got = Dragon4.BigInt.DivideNormalized(n, d);
            if ((got != q || n.ToBigInteger() != expectedRemainder) && wrong.Count < 10)
                wrong.Add($"trial {trial}: quotient {got} (expected {q}), remainder {n.ToBigInteger()} (expected {expectedRemainder})");
        }
        wrong.Should().BeEmpty();
    }

    /// <summary>
    ///     The long digit runs the chunk step serves give NumPy's text: 1e300 and the largest float64 to every exact
    ///     place, the smallest subnormals' exact expansions (a zero remainder mid-chunk ends them), min_digits far past the
    ///     shortest digits (the saturated unique loop), and cuts that end one digit before / at / after a chunk boundary.
    /// </summary>
    [TestMethod]
    public void LongDigitRuns_AreNumPys()
    {
        var cases = new (string Label, Func<object> Call, string NumPy)[]
        {
            ("1e300 exact p0 len", () => np.format_float_positional(1e300, unique: false, precision: 0).Length, "302"),
            ("1e300 exact p0 head", () => np.format_float_positional(1e300, unique: false, precision: 0)[..40], "1000000000000000052504760255204420248704"),
            ("1e300 exact p0 tail", () => np.format_float_positional(1e300, unique: false, precision: 0)[^30..], "15669472196386865459400540160."),
            ("max exact pos p0 len", () => np.format_float_positional(double.MaxValue, unique: false, precision: 0).Length, "310"),
            ("max exact pos p0 head", () => np.format_float_positional(double.MaxValue, unique: false, precision: 0)[..40], "1797693134862315708145274237317043567980"),
            ("max exact pos p0 tail", () => np.format_float_positional(double.MaxValue, unique: false, precision: 0)[^25..], "881250404026184124858368."),
            ("max exact sci p40", () => np.format_float_scientific(double.MaxValue, unique: false, precision: 40), "1.7976931348623157081452742373170435679807e+308"),
            ("5e-324 exact p800 len", () => np.format_float_scientific(double.Epsilon, unique: false, precision: 800).Length, "807"),
            ("5e-324 exact p800 trim . len", () => np.format_float_scientific(double.Epsilon, unique: false, precision: 800, trim: ".").Length, "757"),
            ("5e-324 exact p800 trim . tail", () => np.format_float_scientific(double.Epsilon, unique: false, precision: 800, trim: ".")[^40..], "29087538682506419718265533447265625e-324"),
            ("0.1 min60", () => np.format_float_positional(0.1, min_digits: 60), "0.100000000000000005551115123125782702118158340454101562500000"),
            ("pi min40 sci", () => np.format_float_scientific(System.Math.PI, min_digits: 40), "3.1415926535897931159979634685441851615906e+00"),
            ("1e23 min45", () => np.format_float_positional(1e23, min_digits: 45), "99999999999999991611392.000000000000000000000000000000000000000000000"),
            ("1e22 exact p20 frac False", () => np.format_float_positional(1e22, unique: false, precision: 20, fractional: false), "10000000000000000000000."),
            ("2/3 exact p36", () => np.format_float_positional(2.0 / 3, unique: false, precision: 36), "0.666666666666666629659232512494781986"),
            ("2/3 exact p37", () => np.format_float_positional(2.0 / 3, unique: false, precision: 37), "0.6666666666666666296592325124947819859"),
            ("2/3 exact p38", () => np.format_float_positional(2.0 / 3, unique: false, precision: 38), "0.66666666666666662965923251249478198588"),
            ("2/3 exact p39", () => np.format_float_positional(2.0 / 3, unique: false, precision: 39), "0.666666666666666629659232512494781985879"),
            ("2/3 exact p40", () => np.format_float_positional(2.0 / 3, unique: false, precision: 40), "0.6666666666666666296592325124947819858789"),
            ("f32 max exact p0", () => np.format_float_positional(float.MaxValue, unique: false, precision: 0), "340282346638528859811704183484516925440."),
            ("f32 min subnormal exact sci p200 len", () => np.format_float_scientific(float.Epsilon, unique: false, precision: 200).Length, "206"),
            ("f32 min subnormal exact sci p200", () => np.format_float_scientific(float.Epsilon, unique: false, precision: 200, trim: "."), "1.40129846432481707092372958328991613128026194187651577175706828388979108268586060148663818836212158203125e-45"),
            ("f16 min subnormal exact p40", () => np.format_float_positional(BitConverter.UInt16BitsToHalf(1), unique: false, precision: 40), "0.0000000596046447753906250000000000000000"),
        };
        var wrong = new List<string>();
        foreach (var (label, call, want) in cases)
        {
            string got = Outcome(call);
            if (got != want)
                wrong.Add($"{label}: NumPy '{want}', NumSharp '{got}'");
        }
        wrong.Should().BeEmpty();
    }

    /// <summary>
    ///     Powers of two under min_digits: the only values whose last digit can be decided by the high margin alone after
    ///     the request forced NumPy's loop past the shortest digits (low false, high true, twice the remainder below the
    ///     scale, so NumPy rounds UP where the remainder alone would round down). A power of two above the smallest normal
    ///     has unequal margins (the gap above it is twice the gap below); with equal margins that state cannot occur. The
    ///     port drops the margins once the low one reaches the scale (both stay true from there), and a loop that dropped
    ///     them as soon as low or high held would round these eight down. The pins are every such case among the
    ///     float64 / float32 / float16 powers of two with min_digits 1-25 in both notations (found by simulating NumPy's
    ///     loop on exact integers: <c>test/oracle/gen_surface_oracle.py</c> <c>_d4_min_digits_state</c>, which also emits
    ///     them into the surface tier; NumPy's text); the sweep holds the block loop to the System.Numerics reference over
    ///     all the powers of two, both notations, min_digits 1-25.
    /// </summary>
    [TestMethod]
    public void PowersOfTwo_UnderMinDigits_AreNumPys()
    {
        var pins = new (ulong Bits, string NumPy)[]
        {
            (0x17a0000000000000, "6.849404215651260e-195"),
            (0x1c60000000000000, "5.175263503298810e-172"),
            (0x20c0000000000000, "6.109872726999210e-151"),
            (0x4940000000000000, "7.136238463529800e+44"),
            (0x6e30000000000000, "5.783580587434430e+222"),
            (0x7790000000000000, "8.254602048994770e+267"),
            (0x7bb0000000000000, "6.090821257125000e+287"),
            (0x7c50000000000000, "6.237000967296000e+290"),
        };
        var wrong = new List<string>();
        foreach (var (bits, want) in pins)
        {
            string got = np.format_float_scientific(BitConverter.UInt64BitsToDouble(bits), min_digits: 15);
            if (got != want)
                wrong.Add($"0x{bits:x16} sci min15: NumPy '{want}', NumSharp '{got}'");
        }

        // Every normal power of two of the three widths, min_digits 1-25, both notations: the block loop against the
        // reference loop. The shortest-digits fast path is off, so both run NumPy's loop rather than Ryu's digits.
        var values = new List<object>();
        for (int e = 1; e < 0x7ff; e++)
            values.Add(BitConverter.UInt64BitsToDouble((ulong)e << 52));
        for (int e = 1; e < 0xff; e++)
            values.Add(BitConverter.UInt32BitsToSingle((uint)e << 23));
        for (int e = 1; e < 0x1f; e++)
            values.Add(BitConverter.UInt16BitsToHalf((ushort)(e << 10)));
        long total = 0;
        foreach (object x in values)
        {
            for (int minDigits = 1; minDigits <= 25; minDigits++)
            {
                foreach (bool scientific in new[] { false, true })
                {
                    string blocks, reference;
                    try
                    {
                        Dragon4.DisableShortestDigits = true;
                        blocks = scientific ? np.format_float_scientific(x, min_digits: minDigits)
                            : np.format_float_positional(x, min_digits: minDigits);
                        Dragon4.UseReferenceDigits = true;
                        reference = scientific ? np.format_float_scientific(x, min_digits: minDigits)
                            : np.format_float_positional(x, min_digits: minDigits);
                    }
                    finally
                    {
                        // The hooks are [ThreadStatic]: reset them on this thread whatever the formatter threw.
                        Dragon4.DisableShortestDigits = false;
                        Dragon4.UseReferenceDigits = false;
                    }
                    total++;
                    if (blocks != reference && wrong.Count < 20)
                        wrong.Add($"{x.GetType().Name} {x} {(scientific ? "sci" : "pos")} min{minDigits}: blocks '{blocks}', reference '{reference}'");
                }
            }
        }
        wrong.Should().BeEmpty();
        total.Should().Be(values.Count * 50L);
    }

    /// <summary>
    ///     Exact digits (unique=False) of values whose decimal expansion ENDS before the cutoff: NumPy's exact loop stops
    ///     at the digit that leaves a zero remainder, so its digit buffer never ends in a zero, and scientific notation
    ///     with trim '-' keeps the point only when fraction digits were generated (<c>'1e+22'</c>, never <c>'1.e+22'</c>).
    ///     The port's chunk step makes 19 digits per division and must drop the zeros of its last chunk the same way; the
    ///     other trims, and positional notation, cannot tell (they strip or pad the zeros either way), which is why every
    ///     value here is asked at a precision at or past one chunk. NumPy's text; the surface tier holds the whole grid
    ///     (13 values x 6 precisions x 4 trims x 2 notations).
    /// </summary>
    [TestMethod]
    public void ExactDigitsThatEndEarly_AreNumPys()
    {
        var cases = new (object X, int Precision, string Trim, bool Scientific, string NumPy)[]
        {
            (1e22, 30, "k", true, "1.000000000000000000000000000000e+22"),
            (1e22, 30, ".", true, "1.e+22"),
            (1e22, 30, "0", true, "1.0e+22"),
            (1e22, 30, "-", true, "1e+22"),
            (1e22, 30, ".", false, "10000000000000000000000."),
            (1e22, 30, "-", false, "10000000000000000000000"),
            (0.5, 19, "-", true, "5e-01"),
            (1.152921504606846976e18, 40, "-", true, "1.152921504606846976e+18"),
            (System.Math.Pow(2, -10), 40, "-", true, "9.765625e-04"),
            (1e15 + 1, 20, "-", true, "1.000000000000001e+15"),
            (123456789012345678.0, 40, "-", true, "1.2345678901234568e+17"),
            (16777216f, 20, "-", true, "1.6777216e+07"),
            ((Half)65504, 40, "-", true, "6.5504e+04"),
        };
        var wrong = new List<string>();
        foreach (var (x, precision, trim, scientific, want) in cases)
        {
            string got = scientific ? np.format_float_scientific(x, precision: precision, unique: false, trim: trim)
                : np.format_float_positional(x, precision: precision, unique: false, trim: trim);
            if (got != want)
                wrong.Add($"{x.GetType().Name} {x} p{precision} trim '{trim}' {(scientific ? "sci" : "pos")}: NumPy '{want}', NumSharp '{got}'");
        }
        wrong.Should().BeEmpty();
    }

    /// <summary>
    ///     Fraction digits that were generated and are all zeros (a cut whose remainder rounds down, in exact and unique
    ///     mode): every trim mode strips them, then '.' keeps the point, '0' puts one zero back and '-' removes the point
    ///     too (<c>'1'</c>). In positional notation that last step is FormatPositional's own; a number with no fraction
    ///     digits never reaches it (DptZeros prints no point to begin with), so the early-ending exact digits above cannot
    ///     tell. NumPy's text.
    /// </summary>
    [TestMethod]
    public void ZeroFractionDigits_AreTrimmedByEveryMode()
    {
        var cases = new (object X, int Precision, bool Unique, string[] NumPy)[]
        {
            (1.0000001, 3, false, new[] { "1.000", "1.", "1.0", "1" }),
            (1.0001, 3, true, new[] { "1.000", "1.", "1.0", "1" }),
            (100.00049, 3, false, new[] { "100.000", "100.", "100.0", "100" }),
            (1.0001f, 3, true, new[] { "1.000", "1.", "1.0", "1" }),
        };
        string[] trims = { "k", ".", "0", "-" };
        var wrong = new List<string>();
        foreach (var (x, precision, unique, want) in cases)
        {
            for (int t = 0; t < trims.Length; t++)
            {
                string got = np.format_float_positional(x, precision: precision, unique: unique, trim: trims[t]);
                if (got != want[t])
                    wrong.Add($"{x.GetType().Name} {x} p{precision} unique={unique} trim '{trims[t]}': NumPy '{want[t]}', NumSharp '{got}'");
            }
        }
        wrong.Should().BeEmpty();
    }
}
