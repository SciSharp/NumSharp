using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Threading.Tasks;
using NumSharp.Backends;

namespace NumSharp.Tests.Polynomial
{
    /// <summary>
    ///     The <c>numpy.polynomial</c> series algebra (plan U2): <c>{p}mulx</c>, <c>{p}mul</c>, <c>{p}div</c>,
    ///     <c>{p}pow</c>, <c>{p}fromroots</c> of the six bases and <c>X2poly</c> / <c>poly2X</c> of the five non-power
    ///     ones. Three kinds of test live here: NumPy 2.4.2's own <c>TestArithmetic</c> / <c>TestMisc</c> suites ported
    ///     (exact where NumPy asserts equality, almost-equal where it does); byte dumps NumPy produced for the dtype and
    ///     layout quirks (inputs spelled out next to each test); and the facts the byte oracle
    ///     (<c>polyalgebra.jsonl</c> / <c>polyalgebra_parity.jsonl</c>) cannot see — decimal and char (no NumPy
    ///     dtype), the C# call spellings of <c>{p}pow</c>, the per-thread arena's lifetime, the non-finite complex
    ///     product (NumPy's zdotu result, once a documented divergence), and the one documented divergence left
    ///     (fromroots' ±0 root order). The wholeness pass's argument kinds live in
    ///     <see cref="PolynomialAlgebraArgumentKindsTests"/>.
    /// </summary>
    [TestClass]
    public class PolynomialAlgebraTests
    {
        /// <summary>One basis's series algebra plus the additive and evaluation members its suite leans on.</summary>
        /// <param name="Name">The basis prefix NumPy uses (<c>poly</c>, <c>cheb</c>, ...).</param>
        /// <param name="Mulx"><c>{p}mulx(c)</c>.</param>
        /// <param name="Mul"><c>{p}mul(c1, c2)</c>.</param>
        /// <param name="Div"><c>{p}div(c1, c2)</c>.</param>
        /// <param name="Pow"><c>{p}pow(c, pow)</c> at the default maxpower.</param>
        /// <param name="FromRoots"><c>{p}fromroots(roots)</c>.</param>
        /// <param name="Add"><c>{p}add(c1, c2)</c>.</param>
        /// <param name="Val"><c>{p}val(x, c, tensor)</c>.</param>
        /// <param name="ToPower"><c>X2poly(c)</c>; null for the power basis.</param>
        /// <param name="FromPower"><c>poly2X(pol)</c>; null for the power basis.</param>
        private sealed record Basis(string Name, Func<object, NDArray> Mulx, Func<object, object, NDArray> Mul,
            Func<object, object, (NDArray quo, NDArray rem)> Div, Func<object, int, NDArray> Pow,
            Func<object, NDArray> FromRoots, Func<object, object, NDArray> Add, Func<object, object, bool, NDArray> Val,
            Func<object, NDArray> ToPower, Func<object, NDArray> FromPower);

        /// <summary>The six bases, in NumPy's module order.</summary>
        private static readonly Basis[] Bases =
        {
            new("poly", np.polynomial.polynomial.polymulx, np.polynomial.polynomial.polymul, np.polynomial.polynomial.polydiv,
                (c, k) => np.polynomial.polynomial.polypow(c, k), np.polynomial.polynomial.polyfromroots,
                np.polynomial.polynomial.polyadd, np.polynomial.polynomial.polyval, null, null),
            new("cheb", np.polynomial.chebyshev.chebmulx, np.polynomial.chebyshev.chebmul, np.polynomial.chebyshev.chebdiv,
                (c, k) => np.polynomial.chebyshev.chebpow(c, k), np.polynomial.chebyshev.chebfromroots,
                np.polynomial.chebyshev.chebadd, np.polynomial.chebyshev.chebval,
                np.polynomial.chebyshev.cheb2poly, np.polynomial.chebyshev.poly2cheb),
            new("leg", np.polynomial.legendre.legmulx, np.polynomial.legendre.legmul, np.polynomial.legendre.legdiv,
                (c, k) => np.polynomial.legendre.legpow(c, k), np.polynomial.legendre.legfromroots,
                np.polynomial.legendre.legadd, np.polynomial.legendre.legval,
                np.polynomial.legendre.leg2poly, np.polynomial.legendre.poly2leg),
            new("lag", np.polynomial.laguerre.lagmulx, np.polynomial.laguerre.lagmul, np.polynomial.laguerre.lagdiv,
                (c, k) => np.polynomial.laguerre.lagpow(c, k), np.polynomial.laguerre.lagfromroots,
                np.polynomial.laguerre.lagadd, np.polynomial.laguerre.lagval,
                np.polynomial.laguerre.lag2poly, np.polynomial.laguerre.poly2lag),
            new("herm", np.polynomial.hermite.hermmulx, np.polynomial.hermite.hermmul, np.polynomial.hermite.hermdiv,
                (c, k) => np.polynomial.hermite.hermpow(c, k), np.polynomial.hermite.hermfromroots,
                np.polynomial.hermite.hermadd, np.polynomial.hermite.hermval,
                np.polynomial.hermite.herm2poly, np.polynomial.hermite.poly2herm),
            new("herme", np.polynomial.hermite_e.hermemulx, np.polynomial.hermite_e.hermemul, np.polynomial.hermite_e.hermediv,
                (c, k) => np.polynomial.hermite_e.hermepow(c, k), np.polynomial.hermite_e.hermefromroots,
                np.polynomial.hermite_e.hermeadd, np.polynomial.hermite_e.hermeval,
                np.polynomial.hermite_e.herme2poly, np.polynomial.hermite_e.poly2herme),
        };

        /// <summary>
        ///     The basis polynomials' power-series coefficients NumPy's suites convert against (<c>Tlist</c>, <c>Llist</c>,
        ///     <c>Lalist</c>, <c>Hlist</c>, <c>Helist</c> of <c>numpy/polynomial/tests</c>) — each row is the i-th basis
        ///     polynomial, numerators over a common denominator where NumPy writes them that way.
        /// </summary>
        private static readonly Dictionary<string, double[][]> PowerTables = new()
        {
            ["cheb"] = new[]
            {
                new[] { 1.0 }, new[] { 0.0, 1 }, new[] { -1.0, 0, 2 }, new[] { 0.0, -3, 0, 4 }, new[] { 1.0, 0, -8, 0, 8 },
                new[] { 0.0, 5, 0, -20, 0, 16 }, new[] { -1.0, 0, 18, 0, -48, 0, 32 }, new[] { 0.0, -7, 0, 56, 0, -112, 0, 64 },
                new[] { 1.0, 0, -32, 0, 160, 0, -256, 0, 128 }, new[] { 0.0, 9, 0, -120, 0, 432, 0, -576, 0, 256 },
            },
            ["leg"] = new[]
            {
                new[] { 1.0 }, new[] { 0.0, 1 }, Over(2, -1, 0, 3), Over(2, 0, -3, 0, 5), Over(8, 3, 0, -30, 0, 35),
                Over(8, 0, 15, 0, -70, 0, 63), Over(16, -5, 0, 105, 0, -315, 0, 231), Over(16, 0, -35, 0, 315, 0, -693, 0, 429),
                Over(128, 35, 0, -1260, 0, 6930, 0, -12012, 0, 6435), Over(128, 0, 315, 0, -4620, 0, 18018, 0, -25740, 0, 12155),
            },
            ["lag"] = new[]
            {
                Over(1, 1), Over(1, 1, -1), Over(2, 2, -4, 1), Over(6, 6, -18, 9, -1), Over(24, 24, -96, 72, -16, 1),
                Over(120, 120, -600, 600, -200, 25, -1), Over(720, 720, -4320, 5400, -2400, 450, -36, 1),
            },
            ["herm"] = new[]
            {
                new[] { 1.0 }, new[] { 0.0, 2 }, new[] { -2.0, 0, 4 }, new[] { 0.0, -12, 0, 8 }, new[] { 12.0, 0, -48, 0, 16 },
                new[] { 0.0, 120, 0, -160, 0, 32 }, new[] { -120.0, 0, 720, 0, -480, 0, 64 },
                new[] { 0.0, -1680, 0, 3360, 0, -1344, 0, 128 }, new[] { 1680.0, 0, -13440, 0, 13440, 0, -3584, 0, 256 },
                new[] { 0.0, 30240, 0, -80640, 0, 48384, 0, -9216, 0, 512 },
            },
            ["herme"] = new[]
            {
                new[] { 1.0 }, new[] { 0.0, 1 }, new[] { -1.0, 0, 1 }, new[] { 0.0, -3, 0, 1 }, new[] { 3.0, 0, -6, 0, 1 },
                new[] { 0.0, 15, 0, -10, 0, 1 }, new[] { -15.0, 0, 45, 0, -15, 0, 1 }, new[] { 0.0, -105, 0, 105, 0, -21, 0, 1 },
                new[] { 105.0, 0, -420, 0, 210, 0, -28, 0, 1 }, new[] { 0.0, 945, 0, -1260, 0, 378, 0, -36, 0, 1 },
            },
        };

        /// <summary>NumPy's <c>np.array([...]) / d</c> spelling of a table row.</summary>
        /// <param name="d">The denominator.</param>
        /// <param name="numerators">The numerators.</param>
        /// <returns>The row, each value divided once (the same float64 division NumPy performs).</returns>
        private static double[] Over(double d, params double[] numerators) => numerators.Select(v => v / d).ToArray();

        /// <summary>
        ///     Asserts dtype, shape and every byte (logical C order) against a NumPy hex dump — BIT parity, so a signed
        ///     zero or a last-place rounding difference fails.
        /// </summary>
        /// <param name="actual">The result.</param>
        /// <param name="dtype">NumPy's dtype name.</param>
        /// <param name="shape">NumPy's shape.</param>
        /// <param name="hex">NumPy's <c>tobytes().hex()</c> of the C-order copy.</param>
        /// <param name="because">Context for the failure message.</param>
        private static void AssertBytes(NDArray actual, string dtype, long[] shape, string hex, string because = "")
        {
            actual.dtype.name.Should().Be(dtype, because);
            actual.shape.Should().Equal(shape, because);
            Hex(actual).Should().Be(hex, because);
        }

        /// <summary>The logical-C-order bytes of <paramref name="a"/> as lowercase hex (NumPy's <c>tobytes().hex()</c>).</summary>
        /// <param name="a">The array (any layout).</param>
        /// <returns>The hex text.</returns>
        private static string Hex(NDArray a)
        {
            using var c = np.ascontiguousarray(a);
            var bytes = new byte[c.size * c.dtypesize];
            unsafe
            {
                var p = (byte*)c.Storage.Address + c.Shape.offset * c.dtypesize;
                for (int i = 0; i < bytes.Length; i++) bytes[i] = p[i];
            }
            return Convert.ToHexString(bytes).ToLowerInvariant();
        }

        /// <summary><c>[0]*i + [1]</c> — NumPy's test series (the i-th basis polynomial), as a Python list of ints.</summary>
        /// <param name="i">The degree.</param>
        /// <returns>The list.</returns>
        private static object[] Unit(int i) => Enumerable.Repeat((object)0, i).Append(1).ToArray();

        /// <summary><c>[0]*i + tail</c> as a Python list (NumPy's suites build their targets this way).</summary>
        /// <param name="i">Leading zeros.</param>
        /// <param name="tail">The trailing values.</param>
        /// <returns>The values as doubles.</returns>
        private static double[] Padded(int i, params double[] tail) => Enumerable.Repeat(0.0, i).Concat(tail).ToArray();

        /// <summary>NumPy's <c>polyutils.trimcoef(c)</c> of a float64 result (trailing zeros dropped, one kept).</summary>
        /// <param name="a">A 1-D float64 array.</param>
        /// <returns>The trimmed values.</returns>
        private static double[] Trim(NDArray a) => Trim(a.ToArray<double>());

        /// <summary>NumPy's <c>polyutils.trimcoef(c)</c> of plain values.</summary>
        /// <param name="v">The values.</param>
        /// <returns>The trimmed values.</returns>
        private static double[] Trim(double[] v)
        {
            int n = v.Length;
            while (n > 1 && v[n - 1] == 0) n--;
            return v.Take(n).ToArray();
        }

        /// <summary>NumPy's <c>assert_almost_equal(actual, desired)</c> (decimal=7): same length, every |diff| &lt; 1.5e-7.</summary>
        /// <param name="actual">The values.</param><param name="desired">The expected values.</param>
        /// <param name="because">Context.</param>
        private static void AlmostEqual(double[] actual, double[] desired, string because)
        {
            actual.Length.Should().Be(desired.Length, because);
            for (int i = 0; i < actual.Length; i++)
                System.Math.Abs(actual[i] - desired[i]).Should().BeLessThan(1.5e-7, $"{because} [{i}]");
        }

        // =========================================================================================================
        //  NumPy 2.4.2's own suites (numpy/polynomial/tests/test_{polynomial,chebyshev,legendre,...}.py), ported
        // =========================================================================================================

        /// <summary>
        ///     <c>test_{p}mulx</c>: <c>{p}mulx([0]) == [0]</c> (the zero-series shortcut returns the series itself), the
        ///     multiplication of the constant 1 by x, and of every <c>[0]*i + [1]</c> up to degree 4 — the basis's
        ///     three-term recurrence read off directly. NumPy asserts these EXACTLY (laguerre's almost-equal, but its
        ///     values are integers).
        /// </summary>
        [TestMethod]
        public void Mulx_MatchesNumPysSuite()
        {
            var ofOne = new Dictionary<string, double[]>
            {
                ["poly"] = new[] { 0.0, 1 }, ["cheb"] = new[] { 0.0, 1 }, ["leg"] = new[] { 0.0, 1 },
                ["lag"] = new[] { 1.0, -1 }, ["herm"] = new[] { 0.0, 0.5 }, ["herme"] = new[] { 0.0, 1 },
            };
            foreach (var b in Bases)
            {
                b.Mulx(new object[] { 0 }).ToArray<double>().Should().Equal(new[] { 0.0 }, $"{b.Name}mulx([0])");
                b.Mulx(new object[] { 1 }).ToArray<double>().Should().Equal(ofOne[b.Name], $"{b.Name}mulx([1])");
                for (int i = 1; i < 5; i++)
                {
                    double tmp = 2 * i + 1;
                    double[] tgt = b.Name switch
                    {
                        "poly" => Padded(i + 1, 1),
                        "cheb" => Padded(i - 1, .5, 0, .5),
                        "leg" => Padded(i - 1, i / tmp, 0, (i + 1) / tmp),
                        "lag" => Padded(i - 1, -i, 2 * i + 1, -(i + 1)),
                        "herm" => Padded(i - 1, i, 0, .5),
                        _ => Padded(i - 1, i, 0, 1),
                    };
                    b.Mulx(Unit(i)).ToArray<double>().Should().Equal(tgt, $"{b.Name}mulx([0]*{i} + [1])");
                }
            }
        }

        /// <summary>
        ///     <c>test_{p}mul</c> for every pair of unit series of degree 0..4. The power and Chebyshev bases have closed
        ///     forms (<c>x^i x^j = x^(i+j)</c>, <c>T_i T_j = (T_(i+j) + T_|i-j|) / 2</c>) NumPy asserts exactly; the
        ///     recurrence bases are checked the way NumPy checks them — the product's length is <c>i + j + 1</c> and its
        ///     values at 100 points equal the product of the factors' values (almost-equal).
        /// </summary>
        [TestMethod]
        public void Mul_MatchesNumPysSuite()
        {
            var x = np.linspace(-1.0, 1.0, 100);
            foreach (var b in Bases)
                for (int i = 0; i < 5; i++)
                    for (int j = 0; j < 5; j++)
                    {
                        string msg = $"{b.Name}mul at i={i}, j={j}";
                        var pol3 = b.Mul(Unit(i), Unit(j));
                        if (b.Name == "poly" || b.Name == "cheb")
                        {
                            var tgt = new double[i + j + 1];
                            if (b.Name == "poly")
                                tgt[i + j] += 1;
                            else
                            {
                                tgt[i + j] += .5;
                                tgt[System.Math.Abs(i - j)] += .5;
                            }
                            Trim(pol3).Should().Equal(Trim(tgt), msg);
                            continue;
                        }
                        pol3.size.Should().Be(i + j + 1, msg);
                        var val1 = b.Val(x, Unit(i), true).ToArray<double>();
                        var val2 = b.Val(x, Unit(j), true).ToArray<double>();
                        var val3 = b.Val(x, pol3, true).ToArray<double>();
                        AlmostEqual(val3, val1.Zip(val2, (p, q) => p * q).ToArray(), msg);
                    }
        }

        /// <summary>
        ///     <c>test_{p}div</c>: the quotient and remainder rebuild the dividend — EXACTLY for five bases, almost-equal for
        ///     laguerre as NumPy asserts it (<c>add(mul(quo, ci), rem) == tgt</c>) — plus the power basis's own cases: division by a zero series raises (ZeroDivisionError, which
        ///     NumPy raises with an EMPTY message), and the scalar divisor short cut.
        /// </summary>
        [TestMethod]
        public void Div_MatchesNumPysSuite()
        {
            var poly = np.polynomial.polynomial;
            Action zero = () => poly.polydiv(new object[] { 1 }, new object[] { 0 });
            zero.Should().Throw<DivideByZeroException>().Which.Message.Should().BeEmpty();
            var (q1, r1) = poly.polydiv(new object[] { 2 }, new object[] { 2 });
            q1.ToArray<double>().Should().Equal(1.0);
            r1.ToArray<double>().Should().Equal(0.0);
            var (q2, r2) = poly.polydiv(new object[] { 2, 2 }, new object[] { 2 });
            q2.ToArray<double>().Should().Equal(1.0, 1.0);
            r2.ToArray<double>().Should().Equal(0.0);

            foreach (var b in Bases)
                for (int i = 0; i < 5; i++)
                    for (int j = 0; j < 5; j++)
                    {
                        string msg = $"{b.Name}div at i={i}, j={j}";
                        // The power suite divides [0]*i + [1, 2]; the others the unit series.
                        object[] ci = b.Name == "poly" ? Padded(i, 1, 2).Cast<object>().ToArray() : Unit(i);
                        object[] cj = b.Name == "poly" ? Padded(j, 1, 2).Cast<object>().ToArray() : Unit(j);
                        var tgt = b.Add(ci, cj);
                        var (quo, rem) = b.Div(tgt, ci);
                        var res = b.Add(b.Mul(quo, ci), rem);
                        // NumPy asserts laguerre almost-equal (its rebuilt series is 1 - 2^-53 at i=1, j=3 — in NumPy too).
                        if (b.Name == "lag")
                            AlmostEqual(Trim(res), Trim(tgt), msg);
                        else
                            Trim(res).Should().Equal(Trim(tgt), msg);
                    }
        }

        /// <summary>
        ///     <c>test_{p}pow</c>: <c>{p}pow(arange(i + 1), j)</c> equals <c>reduce({p}mul, [c]*j, [1])</c> EXACTLY for
        ///     i, j in 0..4 (for chebpow too, although it multiplies z-series without converting back in between).
        /// </summary>
        [TestMethod]
        public void Pow_EqualsRepeatedMul()
        {
            foreach (var b in Bases)
                for (int i = 0; i < 5; i++)
                    for (int j = 0; j < 5; j++)
                    {
                        var c = np.arange(i + 1);
                        NDArray tgt = np.array(new long[] { 1 });
                        for (int k = 0; k < j; k++)
                            tgt = b.Mul(tgt, c);
                        var res = b.Pow(c, j);
                        Trim(res).Should().Equal(Trim(tgt.astype(np.float64)), $"{b.Name}pow at i={i}, j={j}");
                    }
        }

        /// <summary>
        ///     <c>test_X2poly</c> / <c>test_poly2X</c>: each basis polynomial converts to its power series (the tables of
        ///     NumPy's suites) and back (almost-equal, as NumPy asserts).
        /// </summary>
        [TestMethod]
        public void Conversions_MatchNumPysSuite()
        {
            foreach (var b in Bases.Where(b => b.ToPower != null))
            {
                var table = PowerTables[b.Name];
                for (int i = 0; i < table.Length; i++)
                {
                    AlmostEqual(b.ToPower(Unit(i)).ToArray<double>(), table[i], $"{b.Name}2poly([0]*{i} + [1])");
                    AlmostEqual(b.FromPower(table[i]).ToArray<double>(), Padded(i, 1), $"poly2{b.Name}(table[{i}])");
                }
            }
        }

        /// <summary>
        ///     <c>test_{p}fromroots</c>: no roots is the series <c>[1.]</c>; the Chebyshev nodes of degree i give (scaled by
        ///     <c>2**(i-1)</c>) the power table's T_i / the unit Chebyshev series; every other basis's product has
        ///     <c>i + 1</c> coefficients, a monic power form and zeros at the roots (almost-equal, as NumPy asserts).
        /// </summary>
        [TestMethod]
        public void FromRoots_MatchesNumPysSuite()
        {
            foreach (var b in Bases)
            {
                AlmostEqual(Trim(b.FromRoots(Array.Empty<object>())), new[] { 1.0 }, $"{b.Name}fromroots([])");
                for (int i = 1; i < 5; i++)
                {
                    string msg = $"{b.Name}fromroots of the degree-{i} Chebyshev nodes";
                    var roots = np.cos(np.linspace(-System.Math.PI, 0.0, 2 * i + 1)["1::2"]);
                    var pol = b.FromRoots(roots);
                    if (b.Name == "poly")
                    {
                        var scaled = (pol * System.Math.Pow(2, i - 1)).ToArray<double>();
                        AlmostEqual(Trim(scaled), Trim(PowerTables["cheb"][i]), msg);
                        continue;
                    }
                    if (b.Name == "cheb")
                    {
                        var scaled = (pol * System.Math.Pow(2, i - 1)).ToArray<double>();
                        AlmostEqual(Trim(scaled), Padded(i, 1), msg);
                        continue;
                    }
                    pol.size.Should().Be(i + 1, msg);
                    AlmostEqual(new[] { b.ToPower(pol).ToArray<double>().Last() }, new[] { 1.0 }, msg + " (monic)");
                    AlmostEqual(b.Val(roots, pol, true).ToArray<double>(), new double[i], msg + " (zeros)");
                }
            }
        }

        // =========================================================================================================
        //  NumPy 2.4.2 byte dumps (probed; inputs spelled out)
        // =========================================================================================================

        /// <summary>
        ///     The result dtype follows NumPy's Python, statement by statement — not a promotion rule:
        ///     <list type="bullet">
        ///         <item>a recurrence-basis product whose SHORTER factor has one coefficient binds the Python int
        ///             <c>c1 = 0</c>, so <c>{p}mulx(0)</c> is float64 <c>[0.]</c> and the float32 / float16 result
        ///             turns float64 (with two coefficients it stays float32);</item>
        ///         <item><c>_div</c>'s remainder is computed against the unit series <c>[0]*i + [1]</c> (int64), so a
        ///             float32 division's remainder is float64 while its quotient keeps float32 — polydiv, which does not
        ///             go through <c>_div</c>, keeps float32 in both (its remainder a VIEW);</item>
        ///         <item>poly2X starts from <c>res = 0</c> (float64); X2poly keeps the dtype;</item>
        ///         <item>fromroots' lines are <c>np.array([-r, 1])</c> of a NumPy scalar and a Python int — array coercion
        ///             promotes STRONGLY, so float16 / float32 roots give float64; ints give float64 through as_series;
        ///             no roots give <c>np.ones(1)</c>;</item>
        ///         <item><c>{p}pow(c, 0)</c> is <c>np.array([1], dtype=c.dtype)</c> of the as_series copy.</item>
        ///     </list>
        /// </summary>
        [TestMethod]
        public void DtypeQuirks_AreNumPys()
        {
            var P = np.polynomial;
            var f32 = np.array(new[] { 1.5f, -2.25f });
            var g32 = np.array(new[] { 0.5f });
            AssertBytes(P.legendre.legmul(f32, g32), "float64", new long[] { 2 }, "000000000000e83f000000000000f2bf", "legmul f32 x f32[1]");
            AssertBytes(P.legendre.legmul(f32, f32), "float32", new long[] { 3 }, "00007c400000d8c000005840", "legmul f32 x f32[2]");
            AssertBytes(P.polynomial.polymul(f32, g32), "float32", new long[] { 2 }, "0000403f000090bf", "polymul f32 x f32[1]");
            AssertBytes(P.hermite.hermmul(np.array(new[] { (Half)1.5, (Half)2.0, (Half)(-0.5) }), np.array(new[] { (Half)0.25 })),
                        "float64", new long[] { 3 }, "000000000000d83f000000000000e03f000000000000c0bf", "hermmul f16");

            var (lq, lr) = P.legendre.legdiv(np.array(new[] { 1f, 2f, 3f, 4f }), np.array(new[] { 0.5f, 1.5f }));
            AssertBytes(lq, "float32", new long[] { 3 }, "ed2534bf721c473fe4388e40", "legdiv f32 quo");
            AssertBytes(lr, "float64", new long[] { 1 }, "0aed25b497d0ee3f", "legdiv f32 rem");
            var (pq, pr) = P.polynomial.polydiv(np.array(new[] { 1f, 2f, 3f, 4f }), np.array(new[] { 0.5f, 1.5f }));
            AssertBytes(pq, "float32", new long[] { 3 }, "bd84763fe3388e3fabaa2a40", "polydiv f32 quo");
            AssertBytes(pr, "float32", new long[] { 1 }, "a1bd043f", "polydiv f32 rem");
            pr.flags.owndata.Should().BeFalse("polydiv's remainder is trimseq(c1[:j+1]) — a view of the working copy");

            var h = np.array(new[] { (Half)1.0, (Half)2.0, (Half)3.0 });
            AssertBytes(P.chebyshev.poly2cheb(h), "float64", new long[] { 3 }, "00000000000004400000000000000040000000000000f83f", "poly2cheb f16");
            AssertBytes(P.chebyshev.cheb2poly(h), "float16", new long[] { 3 }, "00c000400046", "cheb2poly f16");

            AssertBytes(P.polynomial.polyfromroots(np.array(new[] { (Half)0.5, (Half)(-2.0) })), "float64", new long[] { 3 },
                        "000000000000f0bf000000000000f83f000000000000f03f", "polyfromroots f16");
            AssertBytes(P.laguerre.lagfromroots(np.array(new[] { 0.5f, -2f })), "float64", new long[] { 3 },
                        "000000000000044000000000000016c00000000000000040", "lagfromroots f32");
            AssertBytes(P.hermite.hermfromroots(np.array(new[] { new Complex(1, 1), new Complex(2, -1) })), "complex128", new long[] { 3 },
                        "0000000000000c40000000000000f03f000000000000f8bf0000000000000000000000000000d03f0000000000000000", "hermfromroots c128");
            AssertBytes(P.polynomial.polyfromroots(new object[] { 1, 2, 3 }), "float64", new long[] { 4 },
                        "00000000000018c0000000000000264000000000000018c0000000000000f03f", "polyfromroots of Python ints");
            AssertBytes(P.polynomial.polyfromroots(Array.Empty<object>()), "float64", new long[] { 1 }, "000000000000f03f", "polyfromroots([])");

            var s32 = np.array(new[] { 1.5f, -0.5f });
            AssertBytes(P.legendre.legpow(s32, 2), "float32", new long[] { 3 }, "555515400000c0bfabaa2a3e", "legpow f32 ^ 2");
            AssertBytes(P.legendre.legpow(s32, 0), "float32", new long[] { 1 }, "0000803f", "legpow f32 ^ 0");
            AssertBytes(P.polynomial.polypow(new object[] { 1, 2 }, 0), "float64", new long[] { 1 }, "000000000000f03f", "polypow ints ^ 0");
        }

        /// <summary>
        ///     <c>polypow</c> and <c>chebpow</c> run <c>np.convolve</c> directly and never trim, so a product that
        ///     underflows keeps its trailing zero; <c>polymul</c> / <c>chebmul</c> trim it — and trimseq returns a VIEW
        ///     (<c>OWNDATA</c> false) of the untrimmed buffer.
        /// </summary>
        [TestMethod]
        public void PowDoesNotTrim_MulDoes()
        {
            var P = np.polynomial;
            var tiny = np.array(new[] { 1.0, 1e-200 });
            AssertBytes(P.polynomial.polypow(tiny, 2), "float64", new long[] { 3 }, "000000000000f03facf74e15927e78160000000000000000", "polypow");
            AssertBytes(P.chebyshev.chebpow(tiny, 2), "float64", new long[] { 3 }, "000000000000f03facf74e15927e78160000000000000000", "chebpow");
            var pm = P.polynomial.polymul(tiny, tiny);
            AssertBytes(pm, "float64", new long[] { 2 }, "000000000000f03facf74e15927e7816", "polymul");
            pm.flags.owndata.Should().BeFalse("polymul trimmed the convolution: a view");
            var cm = P.chebyshev.chebmul(tiny, tiny);
            AssertBytes(cm, "float64", new long[] { 2 }, "000000000000f03facf74e15927e7816", "chebmul");
            cm.flags.owndata.Should().BeFalse("chebmul trimmed the converted series: a view");

            var (q, r) = P.polynomial.polydiv(np.array(new[] { 2.0, 3.0, 1.0 }), np.array(new[] { 1.0, 1.0 }));
            AssertBytes(q, "float64", new long[] { 2 }, "0000000000000040000000000000f03f", "polydiv exact quo");
            q.flags.owndata.Should().BeTrue("the quotient is c1[j+1:]/scl — a fresh array");
            AssertBytes(r, "float64", new long[] { 1 }, "0000000000000000", "polydiv exact rem");
            r.flags.owndata.Should().BeFalse("the remainder is trimseq(c1[:j+1]) — a view");
        }

        /// <summary>
        ///     NumPy's two complex products, per statement: a recurrence basis's one-coefficient factor multiplies an
        ///     ARRAY by a NumPy SCALAR — a ufunc, <c>simd_cmul</c>'s fused form — while the power basis's convolution
        ///     sums <c>zdotu</c>-style four products. Full-mantissa operands, where the forms differ in the last bit.
        /// </summary>
        [TestMethod]
        public void ComplexProducts_UseNumPysProductForms()
        {
            var P = np.polynomial;
            var a = np.array(new[] { new Complex(0.1, 0.7), new Complex(-1.3, 0.2), new Complex(0.9, -0.4) });
            AssertBytes(P.legendre.legmul(a, np.array(new[] { new Complex(1.7, -0.3) })), "complex128", new long[] { 3 },
                        "52b81e85eb51d83f8fc2f5285c8ff23f33333333333301c05c8fc2f5285ce73f8fc2f5285c8ff63f676666666666eebf", "legmul 1 term");
            var b2 = np.array(new[] { new Complex(1.7, -0.3), new Complex(0.2, 1.1) });
            AssertBytes(P.legendre.legmul(a, b2), "complex128", new long[] { 4 },
                        "295c8fc2f528cc3f4a7eb1e4174be63f9eefa7c64b3705c01b2fdd240681f53f703d0ad7a370f13f083a6da0d306febf042b8716d9ced73f47b6f3fdd478e13f",
                        "legmul 2 terms");
            AssertBytes(P.polynomial.polymul(a, b2), "complex128", new long[] { 4 },
                        "52b81e85eb51d83f8fc2f5285c8ff23f33333333333307c05c8fc2f5285cef3fc2f5285c8fc2ed3fb91e85eb51b802c0d8a3703d0ad7e33f2085eb51b81eed3f",
                        "polymul");
        }

        /// <summary>
        ///     <c>{p}mulx</c>'s zero-series short cut returns the as_series copy itself — a <c>-0.0</c> keeps its sign,
        ///     even for laguerre, whose general branch would compute <c>[c0, -c0]</c> — and the general branches' values.
        /// </summary>
        [TestMethod]
        public void Mulx_ZeroSeries_ReturnsTheCopy()
        {
            var P = np.polynomial;
            AssertBytes(P.polynomial.polymulx(np.array(new[] { -0.0 })), "float64", new long[] { 1 }, "0000000000000080", "polymulx([-0.0])");
            AssertBytes(P.laguerre.lagmulx(np.array(new[] { -0.0 })), "float64", new long[] { 1 }, "0000000000000080", "lagmulx([-0.0])");
            AssertBytes(P.laguerre.lagmulx(np.array(new[] { 2.0 })), "float64", new long[] { 2 }, "000000000000004000000000000000c0", "lagmulx([2.0])");
            AssertBytes(P.hermite.hermmulx(np.array(new[] { 1.0, 2.0, 3.0 })), "float64", new long[] { 4 },
                        "00000000000000400000000000001a40000000000000f03f000000000000f83f", "hermmulx([1, 2, 3])");
        }

        /// <summary>
        ///     The argument errors, NumPy's type and text verbatim: the empty ZeroDivisionError, pow's ValueError /
        ///     OverflowError from <c>int(pow)</c> and its two range checks, fromroots' <c>len()</c> TypeErrors (a Python
        ///     float, a 0-d array), and as_series' texts.
        /// </summary>
        [TestMethod]
        public void Errors_AreNumPys()
        {
            var P = np.polynomial;
            var one = new object[] { 1.0 };
            void Raises<TException>(Action call, string message, string because) where TException : Exception
                => call.Should().Throw<TException>(because).Which.Message.Should().Be(message, because);

            Raises<DivideByZeroException>(() => P.polynomial.polydiv(one, new object[] { 0.0 }), "", "polydiv by [0.]");
            Raises<ValueError>(() => P.polynomial.polypow(one, -1), "Power must be a non-negative integer.", "polypow(c, -1)");
            Raises<ValueError>(() => P.polynomial.polypow(one, 2.5), "Power must be a non-negative integer.", "polypow(c, 2.5)");
            Raises<ValueError>(() => P.polynomial.polypow(one, double.NaN), "cannot convert float NaN to integer", "polypow(c, nan)");
            Raises<OverflowException>(() => P.polynomial.polypow(one, double.PositiveInfinity), "cannot convert float infinity to integer", "polypow(c, inf)");
            Raises<ValueError>(() => P.chebyshev.chebpow(one, 17), "Power is too large", "chebpow(c, 17) at the default maxpower 16");
            Raises<ValueError>(() => P.chebyshev.chebpow(one, 3, maxpower: 2), "Power is too large", "chebpow(c, 3, maxpower=2)");
            Raises<TypeError>(() => P.polynomial.polyfromroots(2.5), "object of type 'float' has no len()", "polyfromroots(2.5)");
            Raises<TypeError>(() => P.polynomial.polyfromroots(NDArray.Scalar(2.5)), "len() of unsized object", "polyfromroots(0-d)");
            Raises<ValueError>(() => P.polynomial.polyfromroots(new object[] { true, false }), "Coefficient arrays have no common type", "polyfromroots([True, False])");
            Raises<ValueError>(() => P.polynomial.polymul(Array.Empty<object>(), one), "Coefficient array is empty", "polymul([], [1.])");
            Raises<ValueError>(() => P.polynomial.polymulx(new object[] { new object[] { 1.0, 2.0 } }), "Coefficient array is not 1-d", "polymulx([[1., 2.]])");
        }

        // =========================================================================================================
        //  NumSharp-only facts
        // =========================================================================================================

        /// <summary>
        ///     The C# spellings of <c>{p}pow</c>: an <see cref="int"/> binds the int overload, a <see cref="double"/> the
        ///     double overload (NumPy's float pow), and a <see cref="long"/> — which converts implicitly to double, never
        ///     to int — the double overload too, with the same answer; <c>maxpower</c> is optional, named, and
        ///     <c>null</c> is NumPy's <c>None</c> (no limit) for every basis.
        /// </summary>
        [TestMethod]
        public void Pow_CSharpSpellings_BindNumPysSemantics()
        {
            var P = np.polynomial;
            var c = np.array(new[] { 1.0, 2.0 });
            var expected = new[] { 1.0, 4.0, 4.0 };
            P.polynomial.polypow(c, 2).ToArray<double>().Should().Equal(expected, "int");
            P.polynomial.polypow(c, 2.0).ToArray<double>().Should().Equal(expected, "double");
            P.polynomial.polypow(c, 2L).ToArray<double>().Should().Equal(expected, "long binds the double overload");
            P.chebyshev.chebpow(c, 17, maxpower: null).size.Should().Be(18, "maxpower=None lifts chebpow's default limit of 16");
            P.legendre.legpow(c, 3, maxpower: 3).size.Should().Be(4, "maxpower bounds the power inclusively");
            ((Action)(() => P.legendre.legpow(c, 17))).Should().Throw<ValueError>().WithMessage("Power is too large");
        }

        /// <summary>
        ///     Decimal (NumSharp only — NumPy has no decimal dtype) keeps its dtype through the whole family and computes
        ///     the statements NumPy would, in System.Decimal arithmetic: exact where the values are (products, integer
        ///     roots, conversions), 28-digit where a division is not. Compared by VALUE: System.Decimal can produce a
        ///     negative zero (<c>0.75m + -0.75m</c>), which equals zero.
        /// </summary>
        [TestMethod]
        public void Decimal_SeriesAlgebra_KeepsDecimal()
        {
            var P = np.polynomial;
            void Is(NDArray r, decimal[] values, string because)
            {
                r.dtype.name.Should().Be("decimal", because);
                r.ToArray<decimal>().Should().Equal(values, because);
            }
            Is(P.polynomial.polymul(np.array(new[] { 1.5m, 2m }), np.array(new[] { 3m, -1m })), new[] { 4.5m, 4.5m, -2m }, "polymul");
            Is(P.legendre.legmul(np.array(new[] { 1.5m, -2.25m, 0.5m }), np.array(new[] { 0.5m, 1m })), new[] { 0m, 0.575m, -1.25m, 0.3m }, "legmul");
            Is(P.polynomial.polyfromroots(np.array(new[] { 1m, 2m, 3m })), new[] { -6m, 11m, -6m, 1m }, "polyfromroots");
            Is(P.chebyshev.chebfromroots(np.array(new[] { 1m, 2m, 3m })), new[] { -9m, 11.75m, -3m, 0.25m }, "chebfromroots");
            Is(P.hermite.hermfromroots(np.array(new[] { 1m, 2m, 3m })), new[] { -9m, 6.25m, -1.5m, 0.125m }, "hermfromroots");
            Is(P.polynomial.polypow(np.array(new[] { 1m, 2m }), 3), new[] { 1m, 6m, 12m, 8m }, "polypow");
            Is(P.chebyshev.cheb2poly(np.array(new[] { 1m, 2m, 3m })), new[] { -2m, 2m, 6m }, "cheb2poly");
            Is(P.chebyshev.poly2cheb(np.array(new[] { 1m, 2m, 3m })), new[] { 2.5m, 2m, 1.5m }, "poly2cheb");
            Is(P.laguerre.lagmulx(np.array(new[] { 1m, 2m, 3m })), new[] { -1m, -1m, 11m, -9m }, "lagmulx");
            // A division that does not terminate rounds at 28 digits; the rebuilt dividend agrees to that precision.
            var dividend = np.array(new[] { 1m, 2m, 3m, 4m });
            var divisor = np.array(new[] { 0.5m, 1.5m });
            var (q, r) = P.polynomial.polydiv(dividend, divisor);
            q.dtype.name.Should().Be("decimal");
            r.dtype.name.Should().Be("decimal");
            var rebuilt = P.polynomial.polyadd(P.polynomial.polymul(q, divisor), r).ToArray<decimal>();
            rebuilt.Zip(new[] { 1m, 2m, 3m, 4m }, (x, y) => System.Math.Abs(x - y)).Should().OnlyContain(d => d < 1e-25m);
        }

        /// <summary>Char (NumSharp only) converts through as_series exactly as uint16 does: to float64.</summary>
        [TestMethod]
        public void Char_ConvertsLikeUInt16()
        {
            var P = np.polynomial;
            var viaChar = P.polynomial.polymul(np.array(new[] { 'a', 'b' }), np.array(new[] { 'c' }));
            var viaU16 = P.polynomial.polymul(np.array(new ushort[] { 97, 98 }), np.array(new ushort[] { 99 }));
            viaChar.dtype.name.Should().Be("float64");
            Hex(viaChar).Should().Be(Hex(viaU16));
            Hex(P.legendre.legfromroots(np.array(new[] { 'a', 'c' }))).Should().Be(Hex(P.legendre.legfromroots(np.array(new ushort[] { 97, 99 }))));
        }

        /// <summary>
        ///     The per-thread arena: a call whose temporaries outgrow the 64 KB base block (a 6000 x 6000 product: an
        ///     80 KB convolution) leaves the thread holding only the base block again, and concurrent calls on many
        ///     threads return exactly what the same calls return one at a time (each thread bump-allocates its own).
        /// </summary>
        [TestMethod]
        public void Arena_ReleasesGrowthBlocks_AndIsPerThread()
        {
            var P = np.polynomial;
            var big = np.ones(6000);
            using (var r = P.polynomial.polymul(big, big))
                r.size.Should().Be(11999);
            PolyArena.PeekCurrent.Should().NotBeNull("the call entered this thread's arena");
            PolyArena.PeekCurrent.BlockCount.Should().Be(1, "the outermost exit frees every growth block");

            Func<int, string>[] work =
            {
                s => Hex(P.legendre.legmul(np.arange(3 + s % 7).astype(np.float64), np.arange(2 + s % 5).astype(np.float64))),
                s => { var (q, r) = P.hermite.hermdiv(np.arange(9 + s % 4).astype(np.float64), np.array(new[] { 1.0, 2.0, 0.5 })); return Hex(q) + "/" + Hex(r); },
                s => Hex(P.chebyshev.chebpow(np.array(new[] { 0.5, -1.25, 2.0 }), 2 + s % 4)),
                s => Hex(P.laguerre.lagfromroots(np.linspace(-1.0, 1.0, 3 + s % 9))),
                s => Hex(P.hermite_e.poly2herme(np.arange(5 + s % 6).astype(np.float64))),
            };
            var expected = Enumerable.Range(0, 64).Select(s => work[s % work.Length](s)).ToArray();
            var actual = new string[64];
            Parallel.For(0, 64, new ParallelOptions { MaxDegreeOfParallelism = 8 }, s => actual[s] = work[s % work.Length](s));
            actual.Should().Equal(expected);
        }

        /// <summary>
        ///     A long float64 product without the OpenBLAS backend: NumPy reduces each output position with cblas
        ///     <c>ddot</c>'s vector kernel once a dot reaches 16 terms, whose summation order no managed kernel
        ///     reproduces, so the managed product is ACCURATE (every coefficient within a few ULP of the exact sum), not
        ///     byte-identical — byte parity for these comes with <c>NumSharp.Interop.OpenBLAS</c> (the host-pinned
        ///     <c>polyalgebra_parity</c> tier).
        /// </summary>
        [TestMethod]
        public void LongProduct_WithoutBackend_IsAccurate()
        {
            var engine = BackendFactory.GetEngine();
            var saved = engine.Blas;
            engine.Blas = null;
            try
            {
                var rng = new System.Random(20260930);
                double[] a = Enumerable.Range(0, 200).Select(_ => rng.NextDouble() * 2 - 1).ToArray();
                double[] b = Enumerable.Range(0, 150).Select(_ => rng.NextDouble() * 2 - 1).ToArray();
                var r = np.polynomial.polynomial.polymul(a, b).ToArray<double>();
                r.Length.Should().Be(349);
                for (int k = 0; k < r.Length; k++)
                {
                    // The reference coefficient by Ogita-Rump-Oishi's Dot2 (an error-free product via FMA plus TwoSum):
                    // as accurate as a dot computed in twice the precision, then rounded once. A floating dot of n terms is
                    // within gamma_n * sum|products| (n * 2^-53, the products' roundings included) of the exact value, and
                    // the reference's own rounding adds 2^-53 * |value| at most — hence (n + 1) units of 2^-53, with margin.
                    double s = 0, comp = 0, scale = 0;
                    int terms = 0;
                    for (int i = System.Math.Max(0, k - b.Length + 1); i <= System.Math.Min(k, a.Length - 1); i++)
                    {
                        double x = a[i], y = b[k - i];
                        double prod = x * y, prodErr = System.Math.FusedMultiplyAdd(x, y, -prod);
                        double t = s + prod, z = t - s, sumErr = (s - (t - z)) + (prod - z);
                        s = t;
                        comp += prodErr + sumErr;
                        scale += System.Math.Abs(prod);
                        terms++;
                    }
                    double reference = s + comp;
                    System.Math.Abs(r[k] - reference).Should().BeLessThanOrEqualTo((terms + 1) * 1.2e-16 * scale, $"coefficient {k}");
                }
            }
            finally
            {
                engine.Blas = saved;
            }
        }

        /// <summary>
        ///     The fused chebmulx kernel's vector stage (float64 / float32 halved by a multiply by 0.5, float16 by
        ///     <c>HalfHalveOnGrid</c> and a plain float32 add, complex128 by the prepared Smith divisor) against the same
        ///     kernel emitted scalar-only, whose every op is the literal NumPy statement op — byte for byte (a NaN's payload
        ///     aside, which no gate compares) over every series dtype, lengths across the vector widths, finite / special /
        ///     subnormal values, and both layouts (c apart from prd, and c already converted into prd[1:]). No emission
        ///     falls back to scalar code.
        /// </summary>
        [TestMethod]
        public unsafe void ChebMulx_FusedKernel_VectorStage_MatchesScalarEmission()
        {
            long fallbacks = NumSharp.Backends.Kernels.DirectILKernelGenerator.PolyCalcVectorFallbacks;
            var rng = new System.Random(20260930);
            double[] specials = { 0.0, -0.0, double.PositiveInfinity, double.NegativeInfinity, double.NaN, 5e-324, -1e-310,
                                  2.2250738585072014e-308, 1.7976931348623157e308, 1.0, -3.0, 1e-45, 6e-8, 65504, 3.4028234663852886e38 };
            foreach (var t in new[] { NPTypeCode.Double, NPTypeCode.Single, NPTypeCode.Half, NPTypeCode.Complex, NPTypeCode.Decimal })
            {
                int size = NumSharp.Backends.Kernels.DirectILKernelGenerator.GetTypeSize(t);
                var vector = NumSharp.Backends.Kernels.DirectILKernelGenerator.GetPolyChebMulxKernel(t);
                var scalar = NumSharp.Backends.Kernels.DirectILKernelGenerator.CompilePolyChebMulx(t, allowVector: false);
                for (int n = 2; n <= 70; n++)
                for (int rep = 0; rep < 3; rep++)
                {
                    var c = new byte[n * size];
                    fixed (byte* cp = c)
                    {
                        for (int i = 0; i < n; i++)
                        {
                            double re = rep == 0 ? rng.NextDouble() * 200 - 100 : specials[rng.Next(specials.Length)];
                            double im = rep == 0 ? rng.NextDouble() * 200 - 100 : specials[rng.Next(specials.Length)];
                            if (rep == 2) { re *= 1e-300; im *= 1e-300; }
                            byte* p = cp + i * size;
                            switch (t)
                            {
                                case NPTypeCode.Double: *(double*)p = re; break;
                                case NPTypeCode.Single: *(float*)p = rep == 2 ? (float)(re * 1e263) : (float)re; break;
                                case NPTypeCode.Half: *(Half*)p = rep == 2 ? (Half)(re * 1e295) : (Half)re; break;
                                case NPTypeCode.Complex: *(Complex*)p = new Complex(re, im); break;
                                default: *(decimal*)p = double.IsFinite(re) && System.Math.Abs(re) < 1e20 ? (decimal)System.Math.Round(re, 6) : rng.Next(-99, 99); break;
                            }
                        }
                    }
                    foreach (bool inPlace in new[] { false, true })
                    {
                        byte[] got = RunChebMulx(vector, c, n, size, inPlace), want = RunChebMulx(scalar, c, n, size, inPlace);
                        for (int k = 0; k <= n; k++)
                            SameOrBothNaN(got, want, k, t, size).Should().BeTrue($"{t} n={n} rep={rep} inPlace={inPlace} prd[{k}]");
                    }
                }
            }
            NumSharp.Backends.Kernels.DirectILKernelGenerator.PolyCalcVectorFallbacks.Should().Be(fallbacks, "every dtype's vector stage emits");
        }

        /// <summary>
        ///     chebmulx of a float16 series holding EVERY bit pattern (each as c[j-1] of one output and c[j+1] of another,
        ///     random neighbours) against NumPy's statements modelled with its HALF loop — <c>tmp = c[1:]/2</c> as
        ///     <c>(half)((float)x / 2)</c>, <c>prd[0:-2] += tmp</c> as <c>(half)((float)a + (float)b)</c>, <c>c[0]*0</c> as
        ///     scalarmath's <c>(half)((float)c0 * 0)</c> — byte for byte, NaNs as NaN. Pins the subnormal rounding of the
        ///     halves (<c>HalfHalveOnGrid</c>'s magic-number step) and the single rounding of the sum.
        /// </summary>
        [TestMethod]
        public void ChebMulx_Float16_EveryBitPattern_IsTheHalfLoop()
        {
            var rng = new System.Random(20260930);
            int n = 2 * 65536 + 1;
            var c = new Half[n];
            for (int i = 0; i < n - 1; i++)
                c[i] = BitConverter.UInt16BitsToHalf((ushort)(i % 2 == 0 ? i / 2 : rng.Next(65536)));
            c[n - 1] = (Half)1.0;   // a nonzero last term: as_series' trim keeps the whole series
            var got = np.polynomial.chebyshev.chebmulx(np.array(c)).ToArray<Half>();

            var tmp = new Half[n - 1];
            for (int i = 0; i < n - 1; i++) tmp[i] = (Half)((float)c[i + 1] / 2f);
            var prd = new Half[n + 1];
            prd[0] = (Half)((float)c[0] * 0f);
            prd[1] = c[0];
            Array.Copy(tmp, 0, prd, 2, n - 1);
            for (int i = 0; i < n - 1; i++) prd[i] = (Half)((float)prd[i] + (float)tmp[i]);

            got.Length.Should().Be(n + 1);
            for (int k = 0; k <= n; k++)
                if (!(Half.IsNaN(got[k]) && Half.IsNaN(prd[k])))
                    BitConverter.HalfToUInt16Bits(got[k]).Should().Be(BitConverter.HalfToUInt16Bits(prd[k]), $"prd[{k}]");
        }

        /// <summary>Runs a fused chebmulx kernel over a copy of <paramref name="c"/>.</summary>
        /// <param name="k">The kernel.</param>
        /// <param name="c">The series bytes (n elements).</param>
        /// <param name="n">Terms.</param>
        /// <param name="size">Itemsize.</param>
        /// <param name="inPlace">Read c from prd[1:] (the converted-series layout) instead of a separate buffer.</param>
        /// <returns>prd's bytes (n + 1 elements).</returns>
        private static unsafe byte[] RunChebMulx(NumSharp.Backends.Kernels.PolyChebMulxKernel k, byte[] c, int n, int size, bool inPlace)
        {
            var prd = new byte[(n + 1) * size];
            fixed (byte* pp = prd)
            fixed (byte* cp = c)
            {
                if (inPlace)
                {
                    Buffer.MemoryCopy(cp, pp + size, (long)n * size, (long)n * size);
                    k(pp + size, n, pp);
                }
                else
                {
                    k(cp, n, pp);
                }
            }
            return prd;
        }

        /// <summary>Whether element <paramref name="i"/> of two result buffers holds the same bytes, or NaN in both
        ///     (complex: per component).</summary>
        /// <param name="a">One buffer.</param><param name="b">The other.</param><param name="i">Element index.</param>
        /// <param name="t">The dtype.</param><param name="size">Itemsize.</param>
        /// <returns>True when equal under NaN tokenization.</returns>
        private static unsafe bool SameOrBothNaN(byte[] a, byte[] b, int i, NPTypeCode t, int size)
        {
            if (a.AsSpan(i * size, size).SequenceEqual(b.AsSpan(i * size, size)))
                return true;
            fixed (byte* pa = a)
            fixed (byte* pb = b)
            {
                byte* x = pa + i * size, y = pb + i * size;
                static bool Same(double p, double q) => BitConverter.DoubleToInt64Bits(p) == BitConverter.DoubleToInt64Bits(q) || double.IsNaN(p) && double.IsNaN(q);
                return t switch
                {
                    NPTypeCode.Double => double.IsNaN(*(double*)x) && double.IsNaN(*(double*)y),
                    NPTypeCode.Single => float.IsNaN(*(float*)x) && float.IsNaN(*(float*)y),
                    NPTypeCode.Half => Half.IsNaN(*(Half*)x) && Half.IsNaN(*(Half*)y),
                    NPTypeCode.Complex => Same(((Complex*)x)->Real, ((Complex*)y)->Real) && Same(((Complex*)x)->Imaginary, ((Complex*)y)->Imaginary),
                    _ => false,
                };
            }
        }

        /// <summary>
        ///     [Misaligned] fromroots sorts its roots first (<c>roots.sort()</c>), and where a +0.0 and a -0.0 meet the
        ///     sort decides a zero coefficient's SIGN in the recurrence bases (their products scale arrays by NumPy
        ///     scalars; the convolving bases normalize zeros). NumPy's float64 sort is x86-simd-sort, whose ordering of
        ///     equal keys follows its vector network and so the CPU it dispatches to (on this AVX2 host it keeps small
        ///     arrays' input order: <c>legfromroots([0.0, -0.0])</c> is <c>[1/3, +0.0, 2/3]</c>); NumSharp's sort puts
        ///     -0.0 first on every host, giving <c>[1/3, -0.0, 2/3]</c>. The values are equal; only the zero's sign differs.
        /// </summary>
        [TestMethod]
        [Misaligned]
        public void FromRoots_MixedSignedZeroRoots_SortNegativeZeroFirst()
        {
            var r = np.polynomial.legendre.legfromroots(np.array(new[] { 0.0, -0.0 }));
            // NumPy 2.4.2 on this AVX2 host: 555555555555d53f 0000000000000000 555555555555e53f.
            AssertBytes(r, "float64", new long[] { 3 }, "555555555555d53f0000000000000080555555555555e53f", "legfromroots([0.0, -0.0])");
            // Input order that already has -0.0 first: both libraries agree.
            AssertBytes(np.polynomial.legendre.legfromroots(np.array(new[] { -0.0, 0.0 })), "float64", new long[] { 3 },
                        "555555555555d53f0000000000000080555555555555e53f", "legfromroots([-0.0, 0.0])");
        }

        /// <summary>
        ///     A complex product of NON-FINITE values without the OpenBLAS backend is NumPy's, byte for byte. NumPy
        ///     reduces a complex convolution position with cblas <c>zdotu</c>, which sums the four products separately
        ///     and returns C99 complex <c>re + im*_Complex_I</c> — the real part picks up <c>im*0</c>, a NaN whenever the
        ///     imaginary part is infinite or NaN: <c>polymul([inf+0j, 1+1j], [1+0j, 2-1j])</c> is
        ///     <c>[nan+nanj, nan-infj, 3+1j]</c>, where the plain four sums give <c>[inf+nanj, inf-infj, 3+1j]</c> (this
        ///     test pinned that divergence as [Misaligned] until the managed dot learned the result construction). A
        ///     ONE-term factor is np.convolve's reversed one-element kernel (<c>v[::-1]</c>, stride -16, which cblas
        ///     refuses), so CDOUBLE_dot's plain loop runs and no such NaN appears: <c>polymul([inf+0j, 1+1j], [1+0j])</c>
        ///     is <c>[inf+nanj, 1+1j]</c>.
        /// </summary>
        [TestMethod]
        public void ComplexNonFiniteProduct_WithoutBackend_MatchesZdotu()
        {
            var engine = BackendFactory.GetEngine();
            var saved = engine.Blas;
            engine.Blas = null;
            try
            {
                var inf = new Complex(double.PositiveInfinity, 0);
                var r = np.polynomial.polynomial.polymul(np.array(new[] { inf, new Complex(1, 1) }),
                                                         np.array(new[] { new Complex(1, 0), new Complex(2, -1) }));
                AssertBytes(r, "complex128", new long[] { 3 },
                            "000000000000f8ff000000000000f8ff000000000000f8ff000000000000f0ff0000000000000840000000000000f03f",
                            "zdotu's C99 result: an infinite / NaN imaginary part turns the real part into NaN");
                var one = np.polynomial.polynomial.polymul(np.array(new[] { inf, new Complex(1, 1) }), np.array(new[] { new Complex(1, 0) }));
                AssertBytes(one, "complex128", new long[] { 2 }, "000000000000f07f000000000000f8ff000000000000f03f000000000000f03f",
                            "a one-term factor takes CDOUBLE_dot's plain loop");
                var pw = np.polynomial.polynomial.polypow(np.array(new[] { new Complex(double.PositiveInfinity, 1) }), 3);
                AssertBytes(pw, "complex128", new long[] { 1 }, "000000000000f8ff000000000000f07f",
                            "a power of a one-term series multiplies one-element kernels: the plain loop each time");
            }
            finally
            {
                engine.Blas = saved;
            }
        }
    }
}
