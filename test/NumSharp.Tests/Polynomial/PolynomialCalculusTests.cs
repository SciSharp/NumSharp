using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Threading.Tasks;
using NumSharp.Backends.Kernels;

namespace NumSharp.Tests.Polynomial
{
    /// <summary>
    ///     The <c>numpy.polynomial</c> calculus family (plan U4): <c>{p}der</c> and <c>{p}int</c> of the six bases.
    ///     Three kinds of test live here: NumPy 2.4.2's own <c>TestIntegral</c>/<c>TestDerivative</c> suites ported
    ///     (the algebraic properties, made BITWISE where NumPy's statement sequence makes them exact); byte dumps NumPy
    ///     produced for every dtype family (the probe inputs are spelled out next to each test); and the object-level
    ///     facts the byte oracle (<c>polycalc.jsonl</c>) cannot see — kernel structure (vector lanes, blocks, direct
    ///     load vs house copy, thread safety), decimal and char, which NumPy has no dtype for.
    /// </summary>
    [TestClass]
    public class PolynomialCalculusTests
    {
        /// <summary>One basis's calculus pair plus its evaluator (for the lbnd property).</summary>
        /// <param name="Name">The basis prefix NumPy uses (<c>poly</c>, <c>cheb</c>, ...).</param>
        /// <param name="Der"><c>{p}der(c, m, scl, axis)</c>.</param>
        /// <param name="Int"><c>{p}int(c, m, k, lbnd, scl, axis)</c>.</param>
        /// <param name="Val"><c>{p}val(x, c, tensor)</c>.</param>
        private sealed record Basis(string Name, Func<object, int, object, int, NDArray> Der,
            Func<object, int, object, object, object, int, NDArray> Int, Func<object, NDArray, bool, NDArray> Val);

        /// <summary>The six bases, in NumPy's module order.</summary>
        private static readonly Basis[] Bases =
        {
            new("poly", np.polynomial.polynomial.polyder, np.polynomial.polynomial.polyint, np.polynomial.polynomial.polyval),
            new("cheb", np.polynomial.chebyshev.chebder, np.polynomial.chebyshev.chebint, np.polynomial.chebyshev.chebval),
            new("leg", np.polynomial.legendre.legder, np.polynomial.legendre.legint, np.polynomial.legendre.legval),
            new("lag", np.polynomial.laguerre.lagder, np.polynomial.laguerre.lagint, np.polynomial.laguerre.lagval),
            new("herm", np.polynomial.hermite.hermder, np.polynomial.hermite.hermint, np.polynomial.hermite.hermval),
            new("herme", np.polynomial.hermite_e.hermeder, np.polynomial.hermite_e.hermeint, np.polynomial.hermite_e.hermeval),
        };

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

        /// <summary><c>list(range(j))</c> — NumPy's integration constants, as a Python list of ints.</summary>
        /// <param name="j">The count.</param>
        /// <returns>The list.</returns>
        private static object[] Range(int j) => Enumerable.Range(0, j).Select(v => (object)v).ToArray();

        /// <summary>NumPy's <c>polyutils.trimcoef(c)</c> of a float64 result (trailing zeros dropped, one kept).</summary>
        /// <param name="a">A 1-D float64 array.</param>
        /// <returns>The trimmed values.</returns>
        private static double[] Trim(NDArray a)
        {
            var v = a.ToArray<double>();
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
        //  NumPy 2.4.2's own suites (numpy/polynomial/tests/test_{polynomial,chebyshev,...}.py), ported
        // =========================================================================================================

        /// <summary>
        ///     <c>TestIntegral</c>'s "integration of zero polynomial": <c>{p}int([0], m=i, k=[0]*(i-2)+[1])</c> for
        ///     i = 2..4 — the n == 1 zero branch keeps the series one coefficient long until the unit constant lands,
        ///     then ONE growing order runs. Expected values are the basis's integral of 1 (exact, so compared exactly).
        /// </summary>
        [TestMethod]
        public void Int_OfZeroSeries_WithAUnitConstant_IsTheIntegralOfOne()
        {
            var expected = new Dictionary<string, double[]>
            {
                ["poly"] = new[] { 0.0, 1.0 }, ["cheb"] = new[] { 0.0, 1.0 }, ["leg"] = new[] { 0.0, 1.0 },
                ["lag"] = new[] { 1.0, -1.0 }, ["herm"] = new[] { 0.0, 0.5 }, ["herme"] = new[] { 0.0, 1.0 },
            };
            foreach (var b in Bases)
                for (int i = 2; i < 5; i++)
                {
                    var k = Enumerable.Repeat((object)0, i - 2).Append(1).ToArray();
                    using var res = b.Int(new object[] { 0 }, i, k, null, null, 0);
                    res.ToArray<double>().Should().Equal(expected[b.Name], $"{b.Name} m={i}");
                }
        }

        /// <summary>
        ///     <c>TestIntegral</c>'s lbnd check: the integral's value AT lbnd is the integration constant, for every
        ///     basis series of degree 0..4 (NumPy converts a power series first; any series has the property).
        /// </summary>
        [TestMethod]
        public void Int_WithLbnd_TakesTheConstantAtLbnd()
        {
            foreach (var b in Bases)
                for (int i = 0; i < 5; i++)
                {
                    using var res = b.Int(Unit(i), 1, new object[] { i }, -1, null, 0);
                    using var at = b.Val(-1, res, true);
                    System.Math.Abs(at.GetDouble() - i).Should().BeLessThan(1.5e-7, $"{b.Name} i={i}");
                }
        }

        /// <summary>
        ///     <c>TestIntegral</c>'s four "multiple integrations" checks — default k, defined k, lbnd, scaling:
        ///     <c>{p}int(pol, m=j, ...)</c> equals j successive single-order calls. NumPy runs the same statements in
        ///     the same order either way, so the equality is BITWISE, and the one-buffer multi-order kernel must keep it.
        /// </summary>
        [TestMethod]
        public void Int_MultipleOrders_EqualRepeatedSingleOrders_Bitwise()
        {
            foreach (var b in Bases)
                for (int i = 0; i < 5; i++)
                    for (int j = 2; j < 5; j++)
                    {
                        foreach (var (label, lbnd, scl, withK) in new (string, object, object, bool)[]
                                 {
                                     ("default k", null, null, false), ("defined k", null, null, true),
                                     ("lbnd", -1, null, true), ("scl", null, 2, true),
                                 })
                        {
                            NDArray tgt = np.array(Unit(i).Select(v => Convert.ToDouble(v)).ToArray());
                            for (int kk = 0; kk < j; kk++)
                            {
                                var next = b.Int(tgt, 1, withK ? new object[] { kk } : null, lbnd, scl, 0);
                                tgt.Dispose();
                                tgt = next;
                            }
                            using (tgt)
                            using (var res = b.Int(Unit(i), j, withK ? Range(j) : null, lbnd, scl, 0))
                                Hex(res).Should().Be(Hex(tgt), $"{b.Name} {label} i={i} j={j}");
                        }
                    }
        }

        /// <summary>
        ///     <c>TestIntegral</c>'s closed-form checks for the POWER basis (the others go through poly2X, which the
        ///     property tests above cover): <c>polyint(x^i, k=[i]) = [i, 0.., 1/(i+1)]</c>, and <c>2/(i+1)</c> with scl=2.
        /// </summary>
        [TestMethod]
        public void PolyInt_SingleOrder_IsTheClosedForm()
        {
            var poly = np.polynomial.polynomial;
            for (int i = 0; i < 5; i++)
            {
                double scl = i + 1;
                var tgt = new[] { (double)i }.Concat(Enumerable.Repeat(0.0, i)).Append(1 / scl).ToArray();
                using (var res = poly.polyint(Unit(i), 1, new object[] { i }))
                    AlmostEqual(Trim(res), tgt, $"polyint i={i}");
                tgt[^1] = 2 / scl;
                using (var res = poly.polyint(Unit(i), 1, new object[] { i }, scl: 2))
                    AlmostEqual(Trim(res), tgt, $"polyint scl=2 i={i}");
            }
        }

        /// <summary><c>TestDerivative</c>: the zeroth derivative is the series itself (a K-order copy, not a view).</summary>
        [TestMethod]
        public void Der_ZerothOrder_ReturnsACopyOfTheSeries()
        {
            foreach (var b in Bases)
                for (int i = 0; i < 5; i++)
                {
                    using var src = np.array(Unit(i).Select(v => Convert.ToDouble(v)).ToArray());
                    using var res = b.Der(src, 0, null, 0);
                    res.ToArray<double>().Should().Equal(src.ToArray<double>(), $"{b.Name} i={i}");
                    res.flags.owndata.Should().BeTrue($"{b.Name}: m == 0 returns NumPy's np.array(c, copy=True)");
                }
        }

        /// <summary>
        ///     <c>TestDerivative</c>: differentiation inverts integration, plain and with reciprocal scales
        ///     (<c>{p}der({p}int(tgt, m=j, scl=2), m=j, scl=0.5)</c>).
        /// </summary>
        [TestMethod]
        public void Der_InvertsInt_PlainAndScaled()
        {
            foreach (var b in Bases)
                for (int i = 0; i < 5; i++)
                    for (int j = 2; j < 5; j++)
                    {
                        var tgt = Unit(i).Select(v => Convert.ToDouble(v)).ToArray();
                        using (var integ = b.Int(Unit(i), j, null, null, null, 0))
                        using (var res = b.Der(integ, j, null, 0))
                            AlmostEqual(Trim(res), tgt, $"{b.Name} i={i} j={j}");
                        using (var integ = b.Int(Unit(i), j, null, null, 2, 0))
                        using (var res = b.Der(integ, j, 0.5, 0))
                            AlmostEqual(Trim(res), tgt, $"{b.Name} scaled i={i} j={j}");
                    }
        }

        /// <summary>
        ///     <c>test_{p}int_axis</c> / <c>test_{p}der_axis</c>: along either axis of a 2-D series the result is the
        ///     per-lane 1-D result stacked. For float64 the 1-D (scalarmath) and N-D (ufunc) arithmetics coincide, so
        ///     the equality is BITWISE here (NumPy only asserts closeness).
        /// </summary>
        [TestMethod]
        public void Axis_EqualsTheLanesComputedOneByOne_Bitwise()
        {
            using var c2d = np.sin(np.arange(12.0)).reshape(3, 4);
            foreach (var b in Bases)
            {
                foreach (int axis in new[] { 0, 1 })
                {
                    int lanes = axis == 0 ? 4 : 3;
                    foreach (var (label, call, lane) in new (string, Func<NDArray>, Func<NDArray, NDArray>)[]
                             {
                                 ("der", () => b.Der(c2d, 1, null, axis), l => b.Der(l, 1, null, 0)),
                                 ("int", () => b.Int(c2d, 1, null, null, null, axis), l => b.Int(l, 1, null, null, null, 0)),
                                 ("int k=3", () => b.Int(c2d, 1, 3, null, null, axis), l => b.Int(l, 1, 3, null, null, 0)),
                             })
                    {
                        using var res = call();
                        for (int q = 0; q < lanes; q++)
                        {
                            using var src = axis == 0 ? c2d[$":, {q}"] : c2d[$"{q}, :"];
                            using var want = lane(src);
                            using var got = axis == 0 ? res[$":, {q}"] : res[$"{q}, :"];
                            Hex(got).Should().Be(Hex(want), $"{b.Name} {label} axis={axis} lane={q}");
                        }
                    }
                }
            }
        }

        // =========================================================================================================
        //  NumPy byte dumps, per dtype family
        // =========================================================================================================

        /// <summary>c7 = [1.5, -2.25, 0.75, 3.0, -1.0, 0.5, 2.0] — the float64/float32/float16 probe series.</summary>
        private static readonly double[] C7 = { 1.5, -2.25, 0.75, 3.0, -1.0, 0.5, 2.0 };

        /// <summary>
        ///     float64: <c>{p}der(c7, m=2, scl=0.5)</c> and <c>{p}int(c7, m=2, k=[1.5, -2.0], lbnd=0.25, scl=0.5)</c> —
        ///     two fused orders with a scale, two growing orders with constants and a lower bound.
        /// </summary>
        [TestMethod]
        public void Float64_DerAndInt_MatchNumPyBitForBit()
        {
            var der = new Dictionary<string, string>
            {
                ["poly"] = "000000000000d83f000000000000124000000000000008c000000000000004400000000000002e40",
                ["cheb"] = "00000000006047400000000000804040000000000000554000000000000024400000000000004e40",
                ["leg"] = "0000000000202140000000000080304000000000002042400000000000801f400000000000c04840",
                ["lag"] = "0000000000800f400000000000000540000000000000f83f000000000000f23f000000000000e03f",
                ["herm"] = "000000000000f83f000000000000324000000000000028c000000000000024400000000000004e40",
                ["herme"] = "000000000000d83f000000000000124000000000000008c000000000000004400000000000002e40",
            };
            var integ = new Dictionary<string, string>
            {
                ["poly"] = "f6633f76576d01c0977229975282e53f000000000000c83f000000000000b8bf000000000000903f333333333333a33f11111111111181bf188661188661683f922449922449823f",
                ["cheb"] = "532ee5521e7901c05ccaa55cca83e53fabaaaaaaaaaaaa3f555555555555b1bfefeeeeeeeeee923fbcbbbbbbbbbb7b3f542ee5522ee582bf188661188661483f922449922449623f",
                ["leg"] = "666666ae1e2e01c02cbee22b26a1e43f9ba6699aa669ba3f24d7a0c762cfb1bfcd9985fc536e8f3f134001144001843f40bcac6e769581bf91e15e05b3a44c3f155001155001653f",
                ["lag"] = "1cc771cc4b4af3bfe8799ee7bd77febf000000000000fb3f000000000000c8bf000000000000f9bf000000000000f63f0000000000000000000000000000ecbf000000000000e03f",
                ["herm"] = "542ee5521eee1ec07fecc77e2c9f1c40000000000000a83f00000000000098bf000000000000703f333333333333833f11111111111161bf188661188661483f922449922449623f",
                ["herme"] = "f6633f7677be0ac0a65ccaa5141e0140000000000000c83f000000000000b8bf000000000000903f333333333333a33f11111111111181bf188661188661683f922449922449823f",
            };
            using var c = np.array(C7);
            foreach (var b in Bases)
            {
                using (var r = b.Der(c, 2, 0.5, 0))
                    AssertBytes(r, "float64", new long[] { 5 }, der[b.Name], b.Name + "der");
                using (var r = b.Int(c, 2, new object[] { 1.5, -2.0 }, 0.25, 0.5, 0))
                    AssertBytes(r, "float64", new long[] { 9 }, integ[b.Name], b.Name + "int");
            }
        }

        /// <summary>
        ///     float32 (NEP 50): a Python-float scl is WEAK — converted to float32, the product float32 — while a 0-d
        ///     float64 array scl is STRONG: <c>c *= np.array(0.3)</c> multiplies in float64 and casts back, which moves
        ///     last bits. The integral's <c>k - {p}val(0.3, tmp)</c> stays float32 (weak k and lbnd).
        /// </summary>
        [TestMethod]
        public void Float32_WeakPythonScl_StaysFloat32_StrongZeroDScl_MultipliesInFloat64()
        {
            var weak = new Dictionary<string, string>
            {
                ["poly"] = "cdcc2cbf6766e63ecdcc2c409a9999bf0000403f67666640", ["cheb"] = "9b9931406766b640ceccdc409a9999400000c03f6766e640",
                ["leg"] = "0200c03e9a99c93f0100a84067660640cdccac3f3433d340", ["lag"] = "676666bf9a99c9bfcdccacbf6666e6be000040bf9a9919bf",
                ["herm"] = "cdccacbf6766663fcdccac409a9919c00000c03f6766e640", ["herme"] = "cdcc2cbf6766e63ecdcc2c409a9999bf0000403f67666640",
            };
            var strong = new Dictionary<string, string>
            {
                ["poly"] = "cdcc2cbf6666e63ecccc2c409a9999bf0000403f67666640", ["cheb"] = "999931406766b640ccccdc409a9999400000c03f6766e640",
                ["leg"] = "feffbf3e9999c93f0000a84067660640cdccac3f3433d340", ["lag"] = "656666bf9999c9bfccccacbf6666e6be000040bf9a9919bf",
                ["herm"] = "cdccacbf6666663fccccac409a9919c00000c03f6766e640", ["herme"] = "cdcc2cbf6666e63ecccc2c409a9999bf0000403f67666640",
            };
            var integ = new Dictionary<string, string>
            {
                ["poly"] = "9dbd85be0000c03f000090bf0000803e0000403fcdcc4cbeabaaaa3d2549923e",
                ["cheb"] = "d6d247bf0000903f0000a8bf5655953e0000a03e9a9999beabaa2a3d2549123e",
                ["leg"] = "cf0a0bbfcdccac3f6edb96bf5bb0853ee627c43e7ba887be8c2e3a3dd9891d3e",
                ["lag"] = "62006f3f000070c00000404000001040000080c00000c03f0000c03f000000c0",
                ["herm"] = "f37075420000403f000010bf0000003e0000c03ecdccccbdabaa2a3d2549123e",
                ["herme"] = "9df1de400000c03f000090bf0000803e0000403fcdcc4cbeabaaaa3d2549923e",
            };
            using var c = np.array(C7.Select(v => (float)v).ToArray());
            using var strongScl = np.array(0.3);
            foreach (var b in Bases)
            {
                using (var r = b.Der(c, 1, 0.3, 0))
                    AssertBytes(r, "float32", new long[] { 6 }, weak[b.Name], b.Name + " weak");
                using (var r = b.Der(c, 1, strongScl, 0))
                    AssertBytes(r, "float32", new long[] { 6 }, strong[b.Name], b.Name + " strong");
                using (var r = b.Int(c, 1, new object[] { 0.1 }, 0.3, null, 0))
                    AssertBytes(r, "float32", new long[] { 8 }, integ[b.Name], b.Name + " int");
            }
        }

        /// <summary>
        ///     float16: every statement is NumPy's HALF loop (widen, float32 op, round to half) and every weak int
        ///     constant (<c>2*j</c>, <c>j+1</c>) is converted to float16 first — <c>{p}der(c7h, 2)</c>,
        ///     <c>{p}int(c7h, 1, k=[0.1], lbnd=0.5)</c>.
        /// </summary>
        [TestMethod]
        public void Float16_EveryStatementRoundsLikeNumPysHalfLoop()
        {
            var der = new Dictionary<string, string>
            {
                ["poly"] = "003e804c00ca00498053", ["cheb"] = "d8592058405d0051805b", ["leg"] = "485020548858e04f305a",
                ["lag"] = "e04b4049004680440040", ["herm"] = "0046805400d20051805b", ["herme"] = "003e804c00ca00498053",
            };
            var integ = new Dictionary<string, string>
            {
                ["poly"] = "1ab7003e80bc0034003a66b2552d9234", ["cheb"] = "13b9803c40bdaa340035ccb455299230",
                ["leg"] = "2eb8663db7bc2e3421363db4d129ec30", ["lag"] = "f33b80c30042804000c4003e003e00c0",
                ["herm"] = "3c54003a80b80030003666ae55299230", ["herme"] = "6149003e80bc0034003a66b2552d9234",
            };
            using var c = np.array(C7.Select(v => (Half)v).ToArray());
            foreach (var b in Bases)
            {
                using (var r = b.Der(c, 2, null, 0))
                    AssertBytes(r, "float16", new long[] { 5 }, der[b.Name], b.Name + " der");
                using (var r = b.Int(c, 1, new object[] { 0.1 }, 0.5, null, 0))
                    AssertBytes(r, "float16", new long[] { 8 }, integ[b.Name], b.Name + " int");
            }
        }

        /// <summary>The full-mantissa complex probe series (numpy.random.default_rng(1234)).</summary>
        private static readonly Complex[] CC =
        {
            new(1.9067990667925687, -1.0329348269888596), new(-0.47921705992152885, -0.7258642848710943),
            new(1.6929849350558217, 1.8563169807135051), new(-0.9532303045458232, -0.9454007828996249),
            new(-0.7236117663432098, -0.23597551178595433), new(-1.5276350681334288, 0.43948323769002995),
        };

        /// <summary>
        ///     complex128, 1-D vs N-D: <c>{p}der(cc, 2, scl=s)</c> and <c>{p}int(cc, 2, k=[k], lbnd=lb, scl=s)</c> as a
        ///     1-D series (every recurrence statement NumPy scalarmath — the NAIVE complex product) and as one (6, 1)
        ///     column (every statement a ufunc — the fused simd_cmul). The derivatives agree; the integrals' lbnd
        ///     corrections differ in the last bits for three bases, exactly as NumPy's do.
        /// </summary>
        [TestMethod]
        public void Complex_OneDIsScalarMath_NDIsUfuncs_BothMatchNumPy()
        {
            var s = new Complex(0.7272425931399216, 0.7275153415491691);
            var k = new Complex(0.34976262669939007, 0.31974869591886224);
            var lb = new Complex(0.4715153966319088, -0.5544926837372255);
            var der = new Dictionary<string, string>
            {
                ["poly"] = "7ce46b3272700fc0631ed589c2a60c402da83456b1041840b73f946ef73218c00e31b369aeff07402e779258de5f22c0eec4db38d79322c0609ea36ca42a40c0",
                ["cheb"] = "2048b6a520e4c13ffa014e7e225631c09ca65e54d4b63fc087f5c79055466bc00e31b369aeff27402e779258de5f42c0eec4db38d79342c0609ea36ca42a60c0",
                ["leg"] = "84d8e148ef280bc0588a28036c4502c0ab964a97ccfa11c0a7f6f2b489c154c06ffe6782c47f21407798d5ebcecb3ac02b568df972423dc00bb381915c7659c0",
                ["lag"] = "7e514590b123f1bfb8e9dac787fa21c05842e9dc3680bb3faa346343d28e1dc068d5128d73b9e5bf989c36a9c8fd0fc0173bf95a58b9ddbf34ca05e1d3ddf9bf",
                ["herm"] = "7ce46b3272702fc0631ed589c2a62c402da83456b1043840b73f946ef73238c00e31b369aeff27402e779258de5f42c0eec4db38d79342c0609ea36ca42a60c0",
                ["herme"] = "7ce46b3272700fc0631ed589c2a60c402da83456b1041840b73f946ef73218c00e31b369aeff07402e779258de5f22c0eec4db38d79322c0609ea36ca42a40c0",
            };
            var int1d = new Dictionary<string, string>
            {
                ["poly"] = "02fc1ce37f7acb3f8f5af12edc24e7bfb2270a3204b2fcbf3ee996a99687d33ff90a035cdf79e13f83e540471625f03ffcdab093c863c03f310fe2c59c9fb5bf5398f276a1f5c4bf41148e06d719c33f44919e069b9ea93f90dd59dcf6cfa9bf442e960cd710813fbe921445f2219abf6745d6a698a586bf33e3a2b733b5a3bf",
                ["cheb"] = "9827a5bfb76ce7bf38cb3e372ffbf2bf7cb439414a41e6bfdbdb6f14766a0c408ec0b99c148de33f9c361c355d4bc63fc4391499b296a4bf3a1a43bf63d6803fe423183ad739a9bfd02e8c146815b03ffe064d4cc0b7963fd21e30131f95953f442e960cd710613fbe921445f2217abf6745d6a698a566bf33e3a2b733b583bf",
                ["leg"] = "a602bdc04c38c2bfa002b9d6e619ecbf3cd97f83a2fff7bfb131e371919cfd3f2220531120c4e13ff18a4dfe435cdf3ff03fb23ec4c7603f676c79940fb975bfbcc20875d308b0bf940405f84430b23f8038b2e66165983f27b40132b8d1873f5230514d90af643f803d4fa106ad7fbfada52f7e299b6abf62dcf6f73a2787bf",
                ["lag"] = "6ab18042809debbffebe3d10b9c4fa3f246d65dfe2fbee3f88eafbb161f714c0ff10f9d91f4603c0341d302b294a134000e2efedb6cb16403dcbe589d66214c09c73dfc960bb0dc048e771c66655084080caccefbf9ca23f1cabfa09fb7ff1bf63d06f949edcf23fd0f7d418dfbd0340173bf95a58b9ddbf34ca05e1d3ddf9bf",
                ["herm"] = "1774a6fe016f31c0dcb3b8a4fc0331406b7b7719c9033740c2a5ee33fa1d16c0f90a035cdf79c13f83e540471625d03ffcdab093c863a03f310fe2c59c9f95bf5398f276a1f5a4bf41148e06d719a33f44919e069b9e893f90dd59dcf6cf89bf442e960cd710613fbe921445f2217abf6745d6a698a566bf33e3a2b733b583bf",
                ["herme"] = "c34f5790ead1e9bf5b1fd52b6421f83f1399299268e9074054f7385a05ecfebff90a035cdf79e13f83e540471625f03ffcdab093c863c03f310fe2c59c9fb5bf5398f276a1f5c4bf41148e06d719c33f44919e069b9ea93f90dd59dcf6cfa9bf442e960cd710813fbe921445f2219abf6745d6a698a586bf33e3a2b733b5a3bf",
            };
            // The three bases whose N-D integral differs from the 1-D one (the lbnd correction's {p}val: scalarmath
            // products vs simd_cmul); the rest are equal to the 1-D dump.
            var int2d = new Dictionary<string, string>(int1d)
            {
                ["poly"] = "07fc1ce37f7acb3f8f5af12edc24e7bfb3270a3204b2fcbf3be996a99687d33ff90a035cdf79e13f83e540471625f03ffcdab093c863c03f310fe2c59c9fb5bf5398f276a1f5c4bf41148e06d719c33f44919e069b9ea93f90dd59dcf6cfa9bf442e960cd710813fbe921445f2219abf6745d6a698a586bf33e3a2b733b5a3bf",
                ["lag"] = "5cb18042809debbff4be3d10b9c4fa3f2e6d65dfe2fbee3f86eafbb161f714c0ff10f9d91f4603c0341d302b294a134000e2efedb6cb16403dcbe589d66214c09c73dfc960bb0dc048e771c66655084080caccefbf9ca23f1cabfa09fb7ff1bf63d06f949edcf23fd0f7d418dfbd0340173bf95a58b9ddbf34ca05e1d3ddf9bf",
                ["herm"] = "1874a6fe016f31c0dcb3b8a4fc0331406b7b7719c9033740c2a5ee33fa1d16c0f90a035cdf79c13f83e540471625d03ffcdab093c863a03f310fe2c59c9f95bf5398f276a1f5a4bf41148e06d719a33f44919e069b9e893f90dd59dcf6cf89bf442e960cd710613fbe921445f2217abf6745d6a698a566bf33e3a2b733b583bf",
            };
            using var c1 = np.array(CC);
            using var c2 = c1.reshape(6, 1);
            foreach (var b in Bases)
            {
                using (var r = b.Der(c1, 2, s, 0))
                    AssertBytes(r, "complex128", new long[] { 4 }, der[b.Name], b.Name + " der 1-D");
                using (var r = b.Der(c2, 2, s, 0))
                    AssertBytes(r, "complex128", new long[] { 4, 1 }, der[b.Name], b.Name + " der N-D");
                using (var r = b.Int(c1, 2, new object[] { k }, lb, s, 0))
                    AssertBytes(r, "complex128", new long[] { 8 }, int1d[b.Name], b.Name + " int 1-D");
                using (var r = b.Int(c2, 2, new object[] { k }, lb, s, 0))
                    AssertBytes(r, "complex128", new long[] { 8, 1 }, int2d[b.Name], b.Name + " int N-D");
            }
        }

        /// <summary>
        ///     Integer and bool series are converted to float64 before any arithmetic (NumPy's
        ///     <c>c.astype(np.double)</c>); NumSharp's char series takes the same route as uint16 (its storage), so its
        ///     bytes are NumPy's uint16 dump. <c>{p}der(int32 [3,-1,4,1,-5])</c> keeps lagder's <c>-0.0</c>.
        /// </summary>
        [TestMethod]
        public void IntegerBoolAndCharSeries_ComputeInFloat64()
        {
            var i32 = new Dictionary<string, string>
            {
                ["poly"] = "000000000000f0bf0000000000002040000000000000084000000000000034c0",
                ["cheb"] = "000000000000004000000000000038c0000000000000184000000000000044c0",
                ["leg"] = "000000000000000000000000000008c0000000000000144000000000008041c0",
                ["lag"] = "000000000000f03f000000000000008000000000000010400000000000001440",
                ["herm"] = "00000000000000c00000000000003040000000000000184000000000000044c0",
                ["herme"] = "000000000000f0bf0000000000002040000000000000084000000000000034c0",
            };
            var boolInt = new Dictionary<string, string>
            {
                ["poly"] = "0000000000000000000000000000f03f0000000000000000555555555555d53f",
                ["cheb"] = "0000000000000000000000000000e03f0000000000000000555555555555c53f",
                ["leg"] = "00000000000000009a9999999999e93f00000000000000009a9999999999c93f",
                ["lag"] = "000000000000f03f000000000000f0bf000000000000f03f000000000000f0bf",
                ["herm"] = "0000000000000000000000000000e03f0000000000000000555555555555c53f",
                ["herme"] = "0000000000000000000000000000f03f0000000000000000555555555555d53f",
            };
            var u16 = new Dictionary<string, string>
            {
                ["poly"] = "000000000000f03f000000000000204000000000000008400000000000003440",
                ["cheb"] = "00000000000010400000000000004c4000000000000018400000000000004440",
                ["leg"] = "00000000000000400000000000003b4000000000000014400000000000804140",
                ["lag"] = "00000000000026c000000000000024c000000000000018c000000000000014c0",
                ["herm"] = "0000000000000040000000000000304000000000000018400000000000004440",
                ["herme"] = "000000000000f03f000000000000204000000000000008400000000000003440",
            };
            using var ci = np.array(new[] { 3, -1, 4, 1, -5 });
            using var cb = np.array(new[] { true, false, true });
            using var cu = np.array(new ushort[] { 3, 1, 4, 1, 5 });
            using var cc = np.array(new[] { '\u0003', '\u0001', '\u0004', '\u0001', '\u0005' });
            foreach (var b in Bases)
            {
                using (var r = b.Der(ci, 1, null, 0))
                    AssertBytes(r, "float64", new long[] { 4 }, i32[b.Name], b.Name + " int32");
                using (var r = b.Int(cb, 1, null, null, null, 0))
                    AssertBytes(r, "float64", new long[] { 4 }, boolInt[b.Name], b.Name + " bool");
                using (var r = b.Der(cu, 1, null, 0))
                    AssertBytes(r, "float64", new long[] { 4 }, u16[b.Name], b.Name + " uint16");
                using (var r = b.Der(cc, 1, null, 0))
                    AssertBytes(r, "float64", new long[] { 4 }, u16[b.Name], b.Name + " char");
            }
        }

        /// <summary>
        ///     Decimal (no NumPy dtype): the series stays decimal and every statement is decimal arithmetic. With integer
        ///     coefficients every derivative value is an exact small integer in BOTH decimal and float64, so the two must
        ///     agree exactly; the integrals divide (1/3, 1/5, ...), where decimal is the more precise of the two.
        /// </summary>
        [TestMethod]
        public void Decimal_ComputesInDecimal()
        {
            var vals = new[] { 1, -2, 3, 4, -5, 6 };
            using var cm = np.array(vals.Select(v => (decimal)v).ToArray());
            using var cd = np.array(vals.Select(v => (double)v).ToArray());
            foreach (var b in Bases)
            {
                using (var rm = b.Der(cm, 2, null, 0))
                using (var rd = b.Der(cd, 2, null, 0))
                {
                    rm.dtype.name.Should().Be("decimal", b.Name);
                    rm.ToArray<decimal>().Select(v => (double)v).Should().Equal(rd.ToArray<double>(), b.Name + " der");
                }
                using (var rm = b.Int(cm, 2, new object[] { 1, 2 }, 0.5, null, 0))
                using (var rd = b.Int(cd, 2, new object[] { 1, 2 }, 0.5, null, 0))
                {
                    rm.dtype.name.Should().Be("decimal", b.Name);
                    var m = rm.ToArray<decimal>();
                    var d = rd.ToArray<double>();
                    m.Length.Should().Be(d.Length, b.Name);
                    for (int i = 0; i < m.Length; i++)
                        System.Math.Abs((double)m[i] - d[i]).Should().BeLessThan(1e-12 * System.Math.Max(1, System.Math.Abs(d[i])), $"{b.Name} int [{i}]");
                }
            }
        }

        // =========================================================================================================
        //  Errors (NumPy's texts, in NumPy's order)
        // =========================================================================================================

        /// <summary>
        ///     Every argument error NumPy raises, with its verbatim text: negative orders, too many constants, array
        ///     lbnd/scl on an integral, an out-of-range axis (AxisError, reporting the axis as given), an empty series'
        ///     integral (IndexError from <c>c[0]</c>), a complex scl into a real series (the in-place multiply's
        ///     UFuncTypeError), and an array scl on a derivative that would stretch the series or does not broadcast.
        /// </summary>
        [TestMethod]
        public void Errors_AreNumPysTexts()
        {
            var cheb = np.polynomial.chebyshev;
            using var two = np.ones(new Shape(2, 3));
            Action a;
            a = () => cheb.chebder(new object[] { 1, 2 }, -1);
            a.Should().Throw<ValueError>().WithMessage("The order of derivation must be non-negative");
            a = () => cheb.chebint(new object[] { 1, 2 }, -1);
            a.Should().Throw<ValueError>().WithMessage("The order of integration must be non-negative");
            a = () => cheb.chebint(new object[] { 0 }, 1, new object[] { 0, 0 });
            a.Should().Throw<ValueError>().WithMessage("Too many integration constants");
            a = () => cheb.chebint(new object[] { 0 }, lbnd: new object[] { 0 });
            a.Should().Throw<ValueError>().WithMessage("lbnd must be a scalar.");
            a = () => cheb.chebint(new object[] { 0 }, scl: new object[] { 0 });
            a.Should().Throw<ValueError>().WithMessage("scl must be a scalar.");
            a = () => cheb.chebder(two, 1, axis: 2);
            a.Should().Throw<AxisError>().WithMessage("axis 2 is out of bounds for array of dimension 2*");
            a = () => cheb.chebder(two, 1, axis: -3);
            a.Should().Throw<AxisError>().WithMessage("axis -3 is out of bounds for array of dimension 2*");
            a = () => cheb.chebint(np.zeros(0), 1);
            a.Should().Throw<IndexError>().WithMessage("index 0 is out of bounds for axis 0 with size 0");
            a = () => cheb.chebder(new object[] { 1.0, 2.0, 3.0 }, 1, new Complex(0, 1));
            a.Should().Throw<ArgumentException>().WithMessage(
                "Cannot cast ufunc 'multiply' output from dtype('complex128') to dtype('float64') with casting rule 'same_kind'");
            a = () => cheb.chebint(new object[] { 1.0, 2.0, 3.0 }, 1, scl: new Complex(0, 1));
            a.Should().Throw<ArgumentException>().WithMessage(
                "Cannot cast ufunc 'multiply' output from dtype('complex128') to dtype('float64') with casting rule 'same_kind'");
            a = () => cheb.chebder(np.ones(new Shape(3, 1)), 1, np.ones(4));
            a.Should().Throw<ValueError>().WithMessage(
                "non-broadcastable output operand with shape (3,1) doesn't match the broadcast shape (3,4)");
            a = () => cheb.chebder(np.ones(new Shape(3, 2)), 1, np.ones(3));
            a.Should().Throw<IncorrectShapeException>().WithMessage(
                "operands could not be broadcast together with shapes (3,2) (3,) (3,2) ");

            // An empty series' DERIVATIVE is fine (m >= n: c[:1]*0 of an empty c).
            using var empty = cheb.chebder(np.zeros(0), 1);
            empty.shape.Should().Equal(new long[] { 0 });
        }

        /// <summary>
        ///     The integral's correction <c>tmp[0] += k - {p}val(lbnd, tmp)</c> errs per SERIES KIND: a 1-D series
        ///     stores a scalar with setitem (an array constant: ValueError into a real series, but <c>complex()</c>'s
        ///     TypeError into a complex one — and a complex value into a real series keeps its real part), while an N-D
        ///     series adds a row in place (a complex value is the add's UFuncTypeError, naming NumPy's complex64 for a
        ///     float32 series).
        /// </summary>
        [TestMethod]
        public void IntCorrection_ErrsAndCastsPerSeriesKind()
        {
            var cheb = np.polynomial.chebyshev;
            Action a;
            a = () => cheb.chebint(new object[] { 1.0, 2.0 }, 1, new object[] { np.array(new[] { 1.0, 2.0 }) });
            a.Should().Throw<ValueError>().WithMessage("setting an array element with a sequence.");
            a = () => cheb.chebint(np.array(new[] { new Complex(1, 0), new Complex(2, 0) }), 1,
                                   new object[] { np.array(new[] { 1.0, 2.0 }) });
            a.Should().Throw<TypeError>().WithMessage("only 0-dimensional arrays can be converted to Python scalars");
            a = () => cheb.chebint(np.ones(new Shape(3, 2), NPTypeCode.Single), 1, lbnd: new Complex(0, 0.5));
            a.Should().Throw<ArgumentException>().WithMessage(
                "Cannot cast ufunc 'add' output from dtype('complex64') to dtype('float32') with casting rule 'same_kind'");
            a = () => cheb.chebint(np.ones(new Shape(3, 2)), 1, lbnd: new Complex(0, 0.5));
            a.Should().Throw<ArgumentException>().WithMessage(
                "Cannot cast ufunc 'add' output from dtype('complex128') to dtype('float64') with casting rule 'same_kind'");
            a = () => cheb.chebint(np.ones(new Shape(3, 2)), 1, new object[] { new Complex(0, 1) });
            a.Should().Throw<ArgumentException>().WithMessage(
                "Cannot cast ufunc 'add' output from dtype('complex128') to dtype('float64') with casting rule 'same_kind'");

            // 1-D: the complex value's real part is stored (NumPy's ComplexWarning), in the series dtype.
            using (var r = cheb.chebint(np.array(new[] { 1f, 2f, 3f }), 1, lbnd: new Complex(0.5, 0.25)))
                AssertBytes(r, "float32", new long[] { 4 }, "0000a03f000000bf0000003f0000003f");
            using (var r = cheb.chebint(np.array(new[] { 1.0, 2.0, 3.0 }), 1, lbnd: new Complex(0.5, 0.25)))
                AssertBytes(r, "float64", new long[] { 4 }, "000000000000f43f000000000000e0bf000000000000e03f000000000000e03f");
        }

        // =========================================================================================================
        //  Layout: the result is the object NumPy returns
        // =========================================================================================================

        /// <summary>
        ///     NumPy's results are moveaxis VIEWS of fresh C-order buffers (OWNDATA false), except the three that are not
        ///     fresh buffers: m == 0 (the K-order copy, OWNDATA true — an F series stays F), der's m ≥ n (<c>c[:1]*0</c>,
        ///     moved back as a view — hermeder alone skips the moveaxis and returns the ufunc output itself) and an
        ///     integral that only took the n == 1 zero branch (a view of the moved copy).
        /// </summary>
        [TestMethod]
        public void ResultLayouts_AreNumPys()
        {
            var cheb = np.polynomial.chebyshev;
            using var a = np.arange(12.0).reshape(3, 4);
            using var f = np.asfortranarray(a);
            using var z = np.zeros(new Shape(1, 3));
            var cases = new (string name, Func<NDArray> call, long[] shape, bool c, bool fo, bool own)[]
            {
                ("der ax1", () => cheb.chebder(a, 1, axis: 1), new long[] { 3, 3 }, false, true, false),
                ("der m0", () => cheb.chebder(a, 0, axis: 1), new long[] { 3, 4 }, true, false, true),
                ("int ax1", () => cheb.chebint(a, 1, axis: 1), new long[] { 3, 5 }, false, true, false),
                ("der m>=n", () => cheb.chebder(a, 5, axis: 1), new long[] { 3, 1 }, true, true, false),
                ("hermeder m>=n", () => np.polynomial.hermite_e.hermeder(a, 5, axis: 1), new long[] { 1, 3 }, true, true, true),
                ("der m0 F", () => cheb.chebder(f, 0), new long[] { 3, 4 }, false, true, true),
                ("int n==1 zero", () => cheb.chebint(z, 2, new object[] { 0, 0 }), new long[] { 1, 3 }, true, true, false),
            };
            foreach (var (name, call, shape, c, fo, own) in cases)
            {
                using var r = call();
                r.shape.Should().Equal(shape, name);
                r.flags.c_contiguous.Should().Be(c, name + " C");
                r.flags.f_contiguous.Should().Be(fo, name + " F");
                r.flags.owndata.Should().Be(own, name + " OWNDATA");
            }
            using (var r = cheb.chebint(new object[] { 0.0 }, 3))
                r.ToArray<double>().Should().Equal(new[] { 0.0 });
            using (var r = cheb.chebint(new object[] { 0.0 }, 3, new object[] { 1, 2, 3 }))
                r.ToArray<double>().Should().Equal(new[] { 3.25, 2.0, 0.25 });
            using (var r = np.polynomial.laguerre.lagint(new object[] { 0 }, 2, new object[] { 0, 1 }))
                r.ToArray<double>().Should().Equal(new[] { 1.0 });
        }

        // =========================================================================================================
        //  Kernel structure
        // =========================================================================================================

        /// <summary>
        ///     Every coefficient dtype NumPy computes in (float16/float32/float64/complex128) meets the load stage's
        ///     source dtypes (the same dtype, every integer width, bool, char) with vector code: a missing lane kind or
        ///     conversion would silently compile scalar-only kernels (same bits, slower), which
        ///     <see cref="DirectILKernelGenerator.PolyCalcVectorFallbacks"/> counts.
        /// </summary>
        [TestMethod]
        public void Kernels_EveryDtypeHasVectorLanes_NoScalarFallback()
        {
            long before = DirectILKernelGenerator.PolyCalcVectorFallbacks;
            var sources = new[] { NPTypeCode.Boolean, NPTypeCode.SByte, NPTypeCode.Byte, NPTypeCode.Int16, NPTypeCode.UInt16,
                NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Char,
                NPTypeCode.Half, NPTypeCode.Single, NPTypeCode.Double, NPTypeCode.Complex, NPTypeCode.Decimal };
            foreach (var b in Bases)
                foreach (var t in sources)
                {
                    using var c = (np.arange(7 * 45) % 5).reshape(7, 45).astype(t);
                    b.Der(c, 2, null, 0).Dispose();
                    b.Der(c, 1, 0.5, 0).Dispose();
                    b.Int(c, 2, new object[] { 1 }, 0.5, null, 0).Dispose();
                    b.Int(c, 1, null, null, 2, 0).Dispose();
                }
            (DirectILKernelGenerator.PolyCalcVectorFallbacks - before).Should().Be(0,
                "every coefficient dtype and load-stage source dtype has a vector lane kind and conversion");
        }

        /// <summary>
        ///     Columns are independent, so the kernel may take them in any blocks: a WIDE series (more columns than one
        ///     block, a ragged last block, vector bodies plus scalar tails) equals the same series computed a few
        ///     columns at a time, byte for byte — for float32 and complex128, derivatives and integrals.
        /// </summary>
        [TestMethod]
        public void Kernels_ColumnBlocks_AgreeWithColumnSlices()
        {
            foreach (var t in new[] { NPTypeCode.Single, NPTypeCode.Complex })
                foreach (var b in Bases)
                {
                    using var wide = np.sin(np.arange(9 * 5003.0)).reshape(9, 5003).astype(t);
                    using var der = b.Der(wide, 3, 0.75, 0);
                    using var integ = b.Int(wide, 2, new object[] { 0.5 }, 0.25, 1.5, 0);
                    foreach (var (lo, hi) in new[] { (0, 3), (3, 17), (17, 4099), (4099, 5003) })
                    {
                        using var part = wide[$":, {lo}:{hi}"];
                        using (var want = b.Der(part, 3, 0.75, 0))
                        using (var got = der[$":, {lo}:{hi}"])
                            Hex(got).Should().Be(Hex(want), $"{b.Name} {t} der [{lo},{hi})");
                        using (var want = b.Int(part, 2, new object[] { 0.5 }, 0.25, 1.5, 0))
                        using (var got = integ[$":, {lo}:{hi}"])
                            Hex(got).Should().Be(Hex(want), $"{b.Name} {t} int [{lo},{hi})");
                    }
                }
        }

        /// <summary>
        ///     The load stage reads the caller's series in place when its columns sit at one stride (C, F, strided,
        ///     reversed, any 2-D view) and falls back to a house copy otherwise (a 3-D view whose other axes do not
        ///     merge); both routes, and every layout, must give the bytes of a contiguous copy of the same series.
        /// </summary>
        [TestMethod]
        public void Kernels_DirectLoadAndHouseCopy_AgreeWithAContiguousCopy()
        {
            using var base3 = np.sin(np.arange(4 * 6 * 5.0)).reshape(4, 6, 5);
            var views = new (string name, NDArray v)[]
            {
                ("transposed 3-D", base3.transpose(new[] { 0, 2, 1 })),
                ("strided 3-D", base3[":, ::2, ::-1"]),
                ("reversed rows", base3["::-1"]),
                ("F copy", np.asfortranarray(base3)),
                ("int32 transposed", (base3 * 10).astype(NPTypeCode.Int32).transpose(new[] { 1, 0, 2 })),
            };
            foreach (var (name, v) in views)
                using (v)
                using (var dense = np.ascontiguousarray(v))
                    foreach (var b in Bases)
                        foreach (int axis in new[] { 0, 1, 2 })
                        {
                            using (var got = b.Der(v, 2, 0.5, axis))
                            using (var want = b.Der(dense, 2, 0.5, axis))
                                Hex(got).Should().Be(Hex(want), $"{b.Name} der {name} axis={axis}");
                            using (var got = b.Int(v, 2, new object[] { 1, -1 }, 0.5, 2, axis))
                            using (var want = b.Int(dense, 2, new object[] { 1, -1 }, 0.5, 2, axis))
                                Hex(got).Should().Be(Hex(want), $"{b.Name} int {name} axis={axis}");
                        }
        }

        /// <summary>Concurrent first calls compile one kernel per key and every thread gets NumPy's bytes.</summary>
        [TestMethod]
        public void Kernels_AreThreadSafe_UnderConcurrentFirstUse()
        {
            using var c = np.sin(np.arange(7 * 33.0)).reshape(7, 33).astype(NPTypeCode.Half);
            var want = Bases.ToDictionary(b => b.Name, b => (Hex(b.Der(c, 2, 0.5, 1)), Hex(b.Int(c, 2, new object[] { 1 }, 0.25, null, 1))));
            Parallel.For(0, 64, i =>
            {
                foreach (var b in Bases)
                {
                    using var d = b.Der(c, 2, 0.5, 1);
                    using var n = b.Int(c, 2, new object[] { 1 }, 0.25, null, 1);
                    if (Hex(d) != want[b.Name].Item1 || Hex(n) != want[b.Name].Item2)
                        throw new InvalidOperationException($"{b.Name} diverged on thread {i}");
                }
            });
        }
    }
}
