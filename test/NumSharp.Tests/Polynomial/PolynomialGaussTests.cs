using System;
using System.Collections.Generic;
using System.Numerics;
using NumSharp.Interop.OpenBLAS;

namespace NumSharp.Tests.Polynomial
{
    /// <summary>
    ///     Gauss-Legendre and Gauss-Laguerre quadrature of <c>numpy.polynomial</c> (plan
    ///     <c>docs/plans/numpy-polynomial-remaining-compute.md</c> §3.4, unit U8): <c>leggauss</c> and <c>laggauss</c>. NumPy
    ///     2.4.2's own <c>TestGauss.test_100</c> ported; NumPy's bytes for small degrees; NumPy's error for every argument kind
    ///     with a C# spelling (the probe scripts are quoted next to <see cref="Pins"/> and <see cref="KindErrors"/>); the
    ///     composition against NumPy's statements written out by hand; and the missing-backend path.
    /// </summary>
    /// <remarks>
    ///     Every value comes out of <c>np.linalg.eigvalsh</c>, which needs a LAPACK-capable library (the bundled
    ///     scipy-openblas is one); those tests go <see cref="Assert.Inconclusive(string)"/> on a host without one. The values
    ///     are held to 4 ULP of NumPy's: the bits depend on the OpenBLAS build, its kernel and the thread count, and byte
    ///     parity is the host-pinned <c>PolygaussParity</c> oracle tier. The errors are raised before LAPACK and are asserted
    ///     exactly on every host. <c>[DoNotParallelize]</c>: the tests bind the process-wide OpenBLAS backend
    ///     (<c>TensorEngine.Blas</c>).
    /// </remarks>
    [TestClass]
    [DoNotParallelize]
    public class PolynomialGaussTests
    {
        /// <summary>The two functions by name.</summary>
        private static readonly Dictionary<string, Func<object, (NDArray x, NDArray w)>> Functions = new()
        {
            ["leggauss"] = d => np.polynomial.legendre.leggauss(d),
            ["laggauss"] = d => np.polynomial.laguerre.laggauss(d),
        };

        /// <summary>2**63, the edge of CPython's list repeat.</summary>
        private static readonly BigInteger T = BigInteger.One << 63;

        /// <summary>
        ///     The C# spelling of every argument kind of <see cref="KindErrors"/> (the house map: a C# integer, bool, double,
        ///     <see cref="Complex"/> or <see cref="BigInteger"/> is a Python scalar, <see cref="Half"/> / <c>char</c> are
        ///     np.float16 / np.uint16 scalars, an <see cref="NDArray"/> an ndarray, <c>object[]</c> a list, a <c>ValueTuple</c>
        ///     a tuple, null Python's None).
        /// </summary>
        private static readonly Dictionary<string, Func<object>> Inputs = new()
        {
            ["int 0"] = () => 0, ["int -1"] = () => -1, ["bool false"] = () => false, ["double 3.0"] = () => 3.0,
            ["str 3"] = () => "3", ["null"] = () => null, ["char 0"] = () => (char)0, ["nd0 int64 3"] = () => np.array(3L),
            ["nd0 int64 0"] = () => np.array(0L), ["nd0 uint64 3"] = () => np.array(3UL), ["nd0 int8 2"] = () => np.array((sbyte)2),
            ["nd0 bool true"] = () => np.array(true), ["nd1 [3]"] = () => np.array(new long[] { 3 }),
            ["object[] {3}"] = () => new object[] { 3 }, ["tuple (3,)"] = () => ValueTuple.Create(3),
            ["complex 3"] = () => new Complex(3, 0), ["half 3"] = () => (Half)3, ["big 2**63"] = () => T,
            ["big 2**64"] = () => BigInteger.One << 64, ["big 2**62"] = () => BigInteger.One << 62,
            ["big 2**61"] = () => BigInteger.One << 61, ["big -2**70"] = () => -(BigInteger.One << 70),
        };

        // NumPy 2.4.2's bytes and error texts (scratchpad gauss_pins.py, OPENBLAS_NUM_THREADS=1):
        //   for d in (1, 2, 3, 4, 5, 6, 12): leggauss(d) / laggauss(d), x.tobytes().hex(), w.tobytes().hex();
        //   every "!!" outcome of gauss_facts.py's argument kinds.
        /// <summary>NumPy's float64 bytes of each function's (x, w) at small degrees.</summary>
        private static readonly (string Fn, long Deg, string X, string W)[] Pins =
        {
            ("leggauss", 1, "0000000000000000", "0000000000000040"),
            ("leggauss", 2, "1c339045a779e2bf1c339045a779e23f", "000000000000f03f000000000000f03f"),
            ("leggauss", 3, "48723ff47ec9e8bf000000000000000048723ff47ec9e83f", "731cc7711cc7e13f1cc7711cc771ec3f731cc7711cc7e13f"),
            ("leggauss", 4, "8539f6bc6d8eebbffc3dddd93fc2d5bffc3dddd93fc2d53f8539f6bc6d8eeb3f", "66b6e7f74043d63fcd240c845fdee43fcd240c845fdee43f66b6e7f74043d63f"),
            ("leggauss", 5, "693a53e06cffecbf05b799fd233be1bf000000000000000005b799fd233be13f693a53e06cffec3f", "93036ec39e53ce3f5841ae25daa1de3fddbc9a785634e23f5841ae25daa1de3f93036ec39e53ce3f"),
            ("leggauss", 6, "1da0804ecad6edbf5ec95596a028e5bfc57536d0128bcebfc57536d0128bce3f5ec95596a028e53f1da0804ecad6ed3f", "f5dbe201f6edc53f1e4c79b5b716d73fe84595494df2dd3fe84595494df2dd3f1e4c79b5b716d73ff5dbe201f6edc53f"),
            ("leggauss", 12, "812ee4d8f168efbfb388fb4f87eeecbf368fd8ee0aa3e8bff977c0054fcbe2bf9db1a8208d8ad7bfe430f6f8a507c0bfe430f6f8a507c03f9db1a8208d8ad73ff977c0054fcbe23f368fd8ee0aa3e83fb388fb4f87eeec3f812ee4d8f168ef3f", "536dea9d5d27a83faf61ce2b6060bb3f962df258727dc43f6babb1e66301ca3fae6a255c15e3cd3f22f0d4e60ce4cf3f22f0d4e60ce4cf3fae6a255c15e3cd3f6babb1e66301ca3f962df258727dc43faf61ce2b6060bb3f536dea9d5d27a83f"),
            ("laggauss", 1, "000000000000f03f", "000000000000f03f"),
            ("laggauss", 2, "67880133c3bee23fe69d3f334f500b40", "e69d3f334f50eb3f67880133c3bec23f"),
            ("laggauss", 3, "4636f9e20c9cda3f56075aa9af5a0240f168235de7281940", "87ebc12046c1e63fb34c420b3cd3d13fe8873b67f646853f"),
            ("laggauss", 4, "983394109fa4d43f2caa4131a3eefb3f8f9586ca7f25124056de0fbc46ca2240", "3e0e13d6094de33fb0f6bfa6f2dfd63f894690ae1de9a33f1507c88feeab413f"),
            ("laggauss", 5, "309d6d1a2cded03f8cdcae864c9df63f52e4e8df7ac50c4046550394de571c40cdd35a0c17482940", "44a57ed238b2e03f1d95e6ccc183d93fd6d4c2e1f670b33f3d72292d68966d3f9e3d10465481f83e"),
            ("laggauss", 6, "17bf7cce3c86cc3f4c93ddaadd05f33fbdce2fbe1ff10740cb085c3cbf19174069d68a87c8ac23401c8c25423bf72f40", "0a9a285ead5fdd3fac8ca44024b0da3f16b43eb80906bd3f3835baa72c4c853feb698b59241b313f7c4512be7726ae3e"),
            ("laggauss", 12, "9f007af5f69fbd3f3e6ca26e8493e33f0a8984d3a633f83f3fd644d285ab0640f4a428ee9b651240bb13c747cb601b40ca9cbd3d1d3e2340df74dca319032a40b284b838ea1d31401840eedbad263640cc7ef86beb7c3c40572a96ffaf8c4240", "42223ad95bf1d03fbdb0e83d352dd83f84941450143ecf3f170a5923ae27b73f960586f6b795943f27cbabe7c1d2653f2a60871652a32a3f4b6acf85f48ae13e887108bce664863e9803abefe50f173e6682892e1dee8a3d7f928765465bcd3c"),
        };

        /// <summary>NumPy's error per (function, argument kind of <see cref="Inputs"/>): <c>!! PythonType: message</c>
        ///     (the MemoryError of CPython's list repeat has an empty message).</summary>
        private static readonly (string Fn, string Label, string Error)[] KindErrors =
        {
            ("leggauss", "int 0", "!! ValueError: deg must be a positive integer"),
            ("leggauss", "int -1", "!! ValueError: deg must be a positive integer"),
            ("leggauss", "bool false", "!! ValueError: deg must be a positive integer"),
            ("leggauss", "double 3.0", "!! TypeError: deg must be an integer, received 3.0"),
            ("leggauss", "str 3", "!! TypeError: deg must be an integer, received 3"),
            ("leggauss", "null", "!! TypeError: deg must be an integer, received None"),
            ("leggauss", "char 0", "!! ValueError: deg must be a positive integer"),
            ("leggauss", "nd0 int64 3", "!! ValueError: Series must have maximum degree of at least 1."),
            ("leggauss", "nd0 int64 0", "!! ValueError: deg must be a positive integer"),
            ("leggauss", "nd0 uint64 3", "!! ValueError: Series must have maximum degree of at least 1."),
            ("leggauss", "nd0 int8 2", "!! ValueError: Series must have maximum degree of at least 1."),
            ("leggauss", "nd0 bool true", "!! TypeError: deg must be an integer, received True"),
            ("leggauss", "nd1 [3]", "!! TypeError: deg must be an integer, received [3]"),
            ("leggauss", "object[] {3}", "!! TypeError: deg must be an integer, received [3]"),
            ("leggauss", "tuple (3,)", "!! TypeError: deg must be an integer, received (3,)"),
            ("leggauss", "complex 3", "!! TypeError: deg must be an integer, received (3+0j)"),
            ("leggauss", "half 3", "!! TypeError: deg must be an integer, received 3.0"),
            ("leggauss", "big 2**63", "!! OverflowError: cannot fit 'int' into an index-sized integer"),
            ("leggauss", "big 2**64", "!! OverflowError: cannot fit 'int' into an index-sized integer"),
            ("leggauss", "big 2**62", "!! MemoryError: "),
            ("leggauss", "big 2**61", "!! MemoryError: "),
            ("leggauss", "big -2**70", "!! ValueError: deg must be a positive integer"),
            ("laggauss", "int 0", "!! ValueError: deg must be a positive integer"),
            ("laggauss", "int -1", "!! ValueError: deg must be a positive integer"),
            ("laggauss", "bool false", "!! ValueError: deg must be a positive integer"),
            ("laggauss", "double 3.0", "!! TypeError: deg must be an integer, received 3.0"),
            ("laggauss", "str 3", "!! TypeError: deg must be an integer, received 3"),
            ("laggauss", "null", "!! TypeError: deg must be an integer, received None"),
            ("laggauss", "char 0", "!! ValueError: deg must be a positive integer"),
            ("laggauss", "nd0 int64 3", "!! ValueError: Series must have maximum degree of at least 1."),
            ("laggauss", "nd0 int64 0", "!! ValueError: deg must be a positive integer"),
            ("laggauss", "nd0 uint64 3", "!! ValueError: Series must have maximum degree of at least 1."),
            ("laggauss", "nd0 int8 2", "!! ValueError: Series must have maximum degree of at least 1."),
            ("laggauss", "nd0 bool true", "!! TypeError: deg must be an integer, received True"),
            ("laggauss", "nd1 [3]", "!! TypeError: deg must be an integer, received [3]"),
            ("laggauss", "object[] {3}", "!! TypeError: deg must be an integer, received [3]"),
            ("laggauss", "tuple (3,)", "!! TypeError: deg must be an integer, received (3,)"),
            ("laggauss", "complex 3", "!! TypeError: deg must be an integer, received (3+0j)"),
            ("laggauss", "half 3", "!! TypeError: deg must be an integer, received 3.0"),
            ("laggauss", "big 2**63", "!! OverflowError: cannot fit 'int' into an index-sized integer"),
            ("laggauss", "big 2**64", "!! OverflowError: cannot fit 'int' into an index-sized integer"),
            ("laggauss", "big 2**62", "!! MemoryError: "),
            ("laggauss", "big 2**61", "!! MemoryError: "),
            ("laggauss", "big -2**70", "!! ValueError: deg must be a positive integer"),
        };

        /// <summary>Leaves the process-global OpenBLAS engine off after every test (the LapackEigTests convention).</summary>
        [TestCleanup]
        public void Cleanup() => OpenBlasEngine.Disable();

        /// <summary>Installs the LAPACK backend, or goes Inconclusive on a host that cannot load one.</summary>
        private static void RequireLapack()
        {
            try
            {
                OpenBlasEngine.Enable(threads: 1);
            }
            catch (Exception e)
            {
                Assert.Inconclusive("no CBLAS library on this host: " + e.Message.Split('\n')[0]);
            }

            if (!OpenBlasEngine.LapackAvailable)
                Assert.Inconclusive("the loaded BLAS exports no LAPACK routines (a bare reference CBLAS).");
        }

        // ---------------------------------------------------------------------------------------------
        //  NumPy's values and errors
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     leggauss / laggauss for degrees 1-6 and 12 have NumPy's float64 values (within 4 ULP; bit for bit on the pinned
        ///     host, which the oracle tier holds), shapes and dtypes.
        /// </summary>
        [TestMethod]
        public void NumPyPins_AreReproduced()
        {
            RequireLapack();
            foreach (var (fn, deg, xHex, wHex) in Pins)
            {
                var (x, w) = Functions[fn](deg);
                string label = $"{fn}({deg})";
                foreach (var (name, got, hex) in new[] { ("x", x, xHex), ("w", w, wHex) })
                {
                    Assert.AreEqual("float64", got.dtype.name, $"{label}.{name}: dtype");
                    CollectionAssert.AreEqual(new[] { deg }, got.Shape.Dimensions, $"{label}.{name}: shape");
                    AssertWithinUlps(hex, got, 4, $"{label}.{name}");
                }
            }
        }

        /// <summary>
        ///     Every rejected argument kind raises NumPy's error, in NumPy's order and text, before any LAPACK call (so with no
        ///     backend installed): operator.index's TypeError, the positive-integer ValueError, the companion's ValueError for a
        ///     0-d array (<c>[0] * arr</c> is np.multiply, not a list repeat), and CPython's list-repeat OverflowError and
        ///     MemoryError.
        /// </summary>
        [TestMethod]
        public void ArgumentErrors_MatchNumPy()
        {
            foreach (var (fn, label, error) in KindErrors)
            {
                string got;
                try
                {
                    var (x, w) = Functions[fn](Inputs[label]());
                    got = $"returned {x.dtype.name}[{x.size}]";
                }
                catch (Exception e)
                {
                    string name = e switch
                    {
                        OverflowException => "OverflowError",
                        OutOfMemoryException => "MemoryError",
                        _ => e.GetType().Name,
                    };
                    got = $"!! {name}: {e.Message}";
                }
                Assert.AreEqual(error, got, $"{fn}({label})");
            }
        }

        /// <summary>
        ///     The accepted integer kinds compute the same nodes and weights as the plain int: a bool (<c>[0] * True</c> is one
        ///     zero) and a char (np.uint16 repeats a list like an int).
        /// </summary>
        [TestMethod]
        public void IntegerKinds_RepeatTheList()
        {
            RequireLapack();
            foreach (var (fn, f) in Functions)
            {
                var (x1, w1) = f(1);
                var (xt, wt) = f(true);
                Assert.AreEqual(Hex(x1) + Hex(w1), Hex(xt) + Hex(wt), $"{fn}(True)");
                var (x3, w3) = f(3);
                var (xc, wc) = f((char)3);
                Assert.AreEqual(Hex(x3) + Hex(w3), Hex(xc) + Hex(wc), $"{fn}((char)3)");
                var (xb, wb) = f(new BigInteger(3));
                Assert.AreEqual(Hex(x3) + Hex(w3), Hex(xb) + Hex(wb), $"{fn}(BigInteger 3)");
            }
        }

        // ---------------------------------------------------------------------------------------------
        //  The composition
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     The result is NumPy's statements composed from the public functions, byte for byte on this host: the companion of
        ///     <c>[0]*deg + [1]</c>, eigvalsh, one Newton step through <c>{p}val</c> / <c>{p}der</c>, the weights scaled by their
        ///     largest magnitude, leggauss symmetrising both and normalising the weights to 2, laggauss normalising them to 1 —
        ///     over every degree to 80.
        /// </summary>
        [TestMethod]
        public void Result_IsNumPysStatementsComposed()
        {
            RequireLapack();
            for (int deg = 1; deg <= 80; deg++)
            {
                foreach (bool legendre in new[] { true, false })
                {
                    var c = np.zeros(new Shape(deg + 1), np.int64);
                    c.SetAtIndex(1L, deg);
                    NDArray m = legendre ? np.polynomial.legendre.legcompanion(c) : np.polynomial.laguerre.lagcompanion(c);
                    NDArray x = np.linalg.eigvalsh(m);
                    Func<object, object, NDArray> val = legendre
                        ? (p, s) => np.polynomial.legendre.legval(p, s)
                        : (p, s) => np.polynomial.laguerre.lagval(p, s);
                    Func<object, NDArray> der = legendre
                        ? s => np.polynomial.legendre.legder(s)
                        : s => np.polynomial.laguerre.lagder(s);
                    NDArray dy = val(x, c), df = val(x, der(c));
                    x = x - dy / df;
                    NDArray fm = val(x, c["1:"]);
                    fm = fm / np.max(np.abs(fm));
                    df = df / np.max(np.abs(df));
                    NDArray w = 1.0 / (fm * df);
                    if (legendre)
                    {
                        w = (w + w["::-1"]) / 2.0;
                        x = (x - x["::-1"]) / 2.0;
                        w = w * (2.0 / w.sum().GetDouble());
                    }
                    else
                        w = w / w.sum();
                    var (gx, gw) = legendre ? np.polynomial.legendre.leggauss(deg) : np.polynomial.laguerre.laggauss(deg);
                    string label = $"{(legendre ? "leggauss" : "laggauss")}({deg})";
                    Assert.AreEqual(Hex(x), Hex(gx), $"{label}.x");
                    Assert.AreEqual(Hex(w), Hex(gw), $"{label}.w");
                }
            }
        }

        /// <summary>Each result slot is a fresh, owning, writeable, contiguous float64 vector, separate from the other.</summary>
        [TestMethod]
        public void Results_AreFreshOwningVectors()
        {
            RequireLapack();
            foreach (var (fn, f) in Functions)
            {
                foreach (int deg in new[] { 1, 2, 7 })
                {
                    var (x, w) = f(deg);
                    foreach (var (name, r) in new[] { ("x", x), ("w", w) })
                    {
                        Assert.IsTrue(r.flags.c_contiguous && r.flags.f_contiguous, $"{fn}({deg}).{name}: contiguous");
                        Assert.IsTrue(r.flags.owndata && r.flags.writeable, $"{fn}({deg}).{name}: owning and writeable");
                    }
                    Assert.IsFalse(np.shares_memory(x, w), $"{fn}({deg}): separate arrays");
                }
            }
        }

        /// <summary>
        ///     Without a LAPACK backend a valid degree raises the missing-backend exception from eigvalsh (NumSharp ships no
        ///     managed eigensolver), after every argument check NumPy runs first.
        /// </summary>
        [TestMethod]
        public void WithoutBackend_ValidDegreeRaisesMissingBackend()
        {
            OpenBlasEngine.Disable();
            foreach (var (fn, f) in Functions)
            {
                // MissingBackendException or its OpenBLAS-specific subclass (the text then names the package to reference).
                Exception missing = null;
                try
                {
                    f(3);
                }
                catch (Exception e)
                {
                    missing = e;
                }
                Assert.IsInstanceOfType(missing, typeof(MissingBackendException), $"{fn}(3): {missing?.GetType().Name}");
                // The argument errors come first, with or without a backend.
                Assert.ThrowsException<ValueError>(() => f(0), $"{fn}(0)");
                Assert.ThrowsException<ValueError>(() => f(np.array(3L)), $"{fn}(np.array(3))");
            }
        }

        // ---------------------------------------------------------------------------------------------
        //  NumPy's own tests (numpy/polynomial/tests/test_legendre.py, test_laguerre.py)
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     NumPy's <c>TestGauss.test_100</c> for both bases: the 100 nodes and weights make <c>{p}vander(x, 99)</c>
        ///     orthonormal after normalisation (<c>assert_almost_equal(vv, np.eye(100))</c>), and the weights integrate 1 to
        ///     2 (Legendre) / 1 (Laguerre).
        /// </summary>
        [TestMethod]
        public void NumPysOwnTest100()
        {
            RequireLapack();
            foreach (bool legendre in new[] { true, false })
            {
                var (x, w) = legendre ? np.polynomial.legendre.leggauss(100) : np.polynomial.laguerre.laggauss(100);
                NDArray v = legendre ? np.polynomial.legendre.legvander(x, 99) : np.polynomial.laguerre.lagvander(x, 99);
                NDArray vv = np.dot(v.T * w, v);
                NDArray vd = 1.0 / np.sqrt(vv.diagonal());
                vv = np.expand_dims(vd, 1) * vv * vd;
                double maxDev = np.max(np.abs(vv - np.eye(100))).GetDouble();
                Assert.IsTrue(maxDev < 1.5e-7, $"{(legendre ? "leggauss" : "laggauss")}(100): orthogonality off by {maxDev}");
                double target = legendre ? 2.0 : 1.0;
                Assert.IsTrue(System.Math.Abs(w.sum().GetDouble() - target) < 1.5e-7, "the weights integrate 1");
            }
        }

        // ---------------------------------------------------------------------------------------------
        //  Helpers
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     Asserts NumPy's float64 bytes to within <paramref name="ulps"/> units in the last place per element.
        /// </summary>
        /// <param name="expectedHex">NumPy's bytes.</param>
        /// <param name="actual">NumSharp's result.</param>
        /// <param name="ulps">The tolerance.</param>
        /// <param name="label">The case.</param>
        private static void AssertWithinUlps(string expectedHex, NDArray actual, long ulps, string label)
        {
            byte[] e = Convert.FromHexString(expectedHex), a = actual.tobytes();
            Assert.AreEqual(e.Length, a.Length, $"{label}: length");
            for (int i = 0; i < e.Length; i += 8)
            {
                long d = UlpDistance(BitConverter.ToInt64(a, i), BitConverter.ToInt64(e, i));
                Assert.IsTrue(d <= ulps, $"{label}: element {i / 8} is {d} ULP from NumPy's");
            }
        }

        /// <summary>The distance in units in the last place between two float64 bit patterns read as signed integers.</summary>
        /// <param name="a">One bit pattern.</param>
        /// <param name="b">The other.</param>
        /// <returns>The distance (a sign difference is the sum of both distances from zero).</returns>
        private static long UlpDistance(long a, long b)
        {
            if (a == b)
                return 0;
            static long Ordered(long v) => v < 0 ? long.MinValue - v : v;   // two's-complement order of sign-magnitude
            long d = Ordered(a) - Ordered(b);
            return d < 0 ? -d : d;
        }

        /// <summary>The C-order bytes of <paramref name="a"/> as lowercase hex (NumPy's <c>tobytes().hex()</c>).</summary>
        /// <param name="a">The array.</param>
        /// <returns>The hex text.</returns>
        private static string Hex(NDArray a) => Convert.ToHexString(a.tobytes()).ToLowerInvariant();
    }
}
