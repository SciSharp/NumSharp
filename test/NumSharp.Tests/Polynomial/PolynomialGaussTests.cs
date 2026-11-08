using System;
using System.Collections.Generic;
using System.Numerics;
using NumSharp.Interop.OpenBLAS;

namespace NumSharp.Tests.Polynomial
{
    /// <summary>
    ///     Gauss-Legendre, Gauss-Laguerre and Gauss-Hermite quadrature of <c>numpy.polynomial</c> (plan
    ///     <c>docs/plans/numpy-polynomial-remaining-compute.md</c> §3.4, unit U8): <c>leggauss</c>, <c>laggauss</c>,
    ///     <c>hermgauss</c> and <c>hermegauss</c>. NumPy 2.4.2's own <c>TestGauss.test_100</c> ported; NumPy's bytes for
    ///     small degrees; NumPy's error for every argument kind with a C# spelling (the probe scripts are quoted next to
    ///     <see cref="Pins"/> and <see cref="KindErrors"/>); the compositions against NumPy's statements written out by hand
    ///     (the Hermite ones with NumPy's normalised recurrences as array statements); and the missing-backend path.
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
        /// <summary>The four functions by name.</summary>
        private static readonly Dictionary<string, Func<object, (NDArray x, NDArray w)>> Functions = new()
        {
            ["leggauss"] = d => np.polynomial.legendre.leggauss(d),
            ["laggauss"] = d => np.polynomial.laguerre.laggauss(d),
            ["hermgauss"] = d => np.polynomial.hermite.hermgauss(d),
            ["hermegauss"] = d => np.polynomial.hermite_e.hermegauss(d),
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

        // NumPy 2.4.2's bytes and error texts (scratchpad gauss_pins.py and hg_facts.py, OPENBLAS_NUM_THREADS=1):
        //   for d in (1, 2, 3, 4, 5, 6, 12): {leg,lag,herm,herme}gauss(d), x.tobytes().hex(), w.tobytes().hex();
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
            ("hermgauss", 1, "0000000000000000", "6aefb491f85bfc3f"),
            ("hermgauss", 2, "cc3b7f669ea0e6bfcc3b7f669ea0e63f", "6aefb491f85bec3f6aefb491f85bec3f"),
            ("hermgauss", 3, "2e2109148e98f3bf00000000000000002e2109148e98f33f", "f3f4cd0bfbe7d23ff1f4cd0bfbe7f23ff3f4cd0bfbe7d23f"),
            ("hermgauss", 4, "60f2c38f2f69fabf2c87fccfe9c9e0bf2c87fccfe9c9e03f60f2c38f2f69fa3f", "beaefd00ebd0b43f92399531dbc1e93f92399531dbc1e93fbeaefd00ebd0b43f"),
            ("hermgauss", 5, "e76204a3552900c056994b29a0aceebf000000000000000056994b29a0acee3fe76204a355290040", "20005f689f6e943f4fcd001a0f31d93f8221e312f83fee3f4fcd001a0f31d93f20005f689f6e943f"),
            ("hermgauss", 6, "9ff9fdfa09ce02c0cc7d5447a35ff5bf4038b03bb1e8dbbf4038b03bb1e8db3fcc7d5447a35ff53f9ff9fdfa09ce0240", "3fc250460f8e723f964c072ec81ac43fbe7aa6672a30e73fbe7aa6672a30e73f964c072ec81ac43f3fc250460f8e723f"),
            ("hermgauss", 12, "1d4d5616281e0fc0fdd89fbe432a08c01a054e356e3c02c06fe1b5aa1b90f9bf67f5fc514854eebfe117c6aa831cd4bfe117c6aa831cd43f67f5fc514854ee3f6fe1b5aa1b90f93f1a054e356e3c0240fdd89fbe432a08401d4d5616281e0f40", "151460655bd7913ee3bbe437b479163f5bcfd99a32fe6f3fa085dda45c6caa3f6e6b5cf0e7abd03f396d41408c3ee23f396d41408c3ee23f6e6b5cf0e7abd03fa085dda45c6caa3f5bcfd99a32fe6f3fe3bbe437b479163f151460655bd7913e"),
            ("hermegauss", 1, "0000000000000000", "0527f61f930d0440"),
            ("hermegauss", 2, "000000000000f0bf000000000000f03f", "0527f61f930df43f0527f61f930df43f"),
            ("hermegauss", 3, "ab4c58e87ab6fbbf0000000000000000ab4c58e87ab6fb3f", "59899d2ac4bcda3f5e899d2ac4bcfa3f59899d2ac4bcda3f"),
            ("hermegauss", 4, "9faa985ce1ac02c0feb08cd52abee7bffeb08cd52abee73f9faa985ce1ac0240", "10c5d0633a70bd3fb41ab9798f36f23fb41ab9798f36f23f10c5d0633a70bd3f"),
            ("hermegauss", 5, "14e4391813db06c04174c913a5b0f5bf00000000000000004174c913a5b0f53f14e4391813db0640", "742fe0b037e59c3f76ab4d602cd0e13f19a11722d063f53f76ab4d602cd0e13f742fe0b037e59c3f"),
            ("hermegauss", 6, "deba044814980ac02d82357c103afebf105b83750fbce3bf105b83750fbce33f2d82357c103afe3fdeba044814980a40", "7437224b993d7a3fb9b4591aa66ecc3f37ce5fc38065f03f37ce5fc38065f03fb9b4591aa66ecc3f7437224b993d7a3f"),
            ("hermegauss", 12, "6dcc5c60ec0016c0754edc83591611c000e4f26028ca09c0282f1c19621302c0042c42432d72f5bf0809e6491971dcbf0809e6491971dc3f042c42432d72f53f282f1c196213024000e4f26028ca0940754edc83591611406dcc5c60ec001640", "132c4bde373b993ea2b0d758f7c81f3f26520f25589f763f3d843c4c20afb23fe68e4106bb93d73fbc5165f126cde93fbc5165f126cde93fe68e4106bb93d73f3d843c4c20afb23f26520f25589f763fa2b0d758f7c81f3f132c4bde373b993e"),
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
            ("hermgauss", "int 0", "!! ValueError: deg must be a positive integer"),
            ("hermgauss", "int -1", "!! ValueError: deg must be a positive integer"),
            ("hermgauss", "bool false", "!! ValueError: deg must be a positive integer"),
            ("hermgauss", "double 3.0", "!! TypeError: deg must be an integer, received 3.0"),
            ("hermgauss", "str 3", "!! TypeError: deg must be an integer, received 3"),
            ("hermgauss", "null", "!! TypeError: deg must be an integer, received None"),
            ("hermgauss", "char 0", "!! ValueError: deg must be a positive integer"),
            ("hermgauss", "nd0 int64 3", "!! ValueError: Series must have maximum degree of at least 1."),
            ("hermgauss", "nd0 int64 0", "!! ValueError: deg must be a positive integer"),
            ("hermgauss", "nd0 uint64 3", "!! ValueError: Series must have maximum degree of at least 1."),
            ("hermgauss", "nd0 int8 2", "!! ValueError: Series must have maximum degree of at least 1."),
            ("hermgauss", "nd0 bool true", "!! TypeError: deg must be an integer, received True"),
            ("hermgauss", "nd1 [3]", "!! TypeError: deg must be an integer, received [3]"),
            ("hermgauss", "object[] {3}", "!! TypeError: deg must be an integer, received [3]"),
            ("hermgauss", "tuple (3,)", "!! TypeError: deg must be an integer, received (3,)"),
            ("hermgauss", "complex 3", "!! TypeError: deg must be an integer, received (3+0j)"),
            ("hermgauss", "half 3", "!! TypeError: deg must be an integer, received 3.0"),
            ("hermgauss", "big 2**63", "!! OverflowError: cannot fit 'int' into an index-sized integer"),
            ("hermgauss", "big 2**64", "!! OverflowError: cannot fit 'int' into an index-sized integer"),
            ("hermgauss", "big 2**62", "!! MemoryError: "),
            ("hermgauss", "big 2**61", "!! MemoryError: "),
            ("hermgauss", "big -2**70", "!! ValueError: deg must be a positive integer"),
            ("hermegauss", "int 0", "!! ValueError: deg must be a positive integer"),
            ("hermegauss", "int -1", "!! ValueError: deg must be a positive integer"),
            ("hermegauss", "bool false", "!! ValueError: deg must be a positive integer"),
            ("hermegauss", "double 3.0", "!! TypeError: deg must be an integer, received 3.0"),
            ("hermegauss", "str 3", "!! TypeError: deg must be an integer, received 3"),
            ("hermegauss", "null", "!! TypeError: deg must be an integer, received None"),
            ("hermegauss", "char 0", "!! ValueError: deg must be a positive integer"),
            ("hermegauss", "nd0 int64 3", "!! ValueError: Series must have maximum degree of at least 1."),
            ("hermegauss", "nd0 int64 0", "!! ValueError: deg must be a positive integer"),
            ("hermegauss", "nd0 uint64 3", "!! ValueError: Series must have maximum degree of at least 1."),
            ("hermegauss", "nd0 int8 2", "!! ValueError: Series must have maximum degree of at least 1."),
            ("hermegauss", "nd0 bool true", "!! TypeError: deg must be an integer, received True"),
            ("hermegauss", "nd1 [3]", "!! TypeError: deg must be an integer, received [3]"),
            ("hermegauss", "object[] {3}", "!! TypeError: deg must be an integer, received [3]"),
            ("hermegauss", "tuple (3,)", "!! TypeError: deg must be an integer, received (3,)"),
            ("hermegauss", "complex 3", "!! TypeError: deg must be an integer, received (3+0j)"),
            ("hermegauss", "half 3", "!! TypeError: deg must be an integer, received 3.0"),
            ("hermegauss", "big 2**63", "!! OverflowError: cannot fit 'int' into an index-sized integer"),
            ("hermegauss", "big 2**64", "!! OverflowError: cannot fit 'int' into an index-sized integer"),
            ("hermegauss", "big 2**62", "!! MemoryError: "),
            ("hermegauss", "big 2**61", "!! MemoryError: "),
            ("hermegauss", "big -2**70", "!! ValueError: deg must be a positive integer"),
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
        ///     The four functions for degrees 1-6 and 12 have NumPy's float64 values (within 4 ULP; bit for bit on the pinned
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

        /// <summary>
        ///     hermgauss / hermegauss are NumPy's statements composed from the public functions, byte for byte on this host:
        ///     the companion of <c>[0]*deg + [1]</c>, eigvalsh, NumPy's normalised recurrences written as NumPy writes them
        ///     (array statements over the house ufuncs, the np.float64 scalars as 0-d arrays) for the Newton step and the
        ///     weights, the scaling, the symmetrisation and the normalisation to <c>sqrt(pi)</c> / <c>sqrt(2 pi)</c> — over
        ///     every degree to 80 and four degrees where NumPy's own results already hold NaN (400, 500, 700, 1000): the NaN
        ///     payloads are compared too.
        /// </summary>
        [TestMethod]
        public void HermiteResult_IsNumPysStatementsComposed()
        {
            RequireLapack();
            var degrees = new List<int>();
            for (int d = 1; d <= 80; d++)
                degrees.Add(d);
            degrees.AddRange(new[] { 400, 500, 700, 1000 });
            foreach (int deg in degrees)
            {
                foreach (bool physicists in new[] { true, false })
                {
                    var c = np.zeros(new Shape(deg + 1), np.int64);
                    c.SetAtIndex(1L, deg);
                    NDArray m = physicists ? np.polynomial.hermite.hermcompanion(c) : np.polynomial.hermite_e.hermecompanion(c);
                    NDArray x = np.linalg.eigvalsh(m);
                    NDArray dy = NormedHermite(physicists, x, deg);
                    NDArray df = NormedHermite(physicists, x, deg - 1) * NDArray.Scalar(System.Math.Sqrt(physicists ? 2.0 * deg : deg));
                    x = x - dy / df;
                    NDArray fm = NormedHermite(physicists, x, deg - 1);
                    fm = fm / np.max(np.abs(fm));
                    NDArray w = 1.0 / (fm * fm);
                    w = (w + w["::-1"]) / 2.0;
                    x = (x - x["::-1"]) / 2.0;
                    w = w * NDArray.Scalar(System.Math.Sqrt(physicists ? System.Math.PI : 2 * System.Math.PI) / w.sum().GetDouble());
                    var (gx, gw) = physicists ? np.polynomial.hermite.hermgauss(deg) : np.polynomial.hermite_e.hermegauss(deg);
                    string label = $"{(physicists ? "hermgauss" : "hermegauss")}({deg})";
                    Assert.AreEqual(Hex(x), Hex(gx), $"{label}.x");
                    Assert.AreEqual(Hex(w), Hex(gw), $"{label}.w");
                }
            }
        }

        // ---------------------------------------------------------------------------------------------
        //  NumPy's own tests (numpy/polynomial/tests/test_legendre.py, test_laguerre.py, test_hermite.py,
        //  test_hermite_e.py)
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     NumPy's <c>TestGauss.test_100</c> for the four bases: the 100 nodes and weights make <c>{p}vander(x, 99)</c>
        ///     orthonormal after normalisation (<c>assert_almost_equal(vv, np.eye(100))</c>), and the weights integrate 1 to
        ///     2 (Legendre), 1 (Laguerre), <c>sqrt(pi)</c> (Hermite) and <c>sqrt(2 pi)</c> (HermiteE).
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
            foreach (bool physicists in new[] { true, false })
            {
                var (x, w) = physicists ? np.polynomial.hermite.hermgauss(100) : np.polynomial.hermite_e.hermegauss(100);
                NDArray v = physicists ? np.polynomial.hermite.hermvander(x, 99) : np.polynomial.hermite_e.hermevander(x, 99);
                NDArray vv = np.dot(v.T * w, v);
                NDArray vd = 1.0 / np.sqrt(vv.diagonal());
                vv = np.expand_dims(vd, 1) * vv * vd;
                double maxDev = np.max(np.abs(vv - np.eye(100))).GetDouble();
                Assert.IsTrue(maxDev < 1.5e-7, $"{(physicists ? "hermgauss" : "hermegauss")}(100): orthogonality off by {maxDev}");
                double target = System.Math.Sqrt(physicists ? System.Math.PI : 2 * System.Math.PI);
                Assert.IsTrue(System.Math.Abs(w.sum().GetDouble() - target) < 1.5e-7, "the weights integrate 1");
            }
        }

        /// <summary>
        ///     NumPy's <c>_normed_hermite_n(x, n)</c> / <c>_normed_hermite_e_n(x, n)</c> as NumPy writes them, statement by
        ///     statement over NumSharp's operators (the np.float64 scalars as 0-d arrays): the reference the kernel-backed
        ///     composition is held to.
        /// </summary>
        /// <param name="physicists">Hermite (true) or HermiteE.</param>
        /// <param name="x">The points.</param>
        /// <param name="n">The degree (0 or more).</param>
        /// <returns>The values.</returns>
        private static NDArray NormedHermite(bool physicists, NDArray x, int n)
        {
            double c1Init = 1.0 / System.Math.Sqrt(System.Math.Sqrt(physicists ? System.Math.PI : 2 * System.Math.PI));
            if (n == 0)
                return np.full(x.Shape, c1Init, np.float64);
            NDArray c0 = NDArray.Scalar(0.0), c1 = NDArray.Scalar(c1Init);
            double nd = n;
            for (int i = 0; i < n - 1; i++)
            {
                NDArray tmp = c0;
                c0 = -c1 * NDArray.Scalar(System.Math.Sqrt((nd - 1.0) / nd));
                c1 = tmp + c1 * x * NDArray.Scalar(System.Math.Sqrt((physicists ? 2.0 : 1.0) / nd));
                nd -= 1.0;
            }
            return physicists ? c0 + c1 * x * NDArray.Scalar(System.Math.Sqrt(2.0)) : c0 + c1 * x;
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
