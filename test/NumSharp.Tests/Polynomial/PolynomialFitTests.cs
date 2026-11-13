using System;
using System.Collections.Generic;
using System.Numerics;
using NumSharp.Interop.OpenBLAS;

namespace NumSharp.Tests.Polynomial
{
    /// <summary>
    ///     The least-squares fits of <c>numpy.polynomial</c> (plan <c>docs/plans/numpy-polynomial-remaining-compute.md</c>
    ///     §3.6, unit U6): <c>polyfit</c>, <c>chebfit</c>, <c>legfit</c>, <c>lagfit</c>, <c>hermfit</c>, <c>hermefit</c>.
    ///     NumPy 2.4.2's own <c>TestFitting</c> ported for every basis; NumPy's bytes for the coefficients and the full
    ///     statistics (the probe script <c>fit_pins.py</c> is quoted next to <see cref="Pins"/>); NumPy's outcome for every
    ///     argument kind with a C# spelling; and the result object's contract (<see cref="NDPolyFitResult"/>).
    /// </summary>
    /// <remarks>
    ///     Every value comes out of <c>np.linalg.lstsq</c>, which needs a LAPACK-capable library (the bundled
    ///     scipy-openblas); those tests go <see cref="Assert.Inconclusive(string)"/> on a host without one. The argument
    ///     errors are raised before lstsq and are asserted on every host. <c>[DoNotParallelize]</c>: the tests bind the
    ///     process-wide OpenBLAS backend (<c>TensorEngine.Blas</c>).
    /// </remarks>
    [TestClass]
    [DoNotParallelize]
    public class PolynomialFitTests
    {
        /// <summary>The six fits by basis prefix, through their one facade overload.</summary>
        private static readonly Dictionary<string, Func<object, object, object, object, bool, object, NDPolyFitResult>> Fits = new()
        {
            ["poly"] = (x, y, d, r, f, w) => np.polynomial.polynomial.polyfit(x, y, d, r, f, w),
            ["cheb"] = (x, y, d, r, f, w) => np.polynomial.chebyshev.chebfit(x, y, d, r, f, w),
            ["leg"] = (x, y, d, r, f, w) => np.polynomial.legendre.legfit(x, y, d, r, f, w),
            ["lag"] = (x, y, d, r, f, w) => np.polynomial.laguerre.lagfit(x, y, d, r, f, w),
            ["herm"] = (x, y, d, r, f, w) => np.polynomial.hermite.hermfit(x, y, d, r, f, w),
            ["herme"] = (x, y, d, r, f, w) => np.polynomial.hermite_e.hermefit(x, y, d, r, f, w),
        };

        /// <summary>The six evaluations by basis prefix (NumPy's tests check the fits through them).</summary>
        private static readonly Dictionary<string, Func<object, object, NDArray>> Vals = new()
        {
            ["poly"] = (x, c) => np.polynomial.polynomial.polyval(x, c),
            ["cheb"] = (x, c) => np.polynomial.chebyshev.chebval(x, c),
            ["leg"] = (x, c) => np.polynomial.legendre.legval(x, c),
            ["lag"] = (x, c) => np.polynomial.laguerre.lagval(x, c),
            ["herm"] = (x, c) => np.polynomial.hermite.hermval(x, c),
            ["herme"] = (x, c) => np.polynomial.hermite_e.hermeval(x, c),
        };

        // NumPy 2.5.4's bytes (scratchpad fit_pins_254.py, OPENBLAS_NUM_THREADS=1; NumPy 2.5's OpenBLAS moved gelsd's last bits
        // from 2.4.2's): x = np.linspace(0, 2), y = x(x - 1)(x - 2),
        // w = zeros with w[1::2] = 1; per basis f(x, y, 3), f(x, y, [0, 2, 3]), f(x, y, 3, w=w), and f(x, y, 3, full=True)'s
        // resids, rank, s and rcond — every array's tobytes().hex().
        /// <summary>NumPy's float64 bytes of each basis' fits and full statistics.</summary>
        private static readonly (string P, string C3, string CList, string CW, string Res, int Rank, string S, string Rcond)[] Pins =
        {
            ("poly", "f166ccfd0a2cc3bc080000000000004009000000000008c0060000000000f03f", "58767dedc9cfd83f000000000000000000a0f6c869bee7bfb844fb02212fd33f", "a225ca3feeebf23cd6ffffffffffff3feeffffffffff07c0e8ffffffffffef3f", "d463399b80bbef39", 4, "093120599a42fe3f705a30938521e43fd84969794ff9c43f8128d471a384973f", "000000000000093d"),
            ("cheb", "fbfffffffffff7bff9ffffffffff0540f8fffffffffff7bff0ffffffffffcf3f", "44cb13de6161c33f000000000000000059d5f1c35912d0bfb04ebc82892aab3f", "e9fffffffffff7bff3ffffffffff0540f6fffffffffff7bff5ffffffffffcf3f", "0433218f9efff539", 4, "dce539ca795efd3f97bfa97b5773e83f308bd48d9e8acb3fc4d0c50683c2983f", "000000000000093d"),
            ("leg", "f1ffffffffffefbfc8cccccccccc0440feffffffffffffbf9b9999999999d93f", "7b0936893349cc3f0000000000000000b2aa4a1ab502d7bfeb1af15c2c25b73f", "2bffffffffffefbf7dcccccccccc044008000000000000c0f39999999999d93f", "567651991a0cf439", 4, "4921ade03db0fd3f17804c272bfee63f7025a84e9da6c93fd33e9e1a6f9b983f", "000000000000093d"),
            ("lag", "010000000000004002000000000020c0040000000000284005000000000018c0", "9ce643d2c5cab43f0000000000000000203a2346f544e33f6f33a3c0df19d7bf", "f5ffffffffffff3ff5ffffffffff1fc0f8ffffffffff2740f2ffffffffff17c0", "94bf782952e60e3a", 4, "b24d96d3f3d6fa3fda4065b8f22cf03f908dd2123ce6d93f21e5d93e84528f3f", "000000000000093d"),
            ("herm", "060000000000f8bf040000000000fc3f030000000000e8bffbffffffffffbf3f", "4967617d9f9ecc3f0000000000000000ff472704d876b7bfbce3002a8c92943f", "fcfffffffffff7bff8fffffffffffb3ff4ffffffffffe7bff1ffffffffffbf3f", "2fcb339e42c0f239", 4, "21705c9ab957fc3f5da51a4ced44ec3f2e7f32ffc43ad23f938aae6e698b973f", "000000000000093d"),
            ("herme", "f8ffffffffff07c0fdffffffffff1340fdffffffffff07c0080000000000f03f", "12cf0a488fffc63f000000000000000071bd47adb80bcdbf355d65e227f1ba3f", "feffffffffff07c00000000000001440ffffffffffff07c0eaffffffffffef3f", "e8e31bc2582b013a", 4, "b55853449564f93f7c4bbc0fba58f23f519986c15214da3f45775f8a89b3933f", "000000000000093d"),
        };

        /// <summary>
        ///     NumPy's outcome of a chebfit call per argument kind (fit_pins.py's "kinds"): the result's dtype and shape, or
        ///     <c>!! PythonType: message</c>. The C# spelling follows the house map (<c>object[]</c> a list, null None, a char
        ///     np.uint16, a <see cref="BigInteger"/> a Python int).
        /// </summary>
        private static readonly (string Label, Func<NDPolyFitResult> Call, string Outcome)[] Kinds =
        {
            ("deg -1", () => Cheb(L(1), L(1), -1), "!! ValueError: expected deg >= 0"),
            ("x 2-D", () => Cheb(N(L(1)), L(1), 0), "!! TypeError: expected 1D vector for x"),
            ("x empty", () => Cheb(L(), L(1), 0), "!! TypeError: expected non-empty vector for x"),
            ("y 3-D", () => Cheb(L(1), N(N(L(1))), 0), "!! TypeError: expected 1D or 2D array for y"),
            ("len x>y", () => Cheb(L(1, 2), L(1), 0), "!! TypeError: expected x and y to have same length"),
            ("len y>x", () => Cheb(L(1), L(1, 2), 0), "!! TypeError: expected x and y to have same length"),
            ("w 2-D", () => Cheb(L(1), L(1), 0, w: N(L(1))), "!! TypeError: expected 1D vector for w"),
            ("w long", () => Cheb(L(1), L(1), 0, w: L(1, 1)), "!! TypeError: expected x and w to have same length"),
            ("w short", () => Cheb(L(1, 2), L(1, 2), 0, w: L(1)), "!! TypeError: expected x and w to have same length"),
            ("deg empty int64", () => Cheb(L(1), L(1), np.zeros(new Shape(0), np.int64)), "!! TypeError: deg must be an int or non-empty 1-D array of int"),
            ("x str array", () => Cheb(np.array(new[] { "a", "b" }), L(1, 2), 0), "!! TypeError: ufunc 'add' did not contain a loop with signature matching types (dtype('<U1'), dtype('float64')) -> None"),
            ("y str array", () => Cheb(L(1, 2), np.array(new[] { "a", "b" }), 0), "!! TypeError: ufunc 'add' did not contain a loop with signature matching types (dtype('<U1'), dtype('float64')) -> None"),
            ("deg [-1]", () => Cheb(L(1), L(1), L(-1)), "!! ValueError: expected deg >= 0"),
            ("deg [2,-1,6]", () => Cheb(L(1), L(1), L(2, -1, 6)), "!! ValueError: expected deg >= 0"),
            ("deg []", () => Cheb(L(1), L(1), L()), "!! TypeError: deg must be an int or non-empty 1-D array of int"),
            ("deg True", () => Cheb(L(1), L(1), true), "!! TypeError: deg must be an int or non-empty 1-D array of int"),
            ("deg 3.0", () => Cheb(L(1), L(1), 3.0), "!! TypeError: deg must be an int or non-empty 1-D array of int"),
            ("deg [[1]]", () => Cheb(L(1), L(1), N(L(1))), "!! TypeError: deg must be an int or non-empty 1-D array of int"),
            ("x None", () => Cheb(null, L(1), 0), "!! TypeError: unsupported operand type(s) for +: 'NoneType' and 'float'"),
            ("y None", () => Cheb(L(1), null, 0), "!! TypeError: unsupported operand type(s) for +: 'NoneType' and 'float'"),
            ("x str", () => Cheb("ab", L(1, 2), 0), "!! TypeError: ufunc 'add' did not contain a loop with signature matching types (dtype('<U2'), dtype('float64')) -> None"),
            ("x f16", () => Cheb(np.ones(new Shape(3), np.float16), L(1, 2, 3), 1), "!! TypeError: array type float16 is unsupported in linalg"),
            ("rcond 1j", () => Cheb(L(1, 2), L(1, 2), 1, rcond: new Complex(0, 1)), "!! TypeError: float() argument must be a string or a real number, not 'complex'"),
            ("rcond 0.5+0j", () => Cheb(L(1, 2), L(1, 2), 1, rcond: new Complex(0.5, 0)), "!! TypeError: float() argument must be a string or a real number, not 'complex'"),
            ("rcond str", () => Cheb(L(1, 2), L(1, 2), 1, rcond: "0.5"), "!! TypeError: Cannot cast ufunc 'lstsq' input 2 from dtype('<U3') to dtype('float64') with casting rule 'same_kind'"),
            ("deg 2**63", () => Cheb(L(1, 2), L(1, 2), BigInteger.One << 63), "!! ValueError: Maximum allowed dimension exceeded"),
            ("deg 2**64", () => Cheb(L(1, 2), L(1, 2), BigInteger.One << 64), "!! TypeError: deg must be an int or non-empty 1-D array of int"),
        };

        /// <summary>A Python list (the house map: <c>object[]</c>).</summary>
        /// <param name="items">The items.</param>
        /// <returns>The list.</returns>
        private static object[] L(params object[] items) => items;

        /// <summary>
        ///     A Python list holding one item — the nesting <see cref="L"/> cannot spell: a lone <c>object[]</c> passed to a
        ///     <c>params object[]</c> parameter IS the params array, so <c>L(L(1))</c> would be <c>[1]</c>, not <c>[[1]]</c>.
        /// </summary>
        /// <param name="item">The one item (a nested list, typically).</param>
        /// <returns>The list <c>[item]</c>.</returns>
        private static object[] N(object item) => new[] { item };

        /// <summary><c>chebfit</c> with NumPy's keyword defaults.</summary>
        /// <param name="x">x.</param>
        /// <param name="y">y.</param>
        /// <param name="deg">deg.</param>
        /// <param name="rcond">rcond, or null.</param>
        /// <param name="w">w, or null.</param>
        /// <returns>The fit.</returns>
        private static NDPolyFitResult Cheb(object x, object y, object deg, object rcond = null, object w = null)
            => np.polynomial.chebyshev.chebfit(x, y, deg, rcond, false, w);

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
        ///     Every basis' fits are NumPy's float64 bytes — degree 3, the degree list [0, 2, 3] (re-expanded with zeros),
        ///     weights that zero half the points — and full=True's residuals, rank, singular values and rcond (the default
        ///     <c>len(x)*eps</c>).
        /// </summary>
        [TestMethod]
        public void Pins_AreNumPysBytes()
        {
            RequireLapack();
            NDArray x = np.linspace(0.0, 2.0, 50);
            NDArray y = x * (x - 1) * (x - 2);
            NDArray w = np.zeros_like(x);
            w["1::2"] = 1.0;
            foreach (var (p, c3, cl, cw, res, rank, s, rc) in Pins)
            {
                var f = Fits[p];
                Assert.AreEqual(c3, Hex(f(x, y, 3, null, false, null)), $"{p}fit(x, y, 3)");
                Assert.AreEqual(cl, Hex(f(x, y, new object[] { 0, 2, 3 }, null, false, null)), $"{p}fit(x, y, [0, 2, 3])");
                Assert.AreEqual(cw, Hex(f(x, y, 3, null, false, w)), $"{p}fit(x, y, 3, w=w)");
                var (c, (r, k, sv, rcond)) = f(x, y, 3, null, true, null);
                Assert.AreEqual(c3, Hex(c), $"{p}fit full: c");
                Assert.AreEqual(res, Hex(r), $"{p}fit full: resids");
                Assert.AreEqual("int32", k.dtype.name, $"{p}fit full: rank dtype (np.intc)");
                Assert.AreEqual(rank, k.GetAtIndex<int>(0), $"{p}fit full: rank");
                Assert.AreEqual(s, Hex(sv), $"{p}fit full: s");
                Assert.AreEqual(rc, Hex(rcond), $"{p}fit full: rcond");
            }
        }

        /// <summary>
        ///     The kinds a ufunc-level slip would get wrong without changing any value check, as NumPy 2.5.4's bytes
        ///     (scratchpad fit_pins_254.py, OPENBLAS_NUM_THREADS=1): an integer and a bool x (x + 0.0 is float64, so the default rcond is
        ///     <c>len(x)*eps</c> of float64), complex points (the norm is <c>|z|**2</c>, not <c>z**2</c>), an all-zero column
        ///     (x = zeros: <c>scl[scl == 0] = 1</c> leaves it unscaled, gelsd sees rank 1 and the residuals are empty) and a
        ///     float32 fit (float32 coefficients and statistics, rcond the np.float32 <c>7*eps</c>).
        /// </summary>
        [TestMethod]
        public void KindPins_AreNumPysBytes()
        {
            RequireLapack();
            NDArray y6 = np.array(new[] { 1.0, 2.0, 0.5, 4.0, 3.0, -1.0 });
            var (c, (res, rank, s, rc)) = np.polynomial.polynomial.polyfit(np.arange(6), y6, 2, null, true);
            Assert.AreEqual("56dbb66ddbb6dd3f22d4411dd441ff3f942449922449dabf", Hex(c), "int x: c");
            Assert.AreEqual("8caff88aaf782340", Hex(res), "int x: resids");
            Assert.AreEqual(3, rank.GetAtIndex<int>(0), "int x: rank");
            Assert.AreEqual("9ee32a4dfa30fa3fd33f2d7f43b0e13f28bec6df062abf3f", Hex(s), "int x: s");
            Assert.AreEqual("float64", rc.dtype.name, "int x: rcond dtype");
            Assert.AreEqual("000000000000d83c", Hex(rc), "int x: rcond = 6 * eps");

            (c, (res, rank, s, rc)) = np.polynomial.polynomial.polyfit(
                np.array(new[] { true, false, true, true, false, false }), y6, 1, null, true);
            Assert.AreEqual("565555555555f53f020000000000e03f", Hex(c), "bool x: c");
            Assert.AreEqual("a9aaaaaaaaaa2f40", Hex(res), "bool x: resids");
            Assert.AreEqual(2, rank.GetAtIndex<int>(0), "bool x: rank");
            Assert.AreEqual("fdf04491aee7f43f9438db7b7a51e13f", Hex(s), "bool x: s");
            Assert.AreEqual("000000000000d83c", Hex(rc), "bool x: rcond");

            NDArray xs = np.array(new[]
            {
                new Complex(0.5, 0.5), new Complex(-0.25, 1), new Complex(1, -0.5), new Complex(0.1, 0.2), new Complex(-0.8, -0.3),
            });
            NDArray ys = np.array(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
            (c, (_, _, s, _)) = np.polynomial.chebyshev.chebfit(xs, ys, 2, null, true);
            Assert.AreEqual("complex128", c.dtype.name, "complex x: dtype");
            Assert.AreEqual("c66bbf9a9e5e0c4044fa606882cea7bf41a7cf4b3636ecbfc46637aa2418e63f478a0193ebd9d43f3f369e96c350ccbf", Hex(c), "complex x chebfit: c");
            Assert.AreEqual("42a2f053c062f43f96514620b5ddee3f149fb4867d60e53f", Hex(s), "complex x chebfit: s");
            Assert.AreEqual("c66bbf9a9e5e0c4044fa606882cea7bf41a7cf4b3636dcbfc46637aa2418d63f478a0193ebd9c43f3f369e96c350bcbf",
                Hex(np.polynomial.hermite.hermfit(xs, ys, 2)), "complex x hermfit: c");

            (c, (res, rank, s, _)) = np.polynomial.polynomial.polyfit(np.zeros(new Shape(4)), np.array(new[] { 1.0, 2.0, 3.0, 4.0 }), 2, null, true);
            Assert.AreEqual("000000000000044000000000000000000000000000000000", Hex(c), "zero columns: c = [2.5, 0, 0]");
            Assert.AreEqual(0, res.size, "zero columns: rank deficient, so no residuals");
            Assert.AreEqual(1, rank.GetAtIndex<int>(0), "zero columns: rank");
            Assert.AreEqual("000000000000f03f00000000000000000000000000000000", Hex(s), "zero columns: s");

            // np.linspace(-1, 1, 7, dtype=np.float32): computed in float64, then cast (NumPy's linspace)
            NDArray x32 = np.linspace(-1.0, 1.0, 7).astype(np.float32);
            NDArray y32 = np.cos(x32 * np.array(2.0f));
            (c, (res, rank, s, rc)) = np.polynomial.chebyshev.chebfit(x32, y32, 2, null, true);
            Assert.AreEqual("float32", c.dtype.name, "float32: c dtype");
            Assert.AreEqual("41cc743e6b2311a5939830bf", Hex(c), "float32: c");
            Assert.AreEqual("float32", res.dtype.name, "float32: resids dtype");
            Assert.AreEqual("e0e4a83c", Hex(res), "float32: resids");
            Assert.AreEqual(3, rank.GetAtIndex<int>(0), "float32: rank");
            Assert.AreEqual("78d6883fffff7f3f8d026d3f", Hex(s), "float32: s");
            Assert.AreEqual("float32", rc.dtype.name, "float32: rcond is np.float32(7 * eps)");
            Assert.AreEqual("00006035", Hex(rc), "float32: rcond");
        }

        /// <summary>
        ///     Every argument kind gives NumPy's outcome, in NumPy's order (these are raised before lstsq computes, so they hold
        ///     with or without a backend): deg must be an integer or a non-empty 1-D integer array, the shapes and lengths of
        ///     x / y / w, None and str arguments (Python's and the ufunc add's TypeErrors), a float16 x (lstsq's), complex and str
        ///     rcond, the Vandermonde's dimension error at 2**63.
        /// </summary>
        [TestMethod]
        public void ArgumentKinds_MatchNumPy()
        {
            foreach (bool backend in new[] { false, true })
            {
                if (backend)
                    RequireLapack();
                foreach (var (label, call, outcome) in Kinds)
                    Assert.AreEqual(outcome, Outcome(call), $"{label} (backend {backend})");
            }
        }

        /// <summary>
        ///     The degree kinds that compute: a char (np.uint16) and a 0-d integer array of any width fit as their integer; a
        ///     degree list may be unsorted or repeat a degree (the last fitted row wins, as in NumPy's sequential assignment).
        /// </summary>
        [TestMethod]
        public void DegreeKinds_Compute()
        {
            RequireLapack();
            NDArray x = np.linspace(-1.0, 1.0, 9), y = np.cos(x);
            string reference = Hex(Cheb(x, y, 2));
            Assert.AreEqual(reference, Hex(Cheb(x, y, (char)2)), "char");
            Assert.AreEqual(reference, Hex(Cheb(x, y, np.array((byte)2))), "0-d uint8");
            Assert.AreEqual(reference, Hex(Cheb(x, y, np.array(2L))), "0-d int64");
            Assert.AreEqual(Hex(Cheb(x, y, new object[] { 0, 1, 2 })), Hex(Cheb(x, y, new object[] { 2, 0, 1 })), "unsorted list");
            CollectionAssert.AreEqual(new long[] { 4 }, Cheb(x, y, new object[] { 3, 1, 1 }).shape, "duplicates: lmax + 1");
        }

        // ---------------------------------------------------------------------------------------------
        //  The result object
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     The result IS the coefficient array: it indexes, evaluates (passed as the series of <c>chebval</c>, an object
        ///     parameter), and its memory is NumPy's — a VIEW for a scalar degree (OWNDATA false), an owner for a degree list.
        /// </summary>
        [TestMethod]
        public void Result_IsTheCoefficientArray()
        {
            RequireLapack();
            NDArray x = np.linspace(-1.0, 1.0, 21);
            NDArray y = x * x * x - x;
            var c = np.polynomial.chebyshev.chebfit(x, y, 3);
            Assert.IsInstanceOfType(c, typeof(NDArray));
            NDArray fitted = np.polynomial.chebyshev.chebval(x, c);
            Assert.IsTrue(np.max(np.abs(fitted - y)).GetDouble() < 1e-13, "chebval(x, chebfit(x, y, 3)) reproduces y");
            Assert.IsFalse(c.flags.owndata, "a scalar degree: NumPy's (c.T / scl).T is a view");
            Assert.IsTrue(c.flags.c_contiguous && c.flags.f_contiguous);
            var cl = np.polynomial.chebyshev.chebfit(x, y, new object[] { 1, 3 });
            Assert.IsTrue(cl.flags.owndata, "a degree list: NumPy's cc owns its data");
            Assert.AreEqual(0.0, cl.GetDouble(0), "an unfitted term is zero");
            Assert.IsNull(c.residuals);
            Assert.IsNull(c.rank);
        }

        /// <summary>
        ///     full=True deconstructs as NumPy's <c>(c, [resids, rank, s, rcond])</c>: the coefficients first, then the four
        ///     statistics in NumPy's order; the given rcond is returned (as the 0-d array of its value); without full the
        ///     deconstruction is refused.
        /// </summary>
        [TestMethod]
        public void Full_DeconstructsAsNumPysTuple()
        {
            RequireLapack();
            NDArray x = np.linspace(-1.0, 1.0, 11), y = np.sin(x);
            var (c, (res, rank, s, rcond)) = np.polynomial.legendre.legfit(x, y, 4, 0.25, true);
            Assert.AreEqual(5, c.size);
            Assert.AreEqual(1, res.size, "over-determined and full rank: one residual");
            Assert.AreEqual(5, rank.GetAtIndex<int>(0));
            Assert.AreEqual(5, s.size);
            Assert.AreEqual(0.25, rcond.GetAtIndex<double>(0));
            var bare = np.polynomial.legendre.legfit(x, y, 4);
            Assert.ThrowsException<ValueError>(() =>
            {
                var (c2, info) = bare;
            });
        }

        /// <summary>
        ///     The statistics outlive the scope the fit ran in: a scoped caller that returns the result yields only the
        ///     coefficient array, so the statistics are owned by the result object, not by any scope.
        /// </summary>
        [TestMethod]
        public void Statistics_OutliveTheCallersScope()
        {
            RequireLapack();
            NDPolyFitResult r;
            using (NDScope.Open())
            {
                NDArray x = np.linspace(-1.0, 1.0, 11), y = np.sin(x);
                r = np.polynomial.hermite.hermfit(x, y, 3, null, true);
                NDScope.Detach(r);
            }
            Assert.IsFalse(r.IsDisposed);
            Assert.IsFalse(r.residuals.IsDisposed, "residuals");
            Assert.IsFalse(r.rank.IsDisposed, "rank");
            Assert.IsFalse(r.singular_values.IsDisposed, "singular values");
            Assert.IsFalse(r.rcond.IsDisposed, "rcond");
            Assert.AreEqual(4, r.rank.GetAtIndex<int>(0));
        }

        /// <summary>
        ///     An rcond ARRAY with dimensions is refused (<c>[Misaligned]</c>): NumPy broadcasts its lstsq gufunc over it, which
        ///     gives every result an extra leading axis; NumSharp's lstsq takes a scalar cutoff.
        /// </summary>
        [TestMethod]
        [Misaligned]
        public void RcondArray_IsRefused()
        {
            RequireLapack();
            NDArray x = np.linspace(-1.0, 1.0, 5);
            Assert.ThrowsException<NotSupportedException>(() => Cheb(x, x, 1, rcond: new object[] { 0.5 }));
            Assert.ThrowsException<NotSupportedException>(() => Cheb(x, x, 1, rcond: np.array(new[] { 0.5 })));
            Assert.AreEqual(Hex(Cheb(x, x, 1, rcond: 0.5)), Hex(Cheb(x, x, 1, rcond: np.array(0.5))), "a 0-d array is a scalar");
        }

        /// <summary>
        ///     Without a LAPACK backend a valid fit raises the missing-backend exception from lstsq (NumSharp ships no managed
        ///     least-squares solver), after every argument check NumPy runs first.
        /// </summary>
        [TestMethod]
        public void WithoutBackend_ValidFitRaisesMissingBackend()
        {
            OpenBlasEngine.Disable();
            foreach (var (p, f) in Fits)
            {
                Exception missing = null;
                try
                {
                    f(new object[] { 1.0, 2.0, 3.0 }, new object[] { 1.0, 0.0, 1.0 }, 1, null, false, null);
                }
                catch (Exception e)
                {
                    missing = e;
                }
                Assert.IsInstanceOfType(missing, typeof(MissingBackendException), $"{p}fit: {missing?.GetType().Name}");
                Assert.ThrowsException<ValueError>(() => f(new object[] { 1.0 }, new object[] { 1.0 }, -1, null, false, null), $"{p}fit deg -1");
            }
        }

        // ---------------------------------------------------------------------------------------------
        //  NumPy's own tests (numpy/polynomial/tests/test_*.py, TestFitting)
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     NumPy's <c>TestFitting</c> for every basis: its exceptions; fits of <c>x(x - 1)(x - 2)</c> over
        ///     <c>np.linspace(0, 2)</c> at degree 3 and 4 and their degree lists (strict, then not increasing) reproduce y;
        ///     a 2-D y fits each column; weights that keep half the points give the same coefficients; complex points whose
        ///     squares sum to zero (<c>[1, 1j, -1, -1j]</c>) fit the identity; the even terms of an even function.
        /// </summary>
        [TestMethod]
        public void NumPysOwnTestFitting()
        {
            RequireLapack();
            foreach (var (p, fit) in Fits)
            {
                var val = Vals[p];
                NDPolyFitResult F(object x, object y, object d, object w = null) => fit(x, y, d, null, false, w);
                Assert.ThrowsException<ValueError>(() => F(L(1), L(1), -1));
                Assert.ThrowsException<TypeError>(() => F(N(L(1)), L(1), 0));
                Assert.ThrowsException<TypeError>(() => F(L(), L(1), 0));
                Assert.ThrowsException<TypeError>(() => F(L(1), N(N(L(1))), 0));
                Assert.ThrowsException<TypeError>(() => F(L(1, 2), L(1), 0));
                Assert.ThrowsException<TypeError>(() => F(L(1), L(1, 2), 0));
                Assert.ThrowsException<TypeError>(() => F(L(1), L(1), 0, N(L(1))));
                Assert.ThrowsException<TypeError>(() => F(L(1), L(1), 0, L(1, 1)));
                Assert.ThrowsException<ValueError>(() => F(L(1), L(1), L(-1)));
                Assert.ThrowsException<ValueError>(() => F(L(1), L(1), L(2, -1, 6)));
                Assert.ThrowsException<TypeError>(() => F(L(1), L(1), L()));

                NDArray x = np.linspace(0.0, 2.0, 50);
                NDArray y = x * (x - 1) * (x - 2);
                void Near(NDArray got, NDArray want, string what)
                {
                    double err = np.max(np.abs(got - want)).GetDouble();
                    Assert.IsTrue(err < 1.5e-7, $"{p}fit {what}: off by {err}");
                }
                NDArray coef3 = F(x, y, 3);
                Assert.AreEqual(4, coef3.size);
                Near(val(x, coef3), y, "degree 3");
                Near(val(x, F(x, y, L(0, 1, 2, 3))), y, "degree list 3");
                NDArray coef4 = F(x, y, 4);
                Assert.AreEqual(5, coef4.size);
                Near(val(x, coef4), y, "degree 4");
                Near(val(x, F(x, y, L(0, 1, 2, 3, 4))), y, "degree list 4");
                Near(val(x, F(x, y, L(2, 3, 4, 1, 0))), y, "degree list not increasing");
                NDArray yy = np.stack(new[] { y, y }, axis: 1);
                Near(F(x, yy, 3), np.stack(new[] { coef3, coef3 }, axis: 1), "2-D y");
                Near(F(x, yy, L(0, 1, 2, 3)), np.stack(new[] { coef3, coef3 }, axis: 1), "2-D y, degree list");
                NDArray w = np.zeros_like(x);
                NDArray yw = y.copy();
                w["1::2"] = 1.0;
                yw["0::2"] = 0.0;
                Near(F(x, yw, 3, w), coef3, "weights");
                Near(F(x, yw, L(0, 1, 2, 3), w), coef3, "weights, degree list");
                Near(F(x, np.stack(new[] { yw, yw }, axis: 1), 3, w), np.stack(new[] { coef3, coef3 }, axis: 1), "weights, 2-D y");
                // complex points whose squares sum to zero: the norm is |.|**2, not the square. The identity x in each basis
                // (NumPy's per-basis expectations: x = T1 = P1 = He1, x = H1/2, x = L0 - L1).
                object[] cx = { new Complex(1, 0), new Complex(0, 1), new Complex(-1, 0), new Complex(0, -1) };
                double[] identity = p switch { "lag" => new[] { 1.0, -1.0 }, "herm" => new[] { 0.0, 0.5 }, _ => new[] { 0.0, 1.0 } };
                NDArray ci = F(cx, cx, 1), cil = F(cx, cx, L(0, 1));
                Assert.AreEqual("complex128", ci.dtype.name);
                Near(ci, np.array(identity), "complex identity");
                Near(cil, np.array(identity), "complex identity, degree list");
                // the even terms of an even function fit as well as the full degree (every basis but Laguerre's has parity;
                // NumPy's test_laguerre has no such case)
                if (p == "lag")
                    continue;
                NDArray xe = np.linspace(-1.0, 1.0, 50);
                NDArray ye = xe * xe * xe * xe + xe * xe + 1.0;
                NDArray c1 = F(xe, ye, 4), c2 = F(xe, ye, L(0, 2, 4));
                Near(val(xe, c1), ye, "even, degree 4");
                Near(val(xe, c2), ye, "even, degree list");
                Near(c1, c2, "even coefficients");
            }
        }

        // ---------------------------------------------------------------------------------------------
        //  Helpers
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     A call's outcome in the probe's text: the result's dtype and shape (Python's tuple text), or
        ///     <c>!! PythonType: message</c> — NumSharp's exceptions named after NumPy's (a UFuncTypeError is a TypeError).
        /// </summary>
        /// <param name="call">The call.</param>
        /// <returns>The text.</returns>
        private static string Outcome(Func<NDPolyFitResult> call)
        {
            try
            {
                NDArray r = call();
                return $"{r.dtype.name} {r.Shape.ToPythonTuple()}";
            }
            catch (ValueError e) { return "!! ValueError: " + e.Message; }
            catch (TypeError e) { return "!! TypeError: " + e.Message; }
        }

        /// <summary>The C-order bytes of <paramref name="a"/> as lowercase hex (NumPy's <c>tobytes().hex()</c>).</summary>
        /// <param name="a">The array.</param>
        /// <returns>The hex text.</returns>
        private static string Hex(NDArray a) => Convert.ToHexString(np.ascontiguousarray(a).tobytes()).ToLowerInvariant();
    }
}
