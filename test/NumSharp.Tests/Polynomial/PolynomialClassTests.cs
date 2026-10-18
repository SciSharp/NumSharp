using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Numerics;
using NumSharp.Interop.OpenBLAS;

namespace NumSharp.Tests.Polynomial
{
    /// <summary>
    ///     numpy.polynomial's six series classes (plan <c>docs/plans/numpy-polynomial-remaining-compute.md</c> §3.7, unit U10):
    ///     <see cref="NDPolynomial"/>, <see cref="NDChebyshev"/>, <see cref="NDLegendre"/>, <see cref="NDLaguerre"/>,
    ///     <see cref="NDHermite"/>, <see cref="NDHermiteE"/>. NumPy 2.4.2's <c>test_classes.py</c> and <c>test_symbol.py</c>
    ///     ported for every class (random coefficients replaced by fixed ones), and the C# spelling of NumPy's protocol pinned
    ///     (which operator binds for which left operand, <c>Call</c> at a series, the class defaults as statics, disposal,
    ///     pickling). The <c>polyclass</c> oracle tiers hold the byte-level matrix.
    /// </summary>
    /// <remarks>
    ///     <c>[DoNotParallelize]</c>: some tests bind the process-wide OpenBLAS backend (fit and the roots of degree two and
    ///     more need LAPACK; they go Inconclusive on a host without one) and one assigns a class default. Every expected text
    ///     and value below was printed by NumPy 2.4.2 for the same call.
    /// </remarks>
    [TestClass]
    [DoNotParallelize]
    public class PolynomialClassTests
    {
        /// <summary>The six classes, in NumPy's test order.</summary>
        private static readonly Type[] Classes =
            { typeof(NDPolynomial), typeof(NDLegendre), typeof(NDChebyshev), typeof(NDLaguerre), typeof(NDHermite), typeof(NDHermiteE) };

        /// <summary>Leaves the process-global OpenBLAS engine off after every test.</summary>
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
        //  Helpers: the class-generic spellings NumPy's fixtures use
        // ---------------------------------------------------------------------------------------------

        /// <summary><c>Poly(coef, domain, window, symbol)</c> for a class given by its type.</summary>
        /// <param name="t">The class.</param><param name="coef">The coefficients.</param>
        /// <param name="domain">The domain, or null.</param><param name="window">The window, or null.</param>
        /// <param name="symbol">The symbol.</param>
        /// <returns>The series.</returns>
        private static NDPolyBase New(Type t, object coef, object domain = null, object window = null, string symbol = "x")
        {
            if (t == typeof(NDPolynomial)) return new NDPolynomial(coef, domain, window, symbol);
            if (t == typeof(NDChebyshev)) return new NDChebyshev(coef, domain, window, symbol);
            if (t == typeof(NDLegendre)) return new NDLegendre(coef, domain, window, symbol);
            if (t == typeof(NDLaguerre)) return new NDLaguerre(coef, domain, window, symbol);
            if (t == typeof(NDHermite)) return new NDHermite(coef, domain, window, symbol);
            return new NDHermiteE(coef, domain, window, symbol);
        }

        /// <summary>NumPy's class attribute <c>Poly.domain</c>.</summary>
        /// <param name="t">The class.</param>
        /// <returns>The shared default.</returns>
        private static NDArray ClassDomain(Type t)
        {
            if (t == typeof(NDPolynomial)) return NDPolynomial.default_domain;
            if (t == typeof(NDChebyshev)) return NDChebyshev.default_domain;
            if (t == typeof(NDLegendre)) return NDLegendre.default_domain;
            if (t == typeof(NDLaguerre)) return NDLaguerre.default_domain;
            if (t == typeof(NDHermite)) return NDHermite.default_domain;
            return NDHermiteE.default_domain;
        }

        /// <summary>NumPy's class attribute <c>Poly.window</c>.</summary>
        /// <param name="t">The class.</param>
        /// <returns>The shared default.</returns>
        private static NDArray ClassWindow(Type t)
        {
            if (t == typeof(NDPolynomial)) return NDPolynomial.default_window;
            if (t == typeof(NDChebyshev)) return NDChebyshev.default_window;
            if (t == typeof(NDLegendre)) return NDLegendre.default_window;
            if (t == typeof(NDLaguerre)) return NDLaguerre.default_window;
            if (t == typeof(NDHermite)) return NDHermite.default_window;
            return NDHermiteE.default_window;
        }

        /// <summary><c>Poly.identity(domain, window, symbol)</c>.</summary>
        /// <param name="t">The class.</param><param name="d">The domain.</param><param name="w">The window.</param>
        /// <param name="symbol">The symbol.</param>
        /// <returns>The identity series.</returns>
        private static NDPolyBase Identity(Type t, object d, object w, string symbol = "x")
        {
            if (t == typeof(NDPolynomial)) return NDPolynomial.identity(d, w, symbol);
            if (t == typeof(NDChebyshev)) return NDChebyshev.identity(d, w, symbol);
            if (t == typeof(NDLegendre)) return NDLegendre.identity(d, w, symbol);
            if (t == typeof(NDLaguerre)) return NDLaguerre.identity(d, w, symbol);
            if (t == typeof(NDHermite)) return NDHermite.identity(d, w, symbol);
            return NDHermiteE.identity(d, w, symbol);
        }

        /// <summary><c>Poly.basis(deg, domain, window, symbol)</c>.</summary>
        /// <param name="t">The class.</param><param name="deg">The degree.</param><param name="d">The domain.</param>
        /// <param name="w">The window.</param><param name="symbol">The symbol.</param>
        /// <returns>The basis series.</returns>
        private static NDPolyBase Basis(Type t, object deg, object d = null, object w = null, string symbol = "x")
        {
            if (t == typeof(NDPolynomial)) return NDPolynomial.basis(deg, d, w, symbol);
            if (t == typeof(NDChebyshev)) return NDChebyshev.basis(deg, d, w, symbol);
            if (t == typeof(NDLegendre)) return NDLegendre.basis(deg, d, w, symbol);
            if (t == typeof(NDLaguerre)) return NDLaguerre.basis(deg, d, w, symbol);
            if (t == typeof(NDHermite)) return NDHermite.basis(deg, d, w, symbol);
            return NDHermiteE.basis(deg, d, w, symbol);
        }

        /// <summary><c>Poly.fromroots(roots, domain, window)</c>.</summary>
        /// <param name="t">The class.</param><param name="roots">The roots.</param><param name="d">The domain.</param>
        /// <param name="w">The window.</param>
        /// <returns>The series.</returns>
        private static NDPolyBase FromRoots(Type t, object roots, object d, object w)
        {
            if (t == typeof(NDPolynomial)) return NDPolynomial.fromroots(roots, d, w);
            if (t == typeof(NDChebyshev)) return NDChebyshev.fromroots(roots, d, w);
            if (t == typeof(NDLegendre)) return NDLegendre.fromroots(roots, d, w);
            if (t == typeof(NDLaguerre)) return NDLaguerre.fromroots(roots, d, w);
            if (t == typeof(NDHermite)) return NDHermite.fromroots(roots, d, w);
            return NDHermiteE.fromroots(roots, d, w);
        }

        /// <summary><c>Poly.fit(x, y, deg, domain, w=w, window=window)</c>.</summary>
        /// <param name="t">The class.</param><param name="x">x.</param><param name="y">y.</param><param name="deg">deg.</param>
        /// <param name="d">The domain.</param><param name="w">The weights.</param><param name="window">The window.</param>
        /// <returns>The fitted series.</returns>
        private static NDPolyBase Fit(Type t, object x, object y, object deg, object d = null, object w = null, object window = null)
        {
            if (t == typeof(NDPolynomial)) return NDPolynomial.fit(x, y, deg, d, w: w, window: window);
            if (t == typeof(NDChebyshev)) return NDChebyshev.fit(x, y, deg, d, w: w, window: window);
            if (t == typeof(NDLegendre)) return NDLegendre.fit(x, y, deg, d, w: w, window: window);
            if (t == typeof(NDLaguerre)) return NDLaguerre.fit(x, y, deg, d, w: w, window: window);
            if (t == typeof(NDHermite)) return NDHermite.fit(x, y, deg, d, w: w, window: window);
            return NDHermiteE.fit(x, y, deg, d, w: w, window: window);
        }

        /// <summary><c>Poly.cast(series, domain, window)</c>.</summary>
        /// <param name="t">The target class.</param><param name="s">The series.</param>
        /// <param name="d">The domain.</param><param name="w">The window.</param>
        /// <returns>The converted series.</returns>
        private static NDPolyBase Cast(Type t, NDPolyBase s, object d = null, object w = null)
        {
            if (t == typeof(NDPolynomial)) return NDPolynomial.cast(s, d, w);
            if (t == typeof(NDChebyshev)) return NDChebyshev.cast(s, d, w);
            if (t == typeof(NDLegendre)) return NDLegendre.cast(s, d, w);
            if (t == typeof(NDLaguerre)) return NDLaguerre.cast(s, d, w);
            if (t == typeof(NDHermite)) return NDHermite.cast(s, d, w);
            return NDHermiteE.cast(s, d, w);
        }

        /// <summary>NumPy's <c>assert_poly_almost_equal</c>: equal domains and windows, coefficients within 1e-7.</summary>
        /// <param name="actual">The result.</param><param name="expected">The target.</param>
        /// <param name="label">What is compared (for the failure line).</param>
        private static void AssertPolyAlmostEqual(NDPolyBase actual, NDPolyBase expected, string label = "")
        {
            Assert.IsTrue(np.array_equal(actual.domain, expected.domain), $"{label}: domain {actual.domain} != {expected.domain}");
            Assert.IsTrue(np.array_equal(actual.window, expected.window), $"{label}: window {actual.window} != {expected.window}");
            Assert.AreEqual(expected.coef.size, actual.coef.size, $"{label}: {actual} vs {expected}");
            Assert.IsTrue(np.allclose(actual.coef, expected.coef, 0, 1e-7), $"{label}: {actual} vs {expected}");
        }

        /// <summary>A Python list (the house map: <c>object[]</c>).</summary>
        /// <param name="items">The items.</param>
        /// <returns>The list.</returns>
        private static object[] L(params object[] items) => items;

        /// <summary>NumPy's <c>d1 = Poly.domain + offsets</c>: the class default shifted by fixed offsets (NumPy's random(2)·.25).</summary>
        /// <param name="def">The class default.</param><param name="a">The first offset.</param><param name="b">The second.</param>
        /// <returns>A new two-element array.</returns>
        private static NDArray Shifted(NDArray def, double a, double b) => def + np.array(new[] { a, b });

        // ---------------------------------------------------------------------------------------------
        //  test_classes.py: conversion between classes
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     NumPy's <c>test_conversion</c> for every pair of classes: <c>p1.convert(kind=Poly2, domain=d2, window=w2)</c> has
        ///     the requested domain and window and the same values as p1 — the series evaluated AT the target class's identity
        ///     series, NumPy's recurrence with series values.
        /// </summary>
        [TestMethod]
        public void Convert_EveryPairOfClasses_KeepsTheValues()
        {
            var x = np.linspace(0, 1, 10);
            foreach (var t1 in Classes)
            foreach (var t2 in Classes)
            {
                using var p1 = New(t1, L(0.3, 0.7, 0.2), Shifted(ClassDomain(t1), 0.05, 0.2), Shifted(ClassWindow(t1), 0.1, 0.15));
                var d2 = Shifted(ClassDomain(t2), 0.12, 0.03);
                var w2 = Shifted(ClassWindow(t2), 0.2, 0.07);
                using var p2 = p1.convert(d2, t2, w2);
                Assert.AreEqual(t2, p2.GetType(), $"{t1.Name} -> {t2.Name}");
                Assert.IsTrue(np.allclose(p2.domain, d2), $"{t1.Name} -> {t2.Name}: domain");
                Assert.IsTrue(np.allclose(p2.window, w2), $"{t1.Name} -> {t2.Name}: window");
                Assert.IsTrue(np.allclose(p2.Call(x), p1.Call(x)), $"{t1.Name} -> {t2.Name}: values {p2.Call(x)} vs {p1.Call(x)}");
            }
        }

        /// <summary>NumPy's <c>test_cast</c>: <c>Poly2.cast(p1, domain, window)</c> is the same conversion.</summary>
        [TestMethod]
        public void Cast_EveryPairOfClasses_KeepsTheValues()
        {
            var x = np.linspace(0, 1, 10);
            foreach (var t1 in Classes)
            foreach (var t2 in Classes)
            {
                using var p1 = New(t1, L(0.6, -0.1, 0.4), Shifted(ClassDomain(t1), 0.15, 0.05), Shifted(ClassWindow(t1), 0.02, 0.21));
                var d2 = Shifted(ClassDomain(t2), 0.07, 0.11);
                var w2 = Shifted(ClassWindow(t2), 0.09, 0.04);
                using var p2 = Cast(t2, p1, d2, w2);
                Assert.IsTrue(np.allclose(p2.domain, d2) && np.allclose(p2.window, w2), $"{t1.Name} -> {t2.Name}");
                Assert.IsTrue(np.allclose(p2.Call(x), p1.Call(x)), $"{t1.Name} -> {t2.Name}");
            }
        }

        // ---------------------------------------------------------------------------------------------
        //  test_classes.py: class methods
        // ---------------------------------------------------------------------------------------------

        /// <summary>NumPy's <c>test_identity</c>: the identity series maps every point of the domain to itself.</summary>
        [TestMethod]
        public void Identity_MapsEachPointToItself()
        {
            foreach (var t in Classes)
            {
                var d = Shifted(ClassDomain(t), 0.1, 0.2);
                var w = Shifted(ClassWindow(t), 0.05, 0.15);
                var x = np.linspace(d.GetDouble(0), d.GetDouble(1), 11);
                using var p = Identity(t, d, w);
                Assert.IsTrue(np.array_equal(p.domain, d) && np.array_equal(p.window, w), t.Name);
                Assert.IsTrue(np.allclose(p.Call(x), x), t.Name);
            }
        }

        /// <summary>NumPy's <c>test_basis</c>: <c>Poly.basis(5)</c> has coefficients <c>[0]*5 + [1]</c> and the given domain / window.</summary>
        [TestMethod]
        public void Basis_HasOneUnitCoefficient()
        {
            foreach (var t in Classes)
            {
                var d = Shifted(ClassDomain(t), 0.1, 0.2);
                var w = Shifted(ClassWindow(t), 0.05, 0.15);
                using var p = Basis(t, 5, d, w);
                Assert.IsTrue(np.array_equal(p.domain, d) && np.array_equal(p.window, w), t.Name);
                Assert.IsTrue(np.array_equal(p.coef, np.array(new[] { 0.0, 0, 0, 0, 0, 1 })), t.Name);
                Assert.AreEqual(5L, p.degree(), t.Name);
            }
        }

        /// <summary>
        ///     NumPy's <c>test_fromroots</c>: the requested roots are zeros of a series of the right degree, domain and window,
        ///     and the series is monic in the power basis.
        /// </summary>
        [TestMethod]
        public void FromRoots_RootsAreZeros_AndThePowerSeriesIsMonic()
        {
            var r = np.array(new[] { 0.11, 0.42, 0.57, 0.73, 0.98 });
            foreach (var t in Classes)
            {
                var d = Shifted(ClassDomain(t), 0.1, 0.2);
                var w = Shifted(ClassWindow(t), 0.05, 0.15);
                using var p1 = FromRoots(t, r, d, w);
                Assert.AreEqual(5L, p1.degree(), t.Name);
                Assert.IsTrue(np.array_equal(p1.domain, d) && np.array_equal(p1.window, w), t.Name);
                Assert.IsTrue(np.allclose(p1.Call(r), np.zeros(new Shape(5)), 0, 1e-9), $"{t.Name}: {p1.Call(r)}");
                using var p2 = NDPolynomial.cast(p1, NDPolynomial.default_domain, NDPolynomial.default_window);
                Assert.AreEqual(1.0, p2.coef.GetDouble(5), 1e-9, t.Name);
            }
        }

        /// <summary>
        ///     <c>fromroots</c>' domain argument: omitted or an empty list is the class default (NumPy's <c>domain=[]</c>), null
        ///     the roots' own interval (NumPy's None: <c>getdomain(roots)</c>).
        /// </summary>
        [TestMethod]
        public void FromRoots_DomainOmittedEmptyOrNone()
        {
            using var omitted = NDPolynomial.fromroots(L(1, 3));
            using var emptyList = NDPolynomial.fromroots(L(1, 3), L());
            using var none = NDPolynomial.fromroots(L(1, 3), null);
            Assert.IsTrue(ReferenceEquals(NDPolynomial.default_domain, NDPolynomial.default_domain));
            Assert.IsTrue(np.array_equal(omitted.domain, np.array(new[] { -1.0, 1 })));
            Assert.IsTrue(np.array_equal(emptyList.domain, np.array(new[] { -1.0, 1 })));
            Assert.IsTrue(np.array_equal(none.domain, np.array(new[] { 1.0, 3 })));
            // NumPy: Polynomial.fromroots([1, 3], domain=None) is Polynomial([-1., 0., 1.], domain=[1., 3.]): the roots map
            // to -1 and 1 of the window, so the window variable's polynomial is (t + 1)(t - 1).
            Assert.IsTrue(np.array_equal(none.coef, np.array(new[] { -1.0, 0, 1 })), none.ToString());
            Assert.AreEqual("-1.0 + 0.0 (-2.0 + x) + 1.0 (-2.0 + x)**2", none.__format__("ascii"));
        }

        /// <summary>
        ///     NumPy's <c>test_fit</c>: the default domain covers x, a given domain / window and a degree list are kept, an
        ///     empty domain list is the class default, and weights select the points. Needs LAPACK (lstsq).
        /// </summary>
        [TestMethod]
        public void Fit_Domains_DegreeLists_Weights()
        {
            RequireLapack();
            var x = np.linspace(0, 3, 50);
            var y = x * (x - 1.0) * (x - 2.0);
            foreach (var t in Classes)
            {
                using var p = Fit(t, x, y, 3);
                Assert.IsTrue(np.allclose(p.domain, np.array(new[] { 0.0, 3 })), t.Name);
                Assert.IsTrue(np.allclose(p.Call(x), y), t.Name);
                Assert.AreEqual(3L, p.degree(), t.Name);

                var d = Shifted(ClassDomain(t), 0.1, 0.2);
                var w = Shifted(ClassWindow(t), 0.05, 0.15);
                using var pd = Fit(t, x, y, 3, d, window: w);
                Assert.IsTrue(np.allclose(pd.Call(x), y) && np.allclose(pd.domain, d) && np.allclose(pd.window, w), t.Name);
                using var pl = Fit(t, x, y, L(0, 1, 2, 3), d, window: w);
                Assert.IsTrue(np.allclose(pl.Call(x), y), t.Name);

                using var pc = Fit(t, x, y, 3, L());
                Assert.IsTrue(np.array_equal(pc.domain, ClassDomain(t)) && np.array_equal(pc.window, ClassWindow(t)), t.Name);

                var wt = np.zeros_like(x);
                wt["::2"] = 1.0;
                var z = y + np.linspace(0, 0.25, 50);
                using var p1 = Fit(t, x["::2"], z["::2"], 3);
                using var p2 = Fit(t, x, z, 3, w: wt);
                using var p3 = Fit(t, x, z, L(0, 1, 2, 3), w: wt);
                Assert.IsTrue(np.allclose(p1.Call(x), p2.Call(x)) && np.allclose(p2.Call(x), p3.Call(x)), t.Name);
            }
        }

        /// <summary>
        ///     <c>fit(..., full: true)</c> deconstructs into the series and lstsq's statistics (NumPy's
        ///     <c>series, [resid, rank, sv, rcond] = Poly.fit(..., full=True)</c>); a series that is not a full fit refuses to.
        /// </summary>
        [TestMethod]
        public void Fit_Full_DeconstructsIntoSeriesAndStatistics()
        {
            RequireLapack();
            var x = np.linspace(0, 1, 7);
            var y = x * x + 1.0;
            var (s, (residuals, rank, sv, rcond)) = NDChebyshev.fit(x, y, 2, full: true);
            Assert.AreEqual(3L, s.__len__());
            Assert.AreEqual(3L, Convert.ToInt64(rank.GetAtIndex(0)));
            Assert.AreEqual(3L, sv.size);
            Assert.AreEqual(7 * 2.220446049250313e-16, rcond.GetDouble(0), 1e-30);
            Assert.AreEqual(1L, residuals.size);
            using var plain = new NDChebyshev(L(1, 2));
            Assert.ThrowsException<ValueError>(() => { var (_, _) = plain; });
        }

        // ---------------------------------------------------------------------------------------------
        //  test_classes.py: equality and arithmetic
        // ---------------------------------------------------------------------------------------------

        /// <summary>NumPy's <c>test_equal</c> / <c>test_not_equal</c>: coefficients, domain and window all count.</summary>
        [TestMethod]
        public void Equality_CoefficientsDomainAndWindowCount()
        {
            foreach (var t in Classes)
            {
                using var p1 = New(t, L(1, 2, 3), L(0, 1), L(2, 3));
                using var p2 = New(t, L(1, 1, 1), L(0, 1), L(2, 3));
                using var p3 = New(t, L(1, 2, 3), L(1, 2), L(2, 3));
                using var p4 = New(t, L(1, 2, 3), L(0, 1), L(1, 2));
                Assert.IsTrue(p1 == p1 && !(p1 == p2) && !(p1 == p3) && !(p1 == p4), t.Name);
                Assert.IsTrue(!(p1 != p1) && p1 != p2 && p1 != p3 && p1 != p4, t.Name);
                Assert.IsTrue(p1.__eq__(p1) && p1.__ne__(p2), t.Name);
            }
        }

        /// <summary>
        ///     Equality with a value that is not a series is false in either direction, as in NumPy (the other side's
        ///     <c>__eq__</c> declines, the series' answers False). An NDArray needs <c>(object)</c> in C#: <c>p == arr</c> is
        ///     ambiguous between the series' and NDArray's <c>==</c>.
        /// </summary>
        [TestMethod]
        public void Equality_WithValuesThatAreNotSeries_IsFalse()
        {
            using var p = new NDPolynomial(L(1, 2, 3));
            var arr = np.array(new[] { 1.0, 2, 3 });
            Assert.IsFalse(p == (object)arr);
            Assert.IsFalse(p.__eq__(arr));
            Assert.IsTrue(p != (object)arr);
            Assert.IsFalse(p == 1);
            Assert.IsFalse(p == (object)null);
            Assert.IsFalse(p == "x");
            using var c = new NDChebyshev(L(1, 2, 3));
            Assert.IsFalse(p == c, "a series of another class is never equal");
            using var f32 = new NDPolynomial(np.array(new float[] { 1, 2, 3 }));
            Assert.IsTrue(p == f32, "a float32 series equals a float64 one with the same values (np.all(==))");
        }

        /// <summary>NumPy's <c>test_add</c> / <c>test_sub</c> / <c>test_mul</c>: lists, tuples and arrays on either side.</summary>
        [TestMethod]
        public void AddSubMul_CommuteOverListsTuplesAndArrays()
        {
            double[] c1 = { 0.9, 1.3, 0.6, 1.1 }, c2 = { 0.7, 1.4, 0.8 };
            foreach (var t in Classes)
            {
                using var p1 = New(t, c1);
                using var p2 = New(t, c2);
                object list2 = L(0.7, 1.4, 0.8);
                var tuple2 = (0.7, 1.4, 0.8);
                var arr2 = np.array(c2);
                using (var p3 = (NDPolyBase)((dynamic)p1 + p2))
                {
                    AssertPolyAlmostEqual((NDPolyBase)((dynamic)p2 + p1), p3, $"{t.Name} p2+p1");
                    AssertPolyAlmostEqual((NDPolyBase)((dynamic)p1 + list2), p3, $"{t.Name} p1+list");
                    AssertPolyAlmostEqual((NDPolyBase)((dynamic)list2 + p1), p3, $"{t.Name} list+p1");
                    AssertPolyAlmostEqual((NDPolyBase)((dynamic)p1 + tuple2), p3, $"{t.Name} p1+tuple");
                    AssertPolyAlmostEqual((NDPolyBase)((dynamic)tuple2 + p1), p3, $"{t.Name} tuple+p1");
                    AssertPolyAlmostEqual((NDPolyBase)((dynamic)p1 + arr2), p3, $"{t.Name} p1+arr");
                    AssertPolyAlmostEqual((NDPolyBase)((dynamic)arr2 + p1), p3, $"{t.Name} arr+p1");
                }
                using (var p3 = (NDPolyBase)((dynamic)p1 - p2))
                {
                    using var neg = (NDPolyBase)(-(dynamic)p3);
                    AssertPolyAlmostEqual((NDPolyBase)((dynamic)p2 - p1), neg, $"{t.Name} p2-p1");
                    AssertPolyAlmostEqual((NDPolyBase)((dynamic)list2 - p1), neg, $"{t.Name} list-p1");
                    AssertPolyAlmostEqual((NDPolyBase)((dynamic)arr2 - p1), neg, $"{t.Name} arr-p1");
                    AssertPolyAlmostEqual((NDPolyBase)((dynamic)p1 - tuple2), p3, $"{t.Name} p1-tuple");
                }
                using (var p3 = (NDPolyBase)((dynamic)p1 * p2))
                {
                    AssertPolyAlmostEqual((NDPolyBase)((dynamic)p2 * p1), p3, $"{t.Name} p2*p1");
                    AssertPolyAlmostEqual((NDPolyBase)((dynamic)list2 * p1), p3, $"{t.Name} list*p1");
                    AssertPolyAlmostEqual((NDPolyBase)((dynamic)p1 * arr2), p3, $"{t.Name} p1*arr");
                    using var two = New(t, L(2));
                    AssertPolyAlmostEqual((NDPolyBase)((dynamic)p1 * 2), (NDPolyBase)((dynamic)p1 * two), $"{t.Name} p1*2");
                    AssertPolyAlmostEqual((NDPolyBase)(2 * (dynamic)p1), (NDPolyBase)((dynamic)p1 * two), $"{t.Name} 2*p1");
                }
            }
        }

        /// <summary>
        ///     NumPy's <c>TypeError</c>s of <c>_get_coefficients</c>: another domain, another window, another class; and
        ///     <c>ValueError</c> for another symbol (test_symbol.py's <c>TestBinaryOperatorsDifferentSymbol</c>).
        /// </summary>
        [TestMethod]
        public void BinaryOperators_RefuseOtherDomainWindowClassOrSymbol()
        {
            foreach (var t in Classes)
            {
                using var p = New(t, L(1, 2, 3));
                using var otherDomain = New(t, L(0), ClassDomain(t) + 1.0);
                using var otherWindow = New(t, L(0), null, ClassWindow(t) + 1.0);
                using var otherClass = t == typeof(NDPolynomial) ? (NDPolyBase)new NDChebyshev(L(0)) : new NDPolynomial(L(0));
                using var otherSymbol = New(t, L(4, 5, 6), symbol: "y");
                Func<NDPolyBase, object>[] ops =
                {
                    q => (dynamic)p + q, q => (dynamic)p - q, q => (dynamic)p * q, q => ((dynamic)p).__floordiv__(q),
                    q => (dynamic)p % q, q => ((dynamic)p).__divmod__(q),
                };
                foreach (var op in ops)
                {
                    Assert.AreEqual("Domains differ", Assert.ThrowsException<TypeError>(() => op(otherDomain)).Message, t.Name);
                    Assert.AreEqual("Windows differ", Assert.ThrowsException<TypeError>(() => op(otherWindow)).Message, t.Name);
                    Assert.AreEqual("Polynomial types differ", Assert.ThrowsException<TypeError>(() => op(otherClass)).Message, t.Name);
                    Assert.AreEqual("Polynomial symbols differ", Assert.ThrowsException<ValueError>(() => op(otherSymbol)).Message, t.Name);
                }
            }
        }

        /// <summary>NumPy's <c>test_floordiv</c> / <c>test_mod</c> / <c>test_divmod</c>: <c>p4 = p1*p2 + p3</c> divides back.</summary>
        [TestMethod]
        public void FloorDivModDivMod_RecoverQuotientAndRemainder()
        {
            foreach (var t in Classes)
            {
                using var p1 = New(t, L(0.9, 1.3, 0.6, 1.1));
                using var p2 = New(t, L(0.7, 1.4, 0.8));
                using var p3 = New(t, L(0.55, 1.2));
                using var p4 = (NDPolyBase)((dynamic)((dynamic)p1 * p2) + p3);
                object c2 = L(0.7, 1.4, 0.8);
                var c4 = (object[])Array.ConvertAll(p4.coef.ToArray<double>(), v => (object)v);
                AssertPolyAlmostEqual(p4.__floordiv__(p2), p1, $"{t.Name} p4//p2");
                AssertPolyAlmostEqual(p4.__floordiv__(c2), p1, $"{t.Name} p4//list");
                AssertPolyAlmostEqual(p2.__rfloordiv__(c4), p1, $"{t.Name} list//p2");
                AssertPolyAlmostEqual((NDPolyBase)((dynamic)p4 % p2), p3, $"{t.Name} p4%p2");
                AssertPolyAlmostEqual((NDPolyBase)((dynamic)c4 % p2), p3, $"{t.Name} list%p2");
                var (quo, rem) = p4.__divmod__(p2);
                AssertPolyAlmostEqual(quo, p1, $"{t.Name} divmod quo");
                AssertPolyAlmostEqual(rem, p3, $"{t.Name} divmod rem");
                var (q2, r2) = p2.__rdivmod__(2);
                AssertPolyAlmostEqual(q2, New(t, L(0)), $"{t.Name} divmod(2, p2) quo");
                AssertPolyAlmostEqual(r2, New(t, L(2)), $"{t.Name} divmod(2, p2) rem");
                AssertPolyAlmostEqual(p2.__floordiv__(2), (NDPolyBase)(0.5 * (dynamic)p2), $"{t.Name} p2//2");
            }
        }

        /// <summary>
        ///     NumPy's <c>test_truediv</c>: <c>p / s</c> for every number (not a bool), a <c>TypeError</c> for <c>s / p</c>, for
        ///     non-numbers (<c>()</c>, <c>[]</c>, False, an array) and for a series divisor.
        /// </summary>
        [TestMethod]
        public void TrueDiv_DividesByNumbersOnly()
        {
            foreach (var t in Classes)
            {
                using var p1 = New(t, L(1, 2, 3));
                using var p2 = (NDPolyBase)((dynamic)p1 * 5);
                foreach (object s in new object[] { 5, 5L, 5.0, 5f, new Complex(5, 0), (Half)5, (char)5, (byte)5, (short)5, 5u, 5UL })
                {
                    AssertPolyAlmostEqual((NDPolyBase)((dynamic)p2 / s), p1, $"{t.Name} / {s.GetType().Name}");
                    // both operands dynamic: a statically typed object operand would bind as object, not as its runtime type
                    Assert.ThrowsException<TypeError>(() => (object)((dynamic)s / (dynamic)p2), $"{t.Name} {s.GetType().Name} / p");
                }
                foreach (object s in new object[] { ValueTuple.Create(), L(), false, np.array(new[] { 1 }) })
                    Assert.ThrowsException<TypeError>(() => (object)((dynamic)p2 / s), $"{t.Name} p / {s}");
                foreach (var t2 in Classes)
                    Assert.ThrowsException<TypeError>(() => (object)((dynamic)p2 / New(t2, L(1))), $"{t.Name} p / {t2.Name}");
                Assert.AreEqual($"unsupported types for true division: '{ClassRepr(t)}', '<class 'bool'>'",
                    Assert.ThrowsException<TypeError>(() => (object)((dynamic)p2 / true)).Message);
            }
        }

        /// <summary><c>str(type(p))</c> of a class: <c>&lt;class 'numpy.polynomial.chebyshev.Chebyshev'&gt;</c>.</summary>
        /// <param name="t">The class.</param>
        /// <returns>The text.</returns>
        private static string ClassRepr(Type t)
        {
            string name = t.Name.Substring(2);
            string module = name switch
            {
                "Polynomial" => "polynomial",
                "Chebyshev" => "chebyshev",
                "Legendre" => "legendre",
                "Laguerre" => "laguerre",
                "Hermite" => "hermite",
                _ => "hermite_e",
            };
            return $"<class 'numpy.polynomial.{module}.{name}'>";
        }

        /// <summary>
        ///     NumPy's <c>test_pow</c>: <c>p ** i</c> is the repeated product (domain and window kept); 1.5 and -1 are
        ///     <c>ValueError</c>s; NumPy's default maxpower 100 refuses 101.
        /// </summary>
        [TestMethod]
        public void Pow_RepeatedProduct_RefusesInvalidPowers()
        {
            foreach (var t in Classes)
            {
                var d = Shifted(ClassDomain(t), 0.1, 0.2);
                var w = Shifted(ClassWindow(t), 0.05, 0.15);
                NDPolyBase tgt = New(t, L(1), d, w);
                using var tst = New(t, L(1, 2, 3), d, w);
                for (int i = 0; i < 5; i++)
                {
                    AssertPolyAlmostEqual(tst.__pow__(i), tgt, $"{t.Name} ** {i}");
                    tgt = (NDPolyBase)((dynamic)tgt * tst);
                }
                Assert.AreEqual("Power must be a non-negative integer.", Assert.ThrowsException<ValueError>(() => tgt.__pow__(1.5)).Message);
                Assert.AreEqual("Power must be a non-negative integer.", Assert.ThrowsException<ValueError>(() => tgt.__pow__(-1)).Message);
                Assert.AreEqual("Power is too large", Assert.ThrowsException<ValueError>(() => tst.__pow__(101)).Message);
            }
        }

        /// <summary>
        ///     <c>maxpower</c>: the class default (100, NumPy's <c>ABCPolyBase.maxpower</c>), an instance value (any Python number:
        ///     2.5 allows 2), null for no limit.
        /// </summary>
        [TestMethod]
        public void Maxpower_InstanceValueOverridesTheClassDefault()
        {
            using var p = new NDPolynomial(L(1, 1));
            Assert.AreEqual(100L, p.maxpower);
            p.maxpower = 2.5;
            using var squared = p.__pow__(2);
            Assert.AreEqual(3L, squared.__len__());
            Assert.ThrowsException<ValueError>(() => p.__pow__(3));
            p.maxpower = null;
            using var big = p.__pow__(150);
            Assert.AreEqual(151L, big.__len__());
            using var copy = p.copy();
            Assert.AreEqual(100L, copy.maxpower, "a derived series reads the class default: NumPy's constructor sets no maxpower");
        }

        // ---------------------------------------------------------------------------------------------
        //  test_classes.py: members
        // ---------------------------------------------------------------------------------------------

        /// <summary>NumPy's <c>test_roots</c>: the roots of <c>fromroots(tgt)</c> are tgt (domain mapped back). Needs LAPACK.</summary>
        [TestMethod]
        public void Roots_OfFromRoots_AreTheRoots()
        {
            RequireLapack();
            foreach (var t in Classes)
            {
                var d = ClassDomain(t) * 1.25 + 0.25;
                var tgt = np.linspace(d.GetDouble(0), d.GetDouble(1), 5);
                using var p = FromRoots(t, tgt, d, ClassWindow(t));
                Assert.IsTrue(np.allclose(np.sort(p.roots()), tgt), $"{t.Name}: {p.roots()}");
                using var q = FromRoots(t, tgt, System.Reflection.Missing.Value, null);
                Assert.IsTrue(np.allclose(np.sort(q.roots()), tgt), t.Name);
            }
        }

        /// <summary>
        ///     The complex roots of a float32 series are NumPy's complex64 values (eigvals of the float32 companion), and a float32
        ///     domain maps them back in NumPy's COMPLEX64 loops — NumPy's float32 roundings, carried in complex128 (#569).
        /// </summary>
        [TestMethod]
        public void Roots_Float32SeriesFloat32Domain_MapInComplex64()
        {
            RequireLapack();
            using var p = new NDPolynomial(np.array(new float[] { 1, -3, 2.5f }), np.array(new float[] { 0, 2 }), np.array(new float[] { -1, 1 }));
            var r = p.roots();
            // NumPy 2.4.2: [1.6-0.19999994j, 1.6+0.19999994j], complex64 bits 3fcccccd be4cccc9 / 3fcccccd 3e4cccc9.
            var bits = new[] { 0x3fcccccdu, 0xbe4cccc9u, 0x3fcccccdu, 0x3e4cccc9u };
            for (int i = 0; i < 2; i++)
            {
                var z = (Complex)r.GetAtIndex(i);
                Assert.AreEqual(bits[2 * i], BitConverter.SingleToUInt32Bits((float)z.Real), $"root {i} real");
                Assert.AreEqual(bits[2 * i + 1], BitConverter.SingleToUInt32Bits((float)z.Imaginary), $"root {i} imag");
                Assert.AreEqual((double)(float)z.Real, z.Real, "a complex64 value carried exactly");
            }
        }

        /// <summary>NumPy's <c>test_copy</c>: an equal series with its own coefficient, domain and window arrays.</summary>
        [TestMethod]
        public void Copy_IsEqualButOwnsItsArrays()
        {
            foreach (var t in Classes)
            {
                using var p1 = Basis(t, 5);
                using var p2 = p1.copy();
                Assert.IsTrue(p1 == p2, t.Name);
                Assert.IsFalse(ReferenceEquals(p1, p2), t.Name);
                Assert.IsFalse(ReferenceEquals(p1.coef, p2.coef), t.Name);
                Assert.IsFalse(ReferenceEquals(p1.domain, p2.domain), t.Name);
                Assert.IsFalse(ReferenceEquals(p1.window, p2.window), t.Name);
                Assert.IsFalse(np.shares_memory(p1.coef, p2.coef), t.Name);
            }
        }

        /// <summary>
        ///     NumPy's object identities: <c>+p is p</c>; a getter returns the stored array; series built without a domain share
        ///     the class default; assigning the class default reaches them.
        /// </summary>
        [TestMethod]
        public void ObjectIdentity_PosGettersAndClassDefaults()
        {
            using var p = new NDLegendre(L(1, 2));
            using var q = new NDLegendre(L(3));
            Assert.IsTrue(ReferenceEquals(+p, p));
            Assert.IsTrue(ReferenceEquals(p.coef, p.coef));
            Assert.IsTrue(ReferenceEquals(p.domain, q.domain), "both read the class default");
            Assert.IsTrue(ReferenceEquals(p.domain, NDLegendre.default_domain));
            var before = NDLegendre.default_domain;
            try
            {
                NDLegendre.default_domain = np.array(new[] { 0.0, 4.0 });
                Assert.IsTrue(np.array_equal(p.domain, np.array(new[] { 0.0, 4.0 })), "an instance without a domain reads the new class attribute");
                using var r = new NDLegendre(L(1), L(5, 6));
                Assert.IsTrue(np.array_equal(r.domain, np.array(new[] { 5.0, 6.0 })), "an instance domain is its own");
            }
            finally
            {
                NDLegendre.default_domain = before;
            }
        }

        /// <summary>NumPy's <c>test_integ</c>: defaults, constants, the lower bound, and the domain scaling.</summary>
        [TestMethod]
        public void Integ_DefaultsConstantsLowerBoundAndScaling()
        {
            foreach (var t in Classes)
            {
                using var src = new NDPolynomial(L(2, 6, 12));
                using var p0 = Cast(t, src);
                AssertPolyAlmostEqual(NDPolynomial.cast(p0.integ()), new NDPolynomial(L(0, 2, 3, 4)), $"{t.Name} integ()");
                AssertPolyAlmostEqual(NDPolynomial.cast(p0.integ(2)), new NDPolynomial(L(0, 0, 1, 1, 1)), $"{t.Name} integ(2)");
                AssertPolyAlmostEqual(NDPolynomial.cast(p0.integ(k: 1)), new NDPolynomial(L(1, 2, 3, 4)), $"{t.Name} k=1");
                AssertPolyAlmostEqual(NDPolynomial.cast(p0.integ(2, L(1, 1))), new NDPolynomial(L(1, 1, 1, 1, 1)), $"{t.Name} k=[1,1]");
                AssertPolyAlmostEqual(NDPolynomial.cast(p0.integ(lbnd: 1)), new NDPolynomial(L(-9, 2, 3, 4)), $"{t.Name} lbnd=1");
                AssertPolyAlmostEqual(NDPolynomial.cast(p0.integ(2, lbnd: 1)), new NDPolynomial(L(6, -9, 1, 1, 1)), $"{t.Name} 2 lbnd=1");
                using var ps = Cast(t, src, ClassDomain(t) * 2.0);
                AssertPolyAlmostEqual(NDPolynomial.cast(ps.integ()), new NDPolynomial(L(0, 2, 3, 4)), $"{t.Name} scaled integ()");
                AssertPolyAlmostEqual(NDPolynomial.cast(ps.integ(2)), new NDPolynomial(L(0, 0, 1, 1, 1)), $"{t.Name} scaled integ(2)");
            }
        }

        /// <summary>
        ///     A str or None integration constant raises NumPy's own TypeError (the str has no <c>__sub__</c>, so the NumPy
        ///     value's reflected subtract finds no ufunc loop; None reaches the object loop's CPython subtraction) — and so does
        ///     the one-coefficient zero branch's <c>c[0] += k</c>.
        /// </summary>
        [TestMethod]
        public void Integ_StrOrNoneConstant_RaisesNumPysTypeError()
        {
            using var p = new NDPolynomial(L(1, 2));
            Assert.AreEqual("ufunc 'subtract' did not contain a loop with signature matching types (dtype('<U1'), dtype('float64')) -> None",
                Assert.ThrowsException<TypeError>(() => p.integ(2, L("a"))).Message);
            Assert.AreEqual("unsupported operand type(s) for -: 'NoneType' and 'float'",
                Assert.ThrowsException<TypeError>(() => p.integ(2, L(new object[] { null }))).Message);
            using var zero = new NDPolynomial(L(0));
            Assert.AreEqual("ufunc 'add' did not contain a loop with signature matching types (dtype('float64'), dtype('<U2')) -> None",
                Assert.ThrowsException<TypeError>(() => zero.integ(1, L("ab"))).Message);
            Assert.AreEqual("unsupported operand type(s) for +: 'float' and 'NoneType'",
                Assert.ThrowsException<TypeError>(() => zero.integ(1, L(new object[] { null }))).Message);
        }

        /// <summary>NumPy's <c>test_deriv</c>: differentiation inverts integration, with and without a mapped domain.</summary>
        [TestMethod]
        public void Deriv_InvertsInteg()
        {
            foreach (var t in Classes)
            {
                foreach (bool mapped in new[] { true, false })
                {
                    using var p1 = mapped
                        ? New(t, L(1, 2, 3), Shifted(ClassDomain(t), 0.1, 0.2), Shifted(ClassWindow(t), 0.05, 0.15))
                        : New(t, L(1, 2, 3));
                    using var p2 = p1.integ(2, L(1, 2));
                    using var p3 = p1.integ(1, L(1));
                    Assert.IsTrue(np.allclose(p2.deriv(1).coef, p3.coef), t.Name);
                    Assert.IsTrue(np.allclose(p2.deriv(2).coef, p1.coef), t.Name);
                }
            }
        }

        /// <summary>NumPy's <c>test_linspace</c>: n points of the domain (or a given interval) and the series' values there.</summary>
        [TestMethod]
        public void Linspace_SamplesTheDomainOrAGivenInterval()
        {
            foreach (var t in Classes)
            {
                var d = Shifted(ClassDomain(t), 0.1, 0.2);
                using var p = New(t, L(1, 2, 3), d, Shifted(ClassWindow(t), 0.05, 0.15));
                var xtgt = np.linspace(d.GetDouble(0), d.GetDouble(1), 20);
                var (xres, yres) = p.linspace(20);
                Assert.IsTrue(np.allclose(xres, xtgt) && np.allclose(yres, p.Call(xtgt)), t.Name);
                var (x2, y2) = p.linspace(20, L(0, 2));
                Assert.IsTrue(np.allclose(x2, np.linspace(0, 2, 20)) && np.allclose(y2, p.Call(np.linspace(0, 2, 20))), t.Name);
            }
        }

        /// <summary>NumPy's <c>test_call</c> / <c>test_call_with_list</c>: the series' values, a Python list of points too.</summary>
        [TestMethod]
        public void Call_ValuesAtArraysListsAndScalars()
        {
            foreach (var t in Classes)
            {
                var d = ClassDomain(t);
                var x = np.linspace(d.GetDouble(0), d.GetDouble(1), 11);
                using var src = new NDPolynomial(L(1, 2, 3));
                using var p = Cast(t, src);
                var tgt = 1.0 + x * (2.0 + 3.0 * x);
                Assert.IsTrue(np.allclose(p.Call(x), tgt), t.Name);
                using var q = New(t, L(1, 2, 3));
                Assert.IsTrue(np.array_equal(q.Call(L(-1, 0, 2)), q.Call(np.array(new[] { -1, 0, 2 }))), t.Name);
                // A Python scalar point is a NumPy scalar result (a 0-d array here).
                Assert.AreEqual(0, q.Call(0.5).ndim, t.Name);
            }
        }

        /// <summary>
        ///     <c>p(q)</c> at a series is a series of q's class with q's symbol (test_symbol.py's <c>test_composition</c>): C#
        ///     binds the generic <see cref="NDPolyBase.Call{TSeries}(TSeries)"/> for a series argument.
        /// </summary>
        [TestMethod]
        public void Call_AtASeries_IsACompositionOfTheArgumentsClass()
        {
            using var p = new NDPolynomial(L(3, 2, 1), symbol: "t");
            using var q = new NDPolynomial(L(5, 1, 0, -1), symbol: "λ_1");
            NDPolynomial r = p.Call(q);
            Assert.AreEqual("λ_1", r.symbol);
            using var c = new NDChebyshev(L(0, 1));
            NDChebyshev rc = p.Call(c);
            Assert.AreEqual(typeof(NDChebyshev), rc.GetType());
            var x = np.linspace(-1, 1, 7);
            Assert.IsTrue(np.allclose(rc.Call(x), p.Call(x)), "Chebyshev's T_1 is x, so p(T_1) evaluates as p");
            Assert.ThrowsException<ArgumentException>(() => p.Call((object)c), "a series through object: the generic Call names its result type");
        }

        /// <summary>NumPy's <c>test_cutdeg</c> / <c>test_truncate</c> / <c>test_trim</c>.</summary>
        [TestMethod]
        public void CutdegTruncateTrim()
        {
            foreach (var t in Classes)
            {
                using var p = New(t, L(1, 2, 3));
                Assert.ThrowsException<ValueError>(() => p.cutdeg(.5));
                Assert.ThrowsException<ValueError>(() => p.cutdeg(-1));
                Assert.AreEqual(3L, p.cutdeg(3).__len__());
                Assert.AreEqual(3L, p.cutdeg(2).__len__());
                Assert.AreEqual(2L, p.cutdeg(1).__len__());
                Assert.AreEqual(1L, p.cutdeg(0).__len__());
                Assert.ThrowsException<ValueError>(() => p.truncate(.5));
                Assert.ThrowsException<ValueError>(() => p.truncate(0));
                Assert.AreEqual(3L, p.truncate(4).__len__());
                Assert.AreEqual(2L, p.truncate(2).__len__());

                using var tz = New(t, L(1, 1e-6, 1e-12, 0));
                Assert.IsTrue(np.array_equal(tz.trim().coef, np.array(new[] { 1, 1e-6, 1e-12 })), t.Name);
                Assert.IsTrue(np.array_equal(tz.trim(1e-10).coef, np.array(new[] { 1, 1e-6 })), t.Name);
                Assert.IsTrue(np.array_equal(tz.trim(1e-5).coef, np.array(new[] { 1.0 })), t.Name);
                // trim(tol=None) is not the default: trimcoef's `tol < 0` refuses None, as in NumPy.
                Assert.AreEqual("'<' not supported between instances of 'NoneType' and 'int'",
                    Assert.ThrowsException<TypeError>(() => tz.trim(null)).Message);
            }
        }

        /// <summary>
        ///     NumPy complex scalars in <c>truncate</c> / <c>cutdeg</c> / <c>basis</c>: int() takes the real part (NumPy warns that
        ///     the imaginary part is discarded) and <c>!=</c> compares that int with the complex — so 2+0j works and 2+1j is
        ///     "not an integer"; an explicit 0-d complex ARRAY is refused by int().
        /// </summary>
        [TestMethod]
        public void IntegerArguments_NumPyComplexScalars()
        {
            using var p = new NDPolynomial(L(1, 2, 3));
            var c2 = NDArray.Scalar(new Complex(2, 0));      // an unmarked 0-d array: a NumPy scalar
            var c2i = NDArray.Scalar(new Complex(2, 1));
            Assert.AreEqual(2L, p.truncate(c2).__len__());
            Assert.ThrowsException<ValueError>(() => p.truncate(c2i));
            Assert.AreEqual(2L, p.cutdeg(NDArray.Scalar(new Complex(1, 0))).__len__());
            using var b = NDPolynomial.basis(c2);
            Assert.AreEqual(3L, b.__len__());
            Assert.AreEqual("int() argument must be a string, a bytes-like object or a real number, not 'complex'",
                Assert.ThrowsException<TypeError>(() => p.truncate(np.array(new Complex(2, 0)))).Message);
        }

        /// <summary>NumPy's <c>test_mapparms</c>: the identity map for equal domain and window, then <c>[1, 2]</c> for <c>w = 2d + 1</c>.</summary>
        [TestMethod]
        public void Mapparms_IdentityAndAffine()
        {
            foreach (var t in Classes)
            {
                var d = ClassDomain(t);
                using var p = New(t, L(1), d, d);
                var (off, scl) = p.mapparms();
                Assert.AreEqual(0.0, (double)off, 1e-15, t.Name);
                Assert.AreEqual(1.0, (double)scl, 1e-15, t.Name);
                using var q = New(t, L(1), d, d * 2.0 + 1.0);
                var (off2, scl2) = q.mapparms();
                Assert.AreEqual(1.0, (double)off2, 1e-15, t.Name);
                Assert.AreEqual(2.0, (double)scl2, 1e-15, t.Name);
            }
        }

        /// <summary>NumPy's <c>TestInterpolate</c>: argument errors, the degree, and the approximation of powers on a domain.</summary>
        [TestMethod]
        public void Interpolate_DegreesAndApproximation()
        {
            static object F(NDArray x, object[] _) => x * (x - 1.0) * (x - 2.0);
            Assert.ThrowsException<ValueError>(() => NDChebyshev.interpolate(F, -1));
            Assert.ThrowsException<TypeError>(() => NDChebyshev.interpolate(F, 10.0));
            for (int deg = 1; deg < 5; deg++)
                Assert.AreEqual((long)deg, NDChebyshev.interpolate(F, deg).degree());
            static object Powx(NDArray x, object[] a) => np.power(x, Convert.ToInt32(a[0]));
            var x = np.linspace(0, 2, 10);
            for (int deg = 0; deg < 10; deg++)
                for (int t = 0; t <= deg; t++)
                {
                    using var p = NDChebyshev.interpolate(Powx, deg, L(0, 2), L(t));
                    // NumPy's assert_almost_equal(decimal=11): |difference| < 1.5e-11.
                    Assert.IsTrue(np.allclose(p.Call(x), np.power(x, t), 0, 1.5e-11), $"deg {deg} t {t}");
                }
        }

        // ---------------------------------------------------------------------------------------------
        //  test_symbol.py
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     <c>TestInit</c>: the default symbol, NumPy's refusals (<c>''</c> and <c>'3'</c> ValueError, None TypeError) and the
        ///     valid ones (Python's identifier rule over code points: <c>β</c>, a symbol outside the BMP).
        /// </summary>
        [TestMethod]
        public void Symbol_DefaultValidAndInvalid()
        {
            using var p = new NDPolynomial(L(1, 2, 3));
            Assert.AreEqual("x", p.symbol);
            Assert.AreEqual("Symbol string must be a valid Python identifier",
                Assert.ThrowsException<ValueError>(() => new NDPolynomial(L(1, 2, 3), symbol: "")).Message);
            Assert.ThrowsException<ValueError>(() => new NDPolynomial(L(1, 2, 3), symbol: "3"));
            Assert.AreEqual("Symbol must be a non-empty string",
                Assert.ThrowsException<TypeError>(() => new NDPolynomial(L(1, 2, 3), symbol: null)).Message);
            foreach (var s in new[] { "x", "x_1", "A", "xyz", "β", "if", "\U0001d465", "ℕ" })
                Assert.AreEqual(s, new NDPolynomial(L(1), symbol: s).symbol, s);
            foreach (var s in new[] { "x y", "a-b", "́x", "x$" })
                Assert.ThrowsException<ValueError>(() => new NDPolynomial(L(1), symbol: s), s);
        }

        /// <summary>
        ///     The symbol travels through every operation that makes a series (<c>TestUnaryOperators</c>,
        ///     <c>TestBinaryOperatorsSameSymbol</c>, <c>TestExtraMethods</c>), and through the class methods' symbol argument.
        /// </summary>
        [TestMethod]
        public void Symbol_IsPreservedByOperationsAndClassMethods()
        {
            using var p = new NDPolynomial(L(1, 2, 3, 0), symbol: "z");
            using var rhs = new NDPolynomial(L(4, 5, 6), symbol: "z");
            var arr = np.array(new[] { 4, 5, 6 });
            var results = new List<NDPolyBase>
            {
                -p, p * 10, 10 * p, p.__pow__(3), p + rhs, p - rhs, p * rhs, p + arr, arr + p, arr - p, arr * p,
                p.__divmod__(rhs).quo, p.__divmod__(rhs).rem, p.__rdivmod__(arr).quo, p.copy(), p.trim(), p.truncate(2),
                p.convert(L(-10, 10)), p.convert(window: L(-10, 10)), p.convert(kind: typeof(NDChebyshev)), p.integ(), p.deriv(),
                NDPolynomial.fromroots(L(-2, 2), symbol: "z"), NDPolynomial.identity(L(-1, 1), L(5, 20), "z"),
                NDPolynomial.basis(3, symbol: "z"),
            };
            foreach (var r in results)
                Assert.AreEqual("z", r.symbol, r.ToString());
        }

        /// <summary><c>TestEquality</c>: the symbol takes part in <c>==</c>.</summary>
        [TestMethod]
        public void Symbol_TakesPartInEquality()
        {
            using var p = new NDPolynomial(L(1, 2, 3));
            Assert.IsTrue(p == new NDPolynomial(L(1, 2, 3), symbol: "x"));
            Assert.IsFalse(p == new NDPolynomial(L(1, 2, 3), symbol: "y"));
        }

        // ---------------------------------------------------------------------------------------------
        //  The C# spelling of the protocol
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     Which overload a left operand binds, and that each gives NumPy's answer: a C# number converts to a marked
        ///     literal NDArray (a Python number), Half / char are NumPy scalars, a typed array an ndarray, <c>object[]</c> a list,
        ///     a tuple a tuple, a string a str (CPython's concatenation refusal).
        /// </summary>
        [TestMethod]
        public void LeftOperands_BindTheirPythonMeaning()
        {
            using var p = new NDPolynomial(L(1, 2));
            Assert.IsTrue(np.array_equal((2.0 + p).coef, np.array(new[] { 3.0, 2 })));
            Assert.IsTrue(np.array_equal((2L + p).coef, np.array(new[] { 3.0, 2 })));
            Assert.IsTrue(np.array_equal((new Complex(0, 1) + p).coef, np.array(new[] { new Complex(1, 1), new Complex(2, 0) })));
            Assert.IsTrue(np.array_equal(((Half)2 + p).coef, np.array(new[] { 3.0, 2 })));
            Assert.IsTrue(np.array_equal((new[] { 1.0, 1.0 } + p).coef, np.array(new[] { 2.0, 3 })));
            Assert.IsTrue(np.array_equal((new object[] { 1, 1 } + p).coef, np.array(new[] { 2.0, 3 })));
            Assert.IsTrue(np.array_equal(((1, 1) + p).coef, np.array(new[] { 2.0, 3 })));
            Assert.AreEqual("can only concatenate str (not \"Polynomial\") to str", Assert.ThrowsException<TypeError>(() => "ab" + p).Message);
            Assert.AreEqual("can't multiply sequence by non-int of type 'Polynomial'", Assert.ThrowsException<TypeError>(() => "ab" * p).Message);
            Assert.AreEqual("unsupported operand type(s) for /: 'float' and 'Polynomial'", Assert.ThrowsException<TypeError>(() => 2.0 / p).Message);
            Assert.AreEqual("operand 'Polynomial' does not support ufuncs (__array_ufunc__=None)",
                Assert.ThrowsException<TypeError>(() => p + np.array(new[,] { { 1.0, 2.0 } })).Message);
        }

        /// <summary>NumPy's <c>__hash__ = None</c>: a series is unhashable (its value can change in place).</summary>
        [TestMethod]
        public void GetHashCode_IsUnhashable()
        {
            using var p = new NDHermite(L(1));
            Assert.AreEqual("unhashable type: 'Hermite'", Assert.ThrowsException<TypeError>(() => p.GetHashCode()).Message);
        }

        /// <summary>
        ///     Disposal releases the series' own arrays (idempotent); every member then refuses, and a class default is never
        ///     released.
        /// </summary>
        [TestMethod]
        public void Dispose_ReleasesOwnArrays_ThenRefusesUse()
        {
            var p = new NDLaguerre(L(1, 2));
            var def = NDLaguerre.default_domain;
            p.Dispose();
            p.Dispose();
            Assert.ThrowsException<ObjectDisposedException>(() => p.coef);
            Assert.ThrowsException<ObjectDisposedException>(() => p.Call(0.5));
            Assert.AreEqual(0.0, def.GetDouble(0), "the class default is the class's, not the series'");
        }

        /// <summary>
        ///     NumPy's pickling pair: <c>__getstate__</c> returns copies in <c>__dict__</c>'s key order (the instance attributes in
        ///     assignment order, then the missing domain / window, then <c>symbol</c>), and <c>__setstate__</c> restores a series
        ///     — a state without <c>_symbol</c> makes reading the symbol NumPy's AttributeError.
        /// </summary>
        [TestMethod]
        public void Pickling_StateOrderAndRoundTrip()
        {
            using var p = new NDPolynomial(np.array(new float[] { 1, 2 }));
            Assert.AreEqual("coef,_symbol,domain,window,symbol", string.Join(",", Keys(p.__getstate__())));
            using var r = new NDPolynomial(L(1, 2), window: L(0, 1));
            r.maxpower = 3;
            r.domain = np.array(new[] { 1.0, 2.0 });
            OrderedDictionary state = r.__getstate__();
            Assert.AreEqual("coef,window,_symbol,maxpower,domain,symbol", string.Join(",", Keys(state)));
            Assert.IsFalse(ReferenceEquals(state["coef"], r.coef), "copies");

            using var target = new NDPolynomial(L(9, 9, 9), L(5, 6));
            target.__setstate__(state);
            Assert.IsTrue(target == r);
            Assert.AreEqual(3L, Convert.ToInt64(target.maxpower));

            var partial = new OrderedDictionary { ["coef"] = np.array(new[] { 1.0, 2.0 }) };
            target.__setstate__(partial);
            Assert.AreEqual(2L, target.__len__());
            Assert.AreEqual("'Polynomial' object has no attribute '_symbol'", Assert.ThrowsException<AttributeError>(() => target.symbol).Message);
            Assert.IsTrue(np.array_equal(target.domain, NDPolynomial.default_domain), "a missing domain falls back to the class attribute");
        }

        /// <summary>
        ///     The C# lifetime rules of the attributes: assigning one releases the array it replaces (as <c>poly1d</c> releases
        ///     replaced coefficients; NumPy's reference count would keep a held one alive, C# code copies it first), assigning
        ///     the current array back changes nothing (NumPy's <c>p.coef = p.coef</c> keeps the object), <c>__setstate__</c>
        ///     releases what it replaces too, and the class attributes survive a caller's <c>Dispose</c>.
        /// </summary>
        [TestMethod]
        public void Assignment_ReleasesTheReplacedArray_AndKeepsTheCurrentOne()
        {
            using var p = new NDPolynomial(L(1, 2, 3), L(0, 2));
            var coef = p.coef;
            p.coef = p.coef;
            Assert.IsTrue(ReferenceEquals(coef, p.coef) && !coef.IsDisposed, "the current array assigned back is kept");
            using var c = np.array(new[] { 4.0, 5.0 });
            p.coef = c;
            Assert.IsTrue(coef.IsDisposed, "the replaced coefficients are released");
            Assert.IsFalse(c.IsDisposed, "the assigned array stays the caller's (the series holds a view)");
            Assert.IsTrue(np.array_equal(p.coef, c));

            var domain = p.domain;
            using var d = np.array(new[] { -3.0, 3.0 });
            p.domain = d;
            Assert.IsTrue(domain.IsDisposed, "the instance's replaced domain is released");

            // A default-window series: the window it reads is the class attribute, which an assignment never releases.
            var window = p.window;
            Assert.IsTrue(ReferenceEquals(window, NDPolynomial.default_window));
            using var w = np.array(new[] { 0.0, 1.0 });
            p.window = w;
            Assert.IsFalse(window.IsDisposed, "the class default is the class's");

            using var q = new NDPolynomial(L(7, 8));
            var qCoef = q.coef;
            OrderedDictionary state = p.__getstate__();
            q.__setstate__(state);
            Assert.IsTrue(qCoef.IsDisposed, "__setstate__ releases the coefficients it replaces");
            Assert.IsTrue(q == p);
            foreach (System.Collections.DictionaryEntry e in state)
                (e.Value as NDArray)?.Dispose();

            // A caller disposing the shared class attribute (reached through any default-domain series) cannot free it.
            using (var r = new NDPolynomial(L(1, 2)))
                r.domain.Dispose();
            using var s = new NDPolynomial(L(1, 2));
            Assert.IsTrue(np.array_equal(s.domain, np.array(new[] { -1.0, 1.0 })), "the class attribute outlives a caller's Dispose");
        }

        /// <summary>
        ///     The forward operators with a C# array or a string operand bind overloads of their own: <c>p + new[] { 1.0, 2.0 }</c>
        ///     is NumPy's <c>p + [1, 2]</c> and <c>p + "ab"</c> NumPy's TypeError, with no call-site <see cref="NDArray"/>
        ///     conversion between them and the protocol (NumPy 2.4.2's texts).
        /// </summary>
        [TestMethod]
        public void Operators_CSharpArrayAndStringOperands()
        {
            using var p = new NDChebyshev(L(1, 2, 3));
            using var viaArray = p + new[] { 1.0, 2.0 };
            using var viaList = p + L(1.0, 2.0);
            Assert.IsTrue(viaArray == viaList);
            Assert.IsTrue(np.array_equal(viaArray.coef, np.array(new[] { 2.0, 4.0, 3.0 })));
            using var product = p * new[] { 0.0, 1.0 };
            Assert.IsTrue(np.array_equal(product.coef, np.array(new[] { 1.0, 2.5, 1.0, 1.5 })));
            Assert.AreEqual("unsupported operand type(s) for +: 'Chebyshev' and 'str'",
                Assert.ThrowsException<TypeError>(() => p + "ab").Message);
            Assert.AreEqual("unsupported types for true division: '<class 'numpy.polynomial.chebyshev.Chebyshev'>', '<class 'str'>'",
                Assert.ThrowsException<TypeError>(() => p / "ab").Message);
            Assert.AreEqual("unsupported operand type(s) for %: 'Chebyshev' and 'str'",
                Assert.ThrowsException<TypeError>(() => p % "ab").Message);
        }

        /// <summary>The keys of a state dictionary, in order.</summary>
        /// <param name="state">The state.</param>
        /// <returns>The keys.</returns>
        private static IEnumerable<string> Keys(OrderedDictionary state)
        {
            foreach (var k in state.Keys)
                yield return (string)k;
        }
    }
}
