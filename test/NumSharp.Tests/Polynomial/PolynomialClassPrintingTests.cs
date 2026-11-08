using System;
using System.Numerics;

namespace NumSharp.Tests.Polynomial
{
    /// <summary>
    ///     The series classes' printers (plan <c>docs/plans/numpy-polynomial-remaining-compute.md</c> §3.8, U11's class half):
    ///     <c>str</c> in both styles, <c>format</c>, <c>repr</c> and <c>_repr_latex_</c>, NumPy 2.4.2's <c>test_printing.py</c>
    ///     ported (the class part; its object-array cases are NumPy object dtypes NumSharp does not have), with C#'s own
    ///     spellings: <c>ToString()</c>, <c>string.Format</c> / interpolation through <see cref="IFormattable"/>.
    /// </summary>
    /// <remarks>
    ///     <c>[DoNotParallelize]</c>: <c>set_default_printstyle</c> is one process-wide flag (NumPy's
    ///     <c>ABCPolyBase._use_unicode</c>); every test puts NumPy's default back. Print options are scoped per call
    ///     (<c>np.printoptions</c>). The LaTeX expectations are NumPy's real output (NumPy's own test replaces
    ///     <c>_repr_latex_scalar</c> with <c>str</c>, which C# cannot do), printed by NumPy 2.4.2 for the same series.
    /// </remarks>
    [TestClass]
    [DoNotParallelize]
    public class PolynomialClassPrintingTests
    {
        /// <summary>NumPy's default print style: ASCII on Windows, Unicode elsewhere.</summary>
        private static readonly string DefaultStyle = OperatingSystem.IsWindows() ? "ascii" : "unicode";

        /// <summary>Puts NumPy's default print style back after every test.</summary>
        [TestCleanup]
        public void Cleanup() => np.polynomial.set_default_printstyle(DefaultStyle);

        /// <summary>A Python list (the house map: <c>object[]</c>).</summary>
        /// <param name="items">The items.</param>
        /// <returns>The list.</returns>
        private static object[] L(params object[] items) => items;

        /// <summary>The six classes with <c>Poly(coef, symbol=symbol)</c> constructors.</summary>
        private static readonly (string Name, Func<object, string, NDPolyBase> New)[] Classes =
        {
            ("Polynomial", (c, s) => new NDPolynomial(c, symbol: s)),
            ("Chebyshev", (c, s) => new NDChebyshev(c, symbol: s)),
            ("Legendre", (c, s) => new NDLegendre(c, symbol: s)),
            ("Laguerre", (c, s) => new NDLaguerre(c, symbol: s)),
            ("Hermite", (c, s) => new NDHermite(c, symbol: s)),
            ("HermiteE", (c, s) => new NDHermiteE(c, symbol: s)),
        };

        /// <summary>
        ///     <c>TestStrUnicodeSuperSubscripts</c>: each class with <c>[1, 2, 3]</c>, <c>[-1, 0, 3, -1]</c> and
        ///     <c>arange(12)</c> (which wraps at the default linewidth 75).
        /// </summary>
        [TestMethod]
        public void Str_Unicode_EveryClass()
        {
            np.polynomial.set_default_printstyle("unicode");
            var expected = new (string Name, string A, string B, string C)[]
            {
                ("Polynomial", "1.0 + 2.0·x + 3.0·x²", "-1.0 + 0.0·x + 3.0·x² - 1.0·x³",
                 "0.0 + 1.0·x + 2.0·x² + 3.0·x³ + 4.0·x⁴ + 5.0·x⁵ + 6.0·x⁶ + 7.0·x⁷ +\n8.0·x⁸ + 9.0·x⁹ + 10.0·x¹⁰ + 11.0·x¹¹"),
                ("Chebyshev", "1.0 + 2.0·T₁(x) + 3.0·T₂(x)", "-1.0 + 0.0·T₁(x) + 3.0·T₂(x) - 1.0·T₃(x)",
                 "0.0 + 1.0·T₁(x) + 2.0·T₂(x) + 3.0·T₃(x) + 4.0·T₄(x) + 5.0·T₅(x) +\n6.0·T₆(x) + 7.0·T₇(x) + 8.0·T₈(x) + 9.0·T₉(x) + 10.0·T₁₀(x) + 11.0·T₁₁(x)"),
                ("Legendre", "1.0 + 2.0·P₁(x) + 3.0·P₂(x)", "-1.0 + 0.0·P₁(x) + 3.0·P₂(x) - 1.0·P₃(x)",
                 "0.0 + 1.0·P₁(x) + 2.0·P₂(x) + 3.0·P₃(x) + 4.0·P₄(x) + 5.0·P₅(x) +\n6.0·P₆(x) + 7.0·P₇(x) + 8.0·P₈(x) + 9.0·P₉(x) + 10.0·P₁₀(x) + 11.0·P₁₁(x)"),
                ("Laguerre", "1.0 + 2.0·L₁(x) + 3.0·L₂(x)", "-1.0 + 0.0·L₁(x) + 3.0·L₂(x) - 1.0·L₃(x)",
                 "0.0 + 1.0·L₁(x) + 2.0·L₂(x) + 3.0·L₃(x) + 4.0·L₄(x) + 5.0·L₅(x) +\n6.0·L₆(x) + 7.0·L₇(x) + 8.0·L₈(x) + 9.0·L₉(x) + 10.0·L₁₀(x) + 11.0·L₁₁(x)"),
                ("Hermite", "1.0 + 2.0·H₁(x) + 3.0·H₂(x)", "-1.0 + 0.0·H₁(x) + 3.0·H₂(x) - 1.0·H₃(x)",
                 "0.0 + 1.0·H₁(x) + 2.0·H₂(x) + 3.0·H₃(x) + 4.0·H₄(x) + 5.0·H₅(x) +\n6.0·H₆(x) + 7.0·H₇(x) + 8.0·H₈(x) + 9.0·H₉(x) + 10.0·H₁₀(x) + 11.0·H₁₁(x)"),
                ("HermiteE", "1.0 + 2.0·He₁(x) + 3.0·He₂(x)", "-1.0 + 0.0·He₁(x) + 3.0·He₂(x) - 1.0·He₃(x)",
                 "0.0 + 1.0·He₁(x) + 2.0·He₂(x) + 3.0·He₃(x) + 4.0·He₄(x) + 5.0·He₅(x) +\n6.0·He₆(x) + 7.0·He₇(x) + 8.0·He₈(x) + 9.0·He₉(x) + 10.0·He₁₀(x) +\n11.0·He₁₁(x)"),
            };
            for (int i = 0; i < Classes.Length; i++)
            {
                var (name, make) = Classes[i];
                Assert.AreEqual(expected[i].A, make(L(1, 2, 3), "x").ToString(), name);
                Assert.AreEqual(expected[i].B, make(L(-1, 0, 3, -1), "x").ToString(), name);
                Assert.AreEqual(expected[i].C, make(np.arange(12), "x").ToString(), name);
            }
        }

        /// <summary><c>TestStrAscii</c>: the ASCII printers (<c>x**2</c>, <c>T_1(x)</c>) and their wrapping.</summary>
        [TestMethod]
        public void Str_Ascii_EveryClass()
        {
            np.polynomial.set_default_printstyle("ascii");
            var expected = new (string Name, string A, string B, string C)[]
            {
                ("Polynomial", "1.0 + 2.0 x + 3.0 x**2", "-1.0 + 0.0 x + 3.0 x**2 - 1.0 x**3",
                 "0.0 + 1.0 x + 2.0 x**2 + 3.0 x**3 + 4.0 x**4 + 5.0 x**5 + 6.0 x**6 +\n7.0 x**7 + 8.0 x**8 + 9.0 x**9 + 10.0 x**10 + 11.0 x**11"),
                ("Chebyshev", "1.0 + 2.0 T_1(x) + 3.0 T_2(x)", "-1.0 + 0.0 T_1(x) + 3.0 T_2(x) - 1.0 T_3(x)",
                 "0.0 + 1.0 T_1(x) + 2.0 T_2(x) + 3.0 T_3(x) + 4.0 T_4(x) + 5.0 T_5(x) +\n6.0 T_6(x) + 7.0 T_7(x) + 8.0 T_8(x) + 9.0 T_9(x) + 10.0 T_10(x) +\n11.0 T_11(x)"),
                ("Legendre", "1.0 + 2.0 P_1(x) + 3.0 P_2(x)", "-1.0 + 0.0 P_1(x) + 3.0 P_2(x) - 1.0 P_3(x)",
                 "0.0 + 1.0 P_1(x) + 2.0 P_2(x) + 3.0 P_3(x) + 4.0 P_4(x) + 5.0 P_5(x) +\n6.0 P_6(x) + 7.0 P_7(x) + 8.0 P_8(x) + 9.0 P_9(x) + 10.0 P_10(x) +\n11.0 P_11(x)"),
                ("Laguerre", "1.0 + 2.0 L_1(x) + 3.0 L_2(x)", "-1.0 + 0.0 L_1(x) + 3.0 L_2(x) - 1.0 L_3(x)",
                 "0.0 + 1.0 L_1(x) + 2.0 L_2(x) + 3.0 L_3(x) + 4.0 L_4(x) + 5.0 L_5(x) +\n6.0 L_6(x) + 7.0 L_7(x) + 8.0 L_8(x) + 9.0 L_9(x) + 10.0 L_10(x) +\n11.0 L_11(x)"),
                ("Hermite", "1.0 + 2.0 H_1(x) + 3.0 H_2(x)", "-1.0 + 0.0 H_1(x) + 3.0 H_2(x) - 1.0 H_3(x)",
                 "0.0 + 1.0 H_1(x) + 2.0 H_2(x) + 3.0 H_3(x) + 4.0 H_4(x) + 5.0 H_5(x) +\n6.0 H_6(x) + 7.0 H_7(x) + 8.0 H_8(x) + 9.0 H_9(x) + 10.0 H_10(x) +\n11.0 H_11(x)"),
                ("HermiteE", "1.0 + 2.0 He_1(x) + 3.0 He_2(x)", "-1.0 + 0.0 He_1(x) + 3.0 He_2(x) - 1.0 He_3(x)",
                 "0.0 + 1.0 He_1(x) + 2.0 He_2(x) + 3.0 He_3(x) + 4.0 He_4(x) +\n5.0 He_5(x) + 6.0 He_6(x) + 7.0 He_7(x) + 8.0 He_8(x) + 9.0 He_9(x) +\n10.0 He_10(x) + 11.0 He_11(x)"),
            };
            for (int i = 0; i < Classes.Length; i++)
            {
                var (name, make) = Classes[i];
                Assert.AreEqual(expected[i].A, make(L(1, 2, 3), "x").__str__(), name);
                Assert.AreEqual(expected[i].B, make(L(-1, 0, 3, -1), "x").__str__(), name);
                Assert.AreEqual(expected[i].C, make(np.arange(12), "x").__str__(), name);
            }
        }

        /// <summary><c>test_polynomial_str_domains</c>: a mapped domain prints the scaled symbol in parentheses, in both styles.</summary>
        [TestMethod]
        public void Str_MappedDomain_PrintsTheScaledSymbol()
        {
            np.polynomial.set_default_printstyle("unicode");
            Assert.AreEqual("0.0 + 1.0·x", new NDPolynomial(L(0, 1)).ToString());
            Assert.AreEqual("0.0 + 1.0·(-3.0 + 2.0x)", new NDPolynomial(L(0, 1), L(1, 2)).ToString());
            np.polynomial.set_default_printstyle("ascii");
            Assert.AreEqual("0.0 + 1.0 x", new NDPolynomial(L(0, 1)).ToString());
            Assert.AreEqual("0.0 + 1.0 (-3.0 + 2.0x)", new NDPolynomial(L(0, 1), L(1, 2)).ToString());
        }

        /// <summary>
        ///     <c>TestLinebreaking</c>: a line breaks before a term that would reach the linewidth (75 by default), so one
        ///     less than the linewidth fits; <c>np.printoptions(linewidth=...)</c> moves the break.
        /// </summary>
        [TestMethod]
        public void Str_Linebreaking()
        {
            np.polynomial.set_default_printstyle("ascii");
            var oneLess = new NDPolynomial(L(12345678, 12345678, 12345678, 12345678, 123)).ToString();
            Assert.AreEqual(74, oneLess.Length);
            Assert.AreEqual("12345678.0 + 12345678.0 x + 12345678.0 x**2 + 12345678.0 x**3 + 123.0 x**4", oneLess);
            var onWidth = new NDPolynomial(L(12345678, 12345678, 12345678, 12345678, 1234)).ToString();
            Assert.AreEqual(75, onWidth.Length);
            Assert.AreEqual("12345678.0 + 12345678.0 x + 12345678.0 x**2 + 12345678.0 x**3 +\n1234.0 x**4", onWidth);
            Assert.AreEqual("12345678.0 + 12345678.0 x + 12345678.0 x**2 + 12345678.0 x**3 + 1.0 x**4 +\n12345678.0 x**5",
                new NDPolynomial(L(12345678, 12345678, 12345678, 12345678, 1, 12345678)).ToString());
            Assert.AreEqual("12345678.0 + 12345678.0 x + 12345678.0 x**2 + 12345678.12 x**3 +\n1.0 x**4 + 12345678.0 x**5",
                new NDPolynomial(L(12345678, 12345678, 12345678, 12345678.12, 1, 12345678)).ToString());

            var p = new NDPolynomial(L(0, 10, 200, 3000, 40000, 500000, 600000, 70000, 8000, 900));
            var widths = new (int Lw, string Tgt)[]
            {
                (75, "0.0 + 10.0 x + 200.0 x**2 + 3000.0 x**3 + 40000.0 x**4 + 500000.0 x**5 +\n600000.0 x**6 + 70000.0 x**7 + 8000.0 x**8 + 900.0 x**9"),
                (45, "0.0 + 10.0 x + 200.0 x**2 + 3000.0 x**3 +\n40000.0 x**4 + 500000.0 x**5 +\n600000.0 x**6 + 70000.0 x**7 + 8000.0 x**8 +\n900.0 x**9"),
                (132, "0.0 + 10.0 x + 200.0 x**2 + 3000.0 x**3 + 40000.0 x**4 + 500000.0 x**5 + 600000.0 x**6 + 70000.0 x**7 + 8000.0 x**8 + 900.0 x**9"),
            };
            foreach (var (lw, tgt) in widths)
            {
                using (np.printoptions(linewidth: lw))
                {
                    string s = p.ToString();
                    Assert.AreEqual(tgt, s, $"linewidth {lw}");
                    foreach (var line in s.Split('\n'))
                        Assert.IsTrue(line.Length < lw, $"linewidth {lw}: '{line}'");
                }
            }
        }

        /// <summary><c>test_set_default_printoptions</c>: the style switches every class's <c>str</c>; an unknown style is a ValueError.</summary>
        [TestMethod]
        public void SetDefaultPrintstyle_SwitchesStr()
        {
            var p = new NDPolynomial(L(1, 2, 3));
            var c = new NDChebyshev(L(1, 2, 3));
            np.polynomial.set_default_printstyle("ascii");
            Assert.AreEqual("1.0 + 2.0 x + 3.0 x**2", p.ToString());
            Assert.AreEqual("1.0 + 2.0 T_1(x) + 3.0 T_2(x)", c.ToString());
            np.polynomial.set_default_printstyle("unicode");
            Assert.AreEqual("1.0 + 2.0·x + 3.0·x²", p.ToString());
            Assert.AreEqual("1.0 + 2.0·T₁(x) + 3.0·T₂(x)", c.ToString());
            Assert.ThrowsException<ValueError>(() => np.polynomial.set_default_printstyle("invalid_input"));
        }

        /// <summary>
        ///     <c>test_complex_coefficients</c> (NumPy complex): a complex coefficient prints as Python's <c>str</c>, and the
        ///     sign is taken from NumPy's lexicographic <c>&gt;= 0</c>.
        /// </summary>
        [TestMethod]
        public void Str_ComplexCoefficients()
        {
            var p = new NDPolynomial(L(new Complex(0, 1), new Complex(1, 1), new Complex(-2, 2), new Complex(3, 0)));
            np.polynomial.set_default_printstyle("unicode");
            Assert.AreEqual("1j + (1+1j)·x - (2-2j)·x² + (3+0j)·x³", p.ToString());
            np.polynomial.set_default_printstyle("ascii");
            Assert.AreEqual("1j + (1+1j) x - (2-2j) x**2 + (3+0j) x**3", p.ToString());
        }

        /// <summary>
        ///     <c>TestFormat</c>: <c>'unicode'</c> / <c>'ascii'</c> force a printer, an empty format is <c>str</c>, any other
        ///     format is NumPy's ValueError — through <c>__format__</c>, <c>string.Format</c> and interpolation alike.
        /// </summary>
        [TestMethod]
        public void Format_UnicodeAsciiEmptyAndBad()
        {
            np.polynomial.set_default_printstyle("ascii");
            var p = new NDPolynomial(L(1, 2, 0, -1));
            Assert.AreEqual("1.0 + 2.0·x + 0.0·x² - 1.0·x³", p.__format__("unicode"));
            Assert.AreEqual("1.0 + 2.0·x + 0.0·x² - 1.0·x³", $"{p:unicode}");
            np.polynomial.set_default_printstyle("unicode");
            Assert.AreEqual("1.0 + 2.0 x + 0.0 x**2 - 1.0 x**3", p.__format__("ascii"));
            Assert.AreEqual("1.0 + 2.0 x + 0.0 x**2 - 1.0 x**3", string.Format("{0:ascii}", p));
            np.polynomial.set_default_printstyle("ascii");
            var q = new NDPolynomial(L(1, 2, 3));
            Assert.AreEqual("1.0 + 2.0 x + 3.0 x**2", q.__format__(""));
            Assert.AreEqual("1.0 + 2.0 x + 3.0 x**2", $"{q}");
            Assert.AreEqual("Unsupported format string '.2f' passed to <class 'numpy.polynomial.polynomial.Polynomial'>.__format__. " +
                            "Valid options are 'ascii' and 'unicode'",
                Assert.ThrowsException<ValueError>(() => p.__format__(".2f")).Message);
            Assert.ThrowsException<ValueError>(() => $"{p:.2f}");
        }

        /// <summary><c>test_symbol</c>: the symbol replaces x in every class's Unicode printer.</summary>
        [TestMethod]
        public void Format_Symbol_EveryClass()
        {
            var expected = new[]
            {
                "1.0 + 2.0·z + 3.0·z²", "1.0 + 2.0·T₁(z) + 3.0·T₂(z)", "1.0 + 2.0·P₁(z) + 3.0·P₂(z)", "1.0 + 2.0·L₁(z) + 3.0·L₂(z)",
                "1.0 + 2.0·H₁(z) + 3.0·H₂(z)", "1.0 + 2.0·He₁(z) + 3.0·He₂(z)",
            };
            for (int i = 0; i < Classes.Length; i++)
                Assert.AreEqual(expected[i], $"{Classes[i].New(L(1, 2, 3), "z"):unicode}", Classes[i].Name);
        }

        /// <summary><c>TestRepr</c>: every class's <c>repr</c> (Laguerre's default domain and window are [0, 1]).</summary>
        [TestMethod]
        public void Repr_EveryClass()
        {
            foreach (var (name, make) in Classes)
            {
                string dw = name == "Laguerre" ? "[0., 1.]" : "[-1.,  1.]";
                Assert.AreEqual($"{name}([0., 1.], domain={dw}, window={dw}, symbol='x')", make(L(0, 1), "x").__repr__(), name);
            }
            Assert.AreEqual("Polynomial([1., 2.], domain=[0., 2.], window=[-1.,  1.], symbol='t')",
                new NDPolynomial(L(1, 2), L(0, 2), symbol: "t").__repr__());
        }

        /// <summary>
        ///     <c>TestLatexRepr</c> with NumPy's real scalar formatting: default, translated, scaled and affine arguments of the
        ///     power series (<c>\left( \right)</c>), the basis printers (no surplus parentheses), a multi-character basis name,
        ///     the symbol, a muted zero term, and a complex coefficient (parenthesized).
        /// </summary>
        [TestMethod]
        public void ReprLatex_NumPysOutput()
        {
            Assert.AreEqual(@"$x \mapsto \text{1.0} + \text{2.0}\,x + \text{3.0}\,x^{2}$", new NDPolynomial(L(1, 2, 3))._repr_latex_());
            Assert.AreEqual(@"$x \mapsto \text{1.0} + \text{2.0}\,\left(\text{1.0} + x\right) + \text{3.0}\,\left(\text{1.0} + x\right)^{2}$",
                new NDPolynomial(L(1, 2, 3), L(-2, 0))._repr_latex_());
            Assert.AreEqual(@"$x \mapsto \text{1.0} + \text{2.0}\,\left(\text{2.0}x\right) + \text{3.0}\,\left(\text{2.0}x\right)^{2}$",
                new NDPolynomial(L(1, 2, 3), L(-0.5, 0.5))._repr_latex_());
            Assert.AreEqual(@"$x \mapsto \text{1.0} + \text{2.0}\,\left(\text{1.0} + \text{2.0}x\right) + \text{3.0}\,\left(\text{1.0} + \text{2.0}x\right)^{2}$",
                new NDPolynomial(L(1, 2, 3), L(-1, 0))._repr_latex_());
            Assert.AreEqual(@"$x \mapsto \text{1.0}\,{T}_{0}(x) + \text{2.0}\,{T}_{1}(x) + \text{3.0}\,{T}_{2}(x)$", new NDChebyshev(L(1, 2, 3))._repr_latex_());
            Assert.AreEqual(@"$x \mapsto \text{1.0}\,{T}_{0}(\text{1.0} + \text{2.0}x) + \text{2.0}\,{T}_{1}(\text{1.0} + \text{2.0}x) + \text{3.0}\,{T}_{2}(\text{1.0} + \text{2.0}x)$",
                new NDChebyshev(L(1, 2, 3), L(-1, 0))._repr_latex_());
            Assert.AreEqual(@"$x \mapsto \text{1.0}\,{He}_{0}(x) + \text{2.0}\,{He}_{1}(x) + \text{3.0}\,{He}_{2}(x)$", new NDHermiteE(L(1, 2, 3))._repr_latex_());
            Assert.AreEqual(@"$z \mapsto \text{1.0} + \text{2.0}\,z + \text{3.0}\,z^{2}$", new NDPolynomial(L(1, 2, 3), symbol: "z")._repr_latex_());
            Assert.AreEqual(@"$x \mapsto \color{LightGray}{\text{0.0}} + \text{1.0}\,x\color{LightGray}{ + \text{0.0}\,x^{2}} - \text{2.5}\,x^{3}$",
                new NDPolynomial(L(0, 1, 0, -2.5))._repr_latex_());
            Assert.AreEqual(@"$x \mapsto \text{1j} + (\text{(2+0j)})\,x$", new NDPolynomial(L(new Complex(0, 1), 2))._repr_latex_());
        }

        /// <summary>
        ///     <c>TestPrintOptions</c>: <c>str</c> and <c>_repr_latex_</c> under <c>precision</c>, <c>floatmode='fixed'</c>, the
        ///     switch to scientific notation as the precision grows, and <c>nanstr</c> / <c>infstr</c>.
        /// </summary>
        [TestMethod]
        public void PrintOptions_StrAndLatex()
        {
            np.polynomial.set_default_printstyle("ascii");
            var p = new NDPolynomial(L(1.0 / 2, 1.0 / 7, 1.0 / 7 * 1e8, 1.0 / 7 * 1e9));
            Assert.AreEqual("0.5 + 0.14285714 x + 14285714.28571429 x**2 + (1.42857143e+08) x**3", p.ToString());
            Assert.AreEqual(@"$x \mapsto \text{0.5} + \text{0.14285714}\,x + \text{14285714.28571429}\,x^{2} + \text{(1.42857143e+08)}\,x^{3}$", p._repr_latex_());
            using (np.printoptions(precision: 3))
            {
                Assert.AreEqual("0.5 + 0.143 x + 14285714.286 x**2 + (1.429e+08) x**3", p.ToString());
                Assert.AreEqual(@"$x \mapsto \text{0.5} + \text{0.143}\,x + \text{14285714.286}\,x^{2} + \text{(1.429e+08)}\,x^{3}$", p._repr_latex_());
            }

            var half = new NDPolynomial(L(0.5));
            Assert.AreEqual("0.5", half.ToString());
            using (np.printoptions(floatmode: "fixed"))
                Assert.AreEqual("0.50000000", half.ToString());
            using (np.printoptions(floatmode: "fixed", precision: 4))
                Assert.AreEqual("0.5000", half.ToString());

            string[] switchToExp =
            {
                "1.0 + (1.0e-01) x + (1.0e-02) x**2",
                "1.2 + (1.2e-01) x + (1.2e-02) x**2",
                "1.23 + 0.12 x + (1.23e-02) x**2 + (1.23e-03) x**3",
                "1.235 + 0.123 x + (1.235e-02) x**2 + (1.235e-03) x**3",
                "1.2346 + 0.1235 x + 0.0123 x**2 + (1.2346e-03) x**3 + (1.2346e-04) x**4",
                "1.23457 + 0.12346 x + 0.01235 x**2 + (1.23457e-03) x**3 + (1.23457e-04) x**4",
                "1.234568 + 0.123457 x + 0.012346 x**2 + 0.001235 x**3 + (1.234568e-04) x**4 + (1.234568e-05) x**5",
                "1.2345679 + 0.1234568 x + 0.0123457 x**2 + 0.0012346 x**3 + (1.2345679e-04) x**4 + (1.2345679e-05) x**5",
            };
            for (int i = 0; i < switchToExp.Length; i++)
            {
                var coef = new object[i / 2 + 3];
                for (int k = 0; k < coef.Length; k++)
                    coef[k] = 1.23456789 * System.Math.Pow(10, -k);
                using (np.printoptions(precision: i))
                    Assert.AreEqual(switchToExp[i], new NDPolynomial(coef).ToString().Replace('\n', ' '), $"precision {i}");
            }

            var nonFinite = new NDPolynomial(L(double.NaN, double.PositiveInfinity));
            Assert.AreEqual("nan + inf x", nonFinite.ToString());
            Assert.AreEqual(@"$x \mapsto \text{nan} + \text{inf}\,x$", nonFinite._repr_latex_());
            using (np.printoptions(nanstr: "NAN", infstr: "INF"))
            {
                Assert.AreEqual("NAN + INF x", nonFinite.ToString());
                Assert.AreEqual(@"$x \mapsto \text{NAN} + \text{INF}\,x$", nonFinite._repr_latex_());
            }
        }
    }
}
