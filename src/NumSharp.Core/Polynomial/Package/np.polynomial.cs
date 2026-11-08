namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     The <c>numpy.polynomial</c> package: power, Chebyshev, Legendre, Laguerre and both Hermite series,
        ///     each as a submodule with NumPy's names — <c>var C = np.polynomial.chebyshev;</c> is the port of
        ///     <c>from numpy.polynomial import chebyshev as C</c>.
        /// </summary>
        /// <remarks>
        ///     <para><b>Coefficient order:</b> every function of this package takes its coefficients LOW degree
        ///     first (<c>c[0] + c[1]*x + …</c>). The legacy <see cref="np.polyval(NDArray, NDArray)"/> family takes
        ///     them HIGH degree first and its arguments in the other order — the two are different functions that
        ///     happen to share names.</para>
        ///     <para>https://numpy.org/doc/stable/reference/routines.polynomials-package.html</para>
        /// </remarks>
        public static PolynomialModule polynomial { get; } = new PolynomialModule();
    }

    /// <summary>
    ///     The <c>numpy.polynomial</c> package facade, reachable as <see cref="np.polynomial"/>. Its members are
    ///     the six basis submodules; each is a first-class object, so Python's <c>import … as P</c> becomes
    ///     <c>var P = np.polynomial.polynomial;</c>. (A nested static class cannot model this: C# forbids a member
    ///     named <c>polynomial</c> inside a type of that name.)
    /// </summary>
    /// <remarks>
    ///     <para>Implemented so far, bit-identical to NumPy 2.4.2 for every dtype NumPy has (values, dtypes, result
    ///     layouts and error texts; the one dtype divergence is NumSharp's single complex width, #569):</para>
    ///     <list type="bullet">
    ///         <item>the evaluation family of every basis — <c>{p}val</c>, <c>{p}val2d</c>, <c>{p}val3d</c>,
    ///         <c>{p}grid2d</c>, <c>{p}grid3d</c> and <c>{p}valnd</c> — through one runtime-compiled IL kernel per
    ///         (basis, dtypes, layout) class, the result laid out as NumPy's last ufunc call lays it out (an
    ///         F-ordered or permuted x or series keeps its memory order);</item>
    ///         <item>the additive family of every basis — the constants <c>{p}domain</c>/<c>{p}zero</c>/<c>{p}one</c>/
    ///         <c>{p}x</c>, <c>{p}line</c>, <c>{p}add</c>, <c>{p}sub</c>, <c>{p}trim</c> — and the <see cref="polyutils"/>
    ///         substrate (<c>as_series</c>, <c>trimseq</c>, <c>trimcoef</c>, <c>getdomain</c>, <c>mapparms</c>,
    ///         <c>mapdomain</c>);</item>
    ///         <item>the calculus family of every basis — <c>{p}der</c> and <c>{p}int</c> with every parameter
    ///         (<c>m</c>, <c>k</c>, <c>lbnd</c>, <c>scl</c>, <c>axis</c>) — through one runtime-compiled IL kernel per
    ///         (basis, direction, dtypes) that runs every order in place in one buffer;</item>
    ///         <item>the series algebra of every basis — <c>{p}mulx</c>, <c>{p}mul</c>, <c>{p}div</c>, <c>{p}pow</c>,
    ///         <c>{p}fromroots</c> and the <c>X2poly</c>/<c>poly2X</c> conversions — NumPy's Python statements replayed
    ///         over a pooled per-thread arena, each statement on the house kernel NumPy's statement runs through
    ///         (long products byte-exact with <c>NumSharp.Interop.OpenBLAS</c>, bounded-ULP without it);</item>
    ///         <item>the Vandermonde family of every basis — <c>{p}vander</c>, <c>{p}vander2d</c>, <c>{p}vander3d</c> —
    ///         one whole-array kernel per (basis, dtype, rank), returning NumPy's <c>moveaxis</c> view;</item>
    ///         <item>companion matrices and roots of every basis — <c>{p}companion</c>, <c>{p}roots</c> (roots of
    ///         degree ≥ 2 run <c>np.linalg.eigvals</c>, so they need a LAPACK backend such as
    ///         <c>NumSharp.Interop.OpenBLAS</c>);</item>
    ///         <item>the weight functions <c>{p}weight</c>, the Chebyshev points and Gauss quadratures (<c>chebpts1</c>,
    ///         <c>chebpts2</c>, <c>chebgauss</c>, <c>leggauss</c>, <c>laggauss</c>, <c>hermgauss</c>, <c>hermegauss</c>;
    ///         the last four run <c>np.linalg.eigvalsh</c>), <c>chebinterpolate</c>, the least-squares fits
    ///         <c>{p}fit</c> (<c>np.linalg.lstsq</c>) and <c>polyvalfromroots</c>;</item>
    ///         <item>printing: <see cref="set_default_printstyle"/> and <see cref="PolyUtilsModule.format_float"/>;</item>
    ///         <item>the six series classes — <see cref="NDPolynomial"/>, <see cref="NDChebyshev"/>, <see cref="NDLegendre"/>,
    ///         <see cref="NDLaguerre"/>, <see cref="NDHermite"/>, <see cref="NDHermiteE"/> (factories <see cref="Polynomial"/>,
    ///         <see cref="Chebyshev"/>, …) — NumPy's <c>ABCPolyBase</c> statement for statement over the module functions:
    ///         the arithmetic with Python's operator protocol, evaluation (at a NumPy scalar in scalarmath, as NumPy's
    ///         <c>p(x)</c> does), calculus, roots, fits, basis conversion (NumPy's recurrence run at a series) and printing.</item>
    ///     </list>
    ///     <para>The kernels are emitted
    ///     at run time and have no fallback: under NativeAOT (no dynamic code) a call that reaches one raises
    ///     <see cref="System.PlatformNotSupportedException"/>, and the evaluation, calculus, algebra, Vandermonde and
    ///     companion paths all reach one.</para>
    ///     <para>https://numpy.org/doc/stable/reference/routines.polynomials-package.html</para>
    /// </remarks>
    [ModuleName("np.polynomial")]
    public class PolynomialModule
    {
        /// <summary>Constructs the facade; there is one shared instance behind <see cref="np.polynomial"/>.</summary>
        internal PolynomialModule() { }

        /// <summary>
        ///     <c>numpy.polynomial.polynomial</c> — power series <c>c[0] + c[1]*x + c[2]*x**2 + …</c>
        ///     (coefficients low degree first; see <see cref="PowerSeriesModule"/>).
        /// </summary>
        public PowerSeriesModule polynomial { get; } = new PowerSeriesModule();

        /// <summary><c>numpy.polynomial.chebyshev</c> — Chebyshev series of the first kind.</summary>
        public ChebyshevModule chebyshev { get; } = new ChebyshevModule();

        /// <summary><c>numpy.polynomial.legendre</c> — Legendre series.</summary>
        public LegendreModule legendre { get; } = new LegendreModule();

        /// <summary><c>numpy.polynomial.laguerre</c> — Laguerre series.</summary>
        public LaguerreModule laguerre { get; } = new LaguerreModule();

        /// <summary><c>numpy.polynomial.hermite</c> — physicists' Hermite series.</summary>
        public HermiteModule hermite { get; } = new HermiteModule();

        /// <summary><c>numpy.polynomial.hermite_e</c> — probabilists' Hermite series.</summary>
        public HermiteEModule hermite_e { get; } = new HermiteEModule();

        /// <summary>
        ///     <c>numpy.polynomial.polyutils</c> — the helpers every basis is built on (series normalization and the
        ///     linear domain maps; see <see cref="PolyUtilsModule"/>).
        /// </summary>
        public PolyUtilsModule polyutils { get; } = new PolyUtilsModule();

        /// <summary>
        ///     NumPy's <c>numpy.polynomial.polynomial.Polynomial(coef, domain=None, window=None, symbol='x')</c>: a power series — a new
        ///     <see cref="NDPolynomial"/> (the class itself; its class methods <c>fit</c>, <c>fromroots</c>, <c>identity</c>,
        ///     <c>basis</c> and <c>cast</c> are static members of <see cref="NDPolynomial"/>).
        /// </summary>
        /// <param name="coef">The coefficients, low degree first (see <see cref="NDPolynomial(object, object, object, string)"/>).</param>
        /// <param name="domain">The domain, or null for the class default.</param>
        /// <param name="window">The window, or null for the class default.</param>
        /// <param name="symbol">The variable's name for printing: a Python identifier.</param>
        /// <returns>The series (dispose it when done).</returns>
        /// <exception cref="ValueError">The constructor's errors (an empty or non-1-d series, a domain or window of the wrong
        ///     length, an invalid symbol).</exception>
        /// <exception cref="TypeError">A null symbol.</exception>
        /// <exception cref="System.NotSupportedException">Coefficients NumPy keeps as an object array.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.Polynomial.html</remarks>
        public NDPolynomial Polynomial(object coef, object domain = null, object window = null, string symbol = "x")
            => new NDPolynomial(coef, domain, window, symbol);

        /// <summary>
        ///     NumPy's <c>numpy.polynomial.chebyshev.Chebyshev(coef, domain=None, window=None, symbol='x')</c>: a Chebyshev series — a new
        ///     <see cref="NDChebyshev"/> (the class itself; its class methods <c>fit</c>, <c>fromroots</c>, <c>identity</c>,
        ///     <c>basis</c> and <c>cast</c> are static members of <see cref="NDChebyshev"/>).
        /// </summary>
        /// <param name="coef">The coefficients, low degree first (see <see cref="NDChebyshev(object, object, object, string)"/>).</param>
        /// <param name="domain">The domain, or null for the class default.</param>
        /// <param name="window">The window, or null for the class default.</param>
        /// <param name="symbol">The variable's name for printing: a Python identifier.</param>
        /// <returns>The series (dispose it when done).</returns>
        /// <exception cref="ValueError">The constructor's errors (an empty or non-1-d series, a domain or window of the wrong
        ///     length, an invalid symbol).</exception>
        /// <exception cref="TypeError">A null symbol.</exception>
        /// <exception cref="System.NotSupportedException">Coefficients NumPy keeps as an object array.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.chebyshev.Chebyshev.html</remarks>
        public NDChebyshev Chebyshev(object coef, object domain = null, object window = null, string symbol = "x")
            => new NDChebyshev(coef, domain, window, symbol);

        /// <summary>
        ///     NumPy's <c>numpy.polynomial.legendre.Legendre(coef, domain=None, window=None, symbol='x')</c>: a Legendre series — a new
        ///     <see cref="NDLegendre"/> (the class itself; its class methods <c>fit</c>, <c>fromroots</c>, <c>identity</c>,
        ///     <c>basis</c> and <c>cast</c> are static members of <see cref="NDLegendre"/>).
        /// </summary>
        /// <param name="coef">The coefficients, low degree first (see <see cref="NDLegendre(object, object, object, string)"/>).</param>
        /// <param name="domain">The domain, or null for the class default.</param>
        /// <param name="window">The window, or null for the class default.</param>
        /// <param name="symbol">The variable's name for printing: a Python identifier.</param>
        /// <returns>The series (dispose it when done).</returns>
        /// <exception cref="ValueError">The constructor's errors (an empty or non-1-d series, a domain or window of the wrong
        ///     length, an invalid symbol).</exception>
        /// <exception cref="TypeError">A null symbol.</exception>
        /// <exception cref="System.NotSupportedException">Coefficients NumPy keeps as an object array.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.legendre.Legendre.html</remarks>
        public NDLegendre Legendre(object coef, object domain = null, object window = null, string symbol = "x")
            => new NDLegendre(coef, domain, window, symbol);

        /// <summary>
        ///     NumPy's <c>numpy.polynomial.laguerre.Laguerre(coef, domain=None, window=None, symbol='x')</c>: a Laguerre series — a new
        ///     <see cref="NDLaguerre"/> (the class itself; its class methods <c>fit</c>, <c>fromroots</c>, <c>identity</c>,
        ///     <c>basis</c> and <c>cast</c> are static members of <see cref="NDLaguerre"/>).
        /// </summary>
        /// <param name="coef">The coefficients, low degree first (see <see cref="NDLaguerre(object, object, object, string)"/>).</param>
        /// <param name="domain">The domain, or null for the class default.</param>
        /// <param name="window">The window, or null for the class default.</param>
        /// <param name="symbol">The variable's name for printing: a Python identifier.</param>
        /// <returns>The series (dispose it when done).</returns>
        /// <exception cref="ValueError">The constructor's errors (an empty or non-1-d series, a domain or window of the wrong
        ///     length, an invalid symbol).</exception>
        /// <exception cref="TypeError">A null symbol.</exception>
        /// <exception cref="System.NotSupportedException">Coefficients NumPy keeps as an object array.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.laguerre.Laguerre.html</remarks>
        public NDLaguerre Laguerre(object coef, object domain = null, object window = null, string symbol = "x")
            => new NDLaguerre(coef, domain, window, symbol);

        /// <summary>
        ///     NumPy's <c>numpy.polynomial.hermite.Hermite(coef, domain=None, window=None, symbol='x')</c>: a (physicists') Hermite series — a new
        ///     <see cref="NDHermite"/> (the class itself; its class methods <c>fit</c>, <c>fromroots</c>, <c>identity</c>,
        ///     <c>basis</c> and <c>cast</c> are static members of <see cref="NDHermite"/>).
        /// </summary>
        /// <param name="coef">The coefficients, low degree first (see <see cref="NDHermite(object, object, object, string)"/>).</param>
        /// <param name="domain">The domain, or null for the class default.</param>
        /// <param name="window">The window, or null for the class default.</param>
        /// <param name="symbol">The variable's name for printing: a Python identifier.</param>
        /// <returns>The series (dispose it when done).</returns>
        /// <exception cref="ValueError">The constructor's errors (an empty or non-1-d series, a domain or window of the wrong
        ///     length, an invalid symbol).</exception>
        /// <exception cref="TypeError">A null symbol.</exception>
        /// <exception cref="System.NotSupportedException">Coefficients NumPy keeps as an object array.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.hermite.Hermite.html</remarks>
        public NDHermite Hermite(object coef, object domain = null, object window = null, string symbol = "x")
            => new NDHermite(coef, domain, window, symbol);

        /// <summary>
        ///     NumPy's <c>numpy.polynomial.hermite_e.HermiteE(coef, domain=None, window=None, symbol='x')</c>: a (probabilists') HermiteE series — a new
        ///     <see cref="NDHermiteE"/> (the class itself; its class methods <c>fit</c>, <c>fromroots</c>, <c>identity</c>,
        ///     <c>basis</c> and <c>cast</c> are static members of <see cref="NDHermiteE"/>).
        /// </summary>
        /// <param name="coef">The coefficients, low degree first (see <see cref="NDHermiteE(object, object, object, string)"/>).</param>
        /// <param name="domain">The domain, or null for the class default.</param>
        /// <param name="window">The window, or null for the class default.</param>
        /// <param name="symbol">The variable's name for printing: a Python identifier.</param>
        /// <returns>The series (dispose it when done).</returns>
        /// <exception cref="ValueError">The constructor's errors (an empty or non-1-d series, a domain or window of the wrong
        ///     length, an invalid symbol).</exception>
        /// <exception cref="TypeError">A null symbol.</exception>
        /// <exception cref="System.NotSupportedException">Coefficients NumPy keeps as an object array.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.hermite_e.HermiteE.html</remarks>
        public NDHermiteE HermiteE(object coef, object domain = null, object window = null, string symbol = "x")
            => new NDHermiteE(coef, domain, window, symbol);

        /// <summary>
        ///     Sets how the series classes print their terms (NumPy's <c>numpy.polynomial.set_default_printstyle</c>):
        ///     <c>"unicode"</c> writes superscripts and subscripts (<c>1.0 + 2.0·x + 3.0·x²</c>, <c>T₁(x)</c>),
        ///     <c>"ascii"</c> plain text (<c>1.0 + 2.0 x + 3.0 x**2</c>, <c>T_1(x)</c>).
        /// </summary>
        /// <param name="style"><c>"unicode"</c> or <c>"ascii"</c>, case-sensitive; a one-element <c>U</c> /
        ///     <c>StringDType</c> array holding one of them is read as its element, as NumPy's <c>==</c> reads it.</param>
        /// <exception cref="ValueError">Any other style: <c>Unsupported format string '{str(style)}'. Valid options are
        ///     'ascii' and 'unicode'</c> — also for a number, null (<c>'None'</c>), bytes or a list; an empty array or one of
        ///     two or more elements raises NumPy's truth-value ValueError instead.</exception>
        /// <remarks>
        ///     <para>The style is ONE process-wide setting, like NumPy's class attribute <c>ABCPolyBase._use_unicode</c>
        ///     (not scoped to a thread). NumPy's default is <c>os.name != 'nt'</c>: ASCII on Windows, Unicode elsewhere —
        ///     NumSharp starts from the same platform rule.</para>
        ///     <para>The series classes read it in <c>str(p)</c> (<see cref="NDPolyBase.__str__"/>, <c>ToString()</c>);
        ///     <c>format(p, "ascii")</c> / <c>"unicode"</c> (<see cref="NDPolyBase.__format__"/>) choose per call.</para>
        ///     <para>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.set_default_printstyle.html</para>
        /// </remarks>
        public void set_default_printstyle(object style) => NDPolyPrinting.SetDefaultPrintStyle(style);
    }
}
