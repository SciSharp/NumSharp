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
    ///     <para>Implemented so far: the evaluation family of every basis — <c>{p}val</c>, <c>{p}val2d</c>,
    ///     <c>{p}val3d</c>, <c>{p}grid2d</c>, <c>{p}grid3d</c> and <c>{p}valnd</c> — bit-identical to NumPy 2.4.2
    ///     for every dtype NumPy has, through one runtime-compiled IL kernel per (basis, dtypes, layout) class.</para>
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
    }
}
