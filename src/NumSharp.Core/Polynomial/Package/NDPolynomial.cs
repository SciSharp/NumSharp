using System;
using System.Collections;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NumSharp.Backends.Kernels;

// =============================================================================
// NDPolynomial.cs — numpy.polynomial.polynomial.Polynomial
// =============================================================================
//
// A thin class over NDPolyBase (the port of NumPy's ABCPolyBase): its class record (basis Power, NumPy's names, the
// class attributes), its constructors, the class methods typed with it, the covariant overrides of the members that
// return a series, and the C# operators declared with the class itself. Every body delegates to NDPolyBase.
// GENERATED from one template for the six classes; keep them identical but for the record and the power series' printers.
//
// =============================================================================

namespace NumSharp
{
    /// <summary>
    ///     NumPy's <c>numpy.polynomial.polynomial.Polynomial</c>: a power series: <c>c[0] + c[1]·x + c[2]·x² + …</c>, with a domain (default <c>[-1, 1]</c>) mapped onto a
    ///     window (default <c>[-1, 1]</c>). Arithmetic (<c>+ - * / %</c>, <see cref="__floordiv__"/>,
    ///     <see cref="__divmod__"/>, <see cref="__pow__"/>), evaluation (<see cref="NDPolyBase.Call(object)"/>), calculus
    ///     (<see cref="integ"/>, <see cref="deriv"/>), roots, fitting (<see cref="fit"/>), conversion between bases
    ///     (<see cref="NDPolyBase.convert"/>, <see cref="cast"/>) and printing, all as NumPy 2.4.2 has them.
    /// </summary>
    /// <remarks>
    ///     <para>See <see cref="NDPolyBase"/> for the ownership rule (dispose every series, operator results included) and
    ///     the C# spelling of NumPy's protocol. NumPy's class attributes are <see cref="default_domain"/>,
    ///     <see cref="default_window"/> and <see cref="default_maxpower"/>; the class methods (<see cref="identity"/>,
    ///     <see cref="basis"/>, <see cref="fromroots"/>, <see cref="fit"/>, <see cref="cast"/>) are static members.</para>
    ///     <para>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.Polynomial.html</para>
    /// </remarks>
    public sealed class NDPolynomial : NDPolyBase
    {
        /// <summary>The class record: basis Power, NumPy's names, the class attributes and the blank factory.</summary>
        internal static readonly NDPolyKind Kind = new NDPolyKind(PolyBasis.Power, "Polynomial", "numpy.polynomial.polynomial", null,
            typeof(NDPolynomial), () => default_domain, () => default_window, () => default_maxpower, () => new NDPolynomial());

        /// <summary>
        ///     NumPy's <c>Polynomial(coef, domain=None, window=None, symbol='x')</c>: the series with coefficients
        ///     <paramref name="coef"/> (low degree first), converted and copied as <c>polyutils.as_series</c> does.
        /// </summary>
        /// <param name="coef">The coefficients: a 1-D array_like (a C# array, an <see cref="NDArray"/>, a Python list / tuple,
        ///     a scalar); integers become float64 (a bool array has no common type), a float16 / float32 / complex array
        ///     keeps its dtype.</param>
        /// <param name="domain">The domain (two numbers), or null for the class default (shared, not copied).</param>
        /// <param name="window">The window (two numbers), or null for the class default.</param>
        /// <param name="symbol">The variable's name for printing: a Python identifier.</param>
        /// <exception cref="ValueError"><c>Coefficient array is empty</c>, <c>Coefficient array is not 1-d</c>,
        ///     <c>Coefficient arrays have no common type</c> (a bool array), <c>Domain has wrong number of elements.</c>,
        ///     <c>Window has wrong number of elements.</c>, <c>Symbol string must be a valid Python identifier</c>.</exception>
        /// <exception cref="TypeError"><c>Symbol must be a non-empty string</c> (a null symbol).</exception>
        /// <exception cref="NotSupportedException">Coefficients NumPy keeps as an OBJECT array (None, a Python int past
        ///     uint64, a list holding a non-number).</exception>
        public NDPolynomial(object coef, object domain = null, object window = null, string symbol = "x") : base(Kind)
        {
            Initialize(coef, domain, window, symbol, coefIsFresh: false);
        }

        /// <summary>A blank instance for <see cref="NDPolyKind.Blank"/>: NDPolyBase fills it in, exactly once.</summary>
        private NDPolynomial() : base(Kind)
        {
        }

        // ---------------------------------------------------------------------------------------------
        //  Class attributes
        // ---------------------------------------------------------------------------------------------

        /// <summary>The class attribute <c>domain</c> (created on first use).</summary>
        private static NDArray s_domain;

        /// <summary>The class attribute <c>window</c> (created on first use).</summary>
        private static NDArray s_window;

        /// <summary>The class attribute <c>maxpower</c> (NumPy's <c>ABCPolyBase.maxpower = 100</c>, a Python int).</summary>
        private static object s_maxpower = 100L;

        /// <summary>
        ///     NumPy's class attribute <c>Polynomial.domain</c>: <c>[-1, 1]</c> (float64), shared by every series created without
        ///     a domain — writing into it changes theirs. Assigning replaces it for those series (NumPy's
        ///     <c>Polynomial.domain = ...</c>).
        /// </summary>
        /// <exception cref="ArgumentNullException">Assigning null.</exception>
        public static NDArray default_domain
        {
            get => ClassDefault(ref s_domain, -1.0, 1.0);
            set => SetClassDefault(ref s_domain, value);
        }

        /// <summary>NumPy's class attribute <c>Polynomial.window</c>: <c>[-1, 1]</c> (float64), shared like <see cref="default_domain"/>.</summary>
        /// <exception cref="ArgumentNullException">Assigning null.</exception>
        public static NDArray default_window
        {
            get => ClassDefault(ref s_window, -1.0, 1.0);
            set => SetClassDefault(ref s_window, value);
        }

        /// <summary>
        ///     NumPy's class attribute <c>Polynomial.maxpower</c>: the largest power <see cref="__pow__"/> allows on a series that
        ///     has no <see cref="NDPolyBase.maxpower"/> of its own — a boxed Python int 100; null is None (no limit).
        /// </summary>
        public static object default_maxpower
        {
            get => System.Threading.Volatile.Read(ref s_maxpower);
            set => System.Threading.Volatile.Write(ref s_maxpower, value);
        }

        // ---------------------------------------------------------------------------------------------
        //  Members returning this class (covariant overrides of NDPolyBase's)
        // ---------------------------------------------------------------------------------------------

        /// <inheritdoc cref="NDPolyBase.copy"/>
        public override NDPolynomial copy() => (NDPolynomial)base.copy();

        /// <inheritdoc cref="NDPolyBase.cutdeg"/>
        public override NDPolynomial cutdeg(object deg) => (NDPolynomial)base.cutdeg(deg);

        /// <inheritdoc cref="NDPolyBase.trim"/>
        public override NDPolynomial trim([Optional] object tol) => (NDPolynomial)base.trim(tol);

        /// <inheritdoc cref="NDPolyBase.truncate"/>
        public override NDPolynomial truncate(object size) => (NDPolynomial)base.truncate(size);

        /// <inheritdoc cref="NDPolyBase.integ"/>
        public override NDPolynomial integ(int m = 1, object k = null, object lbnd = null) => (NDPolynomial)base.integ(m, k, lbnd);

        /// <inheritdoc cref="NDPolyBase.deriv"/>
        public override NDPolynomial deriv(int m = 1) => (NDPolynomial)base.deriv(m);

        /// <inheritdoc cref="NDPolyBase.__pow__"/>
        public override NDPolynomial __pow__(object power) => (NDPolynomial)base.__pow__(power);

        /// <inheritdoc cref="NDPolyBase.__floordiv__"/>
        public override NDPolynomial __floordiv__(object other) => (NDPolynomial)base.__floordiv__(other);

        /// <inheritdoc cref="NDPolyBase.__rfloordiv__"/>
        public override NDPolynomial __rfloordiv__(object other) => (NDPolynomial)base.__rfloordiv__(other);

        /// <inheritdoc cref="NDPolyBase.__divmod__"/>
        public new (NDPolynomial quo, NDPolynomial rem) __divmod__(object other)
        {
            var (q, r) = base.__divmod__(other);
            return ((NDPolynomial)q, (NDPolynomial)r);
        }

        /// <inheritdoc cref="NDPolyBase.__rdivmod__"/>
        public new (NDPolynomial quo, NDPolynomial rem) __rdivmod__(object other)
        {
            var (q, r) = base.__rdivmod__(other);
            return ((NDPolynomial)q, (NDPolynomial)r);
        }

        // ---------------------------------------------------------------------------------------------
        //  Class methods
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     NumPy's <c>Polynomial.identity(domain=None, window=None, symbol='x')</c>: the series whose value at every x is x (in
        ///     this basis, mapped through the domain and window).
        /// </summary>
        /// <param name="domain">The domain, or null for the class default.</param>
        /// <param name="window">The window, or null for the class default.</param>
        /// <param name="symbol">The symbol.</param>
        /// <returns>The identity series.</returns>
        /// <exception cref="ValueError">A domain / window the constructor refuses; an invalid symbol.</exception>
        /// <exception cref="IndexError">A domain or window shorter than two.</exception>
        public static NDPolynomial identity(object domain = null, object window = null, string symbol = "x")
            => (NDPolynomial)Identity(Kind, domain, window, symbol);

        /// <summary>
        ///     NumPy's <c>Polynomial.basis(deg, domain=None, window=None, symbol='x')</c>: the basis polynomial of degree
        ///     <paramref name="deg"/> (coefficients <c>[0, …, 0, 1]</c>).
        /// </summary>
        /// <param name="deg">The degree: a non-negative integer (Python's <c>int()</c> and <c>!=</c> decide).</param>
        /// <param name="domain">The domain, or null for the class default.</param>
        /// <param name="window">The window, or null for the class default.</param>
        /// <param name="symbol">The symbol.</param>
        /// <returns>The basis series.</returns>
        /// <exception cref="ValueError"><c>deg must be non-negative integer</c>.</exception>
        /// <exception cref="TypeError"><c>int()</c> refuses <paramref name="deg"/>.</exception>
        public static NDPolynomial basis(object deg, object domain = null, object window = null, string symbol = "x")
            => (NDPolynomial)BasisOf(Kind, deg, domain, window, symbol);

        /// <summary>
        ///     NumPy's <c>Polynomial.fromroots(roots, domain=[], window=None, symbol='x')</c>: the series
        ///     <c>(x - r[0])·(x - r[1])·…</c>, built in the window variable (<c>fromroots(off + scl·roots) / scl**deg</c>).
        /// </summary>
        /// <param name="roots">The roots (1-D).</param>
        /// <param name="domain">Omitted or an empty list: the class default (NumPy's <c>[]</c>); null: the roots' own
        ///     interval (NumPy's None, <c>getdomain(roots)</c>); otherwise the domain.</param>
        /// <param name="window">The window, or null for the class default.</param>
        /// <param name="symbol">The symbol.</param>
        /// <returns>The series.</returns>
        /// <exception cref="ValueError">Empty or non-1-d roots; a domain / window the constructor refuses.</exception>
        /// <exception cref="IndexError">An empty domain array (mapparms' indexing).</exception>
        public static NDPolynomial fromroots(object roots, [Optional] object domain, object window = null, string symbol = "x")
            => (NDPolynomial)FromRootsOf(Kind, roots, domain, window, symbol);

        /// <summary>
        ///     NumPy's <c>Polynomial.fit(x, y, deg, domain=None, rcond=None, full=False, w=None, window=None, symbol='x')</c>: the
        ///     least-squares fit of this basis to the points, on a domain covering x by default (mapped onto the window, which
        ///     keeps the problem well conditioned).
        /// </summary>
        /// <param name="x">The sample points (1-D).</param>
        /// <param name="y">The values (1-D, or 2-D with one data set per column).</param>
        /// <param name="deg">The degree, or a 1-D list of the degrees to fit.</param>
        /// <param name="domain">null: the smallest interval holding x (widened by 1 when x is constant); an empty list: the
        ///     class default; otherwise the domain.</param>
        /// <param name="rcond">The relative cutoff of small singular values, or null for <c>len(x)·eps</c>.</param>
        /// <param name="full">Also keep lstsq's statistics: <c>var (s, (resid, rank, sv, rcond)) = NDPolynomial.fit(…, full: true)</c>.</param>
        /// <param name="w">Weights (1-D), or null.</param>
        /// <param name="window">The window, or null for the class default.</param>
        /// <param name="symbol">The symbol.</param>
        /// <returns>The fitted series (deconstructible into the series and the statistics when <paramref name="full"/>).</returns>
        /// <exception cref="ValueError"><c>expected deg &gt;= 0</c> and the fit's other checks; the constructor's errors.</exception>
        /// <exception cref="TypeError">The fit's argument errors (<c>expected 1D vector for x</c>, …).</exception>
        /// <exception cref="MissingBackendException">No LAPACK backend: lstsq needs <c>NumSharp.Interop.OpenBLAS</c>.</exception>
        public static NDPolynomial fit(object x, object y, object deg, object domain = null, object rcond = null, bool full = false,
            object w = null, object window = null, string symbol = "x")
            => (NDPolynomial)FitOf(Kind, x, y, deg, domain, rcond, full, w, window, symbol);

        /// <summary>
        ///     NumPy's <c>Polynomial.cast(series, domain=None, window=None)</c>: <paramref name="series"/> converted to this class
        ///     (<c>series.convert(domain, Polynomial, window)</c>).
        /// </summary>
        /// <param name="series">The series to convert (any of the six classes).</param>
        /// <param name="domain">The domain, or null for the class default.</param>
        /// <param name="window">The window, or null for the class default.</param>
        /// <returns>The converted series.</returns>
        /// <exception cref="AttributeError"><paramref name="series"/> is null.</exception>
        public static NDPolynomial cast(NDPolyBase series, object domain = null, object window = null)
            => (NDPolynomial)CastOf(Kind, series, domain, window);

        /// <summary>
        ///     The power series' Unicode term (NumPy's <c>Polynomial._str_term_unicode</c>): <c>·x</c> for the first power, else
        ///     the symbol with a superscript power, <c>·x²</c>.
        /// </summary>
        /// <param name="i">The power, as text.</param><param name="argStr">The (scaled) symbol.</param>
        /// <returns>The term.</returns>
        private protected override string StrTermUnicode(string i, string argStr)
            => i == "1" ? "·" + argStr : "·" + argStr + Translate(i, Superscripts);

        /// <summary>The power series' ASCII term (NumPy's <c>Polynomial._str_term_ascii</c>): <c> x</c>, else <c> x**2</c>.</summary>
        /// <param name="i">The power, as text.</param><param name="argStr">The (scaled) symbol.</param>
        /// <returns>The term.</returns>
        private protected override string StrTermAscii(string i, string argStr)
            => i == "1" ? " " + argStr : " " + argStr + "**" + i;

        /// <summary>
        ///     The power series' LaTeX term (NumPy's <c>Polynomial._repr_latex_term</c>): <c>1</c> for the constant, the
        ///     argument for the first power, <c>x^{2}</c> beyond — a scaled argument wrapped in <c>\left( \right)</c>.
        /// </summary>
        /// <param name="i">The power.</param><param name="argStr">The (scaled) symbol.</param>
        /// <param name="needsParens">Whether the argument is a scaled symbol.</param>
        /// <returns>The term.</returns>
        private protected override string ReprLatexTerm(long i, string argStr, bool needsParens)
        {
            if (needsParens)
                argStr = "\\left(" + argStr + "\\right)";
            if (i == 0)
                return "1";
            if (i == 1)
                return argStr;
            return argStr + "^{" + i.ToString(System.Globalization.CultureInfo.InvariantCulture) + "}";
        }

        /// <summary>
        ///     NumPy's <c>series, [resid, rank, sv, rcond] = Polynomial.fit(..., full=True)</c>: the fitted series and lstsq's
        ///     statistics (the residual sums of squares, the rank, the singular values, the rcond used).
        /// </summary>
        /// <param name="series">This series.</param>
        /// <param name="status">The statistics, in NumPy's order. The caller owns them (no scope tracks them; the garbage
        ///     collector releases them).</param>
        /// <exception cref="ValueError">The series is not the result of a <c>fit(..., full: true)</c> (NumPy's series does not
        ///     unpack either).</exception>
        public void Deconstruct(out NDPolynomial series, out (NDArray residuals, NDArray rank, NDArray singular_values, NDArray rcond) status)
        {
            var f = Fit ?? throw new ValueError("the series is not the result of fit(..., full=True): there are no statistics to unpack");
            series = this;
            status = (f.Residuals, f.Rank, f.SingularValues, f.Rcond);
        }

        // ---------------------------------------------------------------------------------------------
        //  Operators (NumPy's protocol, NDPolyBase.Protocol.cs), typed with this class
        // ---------------------------------------------------------------------------------------------

        /// <inheritdoc cref="NDPolyBase.operator -(NDPolyBase)"/>
        public static NDPolynomial operator -(NDPolynomial a) => (NDPolynomial)a.Negative();

        /// <inheritdoc cref="NDPolyBase.operator +(NDPolyBase)"/>
        public static NDPolynomial operator +(NDPolynomial a) => a;

        /// <inheritdoc cref="NDPolyBase.operator +(NDPolyBase, object)"/>
        public static NDPolynomial operator +(NDPolynomial a, object b) => (NDPolynomial)Binary(PyBinOp.Add, a, b);
        /// <inheritdoc cref="NDPolyBase.operator +(NDPolyBase, NDArray)"/>
        public static NDPolynomial operator +(NDPolynomial a, NDArray b) => (NDPolynomial)Binary(PyBinOp.Add, a, b);
        /// <inheritdoc cref="NDPolyBase.operator +(NDPolyBase, object[])"/>
        public static NDPolynomial operator +(NDPolynomial a, object[] b) => (NDPolynomial)Binary(PyBinOp.Add, a, b);
        /// <inheritdoc cref="NDPolyBase.operator +(NDPolyBase, Array)"/>
        public static NDPolynomial operator +(NDPolynomial a, Array b) => (NDPolynomial)Binary(PyBinOp.Add, a, b);
        /// <inheritdoc cref="NDPolyBase.operator +(NDPolyBase, string)"/>
        public static NDPolynomial operator +(NDPolynomial a, string b) => (NDPolynomial)Binary(PyBinOp.Add, a, b);
        /// <inheritdoc cref="NDPolyBase.operator +(NDArray, NDPolyBase)"/>
        public static NDPolynomial operator +(NDArray a, NDPolynomial b) => (NDPolynomial)Binary(PyBinOp.Add, a, b);
        /// <inheritdoc cref="NDPolyBase.operator +(Array, NDPolyBase)"/>
        public static NDPolynomial operator +(Array a, NDPolynomial b) => (NDPolynomial)Binary(PyBinOp.Add, a, b);
        /// <inheritdoc cref="NDPolyBase.operator +(object[], NDPolyBase)"/>
        public static NDPolynomial operator +(object[] a, NDPolynomial b) => (NDPolynomial)Binary(PyBinOp.Add, a, b);
        /// <inheritdoc cref="NDPolyBase.operator +(IList, NDPolyBase)"/>
        public static NDPolynomial operator +(IList a, NDPolynomial b) => (NDPolynomial)Binary(PyBinOp.Add, a, b);
        /// <inheritdoc cref="NDPolyBase.operator +(ITuple, NDPolyBase)"/>
        public static NDPolynomial operator +(ITuple a, NDPolynomial b) => (NDPolynomial)Binary(PyBinOp.Add, a, b);
        /// <inheritdoc cref="NDPolyBase.operator +(string, NDPolyBase)"/>
        public static NDPolynomial operator +(string a, NDPolynomial b) => (NDPolynomial)Binary(PyBinOp.Add, a, b);

        /// <inheritdoc cref="NDPolyBase.operator -(NDPolyBase, object)"/>
        public static NDPolynomial operator -(NDPolynomial a, object b) => (NDPolynomial)Binary(PyBinOp.Sub, a, b);
        /// <inheritdoc cref="NDPolyBase.operator -(NDPolyBase, NDArray)"/>
        public static NDPolynomial operator -(NDPolynomial a, NDArray b) => (NDPolynomial)Binary(PyBinOp.Sub, a, b);
        /// <inheritdoc cref="NDPolyBase.operator -(NDPolyBase, object[])"/>
        public static NDPolynomial operator -(NDPolynomial a, object[] b) => (NDPolynomial)Binary(PyBinOp.Sub, a, b);
        /// <inheritdoc cref="NDPolyBase.operator -(NDPolyBase, Array)"/>
        public static NDPolynomial operator -(NDPolynomial a, Array b) => (NDPolynomial)Binary(PyBinOp.Sub, a, b);
        /// <inheritdoc cref="NDPolyBase.operator -(NDPolyBase, string)"/>
        public static NDPolynomial operator -(NDPolynomial a, string b) => (NDPolynomial)Binary(PyBinOp.Sub, a, b);
        /// <inheritdoc cref="NDPolyBase.operator -(NDArray, NDPolyBase)"/>
        public static NDPolynomial operator -(NDArray a, NDPolynomial b) => (NDPolynomial)Binary(PyBinOp.Sub, a, b);
        /// <inheritdoc cref="NDPolyBase.operator -(Array, NDPolyBase)"/>
        public static NDPolynomial operator -(Array a, NDPolynomial b) => (NDPolynomial)Binary(PyBinOp.Sub, a, b);
        /// <inheritdoc cref="NDPolyBase.operator -(object[], NDPolyBase)"/>
        public static NDPolynomial operator -(object[] a, NDPolynomial b) => (NDPolynomial)Binary(PyBinOp.Sub, a, b);
        /// <inheritdoc cref="NDPolyBase.operator -(IList, NDPolyBase)"/>
        public static NDPolynomial operator -(IList a, NDPolynomial b) => (NDPolynomial)Binary(PyBinOp.Sub, a, b);
        /// <inheritdoc cref="NDPolyBase.operator -(ITuple, NDPolyBase)"/>
        public static NDPolynomial operator -(ITuple a, NDPolynomial b) => (NDPolynomial)Binary(PyBinOp.Sub, a, b);
        /// <inheritdoc cref="NDPolyBase.operator -(string, NDPolyBase)"/>
        public static NDPolynomial operator -(string a, NDPolynomial b) => (NDPolynomial)Binary(PyBinOp.Sub, a, b);

        /// <inheritdoc cref="NDPolyBase.operator *(NDPolyBase, object)"/>
        public static NDPolynomial operator *(NDPolynomial a, object b) => (NDPolynomial)Binary(PyBinOp.Mul, a, b);
        /// <inheritdoc cref="NDPolyBase.operator *(NDPolyBase, NDArray)"/>
        public static NDPolynomial operator *(NDPolynomial a, NDArray b) => (NDPolynomial)Binary(PyBinOp.Mul, a, b);
        /// <inheritdoc cref="NDPolyBase.operator *(NDPolyBase, object[])"/>
        public static NDPolynomial operator *(NDPolynomial a, object[] b) => (NDPolynomial)Binary(PyBinOp.Mul, a, b);
        /// <inheritdoc cref="NDPolyBase.operator *(NDPolyBase, Array)"/>
        public static NDPolynomial operator *(NDPolynomial a, Array b) => (NDPolynomial)Binary(PyBinOp.Mul, a, b);
        /// <inheritdoc cref="NDPolyBase.operator *(NDPolyBase, string)"/>
        public static NDPolynomial operator *(NDPolynomial a, string b) => (NDPolynomial)Binary(PyBinOp.Mul, a, b);
        /// <inheritdoc cref="NDPolyBase.operator *(NDArray, NDPolyBase)"/>
        public static NDPolynomial operator *(NDArray a, NDPolynomial b) => (NDPolynomial)Binary(PyBinOp.Mul, a, b);
        /// <inheritdoc cref="NDPolyBase.operator *(Array, NDPolyBase)"/>
        public static NDPolynomial operator *(Array a, NDPolynomial b) => (NDPolynomial)Binary(PyBinOp.Mul, a, b);
        /// <inheritdoc cref="NDPolyBase.operator *(object[], NDPolyBase)"/>
        public static NDPolynomial operator *(object[] a, NDPolynomial b) => (NDPolynomial)Binary(PyBinOp.Mul, a, b);
        /// <inheritdoc cref="NDPolyBase.operator *(IList, NDPolyBase)"/>
        public static NDPolynomial operator *(IList a, NDPolynomial b) => (NDPolynomial)Binary(PyBinOp.Mul, a, b);
        /// <inheritdoc cref="NDPolyBase.operator *(ITuple, NDPolyBase)"/>
        public static NDPolynomial operator *(ITuple a, NDPolynomial b) => (NDPolynomial)Binary(PyBinOp.Mul, a, b);
        /// <inheritdoc cref="NDPolyBase.operator *(string, NDPolyBase)"/>
        public static NDPolynomial operator *(string a, NDPolynomial b) => (NDPolynomial)Binary(PyBinOp.Mul, a, b);

        /// <inheritdoc cref="NDPolyBase.operator /(NDPolyBase, object)"/>
        public static NDPolynomial operator /(NDPolynomial a, object b) => (NDPolynomial)Binary(PyBinOp.TrueDiv, a, b);
        /// <inheritdoc cref="NDPolyBase.operator /(NDPolyBase, NDArray)"/>
        public static NDPolynomial operator /(NDPolynomial a, NDArray b) => (NDPolynomial)Binary(PyBinOp.TrueDiv, a, b);
        /// <inheritdoc cref="NDPolyBase.operator /(NDPolyBase, object[])"/>
        public static NDPolynomial operator /(NDPolynomial a, object[] b) => (NDPolynomial)Binary(PyBinOp.TrueDiv, a, b);
        /// <inheritdoc cref="NDPolyBase.operator /(NDPolyBase, Array)"/>
        public static NDPolynomial operator /(NDPolynomial a, Array b) => (NDPolynomial)Binary(PyBinOp.TrueDiv, a, b);
        /// <inheritdoc cref="NDPolyBase.operator /(NDPolyBase, string)"/>
        public static NDPolynomial operator /(NDPolynomial a, string b) => (NDPolynomial)Binary(PyBinOp.TrueDiv, a, b);
        /// <inheritdoc cref="NDPolyBase.operator /(NDArray, NDPolyBase)"/>
        public static NDPolynomial operator /(NDArray a, NDPolynomial b) => (NDPolynomial)Binary(PyBinOp.TrueDiv, a, b);
        /// <inheritdoc cref="NDPolyBase.operator /(Array, NDPolyBase)"/>
        public static NDPolynomial operator /(Array a, NDPolynomial b) => (NDPolynomial)Binary(PyBinOp.TrueDiv, a, b);
        /// <inheritdoc cref="NDPolyBase.operator /(object[], NDPolyBase)"/>
        public static NDPolynomial operator /(object[] a, NDPolynomial b) => (NDPolynomial)Binary(PyBinOp.TrueDiv, a, b);
        /// <inheritdoc cref="NDPolyBase.operator /(IList, NDPolyBase)"/>
        public static NDPolynomial operator /(IList a, NDPolynomial b) => (NDPolynomial)Binary(PyBinOp.TrueDiv, a, b);
        /// <inheritdoc cref="NDPolyBase.operator /(ITuple, NDPolyBase)"/>
        public static NDPolynomial operator /(ITuple a, NDPolynomial b) => (NDPolynomial)Binary(PyBinOp.TrueDiv, a, b);
        /// <inheritdoc cref="NDPolyBase.operator /(string, NDPolyBase)"/>
        public static NDPolynomial operator /(string a, NDPolynomial b) => (NDPolynomial)Binary(PyBinOp.TrueDiv, a, b);

        /// <inheritdoc cref="NDPolyBase.operator %(NDPolyBase, object)"/>
        public static NDPolynomial operator %(NDPolynomial a, object b) => (NDPolynomial)Binary(PyBinOp.Mod, a, b);
        /// <inheritdoc cref="NDPolyBase.operator %(NDPolyBase, NDArray)"/>
        public static NDPolynomial operator %(NDPolynomial a, NDArray b) => (NDPolynomial)Binary(PyBinOp.Mod, a, b);
        /// <inheritdoc cref="NDPolyBase.operator %(NDPolyBase, object[])"/>
        public static NDPolynomial operator %(NDPolynomial a, object[] b) => (NDPolynomial)Binary(PyBinOp.Mod, a, b);
        /// <inheritdoc cref="NDPolyBase.operator %(NDPolyBase, Array)"/>
        public static NDPolynomial operator %(NDPolynomial a, Array b) => (NDPolynomial)Binary(PyBinOp.Mod, a, b);
        /// <inheritdoc cref="NDPolyBase.operator %(NDPolyBase, string)"/>
        public static NDPolynomial operator %(NDPolynomial a, string b) => (NDPolynomial)Binary(PyBinOp.Mod, a, b);
        /// <inheritdoc cref="NDPolyBase.operator %(NDArray, NDPolyBase)"/>
        public static NDPolynomial operator %(NDArray a, NDPolynomial b) => (NDPolynomial)Binary(PyBinOp.Mod, a, b);
        /// <inheritdoc cref="NDPolyBase.operator %(Array, NDPolyBase)"/>
        public static NDPolynomial operator %(Array a, NDPolynomial b) => (NDPolynomial)Binary(PyBinOp.Mod, a, b);
        /// <inheritdoc cref="NDPolyBase.operator %(object[], NDPolyBase)"/>
        public static NDPolynomial operator %(object[] a, NDPolynomial b) => (NDPolynomial)Binary(PyBinOp.Mod, a, b);
        /// <inheritdoc cref="NDPolyBase.operator %(IList, NDPolyBase)"/>
        public static NDPolynomial operator %(IList a, NDPolynomial b) => (NDPolynomial)Binary(PyBinOp.Mod, a, b);
        /// <inheritdoc cref="NDPolyBase.operator %(ITuple, NDPolyBase)"/>
        public static NDPolynomial operator %(ITuple a, NDPolynomial b) => (NDPolynomial)Binary(PyBinOp.Mod, a, b);
        /// <inheritdoc cref="NDPolyBase.operator %(string, NDPolyBase)"/>
        public static NDPolynomial operator %(string a, NDPolynomial b) => (NDPolynomial)Binary(PyBinOp.Mod, a, b);
    }
}
