using System;
using System.Collections;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NumSharp.Backends.Kernels;

// =============================================================================
// NDChebyshev.cs — numpy.polynomial.chebyshev.Chebyshev
// =============================================================================
//
// A thin class over NDPolyBase (the port of NumPy's ABCPolyBase): its class record (basis Chebyshev, NumPy's names, the
// class attributes), its constructors, the class methods typed with it, the covariant overrides of the members that
// return a series, and the C# operators declared with the class itself. Every body delegates to NDPolyBase.
// GENERATED from one template for the six classes; keep them identical but for the record and interpolate.
//
// =============================================================================

namespace NumSharp
{
    /// <summary>
    ///     NumPy's <c>numpy.polynomial.chebyshev.Chebyshev</c>: a Chebyshev series of the first kind: <c>c[0]·T₀(x) + c[1]·T₁(x) + …</c>, with a domain (default <c>[-1, 1]</c>) mapped onto a
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
    ///     <para>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.chebyshev.Chebyshev.html</para>
    /// </remarks>
    public sealed class NDChebyshev : NDPolyBase
    {
        /// <summary>The class record: basis Chebyshev, NumPy's names, the class attributes and the blank factory.</summary>
        internal static readonly NDPolyKind Kind = new NDPolyKind(PolyBasis.Chebyshev, "Chebyshev", "numpy.polynomial.chebyshev", "T",
            typeof(NDChebyshev), () => default_domain, () => default_window, () => default_maxpower, () => new NDChebyshev());

        /// <summary>
        ///     NumPy's <c>Chebyshev(coef, domain=None, window=None, symbol='x')</c>: the series with coefficients
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
        public NDChebyshev(object coef, object domain = null, object window = null, string symbol = "x") : base(Kind)
        {
            Initialize(coef, domain, window, symbol, coefIsFresh: false);
        }

        /// <summary>A blank instance for <see cref="NDPolyKind.Blank"/>: NDPolyBase fills it in, exactly once.</summary>
        private NDChebyshev() : base(Kind)
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
        ///     NumPy's class attribute <c>Chebyshev.domain</c>: <c>[-1, 1]</c> (float64), shared by every series created without
        ///     a domain — writing into it changes theirs. Assigning replaces it for those series (NumPy's
        ///     <c>Chebyshev.domain = ...</c>).
        /// </summary>
        /// <exception cref="ArgumentNullException">Assigning null.</exception>
        public static NDArray default_domain
        {
            get => ClassDefault(ref s_domain, -1.0, 1.0);
            set => SetClassDefault(ref s_domain, value);
        }

        /// <summary>NumPy's class attribute <c>Chebyshev.window</c>: <c>[-1, 1]</c> (float64), shared like <see cref="default_domain"/>.</summary>
        /// <exception cref="ArgumentNullException">Assigning null.</exception>
        public static NDArray default_window
        {
            get => ClassDefault(ref s_window, -1.0, 1.0);
            set => SetClassDefault(ref s_window, value);
        }

        /// <summary>
        ///     NumPy's class attribute <c>Chebyshev.maxpower</c>: the largest power <see cref="__pow__"/> allows on a series that
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
        public override NDChebyshev copy() => (NDChebyshev)base.copy();

        /// <inheritdoc cref="NDPolyBase.cutdeg"/>
        public override NDChebyshev cutdeg(object deg) => (NDChebyshev)base.cutdeg(deg);

        /// <inheritdoc cref="NDPolyBase.trim"/>
        public override NDChebyshev trim([Optional] object tol) => (NDChebyshev)base.trim(tol);

        /// <inheritdoc cref="NDPolyBase.truncate"/>
        public override NDChebyshev truncate(object size) => (NDChebyshev)base.truncate(size);

        /// <inheritdoc cref="NDPolyBase.integ"/>
        public override NDChebyshev integ(int m = 1, object k = null, object lbnd = null) => (NDChebyshev)base.integ(m, k, lbnd);

        /// <inheritdoc cref="NDPolyBase.deriv"/>
        public override NDChebyshev deriv(int m = 1) => (NDChebyshev)base.deriv(m);

        /// <inheritdoc cref="NDPolyBase.__pow__"/>
        public override NDChebyshev __pow__(object power) => (NDChebyshev)base.__pow__(power);

        /// <inheritdoc cref="NDPolyBase.__floordiv__"/>
        public override NDChebyshev __floordiv__(object other) => (NDChebyshev)base.__floordiv__(other);

        /// <inheritdoc cref="NDPolyBase.__rfloordiv__"/>
        public override NDChebyshev __rfloordiv__(object other) => (NDChebyshev)base.__rfloordiv__(other);

        /// <inheritdoc cref="NDPolyBase.__divmod__"/>
        public new (NDChebyshev quo, NDChebyshev rem) __divmod__(object other)
        {
            var (q, r) = base.__divmod__(other);
            return ((NDChebyshev)q, (NDChebyshev)r);
        }

        /// <inheritdoc cref="NDPolyBase.__rdivmod__"/>
        public new (NDChebyshev quo, NDChebyshev rem) __rdivmod__(object other)
        {
            var (q, r) = base.__rdivmod__(other);
            return ((NDChebyshev)q, (NDChebyshev)r);
        }

        // ---------------------------------------------------------------------------------------------
        //  Class methods
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     NumPy's <c>Chebyshev.identity(domain=None, window=None, symbol='x')</c>: the series whose value at every x is x (in
        ///     this basis, mapped through the domain and window).
        /// </summary>
        /// <param name="domain">The domain, or null for the class default.</param>
        /// <param name="window">The window, or null for the class default.</param>
        /// <param name="symbol">The symbol.</param>
        /// <returns>The identity series.</returns>
        /// <exception cref="ValueError">A domain / window the constructor refuses; an invalid symbol.</exception>
        /// <exception cref="IndexError">A domain or window shorter than two.</exception>
        public static NDChebyshev identity(object domain = null, object window = null, string symbol = "x")
            => (NDChebyshev)Identity(Kind, domain, window, symbol);

        /// <summary>
        ///     NumPy's <c>Chebyshev.basis(deg, domain=None, window=None, symbol='x')</c>: the basis polynomial of degree
        ///     <paramref name="deg"/> (coefficients <c>[0, …, 0, 1]</c>).
        /// </summary>
        /// <param name="deg">The degree: a non-negative integer (Python's <c>int()</c> and <c>!=</c> decide).</param>
        /// <param name="domain">The domain, or null for the class default.</param>
        /// <param name="window">The window, or null for the class default.</param>
        /// <param name="symbol">The symbol.</param>
        /// <returns>The basis series.</returns>
        /// <exception cref="ValueError"><c>deg must be non-negative integer</c>.</exception>
        /// <exception cref="TypeError"><c>int()</c> refuses <paramref name="deg"/>.</exception>
        public static NDChebyshev basis(object deg, object domain = null, object window = null, string symbol = "x")
            => (NDChebyshev)BasisOf(Kind, deg, domain, window, symbol);

        /// <summary>
        ///     NumPy's <c>Chebyshev.fromroots(roots, domain=[], window=None, symbol='x')</c>: the series
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
        public static NDChebyshev fromroots(object roots, [Optional] object domain, object window = null, string symbol = "x")
            => (NDChebyshev)FromRootsOf(Kind, roots, domain, window, symbol);

        /// <summary>
        ///     NumPy's <c>Chebyshev.fit(x, y, deg, domain=None, rcond=None, full=False, w=None, window=None, symbol='x')</c>: the
        ///     least-squares fit of this basis to the points, on a domain covering x by default (mapped onto the window, which
        ///     keeps the problem well conditioned).
        /// </summary>
        /// <param name="x">The sample points (1-D).</param>
        /// <param name="y">The values (1-D, or 2-D with one data set per column).</param>
        /// <param name="deg">The degree, or a 1-D list of the degrees to fit.</param>
        /// <param name="domain">null: the smallest interval holding x (widened by 1 when x is constant); an empty list: the
        ///     class default; otherwise the domain.</param>
        /// <param name="rcond">The relative cutoff of small singular values, or null for <c>len(x)·eps</c>.</param>
        /// <param name="full">Also keep lstsq's statistics: <c>var (s, (resid, rank, sv, rcond)) = NDChebyshev.fit(…, full: true)</c>.</param>
        /// <param name="w">Weights (1-D), or null.</param>
        /// <param name="window">The window, or null for the class default.</param>
        /// <param name="symbol">The symbol.</param>
        /// <returns>The fitted series (deconstructible into the series and the statistics when <paramref name="full"/>).</returns>
        /// <exception cref="ValueError"><c>expected deg &gt;= 0</c> and the fit's other checks; the constructor's errors.</exception>
        /// <exception cref="TypeError">The fit's argument errors (<c>expected 1D vector for x</c>, …).</exception>
        /// <exception cref="MissingBackendException">No LAPACK backend: lstsq needs <c>NumSharp.Interop.OpenBLAS</c>.</exception>
        public static NDChebyshev fit(object x, object y, object deg, object domain = null, object rcond = null, bool full = false,
            object w = null, object window = null, string symbol = "x")
            => (NDChebyshev)FitOf(Kind, x, y, deg, domain, rcond, full, w, window, symbol);

        /// <summary>
        ///     NumPy's <c>Chebyshev.cast(series, domain=None, window=None)</c>: <paramref name="series"/> converted to this class
        ///     (<c>series.convert(domain, Chebyshev, window)</c>).
        /// </summary>
        /// <param name="series">The series to convert (any of the six classes).</param>
        /// <param name="domain">The domain, or null for the class default.</param>
        /// <param name="window">The window, or null for the class default.</param>
        /// <returns>The converted series.</returns>
        /// <exception cref="AttributeError"><paramref name="series"/> is null.</exception>
        public static NDChebyshev cast(NDPolyBase series, object domain = null, object window = null)
            => (NDChebyshev)CastOf(Kind, series, domain, window);

        /// <summary>
        ///     NumPy's <c>Chebyshev.interpolate(func, deg, domain=None, args=())</c>: the Chebyshev series of degree
        ///     <paramref name="deg"/> interpolating <paramref name="func"/> at the Chebyshev points of the first kind mapped into
        ///     <paramref name="domain"/> (<c>chebinterpolate</c> of <c>lambda x: func(mapdomain(x, window, domain), *args)</c>) —
        ///     close to the minimax approximation of a function continuous on the domain.
        /// </summary>
        /// <param name="func">The function: called once, with the mapped points (a float64 array) and the unpacked
        ///     <paramref name="args"/>; it returns the values at those points (anything <c>np.dot</c> accepts).</param>
        /// <param name="deg">The degree: an integer.</param>
        /// <param name="domain">The interval: null for the class default <c>[-1, 1]</c>.</param>
        /// <param name="args">Extra arguments for <paramref name="func"/> (any Python iterable), or null for none.</param>
        /// <returns>The interpolating series, on <paramref name="domain"/> with the class window and symbol <c>'x'</c>.</returns>
        /// <exception cref="TypeError"><c>deg must be an int</c>; the unpacking's errors; a null func.</exception>
        /// <exception cref="ValueError"><c>expected deg &gt;= 0</c>; a domain the constructor refuses (checked after the
        ///     interpolation, as NumPy does).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.chebyshev.Chebyshev.interpolate.html</remarks>
        public static NDChebyshev interpolate(Func<NDArray, object[], object> func, object deg, object domain = null, object args = null)
            => (NDChebyshev)InterpolateOf(func, deg, domain, args);

        /// <summary>
        ///     NumPy's <c>series, [resid, rank, sv, rcond] = Chebyshev.fit(..., full=True)</c>: the fitted series and lstsq's
        ///     statistics (the residual sums of squares, the rank, the singular values, the rcond used).
        /// </summary>
        /// <param name="series">This series.</param>
        /// <param name="status">The statistics, in NumPy's order. The caller owns them (no scope tracks them; the garbage
        ///     collector releases them).</param>
        /// <exception cref="ValueError">The series is not the result of a <c>fit(..., full: true)</c> (NumPy's series does not
        ///     unpack either).</exception>
        public void Deconstruct(out NDChebyshev series, out (NDArray residuals, NDArray rank, NDArray singular_values, NDArray rcond) status)
        {
            var f = Fit ?? throw new ValueError("the series is not the result of fit(..., full=True): there are no statistics to unpack");
            series = this;
            status = (f.Residuals, f.Rank, f.SingularValues, f.Rcond);
        }

        // ---------------------------------------------------------------------------------------------
        //  Operators (NumPy's protocol, NDPolyBase.Protocol.cs), typed with this class
        // ---------------------------------------------------------------------------------------------

        /// <inheritdoc cref="NDPolyBase.operator -(NDPolyBase)"/>
        public static NDChebyshev operator -(NDChebyshev a) => (NDChebyshev)a.Negative();

        /// <inheritdoc cref="NDPolyBase.operator +(NDPolyBase)"/>
        public static NDChebyshev operator +(NDChebyshev a) => a;

        /// <inheritdoc cref="NDPolyBase.operator +(NDPolyBase, object)"/>
        public static NDChebyshev operator +(NDChebyshev a, object b) => (NDChebyshev)Binary(PyBinOp.Add, a, b);
        /// <inheritdoc cref="NDPolyBase.operator +(NDPolyBase, NDArray)"/>
        public static NDChebyshev operator +(NDChebyshev a, NDArray b) => (NDChebyshev)Binary(PyBinOp.Add, a, b);
        /// <inheritdoc cref="NDPolyBase.operator +(NDPolyBase, object[])"/>
        public static NDChebyshev operator +(NDChebyshev a, object[] b) => (NDChebyshev)Binary(PyBinOp.Add, a, b);
        /// <inheritdoc cref="NDPolyBase.operator +(NDPolyBase, Array)"/>
        public static NDChebyshev operator +(NDChebyshev a, Array b) => (NDChebyshev)Binary(PyBinOp.Add, a, b);
        /// <inheritdoc cref="NDPolyBase.operator +(NDPolyBase, string)"/>
        public static NDChebyshev operator +(NDChebyshev a, string b) => (NDChebyshev)Binary(PyBinOp.Add, a, b);
        /// <inheritdoc cref="NDPolyBase.operator +(NDArray, NDPolyBase)"/>
        public static NDChebyshev operator +(NDArray a, NDChebyshev b) => (NDChebyshev)Binary(PyBinOp.Add, a, b);
        /// <inheritdoc cref="NDPolyBase.operator +(Array, NDPolyBase)"/>
        public static NDChebyshev operator +(Array a, NDChebyshev b) => (NDChebyshev)Binary(PyBinOp.Add, a, b);
        /// <inheritdoc cref="NDPolyBase.operator +(object[], NDPolyBase)"/>
        public static NDChebyshev operator +(object[] a, NDChebyshev b) => (NDChebyshev)Binary(PyBinOp.Add, a, b);
        /// <inheritdoc cref="NDPolyBase.operator +(IList, NDPolyBase)"/>
        public static NDChebyshev operator +(IList a, NDChebyshev b) => (NDChebyshev)Binary(PyBinOp.Add, a, b);
        /// <inheritdoc cref="NDPolyBase.operator +(ITuple, NDPolyBase)"/>
        public static NDChebyshev operator +(ITuple a, NDChebyshev b) => (NDChebyshev)Binary(PyBinOp.Add, a, b);
        /// <inheritdoc cref="NDPolyBase.operator +(string, NDPolyBase)"/>
        public static NDChebyshev operator +(string a, NDChebyshev b) => (NDChebyshev)Binary(PyBinOp.Add, a, b);

        /// <inheritdoc cref="NDPolyBase.operator -(NDPolyBase, object)"/>
        public static NDChebyshev operator -(NDChebyshev a, object b) => (NDChebyshev)Binary(PyBinOp.Sub, a, b);
        /// <inheritdoc cref="NDPolyBase.operator -(NDPolyBase, NDArray)"/>
        public static NDChebyshev operator -(NDChebyshev a, NDArray b) => (NDChebyshev)Binary(PyBinOp.Sub, a, b);
        /// <inheritdoc cref="NDPolyBase.operator -(NDPolyBase, object[])"/>
        public static NDChebyshev operator -(NDChebyshev a, object[] b) => (NDChebyshev)Binary(PyBinOp.Sub, a, b);
        /// <inheritdoc cref="NDPolyBase.operator -(NDPolyBase, Array)"/>
        public static NDChebyshev operator -(NDChebyshev a, Array b) => (NDChebyshev)Binary(PyBinOp.Sub, a, b);
        /// <inheritdoc cref="NDPolyBase.operator -(NDPolyBase, string)"/>
        public static NDChebyshev operator -(NDChebyshev a, string b) => (NDChebyshev)Binary(PyBinOp.Sub, a, b);
        /// <inheritdoc cref="NDPolyBase.operator -(NDArray, NDPolyBase)"/>
        public static NDChebyshev operator -(NDArray a, NDChebyshev b) => (NDChebyshev)Binary(PyBinOp.Sub, a, b);
        /// <inheritdoc cref="NDPolyBase.operator -(Array, NDPolyBase)"/>
        public static NDChebyshev operator -(Array a, NDChebyshev b) => (NDChebyshev)Binary(PyBinOp.Sub, a, b);
        /// <inheritdoc cref="NDPolyBase.operator -(object[], NDPolyBase)"/>
        public static NDChebyshev operator -(object[] a, NDChebyshev b) => (NDChebyshev)Binary(PyBinOp.Sub, a, b);
        /// <inheritdoc cref="NDPolyBase.operator -(IList, NDPolyBase)"/>
        public static NDChebyshev operator -(IList a, NDChebyshev b) => (NDChebyshev)Binary(PyBinOp.Sub, a, b);
        /// <inheritdoc cref="NDPolyBase.operator -(ITuple, NDPolyBase)"/>
        public static NDChebyshev operator -(ITuple a, NDChebyshev b) => (NDChebyshev)Binary(PyBinOp.Sub, a, b);
        /// <inheritdoc cref="NDPolyBase.operator -(string, NDPolyBase)"/>
        public static NDChebyshev operator -(string a, NDChebyshev b) => (NDChebyshev)Binary(PyBinOp.Sub, a, b);

        /// <inheritdoc cref="NDPolyBase.operator *(NDPolyBase, object)"/>
        public static NDChebyshev operator *(NDChebyshev a, object b) => (NDChebyshev)Binary(PyBinOp.Mul, a, b);
        /// <inheritdoc cref="NDPolyBase.operator *(NDPolyBase, NDArray)"/>
        public static NDChebyshev operator *(NDChebyshev a, NDArray b) => (NDChebyshev)Binary(PyBinOp.Mul, a, b);
        /// <inheritdoc cref="NDPolyBase.operator *(NDPolyBase, object[])"/>
        public static NDChebyshev operator *(NDChebyshev a, object[] b) => (NDChebyshev)Binary(PyBinOp.Mul, a, b);
        /// <inheritdoc cref="NDPolyBase.operator *(NDPolyBase, Array)"/>
        public static NDChebyshev operator *(NDChebyshev a, Array b) => (NDChebyshev)Binary(PyBinOp.Mul, a, b);
        /// <inheritdoc cref="NDPolyBase.operator *(NDPolyBase, string)"/>
        public static NDChebyshev operator *(NDChebyshev a, string b) => (NDChebyshev)Binary(PyBinOp.Mul, a, b);
        /// <inheritdoc cref="NDPolyBase.operator *(NDArray, NDPolyBase)"/>
        public static NDChebyshev operator *(NDArray a, NDChebyshev b) => (NDChebyshev)Binary(PyBinOp.Mul, a, b);
        /// <inheritdoc cref="NDPolyBase.operator *(Array, NDPolyBase)"/>
        public static NDChebyshev operator *(Array a, NDChebyshev b) => (NDChebyshev)Binary(PyBinOp.Mul, a, b);
        /// <inheritdoc cref="NDPolyBase.operator *(object[], NDPolyBase)"/>
        public static NDChebyshev operator *(object[] a, NDChebyshev b) => (NDChebyshev)Binary(PyBinOp.Mul, a, b);
        /// <inheritdoc cref="NDPolyBase.operator *(IList, NDPolyBase)"/>
        public static NDChebyshev operator *(IList a, NDChebyshev b) => (NDChebyshev)Binary(PyBinOp.Mul, a, b);
        /// <inheritdoc cref="NDPolyBase.operator *(ITuple, NDPolyBase)"/>
        public static NDChebyshev operator *(ITuple a, NDChebyshev b) => (NDChebyshev)Binary(PyBinOp.Mul, a, b);
        /// <inheritdoc cref="NDPolyBase.operator *(string, NDPolyBase)"/>
        public static NDChebyshev operator *(string a, NDChebyshev b) => (NDChebyshev)Binary(PyBinOp.Mul, a, b);

        /// <inheritdoc cref="NDPolyBase.operator /(NDPolyBase, object)"/>
        public static NDChebyshev operator /(NDChebyshev a, object b) => (NDChebyshev)Binary(PyBinOp.TrueDiv, a, b);
        /// <inheritdoc cref="NDPolyBase.operator /(NDPolyBase, NDArray)"/>
        public static NDChebyshev operator /(NDChebyshev a, NDArray b) => (NDChebyshev)Binary(PyBinOp.TrueDiv, a, b);
        /// <inheritdoc cref="NDPolyBase.operator /(NDPolyBase, object[])"/>
        public static NDChebyshev operator /(NDChebyshev a, object[] b) => (NDChebyshev)Binary(PyBinOp.TrueDiv, a, b);
        /// <inheritdoc cref="NDPolyBase.operator /(NDPolyBase, Array)"/>
        public static NDChebyshev operator /(NDChebyshev a, Array b) => (NDChebyshev)Binary(PyBinOp.TrueDiv, a, b);
        /// <inheritdoc cref="NDPolyBase.operator /(NDPolyBase, string)"/>
        public static NDChebyshev operator /(NDChebyshev a, string b) => (NDChebyshev)Binary(PyBinOp.TrueDiv, a, b);
        /// <inheritdoc cref="NDPolyBase.operator /(NDArray, NDPolyBase)"/>
        public static NDChebyshev operator /(NDArray a, NDChebyshev b) => (NDChebyshev)Binary(PyBinOp.TrueDiv, a, b);
        /// <inheritdoc cref="NDPolyBase.operator /(Array, NDPolyBase)"/>
        public static NDChebyshev operator /(Array a, NDChebyshev b) => (NDChebyshev)Binary(PyBinOp.TrueDiv, a, b);
        /// <inheritdoc cref="NDPolyBase.operator /(object[], NDPolyBase)"/>
        public static NDChebyshev operator /(object[] a, NDChebyshev b) => (NDChebyshev)Binary(PyBinOp.TrueDiv, a, b);
        /// <inheritdoc cref="NDPolyBase.operator /(IList, NDPolyBase)"/>
        public static NDChebyshev operator /(IList a, NDChebyshev b) => (NDChebyshev)Binary(PyBinOp.TrueDiv, a, b);
        /// <inheritdoc cref="NDPolyBase.operator /(ITuple, NDPolyBase)"/>
        public static NDChebyshev operator /(ITuple a, NDChebyshev b) => (NDChebyshev)Binary(PyBinOp.TrueDiv, a, b);
        /// <inheritdoc cref="NDPolyBase.operator /(string, NDPolyBase)"/>
        public static NDChebyshev operator /(string a, NDChebyshev b) => (NDChebyshev)Binary(PyBinOp.TrueDiv, a, b);

        /// <inheritdoc cref="NDPolyBase.operator %(NDPolyBase, object)"/>
        public static NDChebyshev operator %(NDChebyshev a, object b) => (NDChebyshev)Binary(PyBinOp.Mod, a, b);
        /// <inheritdoc cref="NDPolyBase.operator %(NDPolyBase, NDArray)"/>
        public static NDChebyshev operator %(NDChebyshev a, NDArray b) => (NDChebyshev)Binary(PyBinOp.Mod, a, b);
        /// <inheritdoc cref="NDPolyBase.operator %(NDPolyBase, object[])"/>
        public static NDChebyshev operator %(NDChebyshev a, object[] b) => (NDChebyshev)Binary(PyBinOp.Mod, a, b);
        /// <inheritdoc cref="NDPolyBase.operator %(NDPolyBase, Array)"/>
        public static NDChebyshev operator %(NDChebyshev a, Array b) => (NDChebyshev)Binary(PyBinOp.Mod, a, b);
        /// <inheritdoc cref="NDPolyBase.operator %(NDPolyBase, string)"/>
        public static NDChebyshev operator %(NDChebyshev a, string b) => (NDChebyshev)Binary(PyBinOp.Mod, a, b);
        /// <inheritdoc cref="NDPolyBase.operator %(NDArray, NDPolyBase)"/>
        public static NDChebyshev operator %(NDArray a, NDChebyshev b) => (NDChebyshev)Binary(PyBinOp.Mod, a, b);
        /// <inheritdoc cref="NDPolyBase.operator %(Array, NDPolyBase)"/>
        public static NDChebyshev operator %(Array a, NDChebyshev b) => (NDChebyshev)Binary(PyBinOp.Mod, a, b);
        /// <inheritdoc cref="NDPolyBase.operator %(object[], NDPolyBase)"/>
        public static NDChebyshev operator %(object[] a, NDChebyshev b) => (NDChebyshev)Binary(PyBinOp.Mod, a, b);
        /// <inheritdoc cref="NDPolyBase.operator %(IList, NDPolyBase)"/>
        public static NDChebyshev operator %(IList a, NDChebyshev b) => (NDChebyshev)Binary(PyBinOp.Mod, a, b);
        /// <inheritdoc cref="NDPolyBase.operator %(ITuple, NDPolyBase)"/>
        public static NDChebyshev operator %(ITuple a, NDChebyshev b) => (NDChebyshev)Binary(PyBinOp.Mod, a, b);
        /// <inheritdoc cref="NDPolyBase.operator %(string, NDPolyBase)"/>
        public static NDChebyshev operator %(string a, NDChebyshev b) => (NDChebyshev)Binary(PyBinOp.Mod, a, b);
    }
}
