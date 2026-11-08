using System;
using System.Collections;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NumSharp.Backends.Kernels;

// =============================================================================
// NDLaguerre.cs — numpy.polynomial.laguerre.Laguerre
// =============================================================================
//
// A thin class over NDPolyBase (the port of NumPy's ABCPolyBase): its class object (Class: basis Laguerre, NumPy's names, the
// class attributes), its constructors, the class methods typed with it, the covariant overrides of the members that
// return a series, and the C# operators declared with the class itself. Every body delegates to NDPolyBase.
// GENERATED from one template for the six classes; keep them identical but for the record.
//
// =============================================================================

namespace NumSharp
{
    /// <summary>
    ///     NumPy's <c>numpy.polynomial.laguerre.Laguerre</c>: a Laguerre series: <c>c[0]·L₀(x) + c[1]·L₁(x) + …</c>, with a domain (default <c>[0, 1]</c>) mapped onto a
    ///     window (default <c>[0, 1]</c>). Arithmetic (<c>+ - * / %</c>, <see cref="__floordiv__"/>,
    ///     <see cref="__divmod__"/>, <see cref="__pow__"/>), evaluation (<see cref="NDPolyBase.__call__(object)"/>), calculus
    ///     (<see cref="integ"/>, <see cref="deriv"/>), roots, fitting (<see cref="fit"/>), conversion between bases
    ///     (<see cref="NDPolyBase.convert"/>, <see cref="cast"/>) and printing, all as NumPy 2.4.2 has them.
    /// </summary>
    /// <remarks>
    ///     <para>See <see cref="NDPolyBase"/> for the ownership rule (dispose every series, operator results included) and
    ///     the C# spelling of NumPy's protocol. NumPy's class object <c>Laguerre</c> is <see cref="Class"/> (also every
    ///     series' <see cref="NDPolyBase.__class__"/>): the class attributes <c>domain</c>, <c>window</c>, <c>maxpower</c>
    ///     and <c>basis_name</c> keep NumPy's names there. The class methods (<see cref="identity"/>,
    ///     <see cref="basis"/>, <see cref="fromroots"/>, <see cref="fit"/>, <see cref="cast"/>) are static members.</para>
    ///     <para>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.laguerre.Laguerre.html</para>
    /// </remarks>
    public sealed class NDLaguerre : NDPolyBase
    {
        /// <summary>
        ///     NumPy's class object <c>Laguerre</c>: its class attributes under NumPy's names — <c>NDLaguerre.Class.domain</c> is
        ///     NumPy's <c>Laguerre.domain</c>, likewise <c>window</c>, <c>maxpower</c> and <c>basis_name</c> — and its
        ///     <c>__name__</c> / <c>__module__</c>. The same object as <see cref="NDPolyBase.__class__"/> of every Laguerre
        ///     series. Footgun: its attributes are process-wide state, read by every series of the class without its own
        ///     (see <see cref="NDPolyClass"/>).
        /// </summary>
        public static NDPolyClass Class { get; } = new NDPolyClass(PolyBasis.Laguerre, "Laguerre", "numpy.polynomial.laguerre", "L",
            typeof(NDLaguerre), 0.0, 1.0, () => new NDLaguerre());

        /// <summary>
        ///     NumPy's <c>Laguerre(coef, domain=None, window=None, symbol='x')</c>: the series with coefficients
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
        public NDLaguerre(object coef, object domain = null, object window = null, string symbol = "x") : base(Class)
        {
            Initialize(coef, domain, window, symbol, coefIsFresh: false);
        }

        /// <summary>A blank instance for <see cref="NDPolyClass.Blank"/>: NDPolyBase fills it in, exactly once.</summary>
        private NDLaguerre() : base(Class)
        {
        }

        // ---------------------------------------------------------------------------------------------
        //  Members returning this class (covariant overrides of NDPolyBase's)
        // ---------------------------------------------------------------------------------------------

        /// <inheritdoc cref="NDPolyBase.copy"/>
        public override NDLaguerre copy() => (NDLaguerre)base.copy();

        /// <inheritdoc cref="NDPolyBase.cutdeg"/>
        public override NDLaguerre cutdeg(object deg) => (NDLaguerre)base.cutdeg(deg);

        /// <inheritdoc cref="NDPolyBase.trim"/>
        public override NDLaguerre trim([Optional] object tol) => (NDLaguerre)base.trim(tol);

        /// <inheritdoc cref="NDPolyBase.truncate"/>
        public override NDLaguerre truncate(object size) => (NDLaguerre)base.truncate(size);

        /// <inheritdoc cref="NDPolyBase.integ"/>
        public override NDLaguerre integ(int m = 1, object k = null, object lbnd = null) => (NDLaguerre)base.integ(m, k, lbnd);

        /// <inheritdoc cref="NDPolyBase.deriv"/>
        public override NDLaguerre deriv(int m = 1) => (NDLaguerre)base.deriv(m);

        /// <inheritdoc cref="NDPolyBase.__pow__"/>
        public override NDLaguerre __pow__(object power) => (NDLaguerre)base.__pow__(power);

        /// <inheritdoc cref="NDPolyBase.__floordiv__"/>
        public override NDLaguerre __floordiv__(object other) => (NDLaguerre)base.__floordiv__(other);

        /// <inheritdoc cref="NDPolyBase.__rfloordiv__"/>
        public override NDLaguerre __rfloordiv__(object other) => (NDLaguerre)base.__rfloordiv__(other);

        /// <inheritdoc cref="NDPolyBase.__divmod__"/>
        public new (NDLaguerre quo, NDLaguerre rem) __divmod__(object other)
        {
            var (q, r) = base.__divmod__(other);
            return ((NDLaguerre)q, (NDLaguerre)r);
        }

        /// <inheritdoc cref="NDPolyBase.__rdivmod__"/>
        public new (NDLaguerre quo, NDLaguerre rem) __rdivmod__(object other)
        {
            var (q, r) = base.__rdivmod__(other);
            return ((NDLaguerre)q, (NDLaguerre)r);
        }

        // ---------------------------------------------------------------------------------------------
        //  Class methods
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     NumPy's <c>Laguerre.identity(domain=None, window=None, symbol='x')</c>: the series whose value at every x is x (in
        ///     this basis, mapped through the domain and window).
        /// </summary>
        /// <param name="domain">The domain, or null for the class default.</param>
        /// <param name="window">The window, or null for the class default.</param>
        /// <param name="symbol">The symbol.</param>
        /// <returns>The identity series.</returns>
        /// <exception cref="ValueError">A domain / window the constructor refuses; an invalid symbol.</exception>
        /// <exception cref="IndexError">A domain or window shorter than two.</exception>
        public static NDLaguerre identity(object domain = null, object window = null, string symbol = "x")
            => (NDLaguerre)Identity(Class, domain, window, symbol);

        /// <summary>
        ///     NumPy's <c>Laguerre.basis(deg, domain=None, window=None, symbol='x')</c>: the basis polynomial of degree
        ///     <paramref name="deg"/> (coefficients <c>[0, …, 0, 1]</c>).
        /// </summary>
        /// <param name="deg">The degree: a non-negative integer (Python's <c>int()</c> and <c>!=</c> decide).</param>
        /// <param name="domain">The domain, or null for the class default.</param>
        /// <param name="window">The window, or null for the class default.</param>
        /// <param name="symbol">The symbol.</param>
        /// <returns>The basis series.</returns>
        /// <exception cref="ValueError"><c>deg must be non-negative integer</c>.</exception>
        /// <exception cref="TypeError"><c>int()</c> refuses <paramref name="deg"/>.</exception>
        public static NDLaguerre basis(object deg, object domain = null, object window = null, string symbol = "x")
            => (NDLaguerre)BasisOf(Class, deg, domain, window, symbol);

        /// <summary>
        ///     NumPy's <c>Laguerre.fromroots(roots, domain=[], window=None, symbol='x')</c>: the series
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
        public static NDLaguerre fromroots(object roots, [Optional] object domain, object window = null, string symbol = "x")
            => (NDLaguerre)FromRootsOf(Class, roots, domain, window, symbol);

        /// <summary>
        ///     NumPy's <c>Laguerre.fit(x, y, deg, domain=None, rcond=None, full=False, w=None, window=None, symbol='x')</c>: the
        ///     least-squares fit of this basis to the points, on a domain covering x by default (mapped onto the window, which
        ///     keeps the problem well conditioned).
        /// </summary>
        /// <param name="x">The sample points (1-D).</param>
        /// <param name="y">The values (1-D, or 2-D with one data set per column).</param>
        /// <param name="deg">The degree, or a 1-D list of the degrees to fit.</param>
        /// <param name="domain">null: the smallest interval holding x (widened by 1 when x is constant); an empty list: the
        ///     class default; otherwise the domain.</param>
        /// <param name="rcond">The relative cutoff of small singular values, or null for <c>len(x)·eps</c>.</param>
        /// <param name="full">Also keep lstsq's statistics: <c>var (s, (resid, rank, sv, rcond)) = NDLaguerre.fit(…, full: true)</c>.</param>
        /// <param name="w">Weights (1-D), or null.</param>
        /// <param name="window">The window, or null for the class default.</param>
        /// <param name="symbol">The symbol.</param>
        /// <returns>The fitted series (deconstructible into the series and the statistics when <paramref name="full"/>).</returns>
        /// <exception cref="ValueError"><c>expected deg &gt;= 0</c> and the fit's other checks; the constructor's errors.</exception>
        /// <exception cref="TypeError">The fit's argument errors (<c>expected 1D vector for x</c>, …).</exception>
        /// <exception cref="MissingBackendException">No LAPACK backend: lstsq needs <c>NumSharp.Interop.OpenBLAS</c>.</exception>
        public static NDLaguerre fit(object x, object y, object deg, object domain = null, object rcond = null, bool full = false,
            object w = null, object window = null, string symbol = "x")
            => (NDLaguerre)FitOf(Class, x, y, deg, domain, rcond, full, w, window, symbol);

        /// <summary>
        ///     NumPy's <c>Laguerre.cast(series, domain=None, window=None)</c>: <paramref name="series"/> converted to this class
        ///     (<c>series.convert(domain, Laguerre, window)</c>).
        /// </summary>
        /// <param name="series">The series to convert (any of the six classes).</param>
        /// <param name="domain">The domain, or null for the class default.</param>
        /// <param name="window">The window, or null for the class default.</param>
        /// <returns>The converted series.</returns>
        /// <exception cref="AttributeError"><paramref name="series"/> is null.</exception>
        public static NDLaguerre cast(NDPolyBase series, object domain = null, object window = null)
            => (NDLaguerre)CastOf(Class, series, domain, window);

        /// <summary>
        ///     NumPy's <c>series, [resid, rank, sv, rcond] = Laguerre.fit(..., full=True)</c>: the fitted series and lstsq's
        ///     statistics (the residual sums of squares, the rank, the singular values, the rcond used).
        /// </summary>
        /// <param name="series">This series.</param>
        /// <param name="status">The statistics, in NumPy's order. The caller owns them (no scope tracks them; the garbage
        ///     collector releases them).</param>
        /// <exception cref="ValueError">The series is not the result of a <c>fit(..., full: true)</c> (NumPy's series does not
        ///     unpack either).</exception>
        public void Deconstruct(out NDLaguerre series, out (NDArray residuals, NDArray rank, NDArray singular_values, NDArray rcond) status)
        {
            var f = Fit ?? throw new ValueError("the series is not the result of fit(..., full=True): there are no statistics to unpack");
            series = this;
            status = (f.Residuals, f.Rank, f.SingularValues, f.Rcond);
        }

        // ---------------------------------------------------------------------------------------------
        //  Operators (NumPy's protocol, NDPolyBase.Protocol.cs), typed with this class
        // ---------------------------------------------------------------------------------------------

        /// <inheritdoc cref="NDPolyBase.operator -(NDPolyBase)"/>
        public static NDLaguerre operator -(NDLaguerre a) => (NDLaguerre)a.Negative();

        /// <inheritdoc cref="NDPolyBase.operator +(NDPolyBase)"/>
        public static NDLaguerre operator +(NDLaguerre a) => a;

        /// <inheritdoc cref="NDPolyBase.operator +(NDPolyBase, object)"/>
        public static NDLaguerre operator +(NDLaguerre a, object b) => (NDLaguerre)Binary(PyBinOp.Add, a, b);
        /// <inheritdoc cref="NDPolyBase.operator +(NDPolyBase, NDArray)"/>
        public static NDLaguerre operator +(NDLaguerre a, NDArray b) => (NDLaguerre)Binary(PyBinOp.Add, a, b);
        /// <inheritdoc cref="NDPolyBase.operator +(NDPolyBase, object[])"/>
        public static NDLaguerre operator +(NDLaguerre a, object[] b) => (NDLaguerre)Binary(PyBinOp.Add, a, b);
        /// <inheritdoc cref="NDPolyBase.operator +(NDPolyBase, Array)"/>
        public static NDLaguerre operator +(NDLaguerre a, Array b) => (NDLaguerre)Binary(PyBinOp.Add, a, b);
        /// <inheritdoc cref="NDPolyBase.operator +(NDPolyBase, string)"/>
        public static NDLaguerre operator +(NDLaguerre a, string b) => (NDLaguerre)Binary(PyBinOp.Add, a, b);
        /// <inheritdoc cref="NDPolyBase.operator +(NDArray, NDPolyBase)"/>
        public static NDLaguerre operator +(NDArray a, NDLaguerre b) => (NDLaguerre)Binary(PyBinOp.Add, a, b);
        /// <inheritdoc cref="NDPolyBase.operator +(Array, NDPolyBase)"/>
        public static NDLaguerre operator +(Array a, NDLaguerre b) => (NDLaguerre)Binary(PyBinOp.Add, a, b);
        /// <inheritdoc cref="NDPolyBase.operator +(object[], NDPolyBase)"/>
        public static NDLaguerre operator +(object[] a, NDLaguerre b) => (NDLaguerre)Binary(PyBinOp.Add, a, b);
        /// <inheritdoc cref="NDPolyBase.operator +(IList, NDPolyBase)"/>
        public static NDLaguerre operator +(IList a, NDLaguerre b) => (NDLaguerre)Binary(PyBinOp.Add, a, b);
        /// <inheritdoc cref="NDPolyBase.operator +(ITuple, NDPolyBase)"/>
        public static NDLaguerre operator +(ITuple a, NDLaguerre b) => (NDLaguerre)Binary(PyBinOp.Add, a, b);
        /// <inheritdoc cref="NDPolyBase.operator +(string, NDPolyBase)"/>
        public static NDLaguerre operator +(string a, NDLaguerre b) => (NDLaguerre)Binary(PyBinOp.Add, a, b);

        /// <inheritdoc cref="NDPolyBase.operator -(NDPolyBase, object)"/>
        public static NDLaguerre operator -(NDLaguerre a, object b) => (NDLaguerre)Binary(PyBinOp.Sub, a, b);
        /// <inheritdoc cref="NDPolyBase.operator -(NDPolyBase, NDArray)"/>
        public static NDLaguerre operator -(NDLaguerre a, NDArray b) => (NDLaguerre)Binary(PyBinOp.Sub, a, b);
        /// <inheritdoc cref="NDPolyBase.operator -(NDPolyBase, object[])"/>
        public static NDLaguerre operator -(NDLaguerre a, object[] b) => (NDLaguerre)Binary(PyBinOp.Sub, a, b);
        /// <inheritdoc cref="NDPolyBase.operator -(NDPolyBase, Array)"/>
        public static NDLaguerre operator -(NDLaguerre a, Array b) => (NDLaguerre)Binary(PyBinOp.Sub, a, b);
        /// <inheritdoc cref="NDPolyBase.operator -(NDPolyBase, string)"/>
        public static NDLaguerre operator -(NDLaguerre a, string b) => (NDLaguerre)Binary(PyBinOp.Sub, a, b);
        /// <inheritdoc cref="NDPolyBase.operator -(NDArray, NDPolyBase)"/>
        public static NDLaguerre operator -(NDArray a, NDLaguerre b) => (NDLaguerre)Binary(PyBinOp.Sub, a, b);
        /// <inheritdoc cref="NDPolyBase.operator -(Array, NDPolyBase)"/>
        public static NDLaguerre operator -(Array a, NDLaguerre b) => (NDLaguerre)Binary(PyBinOp.Sub, a, b);
        /// <inheritdoc cref="NDPolyBase.operator -(object[], NDPolyBase)"/>
        public static NDLaguerre operator -(object[] a, NDLaguerre b) => (NDLaguerre)Binary(PyBinOp.Sub, a, b);
        /// <inheritdoc cref="NDPolyBase.operator -(IList, NDPolyBase)"/>
        public static NDLaguerre operator -(IList a, NDLaguerre b) => (NDLaguerre)Binary(PyBinOp.Sub, a, b);
        /// <inheritdoc cref="NDPolyBase.operator -(ITuple, NDPolyBase)"/>
        public static NDLaguerre operator -(ITuple a, NDLaguerre b) => (NDLaguerre)Binary(PyBinOp.Sub, a, b);
        /// <inheritdoc cref="NDPolyBase.operator -(string, NDPolyBase)"/>
        public static NDLaguerre operator -(string a, NDLaguerre b) => (NDLaguerre)Binary(PyBinOp.Sub, a, b);

        /// <inheritdoc cref="NDPolyBase.operator *(NDPolyBase, object)"/>
        public static NDLaguerre operator *(NDLaguerre a, object b) => (NDLaguerre)Binary(PyBinOp.Mul, a, b);
        /// <inheritdoc cref="NDPolyBase.operator *(NDPolyBase, NDArray)"/>
        public static NDLaguerre operator *(NDLaguerre a, NDArray b) => (NDLaguerre)Binary(PyBinOp.Mul, a, b);
        /// <inheritdoc cref="NDPolyBase.operator *(NDPolyBase, object[])"/>
        public static NDLaguerre operator *(NDLaguerre a, object[] b) => (NDLaguerre)Binary(PyBinOp.Mul, a, b);
        /// <inheritdoc cref="NDPolyBase.operator *(NDPolyBase, Array)"/>
        public static NDLaguerre operator *(NDLaguerre a, Array b) => (NDLaguerre)Binary(PyBinOp.Mul, a, b);
        /// <inheritdoc cref="NDPolyBase.operator *(NDPolyBase, string)"/>
        public static NDLaguerre operator *(NDLaguerre a, string b) => (NDLaguerre)Binary(PyBinOp.Mul, a, b);
        /// <inheritdoc cref="NDPolyBase.operator *(NDArray, NDPolyBase)"/>
        public static NDLaguerre operator *(NDArray a, NDLaguerre b) => (NDLaguerre)Binary(PyBinOp.Mul, a, b);
        /// <inheritdoc cref="NDPolyBase.operator *(Array, NDPolyBase)"/>
        public static NDLaguerre operator *(Array a, NDLaguerre b) => (NDLaguerre)Binary(PyBinOp.Mul, a, b);
        /// <inheritdoc cref="NDPolyBase.operator *(object[], NDPolyBase)"/>
        public static NDLaguerre operator *(object[] a, NDLaguerre b) => (NDLaguerre)Binary(PyBinOp.Mul, a, b);
        /// <inheritdoc cref="NDPolyBase.operator *(IList, NDPolyBase)"/>
        public static NDLaguerre operator *(IList a, NDLaguerre b) => (NDLaguerre)Binary(PyBinOp.Mul, a, b);
        /// <inheritdoc cref="NDPolyBase.operator *(ITuple, NDPolyBase)"/>
        public static NDLaguerre operator *(ITuple a, NDLaguerre b) => (NDLaguerre)Binary(PyBinOp.Mul, a, b);
        /// <inheritdoc cref="NDPolyBase.operator *(string, NDPolyBase)"/>
        public static NDLaguerre operator *(string a, NDLaguerre b) => (NDLaguerre)Binary(PyBinOp.Mul, a, b);

        /// <inheritdoc cref="NDPolyBase.operator /(NDPolyBase, object)"/>
        public static NDLaguerre operator /(NDLaguerre a, object b) => (NDLaguerre)Binary(PyBinOp.TrueDiv, a, b);
        /// <inheritdoc cref="NDPolyBase.operator /(NDPolyBase, NDArray)"/>
        public static NDLaguerre operator /(NDLaguerre a, NDArray b) => (NDLaguerre)Binary(PyBinOp.TrueDiv, a, b);
        /// <inheritdoc cref="NDPolyBase.operator /(NDPolyBase, object[])"/>
        public static NDLaguerre operator /(NDLaguerre a, object[] b) => (NDLaguerre)Binary(PyBinOp.TrueDiv, a, b);
        /// <inheritdoc cref="NDPolyBase.operator /(NDPolyBase, Array)"/>
        public static NDLaguerre operator /(NDLaguerre a, Array b) => (NDLaguerre)Binary(PyBinOp.TrueDiv, a, b);
        /// <inheritdoc cref="NDPolyBase.operator /(NDPolyBase, string)"/>
        public static NDLaguerre operator /(NDLaguerre a, string b) => (NDLaguerre)Binary(PyBinOp.TrueDiv, a, b);
        /// <inheritdoc cref="NDPolyBase.operator /(NDArray, NDPolyBase)"/>
        public static NDLaguerre operator /(NDArray a, NDLaguerre b) => (NDLaguerre)Binary(PyBinOp.TrueDiv, a, b);
        /// <inheritdoc cref="NDPolyBase.operator /(Array, NDPolyBase)"/>
        public static NDLaguerre operator /(Array a, NDLaguerre b) => (NDLaguerre)Binary(PyBinOp.TrueDiv, a, b);
        /// <inheritdoc cref="NDPolyBase.operator /(object[], NDPolyBase)"/>
        public static NDLaguerre operator /(object[] a, NDLaguerre b) => (NDLaguerre)Binary(PyBinOp.TrueDiv, a, b);
        /// <inheritdoc cref="NDPolyBase.operator /(IList, NDPolyBase)"/>
        public static NDLaguerre operator /(IList a, NDLaguerre b) => (NDLaguerre)Binary(PyBinOp.TrueDiv, a, b);
        /// <inheritdoc cref="NDPolyBase.operator /(ITuple, NDPolyBase)"/>
        public static NDLaguerre operator /(ITuple a, NDLaguerre b) => (NDLaguerre)Binary(PyBinOp.TrueDiv, a, b);
        /// <inheritdoc cref="NDPolyBase.operator /(string, NDPolyBase)"/>
        public static NDLaguerre operator /(string a, NDLaguerre b) => (NDLaguerre)Binary(PyBinOp.TrueDiv, a, b);

        /// <inheritdoc cref="NDPolyBase.operator %(NDPolyBase, object)"/>
        public static NDLaguerre operator %(NDLaguerre a, object b) => (NDLaguerre)Binary(PyBinOp.Mod, a, b);
        /// <inheritdoc cref="NDPolyBase.operator %(NDPolyBase, NDArray)"/>
        public static NDLaguerre operator %(NDLaguerre a, NDArray b) => (NDLaguerre)Binary(PyBinOp.Mod, a, b);
        /// <inheritdoc cref="NDPolyBase.operator %(NDPolyBase, object[])"/>
        public static NDLaguerre operator %(NDLaguerre a, object[] b) => (NDLaguerre)Binary(PyBinOp.Mod, a, b);
        /// <inheritdoc cref="NDPolyBase.operator %(NDPolyBase, Array)"/>
        public static NDLaguerre operator %(NDLaguerre a, Array b) => (NDLaguerre)Binary(PyBinOp.Mod, a, b);
        /// <inheritdoc cref="NDPolyBase.operator %(NDPolyBase, string)"/>
        public static NDLaguerre operator %(NDLaguerre a, string b) => (NDLaguerre)Binary(PyBinOp.Mod, a, b);
        /// <inheritdoc cref="NDPolyBase.operator %(NDArray, NDPolyBase)"/>
        public static NDLaguerre operator %(NDArray a, NDLaguerre b) => (NDLaguerre)Binary(PyBinOp.Mod, a, b);
        /// <inheritdoc cref="NDPolyBase.operator %(Array, NDPolyBase)"/>
        public static NDLaguerre operator %(Array a, NDLaguerre b) => (NDLaguerre)Binary(PyBinOp.Mod, a, b);
        /// <inheritdoc cref="NDPolyBase.operator %(object[], NDPolyBase)"/>
        public static NDLaguerre operator %(object[] a, NDLaguerre b) => (NDLaguerre)Binary(PyBinOp.Mod, a, b);
        /// <inheritdoc cref="NDPolyBase.operator %(IList, NDPolyBase)"/>
        public static NDLaguerre operator %(IList a, NDLaguerre b) => (NDLaguerre)Binary(PyBinOp.Mod, a, b);
        /// <inheritdoc cref="NDPolyBase.operator %(ITuple, NDPolyBase)"/>
        public static NDLaguerre operator %(ITuple a, NDLaguerre b) => (NDLaguerre)Binary(PyBinOp.Mod, a, b);
        /// <inheritdoc cref="NDPolyBase.operator %(string, NDPolyBase)"/>
        public static NDLaguerre operator %(string a, NDLaguerre b) => (NDLaguerre)Binary(PyBinOp.Mod, a, b);
    }
}
