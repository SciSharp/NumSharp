using System;
using System.Collections;
using NumSharp.Backends;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Strings;

// =============================================================================
// NDPolyBase.cs — numpy.polynomial's ABCPolyBase: the state and the members of the six series classes
// =============================================================================
//
// A port of NumPy 2.4.2's numpy/polynomial/_polybase.py, statement for statement. NumPy's base class holds no
// loops: every member binds a module function of the class's basis (_add = chebadd, _val = chebval, ...) or a
// polyutils helper, plus a little Python logic — and that Python logic (the order of the checks, which operand
// a statement converts, what Python's operator protocol does when a method returns NotImplemented) is what this
// file and its partials reproduce. The module functions are the units U1-U9 (NDPolySeries, NDPolyAlgebra,
// NDPolyCalc, NDPolyEval, NDPolyFit, NDPolyInterpolate), called with the basis of the class.
//
// The partials:
//   NDPolyBase.cs           state, construction (NumPy's __init__), the properties and the plain members;
//   NDPolyBase.Protocol.cs  Python's binary-operator protocol: the forward/reflected methods, NotImplemented and
//                           CPython's fallbacks, the operand classification, the C# operators of the base type;
//   NDPolyBase.Algebra.cs   evaluation AT a series (convert / cast run NumPy's recurrence with series values) and
//                           the class methods' shared cores (identity, basis, fromroots, fit, cast, interpolate);
//   NDPolyBase.Printing.cs  __str__ / __format__ / __repr__ / _repr_latex_.
//
// OWNERSHIP. A series OWNS the arrays it stores: the constructor makes them inside its own NDScope and detaches the
// ones it keeps, so neither that scope nor a caller's reclaims them, and Dispose releases them. A series that is
// never disposed keeps its buffers until the garbage collector finalizes the arrays — dispose every series,
// temporaries included (the NDW analyzer reports a dropped one). A getter returns the stored array itself (NumPy:
// `p.coef is p.coef`); a setter stores a VIEW of the caller's array, so the series and the caller each own one
// array object over the shared buffer (writes through either reach both, as in NumPy, where it is one object).
//
// =============================================================================

namespace NumSharp
{
    /// <summary>
    ///     NumPy's <c>numpy.polynomial._polybase.ABCPolyBase</c>: a polynomial series in one of the six bases — power,
    ///     Chebyshev, Legendre, Laguerre, Hermite (physicists') and HermiteE (probabilists') — with a domain mapped onto a
    ///     window, and the arithmetic, calculus, evaluation, conversion and printing NumPy's series classes have. The
    ///     concrete classes are <see cref="NDPolynomial"/>, <see cref="NDChebyshev"/>, <see cref="NDLegendre"/>,
    ///     <see cref="NDLaguerre"/>, <see cref="NDHermite"/> and <see cref="NDHermiteE"/>.
    /// </summary>
    /// <remarks>
    ///     <para><b>What the domain and window do.</b> A series is evaluated at <c>off + scl*x</c>, the linear map sending the
    ///     domain onto the window (<see cref="mapparms"/>): fitting on <c>[0, 100]</c> with a window of <c>[-1, 1]</c> keeps the
    ///     basis well conditioned. Arithmetic between two series needs the same class, domain, window and symbol
    ///     (NumPy's <c>TypeError: Domains differ</c> and siblings).</para>
    ///     <para><b>Ownership.</b> A series owns its coefficient, domain and window arrays and releases them in
    ///     <see cref="Dispose"/>; until then a getter hands out the stored array itself. Dispose every series, the results
    ///     of operators included (<c>using var r = p * q;</c>): an undisposed one keeps its buffers until the garbage
    ///     collector runs. No <see cref="NDScope"/> reclaims a series (it is not an array).</para>
    ///     <para><b>C# spelling of NumPy's protocol.</b> Every member NumPy names keeps NumPy's name, case included.
    ///     <c>p(x)</c> is <see cref="__call__(object)"/>; <c>p ** n</c>, <c>p // q</c> and <c>divmod(p, q)</c> are the
    ///     concrete classes' <c>__pow__</c>, <c>__floordiv__</c> and <c>__divmod__</c> (C# has no such operators);
    ///     <c>+ - * / %</c>, unary <c>-</c> / <c>+</c> and <c>== !=</c> are operators. NumPy's class attributes
    ///     (<c>Polynomial.domain</c>, <c>window</c>, <c>maxpower</c>, <c>basis_name</c>) are members of the class object
    ///     <see cref="NDPolyClass"/>, reached as <c>NDPolynomial.Class</c> or <see cref="__class__"/>: C# cannot declare a
    ///     static <c>NDPolynomial.domain</c> beside the instance <c>p.domain</c>. The C# protocol members
    ///     (<see cref="Dispose"/>, <see cref="Equals(object)"/>, <see cref="GetHashCode"/>, <see cref="ToString()"/>,
    ///     <see cref="GetEnumerator"/>, the concrete classes' <c>Deconstruct</c>) stand in for Python's.</para>
    ///     <para><b>Not hashable.</b> NumPy sets <c>__hash__ = None</c>: <see cref="GetHashCode"/> raises NumPy's
    ///     <c>TypeError: unhashable type</c>, so a series cannot be a dictionary key — its value can change in place.</para>
    /// </remarks>
    public abstract partial class NDPolyBase : IDisposable, IFormattable
    {
        /// <summary>The class object (basis, names, class attributes): the same one for every series of the class.</summary>
        private readonly NDPolyClass _kind;

        /// <summary>The coefficients, low degree first (NumPy's <c>self.coef</c>): an array this series owns.</summary>
        private NDArray _coef;

        /// <summary>The instance domain, or null to read the class attribute (NumPy: an instance attribute only when the
        ///     constructor got a domain, or after an assignment).</summary>
        private NDArray _domain;

        /// <summary>The instance window, or null to read the class attribute.</summary>
        private NDArray _window;

        /// <summary>The symbol (NumPy's <c>self._symbol</c>), validated by the constructor.</summary>
        private string _symbol;

        /// <summary>An instance <c>maxpower</c> (valid when <see cref="_hasMaxpower"/>; null is NumPy's None).</summary>
        private object _maxpower;

        /// <summary>Whether <see cref="maxpower"/> was assigned on this instance (NumPy's instance attribute).</summary>
        private bool _hasMaxpower;

        /// <summary>The <c>full=True</c> statistics of a <c>fit</c> result, or null (read by the concrete classes' Deconstruct).</summary>
        private FitStatus _fit;

        /// <summary>Set by <see cref="Dispose"/>: the stored arrays are released.</summary>
        private bool _disposed;

        /// <summary>
        ///     The names NumPy's instance <c>__dict__</c> holds, in its insertion order (<c>coef</c>, a constructor domain /
        ///     window, <c>_symbol</c>, then the first assignment of each later attribute), which is the key order of
        ///     <see cref="__getstate__"/>. Replaced wholesale by <see cref="__setstate__"/>.
        /// </summary>
        private readonly System.Collections.Generic.List<string> _stateOrder = new(6);

        /// <summary>
        ///     Entries of a <see cref="__setstate__"/> dictionary this series does not model (NumPy keeps any key in
        ///     <c>__dict__</c>; the <c>symbol</c> key NumPy's own <c>__getstate__</c> writes is one), handed back by
        ///     <see cref="__getstate__"/> at their position; null until a state brings one.
        /// </summary>
        private System.Collections.Generic.Dictionary<string, object> _extraState;

        /// <summary>The four statistics NumPy's <c>fit(..., full=True)</c> returns next to the series.</summary>
        private protected sealed class FitStatus
        {
            /// <summary>Creates the record.</summary>
            /// <param name="residuals">lstsq's residuals.</param><param name="rank">The rank.</param>
            /// <param name="singularValues">The singular values.</param><param name="rcond">The rcond used.</param>
            public FitStatus(NDArray residuals, NDArray rank, NDArray singularValues, NDArray rcond)
            {
                Residuals = residuals;
                Rank = rank;
                SingularValues = singularValues;
                Rcond = rcond;
            }

            /// <summary>The sum of squared residuals per data set.</summary>
            public readonly NDArray Residuals;
            /// <summary>The numerical rank of the scaled design matrix (a 0-d int32).</summary>
            public readonly NDArray Rank;
            /// <summary>The singular values of the scaled design matrix.</summary>
            public readonly NDArray SingularValues;
            /// <summary>The rcond the fit used.</summary>
            public readonly NDArray Rcond;
        }

        /// <summary>
        ///     Creates a blank series of a class: no coefficients yet. Every path fills it in through
        ///     <see cref="Initialize"/>, which runs NumPy's <c>__init__</c>; only the concrete classes call this.
        /// </summary>
        /// <param name="kind">The class object.</param>
        private protected NDPolyBase(NDPolyClass kind)
        {
            _kind = kind;
        }

        /// <summary>
        ///     Python's <c>p.__class__</c> (<c>type(p)</c>): the series' class object — the same object as the class's
        ///     static <c>Class</c> (<c>ReferenceEquals(p.__class__, NDChebyshev.Class)</c> is NumPy's
        ///     <c>p.__class__ is Chebyshev</c>). Its members are NumPy's class attributes, so <c>p.__class__.domain</c> is
        ///     the class default whatever domain the series itself holds; it also converts to the C# type where NumSharp
        ///     takes a class (<c>q.convert(kind: p.__class__)</c>).
        /// </summary>
        /// <remarks>Footgun: assigning through it (<c>p.__class__.maxpower = 5</c>) changes the class, so every series of the
        ///     class without its own value, as in Python — use <see cref="maxpower"/> to change one series.</remarks>
        public NDPolyClass __class__ => _kind;

        /// <summary>The basis of the class's module functions.</summary>
        internal PolyBasis Basis => _kind.Basis;

        // ---------------------------------------------------------------------------------------------
        //  Construction: NumPy's __init__
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     NumPy's <c>__init__(coef, domain, window, symbol)</c>, in its order: <c>as_series([coef], trim=False)</c>
        ///     (the coefficients converted and COPIED), then the domain and the window the same way, each followed by its
        ///     two-element check, then the symbol's identifier test.
        /// </summary>
        /// <param name="coef">The coefficients: anything <c>np.array</c> accepts under the house mapping.</param>
        /// <param name="domain">The domain, or null for the class default (read, never copied, until assigned).</param>
        /// <param name="window">The window, or null for the class default.</param>
        /// <param name="symbol">The symbol: a Python identifier.</param>
        /// <param name="coefIsFresh">The caller asserts <paramref name="coef"/> is a new array nobody else references (a module
        ///     function's result): it is then kept instead of copied when it is already what <c>as_series</c> makes —
        ///     an owning, contiguous 1-D array of the common type — which no caller can observe.</param>
        /// <exception cref="ValueError"><c>Coefficient array is empty</c>, <c>Coefficient array is not 1-d</c>,
        ///     <c>Coefficient arrays have no common type</c> (for the coefficients, then the domain, then the window),
        ///     <c>Domain has wrong number of elements.</c>, <c>Window has wrong number of elements.</c>, <c>Symbol string must
        ///     be a valid Python identifier</c>.</exception>
        /// <exception cref="TypeError"><c>Symbol must be a non-empty string</c> (a null symbol).</exception>
        /// <exception cref="NotSupportedException">An argument NumPy turns into an OBJECT array (None, a sequence holding None
        ///     or a non-numeric object, a Python int past uint64): NumPy keeps an object series, which NumSharp has no dtype
        ///     for.</exception>
        /// <exception cref="InvalidOperationException">The series is already initialized (a programming error).</exception>
        private protected void Initialize(object coef, object domain, object window, string symbol, bool coefIsFresh)
        {
            if (_coef is not null)
                throw new InvalidOperationException("a series is initialized exactly once");
            // Every array made here is this scope's (conversions of a list, the copies); the three the series keeps are
            // detached at the end, so a failure part way reclaims everything built so far.
            using var scope = NDScope.Open();
            NDArray c = AsSeries(coef, coefIsFresh);
            NDArray d = domain is null ? null : AsDomain(domain, "Domain");
            NDArray w = window is null ? null : AsDomain(window, "Window");
            ValidateSymbol(symbol);
            NDScope.Detach(c);
            NDScope.Detach(d);
            NDScope.Detach(w);
            _coef = c;
            _domain = d;
            _window = w;
            _symbol = symbol;
            // NumPy's __init__ assigns self.coef, then self.domain / self.window only when given, then self._symbol: the
            // instance __dict__'s insertion order, which __getstate__ reproduces.
            _stateOrder.Add("coef");
            if (d is not null)
                _stateOrder.Add("domain");
            if (w is not null)
                _stateOrder.Add("window");
            _stateOrder.Add("_symbol");
        }

        /// <summary>
        ///     <c>[a] = pu.as_series([a], trim=False)</c>: <c>np.array(a, ndmin=1)</c>, the empty / not-1-d checks,
        ///     <c>np.common_type</c>, and a copy in that type.
        /// </summary>
        /// <param name="a">The argument.</param>
        /// <param name="fresh">The argument is a new array nobody else references (see <see cref="Initialize"/>).</param>
        /// <returns>The array: <paramref name="a"/> itself when fresh and already as_series' result, else a new copy.</returns>
        /// <exception cref="ValueError">Empty, not 1-d, or no common type.</exception>
        /// <exception cref="NotSupportedException">An object array (see <see cref="Initialize"/>).</exception>
        private static NDArray AsSeries(object a, bool fresh)
        {
            var v = NDPolySeries.AsCoefficientArray(a);
            NDPolySeries.Validate(v);
            NPTypeCode t = NDPolySeries.CommonType(new ReadOnlySpan<PolySeriesView>(in v));
            // A module function's result that already is np.array(a, copy=True, dtype=t) — its own buffer, C-contiguous,
            // starting at element 0 — is kept: the copy NumPy makes of it is unobservable (nothing else holds the original).
            if (fresh && a is NDArray nd && nd.typecode == t && nd.ndim == 1 && nd.Shape.IsContiguous && nd.Shape.offset == 0
                && nd.Shape.OwnsData)
                return nd;
            return NDPolySeries.CopyAs(v, v.Len, t);
        }

        /// <summary>
        ///     The constructor's domain / window statement: <see cref="AsSeries"/>, then <c>len(x) != 2</c> raises
        ///     <c>{what} has wrong number of elements.</c>
        /// </summary>
        /// <param name="a">The argument.</param>
        /// <param name="what"><c>Domain</c> or <c>Window</c>.</param>
        /// <returns>A new two-element array (always a copy: NumPy's as_series copies).</returns>
        /// <exception cref="ValueError">As <see cref="AsSeries"/>, or the wrong length.</exception>
        /// <exception cref="NotSupportedException">An object array.</exception>
        private static NDArray AsDomain(object a, string what)
        {
            NDArray r = AsSeries(a, fresh: false);
            if (r.size != 2)
                throw new ValueError(what + " has wrong number of elements.");
            return r;
        }

        /// <summary>
        ///     NumPy's symbol test: <c>symbol.isidentifier()</c> (Python's rule: an XID_Start character or <c>_</c>, then
        ///     XID_Continue characters, over Unicode code points), and the TypeError it substitutes when the symbol has no
        ///     <c>isidentifier</c> method.
        /// </summary>
        /// <param name="symbol">The symbol.</param>
        /// <exception cref="TypeError"><c>Symbol must be a non-empty string</c> (null).</exception>
        /// <exception cref="ValueError"><c>Symbol string must be a valid Python identifier</c> (empty, a leading digit, a space, ...).</exception>
        private protected static void ValidateSymbol(string symbol)
        {
            if (symbol is null)
                throw new TypeError("Symbol must be a non-empty string");
            if (!PyUnicodeData.IsIdentifier(symbol))
                throw new ValueError("Symbol string must be a valid Python identifier");
        }

        /// <summary>
        ///     NumPy's <c>self.__class__(coef, self.domain, self.window, self.symbol)</c>, the statement that ends every method
        ///     returning a series: a new series of this class whose coefficients are <paramref name="coef"/> (as_series'd),
        ///     with COPIES of this series' domain and window (the class defaults included: the new series owns its own) and
        ///     its symbol.
        /// </summary>
        /// <param name="coef">The coefficients.</param>
        /// <param name="fresh">Whether <paramref name="coef"/> is a module function's new array (see <see cref="Initialize"/>).</param>
        /// <returns>The new series.</returns>
        /// <exception cref="ValueError">The constructor's errors: an invalid result, or a domain / window assigned with a
        ///     length other than two.</exception>
        /// <exception cref="AttributeError">A state set by <see cref="__setstate__"/> without <c>_symbol</c> (NumPy reads
        ///     <c>self.symbol</c> for the new series).</exception>
        private protected NDPolyBase Derive(NDArray coef, bool fresh)
        {
            // NumPy's statement reads self.domain, self.window and self.symbol in that order before calling the class.
            NDArray d = domain, w = window;
            string s = symbol;
            var r = _kind.Blank();
            r.Initialize(coef, d, w, s, fresh);
            return r;
        }

        /// <summary>
        ///     The coefficients for internal use, refusing a disposed series and a state without coefficients.
        /// </summary>
        /// <exception cref="ObjectDisposedException">The series was disposed.</exception>
        /// <exception cref="AttributeError">A <see cref="__setstate__"/> dictionary without <c>coef</c>: NumPy's
        ///     <c>'Polynomial' object has no attribute 'coef'</c>.</exception>
        private protected NDArray Coef => _disposed
            ? throw new ObjectDisposedException(_kind.__name__)
            : _coef ?? throw new AttributeError($"'{_kind.__name__}' object has no attribute 'coef'");

        /// <summary>Appends an attribute to NumPy's <c>__dict__</c> order the first time it is assigned on the instance.</summary>
        /// <param name="name">The attribute (<c>domain</c>, <c>window</c>, <c>maxpower</c>).</param>
        private void NoteAttribute(string name)
        {
            if (!_stateOrder.Contains(name))
                _stateOrder.Add(name);
        }

        // ---------------------------------------------------------------------------------------------
        //  Properties
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     The series coefficients, low degree first (NumPy's <c>coef</c>): the array this series owns, the same object on
        ///     every read. Assigning stores a VIEW of the given array (no check, no copy, as NumPy stores the object):
        ///     writes through either reach both. The replaced array is released by the assignment (see
        ///     <see cref="ReleaseReplaced"/>): copy one read earlier to keep it; assigning the current array back changes
        ///     nothing.
        /// </summary>
        /// <exception cref="ArgumentNullException">Assigning null (NumPy would store None; every later member then fails).</exception>
        /// <exception cref="ObjectDisposedException">The series was disposed.</exception>
        /// <exception cref="AttributeError">Reading a state set by <see cref="__setstate__"/> without <c>coef</c>.</exception>
        public NDArray coef
        {
            get => Coef;
            set
            {
                if (value is null) throw new ArgumentNullException(nameof(value));
                ThrowIfDisposed();
                NoteAttribute("coef");
                if (ReferenceEquals(value, _coef))
                    return;   // NumPy's `p.coef = p.coef`: the attribute keeps its identity
                var replaced = _coef;
                _coef = Owned(value);
                ReleaseReplaced(replaced, _coef);
            }
        }

        /// <summary>
        ///     The domain (NumPy's <c>domain</c>): the instance's own array, or the CLASS attribute (<c>Class.domain</c> of
        ///     the series' class, read at every access and shared by every instance created without a domain — writing into
        ///     it changes the class's, as in NumPy). Assigning stores a view of the given array on this instance and releases
        ///     the instance's previous one (never the class attribute; see <see cref="ReleaseReplaced"/>).
        /// </summary>
        /// <exception cref="ArgumentNullException">Assigning null.</exception>
        /// <exception cref="ObjectDisposedException">The series was disposed.</exception>
        public NDArray domain
        {
            get
            {
                ThrowIfDisposed();
                return _domain ?? _kind.domain;
            }
            set
            {
                if (value is null) throw new ArgumentNullException(nameof(value));
                ThrowIfDisposed();
                NoteAttribute("domain");
                if (ReferenceEquals(value, _domain))
                    return;
                var replaced = _domain;
                _domain = Owned(value);
                ReleaseReplaced(replaced, _domain);
            }
        }

        /// <summary>
        ///     The window (NumPy's <c>window</c>): the instance's own array, or the class attribute (<c>Class.window</c>, read
        ///     at every access). Assigning stores a view of the given array on this instance and releases the instance's
        ///     previous one (never the class attribute; see <see cref="ReleaseReplaced"/>).
        /// </summary>
        /// <exception cref="ArgumentNullException">Assigning null.</exception>
        /// <exception cref="ObjectDisposedException">The series was disposed.</exception>
        public NDArray window
        {
            get
            {
                ThrowIfDisposed();
                return _window ?? _kind.window;
            }
            set
            {
                if (value is null) throw new ArgumentNullException(nameof(value));
                ThrowIfDisposed();
                NoteAttribute("window");
                if (ReferenceEquals(value, _window))
                    return;
                var replaced = _window;
                _window = Owned(value);
                ReleaseReplaced(replaced, _window);
            }
        }

        /// <summary>
        ///     Releases an attribute array this series no longer holds, because an assignment or <see cref="__setstate__"/>
        ///     replaced it. Every instance attribute is the series' own (its constructor's copy, or the view an assignment
        ///     stored; a class default is never one), so it is released at once, as <see cref="poly1d"/> releases replaced
        ///     coefficients, instead of being left to the garbage collector. Footgun: a reference read from the getter before
        ///     the replacement is released with it (NumPy's reference count would keep it alive): copy it first to keep it.
        /// </summary>
        /// <param name="replaced">The array no longer held, or null (none: the attribute was the class default or absent).</param>
        /// <param name="kept">The array now held; a replaced array that is still held is not released.</param>
        private static void ReleaseReplaced(NDArray replaced, NDArray kept)
        {
            if (replaced is not null && !ReferenceEquals(replaced, kept))
                replaced.Dispose();
        }

        /// <summary>The symbol the printers write for the variable (NumPy's read-only <c>symbol</c>, default <c>"x"</c>).</summary>
        /// <exception cref="AttributeError">A state set by <see cref="__setstate__"/> without <c>_symbol</c>: NumPy's
        ///     <c>'Polynomial' object has no attribute '_symbol'</c>.</exception>
        public string symbol => _symbol ?? throw new AttributeError($"'{_kind.__name__}' object has no attribute '_symbol'");

        /// <summary>
        ///     The largest power <c>__pow__</c> allows (NumPy's <c>maxpower</c>): the class attribute (<c>Class.maxpower</c>,
        ///     100 unless the class's was assigned) unless assigned on this instance; null is NumPy's None (no limit). Any
        ///     value Python can compare with an int works — NumPy's <c>power &gt; maxpower</c> decides (a float 2.5 allows 2).
        ///     Assigning here changes this series only.
        /// </summary>
        public object maxpower
        {
            get => _hasMaxpower ? _maxpower : _kind.maxpower;
            set
            {
                _maxpower = value;
                _hasMaxpower = true;
                NoteAttribute("maxpower");
            }
        }

        /// <summary>Refuses a disposed series (the setters' and getters' guard; <see cref="Coef"/> also refuses a missing
        ///     <c>coef</c>).</summary>
        /// <exception cref="ObjectDisposedException">The series was disposed.</exception>
        private void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(_kind.__name__);
        }

        /// <summary>The basis letter the default printers write (NumPy's class attribute <c>basis_name</c>, read through the
        ///     series as in Python): null for the power series, <c>T</c>, <c>P</c>, <c>L</c>, <c>H</c> or <c>He</c>
        ///     otherwise.</summary>
        public string basis_name => _kind.basis_name;

        /// <summary>A view of <paramref name="value"/> that no scope tracks: the object a setter stores.</summary>
        /// <param name="value">The caller's array.</param>
        /// <returns>The view.</returns>
        private static NDArray Owned(NDArray value)
        {
            var v = value.view();
            NDScope.Detach(v);
            return v;
        }

        // ---------------------------------------------------------------------------------------------
        //  Comparison helpers
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     NumPy's <c>has_samecoef</c>: <c>len(self.coef) == len(other.coef) and np.all(self.coef == other.coef)</c> —
        ///     equal lengths, then elementwise <c>==</c> (a NaN coefficient is never the same; <c>-0.0</c> equals <c>0.0</c>;
        ///     a float32 series compares with a float64 one by value).
        /// </summary>
        /// <param name="other">The other series.</param>
        /// <returns>Whether the coefficients match.</returns>
        /// <exception cref="AttributeError"><paramref name="other"/> is null (<c>'NoneType' object has no attribute 'coef'</c>).</exception>
        /// <exception cref="ObjectDisposedException">A series was disposed.</exception>
        public bool has_samecoef(NDPolyBase other)
        {
            if (other is null) throw new AttributeError("'NoneType' object has no attribute 'coef'");
            return PyLen(Coef) == PyLen(other.Coef) && AllEqual(Coef, other.Coef);
        }

        /// <summary>NumPy's <c>has_samedomain</c>: <c>np.all(self.domain == other.domain)</c>.</summary>
        /// <param name="other">The other series.</param>
        /// <returns>Whether the domains match elementwise.</returns>
        /// <exception cref="AttributeError"><paramref name="other"/> is null.</exception>
        /// <exception cref="IncorrectShapeException">Domains whose shapes do not broadcast (an assigned domain).</exception>
        public bool has_samedomain(NDPolyBase other)
        {
            if (other is null) throw new AttributeError("'NoneType' object has no attribute 'domain'");
            return AllEqual(domain, other.domain);
        }

        /// <summary>NumPy's <c>has_samewindow</c>: <c>np.all(self.window == other.window)</c>.</summary>
        /// <param name="other">The other series.</param>
        /// <returns>Whether the windows match elementwise.</returns>
        /// <exception cref="AttributeError"><paramref name="other"/> is null.</exception>
        /// <exception cref="IncorrectShapeException">Windows whose shapes do not broadcast (an assigned window).</exception>
        public bool has_samewindow(NDPolyBase other)
        {
            if (other is null) throw new AttributeError("'NoneType' object has no attribute 'window'");
            return AllEqual(window, other.window);
        }

        /// <summary>NumPy's <c>has_sametype</c>: <c>isinstance(other, self.__class__)</c> — any value; a series of another
        ///     class, a number or null is not.</summary>
        /// <param name="other">The value.</param>
        /// <returns>Whether <paramref name="other"/> is a series of this class.</returns>
        public bool has_sametype(object other) => other is NDPolyBase s && s._kind == _kind;

        /// <summary>
        ///     <c>bool(np.all(a == b))</c>: NumPy's elementwise equality (broadcast; mixed dtypes compared in their promoted
        ///     type; NaN unequal), then <c>all</c>. Two float64 pairs — every class default and every domain as_series makes
        ///     from Python numbers — compare without an array.
        /// </summary>
        /// <param name="a">One array.</param><param name="b">The other.</param>
        /// <returns>Whether every element pair is equal.</returns>
        /// <exception cref="IncorrectShapeException">Shapes that do not broadcast (NumPy's ValueError text).</exception>
        private protected static unsafe bool AllEqual(NDArray a, NDArray b)
        {
            if (a.typecode == NPTypeCode.Double && b.typecode == NPTypeCode.Double && a.ndim == 1 && b.ndim == 1
                && a.size == 2 && b.size == 2)
            {
                double* pa = (double*)a.Storage.Address + a.Shape.offset;
                double* pb = (double*)b.Storage.Address + b.Shape.offset;
                long sa = a.Shape.strides[0], sb = b.Shape.strides[0];
                return pa[0] == pb[0] && pa[sa] == pb[sb];
            }
            using var scope = NDScope.Open();   // the comparison's bool array
            return np.all(a == b);
        }

        /// <summary>Python's <c>len(a)</c> of an array: its first dimension.</summary>
        /// <param name="a">The array.</param>
        /// <returns>The length.</returns>
        /// <exception cref="TypeError"><c>len() of unsized object</c> (a 0-d array).</exception>
        private protected static long PyLen(NDArray a)
        {
            if (a.ndim == 0)
                throw new TypeError("len() of unsized object");
            return a.Shape.dimensions[0];
        }

        // ---------------------------------------------------------------------------------------------
        //  Equality and hashing: __eq__ / __ne__ / __hash__ = None
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     NumPy's <c>__eq__</c>: <c>isinstance(other, self.__class__)</c>, then (short-circuiting, in this order)
        ///     <c>np.all</c> of the domains' and the windows' elementwise equality, equal coefficient shapes,
        ///     <c>np.all(self.coef == other.coef)</c>, and equal symbols. A float32 series equals a float64 one with the same
        ///     values; a NaN coefficient never equals.
        /// </summary>
        /// <param name="obj">Anything (a value that is not a series of this class is unequal).</param>
        /// <returns>Whether the two series are NumPy-equal.</returns>
        /// <exception cref="IncorrectShapeException">Domains or windows whose shapes do not broadcast (assigned ones), as
        ///     NumPy's <c>==</c> raises.</exception>
        /// <exception cref="AttributeError">A state without <c>coef</c> / <c>_symbol</c> (see <see cref="__setstate__"/>),
        ///     when the comparison reaches it.</exception>
        public override bool Equals(object obj)
        {
            if (obj is not NDPolyBase other || other._kind != _kind)
                return false;
            return AllEqual(domain, other.domain)
                   && AllEqual(window, other.window)
                   && SameShape(Coef, other.Coef)
                   && AllEqual(Coef, other.Coef)
                   && symbol == other.symbol;
        }

        /// <summary>
        ///     NumPy's <c>p.__eq__(other)</c>, the method behind <c>==</c>: <see cref="Equals(object)"/>. Spelled out for code
        ///     that calls the protocol method by name; <c>p == other</c> is the same comparison.
        /// </summary>
        /// <param name="other">Anything.</param>
        /// <returns>Whether the two are NumPy-equal (NumPy answers <c>np.False_</c> when a domain, window or coefficient
        ///     comparison decides, a Python bool otherwise; both are this bool).</returns>
        /// <exception cref="IncorrectShapeException">See <see cref="Equals(object)"/>.</exception>
        /// <exception cref="AttributeError">See <see cref="Equals(object)"/>.</exception>
        public bool __eq__(object other) => Equals(other);

        /// <summary>NumPy's <c>p.__ne__(other)</c>: <c>not self.__eq__(other)</c>.</summary>
        /// <param name="other">Anything.</param>
        /// <returns>Whether the two are not NumPy-equal.</returns>
        /// <exception cref="IncorrectShapeException">See <see cref="Equals(object)"/>.</exception>
        /// <exception cref="AttributeError">See <see cref="Equals(object)"/>.</exception>
        public bool __ne__(object other) => !Equals(other);

        /// <summary>Python's <c>a.shape == b.shape</c> (the dimensions only).</summary>
        /// <param name="a">One array.</param><param name="b">The other.</param>
        /// <returns>Whether the dimensions are equal.</returns>
        private static bool SameShape(NDArray a, NDArray b)
        {
            if (a.ndim != b.ndim)
                return false;
            for (int i = 0; i < a.ndim; i++)
                if (a.Shape.dimensions[i] != b.Shape.dimensions[i])
                    return false;
            return true;
        }

        /// <summary>
        ///     NumPy's <c>__hash__ = None</c>: a series is not hashable — its coefficients can change in place, so a hash of
        ///     them would break any dictionary holding it.
        /// </summary>
        /// <returns>Never returns.</returns>
        /// <exception cref="TypeError"><c>unhashable type: 'Polynomial'</c> (the class's NumPy name), always.</exception>
        public override int GetHashCode() => throw new TypeError($"unhashable type: '{_kind.__name__}'");

        /// <summary>NumPy's <c>==</c> (<see cref="Equals(object)"/>); two nulls are equal, null against a series is not.</summary>
        /// <param name="a">One series.</param><param name="b">The other.</param>
        /// <returns>Whether they are NumPy-equal.</returns>
        /// <exception cref="IncorrectShapeException">See <see cref="Equals(object)"/>.</exception>
        public static bool operator ==(NDPolyBase a, NDPolyBase b) => a is null ? b is null : a.Equals(b);

        /// <summary>NumPy's <c>!=</c>: <c>not self.__eq__(other)</c>.</summary>
        /// <param name="a">One series.</param><param name="b">The other.</param>
        /// <returns>Whether they are not NumPy-equal.</returns>
        /// <exception cref="IncorrectShapeException">See <see cref="Equals(object)"/>.</exception>
        public static bool operator !=(NDPolyBase a, NDPolyBase b) => !(a == b);

        /// <summary>NumPy's <c>series == other</c> for a value that is not a series (always false: <c>__eq__</c> answers
        ///     False and the other side's reflected <c>__eq__</c> is never reached).</summary>
        /// <param name="a">The series.</param><param name="b">Anything.</param>
        /// <returns>Whether they are NumPy-equal.</returns>
        public static bool operator ==(NDPolyBase a, object b) => a is null ? b is null : a.Equals(b);

        /// <summary>NumPy's <c>series != other</c>.</summary>
        /// <param name="a">The series.</param><param name="b">Anything.</param>
        /// <returns>Whether they are not NumPy-equal.</returns>
        public static bool operator !=(NDPolyBase a, object b) => !(a == b);

        /// <summary>NumPy's <c>other == series</c>: the other side's <c>__eq__</c> declines, so the series' answers.</summary>
        /// <param name="a">Anything.</param><param name="b">The series.</param>
        /// <returns>Whether they are NumPy-equal.</returns>
        public static bool operator ==(object a, NDPolyBase b) => b is null ? a is null : b.Equals(a);

        /// <summary>NumPy's <c>other != series</c>.</summary>
        /// <param name="a">Anything.</param><param name="b">The series.</param>
        /// <returns>Whether they are not NumPy-equal.</returns>
        public static bool operator !=(object a, NDPolyBase b) => !(a == b);

        // ---------------------------------------------------------------------------------------------
        //  Sequence protocol
        // ---------------------------------------------------------------------------------------------

        /// <summary>NumPy's <c>len(p)</c>: <c>len(self.coef)</c>, the number of coefficients (one more than the degree).</summary>
        /// <returns>The length.</returns>
        /// <exception cref="TypeError">A 0-d coefficient array (assigned): <c>len() of unsized object</c>.</exception>
        public long __len__() => PyLen(Coef);

        /// <summary>NumPy's <c>iter(p)</c>: <c>iter(self.coef)</c> — the coefficients (boxed scalars of their dtype; rows of an
        ///     assigned N-D array).</summary>
        /// <returns>The enumerator.</returns>
        /// <exception cref="TypeError">A 0-d coefficient array: <c>iteration over a 0-d array</c>.</exception>
        public IEnumerator __iter__() => Coef.GetEnumerator();

        /// <summary>
        ///     The <c>foreach</c> pattern over the coefficients (<see cref="__iter__"/>). Deliberately not
        ///     <see cref="IEnumerable"/>: NumPy's array coercion never iterates a series (it has no <c>__getitem__</c>), and an
        ///     <see cref="IEnumerable"/> series would be read as a Python list by every NumSharp function taking one.
        /// </summary>
        /// <returns>The enumerator.</returns>
        /// <exception cref="TypeError">A 0-d coefficient array.</exception>
        public IEnumerator GetEnumerator() => __iter__();

        // ---------------------------------------------------------------------------------------------
        //  Plain members
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     NumPy's <c>copy()</c>: <c>self.__class__(self.coef, self.domain, self.window, self.symbol)</c> — a new series
        ///     with copies of the coefficients, the domain and the window (a class default is copied too: the copy owns its
        ///     own).
        /// </summary>
        /// <returns>The copy.</returns>
        /// <exception cref="ValueError">An assigned domain or window of the wrong length (the constructor's check).</exception>
        public virtual NDPolyBase copy()
        {
            using var scope = NDScope.Open();
            return Derive(Coef, fresh: false);
        }

        /// <summary>NumPy's <c>degree()</c>: <c>len(self) - 1</c>. Trailing zero coefficients count (see <see cref="trim"/>).</summary>
        /// <returns>The degree.</returns>
        /// <exception cref="TypeError">A 0-d coefficient array.</exception>
        public long degree() => __len__() - 1;

        /// <summary>
        ///     NumPy's <c>cutdeg(deg)</c>: <c>self.truncate(deg + 1)</c> — the series cut to degree <paramref name="deg"/>,
        ///     with Python's <c>+</c> on whatever <paramref name="deg"/> is (a float 1.0 works, 1.5 fails truncate's integer
        ///     check, a str fails the <c>+</c>).
        /// </summary>
        /// <param name="deg">The degree: a non-negative integer (anything whose <c>+ 1</c> truncate accepts).</param>
        /// <returns>The new series.</returns>
        /// <exception cref="ValueError"><c>size must be a positive integer</c>.</exception>
        /// <exception cref="TypeError">Python's <c>+</c> refuses <paramref name="deg"/> (<c>can only concatenate str (not
        ///     "int") to str</c>, <c>unsupported operand type(s) for +: 'NoneType' and 'int'</c>), or <c>int()</c> refuses the
        ///     sum (an array of one or more dims).</exception>
        public virtual NDPolyBase cutdeg(object deg)
        {
            using var scope = NDScope.Open();
            return truncate(PlusOne(deg));
        }

        /// <summary>
        ///     NumPy's <c>trim(tol=0)</c>: <c>pu.trimcoef(self.coef, tol)</c> in a new series — the trailing coefficients whose
        ///     magnitude is at most <paramref name="tol"/> removed (all of them leaves one zero).
        /// </summary>
        /// <param name="tol">The tolerance: omitted for NumPy's default 0, or a Python / NumPy number (see
        ///     <c>np.polynomial.polyutils.trimcoef</c>). An explicit null is NumPy's <c>tol=None</c>, which trimcoef's
        ///     <c>tol &lt; 0</c> refuses.</param>
        /// <returns>The new series.</returns>
        /// <exception cref="ValueError"><c>tol must be non-negative</c>.</exception>
        /// <exception cref="TypeError">A tolerance that does not compare with 0 (complex, str, None).</exception>
        public virtual NDPolyBase trim([System.Runtime.InteropServices.Optional] object tol)
        {
            using var scope = NDScope.Open();
            // An omitted argument arrives as Missing (the [Optional] parameter's marker), an explicit null as None.
            NDArray c = tol is System.Reflection.Missing
                ? np.polynomial.polyutils.trimcoef(Coef)
                : np.polynomial.polyutils.trimcoef(Coef, tol);
            return Derive(c, fresh: true);
        }

        /// <summary>
        ///     NumPy's <c>truncate(size)</c>: <c>isize = int(size)</c>; <c>isize != size or isize &lt; 1</c> raises; then the
        ///     first <c>isize</c> coefficients (all of them when there are fewer) in a new series.
        /// </summary>
        /// <param name="size">The length: a positive integer (Python's <c>int()</c> and <c>!=</c> decide: 2.0 works, 2.5
        ///     and '2' do not).</param>
        /// <returns>The new series.</returns>
        /// <exception cref="ValueError"><c>size must be a positive integer</c>; or int()'s own ValueError (a NaN, an
        ///     unparsable str — a parsable one fails the <c>!=</c>).</exception>
        /// <exception cref="TypeError">int() refuses <paramref name="size"/> (None, complex, a list, an array of one or more
        ///     dims).</exception>
        /// <exception cref="OverflowException">An infinity.</exception>
        public virtual NDPolyBase truncate(object size)
        {
            using var scope = NDScope.Open();
            // int(size) and `isize != size` are the Python statements {p}pow reads its power with: the same semantics for
            // every C# value (PolyPowerArgument's header), NumPy scalars included.
            var isize = PyIntOf(size, out bool differs);
            if (differs || isize < 1)
                throw new ValueError("size must be a positive integer");
            NDArray c = Coef;
            if (isize < PyLen(c))
                c = c[new Slice(0, (long)isize)];
            return Derive(c, fresh: false);
        }

        /// <summary>
        ///     Python's <c>i = int(v)</c> and <c>i != v</c>, the statements <c>truncate</c> and <c>basis</c> read their integer
        ///     argument with: <see cref="PolyPowerArgument"/>'s port for every C# value, and for a NumPy complex SCALAR
        ///     NumPy's <c>np.complexfloating.__int__</c> — the real part (NumPy warns that the imaginary part is discarded),
        ///     after which <c>!=</c> compares that int with the complex (unequal unless the imaginary part is zero). An explicit
        ///     0-d complex ARRAY is not a scalar: int() refuses it (<c>int() argument must be ... not 'complex'</c>).
        /// </summary>
        /// <param name="v">The argument.</param>
        /// <param name="differs">Whether <c>int(v) != v</c>.</param>
        /// <returns><c>int(v)</c>.</returns>
        /// <exception cref="ValueError">int()'s ValueErrors (a NaN, an unparsable str).</exception>
        /// <exception cref="OverflowException">An infinity.</exception>
        /// <exception cref="TypeError">int() refuses <paramref name="v"/>.</exception>
        private static System.Numerics.BigInteger PyIntOf(object v, out bool differs)
        {
            var op = Operand(v);
            if (op.Kind == PyOperandKind.NumPyNumber && op.Number.Dtype == NPTypeCode.Complex)
            {
                var z = (System.Numerics.Complex)op.Number.ToObject();
                var i = PolyPowerArgument.Int(z.Real);
                differs = z.Imaginary != 0 || PolyPowerArgument.DiffersFromInt(i, z.Real);
                return i;
            }
            // Any other NumPy scalar converts as the C# number of its dtype does; everything else as it is.
            object arg = op.Kind is PyOperandKind.NumPyNumber or PyOperandKind.NumPyBool ? op.Number.ToObject() : v;
            var r = PolyPowerArgument.Int(arg);
            differs = PolyPowerArgument.DiffersFromInt(r, arg);
            return r;
        }

        /// <summary>
        ///     Python's <c>deg + 1</c> for <see cref="cutdeg"/>: CPython / NumPy arithmetic with the Python int 1 for a number,
        ///     the ufunc for an array, CPython's TypeError for anything else.
        /// </summary>
        /// <param name="deg">The operand.</param>
        /// <returns>The sum: a boxed number, a <see cref="PolyNumber"/> for a NumPy scalar, or an array.</returns>
        /// <exception cref="TypeError">A str, list, tuple, None or other object.</exception>
        private static object PlusOne(object deg)
        {
            var op = Operand(deg);
            switch (op.Kind)
            {
                case PyOperandKind.PyInt:
                case PyOperandKind.PyFloat:
                case PyOperandKind.PyComplex:
                case PyOperandKind.PyBool:
                case PyOperandKind.NumPyNumber:
                case PyOperandKind.NumPyBool:
                case PyOperandKind.NdArray:
                {
                    var r = PolyNumber.Binary(BinaryOp.Add, op.Number, PolyNumber.FromPython(PyScalar.Int(1)));
                    // A NumPy scalar sum stays a NumPy scalar (boxed): truncate's int() of an np.complex128 takes its real
                    // part, where a C# Complex would be read as a Python complex, which int() refuses.
                    return r.Kind switch
                    {
                        PolyNumberKind.Array => r.Array,
                        PolyNumberKind.Scalar => (object)r,
                        _ => r.ToObject(),
                    };
                }
                case PyOperandKind.Str:
                    throw new TypeError("can only concatenate str (not \"int\") to str");
                case PyOperandKind.List:
                    throw new TypeError("can only concatenate list (not \"int\") to list");
                case PyOperandKind.Tuple:
                    throw new TypeError("can only concatenate tuple (not \"int\") to tuple");
                default:
                    throw new TypeError($"unsupported operand type(s) for +: '{op.TypeName}' and 'int'");
            }
        }

        /// <summary>
        ///     NumPy's <c>mapparms()</c>: <c>pu.mapparms(self.domain, self.window)</c> — the offset and scale of the map
        ///     <c>off + scl*x</c> sending the domain onto the window (identity when they are equal).
        /// </summary>
        /// <returns><c>(off, scl)</c>, each a boxed C# number of the NumPy scalar's dtype (<c>double</c> for float64,
        ///     <see cref="System.Numerics.Complex"/> for complex128, <c>float</c> for a float32 domain and window, ...).</returns>
        public (object off, object scl) mapparms()
        {
            using var scope = NDScope.Open();
            var (off, scl) = MapParmsOf(domain, window);
            (object off, object scl) r = (off.ToObject(), scl.ToObject());
            scope.Returns((System.Runtime.CompilerServices.ITuple)r);   // an assigned 2-D domain makes array components
            return r;
        }

        /// <summary><c>pu.mapparms(old, new)</c> as NumPy values (normally NumPy scalars: the domains are arrays).</summary>
        /// <param name="old">The source interval.</param><param name="new">The target interval.</param>
        /// <returns>The offset and the scale.</returns>
        private protected static (PolyNumber off, PolyNumber scl) MapParmsOf(NDArray old, NDArray @new)
            => NDPolySeries.MapParms(old, @new);

        /// <summary>
        ///     NumPy's <c>integ(m=1, k=[], lbnd=None)</c>: the definite integral in a new series. NumPy integrates in the
        ///     WINDOW variable, so the lower bound is mapped there (<c>lbnd = off + scl*lbnd</c>, a NumPy scalar; 0 for None)
        ///     and every order is scaled by <c>1./scl</c> — then <c>{p}int(self.coef, m, k, lbnd, 1./scl)</c>.
        /// </summary>
        /// <param name="m">The number of integrations (≥ 0).</param>
        /// <param name="k">The integration constants (null for NumPy's <c>[]</c>): a scalar or a sequence, at most
        ///     <paramref name="m"/> of them, missing ones 0.</param>
        /// <param name="lbnd">The lower bound in the DOMAIN variable: null for NumPy's None (the window's 0), or a scalar.</param>
        /// <returns>The new series (the same domain and window).</returns>
        /// <exception cref="ValueError"><c>The order of integration must be non-negative</c>, <c>Too many integration
        ///     constants</c>, <c>lbnd must be a scalar.</c> (an array bound).</exception>
        /// <exception cref="TypeError">A sequence or str bound: <c>can't multiply sequence by non-int of type
        ///     'numpy.float64'</c> (the scale's NumPy type); another object NumPy's object loop refuses.</exception>
        public virtual NDPolyBase integ(int m = 1, object k = null, object lbnd = null)
        {
            using var scope = NDScope.Open();
            var (off, scl) = MapParmsOf(domain, window);
            object lb = lbnd is null ? null : (object)MapScalar(off, scl, lbnd);
            var inv = PolyNumber.Binary(BinaryOp.Divide, PolyNumber.FromPython(PyScalar.Float(1.0)), scl);
            NDArray c = NDPolyCalc.Int(Basis, Coef, m, k, lb, inv, 0);
            return Derive(c, fresh: true);
        }

        /// <summary>
        ///     <c>off + scl * v</c> for a value <paramref name="v"/> integ maps (NumPy's two Python operators): scalarmath for a
        ///     number, the ufunc for an array; a Python sequence or str meets the NumPy scalar's sequence refusal.
        /// </summary>
        /// <param name="off">The offset (a NumPy scalar).</param><param name="scl">The scale (a NumPy scalar).</param>
        /// <param name="v">The value.</param>
        /// <returns>The mapped value (a NumPy scalar, or an array for an array value).</returns>
        /// <exception cref="TypeError">A list / tuple / str (<c>can't multiply sequence by non-int of type 'numpy.float64'</c>),
        ///     or an object NumPy's object loop refuses (<c>unsupported operand type(s) for *: 'float' and '…'</c>).</exception>
        /// <exception cref="NotSupportedException">A series bound (NumPy evaluates a series at a series there, then fails to
        ///     store it into a float array; NumSharp does not follow that path).</exception>
        private static PolyNumber MapScalar(in PolyNumber off, in PolyNumber scl, object v)
        {
            var op = Operand(v);
            switch (op.Kind)
            {
                case PyOperandKind.List:
                case PyOperandKind.Tuple:
                case PyOperandKind.Str:
                    // np.float32/np.float64/np.complex128 * a sequence: NumPy's scalar declines, the sequence's repeat
                    // needs an index — which an inexact NumPy scalar has none of.
                    throw new TypeError($"can't multiply sequence by non-int of type '{NumPyScalarTypeName(scl)}'");
                case PyOperandKind.Series:
                    throw new NotSupportedException("a series lower bound makes NumPy evaluate the series at a series and store the result into the coefficient array, which fails in NumPy; NumSharp does not follow it");
                case PyOperandKind.None:
                case PyOperandKind.Other:
                    throw new TypeError($"unsupported operand type(s) for *: '{ObjectLoopTypeName(scl)}' and '{op.TypeName}'");
            }
            return PolyNumber.Binary(BinaryOp.Add, off, PolyNumber.Binary(BinaryOp.Multiply, scl, op.Number));
        }

        /// <summary>
        ///     NumPy's <c>deriv(m=1)</c>: <c>{p}der(self.coef, m, scl)</c> in a new series, with the map's scale as the chain
        ///     rule's factor (the derivative is taken in the window variable).
        /// </summary>
        /// <param name="m">The order (≥ 0).</param>
        /// <returns>The new series (the same domain and window).</returns>
        /// <exception cref="ValueError"><c>The order of derivation must be non-negative</c>.</exception>
        public virtual NDPolyBase deriv(int m = 1)
        {
            using var scope = NDScope.Open();
            var (_, scl) = MapParmsOf(domain, window);
            NDArray c = NDPolyCalc.Der(Basis, Coef, m, scl, 0);
            return Derive(c, fresh: true);
        }

        /// <summary>
        ///     NumPy's <c>roots()</c>: <c>{p}roots(self.coef)</c> — the companion matrix's eigenvalues, in the window
        ///     variable — mapped back to the domain (<c>pu.mapdomain(roots, self.window, self.domain)</c>).
        /// </summary>
        /// <returns>The roots in the domain variable (float64, or complex128 when they are complex).</returns>
        /// <exception cref="MissingBackendException">A series of degree ≥ 2 without a LAPACK backend (the eigenvalues need
        ///     <c>NumSharp.Interop.OpenBLAS</c>).</exception>
        public NDArray roots()
        {
            using var scope = NDScope.Open();
            NDArray r = NDPolyAlgebra.Roots(Basis, Coef);
            var (off, scl) = MapParmsOf(window, domain);
            // Complex roots of a float32 series are NumPy's complex64 (eigvals of the float32 companion), carried in
            // complex128: a float32 domain / window maps them back in NumPy's complex64 loops.
            bool complex64 = Coef.typecode == NPTypeCode.Single && r.typecode == NPTypeCode.Complex;
            return scope.Returns(NDPolySeries.MapDomainWith(PolyNumber.FromArray(r), off, scl, complex64).ToNDArray());
        }

        /// <summary>
        ///     NumPy's <c>linspace(n=100, domain=None)</c>: <c>x = np.linspace(domain[0], domain[1], n)</c> and
        ///     <c>y = self(x)</c> — the series sampled at <paramref name="n"/> equally spaced points of the domain (a plotting
        ///     aid).
        /// </summary>
        /// <param name="n">The number of points (≥ 0).</param>
        /// <param name="domain">The interval: null for the series' domain, or anything indexable at 0 and 1.</param>
        /// <returns><c>(x, y)</c>: x in the endpoints' type (float64 for Python numbers and float64 domains, float32 for a
        ///     float32 domain, complex128 for a complex one), y the values.</returns>
        /// <exception cref="ValueError"><c>Number of samples, -1, must be non-negative.</c></exception>
        /// <exception cref="IndexError">An interval shorter than two.</exception>
        public (NDArray x, NDArray y) linspace(long n = 100, object domain = null)
        {
            using var scope = NDScope.Open();
            object d = domain ?? this.domain;
            var start = NDPolySeries.Index(d, 0);
            var stop = NDPolySeries.Index(d, 1);
            NDArray x = Linspace(start, stop, n);
            NDArray y = __call__(x);
            return scope.Returns((x, y));
        }

        /// <summary>
        ///     NumPy's <c>p(x)</c> (<c>__call__</c>): <c>arg = pu.mapdomain(x, self.domain, self.window)</c>, then
        ///     <c>{p}val(arg, self.coef)</c>. A scalar point maps to a NumPy SCALAR (<c>off + scl*x</c>), at which NumPy's
        ///     recurrence runs in scalarmath; an array of points is evaluated by the evaluation kernel.
        /// </summary>
        /// <param name="arg">The point(s): a number (a C# number is a Python scalar; a bool becomes a 0-d bool array, as
        ///     NumPy's mapdomain converts it), an array, or a Python list / tuple. A series argument binds
        ///     <see cref="__call__{TSeries}(TSeries)"/>.</param>
        /// <returns>The values: an array for array points, a 0-d array (NumPy's scalar) for a scalar point.</returns>
        /// <exception cref="ArgumentException">A series passed through an <c>object</c> reference (its result is a series:
        ///     call <see cref="__call__{TSeries}(TSeries)"/> with the series typed as its class).</exception>
        /// <exception cref="TypeError">None or another object NumPy's object loop refuses (<c>unsupported operand type(s) for
        ///     *: 'float' and 'NoneType'</c>); a str (the multiply ufunc's no-loop error).</exception>
        /// <exception cref="ValueError">A ragged list.</exception>
        public NDArray __call__(object arg)
        {
            using var scope = NDScope.Open();
            var op = Operand(arg);
            if (op.Kind == PyOperandKind.Series)
                throw new ArgumentException("a series argument evaluates to a series: call __call__<TSeries>(TSeries) with the series typed as its class", nameof(arg));
            var (off, scl) = MapParmsOf(domain, window);
            PolyNumber px;
            switch (op.Kind)
            {
                case PyOperandKind.None:
                case PyOperandKind.Other:
                    // np.asanyarray makes a 0-d object array; the object loop multiplies scl's Python value by it.
                    throw new TypeError($"unsupported operand type(s) for *: '{ObjectLoopTypeName(scl)}' and '{op.TypeName}'");
                case PyOperandKind.PyBool:
                    // type(True) is bool, not in (int, float, complex): np.asanyarray makes a 0-d bool ARRAY of it.
                    px = PolyNumber.FromArray(NDArray.Scalar((bool)op.Arg));
                    break;
                case PyOperandKind.List:
                case PyOperandKind.Tuple:
                    px = PolyNumber.FromArray(Backends.UnmanagedStorage.FromSequence(op.Arg));
                    break;
                case PyOperandKind.Str:
                    px = PolyNumber.FromArray(np.asanyarray(op.Arg));
                    break;
                default:
                    px = op.Number;
                    break;
            }
            var mapped = NDPolySeries.MapDomainWith(px, off, scl);
            return scope.Returns(ValueAt(mapped));
        }

        /// <summary>
        ///     NumPy's <c>p(series)</c>: the series evaluated AT another series — <c>pu.mapdomain</c> runs NumPy's object loop
        ///     (<c>scl</c>'s Python value times the series, then the NumPy-scalar offset plus it) and <c>{p}val</c> runs its
        ///     recurrence with series values through the class operators (see <c>NDPolyBase.Algebra.cs</c>). This is how
        ///     <c>convert</c> and <c>cast</c> change basis.
        /// </summary>
        /// <typeparam name="TSeries">The argument's class; the result is of that class.</typeparam>
        /// <param name="arg">The series to evaluate at.</param>
        /// <returns>A new series of <typeparamref name="TSeries"/>'s class, in <paramref name="arg"/>'s domain and window.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="arg"/> is null.</exception>
        /// <exception cref="TypeError"><c>Domains differ</c> / <c>Windows differ</c> when the recurrence combines two series
        ///     whose domain or window holds a NaN (NumPy's equality check of the domains fails).</exception>
        public TSeries __call__<TSeries>(TSeries arg) where TSeries : NDPolyBase
        {
            if (arg is null) throw new ArgumentNullException(nameof(arg));
            using var scope = NDScope.Open();
            return (TSeries)EvaluateAtSeries(arg);
        }

        /// <summary>
        ///     <c>{p}val(mapped, self.coef)</c> for a mapped point that is not a series: the kernel for an array of rank ≥ 1;
        ///     NumPy's scalarmath recurrence for a NumPy scalar (a 0-d result of the map is one: NumPy unwraps a 0-d ufunc
        ///     result) with a 1-D series, the kernel's 0-d mode for an assigned N-D series (every operation then meets an
        ///     array, a ufunc either way).
        /// </summary>
        /// <param name="mapped">The mapped point(s).</param>
        /// <returns>The values (a 0-d array for a scalar).</returns>
        private NDArray ValueAt(in PolyNumber mapped)
        {
            if (mapped.IsNdArray)
                return NDPolyEval.Val(Basis, mapped.Array, Coef, tensor: true);
            var s = mapped.Kind == PolyNumberKind.Array ? PolyNumber.ScalarOf(mapped.Array) : mapped;
            if (Coef.ndim == 1)
                return NDPolyEval.ValAtScalar(Basis, s, Coef).ToNDArray();
            return NDPolyEval.Val(Basis, s.ToNDArray(), Coef, tensor: true);
        }

        /// <summary>
        ///     NumPy's <c>convert(domain=None, kind=None, window=None)</c>: this series as a series of class
        ///     <paramref name="kind"/> on <paramref name="domain"/> / <paramref name="window"/> — <c>self(kind.identity(domain,
        ///     window=window, symbol=self.symbol))</c>, NumPy's recurrence run at the target class's identity series.
        /// </summary>
        /// <param name="domain">The target domain: null for <paramref name="kind"/>'s class attribute <c>domain</c>.</param>
        /// <param name="kind">The target class: <c>typeof(NDChebyshev)</c>, or its class object (<c>NDChebyshev.Class</c>,
        ///     <c>q.__class__</c>), which converts to the type; null for this series' class.</param>
        /// <param name="window">The target window: null for <paramref name="kind"/>'s class attribute <c>window</c>.</param>
        /// <returns>A new series of <paramref name="kind"/>.</returns>
        /// <exception cref="AttributeError"><paramref name="kind"/> is not one of the six series classes.</exception>
        /// <exception cref="ValueError">A target domain / window the constructor refuses.</exception>
        /// <exception cref="TypeError"><c>Domains differ</c> / <c>Windows differ</c>: a target domain or window holding a NaN.</exception>
        /// <remarks>Conversion between bases can be numerically ill conditioned (NumPy's note): a high-degree series in a
        ///     wide domain loses digits.</remarks>
        public NDPolyBase convert(object domain = null, Type kind = null, object window = null)
        {
            using var scope = NDScope.Open();
            var k = kind is null ? _kind : NDPolyClass.Of(kind);
            object d = domain ?? k.domain;
            object w = window ?? k.window;
            using var ident = Identity(k, d, w, symbol);
            return EvaluateAtSeries(ident);
        }

        // ---------------------------------------------------------------------------------------------
        //  Pickling protocol: __getstate__ / __setstate__
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     NumPy's <c>__getstate__()</c> (what <c>pickle</c> and <c>copy.deepcopy</c> save): <c>self.__dict__.copy()</c> with
        ///     <c>coef</c>, <c>domain</c> and <c>window</c> replaced by COPIES (the class defaults included) and <c>symbol</c>
        ///     added — in NumPy's key order (the instance attributes in their assignment order, then whichever of
        ///     <c>domain</c> / <c>window</c> was not an instance attribute, then <c>symbol</c>).
        /// </summary>
        /// <returns>
        ///     A new insertion-ordered dictionary: <c>coef</c>, <c>domain</c>, <c>window</c> as new arrays the CALLER owns,
        ///     <c>_symbol</c> and <c>symbol</c> as strings, <c>maxpower</c> when assigned on the instance, and any entry an
        ///     earlier <see cref="__setstate__"/> brought (the same objects: NumPy's copy is shallow).
        /// </returns>
        /// <exception cref="ObjectDisposedException">The series was disposed.</exception>
        /// <exception cref="AttributeError">A state without <c>coef</c> or <c>_symbol</c> (NumPy reads <c>self.coef</c> and
        ///     <c>self.symbol</c>).</exception>
        public System.Collections.Specialized.OrderedDictionary __getstate__()
        {
            ThrowIfDisposed();
            var ret = new System.Collections.Specialized.OrderedDictionary(_stateOrder.Count + 3);
            // ret = self.__dict__.copy(): the attributes in __dict__ order (their values are replaced below where NumPy
            // replaces them, keeping each key at its place).
            foreach (string name in _stateOrder)
            {
                ret[name] = name switch
                {
                    "coef" => _coef,
                    "domain" => _domain,
                    "window" => _window,
                    "_symbol" => _symbol,
                    "maxpower" => _maxpower,
                    _ => _extraState[name],
                };
            }
            // ret['coef'] = self.coef.copy() ... ret['symbol'] = self.symbol, in NumPy's order; the copies are made last so
            // a missing attribute raises before any array is allocated.
            NDArray c = Coef;
            string s = symbol;
            NDArray d = domain, w = window;
            ret["coef"] = Detached(c.copy());
            ret["domain"] = Detached(d.copy());
            ret["window"] = Detached(w.copy());
            ret["symbol"] = s;
            return ret;
        }

        /// <summary>
        ///     NumPy's <c>__setstate__(dict)</c> (<c>self.__dict__ = dict</c>, what <c>pickle</c> restores): this series' state
        ///     becomes exactly the entries of <paramref name="state"/> — <c>coef</c>, <c>domain</c>, <c>window</c>,
        ///     <c>_symbol</c>, <c>maxpower</c> — and an attribute the dictionary lacks is gone: no <c>coef</c> or
        ///     <c>_symbol</c> makes the members that read it raise NumPy's AttributeError; no <c>domain</c> / <c>window</c> /
        ///     <c>maxpower</c> falls back to the class attribute.
        /// </summary>
        /// <param name="state">The state: <see cref="__getstate__"/>'s dictionary, or any dictionary with string keys. The
        ///     arrays are stored as VIEWS (as the property setters store them: writes through either reach both); entries this
        ///     series does not model (NumPy keeps any attribute) are kept and returned by <see cref="__getstate__"/>.</param>
        /// <exception cref="ArgumentNullException"><paramref name="state"/> is null.</exception>
        /// <exception cref="ObjectDisposedException">The series was disposed.</exception>
        /// <exception cref="TypeError">A key that is not a string, a <c>coef</c> / <c>domain</c> / <c>window</c> that is not an
        ///     <see cref="NDArray"/>, or a <c>_symbol</c> that is not a string. NumPy stores any of these as they are (and fails
        ///     later, in the member that reads them); NumSharp's series holds them typed, so it refuses a value it cannot hold
        ///     up front — a documented divergence.</exception>
        /// <remarks>The arrays the series held before are released, as an assignment releases them
        ///     (<see cref="ReleaseReplaced"/>), except one the state hands back itself, which is kept as it is.</remarks>
        public void __setstate__(System.Collections.IDictionary state)
        {
            if (state is null) throw new ArgumentNullException(nameof(state));
            ThrowIfDisposed();
            // Read and check everything first, so a refused state leaves this series unchanged.
            var order = new System.Collections.Generic.List<string>(state.Count);
            NDArray c = null, d = null, w = null;
            string s = null;
            object mp = null;
            bool hasMp = false;
            System.Collections.Generic.Dictionary<string, object> extra = null;
            foreach (System.Collections.DictionaryEntry e in state)
            {
                if (e.Key is not string key)
                    throw new TypeError($"a series' state keys must be str, not '{NDPolySeries.PythonTypeName(e.Key)}'");
                order.Add(key);
                switch (key)
                {
                    case "coef":
                        c = e.Value as NDArray ?? throw new TypeError($"a series' coef must be an NDArray, not '{NDPolySeries.PythonTypeName(e.Value)}'");
                        break;
                    case "domain":
                        d = e.Value as NDArray ?? throw new TypeError($"a series' domain must be an NDArray, not '{NDPolySeries.PythonTypeName(e.Value)}'");
                        break;
                    case "window":
                        w = e.Value as NDArray ?? throw new TypeError($"a series' window must be an NDArray, not '{NDPolySeries.PythonTypeName(e.Value)}'");
                        break;
                    case "_symbol":
                        s = e.Value as string ?? throw new TypeError($"a series' _symbol must be a str, not '{NDPolySeries.PythonTypeName(e.Value)}'");
                        break;
                    case "maxpower":
                        mp = e.Value;
                        hasMp = true;
                        break;
                    default:
                        (extra ??= new System.Collections.Generic.Dictionary<string, object>())[key] = e.Value;
                        break;
                }
            }
            // An array the state hands back itself (the series' current one) is kept as it is; every other is stored as a
            // view, and the arrays the series no longer holds are released.
            NDArray oldCoef = _coef, oldDomain = _domain, oldWindow = _window;
            _coef = c is null ? null : ReferenceEquals(c, oldCoef) ? oldCoef : Owned(c);
            _domain = d is null ? null : ReferenceEquals(d, oldDomain) ? oldDomain : Owned(d);
            _window = w is null ? null : ReferenceEquals(w, oldWindow) ? oldWindow : Owned(w);
            ReleaseReplaced(oldCoef, _coef);
            ReleaseReplaced(oldDomain, _domain);
            ReleaseReplaced(oldWindow, _window);
            _symbol = s;
            _maxpower = mp;
            _hasMaxpower = hasMp;
            _extraState = extra;
            _stateOrder.Clear();
            _stateOrder.AddRange(order);
        }

        /// <summary>An array no scope tracks (a value the caller of <see cref="__getstate__"/> owns).</summary>
        /// <param name="a">The array.</param>
        /// <returns><paramref name="a"/>.</returns>
        private static NDArray Detached(NDArray a)
        {
            NDScope.Detach(a);
            return a;
        }

        // ---------------------------------------------------------------------------------------------
        //  Disposal
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     Releases the arrays this series owns: the coefficients, an instance domain / window, and the statistics of a
        ///     <c>full=True</c> fit. Idempotent. A class default is never released (the class owns it). After disposal every
        ///     member but <see cref="Dispose"/> raises <see cref="ObjectDisposedException"/>, and an array obtained from a getter
        ///     earlier is released too — copy it first to keep it.
        /// </summary>
        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            _coef?.Dispose();
            _domain?.Dispose();
            _window?.Dispose();
            _coef = _domain = _window = null;
            _fit = null;   // the statistics are the caller's once handed out (see the concrete classes' Deconstruct)
        }

        /// <summary>Attaches a fit's <c>full=True</c> statistics (the concrete classes' Deconstruct hands them out).</summary>
        /// <param name="status">The statistics.</param>
        private protected void AttachFit(FitStatus status) => _fit = status;

        /// <summary>The statistics of a <c>full=True</c> fit, or null for any other series.</summary>
        private protected FitStatus Fit => _fit;
    }
}
