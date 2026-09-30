using NumSharp.Backends.Kernels;

namespace NumSharp
{
    /// <summary>
    ///     <c>numpy.polynomial.polynomial</c> — power series <c>p(x) = c[0] + c[1]*x + … + c[n]*x**n</c>, reachable
    ///     as <c>np.polynomial.polynomial</c>. Coefficients are LOW degree first.
    /// </summary>
    /// <remarks>
    ///     <para><b>Not the legacy functions.</b> <c>np.polynomial.polynomial.polyval(x, c)</c> takes the point first
    ///     and the coefficients low→high; the legacy <see cref="np.polyval(NDArray, NDArray)"/> takes
    ///     <c>(p, x)</c> with <c>p</c> high→low. Same name, different function.</para>
    ///     <para>https://numpy.org/doc/stable/reference/routines.polynomials.polynomial.html</para>
    /// </remarks>
    [ModuleName("np.polynomial.polynomial")]
    public class PowerSeriesModule
    {
        /// <summary>Constructs the module; there is one shared instance behind <c>np.polynomial.polynomial</c>.</summary>
        internal PowerSeriesModule() { }

        /// <summary>
        ///     Evaluates the power series <c>c[0] + c[1]*x + … + c[n]*x**n</c> at <paramref name="x"/> by NumPy's
        ///     Horner loop (<c>c0 = c[-1] + x*0; c0 = c[-i] + c0*x</c>), bit-identical to NumPy 2.4.2 — including
        ///     <c>0*inf = nan</c> from the <c>x*0</c> seed and every intermediate's NEP 50 dtype.
        /// </summary>
        /// <param name="x">
        ///     The points. An <see cref="NDArray"/> (any layout) or a C# array is an array; a C# <c>bool</c>, integer,
        ///     <c>float</c>, <c>double</c> or <see cref="System.Numerics.Complex"/> is a <b>Python scalar</b> — it adopts
        ///     the coefficients' dtype (<c>polyval(2.0, float32_c)</c> is float32) and a 1-D series then yields a 0-d
        ///     result. <c>char</c>/<see cref="System.Half"/>/<c>decimal</c> are strong 0-d arrays.
        /// </param>
        /// <param name="c">Coefficients (anything <c>np.array</c> accepts — an <see cref="NDArray"/> of any layout, a typed C# array, a Python tuple / list such as a ValueTuple, <c>object[]</c> or a jagged / <c>NDArray[]</c> array, nested to any depth, or a scalar), low degree first; for an N-D array the series run along axis 0 and the other
        ///     axes index independent series. Integer and bool coefficients are evaluated as float64.</param>
        /// <param name="tensor">When true (the default) and x is an array, every series is evaluated at every point
        ///     and the result has shape <c>c.shape[1:] + x.shape</c>; when false x is broadcast over the series
        ///     (<c>c.shape[1:]</c> against <c>x.shape</c>). Irrelevant for a 1-D series or a scalar x.</param>
        /// <returns>The values, of NumPy's result dtype (float64 for integer coefficients; the promotion of x and c
        ///     otherwise, float32 staying float32 against a Python-scalar x).</returns>
        /// <exception cref="System.ArgumentNullException"><paramref name="x"/> is null.</exception>
        /// <exception cref="ValueError">A ragged tuple / list <paramref name="c"/> or <paramref name="x"/> (np.array's
        ///     inhomogeneous-shape text; c is converted first, so its error wins).</exception>
        /// <exception cref="System.NotSupportedException">A null or str <paramref name="c"/>, or a tuple / list holding a
        ///     str or null (NumPy builds an object / str array, dtypes NumSharp does not have).</exception>
        /// <exception cref="IndexError"><paramref name="c"/> is empty: <c>index -1 is out of bounds for axis 0 with size 0</c>.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="tensor"/> is false and the shapes do not broadcast
        ///     (NumPy's ValueError text).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.polyval.html</remarks>
        public NDArray polyval(object x, object c, bool tensor = true) => NDPolyEval.Val(PolyBasis.Power, x, c, tensor);

        /// <summary>
        ///     Evaluates the 2-D power series <c>Σ c[i,j] * x**i * y**j</c> at the point pairs <c>(x, y)</c>.
        ///     NumPy's <c>_valnd</c>: <c>polyval(x, c)</c> then <c>polyval(y, ·, tensor=False)</c> — its exact
        ///     intermediate, so the bits match.
        /// </summary>
        /// <param name="x">First coordinates: an <see cref="NDArray"/>, a typed C# array, a Python tuple / list (np.array's coercion) or a
        ///     scalar, converted before anything else (NumPy's <c>np.asanyarray</c>) — so a scalar is a STRONG 0-d array
        ///     (a Python int an int64 one).</param>
        /// <param name="y">Second coordinates (converted like <paramref name="x"/>); must have x's shape.</param>
        /// <param name="c">Coefficients (anything <c>np.array</c> accepts — an <see cref="NDArray"/> of any layout, a typed C# array, a Python tuple / list such as a ValueTuple, <c>object[]</c> or a jagged / <c>NDArray[]</c> array, nested to any depth, or a scalar); <c>c[i,j]</c> is the coefficient of <c>x**i * y**j</c>. Extra trailing axes
        ///     index independent series (they lead the result's shape).</param>
        /// <returns>The values, shape <c>c.shape[2:] + x.shape</c>.</returns>
        /// <exception cref="System.ArgumentNullException">A coordinate (or <c>pts</c>) is null; a null c is
        ///     <see cref="System.NotSupportedException"/> (NumPy's object array).</exception>
        /// <exception cref="ValueError">x and y differ in shape (<c>x, y are incompatible</c>), or a tuple / list argument
        ///     is ragged (np.array's inhomogeneous-shape text: the coordinates' before the shape check, c's after
        ///     it).</exception>
        /// <exception cref="System.NotSupportedException">A null or str <paramref name="c"/>, a str coordinate, or a
        ///     tuple / list holding one (NumPy's object / str arrays).</exception>
        /// <exception cref="IndexError"><paramref name="c"/> is empty along axis 0.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.polyval2d.html</remarks>
        public NDArray polyval2d(object x, object y, object c) => NDPolyEval.ValNd(PolyBasis.Power, c, new[] { x, y });

        /// <summary>
        ///     Evaluates the 3-D power series <c>Σ c[i,j,k] * x**i * y**j * z**k</c> at the point triples
        ///     <c>(x, y, z)</c> (NumPy's <c>_valnd</c>: one tensor pass, then two per-point passes).
        /// </summary>
        /// <param name="x">First coordinates: an <see cref="NDArray"/>, a typed C# array, a Python tuple / list (np.array's coercion) or a
        ///     scalar, converted before anything else (NumPy's <c>np.asanyarray</c>) — so a scalar is a STRONG 0-d array
        ///     (a Python int an int64 one).</param>
        /// <param name="y">Second coordinates (converted like <paramref name="x"/>; x's shape).</param>
        /// <param name="z">Third coordinates (converted like <paramref name="x"/>; x's shape).</param>
        /// <param name="c">Coefficients (anything <c>np.array</c> accepts — an <see cref="NDArray"/> of any layout, a typed C# array, a Python tuple / list such as a ValueTuple, <c>object[]</c> or a jagged / <c>NDArray[]</c> array, nested to any depth, or a scalar); <c>c[i,j,k]</c> multiplies <c>x**i * y**j * z**k</c>.</param>
        /// <returns>The values, shape <c>c.shape[3:] + x.shape</c>.</returns>
        /// <exception cref="System.ArgumentNullException">A coordinate (or <c>pts</c>) is null; a null c is
        ///     <see cref="System.NotSupportedException"/> (NumPy's object array).</exception>
        /// <exception cref="ValueError">The coordinates differ in shape (<c>x, y, z are incompatible</c>), or a tuple /
        ///     list argument is ragged (the coordinates' before the shape check, c's after it).</exception>
        /// <exception cref="System.NotSupportedException">A null or str <paramref name="c"/>, a str coordinate, or a
        ///     tuple / list holding one (NumPy's object / str arrays).</exception>
        /// <exception cref="IndexError"><paramref name="c"/> is empty along axis 0.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.polyval3d.html</remarks>
        public NDArray polyval3d(object x, object y, object z, object c) => NDPolyEval.ValNd(PolyBasis.Power, c, new[] { x, y, z });

        /// <summary>
        ///     Evaluates the 2-D power series on the Cartesian product of <paramref name="x"/> and <paramref name="y"/>
        ///     (NumPy's <c>_gridnd</c>: two tensor passes). Unlike <see cref="polyval2d"/> the coordinates are NOT
        ///     converted first, so a Python-scalar coordinate stays weak.
        /// </summary>
        /// <param name="x">First-axis points (array or Python scalar, as in <see cref="polyval"/>).</param>
        /// <param name="y">Second-axis points.</param>
        /// <param name="c">Coefficients (anything <c>np.array</c> accepts — an <see cref="NDArray"/> of any layout, a typed C# array, a Python tuple / list such as a ValueTuple, <c>object[]</c> or a jagged / <c>NDArray[]</c> array, nested to any depth, or a scalar); <c>c[i,j]</c> multiplies <c>x**i * y**j</c>.</param>
        /// <returns>The grid, shape <c>c.shape[2:] + x.shape + y.shape</c>.</returns>
        /// <exception cref="System.ArgumentNullException">A coordinate (or <c>pts</c>) is null; a null c is
        ///     <see cref="System.NotSupportedException"/> (NumPy's object array).</exception>
        /// <exception cref="IndexError"><paramref name="c"/> is empty along an evaluated axis.</exception>
        /// <exception cref="ValueError">A ragged tuple / list c or point argument (np.array's inhomogeneous-shape
        ///     text; c is converted before the first points).</exception>
        /// <exception cref="System.NotSupportedException">A null or str <paramref name="c"/>, or a tuple / list
        ///     holding a str / null.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.polygrid2d.html</remarks>
        public NDArray polygrid2d(object x, object y, object c) => NDPolyEval.GridNd(PolyBasis.Power, c, new[] { x, y });

        /// <summary>
        ///     Evaluates the 3-D power series on the Cartesian product of <paramref name="x"/>, <paramref name="y"/>
        ///     and <paramref name="z"/> (NumPy's <c>_gridnd</c>: three tensor passes).
        /// </summary>
        /// <param name="x">First-axis points.</param>
        /// <param name="y">Second-axis points.</param>
        /// <param name="z">Third-axis points.</param>
        /// <param name="c">Coefficients (anything <c>np.array</c> accepts — an <see cref="NDArray"/> of any layout, a typed C# array, a Python tuple / list such as a ValueTuple, <c>object[]</c> or a jagged / <c>NDArray[]</c> array, nested to any depth, or a scalar); <c>c[i,j,k]</c> multiplies <c>x**i * y**j * z**k</c>.</param>
        /// <returns>The grid, shape <c>c.shape[3:] + x.shape + y.shape + z.shape</c>.</returns>
        /// <exception cref="System.ArgumentNullException">A coordinate (or <c>pts</c>) is null; a null c is
        ///     <see cref="System.NotSupportedException"/> (NumPy's object array).</exception>
        /// <exception cref="IndexError"><paramref name="c"/> is empty along an evaluated axis.</exception>
        /// <exception cref="ValueError">A ragged tuple / list c or point argument (np.array's inhomogeneous-shape
        ///     text; c is converted before the first points).</exception>
        /// <exception cref="System.NotSupportedException">A null or str <paramref name="c"/>, or a tuple / list
        ///     holding a str / null.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.polygrid3d.html</remarks>
        public NDArray polygrid3d(object x, object y, object z, object c) => NDPolyEval.GridNd(PolyBasis.Power, c, new[] { x, y, z });

        /// <summary>
        ///     Evaluates the N-D power series at the points <c>pts = (x, y, …)</c> — the n-dimensional
        ///     <see cref="polyval2d"/>. Added in NumPy's development branch (<c>pu._valnd(polyval, c, *pts)</c>); not
        ///     in NumPy 2.4.2, where the same computation is the private <c>polyutils._valnd</c>.
        /// </summary>
        /// <param name="pts">The coordinates, all of one shape (one per series axis evaluated): each an
        ///     <see cref="NDArray"/>, a typed C# array, a Python tuple / list or a scalar, converted first as in
        ///     <c>_valnd</c> (a scalar is a STRONG 0-d array). An <c>NDArray[]</c> passes as it is.</param>
        /// <param name="c">Coefficients (anything <c>np.array</c> accepts — an <see cref="NDArray"/> of any layout, a typed C# array, a Python tuple / list such as a ValueTuple, <c>object[]</c> or a jagged / <c>NDArray[]</c> array, nested to any depth, or a scalar); axis i is <c>pts[i]</c>'s degree.</param>
        /// <returns>The values, shape <c>c.shape[len(pts):] + pts[0].shape</c>.</returns>
        /// <exception cref="System.ArgumentNullException">A coordinate (or <c>pts</c>) is null; a null c is
        ///     <see cref="System.NotSupportedException"/> (NumPy's object array).</exception>
        /// <exception cref="IndexError"><paramref name="pts"/> is empty (<c>list index out of range</c>).</exception>
        /// <exception cref="ValueError">The coordinates differ in shape, or a tuple / list argument is ragged.</exception>
        /// <exception cref="System.NotSupportedException">A null or str <paramref name="c"/>, or a str coordinate.</exception>
        /// <remarks>https://numpy.org/devdocs/reference/generated/numpy.polynomial.polynomial.polyvalnd.html</remarks>
        public NDArray polyvalnd(object[] pts, object c) => NDPolyEval.ValNd(PolyBasis.Power, c, pts);

        // ---- U1: constants, line, add/sub, trim ----

        private static NDArray s_polydomain, s_polyzero, s_polyone, s_polyx;

        /// <summary>
        ///     <c>polydomain</c> — the default domain of the power series class, float64 <c>[-1.0, 1.0]</c>.
        /// </summary>
        /// <remarks>
        ///     NumPy's module constants are ordinary WRITEABLE ndarrays, the same object on every access, and so is
        ///     this one: a process-wide instance (a write through it persists, as in NumPy). It is shared — do not
        ///     dispose it (a dispose is harmless: the buffer is pinned for the process lifetime) and copy it before
        ///     mutating unless the change is meant to be global.
        ///     <para>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.html</para>
        /// </remarks>
        public NDArray polydomain => NDPolySeries.Constant(ref s_polydomain, static () => np.array(new double[] { -1.0, 1.0 }));

        /// <summary><c>polyzero</c> — the power series representing 0, int64 <c>[0]</c> (shared and writeable, see <see cref="polydomain"/>).</summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.html</remarks>
        public NDArray polyzero => NDPolySeries.Constant(ref s_polyzero, static () => np.array(new long[] { 0 }));

        /// <summary><c>polyone</c> — the power series representing 1, int64 <c>[1]</c> (shared and writeable, see <see cref="polydomain"/>).</summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.html</remarks>
        public NDArray polyone => NDPolySeries.Constant(ref s_polyone, static () => np.array(new long[] { 1 }));

        /// <summary><c>polyx</c> — the power series representing <c>x</c>, int64 <c>[0, 1]</c> (shared and writeable, see <see cref="polydomain"/>).</summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.html</remarks>
        public NDArray polyx => NDPolySeries.Constant(ref s_polyx, static () => np.array(new long[] { 0, 1 }));

        /// <summary>
        ///     The power series of the line <c>off + scl*x</c>: <c>[off, scl]</c> when <c>scl != 0</c>, else <c>[off]</c> — built
        ///     with <c>np.array</c>, so the dtype is the one NumPy DISCOVERS for the entries (<c>polyline(1, 2)</c> is int64,
        ///     <c>polyline(1.0, 2)</c> float64, a bool pair stays bool) rather than a coefficient dtype.
        /// </summary>
        /// <param name="off">The offset: a Python number (C# bool / integer / float / double / <see cref="System.Numerics.Complex"/>),
        ///     a NumPy scalar (<see cref="System.Half"/>, <c>char</c>, <c>decimal</c>, a 0-d NDArray) or an array.</param>
        /// <param name="scl">The scale; its truth value <c>scl != 0</c> picks the one- or two-coefficient form (NaN is nonzero).</param>
        /// <returns>A new 1-D array (2-D when the entries are equal-shape arrays, as NumPy stacks them).</returns>
        /// <exception cref="ValueError">An array scale with several elements or none (NumPy's truth-value errors), or
        ///     entries of different shapes (<c>setting an array element with a sequence…</c>).</exception>
        /// <exception cref="TypeError">NumPy's boolean negative / subtract.</exception>
        /// <exception cref="OverflowException">A Python int out of a NumPy integer dtype's range, or CPython's int/float overflow.</exception>
        /// <exception cref="System.NotSupportedException">An entry NumPy would store in an object array: a Python int beyond
        ///     uint64, null, a string.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.polyline.html</remarks>
        public NDArray polyline(object off, object scl) => NDPolySeries.Line(PolyBasis.Power, off, scl);

        /// <summary>
        ///     Adds two power seriess (<c>polyutils._add</c>): both converted to their common type (integers become float64),
        ///     the shorter added into a copy of the longer, trailing zeros trimmed.
        /// </summary>
        /// <param name="c1">First series, low degree first (an array, a C# array or list, or a single number).</param>
        /// <param name="c2">Second series.</param>
        /// <returns>The sum: a new array, or a VIEW of one when trailing zeros were trimmed (as in NumPy).</returns>
        /// <exception cref="ValueError">An empty or non-1-d series, or no common type (a bool series).</exception>
        /// <exception cref="System.NotSupportedException">A null series (NumPy would build an object array).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.polyadd.html</remarks>
        [NDScoped]
        public NDArray polyadd(object c1, object c2) => NDPolySeries.AddSub(c1, c2, subtract: false);

        /// <summary>
        ///     Subtracts the power series <paramref name="c2"/> from <paramref name="c1"/> (<c>polyutils._sub</c>): both converted
        ///     to their common type; when <paramref name="c2"/> is at least as long it is NEGATED first and
        ///     <paramref name="c1"/> added into it (NumPy's operation order, visible in which NaN survives), then trailing
        ///     zeros are trimmed.
        /// </summary>
        /// <param name="c1">Minuend series, low degree first.</param>
        /// <param name="c2">Subtrahend series.</param>
        /// <returns>The difference: a new array, or a VIEW of one when trailing zeros were trimmed.</returns>
        /// <exception cref="ValueError">An empty or non-1-d series, or no common type (a bool series).</exception>
        /// <exception cref="System.NotSupportedException">A null series (NumPy would build an object array).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.polysub.html</remarks>
        [NDScoped]
        public NDArray polysub(object c1, object c2) => NDPolySeries.AddSub(c1, c2, subtract: true);

        /// <summary>
        ///     <c>polytrim</c> — NumPy's alias of <see cref="PolyUtilsModule.trimcoef(object)"/>: removes the trailing
        ///     zero coefficients (tolerance 0) and returns a fresh copy of the coefficient dtype.
        /// </summary>
        /// <param name="c">The series (1-D).</param>
        /// <returns>The trimmed copy (an all-zero series becomes <c>c[:1]*0</c>).</returns>
        /// <exception cref="ValueError">An empty or non-1-d series, or no common type.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.polytrim.html</remarks>
        [NDScoped]
        public NDArray polytrim(object c) => NDPolySeries.TrimCoef(c, PolyNumber.FromPython(PyScalar.Int(0)));

        /// <summary>
        ///     <c>polytrim</c> — NumPy's alias of <see cref="PolyUtilsModule.trimcoef(object, object)"/>: removes the
        ///     trailing coefficients whose magnitude is at most <paramref name="tol"/> and returns a fresh copy.
        /// </summary>
        /// <param name="c">The series (1-D).</param>
        /// <param name="tol">The non-negative tolerance (a Python number, or a NumPy scalar compared in the promoted dtype).</param>
        /// <returns>The trimmed copy (<c>c[:1]*0</c> when nothing exceeds the tolerance).</returns>
        /// <exception cref="ValueError"><c>tol must be non-negative</c>, an empty or non-1-d series, or no common type.</exception>
        /// <exception cref="TypeError">A complex or null tolerance.</exception>
        /// <exception cref="OverflowException">A Python int tolerance too large to convert.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.polytrim.html</remarks>
        [NDScoped]
        public NDArray polytrim(object c, object tol) => NDPolySeries.TrimCoef(c, PolyUtilsModule.Tolerance(tol));

        /// <summary>
        ///     Differentiates the power series <paramref name="c"/> <paramref name="m"/> times along <paramref name="axis"/>,
        ///     multiplying by <paramref name="scl"/> at every order (the chain-rule factor of a linear change of variable).
        ///     Bit-identical to NumPy 2.4.2's <c>polyder</c>: its recurrence <c>der[j-1] = j*c[j]</c>, with <c>c *= scl</c>
        ///     before every order, in NumPy's statement order and NEP 50 dtypes — one IL kernel pass per call, where NumPy
        ///     runs a Python loop per coefficient.
        /// </summary>
        /// <param name="c">
        ///     The coefficients, LOW degree first: an <see cref="NDArray"/> (any layout; an N-D array holds one series per
        ///     position of its other axes), a typed C# array (<c>double[]</c>, <c>float[,]</c> — an ndarray of its dtype),
        ///     a Python list or tuple (<c>object[]</c>, <c>List&lt;T&gt;</c>, a <c>ValueTuple</c>, a jagged
        ///     <c>double[][]</c> or <c>NDArray[]</c> — nested to any depth, its dtype discovered like <c>np.array</c>'s:
        ///     C# ints are Python ints, so <c>[1, 2.5]</c> is float64), or a single number. Integer and bool series are
        ///     differentiated as float64.
        /// </param>
        /// <param name="m">How many times to differentiate (≥ 0). 0 returns a copy; <c>m &gt;= len(c)</c> returns
        ///     <c>c[:1]*0</c> — zeros, except NaN where c[0] is inf or NaN.</param>
        /// <param name="scl">
        ///     The multiplier applied at each order: null for NumPy's default (the Python int 1); a C# number — a Python
        ///     scalar that ADOPTS the series dtype (<c>0.1</c> on a float32 series multiplies by float32 0.1); a
        ///     <see cref="System.Half"/>/<c>char</c>/<c>decimal</c> or a 0-d NDArray — a NumPy scalar that may WIDEN the
        ///     arithmetic (<c>np.float64</c> on a float32 series multiplies in float64, then rounds); or an array — an
        ///     NDArray, typed C# array, or Python list / tuple — that broadcasts against the series in place (one factor per
        ///     column, say). Even the default 1 is applied: a complex inf coefficient becomes <c>inf+nanj</c> exactly as in
        ///     NumPy. A Python int of any size (<see cref="System.Numerics.BigInteger"/>) converts with NumPy's rounding.
        /// </param>
        /// <param name="axis">The axis the series runs along (negative counts from the end).</param>
        /// <returns>The derivative's coefficients, of the series' dtype: a new array — a view of it in NumPy's layout
        ///     (the series axis back in place), so <c>flags.owndata</c> is false, as in NumPy.</returns>
        /// <exception cref="ValueError"><c>The order of derivation must be non-negative</c>, an array
        ///     <paramref name="scl"/> whose broadcast would stretch the series, or a ragged list / tuple
        ///     (<paramref name="c"/> or <paramref name="scl"/>: np.array's inhomogeneous-shape text).</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range.</exception>
        /// <exception cref="System.ArgumentException">A <paramref name="scl"/> the series cannot absorb in place — NumPy's
        ///     UFuncTypeError, e.g. a complex scl on a real series.</exception>
        /// <exception cref="IncorrectShapeException">An array <paramref name="scl"/> that does not broadcast with the series.</exception>
        /// <exception cref="System.OverflowException">A Python-int <paramref name="scl"/> too large for the series' dtype.</exception>
        /// <exception cref="System.NotSupportedException">A null or string series, a list holding one, or a Python int past
        ///     uint64 in the series (NumPy's object/str arrays).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.polyder.html</remarks>
        [NDScoped]
        public NDArray polyder(object c, int m = 1, object scl = null, int axis = 0) => NDPolyCalc.Der(PolyBasis.Power, c, m, scl, axis);

        /// <summary>
        ///     Integrates the power series <paramref name="c"/> <paramref name="m"/> times along <paramref name="axis"/>:
        ///     each order multiplies by <paramref name="scl"/>, integrates (<c>tmp[j+1] = c[j]/(j+1)</c>) and adds the
        ///     constant that makes the new series equal <c>k[i]</c> at <paramref name="lbnd"/>
        ///     (<c>tmp[0] += k[i] - polyval(lbnd, tmp)</c>). Bit-identical to NumPy 2.4.2's <c>polyint</c>, NumPy's
        ///     statement order and NEP 50 dtypes included.
        /// </summary>
        /// <param name="c">The coefficients, LOW degree first (see <see cref="polyder"/>).</param>
        /// <param name="m">How many times to integrate (≥ 0); 0 returns a copy.</param>
        /// <param name="k">
        ///     The integration constants, one per order (missing ones are 0; more than <paramref name="m"/> is an error):
        ///     null for none, a single number, a Python list or tuple (<c>object[]</c>, <c>IList</c>, a <c>ValueTuple</c> —
        ///     Python numbers, which adopt the series dtype, or nested lists / tuples, each converted only when its order
        ///     uses it, as NumPy does), or an array (an NDArray or typed C# array — NumPy scalars of its dtype; for an N-D
        ///     series a row per order that broadcasts over the series' columns).
        /// </param>
        /// <param name="lbnd">The lower bound (null for NumPy's 0); a scalar, Python (weak — a <see cref="System.Numerics.BigInteger"/>
        ///     too) or NumPy (strong). A list or tuple is not a scalar.</param>
        /// <param name="scl">The multiplier applied at each order (null for NumPy's 1); a scalar — see <see cref="polyder"/>.</param>
        /// <param name="axis">The axis the series runs along.</param>
        /// <returns>The integral's coefficients, of the series' dtype (a view of a new array in NumPy's layout).</returns>
        /// <exception cref="ValueError">NumPy's texts: <c>The order of integration must be non-negative</c>, <c>Too many
        ///     integration constants</c>, <c>lbnd must be a scalar.</c>, <c>scl must be a scalar.</c>, <c>setting an array
        ///     element with a sequence.</c> (an array constant for a 1-D series), a constant that would stretch the series,
        ///     or a ragged list / tuple (np.array's inhomogeneous-shape text).</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range.</exception>
        /// <exception cref="IndexError">An empty series (<c>index 0 is out of bounds for axis 0 with size 0</c>).</exception>
        /// <exception cref="TypeError">An array constant for a complex 1-D series (NumPy's <c>complex()</c> text).</exception>
        /// <exception cref="System.ArgumentException">A scl, constant or complex lbnd the (N-D) series cannot absorb in
        ///     place — NumPy's UFuncTypeError. (A 1-D series keeps the real part, as NumPy's setitem does.)</exception>
        /// <exception cref="IncorrectShapeException">Constants that do not broadcast with the series' columns.</exception>
        /// <exception cref="System.OverflowException">A Python-int scl, lbnd or constant too large for the series' dtype.</exception>
        /// <exception cref="System.NotSupportedException">A null or string series; a str / null constant, or a str bound,
        ///     when an order uses it (NumPy fails there with its str-dtype error).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.polyint.html</remarks>
        [NDScoped]
        public NDArray polyint(object c, int m = 1, object k = null, object lbnd = null, object scl = null, int axis = 0)
            => NDPolyCalc.Int(PolyBasis.Power, c, m, k, lbnd, scl, axis);
    }
}
