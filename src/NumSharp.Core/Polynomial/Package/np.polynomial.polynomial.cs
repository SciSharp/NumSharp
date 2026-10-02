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
        ///     Adds two power series (<c>polyutils._add</c>): both converted to their common type (integers become float64),
        ///     the shorter added into a copy of the longer, trailing zeros trimmed.
        /// </summary>
        /// <param name="c1">First series, low degree first (an array, a C# array or list, or a single number).</param>
        /// <param name="c2">Second series.</param>
        /// <returns>The sum: a new array, or a VIEW of one when trailing zeros were trimmed (as in NumPy).</returns>
        /// <exception cref="ValueError">An empty or non-1-d series, or no common type (a bool series).</exception>
        /// <exception cref="System.NotSupportedException">A null series, or one holding None, a non-numeric object or a Python int past uint64 (NumPy's object array), raised after both series' checks.</exception>
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
        /// <exception cref="System.NotSupportedException">A null series, or one holding None, a non-numeric object or a Python int past uint64 (NumPy's object array), raised after both series' checks.</exception>
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

        // ---------------------------------------------------------------------------------------------
        //  Series algebra (plan U2): polymulx / polymul / polydiv / polypow / polyfromroots
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     Multiplies the power series <paramref name="c"/> by x: <c>prd[0] = c[0]*0; prd[1:] = c</c> — the coefficients shift up one degree (<c>c[0]*0</c> keeps NaN/inf and the sign of zero NumPy's way). The zero series <c>[0]</c> is returned as is.
        ///     Bit-identical to NumPy 2.4.2's <c>polymulx</c> — one IL kernel pass where NumPy runs a Python loop.
        /// </summary>
        /// <param name="c">The coefficients, LOW degree first: an <see cref="NDArray"/> (any layout), a typed C# array, a Python list or tuple (<c>object[]</c>, a <c>ValueTuple</c>, a jagged array — np.array's coercion), or a single number. Integer series are computed in float64; a bool series has no common type.</param>
        /// <returns>A new array one coefficient longer (the zero series: a one-element copy), of <c>np.common_type(c)</c>.</returns>
        /// <exception cref="ValueError">An empty series (<c>Coefficient array is empty</c>), a series that is not 1-d
        ///     (<c>Coefficient array is not 1-d</c>), or no common type (a bool or str series), in NumPy's order; a ragged list
        ///     (np.array's inhomogeneous-shape text).</exception>
        /// <exception cref="System.NotSupportedException">A null series, or one holding None, a non-numeric object or a Python int past
        ///     uint64 (NumPy's object array, a dtype NumSharp does not have) — raised where NumPy starts computing with
        ///     it, after every series check. A str series is NumPy's ValueError (no common type).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.polymulx.html</remarks>
        [NDScoped]
        public NDArray polymulx(object c) => NDPolyAlgebra.Mulx(PolyBasis.Power, c);

        /// <summary>
        ///     Multiplies two power series: <c>np.convolve(c1, c2)</c>, trailing zeros trimmed. The product is np.convolve's arithmetic — NumPy's sequential sum for real series whose shorter factor has at most ~15 terms, cblas <c>?dot</c> beyond that and for every complex product; with <c>NumSharp.Interop.OpenBLAS</c> installed those run through NumPy's own BLAS and are byte-identical, without it they agree to a few ULP.
        /// </summary>
        /// <param name="c1">First factor. The coefficients, LOW degree first: an <see cref="NDArray"/> (any layout), a typed C# array, a Python list or tuple (<c>object[]</c>, a <c>ValueTuple</c>, a jagged array — np.array's coercion), or a single number. Integer series are computed in float64; a bool series has no common type.</param>
        /// <param name="c2">Second factor.</param>
        /// <returns>The product: a new array, or a VIEW of one when trailing zeros were trimmed (as in NumPy).</returns>
        /// <exception cref="ValueError">An empty series (<c>Coefficient array is empty</c>), a series that is not 1-d
        ///     (<c>Coefficient array is not 1-d</c>), or no common type (a bool or str series), in NumPy's order; a ragged list
        ///     (np.array's inhomogeneous-shape text).</exception>
        /// <exception cref="System.NotSupportedException">A null series, or one holding None, a non-numeric object or a Python int past
        ///     uint64 (NumPy's object array, a dtype NumSharp does not have) — raised where NumPy starts computing with
        ///     it, after every series check. A str series is NumPy's ValueError (no common type).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.polymul.html</remarks>
        [NDScoped]
        public NDArray polymul(object c1, object c2) => NDPolyAlgebra.Mul(PolyBasis.Power, c1, c2);

        /// <summary>
        ///     Divides the power series <paramref name="c1"/> by <paramref name="c2"/>, returning quotient and remainder:
        ///     NumPy's synthetic division in place on a copy of <paramref name="c1"/> (<c>c1[i:j] -= (c2[:-1]/c2[-1]) * c1[j]</c>); the remainder is a VIEW of that copy (<c>c1[:lc2-1]</c>, trimmed), as NumPy returns it. A dividend shorter than the divisor gives <c>(c1[:1]*0, c1)</c>, a one-term divisor
        ///     <c>(c1/c2[-1], c1[:1]*0)</c>. Bit-identical to NumPy 2.4.2's <c>polydiv</c>.
        /// </summary>
        /// <param name="c1">The dividend. The coefficients, LOW degree first: an <see cref="NDArray"/> (any layout), a typed C# array, a Python list or tuple (<c>object[]</c>, a <c>ValueTuple</c>, a jagged array — np.array's coercion), or a single number. Integer series are computed in float64; a bool series has no common type.</param>
        /// <param name="c2">The divisor.</param>
        /// <returns>(quo, rem): NumPy's tuple, each a new array or a view of one.</returns>
        /// <exception cref="ValueError">An empty series (<c>Coefficient array is empty</c>), a series that is not 1-d
        ///     (<c>Coefficient array is not 1-d</c>), or no common type (a bool or str series), in NumPy's order; a ragged list
        ///     (np.array's inhomogeneous-shape text).</exception>
        /// <exception cref="System.NotSupportedException">A null series, or one holding None, a non-numeric object or a Python int past
        ///     uint64 (NumPy's object array, a dtype NumSharp does not have) — raised where NumPy starts computing with
        ///     it, after every series check. A str series is NumPy's ValueError (no common type).</exception>
        /// <exception cref="System.DivideByZeroException">The divisor is zero (all its coefficients, after trimming) —
        ///     NumPy's bare <c>ZeroDivisionError</c>, empty message.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.polydiv.html</remarks>
        [NDScoped]
        public (NDArray quo, NDArray rem) polydiv(object c1, object c2) => NDPolyAlgebra.Div(PolyBasis.Power, c1, c2);

        /// <summary>
        ///     Raises the power series <paramref name="c"/> to the power <paramref name="pow"/>: <c>[1]</c> of the series' dtype
        ///     for 0, the (trimmed) series for 1, otherwise <c>np.convolve</c> applied pow - 1 times (no trimming between steps). Bit-identical to NumPy 2.4.2's <c>polypow</c> (the product's parity notes on <see cref="polymul"/> apply).
        /// </summary>
        /// <param name="c">The coefficients, LOW degree first: an <see cref="NDArray"/> (any layout), a typed C# array, a Python list or tuple (<c>object[]</c>, a <c>ValueTuple</c>, a jagged array — np.array's coercion), or a single number. Integer series are computed in float64; a bool series has no common type.</param>
        /// <param name="pow">The power (≥ 0).</param>
        /// <param name="maxpower">The largest power allowed: null (NumPy's None: no limit). NumPy's default keeps an accidental huge power from
        ///     running away.</param>
        /// <returns>The power series, a new array of <c>np.common_type(c)</c>.</returns>
        /// <exception cref="ValueError">The series' errors (see <see cref="polymul"/>), then
        ///     <c>Power must be a non-negative integer.</c>, then <c>Power is too large</c> — NumPy's order.</exception>
        /// <exception cref="System.NotSupportedException">An object series (see <see cref="polymul"/>), raised after the power checks.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.polypow.html</remarks>
        [NDScoped]
        public NDArray polypow(object c, int pow, int? maxpower = null) => NDPolyAlgebra.Pow(PolyBasis.Power, c, pow, maxpower);

        /// <summary>
        ///     <see cref="polypow(object, int, int?)"/> with a Python-float power: NumPy's <c>power = int(pow)</c>
        ///     truncates, so 2.0 is 2 while 2.5 fails <c>power != pow</c>.
        /// </summary>
        /// <param name="c">The coefficients, LOW degree first: an <see cref="NDArray"/> (any layout), a typed C# array, a Python list or tuple (<c>object[]</c>, a <c>ValueTuple</c>, a jagged array — np.array's coercion), or a single number. Integer series are computed in float64; a bool series has no common type.</param>
        /// <param name="pow">The power; it must equal its integer part and be ≥ 0.</param>
        /// <param name="maxpower">The largest power allowed: null (NumPy's None: no limit).</param>
        /// <returns>The power series.</returns>
        /// <exception cref="ValueError">The series' errors; <c>cannot convert float NaN to integer</c>;
        ///     <c>Power must be a non-negative integer.</c> (a fractional or negative power); <c>Power is too large</c>.</exception>
        /// <exception cref="System.OverflowException"><c>cannot convert float infinity to integer</c>; a power beyond
        ///     int64 with no limit (NumPy would run until memory is exhausted).</exception>
        /// <exception cref="System.NotSupportedException">An object series (see <see cref="polymul"/>), raised after the power checks.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.polypow.html</remarks>
        [NDScoped]
        public NDArray polypow(object c, double pow, int? maxpower = null) => NDPolyAlgebra.Pow(PolyBasis.Power, c, pow, maxpower);

        /// <summary>
        ///     <see cref="polypow(object, int, int?)"/> for ANY power argument, with no limit (NumPy's default <c>maxpower=None</c>): <c>power =
        ///     int(pow)</c> with Python's semantics — a bool is 0 / 1, np.float16 (Half) or decimal truncates like a float, a
        ///     str parses with int()'s grammar (and then fails <c>power != pow</c>), a 0-d NDArray converts its element — and
        ///     NumPy's error for every value int() refuses.
        /// </summary>
        /// <param name="c">The coefficients, LOW degree first: an <see cref="NDArray"/> (any layout), a typed C# array, a Python list or tuple (<c>object[]</c>, a <c>ValueTuple</c>, a jagged array — np.array's coercion), or a single number. Integer series are computed in float64; a bool series has no common type.</param>
        /// <param name="pow">The power: any value (see the summary).</param>
        /// <returns>The power series.</returns>
        /// <exception cref="ValueError">The series' errors (see <see cref="polymul"/>); <c>cannot convert float NaN to
        ///     integer</c>; <c>invalid literal for int() with base 10: '…'</c>; <c>Power must be a non-negative
        ///     integer.</c>; <c>Power is too large</c>.</exception>
        /// <exception cref="TypeError">A power int() refuses: a complex, null (None), a list or tuple (<c>int() argument must
        ///     be a string, a bytes-like object or a real number, not '…'</c>), an ndarray of one or more dims (<c>only
        ///     0-dimensional arrays can be converted to Python scalars</c>).</exception>
        /// <exception cref="System.OverflowException"><c>cannot convert float infinity to integer</c>; a power beyond int64
        ///     with no limit (NumPy would run until memory is exhausted).</exception>
        /// <exception cref="System.NotSupportedException">An object series (see <see cref="polymul"/>), raised after the
        ///     power checks.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.polypow.html</remarks>
        [NDScoped]
        public NDArray polypow(object c, object pow) => NDPolyAlgebra.Pow(PolyBasis.Power, c, pow, null);

        /// <summary>
        ///     <see cref="polypow(object, object)"/> with ANY limit: null is None (no limit); anything else is compared as
        ///     NumPy compares <c>power &gt; maxpower</c> — exactly for a Python int / bool / float, a NumPy integer scalar
        ///     (char) or decimal; in float16 for np.float16 (Half); for an ndarray (an NDArray, a typed C# array) in its dtype
        ///     (float16 / float32 / float64 rounding the power first, complex lexicographically, bool as int64, integers
        ///     exactly), the comparison's truth value then needing exactly one element.
        /// </summary>
        /// <param name="c">The coefficients, LOW degree first: an <see cref="NDArray"/> (any layout), a typed C# array, a Python list or tuple (<c>object[]</c>, a <c>ValueTuple</c>, a jagged array — np.array's coercion), or a single number. Integer series are computed in float64; a bool series has no common type.</param>
        /// <param name="pow">The power: any value (see <see cref="polypow(object, object)"/>).</param>
        /// <param name="maxpower">The largest power allowed: any value, null for no limit.</param>
        /// <returns>The power series.</returns>
        /// <exception cref="ValueError">The series' and the power's errors; <c>Power is too large</c>; an ndarray limit with
        ///     no element or several (NumPy's truth-value texts).</exception>
        /// <exception cref="TypeError">A power int() refuses, or a limit that does not compare with an int: a complex, str,
        ///     list or tuple (<c>'&gt;' not supported between instances of 'int' and '…'</c>).</exception>
        /// <exception cref="System.OverflowException"><c>cannot convert float infinity to integer</c>; a power the limit's
        ///     comparison dtype cannot hold (<c>int too large to convert to float</c> for a float limit ndarray or np.float16,
        ///     <c>int too big to convert</c> for a bool one); a power beyond int64 with no limit.</exception>
        /// <exception cref="System.NotSupportedException">An object series, raised after the power checks.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.polypow.html</remarks>
        [NDScoped]
        public NDArray polypow(object c, object pow, object maxpower) => NDPolyAlgebra.Pow(PolyBasis.Power, c, pow, maxpower);

        /// <summary>
        ///     The power series whose roots are <paramref name="roots"/>: the roots sorted, one linear factor per root
        ///     (<c>polyline(-r, 1) = [-r, 1]</c>), multiplied pairwise as a balanced tree (<c>polyutils._fromroots</c>) with <c>polymul</c>.
        ///     Empty roots give <c>[1.]</c>. The factors are built by <c>np.array([-r, …])</c>, whose coercion promotes a
        ///     float16 / float32 root to float64 — so the series is float64 (complex128 for complex roots). Bit-identical to
        ///     NumPy 2.4.2's <c>polyfromroots</c> (the product's parity notes on <see cref="polymul"/> apply); one documented difference: roots that compare equal but differ in bits (+0.0 and -0.0) are ordered
        ///     deterministically here, where NumPy's unstable SIMD sort decides by CPU, which can flip the sign of a zero
        ///     coefficient.
        /// </summary>
        /// <param name="roots">The roots: an <see cref="NDArray"/>, a typed C# array, a Python list or tuple, …
        ///     (anything with a length — NumPy's <c>len(roots)</c> runs first).</param>
        /// <returns>A new array of <c>len(roots) + 1</c> coefficients.</returns>
        /// <exception cref="TypeError">A scalar (<c>object of type 'float' has no len()</c>) or a 0-d array (<c>len() of
        ///     unsized object</c>).</exception>
        /// <exception cref="ValueError">Roots that are not 1-d, a zero-size array of nonzero length, or no common type
        ///     (bool roots).</exception>
        /// <exception cref="System.NotSupportedException">An object root array (None, or roots holding None, a non-numeric object or a Python int past uint64), raised after its checks; str roots are NumPy's ValueError (no common type).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.polyfromroots.html</remarks>
        [NDScoped]
        public NDArray polyfromroots(object roots) => NDPolyAlgebra.FromRoots(PolyBasis.Power, roots);

        // ---------------------------------------------------------------------------------------------
        //  Vandermonde matrices (plan U5): polyvander / polyvander2d / polyvander3d
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     The Vandermonde matrix of degree <paramref name="deg"/>: <c>V[..., i] = x**i</c>, by NumPy's forward recurrence
        ///     <c>v[0] = x*0 + 1; v[1] = x; v[i] = v[i-1] * x</c> — so <c>polyvander(x, n) @ c == polyval(x, c)</c> for a
        ///     series of n + 1 coefficients (up to rounding). Bit-identical to NumPy 2.4.2's <c>polyvander</c>: one IL kernel pass
        ///     where NumPy runs a Python loop over the degrees; every statement is NumPy's array op in its dtype.
        /// </summary>
        /// <param name="x">The points: an <see cref="NDArray"/> of any layout (a 0-d one is one point), a typed C# array, a
        ///     Python list or tuple (<c>object[]</c>, a <c>ValueTuple</c>, a jagged array — np.array's coercion), or a scalar.
        ///     The matrix is computed in <c>(x + 0.0).dtype</c>: float16 / float32 / float64 / complex128 are kept (a float32
        ///     x gives a float32 matrix), bool, integer and char points become float64; -0.0 reads as +0.0.</param>
        /// <param name="deg">The degree (≥ 0).</param>
        /// <returns>A view of shape <c>x.shape + (deg + 1,)</c> (<c>(1, deg + 1)</c> for a scalar x) over a new array — the
        ///     degree axis last, as NumPy's <c>np.moveaxis</c> leaves it (OWNDATA false; F-contiguous for a 1-D x).</returns>
        /// <exception cref="ValueError"><c>deg must be non-negative</c>; a ragged list x (np.array's inhomogeneous-shape
        ///     text); a matrix too large to describe (<c>array is too big; …</c>).</exception>
        /// <exception cref="System.OutOfMemoryException">The matrix cannot be allocated (NumPy's MemoryError).</exception>
        /// <exception cref="System.NotSupportedException">A null or str x, or a list holding a str / None / a Python int
        ///     past uint64 (NumPy's str / object arrays, dtypes NumSharp does not have).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.polyvander.html</remarks>
        [NDScoped]
        public NDArray polyvander(object x, int deg) => NDPolyVander.Vander(PolyBasis.Power, x, deg);

        /// <summary>
        ///     <see cref="polyvander(object, int)"/> with a degree of ANY kind, read as NumPy's <c>polyutils._as_int</c> reads it
        ///     (Python's <c>operator.index</c>): a C# integer, bool (True is 1), <see cref="System.Numerics.BigInteger"/>, char,
        ///     or a 0-d integer <see cref="NDArray"/>. Anything else — a float or double (even 2.0), a complex, a str, null, a
        ///     0-d bool / float array, an array of one or more dims, a list or tuple — raises NumPy's TypeError, whose text
        ///     formats the value as NumPy's f-string does (<c>deg must be an integer, received 1e+20</c>,
        ///     <c>… received [1 2]</c>, <c>… received None</c>).
        /// </summary>
        /// <param name="x">The points (see <see cref="polyvander(object, int)"/>).</param>
        /// <param name="deg">The degree, any value (see the summary).</param>
        /// <returns>The matrix (see <see cref="polyvander(object, int)"/>).</returns>
        /// <exception cref="TypeError">A degree operator.index refuses: <c>deg must be an integer, received …</c> — raised
        ///     before x is looked at.</exception>
        /// <exception cref="ValueError"><c>deg must be non-negative</c> (before x is converted); a ragged list x;
        ///     <c>Maximum allowed dimension exceeded</c> (a degree of 2^63 - 1 or more) or <c>array is too big; …</c>.</exception>
        /// <exception cref="System.OutOfMemoryException">The matrix cannot be allocated (NumPy's MemoryError).</exception>
        /// <exception cref="System.NotSupportedException">A null or str x, or a list NumPy makes a str / object array of.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.polyvander.html</remarks>
        [NDScoped]
        public NDArray polyvander(object x, object deg) => NDPolyVander.Vander(PolyBasis.Power, x, deg);

        /// <summary>
        ///     The 2-D Vandermonde matrix: column <c>a*(deg[1] + 1) + b</c> is <c>x**a * y**b</c> at every point pair — NumPy's
        ///     <c>polyutils._vander_nd_flat</c>, the outer product <c>V_x[..., :, None] * V_y[..., None, :]</c> of the two 1-D
        ///     matrices, flattened. Bit-identical to NumPy 2.4.2's <c>polyvander2d</c>: one IL kernel pass builds each block of
        ///     points' per-axis matrices in cache and writes the products straight into the result.
        /// </summary>
        /// <param name="x">The first coordinates. x and y are STACKED as <c>np.asarray((x, y))</c> — they must have exactly the
        ///     same shape (no broadcasting), and the matrix dtype is np.promote_types over both (array coercion's rule: a
        ///     Python float is float64 there, so a float16 array with a Python-float y is float64) plus 0.0.</param>
        /// <param name="y">The second coordinates (same shape as x).</param>
        /// <param name="deg">The two degrees <c>[x_deg, y_deg]</c>: a list / tuple, a typed C# array, an NDArray or a str of
        ///     length 2, each item read as <see cref="polyvander(object, object)"/>'s deg.</param>
        /// <returns>A view of shape <c>x.shape + ((deg[0] + 1) * (deg[1] + 1),)</c> over a new array (OWNDATA false; a scalar x, y
        ///     is one point).</returns>
        /// <exception cref="TypeError">A deg without a length (<c>object of type 'int' has no len()</c>, <c>len() of unsized
        ///     object</c>) — before the points are converted — or a non-integer degree (checked axis by axis, after the
        ///     points).</exception>
        /// <exception cref="ValueError"><c>Expected 2 dimensions of degrees, got {len}</c>; points of different shapes
        ///     (np.array's inhomogeneous-shape text); <c>deg must be non-negative</c>; <c>array is too big; …</c>.</exception>
        /// <exception cref="IncorrectShapeException">No points (an empty x, y): NumPy's reshape text <c>cannot reshape array of
        ///     size 0 into shape (0,newaxis)</c>.</exception>
        /// <exception cref="System.OutOfMemoryException">A matrix NumPy allocates on the way cannot be allocated.</exception>
        /// <exception cref="System.OverflowException">Scalar points with a Python int past the float range (NumPy's
        ///     <c>int too large to convert to float</c> at its per-element <c>+ 0.0</c>).</exception>
        /// <exception cref="System.NotSupportedException">Points NumPy stacks into a str array, or into an object array it
        ///     computes with as Python objects or fails on. Scalar points that are all numbers but stack into an object array
        ///     (a Python int past uint64 among them) ARE computed: each dimension gets its own number, as in NumPy.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.polyvander2d.html</remarks>
        [NDScoped]
        public NDArray polyvander2d(object x, object y, object deg) => NDPolyVander.VanderNd(PolyBasis.Power, new[] { x, y }, deg);

        /// <summary>
        ///     The 3-D Vandermonde matrix: column <c>(a*(deg[1] + 1) + b)*(deg[2] + 1) + c</c> is <c>x**a * y**b * z**c</c> —
        ///     NumPy's <c>((V_x[..., :, None, None] * V_y[..., None, :, None]) * V_z[..., None, None, :])</c> flattened, the
        ///     first product rounded to the dtype before the second. Bit-identical to NumPy 2.4.2's <c>polyvander3d</c>.
        /// </summary>
        /// <param name="x">The first coordinates (x, y and z are stacked: one shape — see <see cref="polyvander2d"/>).</param>
        /// <param name="y">The second coordinates.</param>
        /// <param name="z">The third coordinates.</param>
        /// <param name="deg">The three degrees <c>[x_deg, y_deg, z_deg]</c> (see <see cref="polyvander2d"/>).</param>
        /// <returns>A view of shape <c>x.shape + ((deg[0] + 1) * (deg[1] + 1) * (deg[2] + 1),)</c> over a new array.</returns>
        /// <exception cref="TypeError">A deg without a length, or a non-integer degree (see <see cref="polyvander2d"/>).</exception>
        /// <exception cref="ValueError"><c>Expected 3 dimensions of degrees, got {len}</c>, and <see cref="polyvander2d"/>'s
        ///     other errors.</exception>
        /// <exception cref="IncorrectShapeException">No points (NumPy's reshape text).</exception>
        /// <exception cref="System.OutOfMemoryException">A matrix NumPy allocates on the way cannot be allocated.</exception>
        /// <exception cref="System.OverflowException">Scalar points with a Python int past the float range (NumPy's
        ///     <c>int too large to convert to float</c> at its per-element <c>+ 0.0</c>).</exception>
        /// <exception cref="System.NotSupportedException">Points NumPy stacks into a str array, or into an object array it
        ///     computes with as Python objects or fails on. Scalar points that are all numbers but stack into an object array
        ///     (a Python int past uint64 among them) ARE computed: each dimension gets its own number, as in NumPy.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.polyvander3d.html</remarks>
        [NDScoped]
        public NDArray polyvander3d(object x, object y, object z, object deg) => NDPolyVander.VanderNd(PolyBasis.Power, new[] { x, y, z }, deg);

        // ---------------------------------------------------------------------------------------------
        //  Companion matrix and roots (plan U7): polycompanion / polyroots
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     The companion matrix of the power series <paramref name="c"/>: ones on the sub-diagonal and <c>-c[:-1] / c[-1]</c> in the last column, so its eigenvalues are the series' roots. Unlike the orthogonal bases' companions it cannot be made symmetric by scaling the basis. The matrix keeps the series' common type and is computed in it throughout (the quotient, then <c>0 - quotient</c> into the zero column). Bit-identical to NumPy 2.4.2's <c>polycompanion</c>.
        /// </summary>
        /// <param name="c">The coefficients, LOW degree first: an <see cref="NDArray"/> (any layout), a typed C# array, a Python list or tuple (<c>object[]</c>, a <c>ValueTuple</c>, a jagged array — np.array's coercion), or a single number. Trailing zeros are trimmed first; integer series are computed in float64; a bool series has no common type.</param>
        /// <returns>A new owning (deg, deg) matrix of the series' common type; a two-term series gives the 1x1 matrix
        ///     <c>[[-c[0] / c[1]]]</c> (NumPy's arithmetic on the two coefficients: scalarmath, or — for an object series — the items' own: CPython for Python
        ///     numbers, ufuncs for 0-d arrays; the 1x1 array is numeric either way).</returns>
        /// <exception cref="ValueError">An empty series (<c>Coefficient array is empty</c>), a series that is not 1-d (<c>Coefficient array is not 1-d</c>), or no common type (a bool or str series), in NumPy's order; a ragged list (np.array's inhomogeneous-shape text). Fewer than two terms after trimming:
        ///     <c>Series must have maximum degree of at least 1.</c></exception>
        /// <exception cref="System.NotSupportedException">An OBJECT series (None, a non-numeric object or a Python int past
        ///     uint64 among the terms makes NumPy's as_series copy an object array) of three or more terms after trimming whose
        ///     arithmetic does not raise — NumPy's result is an object matrix, a dtype NumSharp does not have; or one where NumPy's
        ///     object arithmetic reaches None or a str (CPython's TypeError there). A two-term object series of numbers computes:
        ///     NumPy's linear root of the items is a numeric 1x1 array. A str series is NumPy's ValueError (no common type).</exception>
        /// <exception cref="System.OverflowException">An object series whose arithmetic meets a Python int too large for a
        ///     float — CPython's <c>integer division result too large for a float</c> when it is divided by a Python int,
        ///     <c>int too large to convert to float</c> when it meets anything else — whichever NumPy's statements raise first.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.polycompanion.html</remarks>
        [NDScoped]
        public NDArray polycompanion(object c) => NDPolyAlgebra.Companion(PolyBasis.Power, c);

        /// <summary>
        ///     The roots of the power series <paramref name="c"/>: the eigenvalues of <see cref="polycompanion"/>'s matrix (not rotated: the power series is the one basis NumPy does not rotate), computed by
        ///     <see cref="np.linalg.eigvals"/> and sorted in place, ascending (complex numbers lexicographically, real part
        ///     first). A constant series has no roots — an empty array of its dtype — and a linear one the single root
        ///     <c>-c[0] / c[1]</c> (scalarmath, in its own dtype). Roots far from the origin, and multiple roots, carry larger
        ///     errors (NumPy's documented caveat). With NumSharp.Interop.OpenBLAS referenced the eigenvalues come from NumPy's
        ///     own LAPACK <c>geev</c> and are bit-identical to NumPy 2.4.2's <c>polyroots</c> (one BLAS thread); where
        ///     +0.0 and -0.0 roots meet, NumPy's SIMD sort orders them by CPU and NumSharp's sort deterministically.
        /// </summary>
        /// <param name="c">The coefficients, LOW degree first: an <see cref="NDArray"/> (any layout), a typed C# array, a Python list or tuple (<c>object[]</c>, a <c>ValueTuple</c>, a jagged array — np.array's coercion), or a single number. Trailing zeros are trimmed first; integer series are computed in float64; a bool series has no common type.</param>
        /// <returns>The sorted roots: REAL when every eigenvalue is — a view of the eigenvalue array's real parts for a float64
        ///     series (OWNDATA false, as NumPy's <c>w.real</c>), a new float32 array for a float32 series — otherwise
        ///     complex128 (NumPy's complex64 for a float32 series: NumSharp has one complex width).</returns>
        /// <exception cref="ValueError">An empty series (<c>Coefficient array is empty</c>), a series that is not 1-d (<c>Coefficient array is not 1-d</c>), or no common type (a bool or str series), in NumPy's order; a ragged list (np.array's inhomogeneous-shape text).</exception>
        /// <exception cref="TypeError">A float16 (or decimal) series of degree 2 or more: linalg's
        ///     <c>array type float16 is unsupported in linalg</c>. An OBJECT series of three or more terms whose arithmetic
        ///     does not raise: eigvals' <c>ufunc 'isfinite' not supported for the input types, …</c> on NumPy's object
        ///     companion matrix (raised before LAPACK — no backend needed).</exception>
        /// <exception cref="LinAlgError">A companion matrix holding an infinity or NaN (<c>Array must not contain infs or
        ///     NaNs</c>, checked before the dtype), or eigenvalues that do not converge.</exception>
        /// <exception cref="MissingBackendException">Degree 2 or more with no LAPACK backend: reference
        ///     NumSharp.Interop.OpenBLAS (NumSharp.Core ships no eigensolver).</exception>
        /// <exception cref="System.NotSupportedException">An OBJECT series (None, a non-numeric object or a Python int past
        ///     uint64 among the terms) of one term — NumPy returns <c>np.array([], dtype=object)</c> — or one where NumPy's
        ///     object arithmetic reaches None or a str (CPython's TypeError there). A two-term object series of numbers
        ///     computes: NumPy's linear root of the items is a numeric array.</exception>
        /// <exception cref="System.OverflowException">An object series whose arithmetic meets a Python int too large for a
        ///     float — CPython's <c>integer division result too large for a float</c> when it is divided by a Python int,
        ///     <c>int too large to convert to float</c> when it meets anything else — whichever NumPy's statements raise first.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.polyroots.html</remarks>
        [NDScoped]
        public NDArray polyroots(object c) => NDPolyAlgebra.Roots(PolyBasis.Power, c);
    }
}
