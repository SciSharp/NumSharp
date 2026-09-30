using NumSharp.Backends.Kernels;

namespace NumSharp
{
    /// <summary>
    ///     <c>numpy.polynomial.legendre</c> — Legendre series <c>p(x) = c[0]*P_0(x) + c[1]*P_1(x) + …</c>, reachable as
    ///     <c>np.polynomial.legendre</c>. Coefficients are LOW degree first.
    /// </summary>
    /// <remarks>https://numpy.org/doc/stable/reference/routines.polynomials.legendre.html</remarks>
    [ModuleName("np.polynomial.legendre")]
    public class LegendreModule
    {
        /// <summary>Constructs the module; there is one shared instance behind <c>np.polynomial.legendre</c>.</summary>
        internal LegendreModule() { }

        /// <summary>
        ///     Evaluates the Legendre series with coefficients <paramref name="c"/> at <paramref name="x"/> by Clenshaw's recurrence exactly as NumPy writes it (<c>c0 = c[-i] - c1*((nd - 1)/nd); c1 = tmp + c1*x*((2*nd - 1)/nd); return c0 + c1*x</c>),
        ///     bit-identical to NumPy 2.4.2: the first steps run in the COEFFICIENTS' dtype exactly as NumPy's
        ///     do, and every intermediate carries NumPy's NEP 50 dtype.
        /// </summary>
        /// <param name="x">
        ///     The points. An <see cref="NDArray"/> (any layout) or a C# array is an array; a C# <c>bool</c>, integer,
        ///     <c>float</c>, <c>double</c> or <see cref="System.Numerics.Complex"/> is a <b>Python scalar</b> — it adopts
        ///     the coefficients' dtype (<c>legval(2.0, float32_c)</c> is float32), Python-level arithmetic on it
        ///     (<c>2*x</c>, <c>1 - x</c>) is done in double as CPython does, and a 1-D series then yields a 0-d result.
        ///     <c>char</c>/<see cref="System.Half"/>/<c>decimal</c> are strong 0-d arrays.
        /// </param>
        /// <param name="c">Coefficients (anything <c>np.array</c> accepts — an <see cref="NDArray"/> of any layout, a typed C# array, a Python tuple / list such as a ValueTuple, <c>object[]</c> or a jagged / <c>NDArray[]</c> array, nested to any depth, or a scalar), low degree first; for an N-D array the series run along axis 0 and the other
        ///     axes index independent series. Integer and bool coefficients are evaluated as float64.</param>
        /// <param name="tensor">When true (the default) and x is an array, every series is evaluated at every point
        ///     and the result has shape <c>c.shape[1:] + x.shape</c>; when false x is broadcast over the series
        ///     (<c>c.shape[1:]</c> against <c>x.shape</c>). Irrelevant for a 1-D series or a scalar x.</param>
        /// <returns>The values, of NumPy's result dtype.</returns>
        /// <exception cref="System.ArgumentNullException"><paramref name="x"/> is null.</exception>
        /// <exception cref="ValueError">A ragged tuple / list <paramref name="c"/> or <paramref name="x"/> (np.array's
        ///     inhomogeneous-shape text; c is converted first, so its error wins).</exception>
        /// <exception cref="System.NotSupportedException">A null or str <paramref name="c"/>, or a tuple / list holding a
        ///     str or null (NumPy builds an object / str array, dtypes NumSharp does not have).</exception>
        /// <exception cref="IndexError"><paramref name="c"/> is empty: <c>index -2 is out of bounds for axis 0 with size 0</c>.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="tensor"/> is false and the shapes do not broadcast
        ///     (NumPy's ValueError text).</exception>
        /// <exception cref="System.OverflowException">A Python int of the recurrence does not fit an integer x dtype
        ///     (NumPy's OverflowError).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.legendre.legval.html</remarks>
        public NDArray legval(object x, object c, bool tensor = true) => NDPolyEval.Val(PolyBasis.Legendre, x, c, tensor);

        /// <summary>
        ///     Evaluates the 2-D Legendre series <c>Σ c[i,j] * P_i(x) * P_j(y)</c> at the point pairs <c>(x, y)</c>.
        ///     NumPy's <c>_valnd</c>: <c>legval(x, c)</c> then <c>legval(y, ·, tensor=False)</c> — its exact
        ///     intermediate, so the bits match.
        /// </summary>
        /// <param name="x">First coordinates: an <see cref="NDArray"/>, a typed C# array, a Python tuple / list (np.array's coercion) or a
        ///     scalar, converted before anything else (NumPy's <c>np.asanyarray</c>) — so a scalar is a STRONG 0-d array
        ///     (a Python int an int64 one).</param>
        /// <param name="y">Second coordinates (converted like <paramref name="x"/>); must have x's shape.</param>
        /// <param name="c">Coefficients (anything <c>np.array</c> accepts — an <see cref="NDArray"/> of any layout, a typed C# array, a Python tuple / list such as a ValueTuple, <c>object[]</c> or a jagged / <c>NDArray[]</c> array, nested to any depth, or a scalar); <c>c[i,j]</c> is the coefficient of the degree (i, j) term. Extra trailing
        ///     axes index independent series (they lead the result's shape).</param>
        /// <returns>The values, shape <c>c.shape[2:] + x.shape</c>.</returns>
        /// <exception cref="System.ArgumentNullException">A coordinate (or <c>pts</c>) is null; a null c is
        ///     <see cref="System.NotSupportedException"/> (NumPy's object array).</exception>
        /// <exception cref="ValueError">x and y differ in shape (<c>x, y are incompatible</c>), or a tuple / list argument
        ///     is ragged (np.array's inhomogeneous-shape text: the coordinates' before the shape check, c's after
        ///     it).</exception>
        /// <exception cref="System.NotSupportedException">A null or str <paramref name="c"/>, a str coordinate, or a
        ///     tuple / list holding one (NumPy's object / str arrays).</exception>
        /// <exception cref="IndexError"><paramref name="c"/> is empty along axis 0.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.legendre.legval2d.html</remarks>
        public NDArray legval2d(object x, object y, object c) => NDPolyEval.ValNd(PolyBasis.Legendre, c, new[] { x, y });

        /// <summary>
        ///     Evaluates the 3-D Legendre series at the point triples <c>(x, y, z)</c> (NumPy's <c>_valnd</c>: one tensor
        ///     pass, then two per-point passes).
        /// </summary>
        /// <param name="x">First coordinates: an <see cref="NDArray"/>, a typed C# array, a Python tuple / list (np.array's coercion) or a
        ///     scalar, converted before anything else (NumPy's <c>np.asanyarray</c>) — so a scalar is a STRONG 0-d array
        ///     (a Python int an int64 one).</param>
        /// <param name="y">Second coordinates (converted like <paramref name="x"/>; x's shape).</param>
        /// <param name="z">Third coordinates (converted like <paramref name="x"/>; x's shape).</param>
        /// <param name="c">Coefficients (anything <c>np.array</c> accepts — an <see cref="NDArray"/> of any layout, a typed C# array, a Python tuple / list such as a ValueTuple, <c>object[]</c> or a jagged / <c>NDArray[]</c> array, nested to any depth, or a scalar); <c>c[i,j,k]</c> is the coefficient of the degree (i, j, k) term.</param>
        /// <returns>The values, shape <c>c.shape[3:] + x.shape</c>.</returns>
        /// <exception cref="System.ArgumentNullException">A coordinate (or <c>pts</c>) is null; a null c is
        ///     <see cref="System.NotSupportedException"/> (NumPy's object array).</exception>
        /// <exception cref="ValueError">The coordinates differ in shape (<c>x, y, z are incompatible</c>), or a tuple /
        ///     list argument is ragged (the coordinates' before the shape check, c's after it).</exception>
        /// <exception cref="System.NotSupportedException">A null or str <paramref name="c"/>, a str coordinate, or a
        ///     tuple / list holding one (NumPy's object / str arrays).</exception>
        /// <exception cref="IndexError"><paramref name="c"/> is empty along axis 0.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.legendre.legval3d.html</remarks>
        public NDArray legval3d(object x, object y, object z, object c) => NDPolyEval.ValNd(PolyBasis.Legendre, c, new[] { x, y, z });

        /// <summary>
        ///     Evaluates the 2-D Legendre series on the Cartesian product of <paramref name="x"/> and <paramref name="y"/>
        ///     (NumPy's <c>_gridnd</c>: two tensor passes). Unlike <see cref="legval2d"/> the coordinates are NOT
        ///     converted first, so a Python-scalar coordinate stays weak.
        /// </summary>
        /// <param name="x">First-axis points (array or Python scalar, as in <see cref="legval"/>).</param>
        /// <param name="y">Second-axis points.</param>
        /// <param name="c">Coefficients (anything <c>np.array</c> accepts — an <see cref="NDArray"/> of any layout, a typed C# array, a Python tuple / list such as a ValueTuple, <c>object[]</c> or a jagged / <c>NDArray[]</c> array, nested to any depth, or a scalar); <c>c[i,j]</c> is the coefficient of the degree (i, j) term.</param>
        /// <returns>The grid, shape <c>c.shape[2:] + x.shape + y.shape</c>.</returns>
        /// <exception cref="System.ArgumentNullException">A coordinate (or <c>pts</c>) is null; a null c is
        ///     <see cref="System.NotSupportedException"/> (NumPy's object array).</exception>
        /// <exception cref="IndexError"><paramref name="c"/> is empty along an evaluated axis.</exception>
        /// <exception cref="ValueError">A ragged tuple / list c or point argument (np.array's inhomogeneous-shape
        ///     text; c is converted before the first points).</exception>
        /// <exception cref="System.NotSupportedException">A null or str <paramref name="c"/>, or a tuple / list
        ///     holding a str / null.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.legendre.leggrid2d.html</remarks>
        public NDArray leggrid2d(object x, object y, object c) => NDPolyEval.GridNd(PolyBasis.Legendre, c, new[] { x, y });

        /// <summary>
        ///     Evaluates the 3-D Legendre series on the Cartesian product of <paramref name="x"/>, <paramref name="y"/> and
        ///     <paramref name="z"/> (NumPy's <c>_gridnd</c>: three tensor passes).
        /// </summary>
        /// <param name="x">First-axis points.</param>
        /// <param name="y">Second-axis points.</param>
        /// <param name="z">Third-axis points.</param>
        /// <param name="c">Coefficients (anything <c>np.array</c> accepts — an <see cref="NDArray"/> of any layout, a typed C# array, a Python tuple / list such as a ValueTuple, <c>object[]</c> or a jagged / <c>NDArray[]</c> array, nested to any depth, or a scalar); <c>c[i,j,k]</c> is the coefficient of the degree (i, j, k) term.</param>
        /// <returns>The grid, shape <c>c.shape[3:] + x.shape + y.shape + z.shape</c>.</returns>
        /// <exception cref="System.ArgumentNullException">A coordinate (or <c>pts</c>) is null; a null c is
        ///     <see cref="System.NotSupportedException"/> (NumPy's object array).</exception>
        /// <exception cref="IndexError"><paramref name="c"/> is empty along an evaluated axis.</exception>
        /// <exception cref="ValueError">A ragged tuple / list c or point argument (np.array's inhomogeneous-shape
        ///     text; c is converted before the first points).</exception>
        /// <exception cref="System.NotSupportedException">A null or str <paramref name="c"/>, or a tuple / list
        ///     holding a str / null.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.legendre.leggrid3d.html</remarks>
        public NDArray leggrid3d(object x, object y, object z, object c) => NDPolyEval.GridNd(PolyBasis.Legendre, c, new[] { x, y, z });

        /// <summary>
        ///     Evaluates the N-D Legendre series at the points <c>pts = (x, y, …)</c> — the n-dimensional
        ///     <see cref="legval2d"/>. Added in NumPy's development branch (<c>pu._valnd(legval, c, *pts)</c>); not in
        ///     NumPy 2.4.2, where the same computation is the private <c>polyutils._valnd</c>.
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
        /// <remarks>https://numpy.org/devdocs/reference/generated/numpy.polynomial.legendre.legvalnd.html</remarks>
        public NDArray legvalnd(object[] pts, object c) => NDPolyEval.ValNd(PolyBasis.Legendre, c, pts);

        // ---- U1: constants, line, add/sub, trim ----

        private static NDArray s_legdomain, s_legzero, s_legone, s_legx;

        /// <summary>
        ///     <c>legdomain</c> — the default domain of the Legendre series class, float64 <c>[-1.0, 1.0]</c>.
        /// </summary>
        /// <remarks>
        ///     NumPy's module constants are ordinary WRITEABLE ndarrays, the same object on every access, and so is
        ///     this one: a process-wide instance (a write through it persists, as in NumPy). It is shared — do not
        ///     dispose it (a dispose is harmless: the buffer is pinned for the process lifetime) and copy it before
        ///     mutating unless the change is meant to be global.
        ///     <para>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.legendre.html</para>
        /// </remarks>
        public NDArray legdomain => NDPolySeries.Constant(ref s_legdomain, static () => np.array(new double[] { -1.0, 1.0 }));

        /// <summary><c>legzero</c> — the Legendre series representing 0, int64 <c>[0]</c> (shared and writeable, see <see cref="legdomain"/>).</summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.legendre.html</remarks>
        public NDArray legzero => NDPolySeries.Constant(ref s_legzero, static () => np.array(new long[] { 0 }));

        /// <summary><c>legone</c> — the Legendre series representing 1, int64 <c>[1]</c> (shared and writeable, see <see cref="legdomain"/>).</summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.legendre.html</remarks>
        public NDArray legone => NDPolySeries.Constant(ref s_legone, static () => np.array(new long[] { 1 }));

        /// <summary><c>legx</c> — the Legendre series representing <c>x</c>, int64 <c>[0, 1]</c> (shared and writeable, see <see cref="legdomain"/>).</summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.legendre.html</remarks>
        public NDArray legx => NDPolySeries.Constant(ref s_legx, static () => np.array(new long[] { 0, 1 }));

        /// <summary>
        ///     The Legendre series of the line <c>off + scl*x</c>: <c>[off, scl]</c> when <c>scl != 0</c>, else <c>[off]</c> — built
        ///     with <c>np.array</c>, so the dtype is the one NumPy DISCOVERS for the entries (<c>legline(1, 2)</c> is int64,
        ///     <c>legline(1.0, 2)</c> float64, a bool pair stays bool) rather than a coefficient dtype.
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
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.legendre.legline.html</remarks>
        public NDArray legline(object off, object scl) => NDPolySeries.Line(PolyBasis.Legendre, off, scl);

        /// <summary>
        ///     Adds two Legendre series (<c>polyutils._add</c>): both converted to their common type (integers become float64),
        ///     the shorter added into a copy of the longer, trailing zeros trimmed.
        /// </summary>
        /// <param name="c1">First series, low degree first (an array, a C# array or list, or a single number).</param>
        /// <param name="c2">Second series.</param>
        /// <returns>The sum: a new array, or a VIEW of one when trailing zeros were trimmed (as in NumPy).</returns>
        /// <exception cref="ValueError">An empty or non-1-d series, or no common type (a bool series).</exception>
        /// <exception cref="System.NotSupportedException">A null series, or one holding None, a non-numeric object or a Python int past uint64 (NumPy's object array), raised after both series' checks.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.legendre.legadd.html</remarks>
        [NDScoped]
        public NDArray legadd(object c1, object c2) => NDPolySeries.AddSub(c1, c2, subtract: false);

        /// <summary>
        ///     Subtracts the Legendre series <paramref name="c2"/> from <paramref name="c1"/> (<c>polyutils._sub</c>): both converted
        ///     to their common type; when <paramref name="c2"/> is at least as long it is NEGATED first and
        ///     <paramref name="c1"/> added into it (NumPy's operation order, visible in which NaN survives), then trailing
        ///     zeros are trimmed.
        /// </summary>
        /// <param name="c1">Minuend series, low degree first.</param>
        /// <param name="c2">Subtrahend series.</param>
        /// <returns>The difference: a new array, or a VIEW of one when trailing zeros were trimmed.</returns>
        /// <exception cref="ValueError">An empty or non-1-d series, or no common type (a bool series).</exception>
        /// <exception cref="System.NotSupportedException">A null series, or one holding None, a non-numeric object or a Python int past uint64 (NumPy's object array), raised after both series' checks.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.legendre.legsub.html</remarks>
        [NDScoped]
        public NDArray legsub(object c1, object c2) => NDPolySeries.AddSub(c1, c2, subtract: true);

        /// <summary>
        ///     <c>legtrim</c> — NumPy's alias of <see cref="PolyUtilsModule.trimcoef(object)"/>: removes the trailing
        ///     zero coefficients (tolerance 0) and returns a fresh copy of the coefficient dtype.
        /// </summary>
        /// <param name="c">The series (1-D).</param>
        /// <returns>The trimmed copy (an all-zero series becomes <c>c[:1]*0</c>).</returns>
        /// <exception cref="ValueError">An empty or non-1-d series, or no common type.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.legendre.legtrim.html</remarks>
        [NDScoped]
        public NDArray legtrim(object c) => NDPolySeries.TrimCoef(c, PolyNumber.FromPython(PyScalar.Int(0)));

        /// <summary>
        ///     <c>legtrim</c> — NumPy's alias of <see cref="PolyUtilsModule.trimcoef(object, object)"/>: removes the
        ///     trailing coefficients whose magnitude is at most <paramref name="tol"/> and returns a fresh copy.
        /// </summary>
        /// <param name="c">The series (1-D).</param>
        /// <param name="tol">The non-negative tolerance (a Python number, or a NumPy scalar compared in the promoted dtype).</param>
        /// <returns>The trimmed copy (<c>c[:1]*0</c> when nothing exceeds the tolerance).</returns>
        /// <exception cref="ValueError"><c>tol must be non-negative</c>, an empty or non-1-d series, or no common type.</exception>
        /// <exception cref="TypeError">A complex or null tolerance.</exception>
        /// <exception cref="OverflowException">A Python int tolerance too large to convert.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.legendre.legtrim.html</remarks>
        [NDScoped]
        public NDArray legtrim(object c, object tol) => NDPolySeries.TrimCoef(c, PolyUtilsModule.Tolerance(tol));

        /// <summary>
        ///     Differentiates the Legendre series <paramref name="c"/> <paramref name="m"/> times along <paramref name="axis"/>,
        ///     multiplying by <paramref name="scl"/> at every order. Bit-identical to NumPy 2.4.2's <c>legder</c>: its
        ///     recurrence <c>der[j-1] = (2*j - 1)*c[j]; c[j-2] += c[j]</c> (then <c>der[1] = 3*c[2]</c>,
        ///     <c>der[0] = c[1]</c>), with <c>c *= scl</c> before every order, in NumPy's statement order and NEP 50 dtypes.
        /// </summary>
        /// <param name="c">The coefficients, LOW degree first (see <see cref="PowerSeriesModule.polyder"/>).</param>
        /// <param name="m">How many times to differentiate (≥ 0). 0 returns a copy; <c>m &gt;= len(c)</c> returns
        ///     <c>c[:1]*0</c>.</param>
        /// <param name="scl">The multiplier applied at each order (null for NumPy's 1; a scalar or a broadcasting array).</param>
        /// <param name="axis">The axis the series runs along (negative counts from the end).</param>
        /// <returns>The derivative's coefficients, of the series' dtype (a view of a new array in NumPy's layout).</returns>
        /// <exception cref="ValueError"><c>The order of derivation must be non-negative</c>, or an array scl whose
        ///     broadcast would stretch the series.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range.</exception>
        /// <exception cref="System.ArgumentException">A scl the series cannot absorb in place (NumPy's UFuncTypeError).</exception>
        /// <exception cref="IncorrectShapeException">An array scl that does not broadcast with the series.</exception>
        /// <exception cref="System.NotSupportedException">A null or string series.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.legendre.legder.html</remarks>
        [NDScoped]
        public NDArray legder(object c, int m = 1, object scl = null, int axis = 0) => NDPolyCalc.Der(PolyBasis.Legendre, c, m, scl, axis);

        /// <summary>
        ///     Integrates the Legendre series <paramref name="c"/> <paramref name="m"/> times along <paramref name="axis"/>:
        ///     each order multiplies by <paramref name="scl"/>, integrates (<c>t = c[j]/(2*j + 1); tmp[j+1] = t;
        ///     tmp[j-1] -= t</c>, with <c>tmp[2] = c[1]/3</c>) and adds the constant that makes the new series equal
        ///     <c>k[i]</c> at <paramref name="lbnd"/> (<c>tmp[0] += k[i] - legval(lbnd, tmp)</c>). Bit-identical to NumPy
        ///     2.4.2's <c>legint</c>.
        /// </summary>
        /// <param name="c">The coefficients, LOW degree first.</param>
        /// <param name="m">How many times to integrate (≥ 0); 0 returns a copy.</param>
        /// <param name="k">The integration constants (see <see cref="PowerSeriesModule.polyint"/>).</param>
        /// <param name="lbnd">The lower bound (null for NumPy's 0); a scalar.</param>
        /// <param name="scl">The multiplier applied at each order (null for NumPy's 1); a scalar.</param>
        /// <param name="axis">The axis the series runs along.</param>
        /// <returns>The integral's coefficients, of the series' dtype (a view of a new array in NumPy's layout).</returns>
        /// <exception cref="ValueError">NumPy's argument checks (see <see cref="PowerSeriesModule.polyint"/>).</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range.</exception>
        /// <exception cref="IndexError">An empty series.</exception>
        /// <exception cref="TypeError">An array constant for a complex 1-D series (NumPy's <c>complex()</c> text).</exception>
        /// <exception cref="System.ArgumentException">A value an N-D series cannot absorb in place (NumPy's UFuncTypeError).</exception>
        /// <exception cref="IncorrectShapeException">Constants that do not broadcast with the series' columns.</exception>
        /// <exception cref="System.NotSupportedException">A null or string series, constant or bound.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.legendre.legint.html</remarks>
        [NDScoped]
        public NDArray legint(object c, int m = 1, object k = null, object lbnd = null, object scl = null, int axis = 0)
            => NDPolyCalc.Int(PolyBasis.Legendre, c, m, k, lbnd, scl, axis);

        // ---------------------------------------------------------------------------------------------
        //  Series algebra (plan U2): legmulx / legmul / legdiv / legpow / legfromroots / leg2poly / poly2leg
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     Multiplies the Legendre series <paramref name="c"/> by x: <c>x*P_i = ((i+1)*P_{i+1} + i*P_{i-1})/(2i+1)</c>: NumPy's <c>prd[i+1] = (c[i]*(i+1))/(2i+1); prd[i-1] += (c[i]*i)/(2i+1)</c> per coefficient. The zero series <c>[0]</c> is returned as is.
        ///     Bit-identical to NumPy 2.4.2's <c>legmulx</c> — one IL kernel pass where NumPy runs a Python loop.
        /// </summary>
        /// <param name="c">The coefficients, LOW degree first: an <see cref="NDArray"/> (any layout), a typed C# array, a Python list or tuple (<c>object[]</c>, a <c>ValueTuple</c>, a jagged array — np.array's coercion), or a single number. Integer series are computed in float64; a bool series has no common type.</param>
        /// <returns>A new array one coefficient longer (the zero series: a one-element copy), of <c>np.common_type(c)</c>.</returns>
        /// <exception cref="ValueError">An empty series (<c>Coefficient array is empty</c>), a series that is not 1-d
        ///     (<c>Coefficient array is not 1-d</c>), or no common type (a bool or str series), in NumPy's order; a ragged list
        ///     (np.array's inhomogeneous-shape text).</exception>
        /// <exception cref="System.NotSupportedException">A null series, or one holding None, a non-numeric object or a Python int past
        ///     uint64 (NumPy's object array, a dtype NumSharp does not have) — raised where NumPy starts computing with
        ///     it, after every series check. A str series is NumPy's ValueError (no common type).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.legendre.legmulx.html</remarks>
        [NDScoped]
        public NDArray legmulx(object c) => NDPolyAlgebra.Mulx(PolyBasis.Legendre, c);

        /// <summary>
        ///     Multiplies two Legendre series: NumPy's backward recurrence over series — for each coefficient of the shorter factor, <c>c0 = legsub(c[-i]*xs, (c1*(nd-1))/nd); c1 = legadd(tmp, (legmulx(c1)*(2*nd-1))/nd)</c> — ending in <c>legadd(c0, legmulx(c1))</c>. When either factor is a single term, NumPy's recurrence binds the Python int 0, so a float16 / float32 product is float64 (probed 2.4.2). Bit-identical to NumPy 2.4.2: every statement runs through the kernel NumPy's statement runs through (the house ufunc loops, the U1 combine, the mulx kernel) in NumPy's order and dtypes.
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
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.legendre.legmul.html</remarks>
        [NDScoped]
        public NDArray legmul(object c1, object c2) => NDPolyAlgebra.Mul(PolyBasis.Legendre, c1, c2);

        /// <summary>
        ///     Divides the Legendre series <paramref name="c1"/> by <paramref name="c2"/>, returning quotient and remainder:
        ///     <c>polyutils._div</c>: for each quotient coefficient, the divisor times the i-th basis polynomial (<c>legmul([0]*i + [1], c2)</c>) is scaled and subtracted from the remainder. A dividend shorter than the divisor gives <c>(c1[:1]*0, c1)</c>, a one-term divisor
        ///     <c>(c1/c2[-1], c1[:1]*0)</c>. The quotient keeps the common type of the series, but the remainder is computed against divisor products NumPy promotes with an int list, so it is float64 for float16 / float32 series (probed 2.4.2). Bit-identical to NumPy 2.4.2's <c>legdiv</c>.
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
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.legendre.legdiv.html</remarks>
        [NDScoped]
        public (NDArray quo, NDArray rem) legdiv(object c1, object c2) => NDPolyAlgebra.Div(PolyBasis.Legendre, c1, c2);

        /// <summary>
        ///     Raises the Legendre series <paramref name="c"/> to the power <paramref name="pow"/>: <c>[1]</c> of the series' dtype
        ///     for 0, the (trimmed) series for 1, otherwise <c>legmul</c> applied pow - 1 times. Bit-identical to NumPy 2.4.2's <c>legpow</c>.
        /// </summary>
        /// <param name="c">The coefficients, LOW degree first: an <see cref="NDArray"/> (any layout), a typed C# array, a Python list or tuple (<c>object[]</c>, a <c>ValueTuple</c>, a jagged array — np.array's coercion), or a single number. Integer series are computed in float64; a bool series has no common type.</param>
        /// <param name="pow">The power (≥ 0).</param>
        /// <param name="maxpower">The largest power allowed: 16; null for no limit. NumPy's default keeps an accidental huge power from
        ///     running away.</param>
        /// <returns>The power series, a new array of <c>np.common_type(c)</c>.</returns>
        /// <exception cref="ValueError">The series' errors (see <see cref="legmul"/>), then
        ///     <c>Power must be a non-negative integer.</c>, then <c>Power is too large</c> — NumPy's order.</exception>
        /// <exception cref="System.NotSupportedException">An object series (see <see cref="legmul"/>), raised after the power checks.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.legendre.legpow.html</remarks>
        [NDScoped]
        public NDArray legpow(object c, int pow, int? maxpower = 16) => NDPolyAlgebra.Pow(PolyBasis.Legendre, c, pow, maxpower);

        /// <summary>
        ///     <see cref="legpow(object, int, int?)"/> with a Python-float power: NumPy's <c>power = int(pow)</c>
        ///     truncates, so 2.0 is 2 while 2.5 fails <c>power != pow</c>.
        /// </summary>
        /// <param name="c">The coefficients, LOW degree first: an <see cref="NDArray"/> (any layout), a typed C# array, a Python list or tuple (<c>object[]</c>, a <c>ValueTuple</c>, a jagged array — np.array's coercion), or a single number. Integer series are computed in float64; a bool series has no common type.</param>
        /// <param name="pow">The power; it must equal its integer part and be ≥ 0.</param>
        /// <param name="maxpower">The largest power allowed: 16; null for no limit.</param>
        /// <returns>The power series.</returns>
        /// <exception cref="ValueError">The series' errors; <c>cannot convert float NaN to integer</c>;
        ///     <c>Power must be a non-negative integer.</c> (a fractional or negative power); <c>Power is too large</c>.</exception>
        /// <exception cref="System.OverflowException"><c>cannot convert float infinity to integer</c>; a power beyond
        ///     int64 with no limit (NumPy would run until memory is exhausted).</exception>
        /// <exception cref="System.NotSupportedException">An object series (see <see cref="legmul"/>), raised after the power checks.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.legendre.legpow.html</remarks>
        [NDScoped]
        public NDArray legpow(object c, double pow, int? maxpower = 16) => NDPolyAlgebra.Pow(PolyBasis.Legendre, c, pow, maxpower);

        /// <summary>
        ///     <see cref="legpow(object, int, int?)"/> for ANY power argument, with NumPy's default limit 16: <c>power =
        ///     int(pow)</c> with Python's semantics — a bool is 0 / 1, np.float16 (Half) or decimal truncates like a float, a
        ///     str parses with int()'s grammar (and then fails <c>power != pow</c>), a 0-d NDArray converts its element — and
        ///     NumPy's error for every value int() refuses.
        /// </summary>
        /// <param name="c">The coefficients, LOW degree first: an <see cref="NDArray"/> (any layout), a typed C# array, a Python list or tuple (<c>object[]</c>, a <c>ValueTuple</c>, a jagged array — np.array's coercion), or a single number. Integer series are computed in float64; a bool series has no common type.</param>
        /// <param name="pow">The power: any value (see the summary).</param>
        /// <returns>The power series.</returns>
        /// <exception cref="ValueError">The series' errors (see <see cref="legmul"/>); <c>cannot convert float NaN to
        ///     integer</c>; <c>invalid literal for int() with base 10: '…'</c>; <c>Power must be a non-negative
        ///     integer.</c>; <c>Power is too large</c>.</exception>
        /// <exception cref="TypeError">A power int() refuses: a complex, null (None), a list or tuple (<c>int() argument must
        ///     be a string, a bytes-like object or a real number, not '…'</c>), an ndarray of one or more dims (<c>only
        ///     0-dimensional arrays can be converted to Python scalars</c>).</exception>
        /// <exception cref="System.OverflowException"><c>cannot convert float infinity to integer</c>; a power beyond int64
        ///     with no limit (NumPy would run until memory is exhausted).</exception>
        /// <exception cref="System.NotSupportedException">An object series (see <see cref="legmul"/>), raised after the
        ///     power checks.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.legendre.legpow.html</remarks>
        [NDScoped]
        public NDArray legpow(object c, object pow) => NDPolyAlgebra.Pow(PolyBasis.Legendre, c, pow, NDPolyAlgebra.DefaultMaxPower);

        /// <summary>
        ///     <see cref="legpow(object, object)"/> with ANY limit: null is None (no limit); anything else is compared as
        ///     NumPy compares <c>power &gt; maxpower</c> — exactly for a Python int / bool / float, a NumPy integer scalar
        ///     (char) or decimal; in float16 for np.float16 (Half); for an ndarray (an NDArray, a typed C# array) in its dtype
        ///     (float16 / float32 / float64 rounding the power first, complex lexicographically, bool as int64, integers
        ///     exactly), the comparison's truth value then needing exactly one element.
        /// </summary>
        /// <param name="c">The coefficients, LOW degree first: an <see cref="NDArray"/> (any layout), a typed C# array, a Python list or tuple (<c>object[]</c>, a <c>ValueTuple</c>, a jagged array — np.array's coercion), or a single number. Integer series are computed in float64; a bool series has no common type.</param>
        /// <param name="pow">The power: any value (see <see cref="legpow(object, object)"/>).</param>
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
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.legendre.legpow.html</remarks>
        [NDScoped]
        public NDArray legpow(object c, object pow, object maxpower) => NDPolyAlgebra.Pow(PolyBasis.Legendre, c, pow, maxpower);

        /// <summary>
        ///     The Legendre series whose roots are <paramref name="roots"/>: the roots sorted, one linear factor per root
        ///     (<c>legline(-r, 1) = [-r, 1]</c>), multiplied pairwise as a balanced tree (<c>polyutils._fromroots</c>) with <c>legmul</c>.
        ///     Empty roots give <c>[1.]</c>. The factors are built by <c>np.array([-r, …])</c>, whose coercion promotes a
        ///     float16 / float32 root to float64 — so the series is float64 (complex128 for complex roots). Bit-identical to
        ///     NumPy 2.4.2's <c>legfromroots</c>; one documented difference: roots that compare equal but differ in bits (+0.0 and -0.0) are ordered
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
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.legendre.legfromroots.html</remarks>
        [NDScoped]
        public NDArray legfromroots(object roots) => NDPolyAlgebra.FromRoots(PolyBasis.Legendre, roots);

        /// <summary>
        ///     Converts the Legendre series <paramref name="c"/> to a power series — NumPy's backward recurrence over power-series
        ///     ops (<c>polyadd</c>, <c>polysub</c>, <c>polymulx</c>) from the highest coefficient down; a series of fewer than three terms is returned as is. Keeps the
        ///     series' common type. Bit-identical to NumPy 2.4.2's <c>leg2poly</c>.
        /// </summary>
        /// <param name="c">The Legendre series's coefficients. The coefficients, LOW degree first: an <see cref="NDArray"/> (any layout), a typed C# array, a Python list or tuple (<c>object[]</c>, a <c>ValueTuple</c>, a jagged array — np.array's coercion), or a single number. Integer series are computed in float64; a bool series has no common type.</param>
        /// <returns>The power-series coefficients, low degree first (a new array, or a view of one when trailing zeros
        ///     were trimmed).</returns>
        /// <exception cref="ValueError">An empty series (<c>Coefficient array is empty</c>), a series that is not 1-d
        ///     (<c>Coefficient array is not 1-d</c>), or no common type (a bool or str series), in NumPy's order; a ragged list
        ///     (np.array's inhomogeneous-shape text).</exception>
        /// <exception cref="System.NotSupportedException">A null series, or one holding None, a non-numeric object or a Python int past
        ///     uint64 (NumPy's object array, a dtype NumSharp does not have) — raised where NumPy starts computing with
        ///     it, after every series check. A str series is NumPy's ValueError (no common type).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.legendre.leg2poly.html</remarks>
        [NDScoped]
        public NDArray leg2poly(object c) => NDPolyAlgebra.ToPower(PolyBasis.Legendre, c);

        /// <summary>
        ///     Converts the power series <paramref name="pol"/> to a Legendre series — Horner's scheme over the Legendre series's own ops:
        ///     <c>res = legadd(legmulx(res), pol[i])</c> from the highest coefficient down, starting from the Python int 0,
        ///     which makes the result float64 for any real series (complex128 for a complex one). Bit-identical to NumPy
        ///     2.4.2's <c>poly2leg</c>.
        /// </summary>
        /// <param name="pol">The power-series coefficients. The coefficients, LOW degree first: an <see cref="NDArray"/> (any layout), a typed C# array, a Python list or tuple (<c>object[]</c>, a <c>ValueTuple</c>, a jagged array — np.array's coercion), or a single number. Integer series are computed in float64; a bool series has no common type.</param>
        /// <returns>The Legendre series's coefficients, low degree first (a new array, or a view of one when trailing zeros
        ///     were trimmed).</returns>
        /// <exception cref="ValueError">An empty series (<c>Coefficient array is empty</c>), a series that is not 1-d
        ///     (<c>Coefficient array is not 1-d</c>), or no common type (a bool or str series), in NumPy's order; a ragged list
        ///     (np.array's inhomogeneous-shape text).</exception>
        /// <exception cref="System.NotSupportedException">A null series, or one holding None, a non-numeric object or a Python int past
        ///     uint64 (NumPy's object array, a dtype NumSharp does not have) — raised where NumPy starts computing with
        ///     it, after every series check. A str series is NumPy's ValueError (no common type).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.legendre.poly2leg.html</remarks>
        [NDScoped]
        public NDArray poly2leg(object pol) => NDPolyAlgebra.FromPower(PolyBasis.Legendre, pol);
    }
}
