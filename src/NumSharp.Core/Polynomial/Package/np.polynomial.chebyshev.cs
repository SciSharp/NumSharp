using NumSharp.Backends.Kernels;

namespace NumSharp
{
    /// <summary>
    ///     <c>numpy.polynomial.chebyshev</c> — Chebyshev series of the first kind <c>p(x) = c[0]*T_0(x) + c[1]*T_1(x) + …</c>, reachable as
    ///     <c>np.polynomial.chebyshev</c>. Coefficients are LOW degree first.
    /// </summary>
    /// <remarks>https://numpy.org/doc/stable/reference/routines.polynomials.chebyshev.html</remarks>
    [ModuleName("np.polynomial.chebyshev")]
    public class ChebyshevModule
    {
        /// <summary>Constructs the module; there is one shared instance behind <c>np.polynomial.chebyshev</c>.</summary>
        internal ChebyshevModule() { }

        /// <summary>
        ///     Evaluates the Chebyshev series of the first kind with coefficients <paramref name="c"/> at <paramref name="x"/> by Clenshaw's recurrence exactly as NumPy writes it (<c>x2 = 2*x; c0 = c[-i] - c1; c1 = tmp + c1*x2; return c0 + c1*x</c>),
        ///     bit-identical to NumPy 2.4.2: the first steps run in the COEFFICIENTS' dtype exactly as NumPy's
        ///     do, and every intermediate carries NumPy's NEP 50 dtype.
        /// </summary>
        /// <param name="x">
        ///     The points. An <see cref="NDArray"/> (any layout) or a C# array is an array; a C# <c>bool</c>, integer,
        ///     <c>float</c>, <c>double</c> or <see cref="System.Numerics.Complex"/> is a <b>Python scalar</b> — it adopts
        ///     the coefficients' dtype (<c>chebval(2.0, float32_c)</c> is float32), Python-level arithmetic on it
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
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.chebyshev.chebval.html</remarks>
        public NDArray chebval(object x, object c, bool tensor = true) => NDPolyEval.Val(PolyBasis.Chebyshev, x, c, tensor);

        /// <summary>
        ///     Evaluates the 2-D Chebyshev series of the first kind <c>Σ c[i,j] * T_i(x) * T_j(y)</c> at the point pairs <c>(x, y)</c>.
        ///     NumPy's <c>_valnd</c>: <c>chebval(x, c)</c> then <c>chebval(y, ·, tensor=False)</c> — its exact
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
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.chebyshev.chebval2d.html</remarks>
        public NDArray chebval2d(object x, object y, object c) => NDPolyEval.ValNd(PolyBasis.Chebyshev, c, new[] { x, y });

        /// <summary>
        ///     Evaluates the 3-D Chebyshev series of the first kind at the point triples <c>(x, y, z)</c> (NumPy's <c>_valnd</c>: one tensor
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
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.chebyshev.chebval3d.html</remarks>
        public NDArray chebval3d(object x, object y, object z, object c) => NDPolyEval.ValNd(PolyBasis.Chebyshev, c, new[] { x, y, z });

        /// <summary>
        ///     Evaluates the 2-D Chebyshev series of the first kind on the Cartesian product of <paramref name="x"/> and <paramref name="y"/>
        ///     (NumPy's <c>_gridnd</c>: two tensor passes). Unlike <see cref="chebval2d"/> the coordinates are NOT
        ///     converted first, so a Python-scalar coordinate stays weak.
        /// </summary>
        /// <param name="x">First-axis points (array or Python scalar, as in <see cref="chebval"/>).</param>
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
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.chebyshev.chebgrid2d.html</remarks>
        public NDArray chebgrid2d(object x, object y, object c) => NDPolyEval.GridNd(PolyBasis.Chebyshev, c, new[] { x, y });

        /// <summary>
        ///     Evaluates the 3-D Chebyshev series of the first kind on the Cartesian product of <paramref name="x"/>, <paramref name="y"/> and
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
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.chebyshev.chebgrid3d.html</remarks>
        public NDArray chebgrid3d(object x, object y, object z, object c) => NDPolyEval.GridNd(PolyBasis.Chebyshev, c, new[] { x, y, z });

        /// <summary>
        ///     Evaluates the N-D Chebyshev series of the first kind at the points <c>pts = (x, y, …)</c> — the n-dimensional
        ///     <see cref="chebval2d"/>. Added in NumPy's development branch (<c>pu._valnd(chebval, c, *pts)</c>); not in
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
        /// <remarks>https://numpy.org/devdocs/reference/generated/numpy.polynomial.chebyshev.chebvalnd.html</remarks>
        public NDArray chebvalnd(object[] pts, object c) => NDPolyEval.ValNd(PolyBasis.Chebyshev, c, pts);

        // ---- U1: constants, line, add/sub, trim ----

        private static NDArray s_chebdomain, s_chebzero, s_chebone, s_chebx;

        /// <summary>
        ///     <c>chebdomain</c> — the default domain of the Chebyshev series class, float64 <c>[-1.0, 1.0]</c>.
        /// </summary>
        /// <remarks>
        ///     NumPy's module constants are ordinary WRITEABLE ndarrays, the same object on every access, and so is
        ///     this one: a process-wide instance (a write through it persists, as in NumPy). It is shared — do not
        ///     dispose it (a dispose is harmless: the buffer is pinned for the process lifetime) and copy it before
        ///     mutating unless the change is meant to be global.
        ///     <para>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.chebyshev.html</para>
        /// </remarks>
        public NDArray chebdomain => NDPolySeries.Constant(ref s_chebdomain, static () => np.array(new double[] { -1.0, 1.0 }));

        /// <summary><c>chebzero</c> — the Chebyshev series representing 0, int64 <c>[0]</c> (shared and writeable, see <see cref="chebdomain"/>).</summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.chebyshev.html</remarks>
        public NDArray chebzero => NDPolySeries.Constant(ref s_chebzero, static () => np.array(new long[] { 0 }));

        /// <summary><c>chebone</c> — the Chebyshev series representing 1, int64 <c>[1]</c> (shared and writeable, see <see cref="chebdomain"/>).</summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.chebyshev.html</remarks>
        public NDArray chebone => NDPolySeries.Constant(ref s_chebone, static () => np.array(new long[] { 1 }));

        /// <summary><c>chebx</c> — the Chebyshev series representing <c>x</c>, int64 <c>[0, 1]</c> (shared and writeable, see <see cref="chebdomain"/>).</summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.chebyshev.html</remarks>
        public NDArray chebx => NDPolySeries.Constant(ref s_chebx, static () => np.array(new long[] { 0, 1 }));

        /// <summary>
        ///     The Chebyshev series of the line <c>off + scl*x</c>: <c>[off, scl]</c> when <c>scl != 0</c>, else <c>[off]</c> — built
        ///     with <c>np.array</c>, so the dtype is the one NumPy DISCOVERS for the entries (<c>chebline(1, 2)</c> is int64,
        ///     <c>chebline(1.0, 2)</c> float64, a bool pair stays bool) rather than a coefficient dtype.
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
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.chebyshev.chebline.html</remarks>
        public NDArray chebline(object off, object scl) => NDPolySeries.Line(PolyBasis.Chebyshev, off, scl);

        /// <summary>
        ///     Adds two Chebyshev series (<c>polyutils._add</c>): both converted to their common type (integers become float64),
        ///     the shorter added into a copy of the longer, trailing zeros trimmed.
        /// </summary>
        /// <param name="c1">First series, low degree first (an array, a C# array or list, or a single number).</param>
        /// <param name="c2">Second series.</param>
        /// <returns>The sum: a new array, or a VIEW of one when trailing zeros were trimmed (as in NumPy).</returns>
        /// <exception cref="ValueError">An empty or non-1-d series, or no common type (a bool series).</exception>
        /// <exception cref="System.NotSupportedException">A null series, or one holding None, a non-numeric object or a Python int past uint64 (NumPy's object array), raised after both series' checks.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.chebyshev.chebadd.html</remarks>
        [NDScoped]
        public NDArray chebadd(object c1, object c2) => NDPolySeries.AddSub(c1, c2, subtract: false);

        /// <summary>
        ///     Subtracts the Chebyshev series <paramref name="c2"/> from <paramref name="c1"/> (<c>polyutils._sub</c>): both converted
        ///     to their common type; when <paramref name="c2"/> is at least as long it is NEGATED first and
        ///     <paramref name="c1"/> added into it (NumPy's operation order, visible in which NaN survives), then trailing
        ///     zeros are trimmed.
        /// </summary>
        /// <param name="c1">Minuend series, low degree first.</param>
        /// <param name="c2">Subtrahend series.</param>
        /// <returns>The difference: a new array, or a VIEW of one when trailing zeros were trimmed.</returns>
        /// <exception cref="ValueError">An empty or non-1-d series, or no common type (a bool series).</exception>
        /// <exception cref="System.NotSupportedException">A null series, or one holding None, a non-numeric object or a Python int past uint64 (NumPy's object array), raised after both series' checks.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.chebyshev.chebsub.html</remarks>
        [NDScoped]
        public NDArray chebsub(object c1, object c2) => NDPolySeries.AddSub(c1, c2, subtract: true);

        /// <summary>
        ///     <c>chebtrim</c> — NumPy's alias of <see cref="PolyUtilsModule.trimcoef(object)"/>: removes the trailing
        ///     zero coefficients (tolerance 0) and returns a fresh copy of the coefficient dtype.
        /// </summary>
        /// <param name="c">The series (1-D).</param>
        /// <returns>The trimmed copy (an all-zero series becomes <c>c[:1]*0</c>).</returns>
        /// <exception cref="ValueError">An empty or non-1-d series, or no common type.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.chebyshev.chebtrim.html</remarks>
        [NDScoped]
        public NDArray chebtrim(object c) => NDPolySeries.TrimCoef(c, PolyNumber.FromPython(PyScalar.Int(0)));

        /// <summary>
        ///     <c>chebtrim</c> — NumPy's alias of <see cref="PolyUtilsModule.trimcoef(object, object)"/>: removes the
        ///     trailing coefficients whose magnitude is at most <paramref name="tol"/> and returns a fresh copy.
        /// </summary>
        /// <param name="c">The series (1-D).</param>
        /// <param name="tol">The non-negative tolerance (a Python number, or a NumPy scalar compared in the promoted dtype).</param>
        /// <returns>The trimmed copy (<c>c[:1]*0</c> when nothing exceeds the tolerance).</returns>
        /// <exception cref="ValueError"><c>tol must be non-negative</c>, an empty or non-1-d series, or no common type.</exception>
        /// <exception cref="TypeError">A complex or null tolerance.</exception>
        /// <exception cref="OverflowException">A Python int tolerance too large to convert.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.chebyshev.chebtrim.html</remarks>
        [NDScoped]
        public NDArray chebtrim(object c, object tol) => NDPolySeries.TrimCoef(c, PolyUtilsModule.Tolerance(tol));

        /// <summary>
        ///     Differentiates the Chebyshev series <paramref name="c"/> <paramref name="m"/> times along <paramref name="axis"/>,
        ///     multiplying by <paramref name="scl"/> at every order. Bit-identical to NumPy 2.4.2's <c>chebder</c>: its
        ///     recurrence <c>der[j-1] = (2*j)*c[j]; c[j-2] += (j*c[j])/(j-2)</c> (then <c>der[1] = 4*c[2]</c>,
        ///     <c>der[0] = c[1]</c>), with <c>c *= scl</c> before every order, in NumPy's statement order and NEP 50 dtypes.
        /// </summary>
        /// <param name="c">The coefficients, LOW degree first (see <see cref="PowerSeriesModule.polyder"/>).</param>
        /// <param name="m">How many times to differentiate (≥ 0). 0 returns a copy; <c>m &gt;= len(c)</c> returns
        ///     <c>c[:1]*0</c>.</param>
        /// <param name="scl">The multiplier applied at each order (null for NumPy's 1; a scalar or a broadcasting array —
        ///     see <see cref="PowerSeriesModule.polyder"/>).</param>
        /// <param name="axis">The axis the series runs along (negative counts from the end).</param>
        /// <returns>The derivative's coefficients, of the series' dtype (a view of a new array in NumPy's layout).</returns>
        /// <exception cref="ValueError"><c>The order of derivation must be non-negative</c>, or an array scl whose
        ///     broadcast would stretch the series.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range.</exception>
        /// <exception cref="System.ArgumentException">A scl the series cannot absorb in place (NumPy's UFuncTypeError).</exception>
        /// <exception cref="IncorrectShapeException">An array scl that does not broadcast with the series.</exception>
        /// <exception cref="System.NotSupportedException">A null or string series.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.chebyshev.chebder.html</remarks>
        [NDScoped]
        public NDArray chebder(object c, int m = 1, object scl = null, int axis = 0) => NDPolyCalc.Der(PolyBasis.Chebyshev, c, m, scl, axis);

        /// <summary>
        ///     Integrates the Chebyshev series <paramref name="c"/> <paramref name="m"/> times along <paramref name="axis"/>:
        ///     each order multiplies by <paramref name="scl"/>, integrates (<c>tmp[j+1] = c[j]/(2*(j+1));
        ///     tmp[j-1] -= c[j]/(2*(j-1))</c>, with <c>tmp[2] = c[1]/4</c>) and adds the constant that makes the new series
        ///     equal <c>k[i]</c> at <paramref name="lbnd"/> (<c>tmp[0] += k[i] - chebval(lbnd, tmp)</c>). Bit-identical to
        ///     NumPy 2.4.2's <c>chebint</c>.
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
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.chebyshev.chebint.html</remarks>
        [NDScoped]
        public NDArray chebint(object c, int m = 1, object k = null, object lbnd = null, object scl = null, int axis = 0)
            => NDPolyCalc.Int(PolyBasis.Chebyshev, c, m, k, lbnd, scl, axis);

        // ---------------------------------------------------------------------------------------------
        //  Series algebra (plan U2): chebmulx / chebmul / chebdiv / chebpow / chebfromroots / cheb2poly / poly2cheb
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     Multiplies the Chebyshev series <paramref name="c"/> by x: <c>x*T_0 = T_1</c>, <c>x*T_i = (T_{i-1} + T_{i+1})/2</c>: NumPy's <c>prd[0] = c[0]*0; prd[1] = c[0]; tmp = c[1:]/2; prd[2:] = tmp; prd[0:-2] += tmp</c>. The zero series <c>[0]</c> is returned as is.
        ///     Bit-identical to NumPy 2.4.2's <c>chebmulx</c> — one IL kernel pass where NumPy runs a Python loop.
        /// </summary>
        /// <param name="c">The coefficients, LOW degree first: an <see cref="NDArray"/> (any layout), a typed C# array, a Python list or tuple (<c>object[]</c>, a <c>ValueTuple</c>, a jagged array — np.array's coercion), or a single number. Integer series are computed in float64; a bool series has no common type.</param>
        /// <returns>A new array one coefficient longer (the zero series: a one-element copy), of <c>np.common_type(c)</c>.</returns>
        /// <exception cref="ValueError">An empty series (<c>Coefficient array is empty</c>), a series that is not 1-d
        ///     (<c>Coefficient array is not 1-d</c>), or no common type (a bool or str series), in NumPy's order; a ragged list
        ///     (np.array's inhomogeneous-shape text).</exception>
        /// <exception cref="System.NotSupportedException">A null series, or one holding None, a non-numeric object or a Python int past
        ///     uint64 (NumPy's object array, a dtype NumSharp does not have) — raised where NumPy starts computing with
        ///     it, after every series check. A str series is NumPy's ValueError (no common type).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.chebyshev.chebmulx.html</remarks>
        [NDScoped]
        public NDArray chebmulx(object c) => NDPolyAlgebra.Mulx(PolyBasis.Chebyshev, c);

        /// <summary>
        ///     Multiplies two Chebyshev series: the z-series product: both series doubled into Laurent z-series (<c>zs[n-1:] = c/2; zs + zs[::-1]</c>), convolved (<c>np.convolve</c>) and folded back (<c>c[1:n] *= 2</c>), trailing zeros trimmed. The product is np.convolve's arithmetic on the z-series — NumPy's sequential sum for real series whose shorter factor has at most ~8 terms (a z-series doubles the length), cblas <c>?dot</c> beyond that and for every complex product; with <c>NumSharp.Interop.OpenBLAS</c> installed those run through NumPy's own BLAS and are byte-identical, without it they agree to a few ULP.
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
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.chebyshev.chebmul.html</remarks>
        [NDScoped]
        public NDArray chebmul(object c1, object c2) => NDPolyAlgebra.Mul(PolyBasis.Chebyshev, c1, c2);

        /// <summary>
        ///     Divides the Chebyshev series <paramref name="c1"/> by <paramref name="c2"/>, returning quotient and remainder:
        ///     the z-series division (<c>chebyshev._zseries_div</c>): both series as z-series, a symmetric synthetic division from both ends, the quotient and remainder folded back and trimmed. A dividend shorter than the divisor gives <c>(c1[:1]*0, c1)</c>, a one-term divisor
        ///     <c>(c1/c2[-1], c1[:1]*0)</c>. Bit-identical to NumPy 2.4.2's <c>chebdiv</c>.
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
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.chebyshev.chebdiv.html</remarks>
        [NDScoped]
        public (NDArray quo, NDArray rem) chebdiv(object c1, object c2) => NDPolyAlgebra.Div(PolyBasis.Chebyshev, c1, c2);

        /// <summary>
        ///     Raises the Chebyshev series <paramref name="c"/> to the power <paramref name="pow"/>: <c>[1]</c> of the series' dtype
        ///     for 0, the (trimmed) series for 1, otherwise the z-series convolved pow - 1 times and folded back (no final trimming). Bit-identical to NumPy 2.4.2's <c>chebpow</c> (the product's parity notes on <see cref="chebmul"/> apply).
        /// </summary>
        /// <param name="c">The coefficients, LOW degree first: an <see cref="NDArray"/> (any layout), a typed C# array, a Python list or tuple (<c>object[]</c>, a <c>ValueTuple</c>, a jagged array — np.array's coercion), or a single number. Integer series are computed in float64; a bool series has no common type.</param>
        /// <param name="pow">The power (≥ 0).</param>
        /// <param name="maxpower">The largest power allowed: 16; null for no limit. NumPy's default keeps an accidental huge power from
        ///     running away.</param>
        /// <returns>The power series, a new array of <c>np.common_type(c)</c>.</returns>
        /// <exception cref="ValueError">The series' errors (see <see cref="chebmul"/>), then
        ///     <c>Power must be a non-negative integer.</c>, then <c>Power is too large</c> — NumPy's order.</exception>
        /// <exception cref="System.NotSupportedException">An object series (see <see cref="chebmul"/>), raised after the power checks.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.chebyshev.chebpow.html</remarks>
        [NDScoped]
        public NDArray chebpow(object c, int pow, int? maxpower = 16) => NDPolyAlgebra.Pow(PolyBasis.Chebyshev, c, pow, maxpower);

        /// <summary>
        ///     <see cref="chebpow(object, int, int?)"/> with a Python-float power: NumPy's <c>power = int(pow)</c>
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
        /// <exception cref="System.NotSupportedException">An object series (see <see cref="chebmul"/>), raised after the power checks.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.chebyshev.chebpow.html</remarks>
        [NDScoped]
        public NDArray chebpow(object c, double pow, int? maxpower = 16) => NDPolyAlgebra.Pow(PolyBasis.Chebyshev, c, pow, maxpower);

        /// <summary>
        ///     <see cref="chebpow(object, int, int?)"/> for ANY power argument, with NumPy's default limit 16: <c>power =
        ///     int(pow)</c> with Python's semantics — a bool is 0 / 1, np.float16 (Half) or decimal truncates like a float, a
        ///     str parses with int()'s grammar (and then fails <c>power != pow</c>), a 0-d NDArray converts its element — and
        ///     NumPy's error for every value int() refuses.
        /// </summary>
        /// <param name="c">The coefficients, LOW degree first: an <see cref="NDArray"/> (any layout), a typed C# array, a Python list or tuple (<c>object[]</c>, a <c>ValueTuple</c>, a jagged array — np.array's coercion), or a single number. Integer series are computed in float64; a bool series has no common type.</param>
        /// <param name="pow">The power: any value (see the summary).</param>
        /// <returns>The power series.</returns>
        /// <exception cref="ValueError">The series' errors (see <see cref="chebmul"/>); <c>cannot convert float NaN to
        ///     integer</c>; <c>invalid literal for int() with base 10: '…'</c>; <c>Power must be a non-negative
        ///     integer.</c>; <c>Power is too large</c>.</exception>
        /// <exception cref="TypeError">A power int() refuses: a complex, null (None), a list or tuple (<c>int() argument must
        ///     be a string, a bytes-like object or a real number, not '…'</c>), an ndarray of one or more dims (<c>only
        ///     0-dimensional arrays can be converted to Python scalars</c>).</exception>
        /// <exception cref="System.OverflowException"><c>cannot convert float infinity to integer</c>; a power beyond int64
        ///     with no limit (NumPy would run until memory is exhausted).</exception>
        /// <exception cref="System.NotSupportedException">An object series (see <see cref="chebmul"/>), raised after the
        ///     power checks.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.chebyshev.chebpow.html</remarks>
        [NDScoped]
        public NDArray chebpow(object c, object pow) => NDPolyAlgebra.Pow(PolyBasis.Chebyshev, c, pow, NDPolyAlgebra.DefaultMaxPower);

        /// <summary>
        ///     <see cref="chebpow(object, object)"/> with ANY limit: null is None (no limit); anything else is compared as
        ///     NumPy compares <c>power &gt; maxpower</c> — exactly for a Python int / bool / float, a NumPy integer scalar
        ///     (char) or decimal; in float16 for np.float16 (Half); for an ndarray (an NDArray, a typed C# array) in its dtype
        ///     (float16 / float32 / float64 rounding the power first, complex lexicographically, bool as int64, integers
        ///     exactly), the comparison's truth value then needing exactly one element.
        /// </summary>
        /// <param name="c">The coefficients, LOW degree first: an <see cref="NDArray"/> (any layout), a typed C# array, a Python list or tuple (<c>object[]</c>, a <c>ValueTuple</c>, a jagged array — np.array's coercion), or a single number. Integer series are computed in float64; a bool series has no common type.</param>
        /// <param name="pow">The power: any value (see <see cref="chebpow(object, object)"/>).</param>
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
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.chebyshev.chebpow.html</remarks>
        [NDScoped]
        public NDArray chebpow(object c, object pow, object maxpower) => NDPolyAlgebra.Pow(PolyBasis.Chebyshev, c, pow, maxpower);

        /// <summary>
        ///     The Chebyshev series whose roots are <paramref name="roots"/>: the roots sorted, one linear factor per root
        ///     (<c>chebline(-r, 1) = [-r, 1]</c>), multiplied pairwise as a balanced tree (<c>polyutils._fromroots</c>) with <c>chebmul</c>.
        ///     Empty roots give <c>[1.]</c>. The factors are built by <c>np.array([-r, …])</c>, whose coercion promotes a
        ///     float16 / float32 root to float64 — so the series is float64 (complex128 for complex roots). Bit-identical to
        ///     NumPy 2.4.2's <c>chebfromroots</c> (the product's parity notes on <see cref="chebmul"/> apply); one documented difference: roots that compare equal but differ in bits (+0.0 and -0.0) are ordered
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
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.chebyshev.chebfromroots.html</remarks>
        [NDScoped]
        public NDArray chebfromroots(object roots) => NDPolyAlgebra.FromRoots(PolyBasis.Chebyshev, roots);

        /// <summary>
        ///     Converts the Chebyshev series <paramref name="c"/> to a power series — NumPy's backward recurrence over power-series
        ///     ops (<c>polyadd</c>, <c>polysub</c>, <c>polymulx</c>) from the highest coefficient down; a series of fewer than three terms is returned as is. Keeps the
        ///     series' common type. Bit-identical to NumPy 2.4.2's <c>cheb2poly</c>.
        /// </summary>
        /// <param name="c">The Chebyshev series's coefficients. The coefficients, LOW degree first: an <see cref="NDArray"/> (any layout), a typed C# array, a Python list or tuple (<c>object[]</c>, a <c>ValueTuple</c>, a jagged array — np.array's coercion), or a single number. Integer series are computed in float64; a bool series has no common type.</param>
        /// <returns>The power-series coefficients, low degree first (a new array, or a view of one when trailing zeros
        ///     were trimmed).</returns>
        /// <exception cref="ValueError">An empty series (<c>Coefficient array is empty</c>), a series that is not 1-d
        ///     (<c>Coefficient array is not 1-d</c>), or no common type (a bool or str series), in NumPy's order; a ragged list
        ///     (np.array's inhomogeneous-shape text).</exception>
        /// <exception cref="System.NotSupportedException">A null series, or one holding None, a non-numeric object or a Python int past
        ///     uint64 (NumPy's object array, a dtype NumSharp does not have) — raised where NumPy starts computing with
        ///     it, after every series check. A str series is NumPy's ValueError (no common type).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.chebyshev.cheb2poly.html</remarks>
        [NDScoped]
        public NDArray cheb2poly(object c) => NDPolyAlgebra.ToPower(PolyBasis.Chebyshev, c);

        /// <summary>
        ///     Converts the power series <paramref name="pol"/> to a Chebyshev series — Horner's scheme over the Chebyshev series's own ops:
        ///     <c>res = chebadd(chebmulx(res), pol[i])</c> from the highest coefficient down, starting from the Python int 0,
        ///     which makes the result float64 for any real series (complex128 for a complex one). Bit-identical to NumPy
        ///     2.4.2's <c>poly2cheb</c>.
        /// </summary>
        /// <param name="pol">The power-series coefficients. The coefficients, LOW degree first: an <see cref="NDArray"/> (any layout), a typed C# array, a Python list or tuple (<c>object[]</c>, a <c>ValueTuple</c>, a jagged array — np.array's coercion), or a single number. Integer series are computed in float64; a bool series has no common type.</param>
        /// <returns>The Chebyshev series's coefficients, low degree first (a new array, or a view of one when trailing zeros
        ///     were trimmed).</returns>
        /// <exception cref="ValueError">An empty series (<c>Coefficient array is empty</c>), a series that is not 1-d
        ///     (<c>Coefficient array is not 1-d</c>), or no common type (a bool or str series), in NumPy's order; a ragged list
        ///     (np.array's inhomogeneous-shape text).</exception>
        /// <exception cref="System.NotSupportedException">A null series, or one holding None, a non-numeric object or a Python int past
        ///     uint64 (NumPy's object array, a dtype NumSharp does not have) — raised where NumPy starts computing with
        ///     it, after every series check. A str series is NumPy's ValueError (no common type).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.chebyshev.poly2cheb.html</remarks>
        [NDScoped]
        public NDArray poly2cheb(object pol) => NDPolyAlgebra.FromPower(PolyBasis.Chebyshev, pol);

        // ---------------------------------------------------------------------------------------------
        //  Vandermonde matrices (plan U5): chebvander / chebvander2d / chebvander3d
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     The pseudo-Vandermonde matrix of degree <paramref name="deg"/>: <c>V[..., i] = T_i(x)</c>, by NumPy's forward
        ///     recurrence <c>v[0] = x*0 + 1; x2 = 2*x; v[1] = x; v[i] = v[i-1]*x2 - v[i-2]</c> — so
        ///     <c>chebvander(x, n) @ c == chebval(x, c)</c> up to rounding. Bit-identical to NumPy 2.4.2's <c>chebvander</c>
        ///     (one IL kernel pass; <c>x2</c> is computed once per block of points, as NumPy computes it once).
        /// </summary>
        /// <param name="x">The points (see <see cref="PowerSeriesModule.polyvander(object, int)"/>: any layout; the matrix dtype is
        ///     <c>(x + 0.0).dtype</c>).</param>
        /// <param name="deg">The degree (≥ 0).</param>
        /// <returns>A view of shape <c>x.shape + (deg + 1,)</c> over a new array, the degree axis last (OWNDATA false).</returns>
        /// <exception cref="ValueError"><c>deg must be non-negative</c>; a ragged list x; <c>array is too big; …</c>.</exception>
        /// <exception cref="System.OutOfMemoryException">The matrix cannot be allocated (NumPy's MemoryError).</exception>
        /// <exception cref="System.NotSupportedException">A null or str x, or a list NumPy makes a str / object array of.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.chebyshev.chebvander.html</remarks>
        [NDScoped]
        public NDArray chebvander(object x, int deg) => NDPolyVander.Vander(PolyBasis.Chebyshev, x, deg);

        /// <summary>
        ///     <see cref="chebvander(object, int)"/> with a degree of ANY kind, read as NumPy's <c>polyutils._as_int</c> (Python's
        ///     <c>operator.index</c>) — see <see cref="PowerSeriesModule.polyvander(object, object)"/> for what passes and how a
        ///     refused value reads in the TypeError.
        /// </summary>
        /// <param name="x">The points.</param>
        /// <param name="deg">The degree, any value.</param>
        /// <returns>The matrix.</returns>
        /// <exception cref="TypeError"><c>deg must be an integer, received …</c> — before x is looked at.</exception>
        /// <exception cref="ValueError"><c>deg must be non-negative</c>; a ragged list x; <c>Maximum allowed dimension
        ///     exceeded</c> or <c>array is too big; …</c>.</exception>
        /// <exception cref="System.OutOfMemoryException">The matrix cannot be allocated.</exception>
        /// <exception cref="System.NotSupportedException">A null or str x, or a list NumPy makes a str / object array of.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.chebyshev.chebvander.html</remarks>
        [NDScoped]
        public NDArray chebvander(object x, object deg) => NDPolyVander.Vander(PolyBasis.Chebyshev, x, deg);

        /// <summary>
        ///     The 2-D pseudo-Vandermonde matrix: column <c>a*(deg[1] + 1) + b</c> is <c>T_a(x) * T_b(y)</c> — NumPy's
        ///     <c>_vander_nd_flat</c> outer product of the two 1-D matrices. Bit-identical to NumPy 2.4.2's <c>chebvander2d</c>.
        /// </summary>
        /// <param name="x">The first coordinates (stacked with y: one shape, one promoted dtype — see
        ///     <see cref="PowerSeriesModule.polyvander2d"/>).</param>
        /// <param name="y">The second coordinates.</param>
        /// <param name="deg">The two degrees <c>[x_deg, y_deg]</c>.</param>
        /// <returns>A view of shape <c>x.shape + ((deg[0] + 1) * (deg[1] + 1),)</c> over a new array.</returns>
        /// <exception cref="TypeError">A deg without a length, or a non-integer degree.</exception>
        /// <exception cref="ValueError"><c>Expected 2 dimensions of degrees, got {len}</c>; points of different shapes;
        ///     <c>deg must be non-negative</c>; <c>array is too big; …</c>.</exception>
        /// <exception cref="IncorrectShapeException">No points (NumPy's reshape text).</exception>
        /// <exception cref="System.OutOfMemoryException">A matrix NumPy allocates on the way cannot be allocated.</exception>
        /// <exception cref="System.OverflowException">Scalar points with a Python int past the float range (NumPy's
        ///     <c>int too large to convert to float</c> at its per-element <c>+ 0.0</c>).</exception>
        /// <exception cref="System.NotSupportedException">Points NumPy stacks into a str array, or into an object array it
        ///     computes with as Python objects or fails on. Scalar points that are all numbers but stack into an object array
        ///     (a Python int past uint64 among them) ARE computed: each dimension gets its own number, as in NumPy.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.chebyshev.chebvander2d.html</remarks>
        [NDScoped]
        public NDArray chebvander2d(object x, object y, object deg) => NDPolyVander.VanderNd(PolyBasis.Chebyshev, new[] { x, y }, deg);

        /// <summary>
        ///     The 3-D pseudo-Vandermonde matrix: column <c>(a*(deg[1] + 1) + b)*(deg[2] + 1) + c</c> is
        ///     <c>T_a(x) * T_b(y) * T_c(z)</c>, the first product rounded before the second (NumPy's reduce). Bit-identical to
        ///     NumPy 2.4.2's <c>chebvander3d</c>.
        /// </summary>
        /// <param name="x">The first coordinates (x, y, z stacked: one shape).</param>
        /// <param name="y">The second coordinates.</param>
        /// <param name="z">The third coordinates.</param>
        /// <param name="deg">The three degrees.</param>
        /// <returns>A view of shape <c>x.shape + ((deg[0] + 1) * (deg[1] + 1) * (deg[2] + 1),)</c> over a new array.</returns>
        /// <exception cref="TypeError">A deg without a length, or a non-integer degree.</exception>
        /// <exception cref="ValueError"><c>Expected 3 dimensions of degrees, got {len}</c>, and the 2-D form's other errors.</exception>
        /// <exception cref="IncorrectShapeException">No points (NumPy's reshape text).</exception>
        /// <exception cref="System.OutOfMemoryException">A matrix NumPy allocates on the way cannot be allocated.</exception>
        /// <exception cref="System.OverflowException">Scalar points with a Python int past the float range (NumPy's
        ///     <c>int too large to convert to float</c> at its per-element <c>+ 0.0</c>).</exception>
        /// <exception cref="System.NotSupportedException">Points NumPy stacks into a str array, or into an object array it
        ///     computes with as Python objects or fails on. Scalar points that are all numbers but stack into an object array
        ///     (a Python int past uint64 among them) ARE computed: each dimension gets its own number, as in NumPy.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.chebyshev.chebvander3d.html</remarks>
        [NDScoped]
        public NDArray chebvander3d(object x, object y, object z, object deg) => NDPolyVander.VanderNd(PolyBasis.Chebyshev, new[] { x, y, z }, deg);

        // ---------------------------------------------------------------------------------------------
        //  Companion matrix and roots (plan U7): chebcompanion / chebroots
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     The scaled companion matrix of the Chebyshev series <paramref name="c"/>: the basis is scaled (T_0 by 1, the others by sqrt(1/2)) so that the matrix is symmetric when c is a single Chebyshev polynomial, which gives better eigenvalue estimates — off-diagonals <c>sqrt(.5)</c> then <c>1/2</c>, and the last column reduced by <c>(c[:-1] / c[-1]) * (scl / scl[-1]) * .5</c>. NumPy's scale vector is float64, so a float16 / float32 series' last column is computed in float64 and rounded once into the matrix (the in-place op's output cast) and its off-diagonals are float64 values cast by assignment; a complex series runs the complex128 loops. Bit-identical to NumPy 2.4.2's <c>chebcompanion</c>.
        /// </summary>
        /// <param name="c">The coefficients, LOW degree first: an <see cref="NDArray"/> (any layout), a typed C# array, a Python list or tuple (<c>object[]</c>, a <c>ValueTuple</c>, a jagged array — np.array's coercion), or a single number. Trailing zeros are trimmed first; integer series are computed in float64; a bool series has no common type.</param>
        /// <returns>A new owning (deg, deg) matrix of the series' common type; a two-term series gives the 1x1 matrix
        ///     <c>[[-c[0] / c[1]]]</c> (NumPy's scalarmath on the two coefficients).</returns>
        /// <exception cref="ValueError">An empty series (<c>Coefficient array is empty</c>), a series that is not 1-d (<c>Coefficient array is not 1-d</c>), or no common type (a bool or str series), in NumPy's order; a ragged list (np.array's inhomogeneous-shape text). Fewer than two terms after trimming:
        ///     <c>Series must have maximum degree of at least 1.</c></exception>
        /// <exception cref="System.NotSupportedException">An object series of two or more terms (None, a non-numeric object or a
        ///     Python int past uint64 — NumPy's object array, a dtype NumSharp does not have), raised where NumPy starts
        ///     computing with Python objects: after the length check. A str series is NumPy's ValueError (no common type).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.chebyshev.chebcompanion.html</remarks>
        [NDScoped]
        public NDArray chebcompanion(object c) => NDPolyAlgebra.Companion(PolyBasis.Chebyshev, c);

        /// <summary>
        ///     The roots of the Chebyshev series <paramref name="c"/>: the eigenvalues of <c>chebcompanion(c)[::-1, ::-1]</c> — the ROTATED companion matrix, which NumPy notes reduces the error, computed by
        ///     <see cref="np.linalg.eigvals"/> and sorted in place, ascending (complex numbers lexicographically, real part
        ///     first). A constant series has no roots — an empty array of its dtype — and a linear one the single root
        ///     <c>-c[0] / c[1]</c> (scalarmath, in its own dtype). Roots far from the origin, and multiple roots, carry larger
        ///     errors (NumPy's documented caveat). With NumSharp.Interop.OpenBLAS referenced the eigenvalues come from NumPy's
        ///     own LAPACK <c>geev</c> and are bit-identical to NumPy 2.4.2's <c>chebroots</c> (one BLAS thread); where
        ///     +0.0 and -0.0 roots meet, NumPy's SIMD sort orders them by CPU and NumSharp's sort deterministically.
        /// </summary>
        /// <param name="c">The coefficients, LOW degree first: an <see cref="NDArray"/> (any layout), a typed C# array, a Python list or tuple (<c>object[]</c>, a <c>ValueTuple</c>, a jagged array — np.array's coercion), or a single number. Trailing zeros are trimmed first; integer series are computed in float64; a bool series has no common type.</param>
        /// <returns>The sorted roots: REAL when every eigenvalue is — a view of the eigenvalue array's real parts for a float64
        ///     series (OWNDATA false, as NumPy's <c>w.real</c>), a new float32 array for a float32 series — otherwise
        ///     complex128 (NumPy's complex64 for a float32 series: NumSharp has one complex width).</returns>
        /// <exception cref="ValueError">An empty series (<c>Coefficient array is empty</c>), a series that is not 1-d (<c>Coefficient array is not 1-d</c>), or no common type (a bool or str series), in NumPy's order; a ragged list (np.array's inhomogeneous-shape text).</exception>
        /// <exception cref="TypeError">A float16 (or decimal) series of degree 2 or more: linalg's
        ///     <c>array type float16 is unsupported in linalg</c>.</exception>
        /// <exception cref="LinAlgError">A companion matrix holding an infinity or NaN (<c>Array must not contain infs or
        ///     NaNs</c>, checked before the dtype), or eigenvalues that do not converge.</exception>
        /// <exception cref="MissingBackendException">Degree 2 or more with no LAPACK backend: reference
        ///     NumSharp.Interop.OpenBLAS (NumSharp.Core ships no eigensolver).</exception>
        /// <exception cref="System.NotSupportedException">An object series (None, a non-numeric object or a Python int past
        ///     uint64): NumPy returns an object array or computes with Python objects.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.chebyshev.chebroots.html</remarks>
        [NDScoped]
        public NDArray chebroots(object c) => NDPolyAlgebra.Roots(PolyBasis.Chebyshev, c);
    }
}
