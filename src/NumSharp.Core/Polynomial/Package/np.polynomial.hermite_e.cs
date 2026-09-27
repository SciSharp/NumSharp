using NumSharp.Backends.Kernels;

namespace NumSharp
{
    /// <summary>
    ///     <c>numpy.polynomial.hermite_e</c> — probabilists' Hermite series <c>p(x) = c[0]*He_0(x) + c[1]*He_1(x) + …</c>, reachable as
    ///     <c>np.polynomial.hermite_e</c>. Coefficients are LOW degree first.
    /// </summary>
    /// <remarks>https://numpy.org/doc/stable/reference/routines.polynomials.hermite_e.html</remarks>
    [ModuleName("np.polynomial.hermite_e")]
    public class HermiteEModule
    {
        /// <summary>Constructs the module; there is one shared instance behind <c>np.polynomial.hermite_e</c>.</summary>
        internal HermiteEModule() { }

        /// <summary>
        ///     Evaluates the probabilists' Hermite series with coefficients <paramref name="c"/> at <paramref name="x"/> by Clenshaw's recurrence exactly as NumPy writes it (<c>c0 = c[-i] - c1*(nd - 1); c1 = tmp + c1*x; return c0 + c1*x</c>),
        ///     bit-identical to NumPy 2.4.2: the first steps run in the COEFFICIENTS' dtype exactly as NumPy's
        ///     do, and every intermediate carries NumPy's NEP 50 dtype.
        /// </summary>
        /// <param name="x">
        ///     The points. An <see cref="NDArray"/> (any layout) or a C# array is an array; a C# <c>bool</c>, integer,
        ///     <c>float</c>, <c>double</c> or <see cref="System.Numerics.Complex"/> is a <b>Python scalar</b> — it adopts
        ///     the coefficients' dtype (<c>hermeval(2.0, float32_c)</c> is float32), Python-level arithmetic on it
        ///     (<c>2*x</c>, <c>1 - x</c>) is done in double as CPython does, and a 1-D series then yields a 0-d result.
        ///     <c>char</c>/<see cref="System.Half"/>/<c>decimal</c> are strong 0-d arrays.
        /// </param>
        /// <param name="c">Coefficients, low degree first; for an N-D array the series run along axis 0 and the other
        ///     axes index independent series. Integer and bool coefficients are evaluated as float64.</param>
        /// <param name="tensor">When true (the default) and x is an array, every series is evaluated at every point
        ///     and the result has shape <c>c.shape[1:] + x.shape</c>; when false x is broadcast over the series
        ///     (<c>c.shape[1:]</c> against <c>x.shape</c>). Irrelevant for a 1-D series or a scalar x.</param>
        /// <returns>The values, of NumPy's result dtype.</returns>
        /// <exception cref="System.ArgumentNullException"><paramref name="x"/> or <paramref name="c"/> is null.</exception>
        /// <exception cref="IndexError"><paramref name="c"/> is empty: <c>index -2 is out of bounds for axis 0 with size 0</c>.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="tensor"/> is false and the shapes do not broadcast
        ///     (NumPy's ValueError text).</exception>
        /// <exception cref="System.OverflowException">A Python int of the recurrence does not fit an integer x dtype
        ///     (NumPy's OverflowError).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.hermite_e.hermeval.html</remarks>
        public NDArray hermeval(object x, NDArray c, bool tensor = true) => NDPolyEval.Val(PolyBasis.HermiteE, x, c, tensor);

        /// <summary>
        ///     Evaluates the 2-D probabilists' Hermite series <c>Σ c[i,j] * He_i(x) * He_j(y)</c> at the point pairs <c>(x, y)</c>.
        ///     NumPy's <c>_valnd</c>: <c>hermeval(x, c)</c> then <c>hermeval(y, ·, tensor=False)</c> — its exact
        ///     intermediate, so the bits match.
        /// </summary>
        /// <param name="x">First coordinates. Scalars become 0-d arrays (NumPy's <c>np.asanyarray</c>).</param>
        /// <param name="y">Second coordinates; must have <paramref name="x"/>'s shape.</param>
        /// <param name="c">Coefficients; <c>c[i,j]</c> is the coefficient of the degree (i, j) term. Extra trailing
        ///     axes index independent series (they lead the result's shape).</param>
        /// <returns>The values, shape <c>c.shape[2:] + x.shape</c>.</returns>
        /// <exception cref="System.ArgumentNullException">An argument is null.</exception>
        /// <exception cref="ValueError">x and y differ in shape: <c>x, y are incompatible</c>.</exception>
        /// <exception cref="IndexError"><paramref name="c"/> is empty along axis 0.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.hermite_e.hermeval2d.html</remarks>
        public NDArray hermeval2d(NDArray x, NDArray y, NDArray c) => NDPolyEval.ValNd(PolyBasis.HermiteE, c, new[] { x, y });

        /// <summary>
        ///     Evaluates the 3-D probabilists' Hermite series at the point triples <c>(x, y, z)</c> (NumPy's <c>_valnd</c>: one tensor
        ///     pass, then two per-point passes).
        /// </summary>
        /// <param name="x">First coordinates.</param>
        /// <param name="y">Second coordinates (x's shape).</param>
        /// <param name="z">Third coordinates (x's shape).</param>
        /// <param name="c">Coefficients; <c>c[i,j,k]</c> is the coefficient of the degree (i, j, k) term.</param>
        /// <returns>The values, shape <c>c.shape[3:] + x.shape</c>.</returns>
        /// <exception cref="System.ArgumentNullException">An argument is null.</exception>
        /// <exception cref="ValueError">The coordinates differ in shape: <c>x, y, z are incompatible</c>.</exception>
        /// <exception cref="IndexError"><paramref name="c"/> is empty along axis 0.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.hermite_e.hermeval3d.html</remarks>
        public NDArray hermeval3d(NDArray x, NDArray y, NDArray z, NDArray c) => NDPolyEval.ValNd(PolyBasis.HermiteE, c, new[] { x, y, z });

        /// <summary>
        ///     Evaluates the 2-D probabilists' Hermite series on the Cartesian product of <paramref name="x"/> and <paramref name="y"/>
        ///     (NumPy's <c>_gridnd</c>: two tensor passes). Unlike <see cref="hermeval2d"/> the coordinates are NOT
        ///     converted first, so a Python-scalar coordinate stays weak.
        /// </summary>
        /// <param name="x">First-axis points (array or Python scalar, as in <see cref="hermeval"/>).</param>
        /// <param name="y">Second-axis points.</param>
        /// <param name="c">Coefficients; <c>c[i,j]</c> is the coefficient of the degree (i, j) term.</param>
        /// <returns>The grid, shape <c>c.shape[2:] + x.shape + y.shape</c>.</returns>
        /// <exception cref="System.ArgumentNullException">An argument is null.</exception>
        /// <exception cref="IndexError"><paramref name="c"/> is empty along an evaluated axis.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.hermite_e.hermegrid2d.html</remarks>
        public NDArray hermegrid2d(object x, object y, NDArray c) => NDPolyEval.GridNd(PolyBasis.HermiteE, c, new[] { x, y });

        /// <summary>
        ///     Evaluates the 3-D probabilists' Hermite series on the Cartesian product of <paramref name="x"/>, <paramref name="y"/> and
        ///     <paramref name="z"/> (NumPy's <c>_gridnd</c>: three tensor passes).
        /// </summary>
        /// <param name="x">First-axis points.</param>
        /// <param name="y">Second-axis points.</param>
        /// <param name="z">Third-axis points.</param>
        /// <param name="c">Coefficients; <c>c[i,j,k]</c> is the coefficient of the degree (i, j, k) term.</param>
        /// <returns>The grid, shape <c>c.shape[3:] + x.shape + y.shape + z.shape</c>.</returns>
        /// <exception cref="System.ArgumentNullException">An argument is null.</exception>
        /// <exception cref="IndexError"><paramref name="c"/> is empty along an evaluated axis.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.hermite_e.hermegrid3d.html</remarks>
        public NDArray hermegrid3d(object x, object y, object z, NDArray c) => NDPolyEval.GridNd(PolyBasis.HermiteE, c, new[] { x, y, z });

        /// <summary>
        ///     Evaluates the N-D probabilists' Hermite series at the points <c>pts = (x, y, …)</c> — the n-dimensional
        ///     <see cref="hermeval2d"/>. Added in NumPy's development branch (<c>pu._valnd(hermeval, c, *pts)</c>); not in
        ///     NumPy 2.4.2, where the same computation is the private <c>polyutils._valnd</c>.
        /// </summary>
        /// <param name="pts">The coordinate arrays, all of one shape (one per series axis evaluated).</param>
        /// <param name="c">Coefficients; axis i is <c>pts[i]</c>'s degree.</param>
        /// <returns>The values, shape <c>c.shape[len(pts):] + pts[0].shape</c>.</returns>
        /// <exception cref="System.ArgumentNullException">An argument is null.</exception>
        /// <exception cref="IndexError"><paramref name="pts"/> is empty (<c>list index out of range</c>).</exception>
        /// <exception cref="ValueError">The coordinates differ in shape.</exception>
        /// <remarks>https://numpy.org/devdocs/reference/generated/numpy.polynomial.hermite_e.hermevalnd.html</remarks>
        public NDArray hermevalnd(NDArray[] pts, NDArray c) => NDPolyEval.ValNd(PolyBasis.HermiteE, c, pts);

        // ---- U1: constants, line, add/sub, trim ----

        private static NDArray s_hermedomain, s_hermezero, s_hermeone, s_hermex;

        /// <summary>
        ///     <c>hermedomain</c> — the default domain of the probabilists' Hermite series class, float64 <c>[-1.0, 1.0]</c>.
        /// </summary>
        /// <remarks>
        ///     NumPy's module constants are ordinary WRITEABLE ndarrays, the same object on every access, and so is
        ///     this one: a process-wide instance (a write through it persists, as in NumPy). It is shared — do not
        ///     dispose it (a dispose is harmless: the buffer is pinned for the process lifetime) and copy it before
        ///     mutating unless the change is meant to be global.
        ///     <para>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.hermite_e.html</para>
        /// </remarks>
        public NDArray hermedomain => NDPolySeries.Constant(ref s_hermedomain, static () => np.array(new double[] { -1.0, 1.0 }));

        /// <summary><c>hermezero</c> — the probabilists' Hermite series representing 0, int64 <c>[0]</c> (shared and writeable, see <see cref="hermedomain"/>).</summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.hermite_e.html</remarks>
        public NDArray hermezero => NDPolySeries.Constant(ref s_hermezero, static () => np.array(new long[] { 0 }));

        /// <summary><c>hermeone</c> — the probabilists' Hermite series representing 1, int64 <c>[1]</c> (shared and writeable, see <see cref="hermedomain"/>).</summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.hermite_e.html</remarks>
        public NDArray hermeone => NDPolySeries.Constant(ref s_hermeone, static () => np.array(new long[] { 1 }));

        /// <summary><c>hermex</c> — the probabilists' Hermite series representing <c>x</c>, int64 <c>[0, 1]</c> (shared and writeable, see <see cref="hermedomain"/>).</summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.hermite_e.html</remarks>
        public NDArray hermex => NDPolySeries.Constant(ref s_hermex, static () => np.array(new long[] { 0, 1 }));

        /// <summary>
        ///     The probabilists' Hermite series of the line <c>off + scl*x</c>: <c>[off, scl]</c> when <c>scl != 0</c>, else <c>[off]</c> — built
        ///     with <c>np.array</c>, so the dtype is the one NumPy DISCOVERS for the entries (<c>hermeline(1, 2)</c> is int64,
        ///     <c>hermeline(1.0, 2)</c> float64, a bool pair stays bool) rather than a coefficient dtype.
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
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.hermite_e.hermeline.html</remarks>
        public NDArray hermeline(object off, object scl) => NDPolySeries.Line(PolyBasis.HermiteE, off, scl);

        /// <summary>
        ///     Adds two probabilists' Hermite seriess (<c>polyutils._add</c>): both converted to their common type (integers become float64),
        ///     the shorter added into a copy of the longer, trailing zeros trimmed.
        /// </summary>
        /// <param name="c1">First series, low degree first (an array, a C# array or list, or a single number).</param>
        /// <param name="c2">Second series.</param>
        /// <returns>The sum: a new array, or a VIEW of one when trailing zeros were trimmed (as in NumPy).</returns>
        /// <exception cref="ValueError">An empty or non-1-d series, or no common type (a bool series).</exception>
        /// <exception cref="System.NotSupportedException">A null series (NumPy would build an object array).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.hermite_e.hermeadd.html</remarks>
        [NDScoped]
        public NDArray hermeadd(object c1, object c2) => NDPolySeries.AddSub(c1, c2, subtract: false);

        /// <summary>
        ///     Subtracts the probabilists' Hermite series <paramref name="c2"/> from <paramref name="c1"/> (<c>polyutils._sub</c>): both converted
        ///     to their common type; when <paramref name="c2"/> is at least as long it is NEGATED first and
        ///     <paramref name="c1"/> added into it (NumPy's operation order, visible in which NaN survives), then trailing
        ///     zeros are trimmed.
        /// </summary>
        /// <param name="c1">Minuend series, low degree first.</param>
        /// <param name="c2">Subtrahend series.</param>
        /// <returns>The difference: a new array, or a VIEW of one when trailing zeros were trimmed.</returns>
        /// <exception cref="ValueError">An empty or non-1-d series, or no common type (a bool series).</exception>
        /// <exception cref="System.NotSupportedException">A null series (NumPy would build an object array).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.hermite_e.hermesub.html</remarks>
        [NDScoped]
        public NDArray hermesub(object c1, object c2) => NDPolySeries.AddSub(c1, c2, subtract: true);

        /// <summary>
        ///     <c>hermetrim</c> — NumPy's alias of <see cref="PolyUtilsModule.trimcoef(object)"/>: removes the trailing
        ///     zero coefficients (tolerance 0) and returns a fresh copy of the coefficient dtype.
        /// </summary>
        /// <param name="c">The series (1-D).</param>
        /// <returns>The trimmed copy (an all-zero series becomes <c>c[:1]*0</c>).</returns>
        /// <exception cref="ValueError">An empty or non-1-d series, or no common type.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.hermite_e.hermetrim.html</remarks>
        [NDScoped]
        public NDArray hermetrim(object c) => NDPolySeries.TrimCoef(c, PolyNumber.FromPython(PyScalar.Int(0)));

        /// <summary>
        ///     <c>hermetrim</c> — NumPy's alias of <see cref="PolyUtilsModule.trimcoef(object, object)"/>: removes the
        ///     trailing coefficients whose magnitude is at most <paramref name="tol"/> and returns a fresh copy.
        /// </summary>
        /// <param name="c">The series (1-D).</param>
        /// <param name="tol">The non-negative tolerance (a Python number, or a NumPy scalar compared in the promoted dtype).</param>
        /// <returns>The trimmed copy (<c>c[:1]*0</c> when nothing exceeds the tolerance).</returns>
        /// <exception cref="ValueError"><c>tol must be non-negative</c>, an empty or non-1-d series, or no common type.</exception>
        /// <exception cref="TypeError">A complex or null tolerance.</exception>
        /// <exception cref="OverflowException">A Python int tolerance too large to convert.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.hermite_e.hermetrim.html</remarks>
        [NDScoped]
        public NDArray hermetrim(object c, object tol) => NDPolySeries.TrimCoef(c, PolyUtilsModule.Tolerance(tol));
    }
}
