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
    }
}
