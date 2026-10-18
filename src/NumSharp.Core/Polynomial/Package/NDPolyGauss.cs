using System;
using System.Numerics;
using NumSharp.Backends.Kernels;

// =============================================================================
// NDPolyGauss.cs — leggauss, laggauss, hermgauss, hermegauss: Gauss-Legendre, Gauss-Laguerre and Gauss-Hermite
// quadrature
// =============================================================================
//
// NumPy 2.4.2 (numpy/polynomial/legendre.py, laguerre.py), statement for statement:
//
//   ideg = pu._as_int(deg, "deg")
//   if ideg <= 0:
//       raise ValueError("deg must be a positive integer")
//   c = np.array([0] * deg + [1])
//   m = legcompanion(c)                        # lagcompanion
//   x = np.linalg.eigvalsh(m)
//   dy = legval(x, c)
//   df = legval(x, legder(c))
//   x -= dy / df                               # one Newton step on the roots
//   fm = legval(x, c[1:])
//   fm /= np.abs(fm).max()
//   df /= np.abs(df).max()
//   w = 1 / (fm * df)
//   leggauss:  w = (w + w[::-1]) / 2;  x = (x - x[::-1]) / 2;  w *= 2. / w.sum()
//   laggauss:  w /= w.sum()
//   return x, w
//
// ROUTE. A composition: every statement already has a bit-exact NumSharp engine — U7's companion matrix, the OpenBLAS
// backend's eigvalsh (LAPACK syevd, the library NumPy itself loads), U3's evaluation kernel, U4's derivative, the house
// ufuncs, and np.max / np.sum, which are NumPy's own reduction schedules (a sum is NumPy's pairwise add.reduce) — so the
// composition is bit-exact by construction. NumPy's time goes into the three {p}val calls, each a Python loop over the
// deg + 1 coefficients; here each is one kernel call, and from degree ~100 the call is eigvalsh-bound on both sides.
// SPEED (benchmark/polynomial/polygauss_results.md, 22 cells, all bit-exact): geomean 5.45x NumPy; leggauss 1.63-9.9x,
// laggauss 2.1-18x; the lowest cells are degree 500, where both sides spend ~98% of the call in the same LAPACK syevd.
//
// `c = np.array([0] * deg + [1])` uses `deg` ITSELF, not ideg, so NumPy's sequence rules leak through:
//   * a Python int (a C# integer or BigInteger), a bool or an np.uint16 (a C# char) repeats the list: deg zeros and a 1
//     (int64), True giving one zero;
//   * a 0-d integer ARRAY is not a list repeat: `[0] * arr` is np.multiply, so c is [1] and the companion raises
//     "Series must have maximum degree of at least 1." for any positive array (probed for int64, uint64 and int8);
//   * a count past npy_intp cannot repeat a list: CPython's "cannot fit 'int' into an index-sized integer"
//     (OverflowError); a count of 2**60 - 1 or more is CPython's list size limit (an empty MemoryError, raised before
//     any allocation); smaller counts that do not fit in memory fail where NumSharp allocates.
//
// hermgauss / hermegauss (numpy/polynomial/hermite.py, hermite_e.py) build the same series (hermgauss with
// dtype=np.float64, which the companion's as_series conversion makes unobservable) and the same eigenvalues, but evaluate
// the NORMALISED Hermite functions instead of {p}val / {p}der (`_normed_hermite_n` / `_normed_hermite_e_n`, whose values
// do not overflow until much higher degrees):
//
//   dy = _normed_hermite_n(x, ideg)
//   df = _normed_hermite_n(x, ideg - 1) * np.sqrt(2 * ideg)        # hermite_e: np.sqrt(ideg)
//   x -= dy / df
//   fm = _normed_hermite_n(x, ideg - 1)
//   fm /= np.abs(fm).max()
//   w = 1 / (fm * fm)
//   w = (w + w[::-1]) / 2;  x = (x - x[::-1]) / 2
//   w *= np.sqrt(np.pi) / w.sum()                                    # hermite_e: np.sqrt(2 * np.pi)
//
// NumPy runs each recurrence as a Python loop of six ufunc calls per step; here each is ONE emitted kernel pass over the
// points (DirectILKernelGenerator.PolyNormedHermite.cs, which documents why the bits are NumPy's — NaN payloads included,
// which the recurrence reaches from degree ~1000, where its outer values overflow), over a step table written once per
// call: the degree-(n - 1) recurrences read the degree-n table from its second row.
//
// =============================================================================

namespace NumSharp
{
    /// <summary>
    ///     <c>leggauss</c>, <c>laggauss</c>, <c>hermgauss</c> and <c>hermegauss</c>: NumPy 2.4.2's statements (see the file
    ///     header) composed from the bit-exact companion, eigvalsh, evaluation, derivative, normalised-recurrence, ufunc and
    ///     reduction engines.
    /// </summary>
    internal static class NDPolyGauss
    {
        /// <summary>
        ///     <c>leggauss(deg)</c> / <c>laggauss(deg)</c>: the sample points and weights of Gauss quadrature for the basis's
        ///     weight function.
        /// </summary>
        /// <param name="basis"><see cref="PolyBasis.Legendre"/> or <see cref="PolyBasis.Laguerre"/>.</param>
        /// <param name="deg">The number of points: an integer under Python's <c>operator.index</c>.</param>
        /// <returns>(x, w): two new float64 arrays of <c>deg</c> elements.</returns>
        /// <exception cref="TypeError"><c>deg must be an integer, received {deg}</c>.</exception>
        /// <exception cref="ValueError"><c>deg must be a positive integer</c>, or the companion's <c>Series must have maximum
        ///     degree of at least 1.</c> for a 0-d array <paramref name="deg"/> (see the file header).</exception>
        /// <exception cref="OverflowException">A count past npy_intp (CPython's list repeat).</exception>
        /// <exception cref="OutOfMemoryException">A count CPython's list cannot hold, or one the machine cannot hold.</exception>
        /// <exception cref="NotSupportedException">No LAPACK backend is installed (<c>np.linalg.eigvalsh</c> needs
        ///     <c>NumSharp.Interop.OpenBLAS</c>).</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="basis"/> is not Legendre or Laguerre.</exception>
        public static (NDArray x, NDArray w) Gauss(PolyBasis basis, object deg)
        {
            if (basis is not (PolyBasis.Legendre or PolyBasis.Laguerre))
                throw new ArgumentOutOfRangeException(nameof(basis), basis, "leggauss / laggauss only");
            BigInteger ideg = PolyIndexArgument.AsInt(deg, "deg");
            if (ideg.Sign <= 0)
                throw new ValueError("deg must be a positive integer");

            // c = np.array([0] * deg + [1]); m = {p}companion(c); x = np.linalg.eigvalsh(m)
            NDArray c = SeriesOfDegree(deg, ideg);
            NDArray m = NDPolyAlgebra.Companion(basis, c);
            NDArray x = np.linalg.eigvalsh(m);

            // One Newton step: dy = {p}val(x, c); df = {p}val(x, {p}der(c)); x -= dy / df (in place, as NumPy does).
            NDArray dy = NDPolyEval.Val(basis, x, c, true);
            NDArray df = NDPolyEval.Val(basis, x, NDPolyCalc.Der(basis, c, 1, null, 0), true);
            np.subtract(x, np.divide(dy, df), @out: x);

            // The weights, scaled against overflow: fm /= |fm|.max(); df /= |df|.max(); w = 1 / (fm * df).
            NDArray fm = NDPolyEval.Val(basis, x, c["1:"], true);
            np.divide(fm, np.max(np.abs(fm)), @out: fm);
            np.divide(df, np.max(np.abs(df)), @out: df);
            // `1 / (fm * df)`: the Python int 1 is weak, so the loop is float64 and each element is 1.0 / v — computed into
            // the product's own fresh buffer (the same bytes as NumPy's new array; a float64 operand keeps the division on
            // the same-dtype kernel, where an int64 0-d one cost 0.8 us more at degree 5).
            NDArray w = np.multiply(fm, df);
            np.divide(NDArray.Scalar(1.0), w, @out: w);

            if (basis == PolyBasis.Legendre)
            {
                // The Legendre nodes are symmetric: w = (w + w[::-1]) / 2; x = (x - x[::-1]) / 2; w *= 2. / w.sum().
                // `/ 2` is a weak Python int: the float64 division by 2.0, done in the sum's / difference's fresh buffer.
                NDArray two = NDArray.Scalar(2.0);
                w = np.add(w, w["::-1"]);
                np.divide(w, two, @out: w);
                x = np.subtract(x, x["::-1"]);
                np.divide(x, two, @out: x);
                // `2. / w.sum()`: a Python float divided by the np.float64 sum (scalar math, one IEEE division).
                double scale = 2.0 / w.sum().GetDouble();
                np.multiply(w, NDArray.Scalar(scale), @out: w);
            }
            else
            {
                // w /= w.sum()
                np.divide(w, w.sum(), @out: w);
            }
            return (x, w);
        }

        /// <summary>
        ///     <c>hermgauss(deg)</c> / <c>hermegauss(deg)</c>: the sample points and weights of Gauss-Hermite quadrature for
        ///     the weight <c>exp(-x²)</c> (physicists') or <c>exp(-x²/2)</c> (probabilists').
        /// </summary>
        /// <param name="basis"><see cref="PolyBasis.Hermite"/> or <see cref="PolyBasis.HermiteE"/>.</param>
        /// <param name="deg">The number of points: an integer under Python's <c>operator.index</c>.</param>
        /// <returns>(x, w): two new float64 arrays of <c>deg</c> elements. From degree ~400 NumPy's own weights hold NaN
        ///     (the overflow-scaled <c>1 / (fm * fm)</c> reaches inf and the final scale is 0), from ~1000 its points too —
        ///     reproduced, NaN payloads included as far as the house reductions reproduce them.</returns>
        /// <exception cref="TypeError"><c>deg must be an integer, received {deg}</c>.</exception>
        /// <exception cref="ValueError"><c>deg must be a positive integer</c>, or the companion's <c>Series must have maximum
        ///     degree of at least 1.</c> for a 0-d array <paramref name="deg"/> (see the file header).</exception>
        /// <exception cref="OverflowException">A count past npy_intp (CPython's list repeat).</exception>
        /// <exception cref="OutOfMemoryException">A count CPython's list cannot hold, or one the machine cannot hold.</exception>
        /// <exception cref="NotSupportedException">No LAPACK backend is installed (<c>np.linalg.eigvalsh</c> needs
        ///     <c>NumSharp.Interop.OpenBLAS</c>).</exception>
        /// <exception cref="InvalidOperationException">IL generation is disabled (the recurrence is an emitted kernel).</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="basis"/> is not Hermite or HermiteE.</exception>
        public static unsafe (NDArray x, NDArray w) HermiteGauss(PolyBasis basis, object deg)
        {
            if (basis is not (PolyBasis.Hermite or PolyBasis.HermiteE))
                throw new ArgumentOutOfRangeException(nameof(basis), basis, "hermgauss / hermegauss only");
            BigInteger ideg = PolyIndexArgument.AsInt(deg, "deg");
            if (ideg.Sign <= 0)
                throw new ValueError("deg must be a positive integer");
            bool physicists = basis == PolyBasis.Hermite;

            // c = np.array([0] * deg + [1], dtype=np.float64) (hermegauss: no dtype); m = {p}companion(c); x = eigvalsh(m).
            // hermgauss's float64 series is the int64 one converted: the companion's as_series converts it anyway.
            NDArray c = SeriesOfDegree(deg, ideg);
            NDArray m = NDPolyAlgebra.Companion(basis, c);
            NDArray x = np.linalg.eigvalsh(m);
            // A valid deg reached here built a (deg x deg) matrix, so it fits a long (SeriesOfDegree refused past 2**60).
            long n = (long)ideg;
            var kind = physicists ? PolyNormedHermite.Hermite : PolyNormedHermite.HermiteE;

            // The step table of the degree-n recurrence, once: row j for nd = n - j (j < n - 1); the degree-(n - 1) calls
            // read it from row 1. At least one double, so a degree-1 call still has a buffer to point at (it reads none).
            NDArray table = new NDArray(NPTypeCode.Double, Shape.Vector(Math.Max(2 * (n - 1), 1)), false);
            double* steps = (double*)table.Unsafe.Address;
            DirectILKernelGenerator.GetPolyNormedHermiteStepsKernel(kind)(steps, n - 1, n);

            // One Newton step: dy = h_n(x); df = h_{n-1}(x) * np.sqrt(2 * ideg) (hermite_e: np.sqrt(ideg)); x -= dy / df.
            // np.sqrt of the Python int converts it to float64 first (exact) — then one IEEE sqrt, as Math.Sqrt is. The
            // product goes into h_{n-1}'s own fresh buffer: the same bytes as NumPy's new array.
            NDArray dy = NormedHermite(kind, x, steps, n);
            NDArray df = NormedHermite(kind, x, steps + 2, n - 1);
            np.multiply(df, NDArray.Scalar(Math.Sqrt(physicists ? 2.0 * n : n)), @out: df);
            np.subtract(x, np.divide(dy, df), @out: x);

            // The weights, scaled against overflow: fm = h_{n-1}(x); fm /= |fm|.max(); w = 1 / (fm * fm) — the Python
            // int 1 is weak, so each element is 1.0 / v, computed into the product's fresh buffer.
            NDArray fm = NormedHermite(kind, x, steps + 2, n - 1);
            np.divide(fm, np.max(np.abs(fm)), @out: fm);
            NDArray w = np.multiply(fm, fm);
            np.divide(NDArray.Scalar(1.0), w, @out: w);

            // The Hermite nodes are symmetric: w = (w + w[::-1]) / 2; x = (x - x[::-1]) / 2 (the weak `/ 2` divides by 2.0
            // in the sum's / difference's fresh buffer).
            NDArray two = NDArray.Scalar(2.0);
            w = np.add(w, w["::-1"]);
            np.divide(w, two, @out: w);
            x = np.subtract(x, x["::-1"]);
            np.divide(x, two, @out: x);
            // `w *= np.sqrt(np.pi) / w.sum()` (hermite_e: np.sqrt(2 * np.pi), an exact doubling of np.pi): the np.float64
            // scalar quotient (one IEEE division), then the in-place multiply.
            double scale = Math.Sqrt(physicists ? Math.PI : 2 * Math.PI) / w.sum().GetDouble();
            np.multiply(w, NDArray.Scalar(scale), @out: w);
            table.Dispose();
            return (x, w);
        }

        /// <summary>
        ///     NumPy's <c>_normed_hermite_n(x, n)</c> / <c>_normed_hermite_e_n(x, n)</c> at the contiguous float64 points
        ///     <paramref name="x"/> (see <see cref="PolyNormedHermiteKernel"/>).
        /// </summary>
        /// <param name="kind">The recurrence.</param>
        /// <param name="x">eigvalsh's points: 1-D, C-contiguous float64 (updated in place by the Newton step, never re-laid out).</param>
        /// <param name="table">The step table of degree <paramref name="n"/> (<see cref="PolyNormedHermiteStepsKernel"/>),
        ///     at least <c>2 * (n - 1)</c> doubles; unread for degrees 0 and 1.</param>
        /// <param name="n">The degree (0 or more).</param>
        /// <returns>A new float64 array of <c>x.size</c> elements.</returns>
        /// <exception cref="InvalidOperationException">IL generation is disabled.</exception>
        /// <remarks>
        ///     Degree 0 is NumPy's <c>np.full(x.shape, 1 / np.sqrt(np.sqrt(np.pi)))</c>: the recurrence's initial c1 in every
        ///     slot, whatever the points (a NaN point included).
        /// </remarks>
        private static unsafe NDArray NormedHermite(PolyNormedHermite kind, NDArray x, double* table, long n)
        {
            if (n == 0)
                return np.full(Shape.Vector(x.size), DirectILKernelGenerator.PolyNormedHermiteC1(kind), np.float64);
            var y = new NDArray(NPTypeCode.Double, Shape.Vector(x.size), false);
            DirectILKernelGenerator.GetPolyNormedHermiteKernel(kind)(
                (double*)x.Unsafe.Address, (double*)y.Unsafe.Address, x.size, table, n - 1);
            return y;
        }

        /// <summary>
        ///     NumPy's <c>c = np.array([0] * deg + [1])</c> for an argument <c>operator.index</c> accepted.
        /// </summary>
        /// <param name="deg">The argument as given.</param>
        /// <param name="ideg">Its integer (positive).</param>
        /// <returns>The coefficients: deg zeros and a one (int64) for a Python int / bool / char; NumPy's elementwise
        ///     <c>[0] * deg + [1]</c> (the one-element series [1]) for a 0-d array.</returns>
        /// <exception cref="OverflowException"><paramref name="ideg"/> past npy_intp (CPython's list repeat).</exception>
        /// <exception cref="OutOfMemoryException"><paramref name="ideg"/> at or past CPython's list size limit.</exception>
        private static NDArray SeriesOfDegree(object deg, BigInteger ideg)
        {
            if (deg is NDArray d)
            {
                // `[0] * arr` is np.multiply(list, array), not a list repeat, and `+ [1]` np.add: the one-element [1] (in
                // the operands' promoted dtype), which the companion matrix refuses.
                return np.add(np.multiply(np.array(new long[] { 0 }), d), np.array(new long[] { 1 }));
            }
            // list_repeat takes a Py_ssize_t (CPython's text for a larger int), and list_new_prealloc refuses a list of
            // more than PY_SSIZE_T_MAX / sizeof(PyObject*) items with an empty MemoryError — before anything is allocated.
            if (ideg > long.MaxValue)
                throw new OverflowException("cannot fit 'int' into an index-sized integer");
            if (ideg >= (1L << 60) - 1)
                throw new OutOfMemoryException(string.Empty);
            long n = (long)ideg;
            // np.array of Python ints is int64: deg zeros, then the 1.
            var c = np.zeros(new Shape(n + 1), np.int64);
            c.SetAtIndex(1L, n);
            return c;
        }
    }
}
