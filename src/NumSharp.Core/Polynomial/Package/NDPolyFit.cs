using System;
using System.Numerics;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

// =============================================================================
// NDPolyFit.cs — {p}fit for the six bases: polyutils._fit, least-squares fits of a series to data
// =============================================================================
//
// NumPy 2.4.2 (numpy/polynomial/polyutils.py), statement for statement — every {p}fit is
// `pu._fit({p}vander, x, y, deg, rcond, full, w)`:
//
//   x = np.asarray(x) + 0.0
//   y = np.asarray(y) + 0.0
//   deg = np.asarray(deg)
//   if deg.ndim > 1 or deg.dtype.kind not in 'iu' or deg.size == 0:
//       raise TypeError("deg must be an int or non-empty 1-D array of int")
//   if deg.min() < 0:
//       raise ValueError("expected deg >= 0")
//   if x.ndim != 1:                raise TypeError("expected 1D vector for x")
//   if x.size == 0:                raise TypeError("expected non-empty vector for x")
//   if y.ndim < 1 or y.ndim > 2:   raise TypeError("expected 1D or 2D array for y")
//   if len(x) != len(y):           raise TypeError("expected x and y to have same length")
//   if deg.ndim == 0:
//       lmax = deg;  order = lmax + 1;  van = vander_f(x, lmax)
//   else:
//       deg = np.sort(deg);  lmax = deg[-1];  order = len(deg);  van = vander_f(x, lmax)[:, deg]
//   lhs = van.T;  rhs = y.T
//   if w is not None:
//       w = np.asarray(w) + 0.0
//       if w.ndim != 1:            raise TypeError("expected 1D vector for w")
//       if len(x) != len(w):       raise TypeError("expected x and w to have same length")
//       lhs = lhs * w;  rhs = rhs * w
//   if rcond is None:
//       rcond = len(x) * np.finfo(x.dtype).eps
//   if issubclass(lhs.dtype.type, np.complexfloating):
//       scl = np.sqrt((np.square(lhs.real) + np.square(lhs.imag)).sum(1))
//   else:
//       scl = np.sqrt(np.square(lhs).sum(1))
//   scl[scl == 0] = 1
//   c, resids, rank, s = np.linalg.lstsq(lhs.T / scl, rhs.T, rcond)
//   c = (c.T / scl).T
//   if deg.ndim > 0:  cc = np.zeros((lmax + 1,) + c.shape[1:], dtype=c.dtype);  cc[deg] = c;  c = cc
//   if rank != order and not full:  warnings.warn("The fit may be poorly conditioned", RankWarning)
//   return (c, [resids, rank, s, rcond]) if full else c
//
// ROUTE. A composition of engines that are already NumPy's bit for bit: the house `+ 0.0` (PolyNumber's ufunc add with
// NumPy's weak Python float), U5's Vandermonde kernel, the house ufuncs, np.sum's add.reduce schedules, and
// np.linalg.lstsq (LAPACK gelsd through NumSharp.Interop.OpenBLAS, the library NumPy loads). Two statements are spelled
// in a layout-free form with the same bytes:
//   * `van[:, deg]` for a degree LIST is np.take of rows of the Vandermonde's (lmax + 1, n) buffer: NumPy's fancy index
//     returns a transposed view whose `.T` is C-ordered, and the norm's np.sum along axis 1 must walk that C order (its
//     pairwise schedule follows the layout; NumSharp's own `a[:, idx]` is C-ordered (n, k), whose `.T` is F);
//   * `lhs.T / scl` is computed as `lhs / scl[:, None]` and transposed: the same element-wise quotients, without the
//     F-ordered broadcast division (plan finding F2) — lstsq copies its operand into LAPACK's column order either way.
//
// RESULT. NumPy returns c (an ndarray) or the tuple (c, [resids, rank, s, rcond]); NumSharp returns an NDPolyFitResult,
// which IS the coefficient array (an NDArray: indexing, operators and every package function take it as is) and, for
// full=True, also carries the four statistics and deconstructs as NumPy's tuple. c's ownership is NumPy's: for a scalar
// degree `(c.T / scl).T` is a VIEW of the quotient (OWNDATA false), for a degree list `cc` owns its data.
//
// NOT REPRODUCED. RankWarning is not emitted (house convention). A None x / y / w is Python's TypeError for the object
// array's `+ 0.0`; any other object array (a Python int past uint64, a non-numeric item) is NotSupportedException.
//
// =============================================================================

namespace NumSharp
{
    /// <summary>
    ///     <c>{p}fit</c> for the six bases: NumPy 2.4.2's <c>polyutils._fit</c> (see the file header), composed from the
    ///     bit-exact <c>+ 0.0</c>, Vandermonde, ufunc, reduction and lstsq engines.
    /// </summary>
    internal static class NDPolyFit
    {
        /// <summary>NumPy's Python float <c>0.0</c> of <c>np.asarray(a) + 0.0</c> (weak: an array keeps its float dtype).</summary>
        private static readonly PolyNumber s_pyZero = PolyNumber.FromPython(PyScalar.Float(0.0));

        /// <summary>
        ///     <c>{p}fit(x, y, deg, rcond=None, full=False, w=None)</c>: the least-squares coefficients of a series in the basis
        ///     fitted to the points (x, y).
        /// </summary>
        /// <param name="basis">The basis (its Vandermonde function is NumPy's <c>vander_f</c>).</param>
        /// <param name="x">The sample points: anything <c>np.asarray</c> takes, 1-D after <c>+ 0.0</c>.</param>
        /// <param name="y">The values at the points: 1-D, or 2-D with one data set per column.</param>
        /// <param name="deg">An integer degree, or a 1-D array_like of integer degrees (only those terms are fitted).</param>
        /// <param name="rcond">The relative cutoff of the small singular values, or null for <c>len(x)*eps(x.dtype)</c>.</param>
        /// <param name="full">Also return the residuals, rank, singular values and rcond.</param>
        /// <param name="w">Weights for the points (1-D, length len(x)), or null.</param>
        /// <returns>The coefficients as an <see cref="NDPolyFitResult"/> (with the statistics when <paramref name="full"/>).</returns>
        /// <exception cref="TypeError">NumPy's argument TypeErrors (see the file header); a None x / y / w (Python's
        ///     <c>unsupported operand type(s) for +: 'NoneType' and 'float'</c>); a str x / y / w (the ufunc add's no-loop
        ///     text); lstsq's <c>array type float16 is unsupported in linalg</c>.</exception>
        /// <exception cref="ValueError"><c>expected deg &gt;= 0</c>; np.asarray's inhomogeneous-shape text; the Vandermonde's
        ///     size errors (<c>Maximum allowed dimension exceeded</c>, <c>array is too big; …</c>).</exception>
        /// <exception cref="NotSupportedException">An object array argument NumPy computes with as Python objects; no LAPACK
        ///     backend (<c>np.linalg.lstsq</c> needs <c>NumSharp.Interop.OpenBLAS</c>).</exception>
        /// <exception cref="OutOfMemoryException">A Vandermonde matrix the machine cannot hold (NumPy's MemoryError).</exception>
        public static NDPolyFitResult Fit(PolyBasis basis, object x, object y, object deg, object rcond, bool full, object w)
        {
            // x = np.asarray(x) + 0.0; y = np.asarray(y) + 0.0; deg = np.asarray(deg) — in NumPy's order, so a bad x is
            // reported before a bad y, and both before any degree error. For a numeric NDArray x the sum is not
            // materialized: the statements below read only its shape (which the add keeps) and its dtype, and NumPy's
            // vander_f adds 0.0 again, which the Vandermonde kernel does in its load stage — the add is idempotent, so the
            // matrix holds the bytes of vander_f(x + 0.0). A numeric NDArray's add cannot raise, so no error moves.
            NDArray xa = x is NDArray xnd && xnd.typecode != NPTypeCode.String ? xnd : PlusZero(x);
            NPTypeCode xType = PolyTyping.IsIntLike(xa.typecode) ? NPTypeCode.Double : xa.typecode;
            NDArray ya = PlusZero(y);
            NDArray degArr = DegreeArray(deg);
            if (DegreeMinimum(degArr).Sign < 0)
                throw new ValueError("expected deg >= 0");
            if (xa.ndim != 1)
                throw new TypeError("expected 1D vector for x");
            if (xa.size == 0)
                throw new TypeError("expected non-empty vector for x");
            if (ya.ndim < 1 || ya.ndim > 2)
                throw new TypeError("expected 1D or 2D array for y");
            long n = xa.Shape.dimensions[0];
            if (n != ya.Shape.dimensions[0])
                throw new TypeError("expected x and y to have same length");

            // van = vander_f(x, lmax); lhs = van.T (the Vandermonde's own (lmax + 1, n) C-ordered buffer). A degree list fits
            // only its terms: NumPy's van[:, deg] after sorting, here the same rows taken from that buffer (see the header).
            NDArray sorted = null;
            NDArray lhs;
            if (degArr.ndim == 0)
            {
                lhs = NDPolyVander.Vander(basis, xa, degArr).T;
            }
            else
            {
                sorted = np.sort(degArr);
                NDArray lmaxArr = sorted[sorted.size - 1];
                NDArray van = NDPolyVander.Vander(basis, xa, lmaxArr);
                lhs = np.take(van.T, sorted, axis: 0);
            }
            // rhs = y.T; lstsq receives rhs.T — y itself (the same view) until the weights multiply it.
            NDArray b = ya;

            // The weights multiply both sides (NumPy: "Don't use inplace operations as they can cause problems with NA").
            if (w is not null)
            {
                NDArray wa = PlusZero(w);
                if (wa.ndim != 1)
                    throw new TypeError("expected 1D vector for w");
                if (n != wa.Shape.dimensions[0])
                    throw new TypeError("expected x and w to have same length");
                lhs = np.multiply(lhs, wa);
                b = np.multiply(ya.T, wa).T;
            }

            // rcond = len(x) * np.finfo(x.dtype).eps: a Python int times a NumPy scalar of x's real float dtype (NEP 50: the
            // product is that dtype — float32 for a float32 x). A given rcond is passed through (converted for lstsq below).
            NDArray rcondArr = rcond is null ? DefaultRcond(xType, n) : RcondArray(rcond);

            // scl = sqrt(sum(|lhs|**2, axis=1)): the column norms of the design matrix, the complex form as NumPy spells it.
            NDArray scl = np.sqrt(ColumnNormsSquared(lhs));
            // scl[scl == 0] = 1: a column that is identically zero is left unscaled (lstsq then sees the zero column). The
            // masked assignment only runs when a zero exists (count_nonzero counts -0.0 as zero and NaN as nonzero, exactly
            // the elements `scl == 0` selects).
            if (np.count_nonzero(scl) != scl.size)
                scl[np.equal(scl, NDArray.Scalar(0.0))] = 1;

            // c, resids, rank, s = np.linalg.lstsq(lhs.T / scl, rhs.T, rcond). lstsq checks its operands' types (float16:
            // "array type float16 is unsupported in linalg") before its gufunc converts rcond, so that check runs first here.
            // The quotient is written over lhs, which is always a fresh array of this call (the Vandermonde buffer under its
            // transposed view, or the take / weights product), C-ordered: the same quotients without a second (k, n)
            // buffer (24.8 MB at n = 100,000, degree 30).
            NDArray sclColumn = np.expand_dims(scl, 1);
            NDArray a = np.divide(lhs, sclColumn, @out: lhs).T;
            np.linalg.CommonType(a, b);
            double rcondValue = RcondValue(rcond, rcondArr);
            var (c, resids, rank, s) = np.linalg.lstsq(a, b, rcondValue);

            // c = (c.T / scl).T: the quotient's transpose, a VIEW (OWNDATA false) as in NumPy.
            NDArray q = c.ndim == 1 ? np.divide(c, scl) : np.divide(c, sclColumn);
            NDArray coef;
            if (sorted is null)
            {
                coef = q;
            }
            else
            {
                // cc = np.zeros((lmax + 1,) + c.shape[1:], dtype=c.dtype); cc[deg] = c — the sorted degrees, a duplicate's
                // last row winning as in NumPy's sequential assignment.
                long rows = (long)PolyIndexArgument.AsInt(sorted[sorted.size - 1], "deg") + 1;
                var dims = new long[q.ndim];
                dims[0] = rows;
                for (int d = 1; d < q.ndim; d++)
                    dims[d] = q.Shape.dimensions[d];
                coef = np.zeros(Shape.Adopt(dims), q.dtype);
                coef[sorted] = q;
            }

            if (!full)
                return new NDPolyFitResult(sorted is null ? coef.Storage.Alias() : coef.Storage, null, null, null, null);
            // The statistics are owned by the result object (detached from the scope this call runs in): an [NDScoped]
            // caller returning the result yields only the coefficient array, so scope-tracked statistics would be reclaimed
            // under it.
            NDScope.Detach(resids);
            NDScope.Detach(rank);
            NDScope.Detach(s);
            if (!ReferenceEquals(rcondArr, rcond))
                NDScope.Detach(rcondArr);
            return new NDPolyFitResult(sorted is null ? coef.Storage.Alias() : coef.Storage, resids, rank, s, rcondArr);
        }

        /// <summary>
        ///     <c>np.square(lhs).sum(1)</c> (complex: <c>(np.square(lhs.real) + np.square(lhs.imag)).sum(1)</c>): the squared
        ///     norms of the design matrix's rows. For float32 / float64 / complex128 one fused <c>np.evaluate</c> pass, which
        ///     streams the squares into np.sum's own add.reduce schedule (the same bytes, without the squared temporary: 3.2 vs
        ///     8.3 µs at 11 × 1,000); the eager statements otherwise (float16 and decimal, whose fits lstsq refuses anyway).
        /// </summary>
        /// <param name="lhs">The design matrix, (k, n), in NumPy's layout (C-ordered).</param>
        /// <returns>The k squared norms, in lhs's real dtype.</returns>
        private static NDArray ColumnNormsSquared(NDArray lhs)
        {
            switch (lhs.typecode)
            {
                case NPTypeCode.Double:
                case NPTypeCode.Single:
                    return np.evaluate(NDExpr.Sum(NDExpr.Square((NDExpr)lhs), 1));
                case NPTypeCode.Complex:
                {
                    var l = (NDExpr)lhs;
                    return np.evaluate(NDExpr.Sum(NDExpr.Square(NDExpr.Real(l)) + NDExpr.Square(NDExpr.Imag(l)), 1));
                }
                default:
                    return np.sum(np.square(lhs), axis: 1);
            }
        }

        /// <summary>
        ///     NumPy's <c>np.asarray(a) + 0.0</c> for x / y / w: an array keeps its float or complex dtype (bool and integers
        ///     become float64), -0.0 becomes +0.0, a Python sequence goes through np.array's coercion first.
        /// </summary>
        /// <param name="a">The argument.</param>
        /// <returns>A new array (a 0-d one for a scalar).</returns>
        /// <exception cref="TypeError">None (Python's <c>unsupported operand type(s) for +: 'NoneType' and 'float'</c>, the
        ///     object array's add), or a str / list of strs (the ufunc add's no-loop text for <c>&lt;U…</c> and float64).</exception>
        /// <exception cref="ValueError">A ragged sequence (np.asarray's inhomogeneous-shape text).</exception>
        /// <exception cref="NotSupportedException">Any other object array (NumPy adds 0.0 to Python objects).</exception>
        private static NDArray PlusZero(object a)
        {
            switch (a)
            {
                case null:
                    // np.asarray(None) is a 0-d object array; its `+ 0.0` is Python's None + 0.0.
                    throw new TypeError("unsupported operand type(s) for +: 'NoneType' and 'float'");
                case string:
                    // A 0-d <U array: NumPy's ufunc add has no loop for it with a float64 (the house add raises NumPy's text).
                    return np.add(np.asanyarray(a), NDArray.Scalar(0.0));
            }
            if (PolySequence.IsSequence(a))
            {
                NDArray arr = PolySequence.ToArrayOrNonNumeric(a, out var nonNumeric);
                if (arr is null)
                {
                    if (nonNumeric.IsObject)
                        throw nonNumeric.Refusal;
                    // A str array: the same no-loop text as a str scalar.
                    return np.add(np.asanyarray(a), NDArray.Scalar(0.0));
                }
                return PolyNumber.Binary(BinaryOp.Add, PolyNumber.FromObject(arr), s_pyZero).ToNDArray();
            }
            if (a is NDArray nd && nd.typecode == NPTypeCode.String)
                return np.add(nd, NDArray.Scalar(0.0));
            return PolyNumber.Binary(BinaryOp.Add, PolyNumber.FromObject(a), s_pyZero).ToNDArray();
        }

        /// <summary>
        ///     <c>np.asarray(deg)</c> and NumPy's <c>deg.ndim &gt; 1 or deg.dtype.kind not in 'iu' or deg.size == 0</c> test:
        ///     a 0-d or 1-D, non-empty integer array.
        /// </summary>
        /// <param name="deg">The argument.</param>
        /// <returns>The degree array (a 0-d int64 / uint64 for a Python int, 0-d uint16 for a char, the caller's array or
        ///     np.asarray's conversion otherwise).</returns>
        /// <exception cref="TypeError"><c>deg must be an int or non-empty 1-D array of int</c>.</exception>
        /// <exception cref="ValueError">A ragged sequence (np.asarray's inhomogeneous-shape text, raised before the test).</exception>
        private static NDArray DegreeArray(object deg)
        {
            NDArray arr;
            switch (deg)
            {
                case NDArray nd:
                    arr = nd;
                    break;
                case char ch:
                    // np.uint16, NumSharp's char (the house map).
                    arr = NDArray.Scalar((ushort)ch);
                    break;
                case sbyte or byte or short or ushort or int or uint or long or ulong or BigInteger:
                {
                    // A Python int: np.asarray makes int64, uint64 past int64, and an OBJECT array past uint64.
                    BigInteger v = PolyIndexArgument.AsInt(deg, "deg");
                    if (v >= long.MinValue && v <= long.MaxValue)
                        arr = NDArray.Scalar((long)v);
                    else if (v.Sign > 0 && v <= ulong.MaxValue)
                        arr = NDArray.Scalar((ulong)v);
                    else
                        throw DegMustBeIntegers();
                    break;
                }
                default:
                    if (PolySequence.IsSequence(deg))
                    {
                        // np.asarray of a list: its coercion (a ragged list is its ValueError); a str / object list fails the
                        // kind test below.
                        arr = PolySequence.ToArrayOrNonNumeric(deg, out _) ?? throw DegMustBeIntegers();
                        break;
                    }
                    if (PolySequence.IsArrayLike(deg))
                    {
                        arr = np.asanyarray(deg);
                        break;
                    }
                    // A bool, float, complex, str, None or other object: np.asarray of it is not an integer array.
                    throw DegMustBeIntegers();
            }
            if (arr.ndim > 1 || arr.typecode == NPTypeCode.Boolean || !PolyTyping.IsIntLike(arr.typecode) || arr.size == 0)
                throw DegMustBeIntegers();
            return arr;
        }

        /// <summary>NumPy's <c>TypeError("deg must be an int or non-empty 1-D array of int")</c>.</summary>
        /// <returns>The exception, for the caller to throw.</returns>
        private static TypeError DegMustBeIntegers() => new TypeError("deg must be an int or non-empty 1-D array of int");

        /// <summary><c>deg.min()</c> of a 0-d or 1-D integer array, as an exact integer.</summary>
        /// <param name="deg">The degree array (non-empty).</param>
        /// <returns>The smallest degree.</returns>
        private static BigInteger DegreeMinimum(NDArray deg)
        {
            if (deg.ndim == 0)
                return PolyIndexArgument.AsInt(deg, "deg");
            return PolyIndexArgument.AsInt(np.amin(deg), "deg");
        }

        /// <summary>
        ///     NumPy's default <c>rcond = len(x) * np.finfo(x.dtype).eps</c>: a Python int times the NumPy scalar eps of x's
        ///     real float dtype, which NEP 50 computes in that dtype.
        /// </summary>
        /// <param name="xType">x's dtype after <c>+ 0.0</c> (float16 / float32 / float64 / complex128, or NumSharp's
        ///     decimal).</param>
        /// <param name="n">len(x).</param>
        /// <returns>The 0-d rcond (float64 for a float64 or complex x, float32 for float32, float16 for float16).</returns>
        private static NDArray DefaultRcond(NPTypeCode xType, long n)
        {
            NPTypeCode real = xType == NPTypeCode.Complex ? NPTypeCode.Double : xType;
            // The two common dtypes directly. eps is a power of two (2^-52, 2^-23), so the product is exact once len(x) is
            // in the dtype: a Python int converts to float64 correctly rounded (PyLong_AsDouble), as C#'s long -> double
            // conversion does, and to float32 exactly up to 2^24 (past it the Python-int conversion is modelled below).
            if (real == NPTypeCode.Double)
                return NDArray.Scalar((double)n * np.finfo(NPTypeCode.Double).eps);
            if (real == NPTypeCode.Single && n <= 1L << 24)
                return NDArray.Scalar((float)n * (float)np.finfo(NPTypeCode.Single).eps);
            NDArray eps = NDArray.Scalar(np.finfo(real).eps).astype(real);
            return PolyNumber.Binary(BinaryOp.Multiply, PolyNumber.FromPython(PyScalar.Int(n)), PolyNumber.FromObject(eps))
                .ToNDArray();
        }

        /// <summary>
        ///     A given rcond as the object NumPy returns in the full statistics: an NDArray itself, a number as the 0-d array
        ///     of its discovered dtype (a Python float is float64, a Python int int64, a bool bool), a str or a sequence as
        ///     np.asarray of it (lstsq refuses both when it converts them, see <see cref="RcondValue"/>).
        /// </summary>
        /// <param name="rcond">The caller's rcond (not null).</param>
        /// <returns>The array (the caller's own array for an NDArray).</returns>
        /// <exception cref="NotSupportedException">A value with no numeric dtype (a sequence holding None, …).</exception>
        private static NDArray RcondArray(object rcond)
        {
            switch (rcond)
            {
                case NDArray nd:
                    return nd;
                case string:
                    return np.asanyarray(rcond);
            }
            if (PolySequence.IsSequence(rcond))
                return PolySequence.ToArray(rcond);
            return PolyNumber.FromObject(rcond).ToNDArray();
        }

        /// <summary>
        ///     rcond as lstsq's gufunc receives it: converted to double (NumPy's lstsq computes in double for every float
        ///     dtype, so a float32 rcond is widened exactly).
        /// </summary>
        /// <param name="rcond">The caller's rcond as given (null for the computed default).</param>
        /// <param name="array">Its array (<see cref="RcondArray"/>, or the default's 0-d).</param>
        /// <returns>The double.</returns>
        /// <exception cref="TypeError">A Python complex (<c>float() argument must be a string or a real number, not
        ///     'complex'</c>, NumPy's conversion of a weak complex to the float64 cutoff); a complex or str array (the
        ///     gufunc's same_kind cast text, <c>Cannot cast ufunc 'lstsq' input 2 from dtype('&lt;U3') to dtype('float64')
        ///     with casting rule 'same_kind'</c>).</exception>
        /// <exception cref="NotSupportedException">An rcond array of one or more dimensions: NumPy broadcasts lstsq's gufunc
        ///     over it (giving every result an extra leading axis), which NumSharp's lstsq does not.</exception>
        private static double RcondValue(object rcond, NDArray array)
        {
            if (rcond is Complex)
                throw new TypeError("float() argument must be a string or a real number, not 'complex'");
            if (array.ndim > 0)
                throw new NotSupportedException(
                    "an rcond array with one or more dimensions broadcasts NumPy's lstsq gufunc over it (every result gains its " +
                    "axes); NumSharp's lstsq takes a scalar cutoff");
            if (array.typecode is NPTypeCode.Complex or NPTypeCode.String)
                throw new TypeError(
                    $"Cannot cast ufunc 'lstsq' input 2 from {array.dtype.ToString(true)} to dtype('float64') with casting rule 'same_kind'");
            // The default rcond is a float64 or float32 0-d array: read it in place (both widen exactly); any other dtype
            // through the cast.
            return array.typecode switch
            {
                NPTypeCode.Double => array.GetAtIndex<double>(0),
                NPTypeCode.Single => array.GetAtIndex<float>(0),
                _ => array.astype(NPTypeCode.Double).GetAtIndex<double>(0),
            };
        }
    }
}
