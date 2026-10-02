using System;
using System.Numerics;
using NumSharp.Backends;
using NumSharp.Backends.Kernels;

// =============================================================================
// NDPolyAlgebra.Roots.cs — numpy.polynomial's companion matrices and roots (plan U7)
// =============================================================================
//
// WHAT NUMPY DOES (numpy/polynomial/*.py, 2.4.2)
// ----------------------------------------------
// {p}companion(c): `[c] = as_series([c])` (trim + np.common_type + copy); fewer than two terms raise
// `Series must have maximum degree of at least 1.`; two terms return the 1x1 matrix of the linear root (scalarmath on
// NumPy scalars: `np.array([[-c[0] / c[1]]])`, lag `1 + c[0] / c[1]`, herm `-.5 * c[0] / c[1]`); otherwise, with
// n = len(c) - 1, `mat = np.zeros((n, n), dtype=c.dtype)`, its diagonals assigned through `mat.reshape(-1)[k::n+1]`
// (top = super, bot = sub, mid = main), and ONE in-place update of the last column:
//
//   poly   bot = 1                                     mat[:, -1] -= c[:-1] / c[-1]
//   cheb   top = [sqrt(.5), .5, ...]; bot = top        mat[:, -1] -= (c[:-1] / c[-1]) * (scl / scl[-1]) * .5
//          scl = [1., sqrt(.5) * (n - 1)]
//   leg    top = arange(1, n) * scl[:-1] * scl[1:]     mat[:, -1] -= (c[:-1] / c[-1]) * (scl / scl[-1]) * (n / (2n - 1))
//          scl = 1. / sqrt(2 * arange(n) + 1); bot = top
//   lag    top = -arange(1, n); mid = 2.*arange(n) + 1.; bot = top
//                                                      mat[:, -1] += (c[:-1] / c[-1]) * n
//   herm   top = sqrt(.5 * arange(1, n)); bot = top    mat[:, -1] -= scl * c[:-1] / (2.0 * c[-1])
//          scl = cumprod([1., 1./sqrt(2.*arange(n-1, 0, -1))])[::-1]
//   herme  top = sqrt(arange(1, n)); bot = top         mat[:, -1] -= scl * c[:-1] / c[-1]
//          scl = cumprod([1., 1./sqrt(arange(n-1, 0, -1))])[::-1]
//
// {p}roots(c): the same as_series; fewer than two terms give `np.array([], dtype=c.dtype)`, two the 1-element array of
// the linear root; otherwise `np.linalg.eigvals` of the companion matrix — ROTATED (`[::-1, ::-1]`, "reduces error")
// for every basis but the power series — sorted in place.
//
// DTYPES (the part that decides the bits)
// ---------------------------------------
// c.dtype is np.common_type's: float16 / float32 / float64 / complex128 (NumSharp's decimal). The helper vectors (scl,
// top, mid) are FLOAT64 — np.sqrt of an int64 ramp, a Python float times one — so wherever one meets the series the
// ufunc loop is result_type(c.dtype, float64): for a float16 / float32 series the cheb / leg / herm / herme last column
// is computed in FLOAT64 and rounded ONCE into the matrix by the in-place op's same_kind output cast, the diagonals are
// float64 values cast by setitem, while poly / lag stay in the series dtype throughout (their updates involve only the
// series and Python ints). A complex series runs complex128 loops with the float64 helper converted to (s + 0j) — the
// house simd_cmul product, operand order kept. Verified against NumPy over 9,600 random / special-value series (every
// basis x float16 / float32 / float64 / complex128, raw bytes, NaN payloads included) with an explicit model of exactly
// these statements before this file was written.
//
// HOW IT RUNS HERE
// ----------------
// Statement for statement over the U2 arena (NDPolyAlgebra.cs), each element operation through the house kernel NumPy's
// own statement runs through in NumSharp: the binary ufunc loops (House: true division, simd_cmul, Smith, the HALF loop),
// the conversions (GetPolyCastKernel), np.sqrt's unary loop, np.multiply.accumulate's sequential scan, the diagonal
// assignments (GetDiagWriteKernel: a strided store, a stride-0 source broadcasting a scalar) and np.arange (the int64
// ramp kernel of DirectILKernelGenerator.PolyRoots.cs). The roots then compose np.linalg.eigvals and the in-place sort,
// so they inherit eigvals' backend contract: with NumSharp.Interop.OpenBLAS referenced it is NumPy's own LAPACK geev
// (byte-identical at one thread), without a backend it raises the backend-missing NotSupportedException.
//
// =============================================================================

namespace NumSharp
{
    internal static unsafe partial class NDPolyAlgebra
    {
        /// <summary>NumPy's text for a series of fewer than two terms passed to <c>{p}companion</c>.</summary>
        private const string DegreeTooLow = "Series must have maximum degree of at least 1.";

        /// <summary>
        ///     NumPy's text for <c>np.isfinite</c> of an OBJECT array — the TypeError np.linalg.eigvals' <c>_assert_finite</c>
        ///     raises on the object companion matrix of an object series (its first check that reads the dtype).
        /// </summary>
        private const string ObjectIsFinite = "ufunc 'isfinite' not supported for the input types, and the inputs could not be " +
                                              "safely coerced to any supported types according to the casting rule ''safe''";

        // =============================================================================================
        //  Entry points
        // =============================================================================================

        /// <summary>
        ///     <c>{p}companion(c)</c> (see the file header): as_series, the length checks, the linear root's 1x1 matrix for a
        ///     two-term series, otherwise the basis's (scaled) companion matrix built by NumPy's statements.
        /// </summary>
        /// <param name="basis">The basis.</param>
        /// <param name="c">The series (anything <c>np.array</c> accepts, 1-D).</param>
        /// <returns>A new (deg, deg) matrix of the series' common type (owning, C-contiguous).</returns>
        /// <exception cref="ValueError">An empty or non-1-d series, no common type (a bool or str series), or fewer than two
        ///     terms after trimming (<c>Series must have maximum degree of at least 1.</c>) — in NumPy's order.</exception>
        /// <exception cref="NotSupportedException">An object series (None, a non-numeric object, a Python int past uint64) of
        ///     three or more terms after trimming whose arithmetic does not raise — NumPy's object companion matrix, a dtype
        ///     NumSharp does not have — or one holding None / a str where NumPy's object arithmetic reaches it (CPython's
        ///     TypeError).</exception>
        /// <exception cref="OverflowException">An object series whose arithmetic meets a Python int too large for a float:
        ///     CPython's <c>integer division result too large for a float</c> (Python int / Python int) or <c>int too large to
        ///     convert to float</c> (any other operand), whichever NumPy's statements raise first.</exception>
        public static NDArray Companion(PolyBasis basis, object c)
        {
            var v = NDPolySeries.AsCoefficientArray(c);
            NDPolySeries.Validate(v);
            if (!NDPolySeries.TryCommonType(new ReadOnlySpan<PolySeriesView>(in v), out NPTypeCode t))
            {
                // as_series went on in the object dtype, and NumPy's next statement is the length check on the TRIMMED object
                // array — so a series whose trailing zeros leave one term raises NumPy's ValueError, not the refusal. Two terms
                // make the linear root, arithmetic on the object array's ITEMS whose result NumPy wraps in a numeric array. A
                // longer object series is an object matrix NumSharp cannot hold — unless the last-column arithmetic raises
                // first, which is NumPy's answer then.
                long trimmed = ObjectTrimLength(c, v);
                if (trimmed < 2)
                    throw new ValueError(DegreeTooLow);
                if (trimmed == 2)
                    return ObjectLinearRoot(basis, c, new Shape(1, 1));
                ObjectCompanionArithmetic(basis, c, trimmed);
                throw v.Refusal;
            }
            long len = NDPolySeries.TrimLength(v);
            if (len < 2)
                throw new ValueError(DegreeTooLow);
            var a = PolyArena.Enter();
            try
            {
                var cs = Load(a, v, len, t);
                // `np.array([[root]])`: a fresh owning (1, 1) array of the root scalar's dtype.
                return len == 2
                    ? ScalarArray(LinearRoot(basis, cs), new Shape(1, 1))
                    : CompanionMatrix(a, basis, cs);
            }
            finally
            {
                a.Exit();
            }
        }

        /// <summary>
        ///     <c>{p}roots(c)</c> (see the file header): as_series, the empty / linear short cuts, otherwise the sorted
        ///     eigenvalues of the companion matrix (rotated for every basis but the power series).
        /// </summary>
        /// <param name="basis">The basis.</param>
        /// <param name="c">The series.</param>
        /// <returns>
        ///     The roots, ascending (complex lexicographically): an empty array of the series' common type for a constant
        ///     series, the one linear root for two terms, otherwise <c>np.linalg.eigvals</c>' result sorted in place — real when
        ///     every eigenvalue is (a view of the complex result's real parts, as NumPy's), complex otherwise.
        /// </returns>
        /// <exception cref="ValueError">An empty or non-1-d series, or no common type (a bool or str series).</exception>
        /// <exception cref="TypeError">A float16 or decimal series of degree 2 or more (linalg's
        ///     <c>array type float16 is unsupported in linalg</c>); an OBJECT series of three or more terms whose companion
        ///     arithmetic does not raise (eigvals' <c>ufunc 'isfinite' not supported for the input types, …</c> on NumPy's object
        ///     companion matrix — raised before LAPACK, so no backend is needed).</exception>
        /// <exception cref="LinAlgError">A companion matrix holding an infinity or NaN (<c>Array must not contain infs or
        ///     NaNs</c>), or eigenvalues that do not converge.</exception>
        /// <exception cref="MissingBackendException">Degree 2 or more with no LAPACK backend (reference
        ///     NumSharp.Interop.OpenBLAS — NumSharp.Core ships no eigensolver).</exception>
        /// <exception cref="NotSupportedException">An object series of one term (NumPy's <c>np.array([], dtype=object)</c>), or
        ///     one holding None / a str where NumPy's object arithmetic reaches it (CPython's TypeError) — NumSharp has no
        ///     object dtype.</exception>
        /// <exception cref="OverflowException">An object series whose arithmetic meets a Python int too large for a float:
        ///     CPython's <c>integer division result too large for a float</c> (Python int / Python int) or <c>int too large to
        ///     convert to float</c> (any other operand), whichever NumPy's statements raise first.</exception>
        public static NDArray Roots(PolyBasis basis, object c)
        {
            var v = NDPolySeries.AsCoefficientArray(c);
            NDPolySeries.Validate(v);
            // An object series: one term is `np.array([], dtype=object)` — refused, NumSharp has no object dtype. Two terms are
            // the linear root, computed from the ITEMS into a numeric array exactly as NumPy's `np.array([root])` is. Three or
            // more build NumPy's object companion matrix, whose arithmetic may raise first; otherwise eigvals' first check that
            // reads the dtype, isfinite, rejects the object matrix with a TypeError — never an object result.
            if (!NDPolySeries.TryCommonType(new ReadOnlySpan<PolySeriesView>(in v), out NPTypeCode t))
            {
                long trimmed = ObjectTrimLength(c, v);
                if (trimmed == 2)
                    return ObjectLinearRoot(basis, c, new Shape(1));
                if (trimmed < 2)
                    throw v.Refusal;
                ObjectCompanionArithmetic(basis, c, trimmed);
                throw new TypeError(ObjectIsFinite);
            }
            long len = NDPolySeries.TrimLength(v);
            if (len < 2)
                return new NDArray(t, new Shape(0), false);   // np.array([], dtype=c.dtype)
            NDArray m;
            var a = PolyArena.Enter();
            try
            {
                var cs = Load(a, v, len, t);
                if (len == 2)
                    return ScalarArray(LinearRoot(basis, cs), new Shape(1));   // np.array([root])
                m = CompanionMatrix(a, basis, cs);
            }
            finally
            {
                a.Exit();
            }
            // NumPy rotates every basis's companion but the power series' — `[::-1, ::-1]`, a VIEW (eigvals copies it into
            // LAPACK's column-major layout through its strides, as the backend does here). np.flip over every axis is that
            // view (negated strides, shifted offset) without parsing a slice string.
            var target = basis == PolyBasis.Power ? m : np.flip(m);
            var r = np.linalg.eigvals(target);
            r.sort();
            return r;
        }

        // =============================================================================================
        //  The pieces
        // =============================================================================================

        /// <summary>
        ///     The root of a two-term series — NumPy's scalarmath on the as_series copy's NumPy scalars: <c>-c[0] / c[1]</c>
        ///     (power, Chebyshev, Legendre, HermiteE), <c>1 + c[0] / c[1]</c> (Laguerre: the Python int 1 is weak, so the sum
        ///     keeps the series dtype), <c>-.5 * c[0] / c[1]</c> (Hermite: the product with the Python float first).
        /// </summary>
        /// <param name="basis">The basis.</param>
        /// <param name="c">The two-term series.</param>
        /// <returns>A NumPy scalar of the series dtype.</returns>
        private static PolyNumber LinearRoot(PolyBasis basis, in PolySer c)
            => LinearRoot(basis, PolyNumber.FromScalar(c.T, c.At(0)), PolyNumber.FromScalar(c.T, c.At(1)));

        /// <summary>
        ///     The linear root's expression on two operands of any kind (see <see cref="LinearRoot(PolyBasis, in PolySer)"/>):
        ///     NumPy scalars of the as_series copy, or the ITEMS of an object series — Python numbers (CPython arithmetic:
        ///     exact ints, correctly rounded int / int), NumPy scalars (scalarmath) and 0-d arrays (ufuncs), the three
        ///     arithmetics <see cref="PolyNumber.Binary"/> dispatches between.
        /// </summary>
        /// <param name="basis">The basis.</param>
        /// <param name="c0">The constant term.</param>
        /// <param name="c1">The linear term.</param>
        /// <returns>The root, of the kind and dtype NumPy's expression produces.</returns>
        /// <exception cref="OverflowException">Python int operands whose quotient exceeds the float range (CPython's
        ///     <c>integer division result too large for a float</c>). The linear term is never zero here: trimseq removed
        ///     it, so CPython's ZeroDivisionError cannot arise.</exception>
        private static PolyNumber LinearRoot(PolyBasis basis, in PolyNumber c0, in PolyNumber c1)
        {
            return basis switch
            {
                PolyBasis.Laguerre => PolyNumber.Binary(BinaryOp.Add, PolyNumber.FromPython(PyScalar.Int(1)),
                    PolyNumber.Binary(BinaryOp.Divide, c0, c1)),
                PolyBasis.Hermite => PolyNumber.Binary(BinaryOp.Divide,
                    PolyNumber.Binary(BinaryOp.Multiply, PolyNumber.FromPython(PyScalar.Float(-0.5)), c0), c1),
                _ => PolyNumber.Binary(BinaryOp.Divide, PolyNumber.Negate(c0), c1),
            };
        }

        /// <summary>
        ///     <c>np.array([...[x]...])</c> of one NumPy scalar: a fresh, owning array of <paramref name="shape"/> (one element)
        ///     and the scalar's dtype — not a reshaped view, so its flags are NumPy's (C and F contiguous, OWNDATA).
        /// </summary>
        /// <param name="x">The scalar.</param>
        /// <param name="shape">(1,) or (1, 1).</param>
        /// <returns>The array.</returns>
        private static NDArray ScalarArray(in PolyNumber x, Shape shape)
        {
            NPTypeCode t = x.DiscoveredDtype();
            var r = new NDArray(t, shape, false);
            x.WriteElement(t, (byte*)r.Storage.Address);
            return r;
        }

        /// <summary>
        ///     The linear root of a two-term OBJECT series (a Python int past uint64 among the terms makes NumPy's as_series
        ///     copy an object array, whose items keep their kinds — Python numbers, NumPy scalars, 0-d arrays): NumPy's
        ///     expression evaluated on the items, then <c>np.array([...])</c> of the result — a numeric array (float64 for a
        ///     Python float, complex128 for a Python complex, a NumPy scalar's or 0-d array's own dtype), not an object one.
        /// </summary>
        /// <param name="basis">The basis.</param>
        /// <param name="c">The caller's argument: a Python sequence whose first two items are the trimmed series.</param>
        /// <param name="shape"><c>(1, 1)</c> for <c>{p}companion</c>, <c>(1,)</c> for <c>{p}roots</c>.</param>
        /// <returns>The fresh owning array.</returns>
        /// <exception cref="NotSupportedException">A None or str item: NumPy's object arithmetic raises its TypeError there
        ///     (<c>bad operand type for unary -: 'NoneType'</c>), and NumSharp has no object dtype to compute it with.</exception>
        /// <exception cref="OverflowException">Python int items whose quotient exceeds the float range (CPython's
        ///     <c>integer division result too large for a float</c>).</exception>
        private static NDArray ObjectLinearRoot(PolyBasis basis, object c, Shape shape)
        {
            object[] items = PolySequence.Items(c);
            // FromObject keeps each item's kind: a BigInteger is a Python int, Half a NumPy float16 scalar, an NDArray a 0-d
            // array (its operations are ufuncs) — the same objects NumPy's object array holds.
            var root = LinearRoot(basis, PolyNumber.FromObject(items[0]), PolyNumber.FromObject(items[1]));
            return ScalarArray(root, shape);
        }

        /// <summary>
        ///     The element operations of NumPy's OBJECT companion statements that can raise, run in NumPy's order for an object
        ///     series of three or more terms — so the caller reports NumPy's first error, or knows there is none (NumPy then
        ///     holds an object matrix: <c>{p}companion</c>'s result, which NumSharp refuses, and <c>{p}roots</c>' isfinite
        ///     TypeError).
        /// </summary>
        /// <param name="basis">The basis.</param>
        /// <param name="c">The caller's argument: a Python sequence whose first <paramref name="len"/> items are the trimmed
        ///     series.</param>
        /// <param name="len">The trimmed length (≥ 3).</param>
        /// <remarks>
        ///     <para>
        ///     NumPy's last-column statement (file header) runs object-dtype ufuncs: each item meets its operand through
        ///     Python's own operator, element by element in order, and the first exception ends the statement. With numeric
        ///     items only one error is possible — a Python int too large for a float (≥ 2**1024 after rounding): CPython's
        ///     <c>integer division result too large for a float</c> when it is divided by a Python int (or bool) and the
        ///     quotient overflows, <c>int too large to convert to float</c> when it meets anything else. Only the FIRST array
        ///     operation can meet a Python int — its results are floats (true division) or the products of a Python float —
        ///     except Hermite's <c>2.0 * c[-1]</c> and HermiteE's <c>/ c[-1]</c>, which meet the leading item again:
        ///     </para>
        ///     <list type="bullet">
        ///       <item>poly / cheb / leg / lag: <c>c[:-1] / c[-1]</c>; what follows multiplies floats and adds them to the
        ///         matrix's numbers.</item>
        ///       <item>herm: <c>scl * c[:-1]</c>, then the plain Python product <c>2.0 * c[-1]</c>; their quotient divides floats
        ///         by a nonzero float.</item>
        ///       <item>herme: <c>scl * c[:-1]</c>, then <c>... / c[-1]</c>.</item>
        ///     </list>
        ///     <para>
        ///     An operand that is not the object array's own item — the float64 helper <c>scl</c>, a divisor <c>c[-1]</c> — is
        ///     first CAST to the object dtype by the ufunc, which makes a NumPy scalar or 0-d array its Python value
        ///     (<see cref="ObjectLoopValue"/>): <c>huge / np.int64(3)</c> inside the loop is Python int / Python int (the
        ///     division text), where the bare expression would be the conversion text. The helper's values cannot change
        ///     whether a product raises (a Python float meeting a too-large int raises whatever its value), so 1.0 stands for
        ///     each. None, a str or another object an operation reaches is CPython's TypeError there — NotSupportedException
        ///     here (<see cref="PolyNumber.FromObject"/>), the documented object-arithmetic divergence.
        ///     </para>
        /// </remarks>
        /// <exception cref="OverflowException">CPython's overflow texts above, from the first element that raises.</exception>
        /// <exception cref="NotSupportedException">An operation reaches None, a str or another object NumSharp cannot read as a
        ///     number (NumPy raises CPython's TypeError there).</exception>
        private static void ObjectCompanionArithmetic(PolyBasis basis, object c, long len)
        {
            object[] items = PolySequence.Items(c);
            long n = len - 1;
            if (basis == PolyBasis.Hermite || basis == PolyBasis.HermiteE)
            {
                // `scl * c[:-1]`: the float64 helper array, cast to the object dtype, times each item, in order.
                var helper = PolyNumber.FromPython(PyScalar.Float(1.0));
                var products = new PolyNumber[n];
                for (long i = 0; i < n; i++)
                    products[i] = PolyNumber.Binary(BinaryOp.Multiply, helper, PolyNumber.FromObject(items[i]));
                if (basis == PolyBasis.Hermite)
                {
                    // `(2.0 * c[-1])`: Python's own operator on the item (outside any ufunc, so the item keeps its kind).
                    PolyNumber.Binary(BinaryOp.Multiply, PolyNumber.FromPython(PyScalar.Float(2.0)), PolyNumber.FromObject(items[n]));
                    return;
                }
                var divisor = ObjectLoopValue(items[n]);
                for (long i = 0; i < n; i++)
                    PolyNumber.Binary(BinaryOp.Divide, products[i], divisor);
                return;
            }
            // `c[:-1] / c[-1]` (power, Chebyshev, Legendre, Laguerre): the divisor cast to the object dtype once, before the loop.
            var d = ObjectLoopValue(items[n]);
            for (long i = 0; i < n; i++)
                PolyNumber.Binary(BinaryOp.Divide, PolyNumber.FromObject(items[i]), d);
        }

        /// <summary>
        ///     An operand as an object-dtype ufunc loop receives it after NumPy's cast to the object dtype: a Python number
        ///     unchanged; a NumPy scalar or 0-d array its element as the Python value NumPy's getitem makes — float16 / float32 /
        ///     float64 a Python float (exact widening), an integer (char included) a Python int, a bool a Python bool, a complex
        ///     a Python complex.
        /// </summary>
        /// <param name="item">The operand (an item of the caller's sequence).</param>
        /// <returns>The Python value.</returns>
        /// <exception cref="NotSupportedException">None, a str or another object NumSharp cannot read as a number (NumPy's
        ///     object arithmetic raises CPython's TypeError when it reaches it), or a decimal — NumSharp's dtype with no NumPy
        ///     analog, so no Python value an object array could hold.</exception>
        private static PolyNumber ObjectLoopValue(object item)
        {
            var x = PolyNumber.FromObject(item);
            if (x.IsPython)
                return x;
            // The element, boxed as its CLR type (a 0-d array's single element, or the NumPy scalar's value).
            object e = x.Kind == PolyNumberKind.Array ? x.Array.GetAtIndex(0) : x.ToObject();
            return PolyNumber.FromPython(e switch
            {
                bool b => PyScalar.Bool(b),
                Half h => PyScalar.Float((double)h),
                float f => PyScalar.Float(f),
                double f => PyScalar.Float(f),
                Complex z => PyScalar.Cplx(z),
                char ch => PyScalar.Int(ch),
                ulong u => PyScalar.Int(new BigInteger(u)),
                decimal => throw new NotSupportedException(
                    "a decimal item has no Python value NumPy's object array could hold (NumSharp's decimal has no NumPy analog)"),
                _ => PyScalar.Int(Convert.ToInt64(e)),   // the remaining signed / unsigned integer scalars
            });
        }

        /// <summary>
        ///     The length trimseq keeps of an OBJECT array argument (as_series went on in the object dtype): Python's
        ///     <c>seq[i] != 0</c> scanned from the end — a Python or NumPy number equal to zero is trimmed (-0.0 included, a
        ///     NaN is not), None, a str or any other object stops the scan; an all-zero tail leaves one term.
        /// </summary>
        /// <param name="c">The caller's argument (for a view longer than one element, a Python sequence of scalars: the
        ///     view passed as_series' 1-d check).</param>
        /// <param name="v">Its converted view.</param>
        /// <returns>The trimmed length (≥ 1).</returns>
        private static long ObjectTrimLength(object c, in PolySeriesView v)
        {
            if (v.Len <= 1)
                return v.Len;
            object[] items = PolySequence.Items(c);
            for (long i = items.Length - 1; i > 0; i--)
                if (!IsPythonZero(items[i]))
                    return i + 1;
            return 1;
        }

        /// <summary>
        ///     Python's <c>x == 0</c> for one item of an object array: true only for a number equal to zero (a Python bool /
        ///     int / float / complex, a NumPy scalar, a 0-d array's element); None, a str and every other object compare
        ///     unequal.
        /// </summary>
        /// <param name="o">The item.</param>
        /// <returns>True when Python's <c>o != 0</c> is False.</returns>
        private static bool IsPythonZero(object o)
        {
            // An object array keeps a 0-d array item AS an array, and `ndarray != 0` is its one-element elementwise comparison,
            // whose truth value trimseq reads: `np.array(0.0)` and `np.array(-0.0)` are trimmed like the number zero.
            if (o is NDArray nd)
                return nd.ndim == 0 && !PolyNumber.FromObject(nd).NotZero();
            if (o is null || o is string || !PolyNumber.IsScalarValue(o))
                return false;
            try
            {
                return !PolyNumber.FromObject(o).NotZero();
            }
            catch (NotSupportedException)
            {
                // An object NumSharp cannot read as a number is not a number Python would call zero.
                return false;
            }
        }

        /// <summary>
        ///     The companion matrix of a series of three or more terms (see the file header), its statements replayed in
        ///     NumPy's order and dtypes: the zero matrix, the basis's diagonals, the last-column update.
        /// </summary>
        /// <param name="a">The arena (holds the helper vectors).</param>
        /// <param name="basis">The basis.</param>
        /// <param name="c">The as_series copy (≥ 3 terms, its dtype the matrix dtype).</param>
        /// <returns>A new owning C-contiguous (n, n) matrix, n = len(c) - 1.</returns>
        private static NDArray CompanionMatrix(PolyArena a, PolyBasis basis, in PolySer c)
        {
            long n = c.N - 1;
            NPTypeCode t = c.T;
            int size = c.Size;
            // np.zeros((n, n), dtype=c.dtype). Up to the pool's cap a zero-FILLED allocation hands out fresh OS pages
            // (calloc), and the diagonal writes below touch every 4 KB page once — one demand-zero fault each: a float64
            // companion of degree 200 spent ~70 of its ~75 µs faulting (measured). A recycled pool buffer zeroed in one
            // memset pass costs a few µs instead (the write-once rule np.tri follows); above the cap the lazy OS-zeroed
            // pages are kept, so only the pages a diagonal touches fault. (double arithmetic: n * n * size cannot overflow.)
            NDArray mat;
            double bytes = (double)n * n * size;
            if (bytes <= long.MaxValue && np.PrefersWriteOnce((long)bytes))
            {
                mat = new NDArray(t, new Shape(n, n), false);
                np.ZeroBytes((byte*)mat.Storage.Address, (long)bytes);
            }
            else
                mat = new NDArray(t, new Shape(n, n), true);
            byte* m = (byte*)mat.Storage.Address;
            var mark = a.Position;
            switch (basis)
            {
                case PolyBasis.Power:
                {
                    // bot = mat.reshape(-1)[n::n+1]; bot[...] = 1
                    ulong* one = stackalloc ulong[2];
                    WeakInt(1, t, (byte*)one);
                    StoreDiagonal(m, n, size, n, (byte*)one, 0, n - 1);
                    // mat[:, -1] -= c[:-1] / c[-1]
                    LastColumnUpdate(a, BinaryOp.Subtract, m, n, t, LeadingQuotient(a, c));
                    break;
                }
                case PolyBasis.Chebyshev:
                {
                    // scl = np.array([1.] + [np.sqrt(.5)] * (n - 1)) — np.sqrt of a Python float is the correctly rounded
                    // float64 square root, the value Math.Sqrt returns.
                    double s = Math.Sqrt(0.5);
                    var scl = New(a, n, NPTypeCode.Double);
                    *(double*)scl.P = 1.0;
                    StoreStrided(scl.P + sizeof(double), sizeof(double), (byte*)&s, 0, n - 1, sizeof(double));
                    // top[0] = np.sqrt(.5); top[1:] = 1 / 2  (setitem: each float64 cast to the matrix dtype); bot[...] = top
                    ulong* tv = stackalloc ulong[2];
                    CastElement((byte*)&s, NPTypeCode.Double, t, (byte*)tv);
                    StoreDiagonal(m, n, size, 1, (byte*)tv, 0, 1);
                    double half = 0.5;
                    CastElement((byte*)&half, NPTypeCode.Double, t, (byte*)tv);
                    StoreDiagonal(m, n, size, 1 + (n + 1), (byte*)tv, 0, n - 2);
                    StoreDiagonal(m, n, size, n, m + size, (n + 1) * size, n - 1);
                    // mat[:, -1] -= (c[:-1] / c[-1]) * (scl / scl[-1]) * .5
                    var ratio = OpScalar(BinaryOp.Divide, scl, scl.At(n - 1), New(a, n, NPTypeCode.Double).P);
                    var prod = Promoted(a, BinaryOp.Multiply, LeadingQuotient(a, c), ratio);
                    var upd = OpPyFloat(BinaryOp.Multiply, prod, 0.5, prod.P);
                    LastColumnUpdate(a, BinaryOp.Subtract, m, n, t, upd);
                    break;
                }
                case PolyBasis.Legendre:
                {
                    // scl = 1. / np.sqrt(2 * np.arange(n) + 1)   (the int64 ramp converted by np.sqrt's float64 loop)
                    var scl = Converted(a, Ramp(a, n, 1, 2), NPTypeCode.Double);
                    SqrtInPlace(scl);
                    PyFloatOp(BinaryOp.Divide, 1.0, scl, scl.P);
                    // top[...] = np.arange(1, n) * scl[:n - 1] * scl[1:n]   (float64, left to right), cast by setitem; bot = top
                    var k = Converted(a, Ramp(a, n - 1, 1, 1), NPTypeCode.Double);
                    House(BinaryOp.Multiply, ExecutionPath.SimdFull, NPTypeCode.Double, k.P, scl.P, k.P, n - 1);
                    House(BinaryOp.Multiply, ExecutionPath.SimdFull, NPTypeCode.Double, k.P, scl.At(1), k.P, n - 1);
                    var top = Converted(a, k, t);
                    StoreDiagonal(m, n, size, 1, top.P, size, n - 1);
                    StoreDiagonal(m, n, size, n, m + size, (n + 1) * size, n - 1);
                    // mat[:, -1] -= (c[:-1] / c[-1]) * (scl / scl[-1]) * (n / (2 * n - 1))   (a Python float: int / int,
                    // correctly rounded — the IEEE quotient of the two exactly converted ints)
                    var ratio = OpScalar(BinaryOp.Divide, scl, scl.At(n - 1), New(a, n, NPTypeCode.Double).P);
                    var prod = Promoted(a, BinaryOp.Multiply, LeadingQuotient(a, c), ratio);
                    var upd = OpPyFloat(BinaryOp.Multiply, prod, (double)n / (2 * n - 1), prod.P);
                    LastColumnUpdate(a, BinaryOp.Subtract, m, n, t, upd);
                    break;
                }
                case PolyBasis.Laguerre:
                {
                    // top[...] = -np.arange(1, n)   (int64, cast by setitem)
                    var top = Converted(a, Ramp(a, n - 1, -1, -1), t);
                    StoreDiagonal(m, n, size, 1, top.P, size, n - 1);
                    // mid[...] = 2. * np.arange(n) + 1.   (float64 — exact — cast by setitem)
                    var mid = Converted(a, Converted(a, Ramp(a, n, 1, 2), NPTypeCode.Double), t);
                    StoreDiagonal(m, n, size, 0, mid.P, size, n);
                    StoreDiagonal(m, n, size, n, m + size, (n + 1) * size, n - 1);
                    // mat[:, -1] += (c[:-1] / c[-1]) * n   (the Python int n is weak: the series dtype throughout)
                    var q = LeadingQuotient(a, c);
                    LastColumnUpdate(a, BinaryOp.Add, m, n, t, OpInt(BinaryOp.Multiply, q, n, q.P));
                    break;
                }
                case PolyBasis.Hermite:
                case PolyBasis.HermiteE:
                {
                    bool physicists = basis == PolyBasis.Hermite;
                    // scl = np.hstack((1., 1. / np.sqrt(2. * np.arange(n - 1, 0, -1))))   (HermiteE: no 2. *)
                    var s0 = New(a, n, NPTypeCode.Double);
                    *(double*)s0.P = 1.0;
                    var tail = new PolySer(s0.At(1), n - 1, NPTypeCode.Double);
                    var ks = Ramp(a, n - 1, n - 1, -1);
                    DirectILKernelGenerator.GetPolyCastKernel(NPTypeCode.Int64, NPTypeCode.Double)(ks.P, n - 1, sizeof(long), tail.P);
                    if (physicists)
                        PyFloatOp(BinaryOp.Multiply, 2.0, tail, tail.P);
                    SqrtInPlace(tail);
                    PyFloatOp(BinaryOp.Divide, 1.0, tail, tail.P);
                    // scl = np.multiply.accumulate(scl)[::-1]   (the sequential scan, read backwards)
                    var acc = New(a, n, NPTypeCode.Double);
                    CumProd(s0, acc);
                    var scl = New(a, n, NPTypeCode.Double);
                    StoreStrided(scl.P, sizeof(double), acc.At(n - 1), -sizeof(double), n, sizeof(double));
                    // top[...] = np.sqrt(.5 * np.arange(1, n))   (HermiteE: np.sqrt(np.arange(1, n))), cast by setitem; bot = top
                    var k = Converted(a, Ramp(a, n - 1, 1, 1), NPTypeCode.Double);
                    if (physicists)
                        PyFloatOp(BinaryOp.Multiply, 0.5, k, k.P);
                    SqrtInPlace(k);
                    var top = Converted(a, k, t);
                    StoreDiagonal(m, n, size, 1, top.P, size, n - 1);
                    StoreDiagonal(m, n, size, n, m + size, (n + 1) * size, n - 1);
                    // mat[:, -1] -= scl * c[:-1] / (2.0 * c[-1])   (HermiteE: / c[-1]) — (scl * c[:-1]) first, in
                    // result_type(float64, c.dtype); the divisor a NumPy scalar of the series dtype (2.0 * c[-1] is
                    // scalarmath), so it widens into the product's loop.
                    var prod = Promoted(a, BinaryOp.Multiply, scl, c.Sub(0, n));
                    var div = PolyNumber.FromScalar(t, c.At(n));
                    if (physicists)
                        div = PolyNumber.Binary(BinaryOp.Multiply, PolyNumber.FromPython(PyScalar.Float(2.0)), div);
                    ulong* dv = stackalloc ulong[2];
                    div.WriteElement(t, (byte*)dv);
                    var upd = OpPromotedScalar(a, BinaryOp.Divide, prod, (byte*)dv, t);
                    LastColumnUpdate(a, BinaryOp.Subtract, m, n, t, upd);
                    break;
                }
                default:
                    throw new ArgumentOutOfRangeException(nameof(basis), basis, "not a numpy.polynomial basis");
            }
            a.Release(mark);
            return mat;
        }

        // =============================================================================================
        //  Statement helpers (one house kernel call each)
        // =============================================================================================

        /// <summary>
        ///     <c>c[:-1] / c[-1]</c> — the array op every builder but Hermite / HermiteE starts its last-column update with:
        ///     the series divided by its NumPy-scalar last coefficient, in the series dtype (true division; Smith's
        ///     algorithm for complex, the float32-computed HALF loop for float16).
        /// </summary>
        /// <param name="a">The arena.</param>
        /// <param name="c">The series (≥ 2 terms).</param>
        /// <returns>A fresh series of len(c) - 1 elements.</returns>
        private static PolySer LeadingQuotient(PolyArena a, in PolySer c)
        {
            long n = c.N - 1;
            return OpScalar(BinaryOp.Divide, c.Sub(0, n), c.At(n), New(a, n, c.T).P);
        }

        /// <summary><c>np.arange(start, start + n*step, step)</c> as int64, into the arena (the ramp kernel).</summary>
        /// <param name="a">The arena.</param>
        /// <param name="n">Element count (≥ 0).</param>
        /// <param name="start">First value.</param>
        /// <param name="step">Difference between consecutive values.</param>
        /// <returns>A fresh int64 series.</returns>
        private static PolySer Ramp(PolyArena a, long n, long start, long step)
        {
            var r = New(a, n, NPTypeCode.Int64);
            DirectILKernelGenerator.GetPolyRampKernel()((long*)r.P, n, start, step);
            return r;
        }

        /// <summary>
        ///     <c>s.astype(t)</c> — the conversion NumPy's ufunc loop or setitem applies to an operand of another dtype
        ///     (the house cast: int64 → float exact below 2^53, float64 → float16 rounded once, x → complex as x + 0j).
        /// </summary>
        /// <param name="a">The arena.</param>
        /// <param name="s">The series.</param>
        /// <param name="t">The target dtype.</param>
        /// <returns><paramref name="s"/> itself when it already has dtype <paramref name="t"/>, else a fresh converted copy.</returns>
        private static PolySer Converted(PolyArena a, in PolySer s, NPTypeCode t)
        {
            if (s.T == t)
                return s;
            var r = New(a, s.N, t);
            DirectILKernelGenerator.GetPolyCastKernel(s.T, t)(s.P, s.N, s.Size, r.P);
            return r;
        }

        /// <summary><c>np.sqrt</c> of a float64 series, in place (the house unary loop: the correctly rounded square root).</summary>
        /// <param name="s">The float64 series (overwritten).</param>
        private static void SqrtInPlace(in PolySer s)
        {
            long n = s.N;
            if (n <= 0)
                return;
            long stride = 1;
            // The contiguous kernel reads only the pointers and the count; the stride / shape arguments are well-formed anyway.
            DirectILKernelGenerator.GetUnaryKernel(new UnaryKernelKey(NPTypeCode.Double, NPTypeCode.Double, UnaryOp.Sqrt, true))
                (s.P, s.P, &stride, &n, 1, n);
        }

        /// <summary>
        ///     <c>np.multiply.accumulate</c> of a float64 series into <paramref name="dst"/> — the house scan, sequential
        ///     (<c>dst[i] = dst[i-1] * src[i]</c> from the multiplicative identity, which reproduces NumPy's
        ///     <c>out[0] = in[0]</c> for the finite values the Hermite scales hold).
        /// </summary>
        /// <param name="src">The series.</param>
        /// <param name="dst">The result (same length; may not overlap <paramref name="src"/> partially).</param>
        private static void CumProd(in PolySer src, in PolySer dst)
        {
            long n = src.N;
            if (n <= 0)
                return;
            long stride = 1;
            DirectILKernelGenerator.GetCumulativeKernel(new CumulativeKernelKey(NPTypeCode.Double, NPTypeCode.Double, ReductionOp.CumProd, true))
                (src.P, dst.P, &stride, &n, 1, n);
        }

        /// <summary><c>s OP f</c> for a Python float <paramref name="f"/> (weak under NEP 50: converted into s's dtype first).</summary>
        /// <param name="op">The op.</param>
        /// <param name="s">The series.</param>
        /// <param name="f">The Python float.</param>
        /// <param name="dst">The result's memory (s.N elements; may be s.P).</param>
        /// <returns>The result.</returns>
        private static PolySer OpPyFloat(BinaryOp op, in PolySer s, double f, byte* dst)
        {
            ulong* fv = stackalloc ulong[2];
            CastElement((byte*)&f, NPTypeCode.Double, s.T, (byte*)fv);
            House(op, ExecutionPath.SimdScalarRight, s.T, s.P, (byte*)fv, dst, s.N);
            return new PolySer(dst, s.N, s.T);
        }

        /// <summary><c>f OP s</c> for a Python float <paramref name="f"/> on the LEFT (<c>1. / x</c>, <c>2. * x</c>, <c>.5 * x</c>).</summary>
        /// <param name="op">The op.</param>
        /// <param name="f">The Python float.</param>
        /// <param name="s">The series.</param>
        /// <param name="dst">The result's memory (s.N elements; may be s.P).</param>
        /// <returns>The result.</returns>
        private static PolySer PyFloatOp(BinaryOp op, double f, in PolySer s, byte* dst)
        {
            ulong* fv = stackalloc ulong[2];
            CastElement((byte*)&f, NPTypeCode.Double, s.T, (byte*)fv);
            House(op, ExecutionPath.SimdScalarLeft, s.T, (byte*)fv, s.P, dst, s.N);
            return new PolySer(dst, s.N, s.T);
        }

        /// <summary>
        ///     <c>x OP y</c> for two equal-length arrays of possibly different dtypes: both converted to the ufunc's loop dtype
        ///     <c>result_type(x, y)</c>, then one house loop in operand order (a complex product's last bit depends on it).
        /// </summary>
        /// <param name="a">The arena.</param>
        /// <param name="op">The op.</param>
        /// <param name="x">Left operand.</param>
        /// <param name="y">Right operand (same length).</param>
        /// <returns>A fresh series of the loop dtype.</returns>
        private static PolySer Promoted(PolyArena a, BinaryOp op, in PolySer x, in PolySer y)
        {
            NPTypeCode r = PolyTyping.Promote(x.T, y.T);
            var xr = Converted(a, x, r);
            var yr = Converted(a, y, r);
            var dst = New(a, x.N, r);
            House(op, ExecutionPath.SimdFull, r, xr.P, yr.P, dst.P, x.N);
            return dst;
        }

        /// <summary>
        ///     <c>s OP x</c> for an array and a NumPy scalar of dtype <paramref name="tx"/>: a NumPy scalar is STRONG under
        ///     NEP 50, so the loop dtype is <c>result_type(s, x)</c> and both are converted to it first.
        /// </summary>
        /// <param name="a">The arena.</param>
        /// <param name="op">The op.</param>
        /// <param name="s">The array.</param>
        /// <param name="x">The scalar's element.</param>
        /// <param name="tx">The scalar's dtype.</param>
        /// <returns>A fresh series of the loop dtype.</returns>
        private static PolySer OpPromotedScalar(PolyArena a, BinaryOp op, in PolySer s, byte* x, NPTypeCode tx)
        {
            NPTypeCode r = PolyTyping.Promote(s.T, tx);
            var sr = Converted(a, s, r);
            ulong* xv = stackalloc ulong[2];
            CastElement(x, tx, r, (byte*)xv);
            var dst = New(a, s.N, r);
            House(op, ExecutionPath.SimdScalarRight, r, sr.P, (byte*)xv, dst.P, s.N);
            return dst;
        }

        /// <summary>
        ///     NumPy's in-place <c>mat[:, -1] OP= x</c>: the ufunc reads the strided last column, runs in
        ///     <c>result_type(mat, x)</c> (x's dtype for every builder), and writes back through the same_kind output cast —
        ///     so a float16 / float32 matrix receives one rounding of the float64 result.
        /// </summary>
        /// <param name="a">The arena.</param>
        /// <param name="op">Add or Subtract.</param>
        /// <param name="mat">Element (0, 0) of the C-contiguous (n, n) matrix.</param>
        /// <param name="n">The matrix order.</param>
        /// <param name="t">The matrix dtype.</param>
        /// <param name="x">The right operand (n elements).</param>
        private static void LastColumnUpdate(PolyArena a, BinaryOp op, byte* mat, long n, NPTypeCode t, in PolySer x)
        {
            int size = DirectILKernelGenerator.GetTypeSize(t);
            byte* col = mat + (n - 1) * size;
            long colStride = n * size;
            NPTypeCode r = PolyTyping.Promote(t, x.T);
            // Gather the column, widened into the loop dtype (the cast kernel reads a strided source).
            var acc = New(a, n, r);
            DirectILKernelGenerator.GetPolyCastKernel(t, r)(col, n, colStride, acc.P);
            var xr = Converted(a, x, r);
            House(op, ExecutionPath.SimdFull, r, acc.P, xr.P, acc.P, n);
            // The output cast back to the matrix dtype, then the strided store into the column.
            var back = Converted(a, acc, t);
            StoreStrided(col, colStride, back.P, size, n, size);
        }

        /// <summary>
        ///     <c>mat.reshape(-1)[first::n+1][...] = src</c> for the first <paramref name="count"/> slots: a strided store
        ///     along a diagonal (<paramref name="srcStride"/> 0 broadcasts one element; a stride of <c>(n+1)*size</c> copies
        ///     another diagonal of the same matrix).
        /// </summary>
        /// <param name="mat">Element (0, 0).</param>
        /// <param name="n">The matrix order.</param>
        /// <param name="size">The itemsize.</param>
        /// <param name="first">Flat index of the diagonal's first element (0 main, 1 super, n sub).</param>
        /// <param name="src">The first source element.</param>
        /// <param name="srcStride">Byte stride between source elements.</param>
        /// <param name="count">Elements written (≤ 0 writes nothing).</param>
        private static void StoreDiagonal(byte* mat, long n, int size, long first, byte* src, long srcStride, long count)
            => StoreStrided(mat + first * size, (n + 1) * size, src, srcStride, count, size);

        /// <summary>The house strided store (<see cref="DirectILKernelGenerator.GetDiagWriteKernel"/>): <c>dst[i*dstStride] = src[i*srcStride]</c>.</summary>
        /// <param name="dst">First destination element.</param>
        /// <param name="dstStride">Signed byte stride between destination elements.</param>
        /// <param name="src">First source element.</param>
        /// <param name="srcStride">Signed byte stride between source elements (0 broadcasts).</param>
        /// <param name="count">Elements written (≤ 0 writes nothing).</param>
        /// <param name="size">The itemsize (1, 2, 4, 8 or 16).</param>
        /// <exception cref="InvalidOperationException">IL generation is disabled.</exception>
        private static void StoreStrided(byte* dst, long dstStride, byte* src, long srcStride, long count, int size)
        {
            if (count <= 0)
                return;
            var k = DirectILKernelGenerator.GetDiagWriteKernel(DirectILKernelGenerator.CopyKindFor(size))
                    ?? throw new InvalidOperationException("IL generation is disabled");
            k(dst, dstStride, src, srcStride, count);
        }
    }
}
