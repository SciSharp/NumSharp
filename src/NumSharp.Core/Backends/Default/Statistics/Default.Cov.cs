using System;
using NumSharp.Backends.Kernels;

// =============================================================================
// Default.Cov.cs - DefaultEngine's np.cov (numpy/lib/_function_base_impl.py: cov)
// =============================================================================
//
// DefaultEngine's override of TensorEngine.Cov (TensorEngine.FunctionBase.cs): NumPy's statements, statement for
// statement - the operand checks, `X = array(m, ndmin=2, dtype=dtype)` and the empty-variables return, y's conversion
// and concatenation, the frequency / analytic weight checks, the weighted mean and its weight sum (NumPy's
// `average(..., returned=True)`), the normalization factor, the in-place centring `X -= avg[:, None]`, the product
// `dot(X, X_T.conj())` (a managed Gram kernel for a few real variables without a BLAS backend), the scale and the closing
// squeeze; and over text, NumPy's statements up to the reduction that refuses it.
//
// The body is the static CovFunction, because the text route (CovOfText, static) calls it again for parsed text; the
// override owns the [NDScoped] scope both run under.
//
// History: the np-layer dtype plan's P2 moved these statements from the public np.cov into the internal operation class
// CovEngine; its phase PE (E1) made that class this DefaultEngine partial. The facade keeps the matrix boundary's plain
// region and the null check. Its phase P6 made each np.* call of the statements (np.average, np.dot, np.concatenate,
// ...) the member its facade dispatches to, and moved the file from beside the facade to Backends/Default/Statistics/.
// =============================================================================

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     DefaultEngine's <see cref="TensorEngine.Cov"/>: <see cref="CovFunction"/>, NumPy's whole body, under this
        ///     member's scope.
        /// </summary>
        /// <param name="m">The variables (not null): a 1-D or 2-D array, rows are variables unless
        ///     <paramref name="rowvar"/> is false.</param>
        /// <param name="y">An additional set of variables, or null.</param>
        /// <param name="rowvar">Whether rows are variables (columns observations).</param>
        /// <param name="bias">Normalize by N instead of N - 1 (overridden by <paramref name="ddof"/>).</param>
        /// <param name="ddof">The delta degrees of freedom, or null for <paramref name="bias"/>'s default.</param>
        /// <param name="fweights">The integer frequency weights, or null.</param>
        /// <param name="aweights">The analytic weights, or null.</param>
        /// <param name="dtype">The requested dtype, or null for at least float64.</param>
        /// <returns>The covariance matrix, which survives the scope (every intermediate is reclaimed).</returns>
        /// <exception cref="Exception">The errors of <see cref="CovFunction"/>, in its order.</exception>
        /// <remarks>
        ///     The scope was np.cov's until the plan's phase PE (the body was <c>[NDScopedCovered]</c> under it); as a public
        ///     entry point the override owns it now, so a direct engine caller's intermediates are reclaimed too. np.cov keeps
        ///     its own <c>[NDScoped]</c> (nested scopes).
        /// </remarks>
        [NDScoped]
        public override NDArray Cov(NDArray m, NDArray y = null, bool rowvar = true, bool bias = false, int? ddof = null,
            NDArray fweights = null, NDArray aweights = null, DType dtype = null)
            => CovFunction(m, y, rowvar, bias, ddof, fweights, aweights, dtype);

        /// <summary>
        ///     <c>cov(m, y, rowvar, bias, ddof, fweights, aweights, dtype)</c> after the facade's checks: NumPy's whole body.
        /// </summary>
        /// <param name="m">The variables (not null): a 1-D or 2-D array, rows are variables unless
        ///     <paramref name="rowvar"/> is false.</param>
        /// <param name="y">An additional set of variables, or null.</param>
        /// <param name="rowvar">Whether rows are variables (columns observations).</param>
        /// <param name="bias">Normalize by N instead of N - 1 (overridden by <paramref name="ddof"/>).</param>
        /// <param name="ddof">The delta degrees of freedom, or null for <paramref name="bias"/>'s default.</param>
        /// <param name="fweights">The integer frequency weights, or null.</param>
        /// <param name="aweights">The analytic weights, or null.</param>
        /// <param name="dtype">The requested dtype, or null for at least float64.</param>
        /// <returns>The covariance matrix: a fresh array the caller owns (a view of one for a squeezed result, a reshape
        ///     view of an empty array for no variables).</returns>
        /// <exception cref="ValueError">More than 2 dimensions; text variables whose <paramref name="m"/> and
        ///     <paramref name="y"/> have different observation counts; a negative weight; text that does not parse into a
        ///     numeric <paramref name="dtype"/>; a <c>StringDType</c> request over no observations.</exception>
        /// <exception cref="IncorrectShapeException">Numeric variables with different observation counts (concatenate).</exception>
        /// <exception cref="TypeError">A non-integral frequency weight; text the mean refuses.</exception>
        /// <exception cref="RuntimeError">A malformed weight array.</exception>
        /// <exception cref="DTypePromotionError"><c>StringDType</c> variables without <paramref name="dtype"/>, or a
        ///     <c>StringDType</c> request with weights.</exception>
        /// <exception cref="ArgumentException">An integer or bool <paramref name="dtype"/> (the in-place centring's cast
        ///     refusal).</exception>
        /// <exception cref="DivideByZeroException">Weights that sum to zero.</exception>
        /// <exception cref="OverflowException">An integer text outside an integer <paramref name="dtype"/>, or decimal
        ///     arithmetic out of range.</exception>
        /// <exception cref="UnicodeEncodeError">A U → S conversion of non-ASCII text.</exception>
        /// <exception cref="UnicodeDecodeError">An S → U conversion of non-ASCII bytes.</exception>
        /// <exception cref="InvalidOperationException">Not reached (see <see cref="CovOfText"/>).</exception>
        [NDScopedCovered] // only callers: the [NDScoped] Cov override and CovOfText's parsed-text call, which runs under it
        internal NDArray CovFunction(NDArray m, NDArray y, bool rowvar, bool bias, int? ddof, NDArray fweights,
            NDArray aweights, DType dtype)
        {
            // --- Check inputs. NumPy's "ddof must be integer" ValueError is enforced by the int? type. ---
            if (m.ndim > 2)
                throw new ValueError("m has more than 2 dimensions");
            if (y is not null && y.ndim > 2)
                throw new ValueError("y has more than 2 dimensions");

            // Text (an operand, or a text dtype= request: S, U or StringDType): NumPy's statements run up to the first
            // reduction, which refuses text — or the empty-variables early return, which yields the (0, 0) float64 result.
            if (Backends.Strings.NDStringUfuncs.IsText(m) || Backends.Strings.NDStringUfuncs.IsText(y) ||
                Backends.Strings.NDStringUfuncs.IsTextRequest(dtype))
                return CovOfText(m, y, rowvar, bias, ddof, fweights, aweights, dtype);

            // Result dtype: at least float64 (complex input stays complex). np.result_type(m, [y,] float64)'s statement
            // promotes the descriptor before the arrays (array_result_type's order).
            NPTypeCode resultType = dtype?.GetTypeCode() ?? (y is null
                ? DTypePromotion.ResultType(NPTypeCode.Double, m)
                : DTypePromotion.ResultType(NPTypeCode.Double, m, y));

            // Every intermediate below — atleast_2d/astype/T/concatenate, the weight casts and
            // validation masks, avg and its column view, Xc/Xt and the pre-scale c — is scope-
            // tracked and reclaimed when the Cov override's scope exits (throw paths included); only the
            // yielded result survives. Inputs were constructed before the scope opened: never touched.

            int mNdim = m.ndim;

            // X = array(m, ndmin=2, dtype=dtype): a fresh 2-D array in the result dtype.
            NDArray X = NDArray.atleast_2d(m).astype(resultType);
            if (!rowvar && mNdim != 1)
                X = X.T;

            // Empty variables => (0, 0) float64, regardless of input dtype (NumPy: np.array([]).reshape(0, 0)).
            if (X.Shape.dimensions[0] == 0)
                return EmptyCovariance();

            if (y is not null)
            {
                NDArray Y = NDArray.atleast_2d(y).astype(resultType);
                if (!rowvar && Y.Shape.dimensions[0] != 1)
                    Y = Y.T;
                X = NDArray.concatenate(new[] { X, Y }, 0, null, null, "same_kind");
            }

            int ddofValue = ddof ?? (bias ? 0 : 1);

            long nobs = X.Shape.dimensions[1];

            // --- Combined weights w = fweights * aweights (either may be absent). ---
            NDArray w = CovWeights(fweights, aweights, nobs, out NDArray aweightsCast);

            // --- Weighted mean along the observation axis. ---
            NDArray avg = X.TensorEngine.Average(X, new[] { 1 }, w, false, false).avg;

            // Real weight sum (NumPy's w_sum[0]; its imaginary part, present only for complex input,
            // is always zero). `fact` is mathematically real, so it is carried in a double.
            double wSum = w is null ? nobs : w.TensorEngine.Sum(w, null, null, false).GetDouble(0);

            // --- Normalization factor (NumPy's `fact`). ---
            double fact;
            if (w is null)
                fact = nobs - ddofValue;
            else if (ddofValue == 0)
                fact = wSum;
            else if (aweightsCast is null)
                fact = wSum - ddofValue;
            else
            {
                // ddof * sum(w*aweights)
                NDArray weighted = w * aweightsCast;
                double numer = ddofValue * weighted.TensorEngine.Sum(weighted, null, null, false).GetDouble(0);
                // NumPy computes `.../ w_sum`. For complex input w_sum is complex128, so NumPy divides
                // via Smith's method as numer*(1/w_sum) — a reciprocal-multiply whose real part rounds
                // 1 ULP differently from real division. It only matters when fact lands ~0 (degenerate
                // degrees of freedom), but there it decides finite-vs-inf, so match it.
                double ratio = resultType == NPTypeCode.Complex ? numer * (1.0 / wSum) : numer / wSum;
                fact = wSum - ratio;
            }

            // --- Center the observations, weight them, and form the covariance via one GEMM. ---
            NDArray Xc = X - avg.reshape(avg.size, 1);
            // NumPy does `X -= avg[:, None]` in place: the subtract's output is X, so the difference is cast back into X's
            // dtype under the in-place operator's same_kind rule. A float or complex X keeps its dtype (a float32 / float16
            // X takes a float64 weighted mean's difference rounded back); an integer or bool X — an integer or bool dtype=
            // request — cannot take the float64 mean's difference: NumPy's UFuncTypeError ("Cannot cast ufunc 'subtract'
            // output from dtype('float64') to dtype('int32') with casting rule 'same_kind'"), the house ArgumentException.
            if (Xc.typecode != X.typecode)
            {
                Backends.DefaultEngine.ValidateOutCast(Xc.typecode, X.typecode, "subtract");
                Xc = Xc.astype(X.typecode);
            }

            // c = dot(Xc, Xc_T.conj()) — a SYMMETRIC Gram (syrk). For a small number of real
            // (Single/Double) variables over a contiguous Xc, a dedicated pairwise-SIMD Gram
            // kernel computes it ~10x faster than routing the (M,K)@(K,M) product — with its
            // tiny inner loop and strided transposed operand — through the general GEMM.
            // Complex/float16, many variables, or a non-contiguous Xc fall back to np.dot.
            // With a BLAS backend installed the product MUST go through np.dot: NumPy computes cov through
            // BLAS (gemm/syrk), so that is the only route that is byte-identical to it — the managed Gram
            // kernel is the no-backend fast path and its accumulation order differs by ~1 ULP.
            NDArray c = resultType == NPTypeCode.Complex || Xc.TensorEngine.Blas is not null ? null : TryGramCov(Xc, w);
            if (c is null)
            {
                NDArray Xt = w is null ? Xc.T : (Xc * w).T;
                // conjugation only matters (and only allocates) for complex data.
                c = resultType == NPTypeCode.Complex
                    ? Xc.TensorEngine.Dot(Xc, Xt.TensorEngine.Conjugate(Xt, null, null, null), null)
                    : Xc.TensorEngine.Dot(Xc, Xt, null);
            }

            // scale = true_divide(1, fact). NumPy replaces fact <= 0 with 0.0 (so 1/0 => +inf).
            NDArray scale = NewScalar(fact <= 0.0 ? double.PositiveInfinity : 1.0 / fact);

            NPTypeCode cType = c.typecode;
            c = c * scale;
            if (c.typecode != cType) // NumPy's in-place `c *= ...` preserves c's dtype.
                c = c.astype(cType);

            // NumPy's c.squeeze(), the ndarray method: a view of c — the view is yielded; c's tracked wrapper is
            // released at scope exit while ARC keeps the buffer alive through the view.
            return c.squeeze();
        }

        /// <summary>
        ///     NumPy's frequency- and analytic-weight checks and their product <c>w = fweights * aweights</c> (either may
        ///     be absent), in NumPy's statement order.
        /// </summary>
        /// <param name="fweights">The integer frequency weights, or null.</param>
        /// <param name="aweights">The analytic weights, or null.</param>
        /// <param name="nobs">The number of observations the weights must match.</param>
        /// <param name="aweightsCast">The analytic weights as float64 (null when absent); cov's normalization reads them.</param>
        /// <returns>The combined float64 weights, or null when neither is given.</returns>
        /// <exception cref="TypeError">A frequency weight is not integral ("fweights must be integer").</exception>
        /// <exception cref="RuntimeError">A weight array is multidimensional or does not have <paramref name="nobs"/>
        ///     elements.</exception>
        /// <exception cref="ValueError">A weight is negative.</exception>
        /// <remarks>
        ///     Shared by the numeric path and the text path (<see cref="CovOfText"/>): NumPy checks the weights before its
        ///     first reduction, so a bad weight wins over the refusal of text. The float64 temporaries belong to the
        ///     caller's <c>[NDScoped]</c> scope.
        /// </remarks>
        [NDScopedCovered] // only caller: CovFunction, under the [NDScoped] Cov override (casts and masks are reclaimed)
        private static NDArray CovWeights(NDArray fweights, NDArray aweights, long nobs, out NDArray aweightsCast)
        {
            NDArray w = null;
            aweightsCast = null;
            if (fweights is not null)
            {
                NDArray fw = fweights.astype(NPTypeCode.Double);
                NDArray rounded = fw.TensorEngine.Round(fw, decimals: 0, dtype: null, @out: null);
                NDArray integral = fw == rounded;
                if (!integral.TensorEngine.All(integral))
                    throw new TypeError("fweights must be integer");
                if (fw.ndim > 1)
                    throw new RuntimeError("cannot handle multidimensional fweights");
                if (fw.Shape.dimensions[0] != nobs)
                    throw new RuntimeError("incompatible numbers of samples and fweights");
                NDArray negative = fw < 0.0;
                if (negative.TensorEngine.Any(negative))
                    throw new ValueError("fweights cannot be negative");
                w = fw;
            }
            if (aweights is not null)
            {
                aweightsCast = aweights.astype(NPTypeCode.Double);
                if (aweightsCast.ndim > 1)
                    throw new RuntimeError("cannot handle multidimensional aweights");
                if (aweightsCast.Shape.dimensions[0] != nobs)
                    throw new RuntimeError("incompatible numbers of samples and aweights");
                NDArray negative = aweightsCast < 0.0;
                if (negative.TensorEngine.Any(negative))
                    throw new ValueError("aweights cannot be negative");
                w = w is null ? aweightsCast : w * aweightsCast;
            }

            return w;
        }

        /// <summary>
        ///     NumPy's covariance of no variables, <c>np.array([]).reshape(0, 0)</c>: a <c>(0, 0)</c> float64 array that is a
        ///     reshape VIEW of a fresh empty array (<c>flags.owndata</c> False), whatever the input dtype — text included.
        /// </summary>
        /// <returns>The empty covariance (it keeps its base's buffer alive; disposing the base inside an
        ///     <c>[NDScoped]</c> caller is safe).</returns>
        private NDArray EmptyCovariance()
            => new NDArray(Unmanaged.ArraySlice.FromArray(Array.Empty<double>(), true), Shape.Vector(0)) { TensorEngine = this }.reshape(0, 0);

        /// <summary>
        ///     <c>np.cov</c> over text (a string operand, or a text <c>dtype=</c> request: <c>S</c>, <c>U</c> or
        ///     <c>StringDType</c>), following NumPy's statements: with a NUMERIC <c>dtype=</c>,
        ///     <c>X = array(m, ndmin=2, dtype=dtype)</c> parses text <paramref name="m"/> (each element by <c>int()</c> /
        ///     <c>float()</c> / <c>complex()</c>), the empty-variables early return comes next, then
        ///     <c>array(y, dtype=dtype)</c> parses text <paramref name="y"/>, and the covariance of the numbers follows.
        ///     Otherwise the statements run up to the first reduction, each as NumPy writes it: the observations' descriptor
        ///     (<c>result_type(m, y, float64)</c> without <c>dtype=</c> — text absorbs float64 as 32 characters — else
        ///     <c>array(m, dtype=dtype)</c>'s), <c>X = array(m, ndmin=2, dtype=...)</c> (numbers formatted, text converted:
        ///     U → S encodes ASCII, S → U decodes it), the empty-variables early return, <c>y</c>'s conversion and the
        ///     concatenation, the weight checks, and then <c>average(X, axis=1, weights=w)</c>, which refuses the text:
        ///     for fixed-width text its <c>a.mean</c> with add.reduce's text, or with weights its weight sum with the SIZED
        ///     string descriptor <c>result_type</c> gives; for <c>StringDType</c> (requested: no operand reaches it
        ///     otherwise) <c>a.mean</c>'s add.reduce concatenates, so its <c>true_divide</c> by the count refuses (add.reduce
        ///     itself refuses an axis of no observations: no identity), and with weights the weighted mean's
        ///     <c>result_type(a.dtype, w.dtype)</c> has no common DType.
        /// </summary>
        /// <param name="m">The variables.</param>
        /// <param name="y">The additional variables, or null.</param>
        /// <param name="rowvar">Whether rows are variables.</param>
        /// <param name="bias">The <c>bias</c> argument (passed on to the numeric covariance of parsed text).</param>
        /// <param name="ddof">The <c>ddof</c> argument (passed on to the numeric covariance of parsed text).</param>
        /// <param name="fweights">The frequency weights, or null.</param>
        /// <param name="aweights">The analytic weights, or null.</param>
        /// <param name="dtype">The <c>dtype=</c> request, or null.</param>
        /// <returns>The covariance of the parsed numbers (a numeric <paramref name="dtype"/>), or the empty float64
        ///     <c>(0, 0)</c> covariance NumPy returns when there are no variables.</returns>
        /// <exception cref="ValueError">A text element that does not parse into a numeric <paramref name="dtype"/>
        ///     (<c>could not convert string to float: np.str_('ab')</c>; <paramref name="m"/>'s before <paramref name="y"/>'s).</exception>
        /// <exception cref="OverflowException">An integer text outside an integer <paramref name="dtype"/>.</exception>
        /// <exception cref="UnicodeEncodeError">A U → S conversion of non-ASCII text.</exception>
        /// <exception cref="UnicodeDecodeError">An S → U conversion of non-ASCII bytes.</exception>
        /// <exception cref="ValueError">The variables of <paramref name="m"/> and <paramref name="y"/> do not have the same
        ///     number of observations (concatenate's text), or a negative weight.</exception>
        /// <exception cref="TypeError">add.reduce's refusal of text, a weighted mean's class-only refusal, the mean's
        ///     division of <c>StringDType</c> text (<c>ufunc 'divide' did not contain a loop with signature matching types
        ///     (&lt;class 'numpy.dtypes.StringDType'&gt;, &lt;class 'numpy.dtypes.Int64DType'&gt;) -&gt; None</c>), or a
        ///     non-integral frequency weight.</exception>
        /// <exception cref="ValueError">A <c>StringDType</c> request over no observations (add.reduce's <c>zero-size array to
        ///     reduction operation add which has no identity</c>).</exception>
        /// <exception cref="DTypePromotionError">A <c>StringDType</c> request with weights (the weighted mean's
        ///     <c>result_type</c> of the text and the float64 weights), or <c>StringDType</c> operands without
        ///     <c>dtype=</c> (<c>result_type(m, y, float64)</c>).</exception>
        /// <exception cref="RuntimeError">A malformed weight array.</exception>
        /// <exception cref="InvalidOperationException">Not reached: np.average refuses every text array, as NumPy's mean
        ///     does.</exception>
        [NDScopedCovered] // only caller: CovFunction, under the [NDScoped] Cov override (temporaries are reclaimed)
        private NDArray CovOfText(NDArray m, NDArray y, bool rowvar, bool bias, int? ddof, NDArray fweights,
            NDArray aweights, DType dtype)
        {
            DType xd, yd;
            // A numeric request parses the text. A StringDType request is text like S and U (the descriptor it asks for,
            // which X and y become); read as numeric, its parse produced StringDType text again and cov recursed into this
            // branch until the stack overflowed.
            if (dtype is not null && !Backends.Strings.NDStringUfuncs.IsTextRequest(dtype))
            {
                // X = array(m, ndmin=2, dtype=dtype): np.array with a dtype casts unsafely, so text is parsed (a number is
                // left for the numeric path's own cast).
                var xm = Backends.Strings.NDStringUfuncs.IsText(m) ? m.astype(dtype, true, 'K', "unsafe") : m;
                long[] parsedDims = AtLeast2dDims(xm);
                if (!rowvar && xm.ndim != 1)
                    Array.Reverse(parsedDims);
                // `if X.shape[0] == 0: return np.array([]).reshape(0, 0)` — before y is converted at all.
                if (parsedDims[0] == 0)
                    return EmptyCovariance();
                // y = array(y, ndmin=2, dtype=dtype): the same parse for text y; then the covariance of the numbers.
                var ym = y is not null && Backends.Strings.NDStringUfuncs.IsText(y) ? y.astype(dtype, true, 'K', "unsafe") : y;
                return CovFunction(xm, ym, rowvar, bias, ddof, fweights, aweights, dtype);
            }
            if (dtype is null)
            {
                // dtype = result_type(m, y, float64): every array is strong, and fixed-width text absorbs float64 as 32
                // characters. StringDType has no common DType with float64: the SEQUENCE promotion refuses it with its own
                // text, naming the float64 dtype first (NumPy promotes the dtype arguments before the arrays).
                DType r;
                if (Backends.Strings.NDStringUfuncs.IsVString(m) || Backends.Strings.NDStringUfuncs.IsVString(y))
                    r = y is null ? DTypePromotion.ResultType(np.float64, m) : DTypePromotion.ResultType(np.float64, m, y);
                else
                    r = y is null ? DTypePromotion.PromoteTypes(m.dtype, np.float64)
                        : DTypePromotion.PromoteTypes(DTypePromotion.PromoteTypes(m.dtype, y.dtype), np.float64);
                xd = yd = r;
            }
            else if (dtype.itemsize != 0)
                xd = yd = dtype;
            else
            {
                // An unsized request keeps each operand's own text length (a number takes its REQUIRED_STR_LEN).
                xd = TextDescriptorFor(m.dtype, dtype);
                yd = y is null ? null : TextDescriptorFor(y.dtype, dtype);
            }

            // X = array(m, ndmin=2, dtype=dtype): numbers are formatted and text converted (np.array with a dtype casts
            // unsafely); only a text-to-text conversion can fail, and only on its content (U -> S encodes ASCII, S -> U
            // decodes it).
            NDArray X = NDArray.atleast_2d(m).astype(xd, true, 'K', "unsafe");
            if (!rowvar && m.ndim != 1)
                X = X.T;
            if (X.Shape.dimensions[0] == 0)
                return EmptyCovariance();

            if (y is not null)
            {
                // y = array(y, copy=None, ndmin=2, dtype=dtype), then X = np.concatenate((X, y), axis=0): the observation
                // counts must agree. NumPy's concatenate raises a ValueError there; NumSharp's raises its house
                // IncorrectShapeException with the same text, so the check is made here with NumPy's type.
                NDArray Y = NDArray.atleast_2d(y).astype(yd, true, 'K', "unsafe");
                if (!rowvar && Y.Shape.dimensions[0] != 1)
                    Y = Y.T;
                if (Y.Shape.dimensions[1] != X.Shape.dimensions[1])
                    throw new ValueError(
                        "all the input array dimensions except for the concatenation axis must match exactly, but along " +
                        $"dimension 1, the array at index 0 has size {X.Shape.dimensions[1]} and the array at index 1 has size {Y.Shape.dimensions[1]}");
                X = NDArray.concatenate(new[] { X, Y }, 0, null, null, "same_kind");
            }

            NDArray w = CovWeights(fweights, aweights, X.Shape.dimensions[1], out _);
            // avg, w_sum = average(X, axis=1, weights=w, returned=True): the mean refuses the text - add.reduce's text for
            // S / U, or, weighted, the weight sum's dtype= refusal of the sized string descriptor result_type gives; for
            // StringDType the division of the concatenated text by the count (add.reduce's "no identity" over no
            // observations), or, weighted, result_type's DTypePromotionError.
            X.TensorEngine.Average(X, new[] { 1 }, w, false, false);
            throw new InvalidOperationException("np.average accepted text, which NumPy's mean refuses.");
        }

        /// <summary>
        ///     The descriptor <c>np.array(x, dtype=&lt;unsized string&gt;)</c> gives an operand: text of the requested class
        ///     keeps its descriptor, anything else is cast to the class (its own length, or a number's REQUIRED_STR_LEN).
        /// </summary>
        /// <param name="from">The operand's descriptor.</param>
        /// <param name="request">The unsized string request.</param>
        /// <returns>The sized string descriptor.</returns>
        private static DType TextDescriptorFor(DType from, DType request)
            => from.Meta == request.Meta ? from : DTypePromotion.CastDescrToDType(from, request.Meta);

        /// <summary>The dimensions of <c>np.array(x, ndmin=2)</c>: ones prepended up to two dimensions.</summary>
        /// <param name="x">The operand.</param>
        /// <returns>A fresh dimension array of length max(ndim, 2).</returns>
        private static long[] AtLeast2dDims(NDArray x)
        {
            var dims = x.Shape.dimensions ?? Array.Empty<long>();
            var result = new long[Math.Max(dims.Length, 2)];
            int lead = result.Length - dims.Length;
            for (int i = 0; i < lead; i++)
                result[i] = 1;
            Array.Copy(dims, 0, result, lead, dims.Length);
            return result;
        }

        /// <summary>
        ///     Above this variable count the pairwise-Gram's row re-reads exceed cache and the
        ///     general GEMM's blocking wins, so cov keeps routing through <c>np.dot</c>. The
        ///     covariance shapes that matter (a handful of variables over many observations)
        ///     sit far below it; measured crossover for float64 at 100K observations is ~16–32.
        /// </summary>
        private const int GramCovMaxVars = 16;

        /// <summary>
        ///     Fast path for cov's <c>dot(Xc, Xc.T)</c> / <c>dot(Xc, (Xc*w).T)</c> — a
        ///     symmetric Gram matrix — via <see cref="DirectILKernelGenerator.GetGramKernel"/>.
        ///     Returns <c>null</c> (→ caller uses <c>np.dot</c>) unless every fast-path
        ///     condition holds: a real Single/Double dtype, a C-contiguous non-broadcast
        ///     <paramref name="Xc"/>, at least one observation, and no more than
        ///     <see cref="GramCovMaxVars"/> variables. For the weighted case the second Gram
        ///     operand is <c>Xc*w</c> (a fresh contiguous product), matching cov's
        ///     <c>(Xc*w).T</c>; the result is still symmetric, so only the upper triangle is
        ///     computed and mirrored — bit-identical to NumPy's independent per-entry dots.
        /// </summary>
        private unsafe NDArray TryGramCov(NDArray Xc, NDArray w)
        {
            NPTypeCode dt = Xc.typecode;
            if (dt != NPTypeCode.Double && dt != NPTypeCode.Single) return null;
            if (!Xc.Shape.IsContiguous || Xc.Shape.IsBroadcasted) return null;

            int m = (int)Xc.Shape.dimensions[0];
            long k = Xc.Shape.dimensions[1];
            if (m < 1 || m > GramCovMaxVars || k < 1) return null;

            var kernel = DirectILKernelGenerator.GetGramKernel(dt);
            if (kernel is null) return null;

            // Second operand: Xc itself (unweighted) or the weighted product Xc*w. The
            // product is a fresh C-contiguous array in the same dtype; bail if either fails
            // to hold so the kernel only ever sees two contiguous (m, k) buffers.
            NDArray b = Xc;
            if (w is not null)
            {
                b = Xc * w;
                if (b.typecode != dt) b = b.astype(dt);
                if (!b.Shape.IsContiguous || b.Shape.dimensions[0] != m || b.Shape.dimensions[1] != k) return null;
            }

            var g = NewArray(Xc.dtype, new Shape(m, m), false);
            int es = Xc.dtypesize;
            byte* ap = (byte*)Xc.Address + Xc.Shape.offset * es;
            byte* bp = (byte*)b.Address + b.Shape.offset * es;
            kernel((void*)ap, (void*)bp, (void*)g.Address, m, k);
            return g;
        }
    }
}
