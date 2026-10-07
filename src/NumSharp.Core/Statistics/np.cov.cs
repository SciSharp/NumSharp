using System;
using NumSharp.Backends.Kernels;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Estimate a covariance matrix, given data and weights.
        ///
        ///     Covariance indicates the level to which two variables vary together. Given
        ///     N-dimensional samples <c>X = [x_1, x_2, ..., x_N]^T</c>, the covariance matrix
        ///     element <c>C_ij</c> is the covariance of <c>x_i</c> and <c>x_j</c>; <c>C_ii</c> is
        ///     the variance of <c>x_i</c>.
        /// </summary>
        /// <param name="m">
        ///     A 1-D or 2-D array containing multiple variables and observations. Each row
        ///     represents a variable, each column a single observation of all those variables
        ///     (see <paramref name="rowvar"/>).
        /// </param>
        /// <param name="y">An additional set of variables and observations, same form as <paramref name="m"/>.</param>
        /// <param name="rowvar">
        ///     If true (default) each row is a variable with observations in the columns.
        ///     Otherwise the relationship is transposed: each column is a variable, rows are observations.
        /// </param>
        /// <param name="bias">
        ///     Default normalization (false) is by <c>(N - 1)</c> (unbiased). If true, normalization
        ///     is by <c>N</c>. Overridden by <paramref name="ddof"/>.
        /// </param>
        /// <param name="ddof">
        ///     If not null, overrides the default implied by <paramref name="bias"/>. <c>ddof=1</c>
        ///     returns the unbiased estimate even when weights are given; <c>ddof=0</c> returns the
        ///     simple average.
        /// </param>
        /// <param name="fweights">1-D array of integer frequency weights (number of times each observation is repeated).</param>
        /// <param name="aweights">1-D array of observation vector weights (relative importance of each observation).</param>
        /// <param name="dtype">Data-type of the result. By default the result has at least float64 precision.</param>
        /// <returns>The covariance matrix of the variables.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.cov.html</remarks>
        [NDScoped]
        public static NDArray cov(NDArray m, NDArray y = null, bool rowvar = true, bool bias = false,
            int? ddof = null, NDArray fweights = null, NDArray aweights = null, DType dtype = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(m, y, fweights);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            if (m is null) throw new ArgumentNullException(nameof(m));

            // --- Check inputs. NumPy's "ddof must be integer" ValueError is enforced by the int? type. ---
            if (m.ndim > 2)
                throw new ValueError("m has more than 2 dimensions");
            if (y is not null && y.ndim > 2)
                throw new ValueError("y has more than 2 dimensions");

            // Text (an operand, or a string dtype= request): NumPy's statements run up to the first reduction, which
            // refuses text — or the empty-variables early return, which yields the (0, 0) float64 result.
            if (Backends.Strings.NDStringUfuncs.IsString(m) || Backends.Strings.NDStringUfuncs.IsString(y) ||
                Backends.Strings.NDStringUfuncs.IsStringRequest(dtype))
                return CovOfText(m, y, rowvar, fweights, aweights, dtype);

            // Result dtype: at least float64 (complex input stays complex).
            NPTypeCode resultType = dtype?.GetTypeCode() ?? (y is null
                ? np.result_type(m, NPTypeCode.Double)
                : np.result_type(m, y, NPTypeCode.Double));

            // Every intermediate below — atleast_2d/astype/T/concatenate, the weight casts and
            // validation masks, avg and its column view, Xc/Xt and the pre-scale c — is scope-
            // tracked and reclaimed at method exit (throw paths included); only the yielded
            // result survives. Inputs were constructed before the scope opened: never touched.

            int mNdim = m.ndim;

            // X = array(m, ndmin=2, dtype=dtype): a fresh 2-D array in the result dtype.
            NDArray X = np.atleast_2d(m).astype(resultType);
            if (!rowvar && mNdim != 1)
                X = X.T;

            // Empty variables => (0, 0) float64, regardless of input dtype (NumPy: np.array([]).reshape(0, 0)).
            if (X.Shape.dimensions[0] == 0)
                return EmptyCovariance();

            if (y is not null)
            {
                NDArray Y = np.atleast_2d(y).astype(resultType);
                if (!rowvar && Y.Shape.dimensions[0] != 1)
                    Y = Y.T;
                X = np.concatenate((X, Y), 0);
            }

            int ddofValue = ddof ?? (bias ? 0 : 1);

            long nobs = X.Shape.dimensions[1];

            // --- Combined weights w = fweights * aweights (either may be absent). ---
            NDArray w = CovWeights(fweights, aweights, nobs, out NDArray aweightsCast);

            // --- Weighted mean along the observation axis. ---
            NDArray avg = np.average(X, 1, w);

            // Real weight sum (NumPy's w_sum[0]; its imaginary part, present only for complex input,
            // is always zero). `fact` is mathematically real, so it is carried in a double.
            double wSum = w is null ? nobs : np.sum(w).GetDouble(0);

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
                double numer = ddofValue * np.sum(w * aweightsCast).GetDouble(0); // ddof * sum(w*aweights)
                // NumPy computes `.../ w_sum`. For complex input w_sum is complex128, so NumPy divides
                // via Smith's method as numer*(1/w_sum) — a reciprocal-multiply whose real part rounds
                // 1 ULP differently from real division. It only matters when fact lands ~0 (degenerate
                // degrees of freedom), but there it decides finite-vs-inf, so match it.
                double ratio = resultType == NPTypeCode.Complex ? numer * (1.0 / wSum) : numer / wSum;
                fact = wSum - ratio;
            }

            // --- Center the observations, weight them, and form the covariance via one GEMM. ---
            NDArray Xc = X - avg.reshape(avg.size, 1);
            // NumPy does `X -= avg[:, None]` in place, so X keeps its dtype even if avg was promoted.
            if (Xc.typecode != X.typecode)
                Xc = Xc.astype(X.typecode);

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
                    ? np.dot(Xc, np.conjugate(Xt))
                    : np.dot(Xc, Xt);
            }

            // scale = true_divide(1, fact). NumPy replaces fact <= 0 with 0.0 (so 1/0 => +inf).
            NDArray scale = NDArray.Scalar(fact <= 0.0 ? double.PositiveInfinity : 1.0 / fact);

            NPTypeCode cType = c.typecode;
            c = c * scale;
            if (c.typecode != cType) // NumPy's in-place `c *= ...` preserves c's dtype.
                c = c.astype(cType);

            // squeeze returns a view of c — the view is yielded; c's tracked wrapper is released
            // at scope exit while ARC keeps the buffer alive through the view.
            return np.squeeze(c);
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
        [NDScopedCovered] // only callers: the [NDScoped] cov (the casts and validation masks are scope-reclaimed)
        private static NDArray CovWeights(NDArray fweights, NDArray aweights, long nobs, out NDArray aweightsCast)
        {
            NDArray w = null;
            aweightsCast = null;
            if (fweights is not null)
            {
                NDArray fw = fweights.astype(NPTypeCode.Double);
                if (!np.all(fw == np.around(fw)))
                    throw new TypeError("fweights must be integer");
                if (fw.ndim > 1)
                    throw new RuntimeError("cannot handle multidimensional fweights");
                if (fw.Shape.dimensions[0] != nobs)
                    throw new RuntimeError("incompatible numbers of samples and fweights");
                if (np.any(fw < 0.0))
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
                if (np.any(aweightsCast < 0.0))
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
        private static NDArray EmptyCovariance() => np.array(Array.Empty<double>()).reshape(0, 0);

        /// <summary>
        ///     <c>np.cov</c> over text (a string operand, or a string <c>dtype=</c> request), following NumPy's statements
        ///     up to the first reduction: the observations' descriptor (<c>result_type(m, y, float64)</c> without
        ///     <c>dtype=</c> — text absorbs float64 as 32 characters — else <c>array(m, dtype=dtype)</c>'s), the
        ///     text-to-text conversion (which can fail: U → S encodes ASCII, S → U decodes it), the empty-variables early
        ///     return, <c>y</c>'s conversion and concatenation, the weight checks, and then <c>average</c>: its
        ///     <c>a.mean</c> refuses text with add.reduce's text, or with weights its weight sum refuses the SIZED string
        ///     descriptor by class.
        /// </summary>
        /// <param name="m">The variables.</param>
        /// <param name="y">The additional variables, or null.</param>
        /// <param name="rowvar">Whether rows are variables.</param>
        /// <param name="fweights">The frequency weights, or null.</param>
        /// <param name="aweights">The analytic weights, or null.</param>
        /// <param name="dtype">The <c>dtype=</c> request, or null.</param>
        /// <returns>The empty float64 <c>(0, 0)</c> covariance NumPy returns when there are no variables.</returns>
        /// <exception cref="StringComputationNotSupportedException">A numeric <paramref name="dtype"/> with a text operand:
        ///     NumPy parses the text as numbers (the next phase).</exception>
        /// <exception cref="UnicodeEncodeError">A U → S conversion of non-ASCII text.</exception>
        /// <exception cref="UnicodeDecodeError">An S → U conversion of non-ASCII bytes.</exception>
        /// <exception cref="ValueError">The variables of <paramref name="m"/> and <paramref name="y"/> do not have the same
        ///     number of observations (concatenate's text), or a negative weight.</exception>
        /// <exception cref="TypeError">add.reduce's refusal of text, a weighted mean's class-only refusal, or a non-integral
        ///     frequency weight.</exception>
        /// <exception cref="RuntimeError">A malformed weight array.</exception>
        [NDScopedCovered] // only caller: the [NDScoped] cov (the conversion temporaries are scope-reclaimed)
        private static NDArray CovOfText(NDArray m, NDArray y, bool rowvar, NDArray fweights, NDArray aweights, DType dtype)
        {
            DType xd, yd;
            if (dtype is null)
            {
                // dtype = result_type(m, y, float64): every array is strong, and text absorbs float64 as 32 characters.
                var r = y is null ? DTypePromotion.PromoteTypes(m.dtype, np.float64)
                    : DTypePromotion.PromoteTypes(DTypePromotion.PromoteTypes(m.dtype, y.dtype), np.float64);
                xd = yd = r;
            }
            else if (!Backends.Strings.NDStringUfuncs.IsStringRequest(dtype))
                // array(text, dtype=float64) parses the text as numbers: the computational phase.
                throw Backends.Strings.NDStringUfuncs.NextPhase("cov", Backends.Strings.NDStringUfuncs.IsString(m) ? m : y);
            else if (dtype.itemsize != 0)
                xd = yd = dtype;
            else
            {
                // An unsized request keeps each operand's own text length (a number takes its REQUIRED_STR_LEN).
                xd = TextDescriptorFor(m.dtype, dtype);
                yd = y is null ? null : TextDescriptorFor(y.dtype, dtype);
            }

            // X = array(m, ndmin=2, dtype=dtype): only a text-to-text conversion can fail, and only on its content.
            ConvertTextForCov(m, xd);
            long[] xDims = AtLeast2dDims(m);
            if (!rowvar && m.ndim != 1)
                Array.Reverse(xDims);
            if (xDims[0] == 0)
                return EmptyCovariance();

            DType observed = xd;
            if (y is not null)
            {
                ConvertTextForCov(y, yd);
                long[] yDims = AtLeast2dDims(y);
                if (!rowvar && yDims[0] != 1)
                    Array.Reverse(yDims);
                // X = np.concatenate((X, y), axis=0): the observation counts must agree.
                if (yDims[1] != xDims[1])
                    throw new ValueError(
                        "all the input array dimensions except for the concatenation axis must match exactly, but along " +
                        $"dimension 1, the array at index 0 has size {xDims[1]} and the array at index 1 has size {yDims[1]}");
                observed = DTypePromotion.PromoteTypes(xd, yd);
            }

            NDArray w = CovWeights(fweights, aweights, xDims[1], out _);
            if (w is not null)
                throw new TypeError(Backends.DefaultEngine.GeneralDTypeOnlyText);
            throw Backends.Strings.NDStringUfuncs.AddNotCompatible(observed, "reduce");
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

        /// <summary>
        ///     Runs the text-to-text conversion <c>np.array(x, dtype=descr)</c> makes, for its errors only: U → S encodes
        ///     ASCII and S → U decodes it, so non-ASCII content raises NumPy's Unicode error. A number converted to text
        ///     cannot fail (and is the computational phase), so it is skipped, as is a conversion within one class.
        /// </summary>
        /// <param name="x">The operand.</param>
        /// <param name="descr">The descriptor it is converted to.</param>
        /// <exception cref="UnicodeEncodeError">A U → S conversion of non-ASCII text.</exception>
        /// <exception cref="UnicodeDecodeError">An S → U conversion of non-ASCII bytes.</exception>
        [NDScopedCovered] // only caller: CovOfText (the converted copy is scope-reclaimed)
        private static void ConvertTextForCov(NDArray x, DType descr)
        {
            if (Backends.Strings.NDStringUfuncs.IsString(x) && descr.Meta is StringDTypeMeta && descr.Meta != x.dtype.Meta)
                x.astype(descr);
        }

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
        private static unsafe NDArray TryGramCov(NDArray Xc, NDArray w)
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

            var g = new NDArray(Xc.dtype, new Shape(m, m), false);
            int es = Xc.dtypesize;
            byte* ap = (byte*)Xc.Address + Xc.Shape.offset * es;
            byte* bp = (byte*)b.Address + b.Shape.offset * es;
            kernel((void*)ap, (void*)bp, (void*)g.Address, m, k);
            return g;
        }
    }
}
