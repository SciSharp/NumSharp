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
            int? ddof = null, NDArray fweights = null, NDArray aweights = null, NPTypeCode? dtype = null)
        {
            if (m is null) throw new ArgumentNullException(nameof(m));

            // --- Check inputs. NumPy's "ddof must be integer" ValueError is enforced by the int? type. ---
            if (m.ndim > 2)
                throw new ValueError("m has more than 2 dimensions");
            if (y is not null && y.ndim > 2)
                throw new ValueError("y has more than 2 dimensions");

            // Result dtype: at least float64 (complex input stays complex).
            NPTypeCode resultType = dtype ?? (y is null
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
            if (X.shape[0] == 0)
                return np.zeros(new Shape(0, 0), NPTypeCode.Double);

            if (y is not null)
            {
                NDArray Y = np.atleast_2d(y).astype(resultType);
                if (!rowvar && Y.shape[0] != 1)
                    Y = Y.T;
                X = np.concatenate((X, Y), 0);
            }

            int ddofValue = ddof ?? (bias ? 0 : 1);

            long nobs = X.shape[1];

            // --- Combined weights w = fweights * aweights (either may be absent). ---
            NDArray w = null;
            NDArray aweightsCast = null;
            if (fweights is not null)
            {
                NDArray fw = fweights.astype(NPTypeCode.Double);
                if (!np.all(fw == np.around(fw)))
                    throw new TypeError("fweights must be integer");
                if (fw.ndim > 1)
                    throw new RuntimeError("cannot handle multidimensional fweights");
                if (fw.shape[0] != nobs)
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
                if (aweightsCast.shape[0] != nobs)
                    throw new RuntimeError("incompatible numbers of samples and aweights");
                if (np.any(aweightsCast < 0.0))
                    throw new ValueError("aweights cannot be negative");
                w = w is null ? aweightsCast : w * aweightsCast;
            }

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
