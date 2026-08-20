using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Least squares polynomial fit — fit a polynomial <c>p[0]*x**deg + ... + p[deg]</c> of degree
        ///     <paramref name="deg"/> to points <c>(x, y)</c>, minimising the squared error.
        /// </summary>
        /// <param name="x">x-coordinates, shape <c>(M,)</c>.</param>
        /// <param name="y">y-coordinates, shape <c>(M,)</c> or <c>(M, K)</c> (one dataset per column).</param>
        /// <param name="deg">Degree of the fitting polynomial.</param>
        /// <param name="rcond">Relative condition number; default <c>len(x) * eps</c>.</param>
        /// <param name="full">When true, the returned <see cref="PolyfitResult"/> also carries the SVD diagnostics.</param>
        /// <param name="w">Optional weights, shape <c>(M,)</c>.</param>
        /// <param name="cov">
        ///     <c>null</c>/<c>false</c> (default), <c>true</c>, or the string <c>"unscaled"</c> — when truthy the
        ///     result also carries the covariance matrix (only meaningful when <paramref name="full"/> is false).
        /// </param>
        /// <returns>
        ///     A <see cref="PolyfitResult"/> that converts implicitly to the coefficient array; deconstruct it
        ///     for the <c>full</c> five-tuple or the <c>cov</c> two-tuple.
        /// </returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html</remarks>
        public static PolyfitResult polyfit(NDArray x, NDArray y, int deg, double? rcond = null,
            bool full = false, NDArray w = null, object cov = null)
        {
            int order = deg + 1;
            x = np.asarray(x) + 0.0;   // to floating (float32 stays float32; int/bool -> float64)
            y = np.asarray(y) + 0.0;

            if (deg < 0)
                throw new ValueError("expected deg >= 0");
            if (x.ndim != 1)
                throw new TypeError("expected 1D vector for x");
            if (x.size == 0)
                throw new TypeError("expected non-empty vector for x");
            if (y.ndim < 1 || y.ndim > 2)
                throw new TypeError("expected 1D or 2D array for y");
            if (x.shape[0] != y.shape[0])
                throw new TypeError("expected x and y to have same length");

            double rc = rcond ?? (x.size * np.finfo(x.dtype).eps);

            // Least-squares system for powers of x — the Vandermonde matrix.
            NDArray lhs = np.vander(x, order);
            NDArray rhs = y;

            if (w is not null)
            {
                w = np.asarray(w) + 0.0;
                if (w.ndim != 1)
                    throw new TypeError("expected a 1-d array for weights");
                if (w.shape[0] != y.shape[0])
                    throw new TypeError("expected w and y to have the same length");
                NDArray wcol = np.expand_dims(w, -1);
                lhs = lhs * wcol;
                rhs = rhs.ndim == 2 ? rhs * wcol : rhs * w;
            }

            // Scale the columns to improve the condition number, then solve and unscale.
            NDArray scale = np.sqrt(np.sum(lhs * lhs, 0));
            lhs = lhs / scale;
            var (c, resids, rank, s) = np.linalg.lstsq(lhs, rhs, rc);
            c = (c.T / scale).T;                  // broadcast the scale back into the coefficients

            // NumPy warns (RankWarning) when the fit is rank-deficient and !full — NumSharp omits the warning.

            if (full)
                return new PolyfitResult(c, resids, rank, s, rc, null);

            if (CovTruthy(cov))
            {
                NDArray vbase = np.linalg.inv(np.dot(lhs.T, lhs));
                vbase = vbase / np.outer(scale, scale);

                bool unscaled = cov is string cs && cs == "unscaled";
                if (!unscaled && x.size <= order)
                    throw new ValueError("the number of data points must exceed order "
                                         + "to scale the covariance matrix");

                NDArray fac = unscaled ? NDArray.Scalar(1.0) : resids / (double)(x.size - order);
                NDArray v = y.ndim == 1 ? vbase * fac : np.expand_dims(vbase, -1) * fac;
                return new PolyfitResult(c, resids, rank, s, rc, v);
            }

            return new PolyfitResult(c, resids, rank, s, rc, null);
        }

        /// <summary>NumPy's <c>elif cov:</c> truthiness — true, or a non-empty string other than nothing.</summary>
        private static bool CovTruthy(object cov)
        {
            switch (cov)
            {
                case null: return false;
                case bool b: return b;
                case string s: return s.Length != 0;
                default: return true;
            }
        }
    }

    /// <summary>
    ///     The polymorphic return of <see cref="np.polyfit"/>. Converts implicitly to the coefficient array
    ///     (the bare-return case); deconstructs to NumPy's <c>full</c> five-tuple
    ///     <c>(coeffs, residuals, rank, singular_values, rcond)</c> or the <c>cov</c> two-tuple
    ///     <c>(coeffs, covariance)</c>.
    /// </summary>
    public readonly struct PolyfitResult
    {
        /// <summary>Polynomial coefficients, highest power first — <c>(deg+1,)</c> or <c>(deg+1, K)</c>.</summary>
        public readonly NDArray coeffs;

        /// <summary>Sum of squared residuals of the least-squares fit (empty unless full-rank overdetermined).</summary>
        public readonly NDArray residuals;

        /// <summary>Effective rank of the scaled Vandermonde matrix.</summary>
        public readonly NDArray rank;

        /// <summary>Singular values of the scaled Vandermonde matrix.</summary>
        public readonly NDArray singular_values;

        /// <summary>The <c>rcond</c> value used.</summary>
        public readonly double rcond;

        /// <summary>The covariance matrix — <c>null</c> unless <c>cov</c> was requested.</summary>
        public readonly NDArray covariance;

        internal PolyfitResult(NDArray coeffs, NDArray residuals, NDArray rank,
            NDArray singularValues, double rcond, NDArray covariance)
        {
            this.coeffs = coeffs;
            this.residuals = residuals;
            this.rank = rank;
            this.singular_values = singularValues;
            this.rcond = rcond;
            this.covariance = covariance;
        }

        /// <summary>The bare return: the coefficient array.</summary>
        public static implicit operator NDArray(PolyfitResult r) => r.coeffs;

        /// <summary>NumPy's <c>full=True</c> return: <c>(coeffs, residuals, rank, singular_values, rcond)</c>.</summary>
        public void Deconstruct(out NDArray coeffs, out NDArray residuals, out NDArray rank,
            out NDArray singularValues, out double rcond)
        {
            coeffs = this.coeffs;
            residuals = this.residuals;
            rank = this.rank;
            singularValues = this.singular_values;
            rcond = this.rcond;
        }

        /// <summary>NumPy's <c>cov=True</c> return: <c>(coeffs, covariance)</c>.</summary>
        public void Deconstruct(out NDArray coeffs, out NDArray covariance)
        {
            coeffs = this.coeffs;
            covariance = this.covariance;
        }
    }
}
