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
        ///     for the <c>full</c> five-tuple or the <c>cov</c> two-tuple. It carries exactly the arrays NumPy's
        ///     return holds: the SVD diagnostics (<c>residuals</c>/<c>rank</c>/<c>singular_values</c>) only when
        ///     <paramref name="full"/> is true, the covariance only when <paramref name="cov"/> is truthy — every
        ///     other slot is <c>null</c>.
        /// </returns>
        /// <exception cref="ValueError">A negative <paramref name="deg"/>, or a scaled covariance with no more points than
        ///     coefficients.</exception>
        /// <exception cref="TypeError">NumPy's shape refusals (x not 1-D or empty, y not 1-D / 2-D, lengths that differ, w
        ///     not 1-D or of another length), text the float conversion refuses, or a float16 / decimal x numpy.linalg
        ///     refuses.</exception>
        /// <exception cref="NotSupportedException"><paramref name="x"/> or <paramref name="y"/> is null.</exception>
        /// <exception cref="OpenBlasMissingBackendException">No LAPACK backend is installed (lstsq needs one).</exception>
        /// <exception cref="LinAlgError">lstsq's SVD does not converge, or the covariance's inverse meets a singular
        ///     matrix.</exception>
        /// <exception cref="IncorrectShapeException">Weights that do not broadcast against a 2-D y's columns.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html
        ///     <para>
        ///     The lstsq diagnostics a non-<c>full</c> call does not return are released with the method's other
        ///     temporaries. They used to ride along in the result regardless of <paramref name="full"/>, so the
        ///     ordinary spelling <c>NDArray c = np.polyfit(x, y, deg)</c> — which keeps only the coefficients —
        ///     stranded the residual and singular-value buffers (2 pooled arrays per call) until a GC.
        ///     </para>
        ///     <para>The statements are <see cref="TensorEngine.Polyfit"/>, on <paramref name="x"/>'s engine (the default one
        ///     for null), inside the matrix boundary's plain region.</para>
        /// </remarks>
        public static PolyfitResult polyfit(NDArray x, NDArray y, int deg, double? rcond = null,
            bool full = false, NDArray w = null, object cov = null)
        {
            if (x is null) throw new ArgumentNullException(nameof(x));
            using var plain = NDMatrixBoundary.PlainFor(x, y, w);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return x.TensorEngine.Polyfit(x, y, deg, rcond, full, w, cov);
        }
    }

    /// <summary>
    ///     The polymorphic return of <see cref="np.polyfit"/>. Converts implicitly to the coefficient array
    ///     (the bare-return case); deconstructs to NumPy's <c>full</c> five-tuple
    ///     <c>(coeffs, residuals, rank, singular_values, rcond)</c> or the <c>cov</c> two-tuple
    ///     <c>(coeffs, covariance)</c>.
    /// </summary>
    public readonly struct PolyfitResult : INDArrayCarrier
    {
        /// <summary>Polynomial coefficients, highest power first — <c>(deg+1,)</c> or <c>(deg+1, K)</c>.</summary>
        public readonly NDArray coeffs;

        /// <summary>Sum of squared residuals of the least-squares fit (empty unless full-rank overdetermined);
        /// <c>null</c> unless <c>full</c> was requested (NumPy returns it only then).</summary>
        public readonly NDArray residuals;

        /// <summary>Effective rank of the scaled Vandermonde matrix; <c>null</c> unless <c>full</c> was requested.</summary>
        public readonly NDArray rank;

        /// <summary>Singular values of the scaled Vandermonde matrix; <c>null</c> unless <c>full</c> was requested.</summary>
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

        void INDArrayCarrier.YieldTo(NDScope scope)
        {
            scope.Returns(coeffs);
            scope.Returns(residuals);
            scope.Returns(rank);
            scope.Returns(singular_values);
            scope.Returns(covariance);
        }
    }
}
