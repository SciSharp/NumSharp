using System;

namespace NumSharp
{
    /// <summary>
    ///     The legacy <c>RandomState</c> samplers — a line-by-line port of
    ///     <c>numpy/random/src/legacy/legacy-distributions.c</c>, NumPy's frozen copy of the NumPy 1.16 algorithms that
    ///     keeps every seeded <c>RandomState</c> stream stable across releases.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     These are instance members because the C functions take an <c>aug_bitgen_t</c> — the bit generator AUGMENTED
    ///     with <c>RandomState</c>'s cached Gaussian (<see cref="NextGaussian"/>) — and the binomial samplers additionally
    ///     take <c>RandomState</c>'s <see cref="_binomial"/> cache. Every expression keeps NumPy's operand order and every
    ///     draw is taken in NumPy's order (see <see cref="Distributions"/> for the byte-parity rules); several legacy formulas
    ///     are deliberately less precise than their <see cref="Generator"/> successors (<c>exp(x) - 1</c> rather than
    ///     <c>expm1</c>, <c>log(1 - p)</c> rather than <c>log1p</c>) and must stay that way to reproduce the stream.
    ///     </para>
    ///     <para>
    ///     Two NumPy inputs never return — <c>zipf(a)</c> for <c>a &gt;= 1025</c> and <c>vonmises(mu, kappa)</c> once
    ///     <c>4*kappa^2</c> overflows (including <c>kappa = inf</c>) — because the rejection test compares NaNs. The ports
    ///     detect exactly those inputs and return the distribution's limit (1, and <c>mu</c> wrapped to <c>[-pi, pi]</c>)
    ///     without drawing, instead of hanging. Callers hold the bit generator's lock.
    ///     </para>
    /// </remarks>
    public partial class NumPyRandom
    {
        /// <summary>
        ///     <c>RandomState</c>'s binomial setup cache (NumPy's <c>self._binomial</c>), shared by <c>binomial</c> and
        ///     <c>multinomial</c> and never reset — not by re-seeding, not by <c>set_state</c> (see <see cref="BinomialState"/>).
        /// </summary>
        private readonly BinomialState _binomial = new BinomialState();

        /// <summary>Whether <paramref name="size"/> is NumPy's <c>size=None</c> (or <c>()</c>): draw one value, return it 0-d.</summary>
        /// <param name="size">The requested output shape; <c>default</c> stands for <c>None</c>.</param>
        /// <returns>True when a single scalar draw is requested.</returns>
        private static bool IsScalarDraw(Shape size) => size.IsEmpty || size.IsScalar;

        /// <summary>
        ///     Allocates a sampler's output — NumPy's <c>np.empty(size, dtype)</c> — as a fresh C-contiguous array of
        ///     <paramref name="size"/>'s DIMENSIONS, uninitialized (every element is written by the fill that follows).
        /// </summary>
        /// <param name="typeCode">The output dtype.</param>
        /// <param name="size">The requested shape.</param>
        /// <returns>The uninitialized output.</returns>
        /// <remarks>
        ///     Rebuilding the shape from its dimensions matters: a <paramref name="size"/> taken from a view carries that view's
        ///     strides and offset, and allocating <c>size</c> elements under those strides would write past the buffer.
        /// </remarks>
        /// <exception cref="ValueError">A dimension is negative (<c>negative dimensions are not allowed</c>).</exception>
        private static NDArray LegacyOutput(NPTypeCode typeCode, Shape size)
            => new NDArray(typeCode, new Shape(size.dimensions), false);

        /// <summary>NumPy's <c>legacy_standard_exponential</c>: <c>-log(1 - U)</c> (U = 0 gives <c>-0.0</c>; no redraw).</summary>
        /// <returns>The draw.</returns>
        private double LegacyStandardExponential() => -Math.Log(1.0 - randomizer.NextDouble());

        /// <summary>NumPy's <c>legacy_exponential</c>: <c>scale * legacy_standard_exponential</c>.</summary>
        /// <param name="scale">The scale (validated by the caller).</param>
        /// <returns>The draw.</returns>
        private double LegacyExponential(double scale) => scale * LegacyStandardExponential();

        /// <summary>
        ///     NumPy's <c>legacy_standard_gamma</c>: exponential for <c>shape == 1</c>, 0 (no draw) for <c>shape == 0</c>,
        ///     Johnk/Ahrens-Dieter rejection for <c>shape &lt; 1</c>, and Marsaglia-Tsang (on the cached-Gaussian polar
        ///     normals) above.
        /// </summary>
        /// <param name="shape">The shape (validated non-negative by the caller; NaN takes the Marsaglia branch and yields NaN).</param>
        /// <returns>The draw.</returns>
        /// <remarks>
        ///     The Marsaglia constant is <c>c = 1 / sqrt(9 * b)</c>, not the algebraically equal <c>(1/3) / sqrt(b)</c> — the two
        ///     round differently, and the old NumSharp spelling left chisquare/wald/dirichlet streams a few ULP off.
        /// </remarks>
        private double LegacyStandardGamma(double shape)
        {
            double b, c;
            double U, V, X, Y;

            if (shape == 1.0)
            {
                return LegacyStandardExponential();
            }
            else if (shape == 0.0)
            {
                return 0.0;
            }
            else if (shape < 1.0)
            {
                for (;;)
                {
                    U = randomizer.NextDouble();
                    V = LegacyStandardExponential();
                    if (U <= 1.0 - shape)
                    {
                        X = Math.Pow(U, 1.0 / shape);
                        if (X <= V)
                            return X;
                    }
                    else
                    {
                        Y = -Math.Log((1 - U) / shape);
                        X = Math.Pow(1.0 - shape + shape * Y, 1.0 / shape);
                        if (X <= (V + Y))
                            return X;
                    }
                }
            }
            else
            {
                b = shape - 1.0 / 3.0;
                c = 1.0 / Math.Sqrt(9 * b);
                for (;;)
                {
                    do
                    {
                        X = NextGaussian();
                        V = 1.0 + c * X;
                    } while (V <= 0.0);

                    V = V * V * V;
                    U = randomizer.NextDouble();
                    if (U < 1.0 - 0.0331 * (X * X) * (X * X))
                        return (b * V);
                    if (Math.Log(U) < 0.5 * X * X + b * (1.0 - V + Math.Log(V)))
                        return (b * V);
                }
            }
        }

        /// <summary>NumPy's <c>legacy_gamma</c>: <c>scale * legacy_standard_gamma(shape)</c>.</summary>
        /// <param name="shape">The shape.</param>
        /// <param name="scale">The scale.</param>
        /// <returns>The draw.</returns>
        private double LegacyGamma(double shape, double scale) => scale * LegacyStandardGamma(shape);

        /// <summary>NumPy's <c>legacy_pareto</c>: <c>exp(E / a) - 1</c> (NOT <c>expm1</c> — the legacy rounding).</summary>
        /// <param name="a">The shape.</param>
        /// <returns>The draw.</returns>
        private double LegacyPareto(double a) => Math.Exp(LegacyStandardExponential() / a) - 1;

        /// <summary>NumPy's <c>legacy_weibull</c>: 0 with NO draw for <c>a == 0</c>, else <c>E^(1/a)</c>.</summary>
        /// <param name="a">The shape.</param>
        /// <returns>The draw.</returns>
        private double LegacyWeibull(double a)
        {
            if (a == 0.0)
                return 0.0;
            return Math.Pow(LegacyStandardExponential(), 1.0 / a);
        }

        /// <summary>NumPy's <c>legacy_power</c>: <c>(1 - exp(-E))^(1/a)</c> (NOT <c>-expm1</c>).</summary>
        /// <param name="a">The exponent parameter.</param>
        /// <returns>The draw.</returns>
        private double LegacyPower(double a) => Math.Pow(1 - Math.Exp(-LegacyStandardExponential()), 1.0 / a);

        /// <summary>NumPy's <c>legacy_chisquare</c>: <c>2 * legacy_standard_gamma(df / 2)</c>.</summary>
        /// <param name="df">The degrees of freedom.</param>
        /// <returns>The draw.</returns>
        private double LegacyChisquare(double df) => 2.0 * LegacyStandardGamma(df / 2.0);

        /// <summary>
        ///     NumPy's <c>legacy_rayleigh</c>: <c>mode * sqrt(-2 * log1p(-U))</c> — it takes the bare bit generator, so it
        ///     neither reads nor fills the Gaussian cache.
        /// </summary>
        /// <param name="mode">The scale.</param>
        /// <returns>The draw.</returns>
        /// <remarks><see cref="Generator.Log1p"/> reproduces the CRT's <c>log1p</c> bit for bit (NumPy's <c>npy_log1p</c>).</remarks>
        private double LegacyRayleigh(double mode) => mode * Math.Sqrt(-2.0 * Generator.Log1p(-randomizer.NextDouble()));

        /// <summary>
        ///     NumPy's <c>legacy_noncentral_chisquare</c>: plain chi-square for <c>nonc == 0</c>; for <c>df &gt; 1</c> a
        ///     <c>df - 1</c> chi-square plus a shifted squared normal; otherwise a Poisson-mixed chi-square — whose NaN guard
        ///     sits AFTER the draws so a NaN <paramref name="nonc"/> still advances the stream as NumPy's does.
        /// </summary>
        /// <param name="df">The degrees of freedom.</param>
        /// <param name="nonc">The non-centrality.</param>
        /// <returns>The draw.</returns>
        private double LegacyNoncentralChisquare(double df, double nonc)
        {
            double @out;
            if (nonc == 0)
                return LegacyChisquare(df);
            if (1 < df)
            {
                double Chi2 = LegacyChisquare(df - 1);
                double n = NextGaussian() + Math.Sqrt(nonc);
                return Chi2 + n * n;
            }
            else
            {
                long i = Distributions.RandomPoisson(randomizer, nonc / 2.0);
                @out = LegacyChisquare(df + 2 * i);
                /* Insert nan guard here to avoid changing the stream */
                if (double.IsNaN(nonc))
                    return Distributions.NPY_NAN;
                return @out;
            }
        }

        /// <summary>NumPy's <c>legacy_noncentral_f</c>: <c>(ncchi2(dfnum, nonc) * dfden) / (chi2(dfden) * dfnum)</c>.</summary>
        /// <param name="dfnum">The numerator degrees of freedom.</param>
        /// <param name="dfden">The denominator degrees of freedom.</param>
        /// <param name="nonc">The non-centrality.</param>
        /// <returns>The draw.</returns>
        private double LegacyNoncentralF(double dfnum, double dfden, double nonc)
        {
            double t = LegacyNoncentralChisquare(dfnum, nonc) * dfden;
            return t / (LegacyChisquare(dfden) * dfnum);
        }

        /// <summary>
        ///     NumPy's <c>legacy_wald</c>: Michael/Schucany/Haas with the legacy <c>mean + mu_2l * (Y - sqrt(4*scale*Y + Y*Y))</c>
        ///     spelling (the Generator rewrote it to avoid cancellation; the legacy form must keep it).
        /// </summary>
        /// <param name="mean">The mean.</param>
        /// <param name="scale">The scale.</param>
        /// <returns>The draw.</returns>
        private double LegacyWald(double mean, double scale)
        {
            double U, X, Y;
            double mu_2l;

            mu_2l = mean / (2 * scale);
            Y = NextGaussian();
            Y = mean * Y * Y;
            X = mean + mu_2l * (Y - Math.Sqrt(4 * scale * Y + Y * Y));
            U = randomizer.NextDouble();
            if (U <= mean / (mean + X))
                return X;
            else
                return mean * mean / X;
        }

        /// <summary>NumPy's <c>legacy_normal</c>: <c>loc + scale * legacy_gauss</c>.</summary>
        /// <param name="loc">The mean.</param>
        /// <param name="scale">The standard deviation.</param>
        /// <returns>The draw.</returns>
        private double LegacyNormal(double loc, double scale) => loc + scale * NextGaussian();

        /// <summary>NumPy's <c>legacy_lognormal</c>: <c>exp(legacy_normal(mean, sigma))</c>.</summary>
        /// <param name="mean">The underlying normal's mean.</param>
        /// <param name="sigma">The underlying normal's standard deviation.</param>
        /// <returns>The draw.</returns>
        private double LegacyLognormal(double mean, double sigma) => Math.Exp(LegacyNormal(mean, sigma));

        /// <summary>NumPy's <c>legacy_standard_t</c>: <c>sqrt(df/2) * N / sqrt(G(df/2))</c>, the normal drawn first.</summary>
        /// <param name="df">The degrees of freedom.</param>
        /// <returns>The draw.</returns>
        private double LegacyStandardT(double df)
        {
            double num, denom;

            num = NextGaussian();
            denom = LegacyStandardGamma(df / 2);
            return Math.Sqrt(df / 2) * num / Math.Sqrt(denom);
        }

        /// <summary>
        ///     NumPy's <c>legacy_negative_binomial</c>: a Poisson whose mean is a legacy gamma draw with scale
        ///     <c>(1 - p) / p</c>.
        /// </summary>
        /// <param name="n">The number of successes.</param>
        /// <param name="p">The success probability.</param>
        /// <returns>The count of failures.</returns>
        /// <remarks>
        ///     <c>p == 0</c> makes the mean infinite; PTRS then converts an infinite/NaN floor through C's integer cast, which
        ///     yields <see cref="long.MinValue"/> here (C <c>long</c> is 64-bit in NumSharp's legacy model; NumPy's win-amd64
        ///     build, with a 32-bit <c>long</c>, returns <c>-2147483648</c>).
        /// </remarks>
        private long LegacyNegativeBinomial(double n, double p)
        {
            double Y = LegacyGamma(n, (1 - p) / p);
            return Distributions.RandomPoisson(randomizer, Y);
        }

        /// <summary>
        ///     NumPy's <c>legacy_standard_cauchy</c>: the ratio of two polar normals, numerator drawn FIRST (MSVC and GCC both
        ///     evaluate the C division's operands left to right here, which NumPy's own expected values confirm).
        /// </summary>
        /// <returns>The draw.</returns>
        private double LegacyStandardCauchy()
        {
            double num = NextGaussian();
            return num / NextGaussian();
        }

        /// <summary>
        ///     NumPy's <c>legacy_beta</c>: Johnk's algorithm when both shapes are <c>&lt;= 1</c> (with the log-space fallback
        ///     when both powers underflow), else the ratio of two legacy gammas.
        /// </summary>
        /// <param name="a">Alpha.</param>
        /// <param name="b">Beta.</param>
        /// <returns>The draw.</returns>
        private double LegacyBeta(double a, double b)
        {
            double Ga, Gb;

            if ((a <= 1.0) && (b <= 1.0))
            {
                double U, V, X, Y;
                /* Use Johnk's algorithm */

                while (true)
                {
                    U = randomizer.NextDouble();
                    V = randomizer.NextDouble();
                    X = Math.Pow(U, 1.0 / a);
                    Y = Math.Pow(V, 1.0 / b);

                    if ((X + Y) <= 1.0)
                    {
                        if (X + Y > 0)
                        {
                            return X / (X + Y);
                        }
                        else
                        {
                            double logX = Math.Log(U) / a;
                            double logY = Math.Log(V) / b;
                            double logM = logX > logY ? logX : logY;
                            logX -= logM;
                            logY -= logM;

                            return Math.Exp(logX - Math.Log(Math.Exp(logX) + Math.Exp(logY)));
                        }
                    }
                }
            }
            else
            {
                Ga = LegacyStandardGamma(a);
                Gb = LegacyStandardGamma(b);
                return Ga / (Ga + Gb);
            }
        }

        /// <summary>
        ///     NumPy's <c>legacy_f</c>: <c>(chi2(dfnum) * dfden) / (chi2(dfden) * dfnum)</c>, the numerator chi-square drawn
        ///     first.
        /// </summary>
        /// <param name="dfnum">The numerator degrees of freedom.</param>
        /// <param name="dfden">The denominator degrees of freedom.</param>
        /// <returns>The draw.</returns>
        private double LegacyF(double dfnum, double dfden)
        {
            double num = LegacyChisquare(dfnum) * dfden;
            return num / (LegacyChisquare(dfden) * dfnum);
        }

        /// <summary>
        ///     NumPy's <c>legacy_random_binomial_inversion</c>: sequential CDF search with <c>q^n = exp(n * log(q))</c> — the
        ///     legacy spelling (the modern sampler uses <c>log1p(-p)</c>), cached in <see cref="_binomial"/>.
        /// </summary>
        /// <param name="n">The trial count.</param>
        /// <param name="p">The success probability (<c>&lt;= 0.5</c>).</param>
        /// <returns>The count of successes.</returns>
        private long LegacyBinomialInversion(long n, double p)
        {
            var binomial = _binomial;
            double q, qn, np, px, U;
            long X, bound;

            if (!binomial.has_binomial || (binomial.nsave != n) || (binomial.psave != p))
            {
                binomial.nsave = n;
                binomial.psave = p;
                binomial.has_binomial = true;
                binomial.q = q = 1.0 - p;
                binomial.r = qn = Math.Exp(n * Math.Log(q));
                binomial.c = np = n * p;
                // C's MIN macro — `n < y ? n : y` in double — then the (RAND_INT_TYPE) cast.
                double limit = np + 10.0 * Math.Sqrt(np * q + 1);
                binomial.m = bound = Distributions.ToInt64(n < limit ? n : limit);
            }
            else
            {
                q = binomial.q;
                qn = binomial.r;
                np = binomial.c;
                bound = binomial.m;
            }
            X = 0;
            px = qn;
            U = randomizer.NextDouble();
            while (U > px)
            {
                X++;
                if (X > bound)
                {
                    X = 0;
                    px = qn;
                    U = randomizer.NextDouble();
                }
                else
                {
                    U -= px;
                    px = ((n - X + 1) * p * px) / (X * q);
                }
            }
            return X;
        }

        /// <summary>
        ///     NumPy's <c>legacy_random_binomial</c> (<c>legacy_random_binomial_original</c>): the legacy inversion for
        ///     <c>n*min(p,1-p) &lt;= 30</c>, BTPE above, mirrored through <c>1-p</c> when <c>p &gt; 0.5</c>. Unlike the modern
        ///     sampler there is NO early exit — <c>n == 0</c> and <c>p == 0</c> still draw one uniform.
        /// </summary>
        /// <param name="p">The success probability (validated to <c>[0, 1]</c>).</param>
        /// <param name="n">The trial count (validated non-negative).</param>
        /// <returns>The count of successes.</returns>
        private long LegacyBinomial(double p, long n)
        {
            double q;

            if (p <= 0.5)
            {
                if (p * n <= 30.0)
                    return LegacyBinomialInversion(n, p);
                else
                    return Distributions.RandomBinomialBtpe(randomizer, n, p, _binomial);
            }
            else
            {
                q = 1.0 - p;
                if (q * n <= 30.0)
                    return n - LegacyBinomialInversion(n, q);
                else
                    return n - Distributions.RandomBinomialBtpe(randomizer, n, q, _binomial);
            }
        }

        /// <summary>
        ///     NumPy's legacy <c>random_hypergeometric_hyp</c>: the urn simulation used for <c>sample &lt;= 10</c>.
        /// </summary>
        /// <param name="good">Good items.</param>
        /// <param name="bad">Bad items.</param>
        /// <param name="sample">Items drawn (1..10).</param>
        /// <returns>Good items drawn.</returns>
        /// <remarks>Draws nothing when <c>min(good, bad) == 0</c> — the answer is then already known.</remarks>
        private long LegacyHypergeometricHyp(long good, long bad, long sample)
        {
            long d1, k, z;
            double d2, u, y;

            d1 = bad + good - sample;
            d2 = (double)(bad < good ? bad : good);

            y = d2;
            k = sample;
            while (y > 0.0)
            {
                u = randomizer.NextDouble();
                y -= Distributions.ToInt64(Math.Floor(u + y / (d1 + k)));
                k--;
                if (k == 0)
                    break;
            }
            z = Distributions.ToInt64(d2 - y);
            if (good > bad)
                z = sample - z;
            return z;
        }

        /// <summary>
        ///     NumPy's legacy <c>random_hypergeometric_hrua</c>: Stadlober's ratio-of-uniforms (HRUA*) for
        ///     <c>sample &gt; 10</c>, with Frohne's two corrections.
        /// </summary>
        /// <param name="good">Good items.</param>
        /// <param name="bad">Bad items.</param>
        /// <param name="sample">Items drawn (&gt; 10).</param>
        /// <returns>Good items drawn.</returns>
        private long LegacyHypergeometricHrua(long good, long bad, long sample)
        {
            /* D1 = 2*sqrt(2/e) */
            /* D2 = 3 - 2*sqrt(3/e) */
            const double D1 = 1.7155277699214135;
            const double D2 = 0.8989161620588988;

            long mingoodbad, maxgoodbad, popsize, m, d9;
            double d4, d5, d6, d7, d8, d10, d11;
            long Z;
            double T, W, X, Y;

            mingoodbad = good < bad ? good : bad;
            popsize = good + bad;
            maxgoodbad = good > bad ? good : bad;
            m = sample < popsize - sample ? sample : popsize - sample;
            d4 = ((double)mingoodbad) / popsize;
            d5 = 1.0 - d4;
            d6 = m * d4 + 0.5;
            d7 = Math.Sqrt((double)(popsize - m) * sample * d4 * d5 / (popsize - 1) + 0.5);
            d8 = D1 * d7 + D2;
            d9 = Distributions.ToInt64(Math.Floor((double)(m + 1) * (mingoodbad + 1) / (popsize + 2)));
            d10 = (Distributions.RandomLoggam(d9 + 1) + Distributions.RandomLoggam(mingoodbad - d9 + 1) +
                   Distributions.RandomLoggam(m - d9 + 1) + Distributions.RandomLoggam(maxgoodbad - m + d9 + 1));
            // C's nested MIN macros: min(min(m, mingoodbad) + 1.0, floor(d6 + 16 * d7)).
            double lhs = (m < mingoodbad ? m : mingoodbad) + 1.0;
            double rhs = Math.Floor(d6 + 16 * d7);
            d11 = lhs < rhs ? lhs : rhs;
            /* 16 for 16-decimal-digit precision in D1 and D2 */

            while (true)
            {
                X = randomizer.NextDouble();
                Y = randomizer.NextDouble();
                W = d6 + d8 * (Y - 0.5) / X;

                /* fast rejection: */
                if ((W < 0.0) || (W >= d11))
                    continue;

                Z = Distributions.ToInt64(Math.Floor(W));
                T = d10 - (Distributions.RandomLoggam(Z + 1) + Distributions.RandomLoggam(mingoodbad - Z + 1) +
                           Distributions.RandomLoggam(m - Z + 1) + Distributions.RandomLoggam(maxgoodbad - m + Z + 1));

                /* fast acceptance: */
                if ((X * (4.0 - X) - 3.0) <= T)
                    break;

                /* fast rejection: */
                if (X * (X - T) >= 1)
                    continue;
                /* log(0.0) is ok here, since always accept */
                if (2.0 * Math.Log(X) <= T)
                    break; /* acceptance */
            }

            /* this is a correction to HRUA* by Ivan Frohne in rv.py */
            if (good > bad)
                Z = m - Z;

            /* another fix from rv.py to allow sample to exceed popsize/2 */
            if (m < sample)
                Z = good - Z;

            return Z;
        }

        /// <summary>
        ///     NumPy's <c>legacy_random_hypergeometric</c> (<c>random_hypergeometric_original</c>): HRUA for
        ///     <c>sample &gt; 10</c>, the urn simulation for <c>1..10</c>, and 0 for a non-positive sample.
        /// </summary>
        /// <param name="good">Good items.</param>
        /// <param name="bad">Bad items.</param>
        /// <param name="sample">Items drawn.</param>
        /// <returns>Good items drawn.</returns>
        private long LegacyHypergeometric(long good, long bad, long sample)
        {
            if (sample > 10)
                return LegacyHypergeometricHrua(good, bad, sample);
            else if (sample > 0)
                return LegacyHypergeometricHyp(good, bad, sample);
            else
                return 0;
        }

        /// <summary>
        ///     NumPy's <c>legacy_random_zipf</c>: rejection from a Pareto envelope over <c>U = 1 - next_double</c>, truncated to
        ///     C <c>long</c> (the pre-<see cref="Generator"/> algorithm, without the modern <c>Umin</c> window).
        /// </summary>
        /// <param name="a">The exponent (validated <c>&gt; 1</c>).</param>
        /// <returns>The draw (<c>&gt;= 1</c>).</returns>
        /// <remarks>
        ///     For <c>a &gt;= 1025</c> (and <c>a = inf</c>) <c>2^(a-1)</c> overflows, every acceptance test compares NaN, and
        ///     NumPy's loop never exits. This port returns 1 there without drawing — the value the distribution degenerates to
        ///     (the modern sampler returns 1 for the same range).
        /// </remarks>
        private long LegacyZipf(double a)
        {
            double am1, b;

            am1 = a - 1.0;
            b = Math.Pow(2.0, am1);
            // NumPy hangs here (see remarks): an infinite b makes every candidate's test NaN <= NaN.
            if (double.IsPositiveInfinity(b))
                return 1;
            while (true)
            {
                double T, U, V, X;

                U = 1.0 - randomizer.NextDouble();
                V = randomizer.NextDouble();
                X = Math.Floor(Math.Pow(U, -1.0 / am1));
                /*
                 * The real result may be above what can be represented in a signed
                 * long. Since this is a straightforward rejection algorithm, we can
                 * just reject this value. This function then models a Zipf
                 * distribution truncated to sys.maxint.
                 */
                if (X > (double)long.MaxValue || X < 1.0)
                    continue;

                T = Math.Pow(1.0 + 1.0 / X, am1);
                if (V * X * (T - 1.0) / (b - 1.0) <= T / b)
                    return Distributions.ToInt64(X);
            }
        }

        /// <summary>
        ///     NumPy's <c>legacy_random_geometric</c>: the CDF search for <c>p &gt;= 1/3</c>, else the legacy inversion
        ///     <c>ceil(log1p(-U) / log(1 - p))</c> (no INT64_MAX clamp — an overflow reads as C's indefinite integer).
        /// </summary>
        /// <param name="p">The success probability (validated to <c>(0, 1]</c>).</param>
        /// <returns>The trial of the first success.</returns>
        private long LegacyGeometric(double p)
        {
            if (p >= 0.333333333333333333333333)
                return Distributions.RandomGeometricSearch(randomizer, p);
            return Distributions.ToInt64(Math.Ceiling(Generator.Log1p(-randomizer.NextDouble()) / Math.Log(1 - p)));
        }

        /// <summary>
        ///     NumPy's <c>legacy_vonmises</c>: Best &amp; Fisher's wrapped-Cauchy rejection, a uniform on <c>[-pi, pi)</c> for
        ///     <c>kappa &lt; 1e-8</c>, and — unlike the modern sampler — NO wrapped-normal fallback for large kappa.
        /// </summary>
        /// <param name="mu">The mode.</param>
        /// <param name="kappa">The concentration (validated non-negative; NaN returns NumPy's NaN without drawing).</param>
        /// <returns>The draw, in <c>[-pi, pi]</c>.</returns>
        /// <remarks>
        ///     Once <c>4*kappa*kappa</c> overflows (<c>kappa &gt; ~6.7e153</c>, or infinite) the envelope parameter is NaN and
        ///     NumPy's rejection loop never exits; this port returns <paramref name="mu"/> wrapped exactly as an accepted draw
        ///     would be — the distribution's limit as kappa grows — without drawing.
        /// </remarks>
        private double LegacyVonmises(double mu, double kappa)
        {
            double s;
            double U, V, W, Y, Z;
            double result, mod;
            bool neg;

            if (double.IsNaN(kappa))
                return Distributions.NPY_NAN;
            if (kappa < 1e-8)
            {
                return Math.PI * (2 * randomizer.NextDouble() - 1);
            }
            else
            {
                /* with double precision rho is zero until 1.4e-8 */
                if (kappa < 1e-5)
                {
                    /*
                     * second order taylor expansion around kappa = 0
                     * precise until relatively large kappas as second order is 0
                     */
                    s = (1.0 / kappa + kappa);
                }
                else
                {
                    /* Path for 1e-5 <= kappa <= 1e6 */
                    double r = 1 + Math.Sqrt(1 + 4 * kappa * kappa);
                    double rho = (r - Math.Sqrt(2 * r)) / (2 * kappa);
                    s = (1 + rho * rho) / (2 * rho);
                }

                if (double.IsNaN(s))
                {
                    // NumPy hangs (see remarks): wrap mu as the accepted branch below wraps `result`.
                    result = mu;
                }
                else
                {
                    while (true)
                    {
                        U = randomizer.NextDouble();
                        Z = Math.Cos(Math.PI * U);
                        W = (1 + s * Z) / (s + Z);
                        Y = kappa * (s - W);
                        V = randomizer.NextDouble();
                        /*
                         * V==0.0 is ok here since Y >= 0 always leads
                         * to accept, while Y < 0 always rejects
                         */
                        if ((Y * (2 - Y) - V >= 0) || (Math.Log(Y / V) + 1 - Y >= 0))
                            break;
                    }

                    U = randomizer.NextDouble();

                    result = Math.Acos(W);
                    if (U < 0.5)
                        result = -result;
                    result += mu;
                }

                neg = (result < 0);
                mod = Math.Abs(result);
                // C's fmod: C# `%` on doubles is the same exact truncated remainder.
                mod = ((mod + Math.PI) % (2 * Math.PI)) - Math.PI;
                if (neg)
                    mod *= -1;

                return mod;
            }
        }

        /// <summary>
        ///     NumPy's <c>legacy_logseries</c>: Kemp's second accelerated generator (LK) with the legacy
        ///     <c>log(1 - p)</c> / <c>1 - exp(r*U)</c> spellings (the modern sampler uses <c>log1p</c>/<c>expm1</c>).
        /// </summary>
        /// <param name="p">The shape (validated to <c>[0, 1)</c>).</param>
        /// <returns>The draw (<c>&gt;= 1</c>).</returns>
        /// <remarks>
        ///     An overflowing <c>floor(1 + log(V)/log(q))</c> goes through C's integer cast (<see cref="Distributions.ToInt64"/>)
        ///     and so reads as negative and is REJECTED, as in NumPy — a saturating cast would accept a huge count instead.
        /// </remarks>
        private long LegacyLogseries(double p)
        {
            double q, r, U, V;
            long result;

            r = Math.Log(1.0 - p);

            while (true)
            {
                V = randomizer.NextDouble();
                if (V >= p)
                    return 1;
                U = randomizer.NextDouble();
                q = 1.0 - Math.Exp(r * U);
                if (V <= q * q)
                {
                    result = Distributions.ToInt64(Math.Floor(1 + Math.Log(V) / Math.Log(q)));
                    if ((result < 1) || (V == 0.0))
                        continue;
                    else
                        return result;
                }
                if (V >= q)
                    return 1;
                return 2;
            }
        }
    }
}
