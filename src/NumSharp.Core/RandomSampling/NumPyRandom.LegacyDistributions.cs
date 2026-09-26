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
    ///     The per-call setup NumPy recomputes on every value — Marsaglia-Tsang's <c>1 / sqrt(9b)</c>, <c>log(1 - p)</c>,
    ///     <c>2^(a-1)</c>, the von Mises envelope, HRUA's four <c>loggam</c> terms — lives in small <c>Legacy*Setup</c>
    ///     structs built ONCE per fill (and per call on the scalar path, which is NumPy's own arithmetic). They hold the same
    ///     IEEE expressions over the same inputs, so hoisting them changes no bit while removing the transcendental calls a
    ///     value would otherwise repeat.
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

        /// <summary>
        ///     NumPy's <c>legacy_gauss</c>: the cached polar (Marsaglia) method. Returns the cached value if there is one —
        ///     ZEROING the cache, as NumPy does — else draws <c>(x1, x2)</c> pairs until one lands strictly inside the unit
        ///     circle, returns <c>f * x2</c> and caches <c>f * x1</c>.
        /// </summary>
        /// <param name="src">The draw source — per-draw, or a read-ahead whose no-overdraw accounting the caller keeps.</param>
        /// <returns>A standard normal draw.</returns>
        /// <remarks>
        ///     Every attempt consumes exactly two draws and a cached return consumes none, which is why the pure-Gaussian fills
        ///     account for their read-ahead in PAIRS (see <see cref="LegacyGaussFill"/>) rather than per output.
        /// </remarks>
        private double LegacyGauss(ref DrawBufferDouble src)
        {
            if (_hasGauss)
            {
                double temp = _gaussCache;
                _hasGauss = false;
                _gaussCache = 0.0;
                return temp;
            }

            double f, x1, x2, r2;
            do
            {
                x1 = 2.0 * src.NextDouble() - 1.0;
                x2 = 2.0 * src.NextDouble() - 1.0;
                r2 = x1 * x1 + x2 * x2;
            } while (r2 >= 1.0 || r2 == 0.0);

            /* Polar method, a more efficient version of the Box-Muller approach. */
            f = Math.Sqrt(-2.0 * Math.Log(r2) / r2);
            /* Keep for next call */
            _gaussCache = f * x1;
            _hasGauss = true;
            return f * x2;
        }

        /// <summary>
        ///     Writes the next <paramref name="n"/> <c>legacy_gauss</c> values to <paramref name="dst"/> — exactly the values,
        ///     draws and final Gaussian cache that <paramref name="n"/> consecutive <see cref="LegacyGauss"/> calls produce —
        ///     in two phases per chunk so the transcendental work overlaps.
        /// </summary>
        /// <param name="src">A read-ahead dedicated to this fill (its <see cref="DrawBufferDouble.Owed"/> is managed here).</param>
        /// <param name="dst">The output, <paramref name="n"/> doubles.</param>
        /// <param name="n">The number of values (0 draws nothing).</param>
        /// <remarks>
        ///     <para>
        ///     Why it is exact: a polar attempt always consumes exactly two draws, so the attempts — and which of them land
        ///     inside the unit circle — are fixed by the stream alone. Phase A walks them in NumPy's order and keeps each
        ///     accepted <c>(x1, x2, r2)</c>; phase B then evaluates NumPy's <c>f = sqrt(-2 * log(r2) / r2)</c> for each kept
        ///     pair and writes <c>f * x2</c> (the returned half) followed by <c>f * x1</c> (the half <c>legacy_gauss</c> caches
        ///     and the next call returns). A pending cached value is emitted first, and when <paramref name="n"/> leaves the last
        ///     pair's <c>f * x1</c> unused it goes back into the cache, as the last call would have left it.
        ///     </para>
        ///     <para>
        ///     Why it is faster: one <c>legacy_gauss</c> call chains <c>log</c>, a division and <c>sqrt</c> behind a
        ///     data-dependent rejection branch, so consecutive values cannot overlap; phase B is a branch-free run of
        ///     independent chains the CPU pipelines.
        ///     </para>
        ///     <para>
        ///     The read-ahead is accounted in PAIRS: before each accepted pair a refill may draw at most twice the pairs still
        ///     needed. The per-output "at least one draw" bound the other fills use would be wrong here — one attempt yields
        ///     two outputs, so counting outputs over-counts the cached halves and lets a refill draw a word NumPy never draws.
        ///     Every refill is even (an even capacity and even bounds), so refills land on attempt boundaries and each
        ///     remaining pair consumes at least its one two-draw attempt: the stream never runs ahead of NumPy's.
        ///     </para>
        /// </remarks>
        private unsafe void LegacyGaussFill(ref DrawBufferDouble src, double* dst, long n)
        {
            const int Chunk = 128;
            long i = 0;
            if (n <= 0)
                return;
            if (_hasGauss)
            {
                dst[i++] = _gaussCache;
                _hasGauss = false;
                _gaussCache = 0.0;
            }

            double* x1s = stackalloc double[Chunk];
            double* x2s = stackalloc double[Chunk];
            double* r2s = stackalloc double[Chunk];
            while (i < n)
            {
                long pairsLeft = (n - i + 1) / 2;
                int batch = pairsLeft < Chunk ? (int)pairsLeft : Chunk;

                // Phase A: NumPy's polar attempts, in draw order, until `batch` pairs land strictly inside the circle.
                for (int k = 0; k < batch; k++)
                {
                    double x1, x2, r2;
                    src.Owed = 2 * (pairsLeft - k); // every pair still needed takes at least one two-draw attempt
                    do
                    {
                        x1 = 2.0 * src.NextDouble() - 1.0;
                        x2 = 2.0 * src.NextDouble() - 1.0;
                        r2 = x1 * x1 + x2 * x2;
                    } while (r2 >= 1.0 || r2 == 0.0);
                    x1s[k] = x1;
                    x2s[k] = x2;
                    r2s[k] = r2;
                }

                // Phase B: NumPy's transform of each pair, returned half first, cached half second.
                for (int k = 0; k < batch; k++)
                {
                    double f = Math.Sqrt(-2.0 * Math.Log(r2s[k]) / r2s[k]);
                    dst[i++] = f * x2s[k];
                    if (i < n)
                    {
                        dst[i++] = f * x1s[k];
                    }
                    else
                    {
                        // The final call's cached half stays cached, exactly as legacy_gauss leaves it.
                        _gaussCache = f * x1s[k];
                        _hasGauss = true;
                    }
                }
            }
        }

        /// <summary>NumPy's <c>legacy_standard_exponential</c>: <c>-log(1 - U)</c> (U = 0 gives <c>-0.0</c>; no redraw).</summary>
        /// <param name="src">The draw source — per-draw, or a read-ahead whose no-overdraw accounting the caller keeps.</param>
        /// <returns>The draw.</returns>
        private double LegacyStandardExponential(ref DrawBufferDouble src) => -Math.Log(1.0 - src.NextDouble());

        /// <summary>NumPy's <c>legacy_exponential</c>: <c>scale * legacy_standard_exponential</c>.</summary>
        /// <param name="src">The draw source — per-draw, or a read-ahead whose no-overdraw accounting the caller keeps.</param>
        /// <param name="scale">The scale (validated by the caller).</param>
        /// <returns>The draw.</returns>
        private double LegacyExponential(ref DrawBufferDouble src, double scale) => scale * LegacyStandardExponential(ref src);

        // ------------------------------------------------------------------------------------------------ gamma family

        /// <summary>
        ///     NumPy's <c>legacy_standard_gamma</c>: exponential for <c>shape == 1</c>, 0 (no draw) for <c>shape == 0</c>,
        ///     Johnk/Ahrens-Dieter rejection for <c>shape &lt; 1</c>, and Marsaglia-Tsang (on the cached-Gaussian polar
        ///     normals) above.
        /// </summary>
        /// <param name="src">The draw source — per-draw, or a read-ahead whose no-overdraw accounting the caller keeps.</param>
        /// <param name="g">The shape's setup (validated non-negative by the caller; NaN takes the Marsaglia branch and yields NaN).</param>
        /// <returns>The draw.</returns>
        private double LegacyStandardGamma(ref DrawBufferDouble src, in GammaSetup g)
        {
            double b, c;
            double U, V, X, Y;
            double shape = g.Shape;

            if (shape == 1.0)
            {
                return LegacyStandardExponential(ref src);
            }
            else if (shape == 0.0)
            {
                return 0.0;
            }
            else if (shape < 1.0)
            {
                double invShape = g.InvShape; // 1. / shape
                for (;;)
                {
                    U = src.NextDouble();
                    V = LegacyStandardExponential(ref src);
                    if (U <= 1.0 - shape)
                    {
                        X = Math.Pow(U, invShape);
                        if (X <= V)
                            return X;
                    }
                    else
                    {
                        Y = -Math.Log((1 - U) / shape);
                        X = Math.Pow(1.0 - shape + shape * Y, invShape);
                        if (X <= (V + Y))
                            return X;
                    }
                }
            }
            else
            {
                b = g.B; // shape - 1./3.
                c = g.C; // 1./sqrt(9 * b)
                for (;;)
                {
                    do
                    {
                        X = LegacyGauss(ref src);
                        V = 1.0 + c * X;
                    } while (V <= 0.0);

                    V = V * V * V;
                    U = src.NextDouble();
                    if (U < 1.0 - 0.0331 * (X * X) * (X * X))
                        return (b * V);
                    if (Math.Log(U) < 0.5 * X * X + b * (1.0 - V + Math.Log(V)))
                        return (b * V);
                }
            }
        }

        /// <summary><c>legacy_standard_gamma</c> for a shape that changes per call — builds the setup, NumPy's per-call arithmetic.</summary>
        /// <param name="src">The draw source — per-draw, or a read-ahead whose no-overdraw accounting the caller keeps.</param>
        /// <param name="shape">The shape.</param>
        /// <returns>The draw.</returns>
        private double LegacyStandardGamma(ref DrawBufferDouble src, double shape) => LegacyStandardGamma(ref src, new GammaSetup(shape));

        /// <summary>NumPy's <c>legacy_gamma</c>: <c>scale * legacy_standard_gamma(shape)</c>.</summary>
        /// <param name="src">The draw source — per-draw, or a read-ahead whose no-overdraw accounting the caller keeps.</param>
        /// <param name="g">The shape's setup.</param>
        /// <param name="scale">The scale.</param>
        /// <returns>The draw.</returns>
        private double LegacyGamma(ref DrawBufferDouble src, in GammaSetup g, double scale) => scale * LegacyStandardGamma(ref src, in g);

        /// <summary>NumPy's <c>legacy_chisquare</c>: <c>2 * legacy_standard_gamma(df / 2)</c>.</summary>
        /// <param name="src">The draw source — per-draw, or a read-ahead whose no-overdraw accounting the caller keeps.</param>
        /// <param name="halfDf">The setup of <c>df / 2.0</c>.</param>
        /// <returns>The draw.</returns>
        private double LegacyChisquare(ref DrawBufferDouble src, in GammaSetup halfDf) => 2.0 * LegacyStandardGamma(ref src, in halfDf);

        /// <summary><c>legacy_chisquare</c> for a <paramref name="df"/> that changes per call (the Poisson-mixed noncentral branch).</summary>
        /// <param name="src">The draw source — per-draw, or a read-ahead whose no-overdraw accounting the caller keeps.</param>
        /// <param name="df">The degrees of freedom.</param>
        /// <returns>The draw.</returns>
        private double LegacyChisquare(ref DrawBufferDouble src, double df) => 2.0 * LegacyStandardGamma(ref src, new GammaSetup(df / 2.0));

        /// <summary>The setups of <c>legacy_f</c>: the two chi-squares' <c>df / 2</c> gammas.</summary>
        private readonly struct LegacyFSetup
        {
            /// <summary>The numerator degrees of freedom.</summary>
            internal readonly double Dfnum;

            /// <summary>The denominator degrees of freedom.</summary>
            internal readonly double Dfden;

            /// <summary>The setup of <c>dfnum / 2</c>.</summary>
            internal readonly GammaSetup HalfNum;

            /// <summary>The setup of <c>dfden / 2</c>.</summary>
            internal readonly GammaSetup HalfDen;

            /// <summary>Builds both chi-square setups.</summary>
            /// <param name="dfnum">The numerator degrees of freedom.</param>
            /// <param name="dfden">The denominator degrees of freedom.</param>
            internal LegacyFSetup(double dfnum, double dfden)
            {
                Dfnum = dfnum;
                Dfden = dfden;
                HalfNum = new GammaSetup(dfnum / 2.0);
                HalfDen = new GammaSetup(dfden / 2.0);
            }

            /// <summary>Whether every value draws — false only when BOTH halves are 0 (e.g. both df underflow when halved).</summary>
            internal bool Draws => HalfNum.Draws || HalfDen.Draws;
        }

        /// <summary>
        ///     NumPy's <c>legacy_f</c>: <c>(chi2(dfnum) * dfden) / (chi2(dfden) * dfnum)</c>, the numerator chi-square drawn
        ///     first.
        /// </summary>
        /// <param name="src">The draw source — per-draw, or a read-ahead whose no-overdraw accounting the caller keeps.</param>
        /// <param name="s">The setups.</param>
        /// <returns>The draw.</returns>
        private double LegacyF(ref DrawBufferDouble src, in LegacyFSetup s)
        {
            double num = LegacyChisquare(ref src, in s.HalfNum) * s.Dfden;
            return num / (LegacyChisquare(ref src, in s.HalfDen) * s.Dfnum);
        }

        /// <summary>
        ///     The setup of <c>legacy_noncentral_chisquare</c>: the branch's chi-square gamma, <c>sqrt(nonc)</c>, or the Poisson
        ///     mixing mean <c>nonc / 2</c>.
        /// </summary>
        private readonly struct LegacyNoncentralChisquareSetup
        {
            /// <summary>The degrees of freedom.</summary>
            internal readonly double Df;

            /// <summary>The non-centrality.</summary>
            internal readonly double Nonc;

            /// <summary>The chi-square's <c>df/2</c> setup: of <c>df</c> when <c>nonc == 0</c>, of <c>df - 1</c> when <c>df &gt; 1</c>.</summary>
            internal readonly GammaSetup Chi;

            /// <summary><c>sqrt(nonc)</c> for the <c>df &gt; 1</c> branch.</summary>
            internal readonly double SqrtNonc;

            /// <summary>The Poisson setup of <c>nonc / 2.0</c> for the <c>df &lt;= 1</c> branch.</summary>
            internal readonly PoissonSetup HalfNonc;

            /// <summary>Evaluates the setup of the branch <paramref name="df"/>/<paramref name="nonc"/> select.</summary>
            /// <param name="df">The degrees of freedom.</param>
            /// <param name="nonc">The non-centrality.</param>
            internal LegacyNoncentralChisquareSetup(double df, double nonc)
            {
                Df = df;
                Nonc = nonc;
                Chi = default;
                SqrtNonc = 0.0;
                HalfNonc = default;
                if (nonc == 0)
                {
                    Chi = new GammaSetup(df / 2.0);
                }
                else if (1 < df)
                {
                    // legacy_chisquare(df - 1) halves its argument: the gamma shape is (df - 1) / 2.0.
                    Chi = new GammaSetup((df - 1) / 2.0);
                    SqrtNonc = Math.Sqrt(nonc);
                }
                else
                {
                    // One setup serves every value of the call, so PTRS's rejection log is hoisted too.
                    HalfNonc = new PoissonSetup(nonc / 2.0, hoistRejectionLog: true);
                }
            }

            /// <summary>
            ///     Whether every value draws at least once — the condition a read-ahead fill needs. A gamma of positive (or NaN)
            ///     shape always draws a uniform even when its normal comes from the Gaussian cache, so: the <c>nonc == 0</c>
            ///     branch draws unless <c>df / 2</c> is 0; the <c>df &gt; 1</c> branch always draws (its <c>(df - 1) / 2</c> is
            ///     positive); the Poisson branch draws unless the Poisson mean <c>nonc / 2</c> underflows to 0 — and then its
            ///     chi-square of <c>df + 0</c> still draws unless <c>df / 2</c> is 0 too.
            /// </summary>
            internal bool Draws
            {
                get
                {
                    if (Nonc == 0)
                        return Chi.Draws;
                    if (1 < Df)
                        return true;
                    return HalfNonc.Draws || Df / 2.0 != 0.0;
                }
            }
        }

        /// <summary>
        ///     NumPy's <c>legacy_noncentral_chisquare</c>: plain chi-square for <c>nonc == 0</c>; for <c>df &gt; 1</c> a
        ///     <c>df - 1</c> chi-square plus a shifted squared normal; otherwise a Poisson-mixed chi-square — whose NaN guard
        ///     sits AFTER the draws so a NaN non-centrality still advances the stream as NumPy's does.
        /// </summary>
        /// <param name="src">The draw source — per-draw, or a read-ahead whose no-overdraw accounting the caller keeps.</param>
        /// <param name="s">The setup.</param>
        /// <returns>The draw.</returns>
        private double LegacyNoncentralChisquare(ref DrawBufferDouble src, in LegacyNoncentralChisquareSetup s)
        {
            double @out;
            if (s.Nonc == 0)
                return LegacyChisquare(ref src, in s.Chi);
            if (1 < s.Df)
            {
                double Chi2 = LegacyChisquare(ref src, in s.Chi);
                double n = LegacyGauss(ref src) + s.SqrtNonc;
                return Chi2 + n * n;
            }
            else
            {
                long i = Distributions.RandomPoisson(ref src, in s.HalfNonc);
                // The chi-square's df depends on the Poisson draw, so its setup is per value — NumPy's arithmetic.
                @out = LegacyChisquare(ref src, s.Df + 2 * i);
                /* Insert nan guard here to avoid changing the stream */
                if (double.IsNaN(s.Nonc))
                    return Distributions.NPY_NAN;
                return @out;
            }
        }

        /// <summary>The setups of <c>legacy_noncentral_f</c>: the numerator's noncentral chi-square and the denominator's chi-square.</summary>
        private readonly struct LegacyNoncentralFSetup
        {
            /// <summary>The numerator degrees of freedom.</summary>
            internal readonly double Dfnum;

            /// <summary>The denominator degrees of freedom.</summary>
            internal readonly double Dfden;

            /// <summary>The numerator's noncentral chi-square setup.</summary>
            internal readonly LegacyNoncentralChisquareSetup Num;

            /// <summary>The denominator chi-square's <c>dfden / 2</c> setup.</summary>
            internal readonly GammaSetup HalfDen;

            /// <summary>Builds both setups.</summary>
            /// <param name="dfnum">The numerator degrees of freedom.</param>
            /// <param name="dfden">The denominator degrees of freedom.</param>
            /// <param name="nonc">The non-centrality.</param>
            internal LegacyNoncentralFSetup(double dfnum, double dfden, double nonc)
            {
                Dfnum = dfnum;
                Dfden = dfden;
                Num = new LegacyNoncentralChisquareSetup(dfnum, nonc);
                HalfDen = new GammaSetup(dfden / 2.0);
            }

            /// <summary>Whether every value draws — when either the numerator or the denominator does.</summary>
            internal bool Draws => Num.Draws || HalfDen.Draws;
        }

        /// <summary>NumPy's <c>legacy_noncentral_f</c>: <c>(ncchi2(dfnum, nonc) * dfden) / (chi2(dfden) * dfnum)</c>.</summary>
        /// <param name="src">The draw source — per-draw, or a read-ahead whose no-overdraw accounting the caller keeps.</param>
        /// <param name="s">The setups.</param>
        /// <returns>The draw.</returns>
        private double LegacyNoncentralF(ref DrawBufferDouble src, in LegacyNoncentralFSetup s)
        {
            double t = LegacyNoncentralChisquare(ref src, in s.Num) * s.Dfden;
            return t / (LegacyChisquare(ref src, in s.HalfDen) * s.Dfnum);
        }

        /// <summary>The setup of <c>legacy_standard_t</c>: the <c>df / 2</c> gamma and <c>sqrt(df / 2)</c>.</summary>
        private readonly struct LegacyStandardTSetup
        {
            /// <summary>The setup of <c>df / 2</c>.</summary>
            internal readonly GammaSetup Half;

            /// <summary><c>sqrt(df / 2)</c>.</summary>
            internal readonly double SqrtHalf;

            /// <summary>Evaluates both per-df terms.</summary>
            /// <param name="df">The degrees of freedom.</param>
            internal LegacyStandardTSetup(double df)
            {
                Half = new GammaSetup(df / 2);
                SqrtHalf = Math.Sqrt(df / 2);
            }
        }

        /// <summary>NumPy's <c>legacy_standard_t</c>: <c>sqrt(df/2) * N / sqrt(G(df/2))</c>, the normal drawn first.</summary>
        /// <param name="src">The draw source — per-draw, or a read-ahead whose no-overdraw accounting the caller keeps.</param>
        /// <param name="s">The setup.</param>
        /// <returns>The draw.</returns>
        private double LegacyStandardT(ref DrawBufferDouble src, in LegacyStandardTSetup s)
        {
            double num, denom;

            num = LegacyGauss(ref src);
            denom = LegacyStandardGamma(ref src, in s.Half);
            return s.SqrtHalf * num / Math.Sqrt(denom);
        }

        /// <summary>The setup of <c>legacy_negative_binomial</c>: the shape-<c>n</c> gamma and its scale <c>(1 - p) / p</c>.</summary>
        private readonly struct LegacyNegativeBinomialSetup
        {
            /// <summary>The setup of the gamma shape <c>n</c>.</summary>
            internal readonly GammaSetup Gamma;

            /// <summary>The gamma scale <c>(1 - p) / p</c>.</summary>
            internal readonly double Scale;

            /// <summary>Evaluates both terms.</summary>
            /// <param name="n">The number of successes.</param>
            /// <param name="p">The success probability.</param>
            internal LegacyNegativeBinomialSetup(double n, double p)
            {
                Gamma = new GammaSetup(n);
                Scale = (1 - p) / p;
            }
        }

        /// <summary>
        ///     NumPy's <c>legacy_negative_binomial</c>: a Poisson whose mean is a legacy gamma draw with scale
        ///     <c>(1 - p) / p</c>.
        /// </summary>
        /// <param name="src">The draw source — per-draw, or a read-ahead whose no-overdraw accounting the caller keeps.</param>
        /// <param name="s">The setup.</param>
        /// <returns>The count of failures.</returns>
        /// <remarks>
        ///     <c>p == 0</c> makes the mean infinite; PTRS then converts an infinite/NaN floor through C's integer cast, which
        ///     yields <see cref="long.MinValue"/> here (C <c>long</c> is 64-bit in NumSharp's legacy model; NumPy's win-amd64
        ///     build, with a 32-bit <c>long</c>, returns <c>-2147483648</c>). The Poisson mean changes per value, so its setup
        ///     is built per value — NumPy's arithmetic.
        /// </remarks>
        private long LegacyNegativeBinomial(ref DrawBufferDouble src, in LegacyNegativeBinomialSetup s)
        {
            double Y = LegacyGamma(ref src, in s.Gamma, s.Scale);
            return Distributions.RandomPoisson(ref src, Y);
        }

        /// <summary>
        ///     The setup of <c>legacy_beta</c>: Johnk's <c>1/a</c> and <c>1/b</c> when both shapes are <c>&lt;= 1</c>, else the
        ///     two gammas'.
        /// </summary>
        private readonly struct LegacyBetaSetup
        {
            /// <summary>Alpha.</summary>
            internal readonly double A;

            /// <summary>Beta.</summary>
            internal readonly double B;

            /// <summary>Whether Johnk's algorithm runs (<c>a &lt;= 1 &amp;&amp; b &lt;= 1</c>).</summary>
            internal readonly bool Johnk;

            /// <summary>Johnk's <c>1.0 / a</c>.</summary>
            internal readonly double InvA;

            /// <summary>Johnk's <c>1.0 / b</c>.</summary>
            internal readonly double InvB;

            /// <summary>The gamma-ratio branch's setup of <c>a</c>.</summary>
            internal readonly GammaSetup GammaA;

            /// <summary>The gamma-ratio branch's setup of <c>b</c>.</summary>
            internal readonly GammaSetup GammaB;

            /// <summary>Evaluates the selected branch's per-shape terms.</summary>
            /// <param name="a">Alpha (validated positive or NaN).</param>
            /// <param name="b">Beta (validated positive or NaN).</param>
            internal LegacyBetaSetup(double a, double b)
            {
                A = a;
                B = b;
                Johnk = (a <= 1.0) && (b <= 1.0);
                InvA = InvB = 0.0;
                GammaA = GammaB = default;
                if (Johnk)
                {
                    InvA = 1.0 / a;
                    InvB = 1.0 / b;
                }
                else
                {
                    GammaA = new GammaSetup(a);
                    GammaB = new GammaSetup(b);
                }
            }
        }

        /// <summary>
        ///     NumPy's <c>legacy_beta</c>: Johnk's algorithm when both shapes are <c>&lt;= 1</c> (with the log-space fallback
        ///     when both powers underflow), else the ratio of two legacy gammas.
        /// </summary>
        /// <param name="src">The draw source — per-draw, or a read-ahead whose no-overdraw accounting the caller keeps.</param>
        /// <param name="s">The setup.</param>
        /// <returns>The draw.</returns>
        private double LegacyBeta(ref DrawBufferDouble src, in LegacyBetaSetup s)
        {
            double Ga, Gb;

            if (s.Johnk)
            {
                double U, V, X, Y;
                double a = s.A, b = s.B, invA = s.InvA, invB = s.InvB;
                /* Use Johnk's algorithm */

                while (true)
                {
                    U = src.NextDouble();
                    V = src.NextDouble();
                    X = Math.Pow(U, invA); // 1.0 / a
                    Y = Math.Pow(V, invB); // 1.0 / b

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
                Ga = LegacyStandardGamma(ref src, in s.GammaA);
                Gb = LegacyStandardGamma(ref src, in s.GammaB);
                return Ga / (Ga + Gb);
            }
        }

        // ------------------------------------------------------------------------------------------------ one-parameter transforms

        /// <summary>NumPy's <c>legacy_pareto</c>: <c>exp(E / a) - 1</c> (NOT <c>expm1</c> — the legacy rounding).</summary>
        /// <param name="src">The draw source — per-draw, or a read-ahead whose no-overdraw accounting the caller keeps.</param>
        /// <param name="a">The shape.</param>
        /// <returns>The draw.</returns>
        private double LegacyPareto(ref DrawBufferDouble src, double a) => Math.Exp(LegacyStandardExponential(ref src) / a) - 1;

        /// <summary>NumPy's <c>legacy_weibull</c>: 0 with NO draw for <c>a == 0</c>, else <c>E^(1/a)</c>.</summary>
        /// <param name="src">The draw source — per-draw, or a read-ahead whose no-overdraw accounting the caller keeps.</param>
        /// <param name="a">The shape.</param>
        /// <returns>The draw.</returns>
        private double LegacyWeibull(ref DrawBufferDouble src, double a)
        {
            if (a == 0.0)
                return 0.0;
            return Math.Pow(LegacyStandardExponential(ref src), 1.0 / a);
        }

        /// <summary>NumPy's <c>legacy_power</c>: <c>(1 - exp(-E))^(1/a)</c> (NOT <c>-expm1</c>).</summary>
        /// <param name="src">The draw source — per-draw, or a read-ahead whose no-overdraw accounting the caller keeps.</param>
        /// <param name="a">The exponent parameter.</param>
        /// <returns>The draw.</returns>
        private double LegacyPower(ref DrawBufferDouble src, double a) => Math.Pow(1 - Math.Exp(-LegacyStandardExponential(ref src)), 1.0 / a);

        /// <summary>
        ///     NumPy's <c>legacy_rayleigh</c>: <c>mode * sqrt(-2 * log1p(-U))</c> — it takes the bare bit generator, so it
        ///     neither reads nor fills the Gaussian cache.
        /// </summary>
        /// <param name="src">The draw source — per-draw, or a read-ahead whose no-overdraw accounting the caller keeps.</param>
        /// <param name="mode">The scale.</param>
        /// <returns>The draw.</returns>
        /// <remarks><see cref="Generator.Log1p"/> reproduces the CRT's <c>log1p</c> bit for bit (NumPy's <c>npy_log1p</c>).</remarks>
        private double LegacyRayleigh(ref DrawBufferDouble src, double mode) => mode * Math.Sqrt(-2.0 * Generator.Log1p(-src.NextDouble()));

        // ------------------------------------------------------------------------------------------------ Gaussian-based

        /// <summary>The setup of <c>legacy_wald</c>: <c>mu_2l = mean / (2 * scale)</c>.</summary>
        private readonly struct LegacyWaldSetup
        {
            /// <summary>The mean.</summary>
            internal readonly double Mean;

            /// <summary>The scale.</summary>
            internal readonly double Scale;

            /// <summary><c>mean / (2 * scale)</c>.</summary>
            internal readonly double Mu2l;

            /// <summary>Evaluates <c>mu_2l</c>.</summary>
            /// <param name="mean">The mean.</param>
            /// <param name="scale">The scale.</param>
            internal LegacyWaldSetup(double mean, double scale)
            {
                Mean = mean;
                Scale = scale;
                Mu2l = mean / (2 * scale);
            }
        }

        /// <summary>
        ///     NumPy's <c>legacy_wald</c>: Michael/Schucany/Haas with the legacy <c>mean + mu_2l * (Y - sqrt(4*scale*Y + Y*Y))</c>
        ///     spelling (the Generator rewrote it to avoid cancellation; the legacy form must keep it).
        /// </summary>
        /// <param name="src">The draw source — per-draw, or a read-ahead whose no-overdraw accounting the caller keeps.</param>
        /// <param name="s">The setup.</param>
        /// <returns>The draw.</returns>
        private double LegacyWald(ref DrawBufferDouble src, in LegacyWaldSetup s)
        {
            double U, X, Y;
            double mean = s.Mean, scale = s.Scale, mu_2l = s.Mu2l;

            Y = LegacyGauss(ref src);
            Y = mean * Y * Y;
            X = mean + mu_2l * (Y - Math.Sqrt(4 * scale * Y + Y * Y));
            U = src.NextDouble();
            if (U <= mean / (mean + X))
                return X;
            else
                return mean * mean / X;
        }

        /// <summary>
        ///     Writes the next <paramref name="n"/> <c>legacy_wald</c> values to <paramref name="dst"/> — exactly the values,
        ///     draws and final Gaussian cache that <paramref name="n"/> consecutive <see cref="LegacyWald"/> calls produce — in
        ///     two phases per chunk, like <see cref="LegacyGaussFill"/>.
        /// </summary>
        /// <param name="src">A read-ahead dedicated to this fill (its <see cref="DrawBufferDouble.Owed"/> is managed here).</param>
        /// <param name="dst">The output, <paramref name="n"/> doubles.</param>
        /// <param name="n">The number of values (0 draws nothing).</param>
        /// <param name="s">The setup.</param>
        /// <remarks>
        ///     <para>
        ///     Why it is exact: each <c>legacy_wald</c> takes one <c>legacy_gauss</c> — the pending cached half, or a fresh
        ///     polar pair whose other half it caches for the next value — and then exactly one uniform, whatever the values
        ///     turn out to be. So the draws are fixed by the stream alone: phase A walks them in NumPy's order, recording each
        ///     value's Gaussian source (the cache pending at chunk start, a pair's returned half, or the previous pair's cached
        ///     half) and its uniform; phase B evaluates <c>f = sqrt(-2 * log(r2) / r2)</c> per pair and then NumPy's
        ///     <c>legacy_wald</c> expressions per value. The cache is left exactly as the last call would leave it.
        ///     </para>
        ///     <para>
        ///     The read-ahead may draw at most one double per value still owed: every value draws its uniform, and a refill
        ///     between an attempt's two draws still owes that attempt's second draw on top of every later uniform.
        ///     </para>
        /// </remarks>
        private unsafe void LegacyWaldFill(ref DrawBufferDouble src, double* dst, long n, in LegacyWaldSetup s)
        {
            const int Chunk = 128;
            double mean = s.Mean, scale = s.Scale, mu_2l = s.Mu2l;
            double* x1s = stackalloc double[Chunk];
            double* x2s = stackalloc double[Chunk];
            double* r2s = stackalloc double[Chunk];
            double* fs = stackalloc double[Chunk];
            double* us = stackalloc double[Chunk];
            // Per value, where its Gaussian comes from: -1 = the cached value pending at chunk start; 2p = pair p's returned
            // half (f * x2); 2p + 1 = pair p's cached half (f * x1), taken by the value after the one that drew pair p.
            int* source = stackalloc int[Chunk];

            for (long i = 0; i < n; i += Chunk)
            {
                int m = n - i < Chunk ? (int)(n - i) : Chunk;
                int pairs = 0;
                double startCache = _gaussCache;
                bool cached = _hasGauss; // whether the next legacy_gauss returns a cached half

                // Phase A: the draws, in NumPy's order — each value's Gaussian, then its uniform.
                for (int j = 0; j < m; j++)
                {
                    src.Owed = n - (i + j);
                    if (cached)
                    {
                        source[j] = pairs == 0 ? -1 : 2 * (pairs - 1) + 1;
                        cached = false;
                    }
                    else
                    {
                        double x1, x2, r2;
                        do
                        {
                            x1 = 2.0 * src.NextDouble() - 1.0;
                            x2 = 2.0 * src.NextDouble() - 1.0;
                            r2 = x1 * x1 + x2 * x2;
                        } while (r2 >= 1.0 || r2 == 0.0);
                        x1s[pairs] = x1;
                        x2s[pairs] = x2;
                        r2s[pairs] = r2;
                        source[j] = 2 * pairs;
                        pairs++;
                        cached = true;
                    }
                    us[j] = src.NextDouble();
                }

                // Phase B: legacy_gauss's transform per pair, then legacy_wald's expressions per value.
                for (int p = 0; p < pairs; p++)
                    fs[p] = Math.Sqrt(-2.0 * Math.Log(r2s[p]) / r2s[p]);
                for (int j = 0; j < m; j++)
                {
                    int sj = source[j];
                    double Y = sj < 0 ? startCache : (sj & 1) == 0 ? fs[sj >> 1] * x2s[sj >> 1] : fs[sj >> 1] * x1s[sj >> 1];
                    Y = mean * Y * Y;
                    double X = mean + mu_2l * (Y - Math.Sqrt(4 * scale * Y + Y * Y));
                    dst[i + j] = us[j] <= mean / (mean + X) ? X : mean * mean / X;
                }

                // The cache exactly as the chunk's last legacy_gauss call leaves it.
                if (cached)
                {
                    _gaussCache = fs[pairs - 1] * x1s[pairs - 1];
                    _hasGauss = true;
                }
                else
                {
                    _gaussCache = 0.0;
                    _hasGauss = false;
                }
            }
        }

        /// <summary>NumPy's <c>legacy_normal</c>: <c>loc + scale * legacy_gauss</c>.</summary>
        /// <param name="src">The draw source — per-draw, or a read-ahead whose no-overdraw accounting the caller keeps.</param>
        /// <param name="loc">The mean.</param>
        /// <param name="scale">The standard deviation.</param>
        /// <returns>The draw.</returns>
        private double LegacyNormal(ref DrawBufferDouble src, double loc, double scale) => loc + scale * LegacyGauss(ref src);

        /// <summary>NumPy's <c>legacy_lognormal</c>: <c>exp(legacy_normal(mean, sigma))</c>.</summary>
        /// <param name="src">The draw source — per-draw, or a read-ahead whose no-overdraw accounting the caller keeps.</param>
        /// <param name="mean">The underlying normal's mean.</param>
        /// <param name="sigma">The underlying normal's standard deviation.</param>
        /// <returns>The draw.</returns>
        private double LegacyLognormal(ref DrawBufferDouble src, double mean, double sigma) => Math.Exp(LegacyNormal(ref src, mean, sigma));

        /// <summary>
        ///     NumPy's <c>legacy_standard_cauchy</c>: the ratio of two polar normals, numerator drawn FIRST (MSVC and GCC both
        ///     evaluate the C division's operands left to right here, which NumPy's own expected values confirm).
        /// </summary>
        /// <param name="src">The draw source — per-draw, or a read-ahead whose no-overdraw accounting the caller keeps.</param>
        /// <returns>The draw.</returns>
        private double LegacyStandardCauchy(ref DrawBufferDouble src)
        {
            double num = LegacyGauss(ref src);
            return num / LegacyGauss(ref src);
        }

        // ------------------------------------------------------------------------------------------------ discrete

        /// <summary>
        ///     NumPy's <c>legacy_random_binomial_inversion</c>: sequential CDF search with <c>q^n = exp(n * log(q))</c> — the
        ///     legacy spelling (the modern sampler uses <c>log1p(-p)</c>), cached in <see cref="_binomial"/>.
        /// </summary>
        /// <param name="src">The draw source — per-draw, or a read-ahead whose no-overdraw accounting the caller keeps.</param>
        /// <param name="n">The trial count.</param>
        /// <param name="p">The success probability (<c>&lt;= 0.5</c>).</param>
        /// <returns>The count of successes.</returns>
        /// <remarks>
        ///     The walk itself is shared with the modern sampler (<see cref="BinomialSetup.InversionWalk"/>), which memoizes
        ///     the <c>px</c> sequence per key; the setup statements here are the legacy ones.
        /// </remarks>
        private long LegacyBinomialInversion(ref DrawBufferDouble src, long n, double p)
        {
            BinomialSetup e = _binomial.Current;
            if (!e.has_binomial || (e.nsave != n) || (e.psave != p))
            {
                // NumPy's in-place repopulation of its one entry — never through the modern-only memo (BinomialState):
                // a legacy q^n must not land in a memo slot a later multinomial lookup could hit.
                double q, np;
                e.nsave = n;
                e.psave = p;
                e.has_binomial = true;
                e.q = q = 1.0 - p;
                e.r = Math.Exp(n * Math.Log(q));
                e.c = np = n * p;
                // C's MIN macro — `n < y ? n : y` in double — then the (RAND_INT_TYPE) cast.
                double limit = np + 10.0 * Math.Sqrt(np * q + 1);
                e.m = Distributions.ToInt64(n < limit ? n : limit);
                e.ResetMemos();
            }
            return e.InversionWalk(ref src, n, p);
        }

        /// <summary>
        ///     NumPy's <c>legacy_random_binomial</c> (<c>legacy_random_binomial_original</c>): the legacy inversion for
        ///     <c>n*min(p,1-p) &lt;= 30</c>, BTPE above, mirrored through <c>1-p</c> when <c>p &gt; 0.5</c>. Unlike the modern
        ///     sampler there is NO early exit — <c>n == 0</c> and <c>p == 0</c> still draw one uniform.
        /// </summary>
        /// <param name="src">The draw source — per-draw, or a read-ahead whose no-overdraw accounting the caller keeps.</param>
        /// <param name="p">The success probability (validated to <c>[0, 1]</c>).</param>
        /// <param name="n">The trial count (validated non-negative).</param>
        /// <returns>The count of successes.</returns>
        private long LegacyBinomial(ref DrawBufferDouble src, double p, long n)
        {
            double q;

            if (p <= 0.5)
            {
                if (p * n <= 30.0)
                    return LegacyBinomialInversion(ref src, n, p);
                else
                    return Distributions.RandomBinomialBtpe(ref src, n, p, _binomial);
            }
            else
            {
                q = 1.0 - p;
                if (q * n <= 30.0)
                    return n - LegacyBinomialInversion(ref src, n, q);
                else
                    return n - Distributions.RandomBinomialBtpe(ref src, n, q, _binomial);
            }
        }

        /// <summary>
        ///     NumPy's legacy <c>random_hypergeometric_hyp</c>: the urn simulation used for <c>sample &lt;= 10</c>.
        /// </summary>
        /// <param name="src">The draw source — per-draw, or a read-ahead whose no-overdraw accounting the caller keeps.</param>
        /// <param name="good">Good items.</param>
        /// <param name="bad">Bad items.</param>
        /// <param name="sample">Items drawn (1..10).</param>
        /// <returns>Good items drawn.</returns>
        /// <remarks>Draws nothing when <c>min(good, bad) == 0</c> — the answer is then already known.</remarks>
        private long LegacyHypergeometricHyp(ref DrawBufferDouble src, long good, long bad, long sample)
        {
            long d1, k, z;
            double d2, u, y;

            d1 = bad + good - sample;
            d2 = (double)(bad < good ? bad : good);

            y = d2;
            k = sample;
            while (y > 0.0)
            {
                u = src.NextDouble();
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
        ///     The precomputed ratios <c>y / (d1 + k)</c> of the urn simulation for one <c>(good, bad, sample)</c>, so a fill's
        ///     walk does no division — every <c>(y, k)</c> the walk can reach in its normal steps, keyed by how far <c>y</c> has
        ///     fallen from <c>d2 = min(good, bad)</c>.
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     The C walk is <c>y -= (long)floor(u + y / (d1 + k))</c> with <c>y</c> an integer-valued double starting at
        ///     <c>d2</c> and <c>k</c> counting down from <c>sample</c>. Two facts make a division-free walk exact. First, the
        ///     ratio depends only on <c>(y, k)</c>, so storing <c>y / (double)(d1 + k)</c> — the same division — and adding
        ///     <c>u</c> to it reproduces <c>u + y / (d1 + k)</c> bit for bit. Second, <c>0 &lt; y &lt;= d1 + k</c> holds at every
        ///     step (an urn never holds more of one colour than balls), so the ratio lies in <c>(0, 1]</c>, the sum <c>s</c> in
        ///     <c>(0, 2]</c>, and <c>floor(s)</c> is exactly <c>(s &gt;= 1) + (s &gt;= 2)</c> — the <c>2</c> occurring only when the
        ///     ratio is exactly 1 and <c>u</c> is the largest uniform, which NumPy's floor produces too.
        ///     </para>
        ///     <para>
        ///     Only built when <c>d2 &lt;= 2^53</c>: tracking <c>y</c> as an integer is exact only while every value it takes is an
        ///     exact double. A step past the table (the rare 2-step) divides directly, as NumPy does.
        ///     </para>
        /// </remarks>
        private sealed class LegacyHypTable
        {
            /// <summary><c>d1 = bad + good - sample</c>.</summary>
            internal readonly long D1;

            /// <summary><c>min(good, bad)</c> as an integer (exact as a double — see the class remarks).</summary>
            internal readonly long D2;

            /// <summary>The number of <c>y</c> offsets per <c>k</c> row (<c>sample + 1</c>).</summary>
            internal readonly int Rows;

            /// <summary><c>Ratio[(k - 1) * Rows + (D2 - y)] = y / (double)(D1 + k)</c> for <c>y &gt; 0</c>.</summary>
            internal readonly double[] Ratio;

            /// <summary>Precomputes every ratio a normal walk can reach.</summary>
            /// <param name="good">Good items.</param>
            /// <param name="bad">Bad items.</param>
            /// <param name="sample">Items drawn (1..10).</param>
            internal LegacyHypTable(long good, long bad, long sample)
            {
                D1 = bad + good - sample;
                D2 = bad < good ? bad : good;
                Rows = (int)sample + 1;
                Ratio = new double[sample * Rows];
                for (long k = 1; k <= sample; k++)
                    for (int off = 0; off < Rows; off++)
                    {
                        long y = D2 - off;
                        if (y > 0)
                            Ratio[(k - 1) * Rows + off] = y / (double)(D1 + k);
                    }
            }

            /// <summary>Whether the integer walk is exact for these parameters (<c>min(good, bad) &lt;= 2^53</c>).</summary>
            /// <param name="good">Good items.</param>
            /// <param name="bad">Bad items.</param>
            /// <returns>True when a table may be built.</returns>
            internal static bool Applicable(long good, long bad) => (bad < good ? bad : good) <= (1L << 53);
        }

        /// <summary>
        ///     <see cref="LegacyHypergeometricHyp"/> over a precomputed <see cref="LegacyHypTable"/>: the same draws, steps and
        ///     result, with each step's division read from the table and its floor taken as two comparisons (see the table's
        ///     remarks for why both are exact).
        /// </summary>
        /// <param name="src">The draw source — per-draw, or a read-ahead whose no-overdraw accounting the caller keeps.</param>
        /// <param name="good">Good items.</param>
        /// <param name="bad">Bad items.</param>
        /// <param name="sample">Items drawn (1..10).</param>
        /// <param name="tab">The table for exactly these parameters.</param>
        /// <returns>Good items drawn.</returns>
        private long LegacyHypergeometricHyp(ref DrawBufferDouble src, long good, long bad, long sample, LegacyHypTable tab)
        {
            long d2 = tab.D2, d1 = tab.D1;
            int rows = tab.Rows;
            double[] ratio = tab.Ratio;
            long y = d2; // the C code's double y, exact as an integer here
            long k = sample;
            while (y > 0)
            {
                double u = src.NextDouble();
                long off = d2 - y;
                // The C ratio y / (d1 + k); a walk past the table's offsets (only after a 2-step) divides as NumPy does.
                double t = off < rows ? ratio[(k - 1) * rows + off] : y / (double)(d1 + k);
                double s = u + t;
                // floor(s) for s in (0, 2] — see LegacyHypTable.
                y -= (s >= 1.0 ? 1 : 0) + (s >= 2.0 ? 1 : 0);
                k--;
                if (k == 0)
                    break;
            }
            long z = d2 - y; // (RAND_INT_TYPE)(d2 - y)
            if (good > bad)
                z = sample - z;
            return z;
        }

        /// <summary>
        ///     The per-call setup of NumPy's legacy <c>random_hypergeometric_hrua</c> (<c>sample &gt; 10</c>): <c>d4</c>..<c>d11</c>
        ///     — a <c>sqrt</c>, four <c>loggam</c>s and several divisions — computed once for a fill.
        /// </summary>
        private readonly struct LegacyHruaSetup
        {
            /// <summary><c>min(good, bad)</c>.</summary>
            internal readonly long Mingoodbad;

            /// <summary><c>max(good, bad)</c>.</summary>
            internal readonly long Maxgoodbad;

            /// <summary><c>min(sample, popsize - sample)</c>.</summary>
            internal readonly long M;

            /// <summary><c>m * d4 + 0.5</c>, the envelope's centre.</summary>
            internal readonly double D6;

            /// <summary><c>D1 * d7 + D2</c>, the envelope's scale.</summary>
            internal readonly double D8;

            /// <summary>The four <c>loggam</c> terms at the mode <c>d9</c>.</summary>
            internal readonly double D10;

            /// <summary>The fast-rejection bound on the candidate.</summary>
            internal readonly double D11;

            /// <summary>Evaluates NumPy's setup statements.</summary>
            /// <param name="good">Good items.</param>
            /// <param name="bad">Bad items.</param>
            /// <param name="sample">Items drawn (&gt; 10).</param>
            internal LegacyHruaSetup(long good, long bad, long sample)
            {
                /* D1 = 2*sqrt(2/e) */
                /* D2 = 3 - 2*sqrt(3/e) */
                const double D1 = 1.7155277699214135;
                const double D2 = 0.8989161620588988;

                long popsize, d9;
                double d4, d5, d7;

                Mingoodbad = good < bad ? good : bad;
                popsize = good + bad;
                Maxgoodbad = good > bad ? good : bad;
                M = sample < popsize - sample ? sample : popsize - sample;
                d4 = ((double)Mingoodbad) / popsize;
                d5 = 1.0 - d4;
                D6 = M * d4 + 0.5;
                d7 = Math.Sqrt((double)(popsize - M) * sample * d4 * d5 / (popsize - 1) + 0.5);
                D8 = D1 * d7 + D2;
                d9 = Distributions.ToInt64(Math.Floor((double)(M + 1) * (Mingoodbad + 1) / (popsize + 2)));
                D10 = (Distributions.RandomLoggam(d9 + 1) + Distributions.RandomLoggam(Mingoodbad - d9 + 1) +
                       Distributions.RandomLoggam(M - d9 + 1) + Distributions.RandomLoggam(Maxgoodbad - M + d9 + 1));
                // C's nested MIN macros: min(min(m, mingoodbad) + 1.0, floor(d6 + 16 * d7)).
                double lhs = (M < Mingoodbad ? M : Mingoodbad) + 1.0;
                double rhs = Math.Floor(D6 + 16 * d7);
                D11 = lhs < rhs ? lhs : rhs;
                /* 16 for 16-decimal-digit precision in D1 and D2 */
            }

            /// <summary>The four-<c>loggam</c> sum <c>T</c> subtracts from <c>d10</c> for candidate <paramref name="Z"/> — NumPy's expression, in its order.</summary>
            /// <param name="Z">The candidate count.</param>
            /// <returns>The sum.</returns>
            internal double LoggamSum(long Z)
                => (Distributions.RandomLoggam(Z + 1) + Distributions.RandomLoggam(Mingoodbad - Z + 1) +
                    Distributions.RandomLoggam(M - Z + 1) + Distributions.RandomLoggam(Maxgoodbad - M + Z + 1));
        }

        /// <summary>
        ///     A direct-mapped memo of <see cref="LegacyHruaSetup.LoggamSum"/> by candidate, for one fill: HRUA evaluates four
        ///     <c>loggam</c>s (four <c>log</c> calls and four Stirling series) per attempt, and the candidates cluster around
        ///     the mode, so most attempts repeat a candidate already seen.
        /// </summary>
        /// <remarks>
        ///     The sum depends only on the setup and the candidate, and a miss evaluates NumPy's expression in NumPy's order,
        ///     so <c>T = d10 - sum</c> is bit-identical whether the sum was memoized or not. Candidates are non-negative
        ///     (HRUA rejects <c>W &lt; 0</c> before flooring), so -1 marks an empty slot. 1024 slots (16 KB) cover the candidate
        ///     spread of every moderate population; a wider spread only collides and recomputes.
        /// </remarks>
        private sealed class LoggamSumMemo
        {
            /// <summary>The slot count (a power of two).</summary>
            private const int Size = 1024;

            /// <summary>The candidate each slot holds, -1 when empty.</summary>
            private readonly long[] _keys = new long[Size];

            /// <summary>The memoized sums.</summary>
            private readonly double[] _values = new double[Size];

            /// <summary>Creates an empty memo.</summary>
            internal LoggamSumMemo() => Array.Fill(_keys, -1L);

            /// <summary>The four-<c>loggam</c> sum for <paramref name="Z"/>, from the memo or computed and stored.</summary>
            /// <param name="setup">The fill's setup (the memo belongs to exactly one).</param>
            /// <param name="Z">The candidate (<c>&gt;= 0</c>).</param>
            /// <returns>The sum, bit-identical to <see cref="LegacyHruaSetup.LoggamSum"/>.</returns>
            internal double Get(in LegacyHruaSetup setup, long Z)
            {
                int slot = (int)Z & (Size - 1);
                if (_keys[slot] == Z)
                    return _values[slot];
                double sum = setup.LoggamSum(Z);
                _keys[slot] = Z;
                _values[slot] = sum;
                return sum;
            }
        }

        /// <summary>
        ///     NumPy's legacy <c>random_hypergeometric_hrua</c>: Stadlober's ratio-of-uniforms (HRUA*) for
        ///     <c>sample &gt; 10</c>, with Frohne's two corrections.
        /// </summary>
        /// <param name="src">The draw source — per-draw, or a read-ahead whose no-overdraw accounting the caller keeps.</param>
        /// <param name="good">Good items.</param>
        /// <param name="bad">Bad items.</param>
        /// <param name="sample">Items drawn (&gt; 10).</param>
        /// <param name="s">The setup of exactly these parameters.</param>
        /// <param name="memo">The fill's loggam-sum memo, or null to evaluate every sum (the scalar path).</param>
        /// <returns>Good items drawn.</returns>
        private long LegacyHypergeometricHrua(ref DrawBufferDouble src, long good, long bad, long sample,
                                              in LegacyHruaSetup s, LoggamSumMemo memo)
        {
            long Z;
            double T, W, X, Y;
            double d6 = s.D6, d8 = s.D8, d10 = s.D10, d11 = s.D11;

            while (true)
            {
                X = src.NextDouble();
                Y = src.NextDouble();
                W = d6 + d8 * (Y - 0.5) / X;

                /* fast rejection: */
                if ((W < 0.0) || (W >= d11))
                    continue;

                Z = Distributions.ToInt64(Math.Floor(W));
                T = d10 - (memo != null ? memo.Get(in s, Z) : s.LoggamSum(Z));

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
                Z = s.M - Z;

            /* another fix from rv.py to allow sample to exceed popsize/2 */
            if (s.M < sample)
                Z = good - Z;

            return Z;
        }

        /// <summary>
        ///     NumPy's <c>legacy_random_hypergeometric</c> (<c>random_hypergeometric_original</c>) for one call: HRUA for
        ///     <c>sample &gt; 10</c>, the urn simulation for <c>1..10</c>, and 0 for a non-positive sample — each with its setup
        ///     evaluated per call, as NumPy does.
        /// </summary>
        /// <param name="src">The draw source — per-draw, or a read-ahead whose no-overdraw accounting the caller keeps.</param>
        /// <param name="good">Good items.</param>
        /// <param name="bad">Bad items.</param>
        /// <param name="sample">Items drawn.</param>
        /// <returns>Good items drawn.</returns>
        private long LegacyHypergeometric(ref DrawBufferDouble src, long good, long bad, long sample)
        {
            if (sample > 10)
                return LegacyHypergeometricHrua(ref src, good, bad, sample, new LegacyHruaSetup(good, bad, sample), null);
            else if (sample > 0)
                return LegacyHypergeometricHyp(ref src, good, bad, sample);
            else
                return 0;
        }

        /// <summary>The setup of <c>legacy_random_zipf</c>: <c>am1 = a - 1</c>, <c>b = 2^am1</c> and the exponent <c>-1 / am1</c>.</summary>
        private readonly struct LegacyZipfSetup
        {
            /// <summary><c>a - 1.0</c>.</summary>
            internal readonly double Am1;

            /// <summary><c>pow(2.0, am1)</c> — infinite for <c>a &gt;= 1025</c>, the input NumPy never returns from.</summary>
            internal readonly double B;

            /// <summary><c>-1.0 / am1</c>, which the C loop recomputes on every attempt.</summary>
            internal readonly double NegInvAm1;

            /// <summary>Evaluates NumPy's per-call statements.</summary>
            /// <param name="a">The validated exponent (<c>&gt; 1</c>).</param>
            internal LegacyZipfSetup(double a)
            {
                Am1 = a - 1.0;
                B = Math.Pow(2.0, Am1);
                NegInvAm1 = -1.0 / Am1;
            }

            /// <summary>Whether values draw — false exactly when <see cref="B"/> overflows (the port then returns 1 without drawing).</summary>
            internal bool Draws => !double.IsPositiveInfinity(B);
        }

        /// <summary>
        ///     NumPy's <c>legacy_random_zipf</c>: rejection from a Pareto envelope over <c>U = 1 - next_double</c>, truncated to
        ///     C <c>long</c> (the pre-<see cref="Generator"/> algorithm, without the modern <c>Umin</c> window).
        /// </summary>
        /// <param name="src">The draw source — per-draw, or a read-ahead whose no-overdraw accounting the caller keeps.</param>
        /// <param name="s">The setup.</param>
        /// <param name="tMemo">
        ///     A fill's memo of <c>T = pow(1 + 1/X, am1)</c> for the small candidates <c>X &lt; tMemo.Length</c> (NaN = not yet
        ///     computed), or null to evaluate every <c>T</c> (the scalar path). <c>T</c> depends only on the integer candidate
        ///     and the setup and is evaluated with NumPy's expression, so a memoized read is bit-identical — and candidates are
        ///     overwhelmingly small, so it removes one of the two <c>pow</c> calls an attempt costs.
        /// </param>
        /// <returns>The draw (<c>&gt;= 1</c>).</returns>
        /// <remarks>
        ///     For <c>a &gt;= 1025</c> (and <c>a = inf</c>) <c>2^(a-1)</c> overflows, every acceptance test compares NaN, and
        ///     NumPy's loop never exits. This port returns 1 there without drawing — the value the distribution degenerates to
        ///     (the modern sampler returns 1 for the same range).
        /// </remarks>
        private long LegacyZipf(ref DrawBufferDouble src, in LegacyZipfSetup s, double[] tMemo)
        {
            double am1 = s.Am1, b = s.B, negInvAm1 = s.NegInvAm1;
            // NumPy hangs here (see remarks): an infinite b makes every candidate's test NaN <= NaN.
            if (double.IsPositiveInfinity(b))
                return 1;
            while (true)
            {
                double T, U, V, X;

                U = 1.0 - src.NextDouble();
                V = src.NextDouble();
                X = Math.Floor(Math.Pow(U, negInvAm1)); // -1.0 / am1
                /*
                 * The real result may be above what can be represented in a signed
                 * long. Since this is a straightforward rejection algorithm, we can
                 * just reject this value. This function then models a Zipf
                 * distribution truncated to sys.maxint.
                 */
                if (X > (double)long.MaxValue || X < 1.0)
                    continue;

                // X is an integer >= 1 here; T is NaN only if never computed (am1 > 0 is finite on this path).
                if (tMemo != null && X < tMemo.Length)
                {
                    T = tMemo[(int)X];
                    if (double.IsNaN(T))
                        tMemo[(int)X] = T = Math.Pow(1.0 + 1.0 / X, am1);
                }
                else
                {
                    T = Math.Pow(1.0 + 1.0 / X, am1);
                }
                if (V * X * (T - 1.0) / (b - 1.0) <= T / b)
                    return Distributions.ToInt64(X);
            }
        }

        /// <summary>The setup of <c>legacy_random_geometric</c>: <c>log(1 - p)</c> for the inversion branch (<c>p &lt; 1/3</c>).</summary>
        private readonly struct LegacyGeometricSetup
        {
            /// <summary>The success probability; selects the branch.</summary>
            internal readonly double P;

            /// <summary><c>log(1 - p)</c> (inversion branch only).</summary>
            internal readonly double LogQ;

            /// <summary>Evaluates the inversion's denominator when that branch runs.</summary>
            /// <param name="p">The validated probability.</param>
            internal LegacyGeometricSetup(double p)
            {
                P = p;
                LogQ = p >= 0.333333333333333333333333 ? 0.0 : Math.Log(1 - p);
            }
        }

        /// <summary>
        ///     NumPy's <c>legacy_random_geometric</c>: the CDF search for <c>p &gt;= 1/3</c>, else the legacy inversion
        ///     <c>ceil(log1p(-U) / log(1 - p))</c> (no INT64_MAX clamp — an overflow reads as C's indefinite integer).
        /// </summary>
        /// <param name="src">The draw source — per-draw, or a read-ahead whose no-overdraw accounting the caller keeps.</param>
        /// <param name="s">The setup.</param>
        /// <returns>The trial of the first success.</returns>
        private long LegacyGeometric(ref DrawBufferDouble src, in LegacyGeometricSetup s)
        {
            if (s.P >= 0.333333333333333333333333)
                return Distributions.RandomGeometricSearch(ref src, s.P);
            return Distributions.ToInt64(Math.Ceiling(Generator.Log1p(-src.NextDouble()) / s.LogQ));
        }

        /// <summary>
        ///     The smallest kappa whose <c>4 * kappa * kappa</c> overflows, <c>2^511</c> (about <c>6.7e153</c>) — the first
        ///     concentration whose legacy envelope is NaN (<see cref="LegacyVonmisesSetup.Draws"/> false: NumPy's rejection
        ///     loop never exits there). A broadcast scan compares against it instead of evaluating the envelope per element.
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     Exact: <c>(4 * k) * k</c> for <c>k = 2^511</c> is <c>2^1024</c>, which overflows; for the double just below,
        ///     <c>k = 2^511 (1 - 2^-53)</c>, it is <c>2^1024 (1 - 2^-52 + 2^-106)</c>, which rounds to the finite
        ///     <c>2^1024 - 2^972</c>; and each rounded product is monotonic in <c>k</c>, so every larger kappa overflows and
        ///     every smaller one does not (a bisection of the doubles lands on the same bits, <c>0x5FE0000000000000</c>).
        ///     </para>
        ///     <para>
        ///     Below it the envelope stays finite: <c>r = 1 + sqrt(1 + 4k^2)</c> is then at most about <c>1.3e154</c>, so
        ///     <c>sqrt(2r)</c> and <c>rho</c> are finite and <c>rho</c> is never 0 for <c>kappa &gt;= 1e-5</c>; below <c>1e-5</c>
        ///     the Taylor branch <c>1/kappa + kappa</c> is finite too.
        ///     </para>
        /// </remarks>
        internal const double LegacyVonmisesOverflowKappa = 6.703903964971299e153;

        /// <summary>The setup of <c>legacy_vonmises</c>: the wrapped-Cauchy envelope parameter <c>s</c>.</summary>
        private readonly struct LegacyVonmisesSetup
        {
            /// <summary>The mode.</summary>
            internal readonly double Mu;

            /// <summary>The concentration.</summary>
            internal readonly double Kappa;

            /// <summary>The envelope (<see cref="LegacyVonmisesEnvelope"/>) for <c>kappa &gt;= 1e-8</c>; 0 otherwise.</summary>
            internal readonly double S;

            /// <summary>Evaluates the envelope when the rejection branch runs.</summary>
            /// <param name="mu">The mode.</param>
            /// <param name="kappa">The validated concentration.</param>
            internal LegacyVonmisesSetup(double mu, double kappa)
            {
                Mu = mu;
                Kappa = kappa;
                S = double.IsNaN(kappa) || kappa < 1e-8 ? 0.0 : LegacyVonmisesEnvelope(kappa);
            }

            /// <summary>
            ///     Whether every value draws — false for a NaN kappa (NaN without drawing) and for an overflowing envelope
            ///     (the limit without drawing).
            /// </summary>
            internal bool Draws => !double.IsNaN(Kappa) && (Kappa < 1e-8 || !double.IsNaN(S));
        }

        /// <summary>
        ///     NumPy's <c>legacy_vonmises</c>: Best &amp; Fisher's wrapped-Cauchy rejection, a uniform on <c>[-pi, pi)</c> for
        ///     <c>kappa &lt; 1e-8</c>, and — unlike the modern sampler — NO wrapped-normal fallback for large kappa.
        /// </summary>
        /// <param name="src">The draw source — per-draw, or a read-ahead whose no-overdraw accounting the caller keeps.</param>
        /// <param name="st">The setup (kappa validated non-negative; NaN returns NumPy's NaN without drawing).</param>
        /// <returns>The draw, in <c>[-pi, pi]</c>.</returns>
        /// <remarks>
        ///     Once <c>4*kappa*kappa</c> overflows (<c>kappa &gt; ~6.7e153</c>, or infinite) the envelope parameter is NaN and
        ///     NumPy's rejection loop never exits; this port returns <c>mu</c> wrapped exactly as an accepted draw would be —
        ///     the distribution's limit as kappa grows — without drawing.
        /// </remarks>
        private double LegacyVonmises(ref DrawBufferDouble src, in LegacyVonmisesSetup st)
        {
            double s;
            double U, V, W, Y, Z;
            double result, mod;
            bool neg;
            double mu = st.Mu, kappa = st.Kappa;

            if (double.IsNaN(kappa))
                return Distributions.NPY_NAN;
            if (kappa < 1e-8)
            {
                return Math.PI * (2 * src.NextDouble() - 1);
            }
            else
            {
                s = st.S;

                if (double.IsNaN(s))
                {
                    // NumPy hangs (see remarks): wrap mu as the accepted branch below wraps `result`.
                    result = mu;
                }
                else
                {
                    while (true)
                    {
                        U = src.NextDouble();
                        Z = Math.Cos(Math.PI * U);
                        W = (1 + s * Z) / (s + Z);
                        Y = kappa * (s - W);
                        V = src.NextDouble();
                        /*
                         * V==0.0 is ok here since Y >= 0 always leads
                         * to accept, while Y < 0 always rejects
                         */
                        if ((Y * (2 - Y) - V >= 0) || (Math.Log(Y / V) + 1 - Y >= 0))
                            break;
                    }

                    U = src.NextDouble();

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
        ///     The wrapped-Cauchy envelope parameter <c>s</c> of <c>legacy_vonmises</c> for <c>kappa &gt;= 1e-8</c>: the second-order
        ///     Taylor form below <c>1e-5</c>, else <c>(1 + rho^2) / (2 rho)</c> — NumPy's statements verbatim.
        /// </summary>
        /// <param name="kappa">The concentration, <c>&gt;= 1e-8</c>.</param>
        /// <returns><c>s</c>; NaN once <c>4*kappa^2</c> overflows (the input NumPy never returns from).</returns>
        private static double LegacyVonmisesEnvelope(double kappa)
        {
            /* with double precision rho is zero until 1.4e-8 */
            if (kappa < 1e-5)
            {
                /*
                 * second order taylor expansion around kappa = 0
                 * precise until relatively large kappas as second order is 0
                 */
                return (1.0 / kappa + kappa);
            }
            /* Path for 1e-5 <= kappa <= 1e6 */
            double r = 1 + Math.Sqrt(1 + 4 * kappa * kappa);
            double rho = (r - Math.Sqrt(2 * r)) / (2 * kappa);
            return (1 + rho * rho) / (2 * rho);
        }

        /// <summary>The setup of <c>legacy_logseries</c>: <c>r = log(1 - p)</c>.</summary>
        private readonly struct LegacyLogseriesSetup
        {
            /// <summary>The shape.</summary>
            internal readonly double P;

            /// <summary><c>log(1.0 - p)</c>, which the C code computes on every call.</summary>
            internal readonly double R;

            /// <summary>Evaluates <c>r</c>.</summary>
            /// <param name="p">The validated shape (<c>[0, 1)</c>).</param>
            internal LegacyLogseriesSetup(double p)
            {
                P = p;
                R = Math.Log(1.0 - p);
            }
        }

        /// <summary>
        ///     NumPy's <c>legacy_logseries</c>: Kemp's second accelerated generator (LK) with the legacy
        ///     <c>log(1 - p)</c> / <c>1 - exp(r*U)</c> spellings (the modern sampler uses <c>log1p</c>/<c>expm1</c>).
        /// </summary>
        /// <param name="src">The draw source — per-draw, or a read-ahead whose no-overdraw accounting the caller keeps.</param>
        /// <param name="s">The setup.</param>
        /// <returns>The draw (<c>&gt;= 1</c>).</returns>
        /// <remarks>
        ///     An overflowing <c>floor(1 + log(V)/log(q))</c> goes through C's integer cast (<see cref="Distributions.ToInt64"/>)
        ///     and so reads as negative and is REJECTED, as in NumPy — a saturating cast would accept a huge count instead.
        /// </remarks>
        private long LegacyLogseries(ref DrawBufferDouble src, in LegacyLogseriesSetup s)
        {
            double q, U, V;
            long result;
            double p = s.P, r = s.R;

            while (true)
            {
                V = src.NextDouble();
                if (V >= p)
                    return 1;
                U = src.NextDouble();
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
