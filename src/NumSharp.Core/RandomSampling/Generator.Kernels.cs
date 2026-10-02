using System;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace NumSharp
{
    /// <summary>
    ///     The per-call setup of NumPy's modern <c>random_beta</c>: which algorithm the shapes select and the constants it
    ///     would otherwise recompute on every value — Johnk's <c>1/a</c> and <c>1/b</c>, or the two gammas' setups.
    /// </summary>
    internal readonly struct BetaSetup
    {
        /// <summary>NumPy's <c>BETA_TINY_THRESHOLD</c>: below it (both shapes) a single uniform decides 0 or 1.</summary>
        internal const double TinyThreshold = 3e-103;

        /// <summary>Alpha.</summary>
        internal readonly double A;

        /// <summary>Beta.</summary>
        internal readonly double B;

        /// <summary>Whether Johnk's algorithm runs (<c>a &lt;= 1 &amp;&amp; b &lt;= 1</c>).</summary>
        internal readonly bool Johnk;

        /// <summary>Whether the tiny-shape shortcut runs (both shapes below <see cref="TinyThreshold"/>, Johnk's range).</summary>
        internal readonly bool Tiny;

        /// <summary>Johnk's <c>1.0 / a</c>.</summary>
        internal readonly double InvA;

        /// <summary>Johnk's <c>1.0 / b</c>.</summary>
        internal readonly double InvB;

        /// <summary>The gamma-ratio branch's setup of <c>a</c>.</summary>
        internal readonly GammaSetup GammaA;

        /// <summary>The gamma-ratio branch's setup of <c>b</c>.</summary>
        internal readonly GammaSetup GammaB;

        /// <summary>Evaluates the selected branch's per-shape terms, in NumPy's branch order.</summary>
        /// <param name="a">Alpha (validated positive or NaN; the Generator's dirichlet may pass 0).</param>
        /// <param name="b">Beta (validated positive or NaN; the Generator's dirichlet may pass 0).</param>
        internal BetaSetup(double a, double b)
        {
            A = a;
            B = b;
            Johnk = (a <= 1.0) && (b <= 1.0);
            Tiny = Johnk && a < TinyThreshold && b < TinyThreshold;
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
    ///     The per-call setup of NumPy's modern <c>random_noncentral_chisquare</c>: the branch's chi-square gamma,
    ///     <c>sqrt(nonc)</c>, or the Poisson mixing mean <c>nonc / 2</c>.
    /// </summary>
    internal readonly struct NoncentralChisquareSetup
    {
        /// <summary>The degrees of freedom.</summary>
        internal readonly double Df;

        /// <summary>The non-centrality (NaN returns NaN without drawing — the modern sampler tests it FIRST).</summary>
        internal readonly double Nonc;

        /// <summary>The chi-square's gamma setup: of <c>df / 2</c> when <c>nonc == 0</c>, of <c>(df - 1) / 2</c> when <c>df &gt; 1</c>.</summary>
        internal readonly GammaSetup Chi;

        /// <summary><c>sqrt(nonc)</c> for the <c>df &gt; 1</c> branch.</summary>
        internal readonly double SqrtNonc;

        /// <summary>The Poisson setup of <c>nonc / 2.0</c> for the <c>df &lt;= 1</c> branch (rejection log hoisted).</summary>
        internal readonly PoissonSetup HalfNonc;

        /// <summary>Evaluates the setup of the branch <paramref name="df"/>/<paramref name="nonc"/> select.</summary>
        /// <param name="df">The degrees of freedom.</param>
        /// <param name="nonc">The non-centrality.</param>
        internal NoncentralChisquareSetup(double df, double nonc)
        {
            Df = df;
            Nonc = nonc;
            Chi = default;
            SqrtNonc = 0.0;
            HalfNonc = default;
            if (double.IsNaN(nonc))
                return;
            if (nonc == 0)
            {
                Chi = new GammaSetup(df / 2.0);
            }
            else if (1 < df)
            {
                // random_chisquare(df - 1) halves its argument: the gamma shape is (df - 1) / 2.0.
                Chi = new GammaSetup((df - 1) / 2.0);
                SqrtNonc = Math.Sqrt(nonc);
            }
            else
            {
                HalfNonc = new PoissonSetup(nonc / 2.0, hoistRejectionLog: true);
            }
        }
    }

    /// <summary>
    ///     The per-call setup of NumPy's modern <c>random_vonmises</c>: the wrapped-Cauchy envelope <c>s</c>, or — above
    ///     <c>kappa = 1e6</c> — the wrapped-normal fallback's <c>sqrt(1 / kappa)</c>.
    /// </summary>
    internal readonly struct VonmisesSetup
    {
        /// <summary>The mode.</summary>
        internal readonly double Mu;

        /// <summary>The concentration (NaN returns NaN without drawing).</summary>
        internal readonly double Kappa;

        /// <summary>The envelope for <c>1e-8 &lt;= kappa &lt;= 1e6</c>.</summary>
        internal readonly double S;

        /// <summary><c>sqrt(1. / kappa)</c> for the wrapped-normal fallback (<c>kappa &gt; 1e6</c>, infinity included).</summary>
        internal readonly double SqrtInvKappa;

        /// <summary>Evaluates NumPy's per-call statements for the branch <paramref name="kappa"/> selects.</summary>
        /// <param name="mu">The mode.</param>
        /// <param name="kappa">The validated concentration.</param>
        internal VonmisesSetup(double mu, double kappa)
        {
            Mu = mu;
            Kappa = kappa;
            S = SqrtInvKappa = 0.0;
            if (double.IsNaN(kappa) || kappa < 1e-8)
                return;
            if (kappa < 1e-5)
            {
                /* second order taylor expansion around kappa = 0 */
                S = (1.0 / kappa + kappa);
            }
            else if (kappa <= 1e6)
            {
                /* Path for 1e-5 <= kappa <= 1e6 */
                double r = 1 + Math.Sqrt(1 + 4 * kappa * kappa);
                double rho = (r - Math.Sqrt(2 * r)) / (2 * kappa);
                S = (1 + rho * rho) / (2 * rho);
            }
            else
            {
                SqrtInvKappa = Math.Sqrt(1.0 / kappa);
            }
        }
    }

    /// <summary>
    ///     The per-call setup of NumPy's modern <c>random_zipf</c>: <c>am1 = a - 1</c>, <c>b = 2^am1</c>, the sampling
    ///     window's lower end <c>Umin = INT64_MAX^-am1</c> and the exponent <c>-1 / am1</c>.
    /// </summary>
    internal readonly struct ZipfSetup
    {
        /// <summary>Whether NumPy's own <c>a &gt;= 1025</c> guard applies (returns 1 without drawing).</summary>
        internal readonly bool Degenerate;

        /// <summary><c>a - 1.0</c>.</summary>
        internal readonly double Am1;

        /// <summary><c>pow(2.0, am1)</c>.</summary>
        internal readonly double B;

        /// <summary><c>pow((double)INT64_MAX, -am1)</c>: candidates are drawn from <c>(Umin, 1]</c> so none overflows int64.</summary>
        internal readonly double Umin;

        /// <summary><c>-1.0 / am1</c>, which the C loop recomputes per attempt.</summary>
        internal readonly double NegInvAm1;

        /// <summary>Evaluates NumPy's per-call statements.</summary>
        /// <param name="a">The validated exponent (<c>&gt; 1</c>).</param>
        internal ZipfSetup(double a)
        {
            Degenerate = a >= 1025;
            Am1 = B = Umin = NegInvAm1 = 0.0;
            if (Degenerate)
                return;
            Am1 = a - 1.0;
            B = Math.Pow(2.0, Am1);
            Umin = Math.Pow((double)long.MaxValue, -Am1);
            NegInvAm1 = -1.0 / Am1;
        }
    }

    /// <summary>
    ///     The per-parameter setup of NumPy's modern <c>hypergeometric_hrua</c> — everything its loop reads that does not
    ///     depend on the draws — evaluated once per fill instead of on every value.
    /// </summary>
    /// <remarks>
    ///     NumPy recomputes these on every call (a <c>sqrt</c>, four <c>logfactorial</c>s, a <c>floor</c> and a handful of
    ///     divisions); each is a deterministic function of <c>(good, bad, sample)</c> under the same IEEE operations, so the
    ///     stored values are NumPy's bits. The statements are NumPy's, in NumPy's order.
    /// </remarks>
    internal readonly struct HruaSetup
    {
        /// <summary>Good items.</summary>
        internal readonly long Good;

        /// <summary>Bad items.</summary>
        internal readonly long Bad;

        /// <summary>Items drawn.</summary>
        internal readonly long Sample;

        /// <summary><c>MIN(sample, popsize - sample)</c> (Stadlober's <c>n</c>).</summary>
        internal readonly long ComputedSample;

        /// <summary><c>MIN(good, bad)</c> (Stadlober's <c>M</c>).</summary>
        internal readonly long MinGoodBad;

        /// <summary><c>MAX(good, bad)</c>.</summary>
        internal readonly long MaxGoodBad;

        /// <summary><c>mu + 0.5</c>, the table mountain's centre.</summary>
        internal readonly double A;

        /// <summary><c>D1*c + D2</c>, twice the table mountain's scale.</summary>
        internal readonly double H;

        /// <summary>The mode's four-<c>logfactorial</c> sum.</summary>
        internal readonly double G;

        /// <summary>The exclusive upper bound of the candidates: <c>MIN(MIN(computed_sample, mingoodbad) + 1, floor(a + 16*c))</c>.</summary>
        internal readonly double B;

        /// <summary>Evaluates NumPy's per-call statements for the parameters.</summary>
        /// <param name="good">Good items (the caller routed <c>10 &lt;= sample &lt;= good + bad - 10</c> here).</param>
        /// <param name="bad">Bad items.</param>
        /// <param name="sample">Items drawn.</param>
        internal HruaSetup(long good, long bad, long sample)
        {
            /* D1 = 2*sqrt(2/e), D2 = 3 - 2*sqrt(3/e) */
            const double D1 = 1.7155277699214135;
            const double D2 = 0.8989161620588988;

            Good = good;
            Bad = bad;
            Sample = sample;
            long popsize = good + bad;
            ComputedSample = sample < popsize - sample ? sample : popsize - sample;
            MinGoodBad = good < bad ? good : bad;
            MaxGoodBad = good > bad ? good : bad;

            double p = ((double)MinGoodBad) / popsize;
            double q = ((double)MaxGoodBad) / popsize;

            // mu is the mean of the distribution.
            double mu = ComputedSample * p;
            A = mu + 0.5;

            // var is the variance of the distribution.
            double var = ((double)(popsize - ComputedSample) * ComputedSample * p * q / (popsize - 1));

            double c = Math.Sqrt(var + 0.5);

            // h is 2*s_hat (Stadlober's "table mountain" scale).
            H = D1 * c + D2;

            long m = (long)Math.Floor((double)(ComputedSample + 1) * (MinGoodBad + 1) / (popsize + 2));

            G = (Generator.LogFactorial(m) +
                 Generator.LogFactorial(MinGoodBad - m) +
                 Generator.LogFactorial(ComputedSample - m) +
                 Generator.LogFactorial(MaxGoodBad - ComputedSample + m));

            // b is the upper bound for random samples: C's MIN(MIN(computed_sample, mingoodbad) + 1, floor(a + 16*c)).
            double lhs = (ComputedSample < MinGoodBad ? ComputedSample : MinGoodBad) + 1;
            double rhs = Math.Floor(A + 16 * c);
            B = lhs < rhs ? lhs : rhs;
        }

        /// <summary>
        ///     A NaN-initialized <c>gp</c> memo for a fill of <paramref name="count"/> values, or null when it would not pay
        ///     (a short fill) or would be too large (more than 2^16 candidates).
        /// </summary>
        /// <param name="count">The values the fill draws.</param>
        /// <returns>The memo, or null.</returns>
        internal double[] NewGpMemo(long count)
        {
            if (count < 16 || !(B <= 65536))
                return null;
            var memo = new double[(int)B];
            Array.Fill(memo, double.NaN);
            return memo;
        }
    }

    public sealed partial class Generator
    {
        // =============================================================================================================
        // expm1 and logfactorial
        // =============================================================================================================

        /// <summary><c>-ln 2</c>: below it <c>expm1(x) &lt; -0.5</c> and ucrtbase computes <c>exp(x) - 1</c>.</summary>
        private const double Expm1BandLow = -0.6931471805599453;

        /// <summary><c>ln 1.5</c>: above it <c>expm1(x) &gt; 0.5</c> and ucrtbase computes <c>exp(x) - 1</c>.</summary>
        private const double Expm1BandHigh = 0.4054651081081644;

        /// <summary>
        ///     <c>expm1(x) = e^x - 1</c> for NumPy's modern samplers (<c>pareto</c>, <c>power</c>, <c>logseries</c>), which
        ///     call the C runtime's <c>expm1</c> (<c>npy_expm1</c>).
        /// </summary>
        /// <param name="x">The argument.</param>
        /// <returns><c>e^x - 1</c>.</returns>
        /// <remarks>
        ///     <para>
        ///     Measured against ucrtbase (NumPy's win-amd64 runtime): OUTSIDE <c>[-ln 2, ln 1.5]</c> — where
        ///     <c>|expm1(x)| &gt;= 0.5</c>, so the subtraction loses nothing — ucrtbase returns exactly <c>exp(x) - 1</c>
        ///     (0 mismatches over millions of inputs), and so does this port, bit for bit. INSIDE the band ucrtbase uses an
        ///     undocumented approximation that no public formula reproduces (not correctly rounded: about a third of its
        ///     results sit one ulp or more from the true value; Cephes, FreeBSD/Sun, Kahan's (with <c>log</c> and
        ///     <c>log1p</c>), Newton-on-<c>log1p</c>, Taylor/Horner, <c>sinh</c>/<c>tanh</c>/<c>exp2</c> forms, x87
        ///     <c>F2XM1</c> roundings and AMD's libm all tested and rejected). There this port runs Sun's <c>s_expm1.c</c>,
        ///     which agrees with ucrtbase on about 58% of in-band inputs and is never more than 2 ulp away (measured over
        ///     millions of inputs).
        ///     </para>
        ///     <para>
        ///     Linux NumPy calls glibc's <c>expm1</c> — the same Sun algorithm, but built with FMA on x86-64 — which
        ///     differs from this transcription on ~0.3% of in-band inputs and from <c>exp(x) - 1</c> on ~2-10% of
        ///     out-of-band ones (glibc 2.39). The port favors win-amd64, the project's parity reference. .NET's own
        ///     <c>double.ExpM1</c> is a plain <c>Exp(x) - 1</c> and loses all precision near 0, so it is not used.
        ///     </para>
        /// </remarks>
        internal static double Expm1(double x)
        {
            if (x < Expm1BandLow || x > Expm1BandHigh)
                return Math.Exp(x) - 1.0; // ucrtbase's own formula outside the band (NaN falls through: NaN - 1 = NaN)
            return Expm1Sun(x);
        }

        /// <summary>Sun's (FreeBSD msun / glibc) <c>s_expm1.c</c>, transcribed statement by statement.</summary>
        /// <param name="x">The argument (called for the band <c>[-ln 2, ln 1.5]</c>, but valid everywhere).</param>
        /// <returns><c>e^x - 1</c>, within one ulp.</returns>
        private static double Expm1Sun(double x)
        {
            const double o_threshold = 7.09782712893383973096e+02;
            const double ln2_hi = 6.93147180369123816490e-01;
            const double ln2_lo = 1.90821492927058770002e-10;
            const double invln2 = 1.44269504088896338700e+00;
            const double Q1 = -3.33333333333331316428e-02;
            const double Q2 = 1.58730158725481460165e-03;
            const double Q3 = -7.93650757867487942473e-05;
            const double Q4 = 4.00821782732936239552e-06;
            const double Q5 = -2.01099218183624371326e-07;

            double y, hi, lo, c = 0, t, e, hxs, hfx, r1, twopk;
            int k;
            long bits = BitConverter.DoubleToInt64Bits(x);
            uint hx = (uint)(bits >> 32);
            bool negative = (hx & 0x80000000) != 0;
            hx &= 0x7fffffff;

            /* filter out huge and non-finite argument */
            if (hx >= 0x4043687A)
            {
                if (hx >= 0x40862E42)
                {
                    if (hx >= 0x7ff00000)
                    {
                        uint low = (uint)bits;
                        if (((hx & 0xfffff) | low) != 0)
                            return x + x; /* NaN */
                        return negative ? -1.0 : x; /* exp(+-inf) - 1 = {inf, -1} */
                    }
                    if (x > o_threshold)
                        return double.PositiveInfinity; /* overflow */
                }
                if (negative && x + 1e-300 < 0.0)
                    return 1e-300 - 1.0; /* x < -56*ln2: return -1.0 with inexact */
            }

            /* argument reduction */
            if (hx > 0x3fd62e42)
            {
                /* if |x| > 0.5 ln2 */
                if (hx < 0x3FF0A2B2)
                {
                    /* and |x| < 1.5 ln2 */
                    if (!negative)
                    {
                        hi = x - ln2_hi;
                        lo = ln2_lo;
                        k = 1;
                    }
                    else
                    {
                        hi = x + ln2_hi;
                        lo = -ln2_lo;
                        k = -1;
                    }
                }
                else
                {
                    k = (int)(invln2 * x + (!negative ? 0.5 : -0.5));
                    t = k;
                    hi = x - t * ln2_hi; /* t*ln2_hi is exact here */
                    lo = t * ln2_lo;
                }
                x = hi - lo;
                c = (hi - x) - lo;
            }
            else if (hx < 0x3c900000)
            {
                /* when |x| < 2**-54, return x */
                t = 1e300 + x; /* return x with inexact flags when x != 0 */
                return x - (t - (1e300 + x));
            }
            else
            {
                k = 0;
            }

            /* x is now in primary range */
            hfx = 0.5 * x;
            hxs = x * hfx;
            r1 = 1.0 + hxs * (Q1 + hxs * (Q2 + hxs * (Q3 + hxs * (Q4 + hxs * Q5))));
            t = 3.0 - r1 * hfx;
            e = hxs * ((r1 - t) / (6.0 - x * t));
            if (k == 0)
                return x - (x * e - hxs); /* c is 0 */
            twopk = BitConverter.Int64BitsToDouble((long)(0x3ff + k) << 52);
            e = (x * (e - c) - c);
            e -= hxs;
            if (k == -1)
                return 0.5 * (x - e) - 0.5;
            if (k == 1)
            {
                if (x < -0.25)
                    return -2.0 * (e - (x + 0.5));
                return 1.0 + 2.0 * (x - e);
            }
            if (k <= -2 || k > 56)
            {
                /* suffice to return exp(x) - 1 */
                y = 1.0 - (e - x);
                if (k == 1024)
                    y = y * 2.0 * BitConverter.Int64BitsToDouble(0x7fe0000000000000);
                else
                    y = y * twopk;
                return y - 1.0;
            }
            if (k < 20)
            {
                t = BitConverter.Int64BitsToDouble((long)(0x3ff - (0x200000 >> k)) << 32); /* t = 1 - 2^-k */
                y = t - (e - x);
                y = y * twopk;
            }
            else
            {
                t = BitConverter.Int64BitsToDouble((long)((0x3ff - k) << 20) << 32); /* 2^-k */
                y = x - (e + t);
                y += 1.0;
                y = y * twopk;
            }
            return y;
        }

        /// <summary>
        ///     NumPy's <c>logfactorial(k)</c> (<c>logfactorial.c</c>): <c>log(k!)</c> from a 126-entry table, else the
        ///     Stirling series truncated at the <c>1/k^3</c> term. Used by the modern hypergeometric samplers.
        /// </summary>
        /// <param name="k">The non-negative count.</param>
        /// <returns><c>ln(k!)</c>, as NumPy rounds it.</returns>
        internal static double LogFactorial(long k)
        {
            const double halfln2pi = 0.9189385332046728;
            if (k < LogFact.Length)
                return LogFact[k];
            return (k + 0.5) * Math.Log((double)k) - k + (halfln2pi + (1.0 / k) * (1 / 12.0 - 1 / (360.0 * k * k)));
        }

        /// <summary>NumPy's <c>logfact[]</c>: <c>log(k!)</c> for <c>k = 0..125</c>, the literals verbatim.</summary>
        private static readonly double[] LogFact =
        {
            0.0, 0.0, 0.69314718055994529, 1.791759469228055,
            3.1780538303479458, 4.7874917427820458, 6.5792512120101012, 8.5251613610654147,
            10.604602902745251, 12.801827480081469, 15.104412573075516, 17.502307845873887,
            19.987214495661885, 22.552163853123425, 25.19122118273868, 27.89927138384089,
            30.671860106080672, 33.505073450136891, 36.395445208033053, 39.339884187199495,
            42.335616460753485, 45.380138898476908, 48.471181351835227, 51.606675567764377,
            54.784729398112319, 58.003605222980518, 61.261701761002001, 64.557538627006338,
            67.88974313718154, 71.257038967168015, 74.658236348830158, 78.092223553315307,
            81.557959456115043, 85.054467017581516, 88.580827542197682, 92.136175603687093,
            95.719694542143202, 99.330612454787428, 102.96819861451381, 106.63176026064346,
            110.32063971475739, 114.03421178146171, 117.77188139974507, 121.53308151543864,
            125.3172711493569, 129.12393363912722, 132.95257503561632, 136.80272263732635,
            140.67392364823425, 144.5657439463449, 148.47776695177302, 152.40959258449735,
            156.3608363030788, 160.3311282166309, 164.32011226319517, 168.32744544842765,
            172.35279713916279, 176.39584840699735, 180.45629141754378, 184.53382886144948,
            188.6281734236716, 192.7390472878449, 196.86618167289001, 201.00931639928152,
            205.1681994826412, 209.34258675253685, 213.53224149456327, 217.73693411395422,
            221.95644181913033, 226.1905483237276, 230.43904356577696, 234.70172344281826,
            238.97838956183432, 243.26884900298271, 247.57291409618688, 251.89040220972319,
            256.22113555000954, 260.56494097186322, 264.92164979855278, 269.29109765101981,
            273.67312428569369, 278.06757344036612, 282.4742926876304, 286.89313329542699,
            291.32395009427029, 295.76660135076065, 300.22094864701415, 304.68685676566872,
            309.1641935801469, 313.65282994987905, 318.1526396202093, 322.66349912672615,
            327.1852877037752, 331.71788719692847, 336.26118197919845, 340.81505887079902,
            345.37940706226686, 349.95411804077025, 354.53908551944079, 359.1342053695754,
            363.73937555556347, 368.35449607240474, 372.97946888568902, 377.61419787391867,
            382.25858877306001, 386.91254912321756, 391.57598821732961, 396.24881705179155,
            400.93094827891576, 405.6222961611449, 410.32277652693733, 415.03230672824964,
            419.75080559954472, 424.47819341825709, 429.21439186665157, 433.95932399501481,
            438.71291418612117, 443.47508812091894, 448.24577274538461, 453.02489623849613,
            457.81238798127816, 462.60817852687489, 467.4121995716082, 472.22438392698058,
            477.04466549258564, 481.87297922988796,
        };

        // =============================================================================================================
        // Modern kernels over a DrawBuffer64 (NumPy's distributions.c, the Generator build)
        // =============================================================================================================
        //
        // Every kernel takes its draws from a DrawBuffer64 exactly as the C code takes them from bitgen_t: 64-bit units
        // for the ziggurat samplers, next_double for the uniforms (one unit each). A one-unit buffer IS the per-draw call
        // sequence; a fill hands in a read-ahead whose Owed it keeps at a lower bound of the units still to come. Each
        // expression keeps NumPy's operand order (see Distributions for the byte-parity rules). Callers hold the lock.

        /// <summary>
        ///     NumPy's modern <c>random_standard_gamma</c>: the ziggurat exponential for <c>shape == 1</c>, 0 (no draw) for
        ///     <c>shape == 0</c>, Johnk/Ahrens-Dieter below 1, Marsaglia-Tsang over ziggurat normals above.
        /// </summary>
        /// <param name="src">The draw source.</param>
        /// <param name="g">The shape's setup.</param>
        /// <returns>The draw.</returns>
        internal static double StandardGamma(ref DrawBuffer64 src, in GammaSetup g)
        {
            double b, c;
            double U, V, X, Y;
            double shape = g.Shape;

            if (shape == 1.0)
            {
                return StandardExponential(ref src);
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
                    V = StandardExponential(ref src);
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
                b = g.B;
                c = g.C;
                for (;;)
                {
                    do
                    {
                        X = StandardNormal(ref src);
                        V = 1.0 + c * X;
                    } while (V <= 0.0);

                    V = V * V * V;
                    U = src.NextDouble();
                    if (U < 1.0 - 0.0331 * (X * X) * (X * X))
                        return (b * V);
                    /* log(0.0) ok here */
                    if (Math.Log(U) < 0.5 * X * X + b * (1.0 - V + Math.Log(V)))
                        return (b * V);
                }
            }
        }

        /// <summary>
        ///     NumPy's modern <c>random_beta</c>: the tiny-shape shortcut (one uniform decides 0 or 1), Johnk's algorithm
        ///     with the log-space fallback when a power underflows, or the ratio of two gammas.
        /// </summary>
        /// <param name="src">The draw source.</param>
        /// <param name="s">The setup.</param>
        /// <returns>The draw.</returns>
        internal static double Beta(ref DrawBuffer64 src, in BetaSetup s)
        {
            if (s.Johnk)
            {
                double a = s.A, b = s.B;
                double U, V, X, Y, XpY;
                if (s.Tiny)
                {
                    /* the proportion a/(a + b) and a single uniform decide the (0 or 1) result */
                    U = src.NextDouble();
                    return (a + b) * U < a ? 1.0 : 0.0;
                }

                /* Use Johnk's algorithm */
                while (true)
                {
                    U = src.NextDouble();
                    V = src.NextDouble();
                    X = Math.Pow(U, s.InvA); // 1.0 / a
                    Y = Math.Pow(V, s.InvB); // 1.0 / b
                    XpY = X + Y;
                    /* Reject if both U and V are 0.0, which is approx 1 in 10^106 */
                    if ((XpY <= 1.0) && (U + V > 0.0))
                    {
                        if ((X > 0) && (Y > 0))
                            return X / XpY;
                        /* a power underflowed: work with logarithms (npy_log1p is the CRT's, reproduced by Log1p) */
                        double logX = Math.Log(U) / a;
                        double logY = Math.Log(V) / b;
                        double delta = logX - logY;
                        if (delta > 0)
                            return Math.Exp(-Log1p(Math.Exp(-delta)));
                        return Math.Exp(delta - Log1p(Math.Exp(delta)));
                    }
                }
            }

            double Ga = StandardGamma(ref src, in s.GammaA);
            double Gb = StandardGamma(ref src, in s.GammaB);
            return Ga / (Ga + Gb);
        }

        /// <summary>NumPy's modern <c>random_chisquare</c>: <c>2 * random_standard_gamma(df / 2)</c>.</summary>
        /// <param name="src">The draw source.</param>
        /// <param name="halfDf">The setup of <c>df / 2.0</c>.</param>
        /// <returns>The draw.</returns>
        internal static double Chisquare(ref DrawBuffer64 src, in GammaSetup halfDf) => 2.0 * StandardGamma(ref src, in halfDf);

        /// <summary>NumPy's modern <c>random_f</c>: <c>(chi2(dfnum) * dfden) / (chi2(dfden) * dfnum)</c>, numerator first.</summary>
        /// <param name="src">The draw source.</param>
        /// <param name="halfNum">The setup of <c>dfnum / 2</c>.</param>
        /// <param name="halfDen">The setup of <c>dfden / 2</c>.</param>
        /// <param name="dfnum">The numerator degrees of freedom.</param>
        /// <param name="dfden">The denominator degrees of freedom.</param>
        /// <returns>The draw.</returns>
        internal static double F(ref DrawBuffer64 src, in GammaSetup halfNum, in GammaSetup halfDen, double dfnum, double dfden)
        {
            double subexpr1 = Chisquare(ref src, in halfNum) * dfden;
            double subexpr2 = Chisquare(ref src, in halfDen) * dfnum;
            return subexpr1 / subexpr2;
        }

        /// <summary>NumPy's modern <c>random_standard_cauchy</c>: the ratio of two ziggurat normals, numerator first.</summary>
        /// <param name="src">The draw source.</param>
        /// <returns>The draw.</returns>
        internal static double StandardCauchy(ref DrawBuffer64 src)
        {
            double subexpr1 = StandardNormal(ref src);
            double subexpr2 = StandardNormal(ref src);
            return subexpr1 / subexpr2;
        }

        /// <summary>NumPy's modern <c>random_standard_t</c>: <c>sqrt(df/2) * N / sqrt(G(df/2))</c>, the normal drawn first.</summary>
        /// <param name="src">The draw source.</param>
        /// <param name="half">The setup of <c>df / 2</c>.</param>
        /// <param name="sqrtHalf"><c>sqrt(df / 2)</c>.</param>
        /// <returns>The draw.</returns>
        internal static double StandardT(ref DrawBuffer64 src, in GammaSetup half, double sqrtHalf)
        {
            double num = StandardNormal(ref src);
            double denom = StandardGamma(ref src, in half);
            return sqrtHalf * num / Math.Sqrt(denom);
        }

        /// <summary>
        ///     NumPy's modern <c>random_noncentral_chisquare</c>: NaN for a NaN non-centrality (checked FIRST, no draw), a
        ///     plain chi-square for <c>nonc == 0</c>, a <c>df - 1</c> chi-square plus a shifted squared normal for
        ///     <c>df &gt; 1</c>, else a Poisson-mixed chi-square.
        /// </summary>
        /// <param name="src">The draw source.</param>
        /// <param name="s">The setup.</param>
        /// <returns>The draw.</returns>
        internal static double NoncentralChisquare(ref DrawBuffer64 src, in NoncentralChisquareSetup s)
        {
            if (double.IsNaN(s.Nonc))
                return Distributions.NPY_NAN;
            if (s.Nonc == 0)
                return Chisquare(ref src, in s.Chi);
            if (1 < s.Df)
            {
                double Chi2 = Chisquare(ref src, in s.Chi);
                double n = StandardNormal(ref src) + s.SqrtNonc;
                return Chi2 + n * n;
            }
            long i = Poisson(ref src, in s.HalfNonc);
            // The chi-square's df depends on the Poisson draw, so its setup is per value — NumPy's arithmetic.
            return 2.0 * StandardGamma(ref src, new GammaSetup((s.Df + 2 * i) / 2.0));
        }

        /// <summary>NumPy's modern <c>random_noncentral_f</c>: <c>(ncchi2(dfnum, nonc) * dfden) / (chi2(dfden) * dfnum)</c>.</summary>
        /// <param name="src">The draw source.</param>
        /// <param name="num">The numerator's setup.</param>
        /// <param name="halfDen">The setup of <c>dfden / 2</c>.</param>
        /// <param name="dfnum">The numerator degrees of freedom.</param>
        /// <param name="dfden">The denominator degrees of freedom.</param>
        /// <returns>The draw.</returns>
        internal static double NoncentralF(ref DrawBuffer64 src, in NoncentralChisquareSetup num, in GammaSetup halfDen,
                                           double dfnum, double dfden)
        {
            double t = NoncentralChisquare(ref src, in num) * dfden;
            return t / (Chisquare(ref src, in halfDen) * dfnum);
        }

        /// <summary>
        ///     NumPy's modern <c>random_wald</c>: Michael/Schucany/Haas in the cancellation-free form
        ///     <c>X = mean * (1 - 2 / (1 + sqrt(1 + 4*scale/Y)))</c>, one normal and one uniform per value.
        /// </summary>
        /// <param name="src">The draw source.</param>
        /// <param name="mean">The mean.</param>
        /// <param name="scale">The scale.</param>
        /// <returns>The draw.</returns>
        internal static double Wald(ref DrawBuffer64 src, double mean, double scale)
        {
            double U, X, Y;
            double d;

            Y = StandardNormal(ref src);
            Y = mean * Y * Y;
            d = 1 + Math.Sqrt(1 + 4 * scale / Y);
            X = mean * (1 - 2 / d);
            U = src.NextDouble();
            if (U <= mean / (mean + X))
                return X;
            return mean * mean / X;
        }

        /// <summary>
        ///     NumPy's modern <c>random_vonmises</c>: NaN without drawing for a NaN kappa, a uniform on <c>[-pi, pi)</c>
        ///     below <c>1e-8</c>, Best &amp; Fisher's wrapped-Cauchy rejection up to <c>1e6</c>, and the wrapped normal
        ///     <c>mu + sqrt(1/kappa) * N</c> above (which also makes <c>kappa = inf</c> return <c>mu</c>).
        /// </summary>
        /// <param name="src">The draw source.</param>
        /// <param name="st">The setup.</param>
        /// <returns>The draw.</returns>
        internal static double Vonmises(ref DrawBuffer64 src, in VonmisesSetup st)
        {
            double s;
            double U, V, W, Y, Z;
            double result, mod;
            bool neg;
            double mu = st.Mu, kappa = st.Kappa;

            if (double.IsNaN(kappa))
                return Distributions.NPY_NAN;
            if (kappa < 1e-8)
                return Math.PI * (2 * src.NextDouble() - 1);
            if (kappa > 1e6)
            {
                /* Fallback to wrapped normal distribution for kappa > 1e6 */
                result = mu + st.SqrtInvKappa * StandardNormal(ref src);
                /* Ensure result is within bounds */
                if (result < -Math.PI)
                    result += 2 * Math.PI;
                if (result > Math.PI)
                    result -= 2 * Math.PI;
                return result;
            }

            s = st.S;
            while (true)
            {
                U = src.NextDouble();
                Z = Math.Cos(Math.PI * U);
                W = (1 + s * Z) / (s + Z);
                Y = kappa * (s - W);
                V = src.NextDouble();
                /* V==0.0 is ok here since Y >= 0 always leads to accept, while Y < 0 always rejects */
                if ((Y * (2 - Y) - V >= 0) || (Math.Log(Y / V) + 1 - Y >= 0))
                    break;
            }

            U = src.NextDouble();

            result = Math.Acos(W);
            if (U < 0.5)
                result = -result;
            result += mu;
            neg = (result < 0);
            mod = Math.Abs(result);
            // C's fmod: C# `%` on doubles is the same exact truncated remainder for every finite value. A NaN
            // (mu = NaN) goes through untouched: .NET 8's `%` returns the default NaN (sign bit SET) for a NaN
            // dividend where C's fmod — NumPy — and .NET 10 propagate the input NaN, so the NaN's bits match on
            // every runtime.
            double shifted = mod + Math.PI;
            mod = (double.IsNaN(shifted) ? shifted : shifted % (2 * Math.PI)) - Math.PI;
            if (neg)
                mod *= -1;
            return mod;
        }

        /// <summary>
        ///     NumPy's <c>random_poisson</c> over a <see cref="DrawBuffer64"/> with the setup built once (the noncentral
        ///     chi-square's fixed <c>nonc / 2</c>).
        /// </summary>
        /// <param name="src">The draw source.</param>
        /// <param name="s">The mean's setup.</param>
        /// <returns>The count.</returns>
        /// <remarks>
        ///     The twin of <see cref="Distributions.RandomPoisson(ref DrawBufferDouble, in PoissonSetup)"/> for samplers that
        ///     also draw ziggurat words: their stream is 64-bit units, which a <see cref="DrawBufferDouble"/> cannot carry.
        /// </remarks>
        internal static long Poisson(ref DrawBuffer64 src, in PoissonSetup s)
        {
            if (s.Lam >= 10)
                return PoissonPtrs(ref src, s.Lam, s.Loglam, s.A, s.B, s.Invalpha, s.Vr, s.LogInvalpha);
            if (s.Lam == 0)
                return 0;
            return PoissonMult(ref src, s.Enlam);
        }

        /// <summary>
        ///     NumPy's <c>random_poisson</c> over a <see cref="DrawBuffer64"/> for a mean that changes per call (the negative
        ///     binomial's gamma-drawn mean): the setup in locals, exactly NumPy's per-call arithmetic, no struct.
        /// </summary>
        /// <param name="src">The draw source.</param>
        /// <param name="lam">The mean.</param>
        /// <returns>The count.</returns>
        internal static long Poisson(ref DrawBuffer64 src, double lam)
        {
            if (lam >= 10)
            {
                Distributions.PtrsSetup(lam, out _, out double loglam, out double b, out double a, out double invalpha, out double vr);
                return PoissonPtrs(ref src, lam, loglam, a, b, invalpha, vr, double.NaN);
            }
            if (lam == 0)
                return 0;
            return PoissonMult(ref src, Math.Exp(-lam));
        }

        /// <summary>NumPy's <c>random_poisson_mult</c> over a <see cref="DrawBuffer64"/> (see <see cref="Distributions.RandomPoissonMult"/>).</summary>
        /// <param name="src">The draw source.</param>
        /// <param name="enlam"><c>exp(-lam)</c>.</param>
        /// <returns>The count.</returns>
        private static long PoissonMult(ref DrawBuffer64 src, double enlam)
        {
            long X = 0;
            double prod = 1.0;
            while (true)
            {
                double U = src.NextDouble();
                prod *= U;
                if (prod > enlam)
                    X += 1;
                else
                    return X;
            }
        }

        /// <summary>NumPy's <c>random_poisson_ptrs</c> trial loop over a <see cref="DrawBuffer64"/> (see <see cref="Distributions.RandomPoissonPtrs"/>).</summary>
        /// <param name="src">The draw source.</param>
        /// <param name="lam">The mean, <c>&gt;= 10</c>.</param>
        /// <param name="loglam"><c>log(lam)</c>.</param>
        /// <param name="a">The envelope's <c>a</c>.</param>
        /// <param name="b">The envelope's <c>b</c>.</param>
        /// <param name="invalpha">The envelope's <c>invalpha</c>.</param>
        /// <param name="vr">The squeeze bound.</param>
        /// <param name="logInvalpha">A hoisted <c>log(invalpha)</c>, or NaN to compute it per rejection trial.</param>
        /// <returns>The count.</returns>
        private static long PoissonPtrs(ref DrawBuffer64 src, double lam, double loglam, double a, double b, double invalpha,
                                        double vr, double logInvalpha)
        {
            while (true)
            {
                double U = src.NextDouble() - 0.5;
                double V = src.NextDouble();
                double us = 0.5 - Math.Abs(U);
                long k = Distributions.ToInt64(Math.Floor((2 * a / us + b) * U + lam + 0.43));
                if ((us >= 0.07) && (V <= vr))
                    return k;
                if ((k < 0) || ((us < 0.013) && (V > us)))
                    continue;
                double li = double.IsNaN(logInvalpha) ? Math.Log(invalpha) : logInvalpha;
                if ((Math.Log(V) + li - Math.Log(a / (us * us) + b)) <=
                    (-lam + (double)k * loglam - Distributions.RandomLoggam((double)k + 1)))
                    return k;
            }
        }

        /// <summary>
        ///     NumPy's modern <c>random_negative_binomial</c>: a Poisson whose mean is a gamma draw with scale
        ///     <c>(1 - p) / p</c>.
        /// </summary>
        /// <param name="src">The draw source.</param>
        /// <param name="gamma">The setup of the gamma shape <c>n</c>.</param>
        /// <param name="scale"><c>(1 - p) / p</c>.</param>
        /// <returns>The count of failures.</returns>
        internal static long NegativeBinomial(ref DrawBuffer64 src, in GammaSetup gamma, double scale)
        {
            double Y = scale * StandardGamma(ref src, in gamma);
            return Poisson(ref src, Y);
        }

        /// <summary>
        ///     NumPy's modern <c>random_geometric_inversion</c> (the <c>p &lt; 1/3</c> branch):
        ///     <c>ceil(-E / log1p(-p))</c> over a ziggurat exponential, clamped to <c>INT64_MAX</c>.
        /// </summary>
        /// <param name="src">The draw source.</param>
        /// <param name="log1pNegP">The setup's <c>npy_log1p(-p)</c>, which the C code recomputes per value.</param>
        /// <returns>The trial of the first success.</returns>
        internal static long GeometricInversion(ref DrawBuffer64 src, double log1pNegP)
        {
            double z = Math.Ceiling(-StandardExponential(ref src) / log1pNegP);
            /* 9.223372036854776e+18 is the smallest double that is larger than INT64_MAX. */
            if (z >= 9.223372036854776e+18)
                return long.MaxValue;
            return (long)z;
        }

        // =============================================================================================================
        // Modern next_double-only kernels (they run over a DrawBufferDouble, like the shared distributions.c ports)
        // =============================================================================================================

        /// <summary>
        ///     NumPy's modern <c>random_zipf</c>: rejection from a Pareto envelope with the candidate drawn from
        ///     <c>U in (Umin, 1]</c> so it never overflows int64; NumPy's own guard returns 1 (no draw) for <c>a &gt;= 1025</c>.
        /// </summary>
        /// <param name="src">The draw source.</param>
        /// <param name="s">The setup.</param>
        /// <param name="tMemo">A fill's memo of <c>pow(1 + 1/X, am1)</c> for the small candidates (NaN = not yet computed), or
        ///     null to evaluate every one (the scalar path) — bit-neutral, see the legacy <c>LegacyZipf</c>.</param>
        /// <returns>The draw (<c>&gt;= 1</c>).</returns>
        internal static long Zipf(ref DrawBufferDouble src, in ZipfSetup s, double[] tMemo)
        {
            if (s.Degenerate)
                return 1;
            double am1 = s.Am1, b = s.B, Umin = s.Umin, negInvAm1 = s.NegInvAm1;
            while (true)
            {
                double U01, T, U, V, X;

                /* U is sampled from (Umin, 1]. Note that Umin might be 0, and we don't want U to be 0. */
                U01 = src.NextDouble();
                U = U01 * Umin + (1 - U01);
                V = src.NextDouble();
                X = Math.Floor(Math.Pow(U, negInvAm1)); // -1.0 / am1
                if (X > (double)long.MaxValue || X < 1.0)
                    continue;

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
                    return (long)X;
            }
        }

        /// <summary>
        ///     NumPy's modern <c>random_logseries</c>: Kemp's LK with <c>r = log1p(-p)</c> and <c>q = -expm1(r*U)</c>.
        /// </summary>
        /// <param name="src">The draw source.</param>
        /// <param name="p">The shape (validated to <c>[0, 1)</c>).</param>
        /// <param name="r">The setup's <c>npy_log1p(-p)</c>.</param>
        /// <returns>The draw (<c>&gt;= 1</c>).</returns>
        /// <remarks>
        ///     <c>q</c> goes through <see cref="Expm1"/>, so inside <c>r*U in [-ln 2, 0]</c> it can sit one ulp from Windows
        ///     NumPy's (see there); that only moves a count when <c>V</c> lands within that ulp of a boundary.
        /// </remarks>
        internal static long Logseries(ref DrawBufferDouble src, double p, double r)
        {
            while (true)
            {
                double V = src.NextDouble();
                if (V >= p)
                    return 1;
                double U = src.NextDouble();
                double q = -Expm1(r * U);
                if (V <= q * q)
                {
                    long result = Distributions.ToInt64(Math.Floor(1 + Math.Log(V) / Math.Log(q)));
                    if ((result < 1) || (V == 0.0))
                        continue;
                    return result;
                }
                if (V >= q)
                    return 1;
                return 2;
            }
        }

        // =============================================================================================================
        // Modern hypergeometric (random_hypergeometric.c)
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>random_interval</c> for a <paramref name="max"/> that fits 32 bits, over a 32-bit read-ahead: the
        ///     smallest all-ones mask covering <paramref name="max"/>, then masked <c>next_uint32</c> words until one lands
        ///     inside — the same words, in the same order, as <see cref="BoundedIntegers.RandomInterval"/> draws.
        /// </summary>
        /// <param name="src">The 32-bit draw source (a read-ahead, or capacity 1 for per-draw calls).</param>
        /// <param name="max">The inclusive upper bound, <c>0 &lt;= max &lt;= 0xFFFFFFFF</c> (the caller guarantees it).</param>
        /// <returns>A value in <c>[0, max]</c>; 0 without drawing when <paramref name="max"/> is 0.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static ulong RandomInterval(ref DrawBuffer32 src, ulong max)
        {
            if (max == 0)
                return 0;
            ulong mask = BoundedIntegers.GenMask(max);
            ulong value;
            while ((value = src.NextUInt32() & mask) > max) { }
            return value;
        }

        /// <summary>
        ///     NumPy's <c>hypergeometric_sample</c> over a 32-bit read-ahead — the urn walk of the <c>hypergeometric</c> fill,
        ///     where every value's words are <c>next_uint32</c> draws and no other kind of draw interleaves them.
        /// </summary>
        /// <param name="src">The 32-bit draw source.</param>
        /// <param name="good">Good items (below <c>10^9</c>, so every <c>random_interval</c> bound fits 32 bits).</param>
        /// <param name="bad">Bad items (below <c>10^9</c>).</param>
        /// <param name="sample">Items drawn.</param>
        /// <returns>Good items drawn.</returns>
        /// <remarks>
        ///     Each loop iteration draws at least one word: the loop runs only while <c>remaining_total &gt;
        ///     remaining_good &gt; 0</c>, so the decremented bound is at least 1 and <see cref="RandomInterval(ref DrawBuffer32, ulong)"/>
        ///     draws. A value therefore draws at least one word exactly when <c>sample &gt; 0</c>, <c>good &gt; 0</c> and
        ///     <c>bad &gt; 0</c> (after NumPy's complement), which is what makes a whole fill all-or-nothing.
        /// </remarks>
        internal static long HypergeometricSample(ref DrawBuffer32 src, long good, long bad, long sample)
        {
            long remaining_total, remaining_good, result, computed_sample;
            long total = good + bad;

            if (sample > total / 2)
                computed_sample = total - sample;
            else
                computed_sample = sample;

            remaining_total = total;
            remaining_good = good;

            while ((computed_sample > 0) && (remaining_good > 0) && (remaining_total > remaining_good))
            {
                // random_interval(max) is inclusive of max, so remaining_total is decremented first.
                --remaining_total;
                if ((long)RandomInterval(ref src, (ulong)remaining_total) < remaining_good)
                {
                    // Selected a "good" one, so decrement remaining_good.
                    --remaining_good;
                }
                --computed_sample;
            }

            if (remaining_total == remaining_good)
            {
                // Only "good" choices are left.
                remaining_good -= computed_sample;
            }

            if (sample > total / 2)
                result = remaining_good;
            else
                result = good - remaining_good;

            return result;
        }

        /// <summary>
        ///     Fills <paramref name="count"/> values of NumPy's <c>hypergeometric_sample</c> (the urn walk) for one parameter
        ///     set: the same words, comparisons and results as <paramref name="count"/> calls of
        ///     <see cref="HypergeometricSample(ref DrawBuffer32, long, long, long)"/>, with the 32-bit read-ahead's cursor kept
        ///     in locals.
        /// </summary>
        /// <param name="bg">The bit generator (lock held by the caller).</param>
        /// <param name="dst">The int64 output (<paramref name="count"/> slots).</param>
        /// <param name="count">The values to draw.</param>
        /// <param name="good">Good items (below <c>10^9</c>: every bound fits 32 bits).</param>
        /// <param name="bad">Bad items (below <c>10^9</c>).</param>
        /// <param name="sample">Items drawn per value.</param>
        /// <remarks>
        ///     <para>
        ///     The walk is a chain of tiny steps — a masked word, a rejection test, a comparison — so where the words live
        ///     decides its speed: a <see cref="DrawBuffer32"/> passed by reference keeps its position in memory, and every
        ///     word then waits on a store-to-load round trip. Here the position, the fill count and the buffer are locals.
        ///     The refill rule is the read-ahead's (at most the values still owed, at least one — see
        ///     <see cref="HypergeometricSample(ref DrawBuffer32, long, long, long)"/> for why every value draws at least one
        ///     word or none does), so the stream is NumPy's word for word.
        ///     </para>
        ///     <para>
        ///     Two spellings differ from the C and change no result: <c>gen_mask</c> becomes one <c>lzcnt</c> (the bound is at
        ///     least 1 inside the loop), and <c>if (value &lt; remaining_good) --remaining_good</c> becomes a branch-free
        ///     subtraction of the comparison — the comparison is a coin flip, so as a branch it mispredicts half the time.
        ///     </para>
        /// </remarks>
        internal static unsafe void HypergeometricSampleFill(BitGenerator bg, long* dst, long count, long good, long bad, long sample)
        {
            long total = good + bad;
            bool complement = sample > total / 2;
            long computed0 = complement ? total - sample : sample;

            uint* buf = stackalloc uint[DrawBuffer32.Capacity];
            int pos = 0, avail = 0;
            for (long i = 0; i < count; i++)
            {
                long remaining_total = total, remaining_good = good, computed_sample = computed0;
                while ((computed_sample > 0) && (remaining_good > 0) && (remaining_total > remaining_good))
                {
                    // random_interval(max) is inclusive of max, so remaining_total is decremented first; it stays >= 1
                    // because it exceeded remaining_good >= 1.
                    --remaining_total;
                    ulong max = (ulong)remaining_total;
                    ulong mask = ulong.MaxValue >> BitOperations.LeadingZeroCount(max);
                    ulong value;
                    do
                    {
                        if (pos == avail)
                        {
                            // This value still draws and every later one draws at least once: count - i words are owed.
                            long owed = count - i;
                            avail = owed < DrawBuffer32.Capacity ? (int)owed : DrawBuffer32.Capacity;
                            bg.FillUInt32(buf, avail);
                            pos = 0;
                        }
                        value = buf[pos++] & mask;
                    } while (value > max);
                    // Selected a "good" one: decrement remaining_good (branch-free; value <= max < 2^31, so the signed C
                    // comparison and this unsigned one agree).
                    remaining_good -= value < (ulong)remaining_good ? 1L : 0L;
                    --computed_sample;
                }

                if (remaining_total == remaining_good)
                {
                    // Only "good" choices are left.
                    remaining_good -= computed_sample;
                }

                dst[i] = complement ? remaining_good : good - remaining_good;
            }
        }

        /// <summary>
        ///     NumPy's <c>hypergeometric_sample</c>: an urn simulation over <c>random_interval</c> draws (the small-sample
        ///     path), drawn straight from the bit generator — for callers that interleave it with 64-bit draws (the
        ///     multivariate marginals) or draw one value, where a 32-bit read-ahead would reorder the stream.
        /// </summary>
        /// <param name="bg">The bit generator (lock held by the caller).</param>
        /// <param name="good">Good items.</param>
        /// <param name="bad">Bad items.</param>
        /// <param name="sample">Items drawn.</param>
        /// <returns>Good items drawn.</returns>
        internal static long HypergeometricSample(BitGenerator bg, long good, long bad, long sample)
        {
            long remaining_total, remaining_good, result, computed_sample;
            long total = good + bad;

            if (sample > total / 2)
                computed_sample = total - sample;
            else
                computed_sample = sample;

            remaining_total = total;
            remaining_good = good;

            while ((computed_sample > 0) && (remaining_good > 0) && (remaining_total > remaining_good))
            {
                // random_interval(max) is inclusive of max, so remaining_total is decremented first.
                --remaining_total;
                if ((long)BoundedIntegers.RandomInterval(bg, (ulong)remaining_total) < remaining_good)
                {
                    // Selected a "good" one, so decrement remaining_good.
                    --remaining_good;
                }
                --computed_sample;
            }

            if (remaining_total == remaining_good)
            {
                // Only "good" choices are left.
                remaining_good -= computed_sample;
            }

            if (sample > total / 2)
                result = remaining_good;
            else
                result = good - remaining_good;

            return result;
        }

        /// <summary>
        ///     NumPy's modern <c>hypergeometric_hrua</c> (Stadlober's ratio-of-uniforms with <c>logfactorial</c>), over a
        ///     <see cref="DrawBufferDouble"/> — its only draws are <c>next_double</c>.
        /// </summary>
        /// <param name="src">The draw source.</param>
        /// <param name="good">Good items.</param>
        /// <param name="bad">Bad items.</param>
        /// <param name="sample">Items drawn.</param>
        /// <returns>Good items drawn.</returns>
        internal static long HypergeometricHrua(ref DrawBufferDouble src, long good, long bad, long sample)
        {
            var setup = new HruaSetup(good, bad, sample);
            return HypergeometricHrua(ref src, in setup, null);
        }

        /// <summary>
        ///     NumPy's modern <c>hypergeometric_hrua</c> loop over a precomputed <see cref="HruaSetup"/>, optionally reading
        ///     each candidate's <c>gp</c> (its four-<c>logfactorial</c> sum) from a per-fill memo.
        /// </summary>
        /// <param name="src">The draw source.</param>
        /// <param name="s">The parameters' setup (NumPy's per-call statements, evaluated once).</param>
        /// <param name="gpMemo">Null, or a NaN-initialized array of at least <c>s.B</c> slots: slot <c>K</c> caches
        ///     candidate <c>K</c>'s <c>gp</c> after its first evaluation.</param>
        /// <returns>Good items drawn.</returns>
        /// <remarks>
        ///     <c>gp</c> depends only on the parameters and <c>K</c>, and a miss evaluates NumPy's expression in NumPy's
        ///     order, so a memoized read is bit-identical — the fill pays each distinct candidate's four table lookups (or
        ///     Stirling <c>log</c>s past 125) once instead of on every visit. A NaN slot means "not yet": <c>gp</c> itself is a
        ///     sum of finite <c>logfactorial</c>s (every argument is non-negative for <c>0 &lt;= K &lt; B</c>), never NaN.
        /// </remarks>
        internal static long HypergeometricHrua(ref DrawBufferDouble src, in HruaSetup s, double[] gpMemo)
        {
            long computed_sample = s.ComputedSample, mingoodbad = s.MinGoodBad, maxgoodbad = s.MaxGoodBad;
            double a = s.A, h = s.H, g = s.G, b = s.B;
            long K;

            while (true)
            {
                double U, V, X, T;
                double gp;

                U = src.NextDouble();
                V = src.NextDouble(); // "U star" in Stadlober (1989)
                X = a + h * (V - 0.5) / U;

                // fast rejection:
                if ((X < 0.0) || (X >= b))
                    continue;

                K = (long)Math.Floor(X);

                if (gpMemo == null || double.IsNaN(gp = gpMemo[K]))
                {
                    gp = (LogFactorial(K) +
                          LogFactorial(mingoodbad - K) +
                          LogFactorial(computed_sample - K) +
                          LogFactorial(maxgoodbad - computed_sample + K));
                    if (gpMemo != null)
                        gpMemo[K] = gp;
                }

                T = g - gp;

                // fast acceptance:
                if ((U * (4.0 - U) - 3.0) <= T)
                    break;

                // fast rejection:
                if (U * (U - T) >= 1)
                    continue;

                if (2.0 * Math.Log(U) <= T)
                    break; // acceptance
            }

            if (s.Good > s.Bad)
                K = computed_sample - K;

            if (computed_sample < s.Sample)
                K = s.Good - K;

            return K;
        }

        /// <summary>
        ///     NumPy's modern <c>random_hypergeometric</c>: HRUA for <c>10 &lt;= sample &lt;= good + bad - 10</c>, the urn
        ///     simulation otherwise.
        /// </summary>
        /// <param name="src">A per-draw (capacity-1) buffer over <paramref name="bg"/> for HRUA's doubles — a read-ahead
        ///     would reorder them against the urn path's 32-bit draws when the two alternate (multivariate marginals).</param>
        /// <param name="bg">The bit generator, for the urn path's <c>random_interval</c>.</param>
        /// <param name="good">Good items.</param>
        /// <param name="bad">Bad items.</param>
        /// <param name="sample">Items drawn.</param>
        /// <returns>Good items drawn.</returns>
        internal static long Hypergeometric(ref DrawBufferDouble src, BitGenerator bg, long good, long bad, long sample)
        {
            if ((sample >= 10) && (sample <= good + bad - 10))
                return HypergeometricHrua(ref src, good, bad, sample);
            return HypergeometricSample(bg, good, bad, sample);
        }
    }
}
