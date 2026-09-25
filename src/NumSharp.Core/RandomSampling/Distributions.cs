using System;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace NumSharp
{
    /// <summary>
    ///     NumPy's <c>binomial_t</c> — the per-generator cache of the binomial samplers' setup (the BTPE constants, or the
    ///     inversion's <c>q</c>/<c>q^n</c>/bound), keyed on the last <c>(n, p)</c> drawn.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The cache is observable, which is why it is a shared object rather than recomputed per call: NumPy's legacy
    ///     <c>RandomState</c> owns ONE of these and lends it to both <c>binomial</c> (legacy inversion, <c>exp(n*log(q))</c>)
    ///     and <c>multinomial</c> (the modern inversion, <c>exp(n*log1p(-p))</c>). Both write the same fields under the same
    ///     <c>(n, p)</c> key, so a <c>multinomial</c> call can leave a <c>q^n</c> that a later <c>binomial</c> with a matching
    ///     key reuses instead of recomputing — a one-ULP-level difference NumPy's stream carries and this port reproduces.
    ///     </para>
    ///     <para>
    ///     <see cref="m"/> means the inversion's search bound for the inversion sampler and the mode for BTPE; the key
    ///     <c>(n, p)</c> decides the algorithm, so one key never sees both meanings. Nothing resets the cache — not
    ///     re-seeding, not <c>set_state</c> — exactly as in NumPy. Fields keep NumPy's names for a line-by-line comparison
    ///     with <c>distributions.c</c>. Access is serialized by the owning bit generator's lock.
    ///     </para>
    /// </remarks>
    internal sealed class BinomialState
    {
        /// <summary>Whether the fields below hold a setup (NumPy's <c>has_binomial</c>).</summary>
        internal bool has_binomial;

        /// <summary>The cached key's probability.</summary>
        internal double psave;

        /// <summary>The cached key's trial count.</summary>
        internal long nsave;

        /// <summary>BTPE: <c>min(p, 1-p)</c>; inversion: <c>q^n</c>.</summary>
        internal double r;

        /// <summary><c>1 - r</c> (BTPE) or <c>1 - p</c> (inversion).</summary>
        internal double q;

        /// <summary>BTPE: <c>n*r + r</c>.</summary>
        internal double fm;

        /// <summary>BTPE: the mode <c>floor(fm)</c>; inversion: the search bound.</summary>
        internal long m;

        /// <summary>BTPE: half-width of the triangular region.</summary>
        internal double p1;

        /// <summary>BTPE: <c>m + 0.5</c>.</summary>
        internal double xm;

        /// <summary>BTPE: left edge of the triangle.</summary>
        internal double xl;

        /// <summary>BTPE: right edge of the triangle.</summary>
        internal double xr;

        /// <summary>BTPE: parallelogram height; inversion: <c>n*p</c>.</summary>
        internal double c;

        /// <summary>BTPE: left exponential tail rate.</summary>
        internal double laml;

        /// <summary>BTPE: right exponential tail rate.</summary>
        internal double lamr;

        /// <summary>BTPE: cumulative area through the parallelograms.</summary>
        internal double p2;

        /// <summary>BTPE: cumulative area through the left tail.</summary>
        internal double p3;

        /// <summary>BTPE: total area (the scale of the region selector).</summary>
        internal double p4;
    }

    /// <summary>
    ///     A line-by-line port of the scalar samplers in <c>numpy/random/src/distributions/distributions.c</c> that NumPy's
    ///     legacy <c>RandomState</c> and its <see cref="Generator"/> share, drawing from a <see cref="BitGenerator"/>'s
    ///     <c>next_double</c> exactly as the C code draws from <c>bitgen_t</c>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Byte-parity rules followed throughout: every expression keeps NumPy's operand ORDER and parenthesisation (float
    ///     addition is not associative, so <c>a + b + c</c> stays <c>(a + b) + c</c>); integer sub-expressions stay integer
    ///     until C would convert them (e.g. BTPE's <c>-k * k</c> is an integer product); each draw is taken in the order the C
    ///     statements take it; and a C conversion of an out-of-range double to an integer goes through <see cref="ToInt64"/>,
    ///     never a bare cast (see there for why).
    ///     </para>
    ///     <para>
    ///     <c>RAND_INT_TYPE</c> is C <c>long</c> in NumPy's legacy build and <c>int64_t</c> in the Generator build; NumSharp's
    ///     legacy integer samplers return int64 (the LP64 shape — see <see cref="RandomConstraints.LegacyPoissonLamMax"/>), so
    ///     both builds are the int64 code here. Callers hold the bit generator's lock.
    ///     </para>
    /// </remarks>
    internal static class Distributions
    {
        /// <summary>
        ///     NumPy's positive quiet NaN (<c>NPY_NAN</c>, bits <c>0x7ff8000000000000</c>). The samplers that return
        ///     <c>NPY_NAN</c> use this rather than <see cref="double.NaN"/>, whose bits are <c>0xfff8000000000000</c> on .NET.
        /// </summary>
        internal static readonly double NPY_NAN = BitConverter.Int64BitsToDouble(0x7FF8000000000000);

        /// <summary>
        ///     C's <c>(int64_t)x</c> as x86-64 executes it (<c>cvttsd2si</c>): truncation toward zero, and the "integer
        ///     indefinite" value <see cref="long.MinValue"/> for NaN and for anything outside <c>[-2^63, 2^63)</c>.
        /// </summary>
        /// <param name="x">The double to convert.</param>
        /// <returns>The truncated value, or <see cref="long.MinValue"/> when it is not representable.</returns>
        /// <remarks>
        ///     A bare C# <c>(long)x</c> is NOT a substitute: out-of-range conversions are unspecified in C#, and the runtime
        ///     changed them between the two target frameworks (.NET 8 on x64 returns <see cref="long.MinValue"/>, .NET 9+
        ///     saturates, and NaN becomes 0). NumPy's samplers feed out-of-range doubles to such casts on purpose — e.g.
        ///     <c>legacy_logseries</c> relies on an overflowed <c>floor(1 + log(V)/log(q))</c> reading as a negative count and
        ///     being rejected — so the conversion must be pinned for both frameworks to reproduce one stream.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static long ToInt64(double x)
        {
            // -2^63 and 2^63 are both exact doubles; NaN fails both comparisons and lands on the indefinite value.
            if (x >= -9223372036854775808.0 && x < 9223372036854775808.0)
                return (long)x;
            return long.MinValue;
        }

        /// <summary>
        ///     NumPy's <c>kahan_sum</c> (<c>_common.pyx</c>): a compensated left-to-right sum of <paramref name="n"/> doubles —
        ///     the sum <c>multinomial</c> and weighted <c>choice</c> validate probabilities with.
        /// </summary>
        /// <param name="darr">The values.</param>
        /// <param name="n">How many to add; <c>n &lt;= 0</c> sums to 0 without touching <paramref name="darr"/>.</param>
        /// <returns>The compensated sum.</returns>
        internal static unsafe double KahanSum(double* darr, long n)
        {
            double c, y, t, sum;
            if (n <= 0)
                return 0.0;
            sum = darr[0];
            c = 0.0;
            for (long i = 1; i < n; i++)
            {
                y = darr[i] - c;
                t = sum + y;
                c = (t - sum) - y;
                sum = t;
            }
            return sum;
        }

        /// <summary>
        ///     NumPy's <c>random_loggam</c>: <c>log(Gamma(x))</c> by the Stirling series with the argument shifted up to 7.
        /// </summary>
        /// <param name="x">The argument (the samplers pass <c>k + 1</c> for a count <c>k &gt;= 0</c>).</param>
        /// <returns><c>ln Γ(x)</c>, exactly as NumPy rounds it.</returns>
        internal static double RandomLoggam(double x)
        {
            double x0, x2, lg2pi, gl, gl0;
            long k, n;

            if ((x == 1.0) || (x == 2.0))
                return 0.0;
            else if (x < 7.0)
                n = ToInt64(7 - x);
            else
                n = 0;

            x0 = x + n;
            x2 = (1.0 / x0) * (1.0 / x0);
            /* log(2 * M_PI) */
            lg2pi = 1.8378770664093453e+00;
            gl0 = LoggamCoefficients[9];
            // Separate multiply and add (no FMA): the C statements are two, and MSVC compiles them as two.
            for (k = 8; k >= 0; k--)
            {
                gl0 *= x2;
                gl0 += LoggamCoefficients[k];
            }
            gl = gl0 / x0 + 0.5 * lg2pi + (x0 - 0.5) * Math.Log(x0) - x0;
            if (x < 7.0)
            {
                for (k = 1; k <= n; k++)
                {
                    gl -= Math.Log(x0 - 1.0);
                    x0 -= 1.0;
                }
            }
            return gl;
        }

        /// <summary>The Stirling-series coefficients of <see cref="RandomLoggam"/> (NumPy's static <c>a[10]</c>).</summary>
        private static readonly double[] LoggamCoefficients =
        {
            8.333333333333333e-02, -2.777777777777778e-03,
            7.936507936507937e-04, -5.952380952380952e-04,
            8.417508417508418e-04, -1.917526917526918e-03,
            6.410256410256410e-03, -2.955065359477124e-02,
            1.796443723688307e-01, -1.39243221690590e+00,
        };

        /// <summary>
        ///     NumPy's <c>random_poisson_mult</c>: multiply uniforms until the product drops to <c>exp(-lam)</c> (the
        ///     <c>lam &lt; 10</c> branch).
        /// </summary>
        /// <param name="bg">The draw source.</param>
        /// <param name="lam">The mean (NaN yields 0 after one draw, since no product exceeds a NaN bound).</param>
        /// <returns>The count.</returns>
        internal static long RandomPoissonMult(BitGenerator bg, double lam)
        {
            long X;
            double prod, U, enlam;

            enlam = Math.Exp(-lam);
            X = 0;
            prod = 1.0;
            while (true)
            {
                U = bg.NextDouble();
                prod *= U;
                if (prod > enlam)
                    X += 1;
                else
                    return X;
            }
        }

        /// <summary>
        ///     NumPy's <c>random_poisson_ptrs</c>: Hörmann's transformed rejection with squeeze (PTRS, the <c>lam &gt;= 10</c>
        ///     branch), two draws per trial.
        /// </summary>
        /// <param name="bg">The draw source.</param>
        /// <param name="lam">The mean, <c>&gt;= 10</c>.</param>
        /// <returns>The count.</returns>
        internal static long RandomPoissonPtrs(BitGenerator bg, double lam)
        {
            long k;
            double U, V, slam, loglam, a, b, invalpha, vr, us;

            slam = Math.Sqrt(lam);
            loglam = Math.Log(lam);
            b = 0.931 + 2.53 * slam;
            a = -0.059 + 0.02483 * b;
            invalpha = 1.1239 + 1.1328 / (b - 3.4);
            vr = 0.9277 - 3.6224 / (b - 2);

            while (true)
            {
                U = bg.NextDouble() - 0.5;
                V = bg.NextDouble();
                us = 0.5 - Math.Abs(U);
                // C's (RAND_INT_TYPE)floor(...): us == 0 sends the argument to +-inf/NaN, which must read as a negative
                // count and be rejected below, not saturate to a huge accepted value.
                k = ToInt64(Math.Floor((2 * a / us + b) * U + lam + 0.43));
                if ((us >= 0.07) && (V <= vr))
                    return k;
                if ((k < 0) || ((us < 0.013) && (V > us)))
                    continue;
                /* log(V) == log(0.0) ok here */
                /* if U==0.0 so that us==0.0, log is ok since always returns */
                if ((Math.Log(V) + Math.Log(invalpha) - Math.Log(a / (us * us) + b)) <=
                    (-lam + (double)k * loglam - RandomLoggam((double)k + 1)))
                    return k;
            }
        }

        /// <summary>
        ///     NumPy's <c>random_poisson</c>: PTRS for <c>lam &gt;= 10</c>, 0 (no draw) for <c>lam == 0</c>, else the product
        ///     method.
        /// </summary>
        /// <param name="bg">The draw source.</param>
        /// <param name="lam">The mean (validated by the caller).</param>
        /// <returns>The count.</returns>
        internal static long RandomPoisson(BitGenerator bg, double lam)
        {
            if (lam >= 10)
                return RandomPoissonPtrs(bg, lam);
            else if (lam == 0)
                return 0;
            else
                return RandomPoissonMult(bg, lam);
        }

        /// <summary>
        ///     NumPy's <c>random_binomial_btpe</c>: Kachitvichyanukul &amp; Schmeiser's BTPE rejection sampler for
        ///     <c>n*min(p,1-p) &gt; 30</c>, with its setup cached in <paramref name="binomial"/>.
        /// </summary>
        /// <param name="bg">The draw source.</param>
        /// <param name="n">The trial count.</param>
        /// <param name="p">The success probability (every NumPy caller passes <c>p &lt;= 0.5</c>).</param>
        /// <param name="binomial">The cache (see <see cref="BinomialState"/>).</param>
        /// <returns>The count of successes.</returns>
        /// <remarks>The <c>goto</c> structure is NumPy's, kept so each step can be compared with the C label it ports.</remarks>
        internal static long RandomBinomialBtpe(BitGenerator bg, long n, double p, BinomialState binomial)
        {
            double r, q, fm, p1, xm, xl, xr, c, laml, lamr, p2, p3, p4;
            double a, u, v, s, F, rho, t, A, nrq, x1, x2, f1, f2, z, z2, w, w2, x;
            long m, y, k, i;

            if (!binomial.has_binomial || (binomial.nsave != n) || (binomial.psave != p))
            {
                /* initialize */
                binomial.nsave = n;
                binomial.psave = p;
                binomial.has_binomial = true;
                // C's MIN macro, `p < 1-p ? p : 1-p`, spelled out (Math.Min differs on the signed-zero/NaN corners).
                binomial.r = r = p < 1.0 - p ? p : 1.0 - p;
                binomial.q = q = 1.0 - r;
                binomial.fm = fm = n * r + r;
                binomial.m = m = ToInt64(Math.Floor(binomial.fm));
                binomial.p1 = p1 = Math.Floor(2.195 * Math.Sqrt(n * r * q) - 4.6 * q) + 0.5;
                binomial.xm = xm = m + 0.5;
                binomial.xl = xl = xm - p1;
                binomial.xr = xr = xm + p1;
                binomial.c = c = 0.134 + 20.5 / (15.3 + m);
                a = (fm - xl) / (fm - xl * r);
                binomial.laml = laml = a * (1.0 + a / 2.0);
                a = (xr - fm) / (xr * q);
                binomial.lamr = lamr = a * (1.0 + a / 2.0);
                binomial.p2 = p2 = p1 * (1.0 + 2.0 * c);
                binomial.p3 = p3 = p2 + c / laml;
                binomial.p4 = p4 = p3 + c / lamr;
            }
            else
            {
                r = binomial.r;
                q = binomial.q;
                fm = binomial.fm;
                m = binomial.m;
                p1 = binomial.p1;
                xm = binomial.xm;
                xl = binomial.xl;
                xr = binomial.xr;
                c = binomial.c;
                laml = binomial.laml;
                lamr = binomial.lamr;
                p2 = binomial.p2;
                p3 = binomial.p3;
                p4 = binomial.p4;
            }

        Step10:
            nrq = n * r * q;
            u = bg.NextDouble() * p4;
            v = bg.NextDouble();
            if (u > p1)
                goto Step20;
            y = ToInt64(Math.Floor(xm - p1 * v + u));
            goto Step60;

        Step20:
            if (u > p2)
                goto Step30;
            x = xl + (u - p1) / c;
            v = v * c + 1.0 - Math.Abs(m - x + 0.5) / p1;
            if (v > 1.0)
                goto Step10;
            y = ToInt64(Math.Floor(x));
            goto Step50;

        Step30:
            if (u > p3)
                goto Step40;
            y = ToInt64(Math.Floor(xl + Math.Log(v) / laml));
            /* Reject if v==0.0 since previous cast is undefined */
            if ((y < 0) || (v == 0.0))
                goto Step10;
            v = v * (u - p2) * laml;
            goto Step50;

        Step40:
            y = ToInt64(Math.Floor(xr - Math.Log(v) / lamr));
            /* Reject if v==0.0 since previous cast is undefined */
            if ((y > n) || (v == 0.0))
                goto Step10;
            v = v * (u - p3) * lamr;

        Step50:
            // llabs as two's-complement negation (Math.Abs(long) would throw on long.MinValue, which llabs does not).
            k = y - m;
            if (k < 0)
                k = -k;
            if ((k > 20) && (k < ((nrq) / 2.0 - 1)))
                goto Step52;

            s = r / q;
            a = s * (n + 1);
            F = 1.0;
            if (m < y)
            {
                for (i = m + 1; i <= y; i++)
                    F *= (a / i - s);
            }
            else if (m > y)
            {
                for (i = y + 1; i <= m; i++)
                    F /= (a / i - s);
            }
            if (v > F)
                goto Step10;
            goto Step60;

        Step52:
            rho = (k / (nrq)) * ((k * (k / 3.0 + 0.625) + 0.16666666666666666) / nrq + 0.5);
            // C's `-k * k` is an INTEGER product, converted to double only for the division.
            t = -k * k / (2 * nrq);
            /* log(0.0) ok here */
            A = Math.Log(v);
            if (A < (t - rho))
                goto Step60;
            if (A > (t + rho))
                goto Step10;

            x1 = (double)y + 1;
            f1 = (double)m + 1;
            z = (double)n + 1 - (double)m;
            w = (double)n - (double)y + 1;
            x2 = x1 * x1;
            f2 = f1 * f1;
            z2 = z * z;
            w2 = w * w;
            if (A > (xm * Math.Log(f1 / x1) + (n - m + 0.5) * Math.Log(z / w) +
                     (y - m) * Math.Log(w * r / (x1 * q)) +
                     (13680.0 - (462.0 - (132.0 - (99.0 - 140.0 / f2) / f2) / f2) / f2) / f1 / 166320.0 +
                     (13680.0 - (462.0 - (132.0 - (99.0 - 140.0 / z2) / z2) / z2) / z2) / z / 166320.0 +
                     (13680.0 - (462.0 - (132.0 - (99.0 - 140.0 / x2) / x2) / x2) / x2) / x1 / 166320.0 +
                     (13680.0 - (462.0 - (132.0 - (99.0 - 140.0 / w2) / w2) / w2) / w2) / w / 166320.0))
            {
                goto Step10;
            }

        Step60:
            if (p > 0.5)
                y = n - y;

            return y;
        }

        /// <summary>
        ///     NumPy's (modern) <c>random_binomial_inversion</c>: sequential CDF search, <c>q^n</c> computed as
        ///     <c>exp(n * log1p(-p))</c> — the variant <see cref="RandomBinomial"/> (and so <c>multinomial</c>) uses.
        /// </summary>
        /// <param name="bg">The draw source.</param>
        /// <param name="n">The trial count.</param>
        /// <param name="p">The success probability (<c>&lt;= 0.5</c> from every caller).</param>
        /// <param name="binomial">The cache (see <see cref="BinomialState"/>).</param>
        /// <returns>The count of successes.</returns>
        /// <remarks>
        ///     The legacy <c>RandomState.binomial</c> uses a copy that computes <c>exp(n * log(q))</c> instead
        ///     (<c>NumPyRandom.LegacyBinomialInversion</c>); the two share the cache, see <see cref="BinomialState"/>.
        /// </remarks>
        internal static long RandomBinomialInversion(BitGenerator bg, long n, double p, BinomialState binomial)
        {
            double q, qn, np, px, U;
            long X, bound;

            if (!binomial.has_binomial || (binomial.nsave != n) || (binomial.psave != p))
            {
                binomial.nsave = n;
                binomial.psave = p;
                binomial.has_binomial = true;
                binomial.q = q = 1.0 - p;
                binomial.r = qn = Math.Exp(n * Generator.Log1p(-p));
                binomial.c = np = n * p;
                // C's MIN macro — `n < y ? n : y` in double — then the (RAND_INT_TYPE) cast.
                double limit = np + 10.0 * Math.Sqrt(np * q + 1);
                binomial.m = bound = ToInt64(n < limit ? n : limit);
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
            U = bg.NextDouble();
            while (U > px)
            {
                X++;
                if (X > bound)
                {
                    X = 0;
                    px = qn;
                    U = bg.NextDouble();
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
        ///     NumPy's (modern) <c>random_binomial</c>: 0 with no draw for <c>n == 0</c> or <c>p == 0</c>, else inversion for
        ///     <c>n*min(p,1-p) &lt;= 30</c> and BTPE above, mirrored through <c>1-p</c> when <c>p &gt; 0.5</c>.
        /// </summary>
        /// <param name="bg">The draw source.</param>
        /// <param name="p">The success probability.</param>
        /// <param name="n">The trial count.</param>
        /// <param name="binomial">The cache (see <see cref="BinomialState"/>).</param>
        /// <returns>The count of successes.</returns>
        internal static long RandomBinomial(BitGenerator bg, double p, long n, BinomialState binomial)
        {
            double q;

            // NumPy compares against the FLOAT literal 0.0f; zero is zero in either width.
            if ((n == 0L) || (p == 0.0f))
                return 0;

            if (p <= 0.5)
            {
                if (p * n <= 30.0)
                    return RandomBinomialInversion(bg, n, p, binomial);
                else
                    return RandomBinomialBtpe(bg, n, p, binomial);
            }
            else
            {
                q = 1.0 - p;
                if (q * n <= 30.0)
                    return n - RandomBinomialInversion(bg, n, q, binomial);
                else
                    return n - RandomBinomialBtpe(bg, n, q, binomial);
            }
        }

        /// <summary>
        ///     NumPy's <c>random_multinomial</c>: one conditional binomial per category, the last category taking whatever is
        ///     left. Writes one row of counts.
        /// </summary>
        /// <typeparam name="TCount">The output count type (C <c>long</c> in the legacy build, <c>int64_t</c> in Generator).</typeparam>
        /// <param name="bg">The draw source.</param>
        /// <param name="n">The number of trials.</param>
        /// <param name="mnix">The row to write — its <paramref name="d"/> slots must be zero on entry (NumPy writes into an
        ///     <c>np.zeros</c> buffer and leaves untouched categories at 0).</param>
        /// <param name="pix">The category probabilities (<paramref name="d"/> values).</param>
        /// <param name="d">The number of categories.</param>
        /// <param name="binomial">The cache shared with the binomial sampler.</param>
        /// <remarks>
        ///     Each count is stored in <typeparamref name="TCount"/> and READ BACK from the row before being subtracted from
        ///     the remainder, exactly as the C code reads <c>mnix[j]</c>; the loop stops early once no trials remain,
        ///     leaving the later categories at their initial zero.
        /// </remarks>
        internal static unsafe void RandomMultinomial<TCount>(BitGenerator bg, long n, TCount* mnix, double* pix, long d,
                                                             BinomialState binomial)
            where TCount : unmanaged, IBinaryInteger<TCount>
        {
            double remaining_p = 1.0;
            long j;
            long dn = n;
            for (j = 0; j < (d - 1); j++)
            {
                mnix[j] = TCount.CreateTruncating(RandomBinomial(bg, pix[j] / remaining_p, dn, binomial));
                dn = dn - long.CreateTruncating(mnix[j]);
                if (dn <= 0)
                    break;
                remaining_p -= pix[j];
            }
            if (dn > 0)
                mnix[d - 1] = TCount.CreateTruncating(dn);
        }

        /// <summary>
        ///     NumPy's <c>random_geometric_search</c>: walk the CDF until it passes one uniform (used for
        ///     <c>p &gt;= 1/3</c>, where the expected walk is short).
        /// </summary>
        /// <param name="bg">The draw source.</param>
        /// <param name="p">The success probability.</param>
        /// <returns>The trial on which the first success occurs (<c>&gt;= 1</c>).</returns>
        internal static long RandomGeometricSearch(BitGenerator bg, double p)
        {
            double U;
            long X;
            double sum, prod, q;

            X = 1;
            sum = prod = p;
            q = 1.0 - p;
            U = bg.NextDouble();
            while (U > sum)
            {
                prod *= q;
                sum += prod;
                X++;
            }
            return X;
        }

        /// <summary>NumPy's <c>random_uniform</c>: <c>lower + range * U</c>.</summary>
        /// <param name="bg">The draw source.</param>
        /// <param name="lower">The lower bound.</param>
        /// <param name="range">The width <c>high - low</c> (the caller checked it is finite).</param>
        /// <returns>The draw.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static double RandomUniform(BitGenerator bg, double lower, double range) => lower + range * bg.NextDouble();

        /// <summary>
        ///     NumPy's <c>random_laplace</c>: inverse CDF of the double exponential, redrawing when <c>U == 0</c>.
        /// </summary>
        /// <param name="bg">The draw source.</param>
        /// <param name="loc">The peak position.</param>
        /// <param name="scale">The exponential decay (validated non-negative by the caller).</param>
        /// <returns>The draw.</returns>
        /// <remarks>NumPy recurses on <c>U == 0</c>; the loop is the same draw sequence without the stack.</remarks>
        internal static double RandomLaplace(BitGenerator bg, double loc, double scale)
        {
            while (true)
            {
                double U = bg.NextDouble();
                if (U >= 0.5)
                    return loc - scale * Math.Log(2.0 - U - U);
                if (U > 0.0)
                    return loc + scale * Math.Log(U + U);
                /* Reject U == 0.0 and call again to get next value */
            }
        }

        /// <summary>
        ///     NumPy's <c>random_gumbel</c>: <c>loc - scale * log(-log(U))</c> with <c>U = 1 - next_double</c>, redrawing
        ///     when <c>U == 1</c>.
        /// </summary>
        /// <param name="bg">The draw source.</param>
        /// <param name="loc">The mode.</param>
        /// <param name="scale">The scale (validated non-negative by the caller).</param>
        /// <returns>The draw — one uniform is consumed even when <paramref name="scale"/> is 0.</returns>
        internal static double RandomGumbel(BitGenerator bg, double loc, double scale)
        {
            while (true)
            {
                double U = 1.0 - bg.NextDouble();
                if (U < 1.0)
                    return loc - scale * Math.Log(-Math.Log(U));
                /* Reject U == 1.0 and call again to get next value */
            }
        }

        /// <summary>
        ///     NumPy's <c>random_logistic</c>: <c>loc + scale * log(U / (1 - U))</c>, redrawing when <c>U == 0</c>.
        /// </summary>
        /// <param name="bg">The draw source.</param>
        /// <param name="loc">The mean.</param>
        /// <param name="scale">The scale (validated non-negative by the caller).</param>
        /// <returns>The draw — one uniform is consumed even when <paramref name="scale"/> is 0.</returns>
        internal static double RandomLogistic(BitGenerator bg, double loc, double scale)
        {
            while (true)
            {
                double U = bg.NextDouble();
                if (U > 0.0)
                    return loc + scale * Math.Log(U / (1.0 - U));
                /* Reject U == 0.0 and call again to get next value */
            }
        }

        /// <summary>NumPy's <c>random_triangular</c>: inverse CDF of the triangular distribution, one uniform per draw.</summary>
        /// <param name="bg">The draw source.</param>
        /// <param name="left">The lower limit.</param>
        /// <param name="mode">The peak (<paramref name="left"/> &lt;= mode &lt;= <paramref name="right"/>, checked by the caller).</param>
        /// <param name="right">The upper limit (&gt; <paramref name="left"/>).</param>
        /// <returns>The draw.</returns>
        internal static double RandomTriangular(BitGenerator bg, double left, double mode, double right)
        {
            double @base, leftbase, ratio, leftprod, rightprod;
            double U;

            @base = right - left;
            leftbase = mode - left;
            ratio = leftbase / @base;
            leftprod = leftbase * @base;
            rightprod = (right - mode) * @base;

            U = bg.NextDouble();
            if (U <= ratio)
                return left + Math.Sqrt(U * leftprod);
            else
                return right - Math.Sqrt((1.0 - U) * rightprod);
        }
    }
}
