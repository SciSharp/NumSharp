using System;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace NumSharp
{
    /// <summary>
    ///     One setup of NumPy's <c>binomial_t</c> — the BTPE constants, or the inversion's <c>q</c>/<c>q^n</c>/bound — for
    ///     the <c>(nsave, psave)</c> key, together with NumSharp's memos of the per-draw terms the samplers derive from it.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Fields keep NumPy's names for a line-by-line comparison with <c>distributions.c</c>. <see cref="m"/> is the
    ///     inversion's search bound for the inversion sampler and the mode for BTPE; the key decides the algorithm
    ///     (<c>n * min(p, 1-p) &lt;= 30</c>), so one setup never carries both meanings.
    ///     </para>
    ///     <para>
    ///     The memos hold terms the C code recomputes on EVERY draw although they depend only on the key: the inversion's
    ///     CDF-walk probabilities (<see cref="px"/>), BTPE's per-attempt <c>nrq</c> and Step50 constants
    ///     (<see cref="nrq"/>, <see cref="s"/>, <see cref="a"/>), Step50's acceptance bound <c>F(y)</c>
    ///     (<see cref="StepF"/>) and Step52's squeeze and Stirling bounds (<see cref="Step52Bounds"/>). Each is the same IEEE
    ///     expression, in the same order, over the same inputs, evaluated the first time it is needed — so reading it back
    ///     changes no bit: the draws, the comparisons and the returned counts stay NumPy's. They turn the inversion walk
    ///     from a chain of divisions into a chain of subtractions, Step50's product loop into a load, and Step52's four
    ///     <c>log</c>s into one. Every (re)population of the key calls <see cref="ResetMemos"/>.
    ///     </para>
    /// </remarks>
    internal sealed class BinomialSetup
    {
        /// <summary>
        ///     How far from the mode Step50's <c>F(y)</c> is memoized (<c>|y - m| &lt;= FWindow</c>). Step50 runs for
        ///     <c>|y - m| &lt;= 20</c> (and for every <c>y</c> once <c>nrq</c> is small), and accepted counts cluster within a
        ///     few standard deviations, so 64 covers nearly every visit; farther candidates compute <c>F</c> directly.
        /// </summary>
        private const int FWindow = 64;

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

        /// <summary>BTPE memo: Step10's <c>nrq = n * r * q</c>, which the C code recomputes on every attempt.</summary>
        internal double nrq;

        /// <summary>BTPE memo: Step50's <c>s = r / q</c>.</summary>
        internal double s;

        /// <summary>BTPE memo: Step50's <c>a = s * (n + 1)</c>.</summary>
        internal double a;

        /// <summary>
        ///     Inversion memo: <c>px[X]</c> is the <c>px</c> the C walk holds at index <c>X</c> — <c>px[0] = q^n</c> and
        ///     <c>px[X] = ((n - X + 1) * p * px[X - 1]) / (X * q)</c> — valid for <c>X &lt; pxCount</c>, extended on demand.
        /// </summary>
        internal double[] px;

        /// <summary>The number of valid <see cref="px"/> entries (0 after a repopulation).</summary>
        internal int pxCount;

        /// <summary>BTPE memo: <c>F(y)</c> at <c>[y - m + FWindow]</c>, NaN until computed; valid while <see cref="fValid"/>.</summary>
        private double[] _f;

        /// <summary>Whether <see cref="_f"/> belongs to the current key (cleared by <see cref="ResetMemos"/>).</summary>
        private bool fValid;

        /// <summary>The slot count of BTPE's Step52 memo (a power of two; direct-mapped by candidate).</summary>
        private const int Step52Slots = 128;

        /// <summary>BTPE Step52 memo: the candidate each slot holds, -1 when empty (Step52 only sees <c>0 &lt;= y &lt;= n</c>).</summary>
        private long[] _s52Key;

        /// <summary>BTPE Step52 memo: the squeeze bounds <c>t - rho</c>, <c>t + rho</c> and the Stirling bound, per slot.</summary>
        private double[] _s52Lo, _s52Hi, _s52Bound;

        /// <summary>Whether the Step52 memo belongs to the current key (cleared by <see cref="ResetMemos"/>).</summary>
        private bool s52Valid;

        /// <summary>
        ///     How many BTPE draws a setup serves before its Step50/Step52 memos start: those memos cost a few kilobytes and a
        ///     fill to set up, which a setup used only a handful of times (a multinomial category key that recurs rarely)
        ///     never earns back — measured: <c>multinomial(1000, [.25,.25,.5])</c> at 1K ran 0.88x NumPy with eager memos.
        /// </summary>
        private const int BtpeMemoWarmup = 64;

        /// <summary>BTPE draws served since the last (re)population (saturates at <see cref="BtpeMemoWarmup"/>).</summary>
        private int btpeUses;

        /// <summary>
        ///     Forgets every memo — call after (re)populating the key. The arrays are kept for reuse; only their validity
        ///     is reset, so repopulating in a scalar loop costs three stores.
        /// </summary>
        internal void ResetMemos()
        {
            pxCount = 0;
            fValid = false;
            s52Valid = false;
            btpeUses = 0;
        }

        /// <summary>
        ///     Counts one BTPE draw against the memo warm-up (see <see cref="BtpeMemoWarmup"/>); call once per draw.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal void CountBtpeUse()
        {
            if (btpeUses < BtpeMemoWarmup)
                btpeUses++;
        }

        /// <summary>
        ///     BTPE Step52's three thresholds for candidate <paramref name="y"/> (at distance <paramref name="k"/> from the
        ///     mode): the squeeze bounds <c>t - rho</c> and <c>t + rho</c> and the Stirling bound <c>log(f(y)/f(m))</c> that
        ///     <c>A = log(v)</c> is compared against — memoized per candidate, direct-mapped.
        /// </summary>
        /// <param name="n">The trial count (the key's <see cref="nsave"/>).</param>
        /// <param name="y">The candidate (<c>0 &lt;= y &lt;= n</c>).</param>
        /// <param name="k"><c>|y - m|</c>.</param>
        /// <param name="lo">The lower squeeze bound <c>t - rho</c> (accept below it).</param>
        /// <param name="hi">The upper squeeze bound <c>t + rho</c> (reject above it).</param>
        /// <param name="bound">The exact bound (reject above it).</param>
        /// <remarks>
        ///     All three depend only on the setup and the candidate — <c>rho</c> and <c>t</c> on <c>k</c> and <c>nrq</c>, the
        ///     Stirling bound on <c>y</c>, <c>m</c>, <c>n</c>, <c>xm</c>, <c>r</c>, <c>q</c> — and a miss evaluates NumPy's
        ///     expressions in NumPy's order, so the comparisons see NumPy's bits. The only difference is WHEN the exact bound
        ///     is evaluated: NumPy skips it when the squeeze decides; a miss here evaluates all three at once (pure work, no
        ///     draws), after which every later visit of the candidate costs one load per threshold instead of four
        ///     <c>log</c>s and sixteen divisions.
        /// </remarks>
        internal void Step52Bounds(long n, long y, long k, out double lo, out double hi, out double bound)
        {
            // Before the setup has earned its memo: NumPy's expressions directly (same bits).
            if (btpeUses < BtpeMemoWarmup)
            {
                ComputeStep52(n, y, k, out lo, out hi, out bound);
                return;
            }
            if (!s52Valid)
            {
                if (_s52Key == null)
                {
                    _s52Key = new long[Step52Slots];
                    _s52Lo = new double[Step52Slots];
                    _s52Hi = new double[Step52Slots];
                    _s52Bound = new double[Step52Slots];
                }
                Array.Fill(_s52Key, -1L);
                s52Valid = true;
            }

            int slot = (int)(y & (Step52Slots - 1));
            if (_s52Key[slot] == y)
            {
                lo = _s52Lo[slot];
                hi = _s52Hi[slot];
                bound = _s52Bound[slot];
                return;
            }

            ComputeStep52(n, y, k, out lo, out hi, out bound);
            _s52Key[slot] = y;
            _s52Lo[slot] = lo;
            _s52Hi[slot] = hi;
            _s52Bound[slot] = bound;
        }

        /// <summary>NumPy's Step52 expressions verbatim: <c>rho</c>, <c>t</c>, and the Stirling-series bound.</summary>
        /// <param name="n">The trial count.</param>
        /// <param name="y">The candidate.</param>
        /// <param name="k"><c>|y - m|</c>.</param>
        /// <param name="lo"><c>t - rho</c>.</param>
        /// <param name="hi"><c>t + rho</c>.</param>
        /// <param name="bound">The exact acceptance bound.</param>
        private void ComputeStep52(long n, long y, long k, out double lo, out double hi, out double bound)
        {
            double rho = (k / (nrq)) * ((k * (k / 3.0 + 0.625) + 0.16666666666666666) / nrq + 0.5);
            // C's `-k * k` is an INTEGER product, converted to double only for the division.
            double t = -k * k / (2 * nrq);
            lo = t - rho;
            hi = t + rho;

            double x1 = (double)y + 1;
            double f1 = (double)m + 1;
            double z = (double)n + 1 - (double)m;
            double w = (double)n - (double)y + 1;
            double x2 = x1 * x1;
            double f2 = f1 * f1;
            double z2 = z * z;
            double w2 = w * w;
            bound = xm * Math.Log(f1 / x1) + (n - m + 0.5) * Math.Log(z / w) +
                    (y - m) * Math.Log(w * r / (x1 * q)) +
                    (13680.0 - (462.0 - (132.0 - (99.0 - 140.0 / f2) / f2) / f2) / f2) / f1 / 166320.0 +
                    (13680.0 - (462.0 - (132.0 - (99.0 - 140.0 / z2) / z2) / z2) / z2) / z / 166320.0 +
                    (13680.0 - (462.0 - (132.0 - (99.0 - 140.0 / x2) / x2) / x2) / x2) / x1 / 166320.0 +
                    (13680.0 - (462.0 - (132.0 - (99.0 - 140.0 / w2) / w2) / w2) / w2) / w / 166320.0;
        }

        /// <summary>
        ///     The sequential CDF search of the inversion sampler for this (inversion-keyed) setup — the walk shared by
        ///     NumPy's <c>random_binomial_inversion</c> and <c>legacy_random_binomial_inversion</c>, which differ only in how
        ///     they compute <c>q^n</c> (already in <see cref="r"/>).
        /// </summary>
        /// <param name="src">The draw source — per-draw, or a read-ahead whose no-overdraw accounting the caller keeps.</param>
        /// <param name="n">The trial count (the key's <see cref="nsave"/>).</param>
        /// <param name="p">The success probability (the key's <see cref="psave"/>).</param>
        /// <returns>The count of successes.</returns>
        /// <remarks>
        ///     <para>
        ///     The C loop keeps <c>px</c> in a register and advances it with one division per step; this walk reads the same
        ///     sequence from <see cref="px"/>, computing an entry — with the C expression, from the entry before it — only the
        ///     first time the walk reaches it. The loop test, the subtraction and the restart at <c>X &gt; bound</c> are the
        ///     C statements, so the draws taken and the count returned are NumPy's.
        ///     </para>
        ///     <para>
        ///     The table never exceeds 86 entries: both inversions run only for <c>n * p &lt;= 30</c> (with
        ///     <c>p &lt;= 0.5</c>), so <c>bound = min(n, np + 10*sqrt(np*q + 1)) &lt;= 30 + 10*sqrt(31) &lt; 86</c>.
        ///     </para>
        /// </remarks>
        internal unsafe long InversionWalk(ref DrawBufferDouble src, long n, double p)
        {
            long bound = m;
            double[] table = px;
            if (table == null || table.Length <= bound)
            {
                px = table = new double[Math.Max(bound + 1, 32)];
                pxCount = 0;
            }
            if (pxCount == 0)
            {
                table[0] = r; // qn
                pxCount = 1;
            }

            double q = this.q;
            int count = pxCount;
            fixed (double* t = table)
            {
                long X = 0;
                double U = src.NextDouble();
                while (U > t[X])
                {
                    X++;
                    if (X > bound)
                    {
                        X = 0;
                        U = src.NextDouble();
                    }
                    else
                    {
                        U -= t[X - 1];
                        if (X == count)
                        {
                            // The C loop's `px = ((n - X + 1) * p * px) / (X * q)`, evaluated once for this key.
                            t[X] = ((n - X + 1) * p * t[X - 1]) / (X * q);
                            pxCount = ++count;
                        }
                    }
                }
                return X;
            }
        }

        /// <summary>
        ///     BTPE Step50's acceptance bound for candidate <paramref name="y"/>: NumPy's
        ///     <c>F = prod_{i=m+1..y} (a/i - s)</c> for <c>m &lt; y</c>, <c>1 / ... / (a/i - s)</c> over <c>(y, m]</c> for
        ///     <c>m &gt; y</c>, and 1 at the mode — memoized per candidate within <c>FWindow</c> of the mode.
        /// </summary>
        /// <param name="y">The candidate count (Step50 only sees <c>0 &lt;= y &lt;= n</c>).</param>
        /// <returns>F, bit-identical to NumPy's loop.</returns>
        /// <remarks>
        ///     F depends only on the key and <paramref name="y"/> (the loop's operands are <see cref="a"/>, <see cref="s"/>,
        ///     <see cref="m"/> and the index), and the loop is evaluated in NumPy's order the first time a candidate is seen,
        ///     so a memoized read returns the very bits the C loop would produce. The window arithmetic cannot overflow
        ///     harmfully: <c>|y - m| &lt;= n &lt; 2^63</c>, and a wrapped offset fails the range test and computes directly.
        /// </remarks>
        internal double StepF(long y)
        {
            long off = y - m + FWindow;
            // Outside the window, or before the setup has earned its memos: NumPy's loop directly (same bits).
            if ((ulong)off > 2 * FWindow || btpeUses < BtpeMemoWarmup)
                return ComputeStepF(y);
            if (!fValid)
            {
                _f ??= new double[2 * FWindow + 1];
                Array.Fill(_f, double.NaN);
                fValid = true;
            }
            // NaN marks "not computed": F starts at 1 and only multiplies or divides by finite factors a/i - s, so it is
            // never NaN (a 0 factor gives 0 or inf); were one ever NaN it would merely be recomputed each time, still exact.
            double f = _f[off];
            if (double.IsNaN(f))
                _f[off] = f = ComputeStepF(y);
            return f;
        }

        /// <summary>NumPy's Step50 loop verbatim (with <c>s</c> and <c>a</c> read from the setup).</summary>
        /// <param name="y">The candidate count.</param>
        /// <returns>F.</returns>
        private double ComputeStepF(long y)
        {
            double F = 1.0;
            if (m < y)
            {
                for (long i = m + 1; i <= y; i++)
                    F *= (a / i - s);
            }
            else if (m > y)
            {
                for (long i = y + 1; i <= m; i++)
                    F /= (a / i - s);
            }
            return F;
        }
    }

    /// <summary>
    ///     NumPy's <c>binomial_t</c> — the per-generator cache of the binomial samplers' setup, keyed on the last
    ///     <c>(n, p)</c> drawn — held as a reference to the <see cref="BinomialSetup"/> that is current.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The cache is observable, which is why it is a shared object rather than recomputed per call: NumPy's legacy
    ///     <c>RandomState</c> owns ONE of these and lends it to both <c>binomial</c> (legacy inversion, <c>exp(n*log(q))</c>)
    ///     and <c>multinomial</c> (the modern inversion, <c>exp(n*log1p(-p))</c>). Both write the same fields under the same
    ///     <c>(n, p)</c> key, so a <c>multinomial</c> call can leave a <c>q^n</c> that a later <c>binomial</c> with a matching
    ///     key reuses instead of recomputing — a one-ULP-level difference NumPy's stream carries and this port reproduces.
    ///     Nothing resets the cache — not re-seeding, not <c>set_state</c> — exactly as in NumPy.
    ///     </para>
    ///     <para>
    ///     <see cref="Memo"/> extends the single entry for a <c>multinomial</c> fill, whose rows cycle through one key per
    ///     category — a pattern on which NumPy's single entry never hits, so every binomial call re-runs its setup. While a
    ///     memo is installed, a miss on <see cref="Current"/> looks the key up there before recomputing; a hit swaps that
    ///     setup in. This is unobservable: the memo only ever holds setups a MODERN call computed (a <c>multinomial</c> fill
    ///     makes no other kind of call, and the memo is removed when the fill ends), and a modern miss would recompute
    ///     exactly those values; the rule that a key hitting <see cref="Current"/> reuses whatever populated it — legacy
    ///     formula included — is untouched. Access is serialized by the owning bit generator's lock.
    ///     </para>
    /// </remarks>
    internal sealed class BinomialState
    {
        /// <summary>log2 of the memo's slot count (1024 direct-mapped slots, so the ~100 keys a many-trial fill cycles through rarely collide).</summary>
        internal const int MemoBits = 10;

        /// <summary>The setup NumPy's single <c>binomial_t</c> holds right now.</summary>
        internal BinomialSetup Current = new BinomialSetup();

        /// <summary>
        ///     The per-key memo of modern setups, direct-mapped by <c>(n, p)</c> — non-null only while a <c>multinomial</c>
        ///     fill runs (see the class remarks for why it must not outlive one).
        /// </summary>
        internal BinomialSetup[] Memo;

        /// <summary>
        ///     Resolves a miss on <see cref="Current"/> for key <c>(<paramref name="n"/>, <paramref name="p"/>)</c>: returns
        ///     the setup to use and makes it current — a memoized one (<paramref name="fresh"/> false), or one the caller
        ///     must populate with NumPy's setup statements and <see cref="BinomialSetup.ResetMemos"/> (<paramref name="fresh"/>
        ///     true).
        /// </summary>
        /// <param name="n">The key's trial count.</param>
        /// <param name="p">The key's probability.</param>
        /// <param name="fresh">True when the returned setup must be (re)populated.</param>
        /// <returns>The setup, already installed as <see cref="Current"/>.</returns>
        /// <remarks>
        ///     Without a memo this is NumPy's single entry: the current setup is repopulated in place. With one, the key's slot
        ///     is reused in place (evicting a colliding key) — safe because every memo slot holds a modern setup and a
        ///     repopulation is itself modern. NaN keys never compare equal, so they always repopulate, as in NumPy.
        /// </remarks>
        internal BinomialSetup Acquire(long n, double p, out bool fresh)
        {
            BinomialSetup[] memo = Memo;
            if (memo == null)
            {
                fresh = true;
                return Current;
            }

            ulong h = ((ulong)n * 0x9E3779B97F4A7C15UL) ^ (ulong)BitConverter.DoubleToInt64Bits(p);
            int slot = (int)((h * 0xBF58476D1CE4E5B9UL) >> (64 - MemoBits));
            BinomialSetup e = memo[slot];
            if (e != null && e.has_binomial && e.nsave == n && e.psave == p)
            {
                Current = e;
                fresh = false;
                return e;
            }
            if (e == null)
                memo[slot] = e = new BinomialSetup();
            Current = e;
            fresh = true;
            return e;
        }
    }

    /// <summary>
    ///     The per-mean setup of NumPy's <c>random_poisson</c>: the product method's <c>exp(-lam)</c>, or PTRS's
    ///     <c>sqrt(lam)</c>, <c>log(lam)</c> and envelope constants — computed ONCE for a fill instead of on every value.
    /// </summary>
    /// <remarks>
    ///     NumPy's C recomputes these on each call (and <c>log(invalpha)</c> on every PTRS rejection test); they are
    ///     deterministic functions of <c>lam</c> evaluated with the same IEEE operations, so a fill that computes them once
    ///     produces the same bits while skipping an <c>exp</c> (product method) or a <c>sqrt</c>, two <c>log</c>s and two
    ///     divisions (PTRS) per value. A caller whose mean changes per value (negative binomial, the Poisson-mixed
    ///     noncentral chi-square) builds one per value, which is exactly NumPy's per-call arithmetic.
    /// </remarks>
    internal readonly struct PoissonSetup
    {
        /// <summary>The mean; selects the branch per value (PTRS for <c>&gt;= 10</c>, none for 0, else the product method).</summary>
        internal readonly double Lam;

        /// <summary>Product method: <c>exp(-lam)</c> (NaN for a NaN mean, which then returns 0 after one draw).</summary>
        internal readonly double Enlam;

        /// <summary>PTRS: <c>sqrt(lam)</c>.</summary>
        internal readonly double Slam;

        /// <summary>PTRS: <c>log(lam)</c>.</summary>
        internal readonly double Loglam;

        /// <summary>PTRS: <c>-0.059 + 0.02483 * b</c>.</summary>
        internal readonly double A;

        /// <summary>PTRS: <c>0.931 + 2.53 * slam</c>.</summary>
        internal readonly double B;

        /// <summary>PTRS: <c>1.1239 + 1.1328 / (b - 3.4)</c>.</summary>
        internal readonly double Invalpha;

        /// <summary>PTRS: <c>0.9277 - 3.6224 / (b - 2)</c>.</summary>
        internal readonly double Vr;

        /// <summary>
        ///     PTRS: <c>log(invalpha)</c>, which the C rejection test recomputes per trial — hoisted only for a fill
        ///     (<c>hoistRejectionLog</c>); NaN otherwise, and the rejection test then computes it where NumPy does.
        /// </summary>
        internal readonly double LogInvalpha;

        /// <summary>Evaluates NumPy's setup statements for <paramref name="lam"/> (only the selected branch's).</summary>
        /// <param name="lam">The validated mean.</param>
        /// <param name="hoistRejectionLog">
        ///     Whether to also evaluate PTRS's per-rejection <c>log(invalpha)</c> now. Worth it for a fill, where one setup
        ///     serves every value; NOT for a setup built per value (a Poisson mean that is itself a draw), where it would add a
        ///     <c>log</c> to every value that only the ~10% of PTRS trials that miss the squeeze need.
        /// </param>
        /// <remarks>
        ///     Built ONCE per fill. A caller whose mean changes per value must not build one per value: constructing this
        ///     72-byte struct costs more than the <c>exp</c> it holds (measured ~22 ns a value, with the <c>exp</c> itself
        ///     hidden by out-of-order execution) — use <see cref="Distributions.RandomPoisson(ref DrawBufferDouble, double)"/>,
        ///     which keeps NumPy's setup in locals.
        /// </remarks>
        internal PoissonSetup(double lam, bool hoistRejectionLog)
        {
            Lam = lam;
            Enlam = Slam = Loglam = A = B = Invalpha = Vr = 0.0;
            LogInvalpha = double.NaN;
            if (lam >= 10)
            {
                Distributions.PtrsSetup(lam, out Slam, out Loglam, out B, out A, out Invalpha, out Vr);
                // invalpha > 1.1239 for every lam >= 10, so its log is finite: NaN can only mean "not hoisted".
                if (hoistRejectionLog)
                    LogInvalpha = Math.Log(Invalpha);
            }
            else if (lam != 0)
            {
                Enlam = Math.Exp(-lam);
            }
        }

        /// <summary>Whether every value draws at least once — false only for <c>lam == 0</c> (0 without a draw).</summary>
        internal bool Draws => Lam != 0;
    }

    /// <summary>
    ///     A line-by-line port of the scalar samplers in <c>numpy/random/src/distributions/distributions.c</c> that NumPy's
    ///     legacy <c>RandomState</c> and its <see cref="Generator"/> share, drawing <c>next_double</c> units through a
    ///     <see cref="DrawBufferDouble"/> exactly as the C code draws from <c>bitgen_t</c>.
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
    ///     <para>
    ///     The draw source is a <see cref="DrawBufferDouble"/>: a one-double buffer IS the per-draw call sequence, and a bulk
    ///     fill hands in a read-ahead (the bit generator's vectorized <c>FillDouble</c>) whose <see cref="DrawBufferDouble.Owed"/>
    ///     the caller keeps at a lower bound of the draws still to come, so the stream never runs ahead of NumPy's. Setup
    ///     that depends only on the parameters (<see cref="PoissonSetup"/>, <see cref="BinomialSetup"/>) is computed once per
    ///     fill and read back, which is bit-neutral because it is the same expression over the same inputs.
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
        /// <param name="src">The draw source — per-draw, or a read-ahead whose no-overdraw accounting the caller keeps.</param>
        /// <param name="enlam">The setup's <c>exp(-lam)</c> (NaN yields 0 after one draw, since no product exceeds it).</param>
        /// <returns>The count.</returns>
        internal static long RandomPoissonMult(ref DrawBufferDouble src, double enlam)
        {
            long X;
            double prod, U;

            X = 0;
            prod = 1.0;
            while (true)
            {
                U = src.NextDouble();
                prod *= U;
                if (prod > enlam)
                    X += 1;
                else
                    return X;
            }
        }

        /// <summary>
        ///     NumPy's per-call PTRS setup statements (<c>random_poisson_ptrs</c>'s prologue) — the one definition both the
        ///     fill's <see cref="PoissonSetup"/> and the per-value <see cref="RandomPoisson(ref DrawBufferDouble, double)"/> use.
        /// </summary>
        /// <param name="lam">The mean, <c>&gt;= 10</c>.</param>
        /// <param name="slam"><c>sqrt(lam)</c>.</param>
        /// <param name="loglam"><c>log(lam)</c>.</param>
        /// <param name="b"><c>0.931 + 2.53 * slam</c>.</param>
        /// <param name="a"><c>-0.059 + 0.02483 * b</c>.</param>
        /// <param name="invalpha"><c>1.1239 + 1.1328 / (b - 3.4)</c>.</param>
        /// <param name="vr"><c>0.9277 - 3.6224 / (b - 2)</c>.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void PtrsSetup(double lam, out double slam, out double loglam, out double b, out double a,
                                       out double invalpha, out double vr)
        {
            slam = Math.Sqrt(lam);
            loglam = Math.Log(lam);
            b = 0.931 + 2.53 * slam;
            a = -0.059 + 0.02483 * b;
            invalpha = 1.1239 + 1.1328 / (b - 3.4);
            vr = 0.9277 - 3.6224 / (b - 2);
        }

        /// <summary>
        ///     NumPy's <c>random_poisson_ptrs</c> trial loop: Hörmann's transformed rejection with squeeze (PTRS, the
        ///     <c>lam &gt;= 10</c> branch), two draws per trial, over the setup values <see cref="PtrsSetup"/> computes.
        /// </summary>
        /// <param name="src">The draw source — per-draw, or a read-ahead whose no-overdraw accounting the caller keeps.</param>
        /// <param name="lam">The mean, <c>&gt;= 10</c>.</param>
        /// <param name="loglam"><c>log(lam)</c>.</param>
        /// <param name="a">The envelope's <c>a</c>.</param>
        /// <param name="b">The envelope's <c>b</c>.</param>
        /// <param name="invalpha">The envelope's <c>invalpha</c>.</param>
        /// <param name="vr">The squeeze bound.</param>
        /// <param name="logInvalpha"><c>log(invalpha)</c> when a fill hoisted it; NaN to compute it per rejection trial as
        ///     NumPy does (identical bits either way).</param>
        /// <returns>The count.</returns>
        /// <remarks>
        ///     The setup arrives as scalars, not as a <see cref="PoissonSetup"/>, so a caller whose mean changes per value
        ///     pays for no struct — see <see cref="PoissonSetup(double, bool)"/> for why that matters.
        /// </remarks>
        internal static long RandomPoissonPtrs(ref DrawBufferDouble src, double lam, double loglam, double a, double b,
                                               double invalpha, double vr, double logInvalpha)
        {
            long k;
            double U, V, us;

            while (true)
            {
                U = src.NextDouble() - 0.5;
                V = src.NextDouble();
                us = 0.5 - Math.Abs(U);
                // C's (RAND_INT_TYPE)floor(...): us == 0 sends the argument to +-inf/NaN, which must read as a negative
                // count and be rejected below, not saturate to a huge accepted value.
                k = ToInt64(Math.Floor((2 * a / us + b) * U + lam + 0.43));
                if ((us >= 0.07) && (V <= vr))
                    return k;
                if ((k < 0) || ((us < 0.013) && (V > us)))
                    continue;
                // log(invalpha): the fill's hoisted value, or computed here per trial as NumPy does (same bits either way).
                double li = double.IsNaN(logInvalpha) ? Math.Log(invalpha) : logInvalpha;
                /* log(V) == log(0.0) ok here */
                /* if U==0.0 so that us==0.0, log is ok since always returns */
                if ((Math.Log(V) + li - Math.Log(a / (us * us) + b)) <=
                    (-lam + (double)k * loglam - RandomLoggam((double)k + 1)))
                    return k;
            }
        }

        /// <summary>
        ///     NumPy's <c>random_poisson</c> for a fill's fixed mean: PTRS for <c>lam &gt;= 10</c>, 0 (no draw) for
        ///     <c>lam == 0</c>, else the product method — over the setup the fill built once.
        /// </summary>
        /// <param name="src">The draw source — per-draw, or a read-ahead whose no-overdraw accounting the caller keeps.</param>
        /// <param name="s">The mean's setup.</param>
        /// <returns>The count.</returns>
        internal static long RandomPoisson(ref DrawBufferDouble src, in PoissonSetup s)
        {
            if (s.Lam >= 10)
                return RandomPoissonPtrs(ref src, s.Lam, s.Loglam, s.A, s.B, s.Invalpha, s.Vr, s.LogInvalpha);
            else if (s.Lam == 0)
                return 0;
            else
                return RandomPoissonMult(ref src, s.Enlam);
        }

        /// <summary>
        ///     NumPy's <c>random_poisson</c> for a mean that changes per call (a negative binomial's gamma-drawn mean, the
        ///     noncentral chi-square's Poisson mixing): the setup is evaluated here, in locals, exactly NumPy's per-call
        ///     arithmetic.
        /// </summary>
        /// <param name="src">The draw source — per-draw, or a read-ahead whose no-overdraw accounting the caller keeps.</param>
        /// <param name="lam">The mean (validated by the caller).</param>
        /// <returns>The count.</returns>
        internal static long RandomPoisson(ref DrawBufferDouble src, double lam)
        {
            if (lam >= 10)
            {
                PtrsSetup(lam, out _, out double loglam, out double b, out double a, out double invalpha, out double vr);
                return RandomPoissonPtrs(ref src, lam, loglam, a, b, invalpha, vr, double.NaN);
            }
            else if (lam == 0)
                return 0;
            else
                return RandomPoissonMult(ref src, Math.Exp(-lam));
        }

        /// <summary>
        ///     NumPy's <c>random_binomial_btpe</c>: Kachitvichyanukul &amp; Schmeiser's BTPE rejection sampler for
        ///     <c>n*min(p,1-p) &gt; 30</c>, with its setup cached in <paramref name="binomial"/>.
        /// </summary>
        /// <param name="src">The draw source — per-draw, or a read-ahead whose no-overdraw accounting the caller keeps.</param>
        /// <param name="n">The trial count.</param>
        /// <param name="p">The success probability (every NumPy caller passes <c>p &lt;= 0.5</c>).</param>
        /// <param name="binomial">The cache (see <see cref="BinomialState"/>).</param>
        /// <returns>The count of successes.</returns>
        /// <remarks>
        ///     The <c>goto</c> structure is NumPy's, kept so each step can be compared with the C label it ports. Step10's
        ///     <c>nrq</c>, Step50's <c>s</c>, <c>a</c> and <c>F</c> product, and Step52's <c>rho</c>, <c>t</c> and Stirling
        ///     bound are read from the setup's memos (<see cref="BinomialSetup"/>) — the same values the C statements
        ///     recompute on each attempt; only Step52's <c>log(v)</c> is still evaluated per trial.
        /// </remarks>
        internal static long RandomBinomialBtpe(ref DrawBufferDouble src, long n, double p, BinomialState binomial)
        {
            double r, q, fm, p1, xm, xl, xr, c, laml, lamr, p2, p3, p4;
            double a, u, v, F, A, nrq, x;
            long m, y, k;

            BinomialSetup e = binomial.Current;
            if (!e.has_binomial || (e.nsave != n) || (e.psave != p))
            {
                e = binomial.Acquire(n, p, out bool fresh);
                if (fresh)
                {
                    /* initialize */
                    e.nsave = n;
                    e.psave = p;
                    e.has_binomial = true;
                    // C's MIN macro, `p < 1-p ? p : 1-p`, spelled out (Math.Min differs on the signed-zero/NaN corners).
                    e.r = r = p < 1.0 - p ? p : 1.0 - p;
                    e.q = q = 1.0 - r;
                    e.fm = fm = n * r + r;
                    e.m = m = ToInt64(Math.Floor(e.fm));
                    e.p1 = p1 = Math.Floor(2.195 * Math.Sqrt(n * r * q) - 4.6 * q) + 0.5;
                    e.xm = xm = m + 0.5;
                    e.xl = xl = xm - p1;
                    e.xr = xr = xm + p1;
                    e.c = c = 0.134 + 20.5 / (15.3 + m);
                    a = (fm - xl) / (fm - xl * r);
                    e.laml = laml = a * (1.0 + a / 2.0);
                    a = (xr - fm) / (xr * q);
                    e.lamr = lamr = a * (1.0 + a / 2.0);
                    e.p2 = p2 = p1 * (1.0 + 2.0 * c);
                    e.p3 = p3 = p2 + c / laml;
                    e.p4 = p4 = p3 + c / lamr;
                    // The memos: Step10's per-attempt nrq and Step50's s/a, as the C statements compute them.
                    e.nrq = n * r * q;
                    e.s = r / q;
                    e.a = e.s * (n + 1);
                    e.ResetMemos();
                }
            }

            r = e.r;
            q = e.q;
            fm = e.fm;
            m = e.m;
            p1 = e.p1;
            xm = e.xm;
            xl = e.xl;
            xr = e.xr;
            c = e.c;
            laml = e.laml;
            lamr = e.lamr;
            p2 = e.p2;
            p3 = e.p3;
            p4 = e.p4;
            nrq = e.nrq; // Step10's `nrq = n * r * q`, identical on every attempt
            e.CountBtpeUse();

        Step10:
            u = src.NextDouble() * p4;
            v = src.NextDouble();
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

            // s = r / q; a = s * (n + 1); F = the product over (m, y] or its reciprocal over (y, m] — memoized per setup.
            F = e.StepF(y);
            if (v > F)
                goto Step10;
            goto Step60;

        Step52:
            // rho, t and the Stirling bound depend only on the candidate — NumPy's expressions, memoized per y
            // (BinomialSetup.Step52Bounds); only log(v) is per trial.
            e.Step52Bounds(n, y, k, out double squeezeLo, out double squeezeHi, out double stirling);
            /* log(0.0) ok here */
            A = Math.Log(v);
            if (A < squeezeLo)
                goto Step60;
            if (A > squeezeHi)
                goto Step10;
            if (A > stirling)
                goto Step10;

        Step60:
            if (p > 0.5)
                y = n - y;

            return y;
        }

        /// <summary>
        ///     NumPy's (modern) <c>random_binomial_inversion</c>: sequential CDF search, <c>q^n</c> computed as
        ///     <c>exp(n * log1p(-p))</c> — the variant <see cref="RandomBinomial"/> (and so <c>multinomial</c>) uses.
        /// </summary>
        /// <param name="src">The draw source — per-draw, or a read-ahead whose no-overdraw accounting the caller keeps.</param>
        /// <param name="n">The trial count.</param>
        /// <param name="p">The success probability (<c>&lt;= 0.5</c> from every caller).</param>
        /// <param name="binomial">The cache (see <see cref="BinomialState"/>).</param>
        /// <returns>The count of successes.</returns>
        /// <remarks>
        ///     The legacy <c>RandomState.binomial</c> uses a copy that computes <c>exp(n * log(q))</c> instead
        ///     (<c>NumPyRandom.LegacyBinomialInversion</c>); the two share the cache, see <see cref="BinomialState"/>, and the
        ///     walk itself (<see cref="BinomialSetup.InversionWalk"/>).
        /// </remarks>
        internal static long RandomBinomialInversion(ref DrawBufferDouble src, long n, double p, BinomialState binomial)
        {
            BinomialSetup e = binomial.Current;
            if (!e.has_binomial || (e.nsave != n) || (e.psave != p))
            {
                e = binomial.Acquire(n, p, out bool fresh);
                if (fresh)
                {
                    double q, np;
                    e.nsave = n;
                    e.psave = p;
                    e.has_binomial = true;
                    e.q = q = 1.0 - p;
                    e.r = Math.Exp(n * Generator.Log1p(-p));
                    e.c = np = n * p;
                    // C's MIN macro — `n < y ? n : y` in double — then the (RAND_INT_TYPE) cast.
                    double limit = np + 10.0 * Math.Sqrt(np * q + 1);
                    e.m = ToInt64(n < limit ? n : limit);
                    e.ResetMemos();
                }
            }
            return e.InversionWalk(ref src, n, p);
        }

        /// <summary>
        ///     NumPy's (modern) <c>random_binomial</c>: 0 with no draw for <c>n == 0</c> or <c>p == 0</c>, else inversion for
        ///     <c>n*min(p,1-p) &lt;= 30</c> and BTPE above, mirrored through <c>1-p</c> when <c>p &gt; 0.5</c>.
        /// </summary>
        /// <param name="src">The draw source — per-draw, or a read-ahead whose no-overdraw accounting the caller keeps.</param>
        /// <param name="p">The success probability.</param>
        /// <param name="n">The trial count.</param>
        /// <param name="binomial">The cache (see <see cref="BinomialState"/>).</param>
        /// <returns>The count of successes.</returns>
        internal static long RandomBinomial(ref DrawBufferDouble src, double p, long n, BinomialState binomial)
        {
            double q;

            // NumPy compares against the FLOAT literal 0.0f; zero is zero in either width.
            if ((n == 0L) || (p == 0.0f))
                return 0;

            if (p <= 0.5)
            {
                if (p * n <= 30.0)
                    return RandomBinomialInversion(ref src, n, p, binomial);
                else
                    return RandomBinomialBtpe(ref src, n, p, binomial);
            }
            else
            {
                q = 1.0 - p;
                if (q * n <= 30.0)
                    return n - RandomBinomialInversion(ref src, n, q, binomial);
                else
                    return n - RandomBinomialBtpe(ref src, n, q, binomial);
            }
        }

        /// <summary>
        ///     NumPy's <c>random_multinomial</c>: one conditional binomial per category, the last category taking whatever is
        ///     left. Writes one row of counts.
        /// </summary>
        /// <typeparam name="TCount">The output count type (C <c>long</c> in the legacy build, <c>int64_t</c> in Generator).</typeparam>
        /// <param name="src">The draw source — per-draw, or a read-ahead whose no-overdraw accounting the caller keeps.</param>
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
        internal static unsafe void RandomMultinomial<TCount>(ref DrawBufferDouble src, long n, TCount* mnix, double* pix, long d,
                                                             BinomialState binomial)
            where TCount : unmanaged, IBinaryInteger<TCount>
        {
            double remaining_p = 1.0;
            long j;
            long dn = n;
            for (j = 0; j < (d - 1); j++)
            {
                mnix[j] = TCount.CreateTruncating(RandomBinomial(ref src, pix[j] / remaining_p, dn, binomial));
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
        /// <param name="src">The draw source — per-draw, or a read-ahead whose no-overdraw accounting the caller keeps.</param>
        /// <param name="p">The success probability.</param>
        /// <returns>The trial on which the first success occurs (<c>&gt;= 1</c>).</returns>
        internal static long RandomGeometricSearch(ref DrawBufferDouble src, double p)
        {
            double U;
            long X;
            double sum, prod, q;

            X = 1;
            sum = prod = p;
            q = 1.0 - p;
            U = src.NextDouble();
            while (U > sum)
            {
                prod *= q;
                sum += prod;
                X++;
            }
            return X;
        }

        /// <summary>NumPy's <c>random_uniform</c>: <c>lower + range * U</c>.</summary>
        /// <param name="src">The draw source — per-draw, or a read-ahead whose no-overdraw accounting the caller keeps.</param>
        /// <param name="lower">The lower bound.</param>
        /// <param name="range">The width <c>high - low</c> (the caller checked it is finite).</param>
        /// <returns>The draw.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static double RandomUniform(ref DrawBufferDouble src, double lower, double range) => lower + range * src.NextDouble();

        /// <summary>
        ///     NumPy's <c>random_laplace</c>: inverse CDF of the double exponential, redrawing when <c>U == 0</c>.
        /// </summary>
        /// <param name="src">The draw source — per-draw, or a read-ahead whose no-overdraw accounting the caller keeps.</param>
        /// <param name="loc">The peak position.</param>
        /// <param name="scale">The exponential decay (validated non-negative by the caller).</param>
        /// <returns>The draw.</returns>
        /// <remarks>NumPy recurses on <c>U == 0</c>; the loop is the same draw sequence without the stack.</remarks>
        internal static double RandomLaplace(ref DrawBufferDouble src, double loc, double scale)
        {
            while (true)
            {
                double U = src.NextDouble();
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
        /// <param name="src">The draw source — per-draw, or a read-ahead whose no-overdraw accounting the caller keeps.</param>
        /// <param name="loc">The mode.</param>
        /// <param name="scale">The scale (validated non-negative by the caller).</param>
        /// <returns>The draw — one uniform is consumed even when <paramref name="scale"/> is 0.</returns>
        internal static double RandomGumbel(ref DrawBufferDouble src, double loc, double scale)
        {
            while (true)
            {
                double U = 1.0 - src.NextDouble();
                if (U < 1.0)
                    return loc - scale * Math.Log(-Math.Log(U));
                /* Reject U == 1.0 and call again to get next value */
            }
        }

        /// <summary>
        ///     NumPy's <c>random_logistic</c>: <c>loc + scale * log(U / (1 - U))</c>, redrawing when <c>U == 0</c>.
        /// </summary>
        /// <param name="src">The draw source — per-draw, or a read-ahead whose no-overdraw accounting the caller keeps.</param>
        /// <param name="loc">The mean.</param>
        /// <param name="scale">The scale (validated non-negative by the caller).</param>
        /// <returns>The draw — one uniform is consumed even when <paramref name="scale"/> is 0.</returns>
        internal static double RandomLogistic(ref DrawBufferDouble src, double loc, double scale)
        {
            while (true)
            {
                double U = src.NextDouble();
                if (U > 0.0)
                    return loc + scale * Math.Log(U / (1.0 - U));
                /* Reject U == 0.0 and call again to get next value */
            }
        }

        /// <summary>NumPy's <c>random_triangular</c>: inverse CDF of the triangular distribution, one uniform per draw.</summary>
        /// <param name="src">The draw source — per-draw, or a read-ahead whose no-overdraw accounting the caller keeps.</param>
        /// <param name="left">The lower limit.</param>
        /// <param name="mode">The peak (<paramref name="left"/> &lt;= mode &lt;= <paramref name="right"/>, checked by the caller).</param>
        /// <param name="right">The upper limit (&gt; <paramref name="left"/>).</param>
        /// <returns>The draw.</returns>
        internal static double RandomTriangular(ref DrawBufferDouble src, double left, double mode, double right)
        {
            double @base, leftbase, ratio, leftprod, rightprod;
            double U;

            @base = right - left;
            leftbase = mode - left;
            ratio = leftbase / @base;
            leftprod = leftbase * @base;
            rightprod = (right - mode) * @base;

            U = src.NextDouble();
            if (U <= ratio)
                return left + Math.Sqrt(U * leftprod);
            else
                return right - Math.Sqrt((1.0 - U) * rightprod);
        }
    }
}
