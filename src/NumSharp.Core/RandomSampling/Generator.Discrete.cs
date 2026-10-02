using System;

namespace NumSharp
{
    public sealed partial class Generator
    {
        /// <summary>
        ///     The Generator's binomial setup cache (NumPy's <c>self._binomial</c>), shared by <see cref="binomial"/> and
        ///     <see cref="multinomial(long, double[], Shape)"/> exactly as in NumPy (see <see cref="BinomialState"/>).
        /// </summary>
        private readonly BinomialState _binomial = new BinomialState();

        /// <summary>
        ///     NumPy's <c>POISSON_LAM_MAX</c> for the Generator: <c>int64 max - sqrt(int64 max) * 10</c>, the largest mean whose
        ///     Poisson draws cannot overflow int64 (also the negative binomial's bound).
        /// </summary>
        private const double PoissonLamMax = RandomConstraints.PoissonLamMax;

        /// <summary>
        ///     Draw samples from a binomial distribution.
        /// </summary>
        /// <param name="n">Number of trials, non-negative (a fractional count truncates, as NumPy's <c>&lt;int64_t&gt;n</c>).</param>
        /// <param name="p">Probability of success, in <c>[0, 1]</c> (NaN rejected).</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) returns a single value.</param>
        /// <returns>The int64 draws.</returns>
        /// <exception cref="ValueError">In NumPy's order: <paramref name="p"/> outside <c>[0, 1]</c> or NaN
        ///     (<c>p &lt; 0, p &gt; 1 or p is NaN</c>), then <paramref name="n"/> negative (<c>n &lt; 0</c>).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.binomial.html
        ///     <br/>NumPy's modern <c>random_binomial</c>: 0 without drawing for <c>n == 0</c> or <c>p == 0</c>, the inversion
        ///     sampler for <c>n*min(p,1-p) &lt;= 30</c> and BTPE above, over this Generator's own setup cache. Byte-identical to
        ///     <c>default_rng(seed).binomial</c>.
        /// </remarks>
        public NDArray binomial(long n, double p, Shape size = default)
        {
            RandomConstraints.Check(p, "p", ConstraintType.CONS_BOUNDED_0_1);
            RandomConstraints.Check((double)n, "n", ConstraintType.CONS_NON_NEGATIVE);

            if (IsNoSize(size))
            {
                unsafe
                {
                    double word;
                    var one = new DrawBufferDouble(_bitGenerator, &word, 1);
                    lock (_bitGenerator.@lock)
                        return NDArray.Scalar(Distributions.RandomBinomial(ref one, p, n, _binomial));
                }
            }

            var ret = DistOutput(NPTypeCode.Int64, size);
            unsafe
            {
                var dst = (long*)ret.Address;
                long count = ret.size;
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                var src = new DrawBufferDouble(_bitGenerator, storage, DrawBufferDouble.Capacity);
                lock (_bitGenerator.@lock)
                {
                    // random_binomial's dispatch, decided once for the fill: a BTPE key (n * min(p, 1-p) > 30) fills with
                    // Step10's triangle inlined per value; the rest (inversion, and the n == 0 / p == 0 shortcut) per call.
                    if (n != 0 && p != 0.0 && (p <= 0.5 ? p * n > 30.0 : (1.0 - p) * n > 30.0))
                    {
                        if (p <= 0.5)
                            Distributions.RandomBinomialBtpeFill(ref src, dst, count, n, p, complement: false, _binomial);
                        else
                            Distributions.RandomBinomialBtpeFill(ref src, dst, count, n, 1.0 - p, complement: true, _binomial);
                    }
                    else
                    {
                        for (long i = 0; i < count; i++)
                        {
                            src.Owed = count - i;
                            dst[i] = Distributions.RandomBinomial(ref src, p, n, _binomial);
                        }
                    }
                }
            }
            return ret;
        }

        /// <summary>
        ///     Draw samples from a negative binomial distribution.
        /// </summary>
        /// <param name="n">Parameter of the distribution, positive and not NaN (need not be an integer).</param>
        /// <param name="p">Probability of success, in <c>(0, 1]</c>.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) returns a single value.</param>
        /// <returns>The int64 draws (the number of failures before <paramref name="n"/> successes).</returns>
        /// <exception cref="ValueError">In NumPy's order: <paramref name="n"/> is NaN (<c>n must not be NaN</c>) or not
        ///     positive (<c>n &lt;= 0</c>); <paramref name="p"/> outside <c>(0, 1]</c>; or the implied Poisson mean
        ///     <c>(1-p)/p * (n + 10*sqrt(n))</c> exceeds the int64-safe bound (<c>n too large or p too small, see
        ///     Generator.negative_binomial Notes</c>).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.negative_binomial.html
        ///     <br/>NumPy's modern <c>random_negative_binomial</c>: a Poisson whose mean is a gamma draw of shape
        ///     <paramref name="n"/> and scale <c>(1 - p) / p</c>. Byte-identical to <c>default_rng(seed).negative_binomial</c>.
        /// </remarks>
        public NDArray negative_binomial(double n, double p, Shape size = default)
        {
            RandomConstraints.Check(n, "n", ConstraintType.CONS_POSITIVE_NOT_NAN);
            RandomConstraints.Check(p, "p", ConstraintType.CONS_BOUNDED_GT_0_1);
            // The largest mean the gamma is likely to produce must stay a valid Poisson mean.
            double maxLam = (1 - p) / p * (n + 10 * Math.Sqrt(n));
            if (maxLam > PoissonLamMax)
                throw new ValueError("n too large or p too small, see Generator.negative_binomial Notes");

            var gamma = new GammaSetup(n);
            double scale = (1 - p) / p;

            if (IsNoSize(size))
            {
                unsafe
                {
                    ulong word;
                    var one = new DrawBuffer64(_bitGenerator, &word, 1);
                    lock (_bitGenerator.@lock)
                        return NDArray.Scalar(NegativeBinomial(ref one, in gamma, scale));
                }
            }

            var ret = DistOutput(NPTypeCode.Int64, size);
            unsafe
            {
                var dst = (long*)ret.Address;
                long count = ret.size;
                ulong* storage = stackalloc ulong[DrawBuffer64.Capacity];
                var src = new DrawBuffer64(_bitGenerator, storage, DrawBuffer64.Capacity);
                lock (_bitGenerator.@lock)
                    for (long i = 0; i < count; i++)
                    {
                        src.Owed = count - i;
                        dst[i] = NegativeBinomial(ref src, in gamma, scale);
                    }
            }
            return ret;
        }

        /// <summary>
        ///     Draw samples from a Poisson distribution.
        /// </summary>
        /// <param name="lam">Expected number of events, non-negative, not NaN, at most <c>int64 max - 10*sqrt(int64 max)</c>. Default 1.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) returns a single value.</param>
        /// <returns>The int64 draws.</returns>
        /// <exception cref="ValueError"><paramref name="lam"/> is negative or NaN (<c>lam &lt; 0 or lam is NaN</c>, checked
        ///     first) or too large (<c>lam value too large</c>).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.poisson.html
        ///     <br/>NumPy's <c>random_poisson</c>: 0 without drawing for <c>lam == 0</c>, the product method below 10 and
        ///     PTRS above, its setup evaluated once per call. Byte-identical to <c>default_rng(seed).poisson</c>.
        /// </remarks>
        public NDArray poisson(double lam = 1.0, Shape size = default)
        {
            RandomConstraints.Check(lam, "lam", ConstraintType.CONS_POISSON);
            var setup = new PoissonSetup(lam, hoistRejectionLog: !IsNoSize(size));

            if (IsNoSize(size))
            {
                unsafe
                {
                    double word;
                    var one = new DrawBufferDouble(_bitGenerator, &word, 1);
                    lock (_bitGenerator.@lock)
                        return NDArray.Scalar(Distributions.RandomPoisson(ref one, in setup));
                }
            }

            var ret = DistOutput(NPTypeCode.Int64, size);
            unsafe
            {
                var dst = (long*)ret.Address;
                long count = ret.size;
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                var src = new DrawBufferDouble(_bitGenerator, storage, DrawBufferDouble.Capacity);
                lock (_bitGenerator.@lock)
                    for (long i = 0; i < count; i++)
                    {
                        src.Owed = count - i;
                        dst[i] = Distributions.RandomPoisson(ref src, in setup);
                    }
            }
            return ret;
        }

        /// <summary>
        ///     Draw samples from a Zipf distribution.
        /// </summary>
        /// <param name="a">Distribution parameter, greater than 1 (NaN rejected).</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) returns a single value.</param>
        /// <returns>The int64 draws (<c>&gt;= 1</c>).</returns>
        /// <exception cref="ValueError"><paramref name="a"/> is not greater than 1 or is NaN (<c>a &lt;= 1 or a is NaN</c>).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.zipf.html
        ///     <br/>NumPy's modern <c>random_zipf</c>: candidates are drawn from a window that keeps them within int64, and
        ///     <c>a &gt;= 1025</c> (infinity included) returns 1 without drawing. Byte-identical to <c>default_rng(seed).zipf</c>.
        /// </remarks>
        public NDArray zipf(double a, Shape size = default)
        {
            RandomConstraints.Check(a, "a", ConstraintType.CONS_GT_1);
            var setup = new ZipfSetup(a);

            if (IsNoSize(size))
            {
                unsafe
                {
                    double word;
                    var one = new DrawBufferDouble(_bitGenerator, &word, 1);
                    lock (_bitGenerator.@lock)
                        return NDArray.Scalar(Zipf(ref one, in setup, null));
                }
            }

            var ret = DistOutput(NPTypeCode.Int64, size);
            unsafe
            {
                var dst = (long*)ret.Address;
                long count = ret.size;
                // A memo of the acceptance test's pow(1 + 1/X, a - 1) for the small candidates (bit-neutral).
                double[] tMemo = null;
                if (count >= 16 && !setup.Degenerate)
                {
                    tMemo = new double[256];
                    Array.Fill(tMemo, double.NaN);
                }
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                var src = new DrawBufferDouble(_bitGenerator, storage, DrawBufferDouble.Capacity);
                lock (_bitGenerator.@lock)
                    for (long i = 0; i < count; i++)
                    {
                        src.Owed = count - i;
                        dst[i] = Zipf(ref src, in setup, tMemo);
                    }
            }
            return ret;
        }

        /// <summary>
        ///     Draw samples from the geometric distribution.
        /// </summary>
        /// <param name="p">The probability of success of an individual trial, in <c>(0, 1]</c>.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) returns a single value.</param>
        /// <returns>The int64 draws (the trial of the first success, <c>&gt;= 1</c>).</returns>
        /// <exception cref="ValueError"><paramref name="p"/> is outside <c>(0, 1]</c> or NaN.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.geometric.html
        ///     <br/>NumPy's modern <c>random_geometric</c>: the CDF search for <c>p &gt;= 1/3</c>, else the inversion
        ///     <c>ceil(-E / log1p(-p))</c> over a ziggurat exponential, clamped to <c>INT64_MAX</c>. Byte-identical to
        ///     <c>default_rng(seed).geometric</c>.
        /// </remarks>
        public NDArray geometric(double p, Shape size = default)
        {
            RandomConstraints.Check(p, "p", ConstraintType.CONS_BOUNDED_GT_0_1);
            bool search = p >= 0.333333333333333333333333;
            double log1pNegP = search ? 0.0 : Log1p(-p);

            if (IsNoSize(size))
            {
                unsafe
                {
                    lock (_bitGenerator.@lock)
                    {
                        if (search)
                        {
                            double word;
                            var one = new DrawBufferDouble(_bitGenerator, &word, 1);
                            return NDArray.Scalar(Distributions.RandomGeometricSearch(ref one, p));
                        }
                        ulong unit;
                        var one64 = new DrawBuffer64(_bitGenerator, &unit, 1);
                        return NDArray.Scalar(GeometricInversion(ref one64, log1pNegP));
                    }
                }
            }

            var ret = DistOutput(NPTypeCode.Int64, size);
            unsafe
            {
                var dst = (long*)ret.Address;
                long count = ret.size;
                if (search)
                {
                    double* storage = stackalloc double[DrawBufferDouble.Capacity];
                    var src = new DrawBufferDouble(_bitGenerator, storage, DrawBufferDouble.Capacity);
                    lock (_bitGenerator.@lock)
                        for (long i = 0; i < count; i++)
                        {
                            src.Owed = count - i;
                            dst[i] = Distributions.RandomGeometricSearch(ref src, p);
                        }
                }
                else
                {
                    ulong* storage = stackalloc ulong[DrawBuffer64.Capacity];
                    var src = new DrawBuffer64(_bitGenerator, storage, DrawBuffer64.Capacity);
                    lock (_bitGenerator.@lock)
                        for (long i = 0; i < count; i++)
                        {
                            src.Owed = count - i;
                            dst[i] = GeometricInversion(ref src, log1pNegP);
                        }
                }
            }
            return ret;
        }

        /// <summary>
        ///     Draw samples from a Hypergeometric distribution.
        /// </summary>
        /// <param name="ngood">Number of ways to make a good selection, non-negative and below <c>10^9</c>.</param>
        /// <param name="nbad">Number of ways to make a bad selection, non-negative and below <c>10^9</c>.</param>
        /// <param name="nsample">Number of items sampled, <c>0 &lt;= nsample &lt;= ngood + nbad</c>.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) returns a single value.</param>
        /// <returns>The int64 draws (good items in the sample).</returns>
        /// <exception cref="ValueError">In NumPy's order: <paramref name="ngood"/> or <paramref name="nbad"/> is at least
        ///     <c>10^9</c>; <c>ngood + nbad &lt; nsample</c>; then a negative <paramref name="ngood"/>,
        ///     <paramref name="nbad"/> or <paramref name="nsample"/> (<c>ngood &lt; 0</c>, ...).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.hypergeometric.html
        ///     <br/>NumPy's modern <c>random_hypergeometric</c> — a different algorithm from RandomState's: HRUA (with
        ///     <c>logfactorial</c>) for <c>10 &lt;= nsample &lt;= ngood + nbad - 10</c>, else an urn simulation over
        ///     <c>random_interval</c>. Byte-identical to <c>default_rng(seed).hypergeometric</c>.
        /// </remarks>
        public NDArray hypergeometric(long ngood, long nbad, long nsample, Shape size = default)
        {
            const long HypergeomMax = 1000000000;
            if (ngood >= HypergeomMax || nbad >= HypergeomMax)
                throw new ValueError($"both ngood and nbad must be less than {HypergeomMax}");
            if (ngood + nbad < nsample)
                throw new ValueError("ngood + nbad < nsample");
            RandomConstraints.Check((double)ngood, "ngood", ConstraintType.CONS_NON_NEGATIVE);
            RandomConstraints.Check((double)nbad, "nbad", ConstraintType.CONS_NON_NEGATIVE);
            RandomConstraints.Check((double)nsample, "nsample", ConstraintType.CONS_NON_NEGATIVE);

            bool hrua = (nsample >= 10) && (nsample <= ngood + nbad - 10);

            if (IsNoSize(size))
            {
                unsafe
                {
                    double word;
                    var one = new DrawBufferDouble(_bitGenerator, &word, 1);
                    lock (_bitGenerator.@lock)
                        return NDArray.Scalar(Hypergeometric(ref one, _bitGenerator, ngood, nbad, nsample));
                }
            }

            var ret = DistOutput(NPTypeCode.Int64, size);
            unsafe
            {
                var dst = (long*)ret.Address;
                long count = ret.size;
                lock (_bitGenerator.@lock)
                {
                    if (hrua)
                    {
                        // HRUA draws only doubles, at least two per value (one U/V pair per attempt): a read-ahead of
                        // them, over the setup evaluated once and a memo of each candidate's logfactorial sum.
                        var setup = new HruaSetup(ngood, nbad, nsample);
                        double[] gpMemo = setup.NewGpMemo(count);
                        double* storage = stackalloc double[DrawBufferDouble.Capacity];
                        var src = new DrawBufferDouble(_bitGenerator, storage, DrawBufferDouble.Capacity);
                        for (long i = 0; i < count; i++)
                        {
                            src.Owed = count - i;
                            dst[i] = HypergeometricHrua(ref src, in setup, gpMemo);
                        }
                    }
                    else
                    {
                        // The urn walk draws only next_uint32 words (every bound is below 2*10^9). With constant
                        // parameters a value draws at least one word either always or never (see HypergeometricSample),
                        // so a 32-bit read-ahead sized by the values still owed never draws past NumPy's stream; when no
                        // value draws, it is never refilled.
                        HypergeometricSampleFill(_bitGenerator, dst, count, ngood, nbad, nsample);
                    }
                }
            }
            return ret;
        }

        /// <summary>
        ///     Draw samples from a logarithmic series distribution.
        /// </summary>
        /// <param name="p">Shape parameter, in <c>[0, 1)</c> (NaN rejected).</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) returns a single value.</param>
        /// <returns>The int64 draws (<c>&gt;= 1</c>).</returns>
        /// <exception cref="ValueError"><paramref name="p"/> is outside <c>[0, 1)</c> or NaN (<c>p &lt; 0, p &gt;= 1 or p is NaN</c>).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.logseries.html
        ///     <br/>NumPy's modern <c>random_logseries</c> (<c>log1p</c>/<c>expm1</c> spellings). The draws are NumPy's; a
        ///     count can differ only where Windows NumPy's <c>expm1</c> and <see cref="Expm1"/> part by an ulp AND a
        ///     uniform lands inside that ulp of a boundary (see <see cref="Logseries"/>).
        /// </remarks>
        public NDArray logseries(double p, Shape size = default)
        {
            RandomConstraints.Check(p, "p", ConstraintType.CONS_BOUNDED_LT_0_1);
            double r = Log1p(-p);

            if (IsNoSize(size))
            {
                unsafe
                {
                    double word;
                    var one = new DrawBufferDouble(_bitGenerator, &word, 1);
                    lock (_bitGenerator.@lock)
                        return NDArray.Scalar(Logseries(ref one, p, r));
                }
            }

            var ret = DistOutput(NPTypeCode.Int64, size);
            unsafe
            {
                var dst = (long*)ret.Address;
                long count = ret.size;
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                var src = new DrawBufferDouble(_bitGenerator, storage, DrawBufferDouble.Capacity);
                lock (_bitGenerator.@lock)
                    for (long i = 0; i < count; i++)
                    {
                        src.Owed = count - i;
                        dst[i] = Logseries(ref src, p, r);
                    }
            }
            return ret;
        }
    }
}
