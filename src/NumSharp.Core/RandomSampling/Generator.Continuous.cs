using System;

namespace NumSharp
{
    public sealed partial class Generator
    {
        // The continuous distributions of NumPy's Generator (_generator.pyx + distributions.c). Each validates its
        // parameters with NumPy's constraints and messages, in NumPy's order, BEFORE drawing; `size = default` is NumPy's
        // `None` (a 0-d float64 array holding one draw) and `()` a 0-d array. Sized draws go through a read-ahead buffer
        // (DrawBuffer64 for samplers that draw ziggurat words, DrawBufferDouble for next_double-only ones) whose Owed is
        // the values still owed: with fixed parameters every value draws at least once, or none ever does (then the
        // buffer never refills), so the stream and the engine state stay NumPy's. The per-parameter setup NumPy
        // recomputes for every value is evaluated once (the same expressions — bit-neutral).

        /// <summary>
        ///     Allocates a distribution's output — NumPy's <c>np.empty(size, dtype)</c> — from <paramref name="size"/>'s
        ///     DIMENSIONS, so a size taken from a view (with its strides/offset) cannot make the allocation too small.
        /// </summary>
        /// <param name="typeCode">The output dtype.</param>
        /// <param name="size">The requested shape.</param>
        /// <returns>A fresh, uninitialized C-contiguous array.</returns>
        /// <exception cref="ValueError">A dimension is negative.</exception>
        private static NDArray DistOutput(NPTypeCode typeCode, Shape size) => new NDArray(typeCode, new Shape(size.dimensions), false);

        /// <summary>
        ///     Draw samples from a Beta distribution.
        /// </summary>
        /// <param name="a">Alpha, positive (<c>&gt; 0</c>; NaN is accepted and samples NaN, as in NumPy).</param>
        /// <param name="b">Beta, positive (<c>&gt; 0</c>; NaN accepted).</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) returns a single value.</param>
        /// <returns>The float64 draws, in <c>[0, 1]</c>.</returns>
        /// <exception cref="ValueError"><paramref name="a"/> (<c>a &lt;= 0</c>, checked first) or <paramref name="b"/> is not positive.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.beta.html
        ///     <br/>NumPy's modern <c>random_beta</c>: when both shapes are <c>&lt;= 1</c>, Johnk's algorithm (with a
        ///     log-space fallback when a power underflows, and a single-uniform 0/1 shortcut when both shapes are below
        ///     <c>3e-103</c>); otherwise the ratio of two standard gammas. Byte-identical to
        ///     <c>default_rng(seed).beta</c>.
        /// </remarks>
        public NDArray beta(double a, double b, Shape size = default)
        {
            RandomConstraints.Check(a, "a", ConstraintType.CONS_POSITIVE);
            RandomConstraints.Check(b, "b", ConstraintType.CONS_POSITIVE);
            var setup = new BetaSetup(a, b);

            if (IsNoSize(size))
            {
                unsafe
                {
                    ulong word;
                    var one = new DrawBuffer64(_bitGenerator, &word, 1);
                    lock (_bitGenerator.@lock)
                        return NDArray.Scalar(Beta(ref one, in setup));
                }
            }

            var ret = DistOutput(NPTypeCode.Double, size);
            unsafe
            {
                var dst = (double*)ret.Address;
                long n = ret.size;
                ulong* storage = stackalloc ulong[DrawBuffer64.Capacity];
                var src = new DrawBuffer64(_bitGenerator, storage, DrawBuffer64.Capacity);
                lock (_bitGenerator.@lock)
                    for (long i = 0; i < n; i++)
                    {
                        src.Owed = n - i;
                        dst[i] = Beta(ref src, in setup);
                    }
            }
            return ret;
        }

        /// <summary>
        ///     Draw samples from a chi-square distribution.
        /// </summary>
        /// <param name="df">Degrees of freedom, positive (<c>&gt; 0</c>; NaN accepted and samples NaN).</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) returns a single value.</param>
        /// <returns>The float64 draws.</returns>
        /// <exception cref="ValueError"><paramref name="df"/> is not positive (<c>df &lt;= 0</c>).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.chisquare.html
        ///     <br/>NumPy's <c>random_chisquare</c>: <c>2 * random_standard_gamma(df / 2)</c>. Byte-identical to
        ///     <c>default_rng(seed).chisquare</c>.
        /// </remarks>
        public NDArray chisquare(double df, Shape size = default)
        {
            RandomConstraints.Check(df, "df", ConstraintType.CONS_POSITIVE);
            var half = new GammaSetup(df / 2.0);

            if (IsNoSize(size))
            {
                unsafe
                {
                    ulong word;
                    var one = new DrawBuffer64(_bitGenerator, &word, 1);
                    lock (_bitGenerator.@lock)
                        return NDArray.Scalar(Chisquare(ref one, in half));
                }
            }

            var ret = DistOutput(NPTypeCode.Double, size);
            unsafe
            {
                var dst = (double*)ret.Address;
                long n = ret.size;
                ulong* storage = stackalloc ulong[DrawBuffer64.Capacity];
                var src = new DrawBuffer64(_bitGenerator, storage, DrawBuffer64.Capacity);
                lock (_bitGenerator.@lock)
                    for (long i = 0; i < n; i++)
                    {
                        src.Owed = n - i;
                        dst[i] = Chisquare(ref src, in half);
                    }
            }
            return ret;
        }

        /// <summary>
        ///     Draw samples from an F distribution.
        /// </summary>
        /// <param name="dfnum">Degrees of freedom in the numerator, positive (NaN accepted).</param>
        /// <param name="dfden">Degrees of freedom in the denominator, positive (NaN accepted).</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) returns a single value.</param>
        /// <returns>The float64 draws.</returns>
        /// <exception cref="ValueError"><paramref name="dfnum"/> (checked first) or <paramref name="dfden"/> is not positive.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.f.html
        ///     <br/>NumPy's <c>random_f</c>: <c>(chisquare(dfnum) * dfden) / (chisquare(dfden) * dfnum)</c>, the numerator
        ///     drawn first. Byte-identical to <c>default_rng(seed).f</c>.
        /// </remarks>
        public NDArray f(double dfnum, double dfden, Shape size = default)
        {
            RandomConstraints.Check(dfnum, "dfnum", ConstraintType.CONS_POSITIVE);
            RandomConstraints.Check(dfden, "dfden", ConstraintType.CONS_POSITIVE);
            var halfNum = new GammaSetup(dfnum / 2.0);
            var halfDen = new GammaSetup(dfden / 2.0);

            if (IsNoSize(size))
            {
                unsafe
                {
                    ulong word;
                    var one = new DrawBuffer64(_bitGenerator, &word, 1);
                    lock (_bitGenerator.@lock)
                        return NDArray.Scalar(F(ref one, in halfNum, in halfDen, dfnum, dfden));
                }
            }

            var ret = DistOutput(NPTypeCode.Double, size);
            unsafe
            {
                var dst = (double*)ret.Address;
                long n = ret.size;
                ulong* storage = stackalloc ulong[DrawBuffer64.Capacity];
                var src = new DrawBuffer64(_bitGenerator, storage, DrawBuffer64.Capacity);
                lock (_bitGenerator.@lock)
                    for (long i = 0; i < n; i++)
                    {
                        src.Owed = n - i;
                        dst[i] = F(ref src, in halfNum, in halfDen, dfnum, dfden);
                    }
            }
            return ret;
        }

        /// <summary>
        ///     Draw samples from a noncentral chi-square distribution.
        /// </summary>
        /// <param name="df">Degrees of freedom, positive (NaN accepted).</param>
        /// <param name="nonc">Non-centrality, non-negative (<c>-0.0</c> rejected; NaN accepted and returns NaN without drawing).</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) returns a single value.</param>
        /// <returns>The float64 draws.</returns>
        /// <exception cref="ValueError"><paramref name="df"/> is not positive (checked first) or <paramref name="nonc"/> is negative.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.noncentral_chisquare.html
        ///     <br/>NumPy's modern <c>random_noncentral_chisquare</c> — unlike the legacy sampler it tests a NaN
        ///     non-centrality BEFORE drawing (so it advances nothing). Byte-identical to
        ///     <c>default_rng(seed).noncentral_chisquare</c>.
        /// </remarks>
        public NDArray noncentral_chisquare(double df, double nonc, Shape size = default)
        {
            RandomConstraints.Check(df, "df", ConstraintType.CONS_POSITIVE);
            RandomConstraints.Check(nonc, "nonc", ConstraintType.CONS_NON_NEGATIVE);
            var setup = new NoncentralChisquareSetup(df, nonc);

            if (IsNoSize(size))
            {
                unsafe
                {
                    ulong word;
                    var one = new DrawBuffer64(_bitGenerator, &word, 1);
                    lock (_bitGenerator.@lock)
                        return NDArray.Scalar(NoncentralChisquare(ref one, in setup));
                }
            }

            var ret = DistOutput(NPTypeCode.Double, size);
            unsafe
            {
                var dst = (double*)ret.Address;
                long n = ret.size;
                ulong* storage = stackalloc ulong[DrawBuffer64.Capacity];
                var src = new DrawBuffer64(_bitGenerator, storage, DrawBuffer64.Capacity);
                lock (_bitGenerator.@lock)
                    for (long i = 0; i < n; i++)
                    {
                        src.Owed = n - i;
                        dst[i] = NoncentralChisquare(ref src, in setup);
                    }
            }
            return ret;
        }

        /// <summary>
        ///     Draw samples from the noncentral F distribution.
        /// </summary>
        /// <param name="dfnum">Numerator degrees of freedom, positive (NaN accepted).</param>
        /// <param name="dfden">Denominator degrees of freedom, positive (NaN accepted).</param>
        /// <param name="nonc">Non-centrality, non-negative (<c>-0.0</c> rejected; NaN accepted).</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) returns a single value.</param>
        /// <returns>The float64 draws.</returns>
        /// <exception cref="ValueError">In NumPy's order: <paramref name="dfnum"/> or <paramref name="dfden"/> is not positive, or <paramref name="nonc"/> is negative.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.noncentral_f.html
        ///     <br/>NumPy's <c>random_noncentral_f</c>: <c>(ncchi2(dfnum, nonc) * dfden) / (chi2(dfden) * dfnum)</c>.
        ///     Byte-identical to <c>default_rng(seed).noncentral_f</c>.
        /// </remarks>
        public NDArray noncentral_f(double dfnum, double dfden, double nonc, Shape size = default)
        {
            RandomConstraints.Check(dfnum, "dfnum", ConstraintType.CONS_POSITIVE);
            RandomConstraints.Check(dfden, "dfden", ConstraintType.CONS_POSITIVE);
            RandomConstraints.Check(nonc, "nonc", ConstraintType.CONS_NON_NEGATIVE);
            var num = new NoncentralChisquareSetup(dfnum, nonc);
            var halfDen = new GammaSetup(dfden / 2.0);

            if (IsNoSize(size))
            {
                unsafe
                {
                    ulong word;
                    var one = new DrawBuffer64(_bitGenerator, &word, 1);
                    lock (_bitGenerator.@lock)
                        return NDArray.Scalar(NoncentralF(ref one, in num, in halfDen, dfnum, dfden));
                }
            }

            var ret = DistOutput(NPTypeCode.Double, size);
            unsafe
            {
                var dst = (double*)ret.Address;
                long n = ret.size;
                ulong* storage = stackalloc ulong[DrawBuffer64.Capacity];
                var src = new DrawBuffer64(_bitGenerator, storage, DrawBuffer64.Capacity);
                lock (_bitGenerator.@lock)
                    for (long i = 0; i < n; i++)
                    {
                        src.Owed = n - i;
                        dst[i] = NoncentralF(ref src, in num, in halfDen, dfnum, dfden);
                    }
            }
            return ret;
        }

        /// <summary>
        ///     Draw samples from a standard Cauchy distribution with mode = 0.
        /// </summary>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) returns a single value.</param>
        /// <returns>The float64 draws.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.standard_cauchy.html
        ///     <br/>NumPy's <c>random_standard_cauchy</c>: the ratio of two ziggurat normals, numerator first — so a fill is
        ///     the Generator's normal fill of consecutive pairs, divided (the same division, the same bits).
        ///     Byte-identical to <c>default_rng(seed).standard_cauchy</c>.
        /// </remarks>
        public NDArray standard_cauchy(Shape size = default)
        {
            if (IsNoSize(size))
            {
                unsafe
                {
                    ulong word;
                    var one = new DrawBuffer64(_bitGenerator, &word, 1);
                    lock (_bitGenerator.@lock)
                        return NDArray.Scalar(StandardCauchy(ref one));
                }
            }

            var ret = DistOutput(NPTypeCode.Double, size);
            unsafe
            {
                const int Chunk = 512;
                var dst = (double*)ret.Address;
                long n = ret.size;
                double* normals = stackalloc double[2 * Chunk];
                lock (_bitGenerator.@lock)
                    for (long i = 0; i < n; i += Chunk)
                    {
                        long m = n - i < Chunk ? n - i : Chunk;
                        // Every value is exactly two normals, so each chunk's normals are the stream's next 2m normals.
                        FillStandardNormal(normals, 2 * m);
                        for (long j = 0; j < m; j++)
                            dst[i + j] = normals[2 * j] / normals[2 * j + 1];
                    }
            }
            return ret;
        }

        /// <summary>
        ///     Draw samples from a standard Student's t distribution with <paramref name="df"/> degrees of freedom.
        /// </summary>
        /// <param name="df">Degrees of freedom, positive (NaN accepted).</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) returns a single value.</param>
        /// <returns>The float64 draws.</returns>
        /// <exception cref="ValueError"><paramref name="df"/> is not positive (<c>df &lt;= 0</c>).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.standard_t.html
        ///     <br/>NumPy's <c>random_standard_t</c>: <c>sqrt(df/2) * N / sqrt(G(df/2))</c>, the normal drawn first.
        ///     Byte-identical to <c>default_rng(seed).standard_t</c>.
        /// </remarks>
        public NDArray standard_t(double df, Shape size = default)
        {
            RandomConstraints.Check(df, "df", ConstraintType.CONS_POSITIVE);
            var half = new GammaSetup(df / 2);
            double sqrtHalf = Math.Sqrt(df / 2);

            if (IsNoSize(size))
            {
                unsafe
                {
                    ulong word;
                    var one = new DrawBuffer64(_bitGenerator, &word, 1);
                    lock (_bitGenerator.@lock)
                        return NDArray.Scalar(StandardT(ref one, in half, sqrtHalf));
                }
            }

            var ret = DistOutput(NPTypeCode.Double, size);
            unsafe
            {
                var dst = (double*)ret.Address;
                long n = ret.size;
                ulong* storage = stackalloc ulong[DrawBuffer64.Capacity];
                var src = new DrawBuffer64(_bitGenerator, storage, DrawBuffer64.Capacity);
                lock (_bitGenerator.@lock)
                    for (long i = 0; i < n; i++)
                    {
                        src.Owed = n - i;
                        dst[i] = StandardT(ref src, in half, sqrtHalf);
                    }
            }
            return ret;
        }

        /// <summary>
        ///     Draw samples from a von Mises distribution.
        /// </summary>
        /// <param name="mu">Mode ("center") of the distribution.</param>
        /// <param name="kappa">Concentration, non-negative (<c>-0.0</c> rejected; NaN accepted and returns NaN without drawing).</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) returns a single value.</param>
        /// <returns>The float64 draws, in <c>[-pi, pi]</c>.</returns>
        /// <exception cref="ValueError"><paramref name="kappa"/> is negative (<c>kappa &lt; 0</c>).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.vonmises.html
        ///     <br/>NumPy's modern <c>random_vonmises</c>: a uniform below <c>kappa = 1e-8</c>, Best &amp; Fisher's
        ///     wrapped-Cauchy rejection up to <c>1e6</c>, the wrapped normal above (so <c>kappa = inf</c> returns
        ///     <paramref name="mu"/>). Byte-identical to <c>default_rng(seed).vonmises</c>.
        /// </remarks>
        public NDArray vonmises(double mu, double kappa, Shape size = default)
        {
            RandomConstraints.Check(kappa, "kappa", ConstraintType.CONS_NON_NEGATIVE);
            var setup = new VonmisesSetup(mu, kappa);

            if (IsNoSize(size))
            {
                unsafe
                {
                    ulong word;
                    var one = new DrawBuffer64(_bitGenerator, &word, 1);
                    lock (_bitGenerator.@lock)
                        return NDArray.Scalar(Vonmises(ref one, in setup));
                }
            }

            var ret = DistOutput(NPTypeCode.Double, size);
            unsafe
            {
                var dst = (double*)ret.Address;
                long n = ret.size;
                ulong* storage = stackalloc ulong[DrawBuffer64.Capacity];
                var src = new DrawBuffer64(_bitGenerator, storage, DrawBuffer64.Capacity);
                lock (_bitGenerator.@lock)
                    for (long i = 0; i < n; i++)
                    {
                        src.Owed = n - i;
                        dst[i] = Vonmises(ref src, in setup);
                    }
            }
            return ret;
        }

        /// <summary>
        ///     Draw samples from a Pareto II (AKA Lomax) distribution with specified shape.
        /// </summary>
        /// <param name="a">Shape of the distribution, positive (NaN accepted).</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) returns a single value.</param>
        /// <returns>The float64 draws.</returns>
        /// <exception cref="ValueError"><paramref name="a"/> is not positive (<c>a &lt;= 0</c>).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.pareto.html
        ///     <br/>NumPy's modern <c>random_pareto</c>: <c>expm1(E / a)</c> over a ziggurat exponential. The draws are
        ///     NumPy's; the value is bit-identical wherever <c>E / a</c> lies outside <c>[-ln 2, ln 1.5]</c> and can sit up
        ///     to 2 ulp from Windows NumPy's inside it, where the MSVC <c>expm1</c> uses an approximation no public formula
        ///     reproduces (see <see cref="Expm1"/>). Measured over 20000 draws: 30% of <c>pareto(3)</c> values differ (1% by
        ///     2 ulp), 8% of <c>pareto(0.5)</c>.
        /// </remarks>
        public NDArray pareto(double a, Shape size = default)
        {
            RandomConstraints.Check(a, "a", ConstraintType.CONS_POSITIVE);

            lock (_bitGenerator.@lock)
            {
                if (IsNoSize(size))
                    return NDArray.Scalar(Expm1(NextStandardExponential() / a));

                var ret = DistOutput(NPTypeCode.Double, size);
                unsafe
                {
                    var dst = (double*)ret.Address;
                    long n = ret.size;
                    // Exactly one exponential per value: the exponential fill, then random_pareto's transform in place.
                    FillStandardExponential(dst, n);
                    for (long i = 0; i < n; i++)
                        dst[i] = Expm1(dst[i] / a);
                }
                return ret;
            }
        }

        /// <summary>
        ///     Draw samples from a Weibull distribution.
        /// </summary>
        /// <param name="a">Shape parameter, non-negative (<c>-0.0</c> rejected; NaN accepted). 0 returns 0 without drawing.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) returns a single value.</param>
        /// <returns>The float64 draws.</returns>
        /// <exception cref="ValueError"><paramref name="a"/> is negative (<c>a &lt; 0</c>).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.weibull.html
        ///     <br/>NumPy's <c>random_weibull</c>: 0 (no draw) for <c>a == 0</c>, else <c>E^(1/a)</c> over a ziggurat
        ///     exponential. Byte-identical to <c>default_rng(seed).weibull</c>.
        /// </remarks>
        public NDArray weibull(double a, Shape size = default)
        {
            RandomConstraints.Check(a, "a", ConstraintType.CONS_NON_NEGATIVE);

            lock (_bitGenerator.@lock)
            {
                if (IsNoSize(size))
                    return NDArray.Scalar(a == 0.0 ? 0.0 : Math.Pow(NextStandardExponential(), 1.0 / a));

                if (a == 0.0)
                    return np.zeros(new Shape(size.dimensions), NPTypeCode.Double); // random_weibull(0) draws nothing

                var ret = DistOutput(NPTypeCode.Double, size);
                unsafe
                {
                    var dst = (double*)ret.Address;
                    long n = ret.size;
                    FillStandardExponential(dst, n);
                    double invA = 1.0 / a; // NumPy's per-value 1. / a
                    for (long i = 0; i < n; i++)
                        dst[i] = Math.Pow(dst[i], invA);
                }
                return ret;
            }
        }

        /// <summary>
        ///     Draws samples in [0, 1] from a power distribution with positive exponent a - 1.
        /// </summary>
        /// <param name="a">Parameter of the distribution, positive (NaN accepted).</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) returns a single value.</param>
        /// <returns>The float64 draws.</returns>
        /// <exception cref="ValueError"><paramref name="a"/> is not positive (<c>a &lt;= 0</c>).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.power.html
        ///     <br/>NumPy's modern <c>random_power</c>: <c>(-expm1(-E))^(1/a)</c> over a ziggurat exponential; like
        ///     <see cref="pareto"/>, bit-identical except where <c>-E</c> lies in <c>[-ln 2, 0]</c>, where the MSVC
        ///     <c>expm1</c> can differ by up to 2 ulp (see <see cref="Expm1"/>) — and the power <c>1/a</c> scales that
        ///     relative difference: measured over 20000 draws, <c>power(2.5)</c> stays within 1 ulp (8% of values differ)
        ///     while <c>power(0.3)</c> reaches 9 ulp (20% of values).
        /// </remarks>
        public NDArray power(double a, Shape size = default)
        {
            RandomConstraints.Check(a, "a", ConstraintType.CONS_POSITIVE);

            lock (_bitGenerator.@lock)
            {
                if (IsNoSize(size))
                    return NDArray.Scalar(Math.Pow(-Expm1(-NextStandardExponential()), 1.0 / a));

                var ret = DistOutput(NPTypeCode.Double, size);
                unsafe
                {
                    var dst = (double*)ret.Address;
                    long n = ret.size;
                    FillStandardExponential(dst, n);
                    double invA = 1.0 / a;
                    for (long i = 0; i < n; i++)
                        dst[i] = Math.Pow(-Expm1(-dst[i]), invA);
                }
                return ret;
            }
        }

        /// <summary>
        ///     Draw samples from the Laplace or double exponential distribution with specified location and scale.
        /// </summary>
        /// <param name="loc">The position of the distribution peak. Default 0.</param>
        /// <param name="scale">The exponential decay, non-negative (<c>-0.0</c> rejected; NaN accepted). Default 1.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) returns a single value.</param>
        /// <returns>The float64 draws.</returns>
        /// <exception cref="ValueError"><paramref name="scale"/> is negative (<c>scale &lt; 0</c>).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.laplace.html
        ///     <br/>NumPy's <c>random_laplace</c> (shared with RandomState), one uniform per value.
        /// </remarks>
        public NDArray laplace(double loc = 0.0, double scale = 1.0, Shape size = default)
        {
            RandomConstraints.Check(scale, "scale", ConstraintType.CONS_NON_NEGATIVE);

            if (IsNoSize(size))
            {
                unsafe
                {
                    double word;
                    var one = new DrawBufferDouble(_bitGenerator, &word, 1);
                    lock (_bitGenerator.@lock)
                        return NDArray.Scalar(Distributions.RandomLaplace(ref one, loc, scale));
                }
            }

            var ret = DistOutput(NPTypeCode.Double, size);
            unsafe
            {
                var dst = (double*)ret.Address;
                long n = ret.size;
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                var src = new DrawBufferDouble(_bitGenerator, storage, DrawBufferDouble.Capacity);
                lock (_bitGenerator.@lock)
                    for (long i = 0; i < n; i++)
                    {
                        src.Owed = n - i;
                        dst[i] = Distributions.RandomLaplace(ref src, loc, scale);
                    }
            }
            return ret;
        }

        /// <summary>
        ///     Draw samples from a Gumbel distribution.
        /// </summary>
        /// <param name="loc">The location of the mode. Default 0.</param>
        /// <param name="scale">The scale parameter, non-negative (<c>-0.0</c> rejected; NaN accepted). Default 1.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) returns a single value.</param>
        /// <returns>The float64 draws.</returns>
        /// <exception cref="ValueError"><paramref name="scale"/> is negative (<c>scale &lt; 0</c>).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.gumbel.html
        ///     <br/>NumPy's <c>random_gumbel</c> (shared with RandomState), one uniform per value.
        /// </remarks>
        public NDArray gumbel(double loc = 0.0, double scale = 1.0, Shape size = default)
        {
            RandomConstraints.Check(scale, "scale", ConstraintType.CONS_NON_NEGATIVE);

            if (IsNoSize(size))
            {
                unsafe
                {
                    double word;
                    var one = new DrawBufferDouble(_bitGenerator, &word, 1);
                    lock (_bitGenerator.@lock)
                        return NDArray.Scalar(Distributions.RandomGumbel(ref one, loc, scale));
                }
            }

            var ret = DistOutput(NPTypeCode.Double, size);
            unsafe
            {
                var dst = (double*)ret.Address;
                long n = ret.size;
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                var src = new DrawBufferDouble(_bitGenerator, storage, DrawBufferDouble.Capacity);
                lock (_bitGenerator.@lock)
                    for (long i = 0; i < n; i++)
                    {
                        src.Owed = n - i;
                        dst[i] = Distributions.RandomGumbel(ref src, loc, scale);
                    }
            }
            return ret;
        }

        /// <summary>
        ///     Draw samples from a logistic distribution.
        /// </summary>
        /// <param name="loc">The mean. Default 0.</param>
        /// <param name="scale">The scale, non-negative (<c>-0.0</c> rejected; NaN accepted). Default 1.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) returns a single value.</param>
        /// <returns>The float64 draws.</returns>
        /// <exception cref="ValueError"><paramref name="scale"/> is negative (<c>scale &lt; 0</c>).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.logistic.html
        ///     <br/>NumPy's <c>random_logistic</c> (shared with RandomState), one uniform per value.
        /// </remarks>
        public NDArray logistic(double loc = 0.0, double scale = 1.0, Shape size = default)
        {
            RandomConstraints.Check(scale, "scale", ConstraintType.CONS_NON_NEGATIVE);

            if (IsNoSize(size))
            {
                unsafe
                {
                    double word;
                    var one = new DrawBufferDouble(_bitGenerator, &word, 1);
                    lock (_bitGenerator.@lock)
                        return NDArray.Scalar(Distributions.RandomLogistic(ref one, loc, scale));
                }
            }

            var ret = DistOutput(NPTypeCode.Double, size);
            unsafe
            {
                var dst = (double*)ret.Address;
                long n = ret.size;
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                var src = new DrawBufferDouble(_bitGenerator, storage, DrawBufferDouble.Capacity);
                lock (_bitGenerator.@lock)
                    for (long i = 0; i < n; i++)
                    {
                        src.Owed = n - i;
                        dst[i] = Distributions.RandomLogistic(ref src, loc, scale);
                    }
            }
            return ret;
        }

        /// <summary>
        ///     Draw samples from a log-normal distribution.
        /// </summary>
        /// <param name="mean">Mean of the underlying normal distribution. Default 0.</param>
        /// <param name="sigma">Standard deviation of the underlying normal, non-negative (<c>-0.0</c> rejected; NaN accepted). Default 1.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) returns a single value.</param>
        /// <returns>The float64 draws.</returns>
        /// <exception cref="ValueError"><paramref name="sigma"/> is negative (<c>sigma &lt; 0</c>).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.lognormal.html
        ///     <br/>NumPy's <c>random_lognormal</c>: <c>exp(mean + sigma * N)</c> over a ziggurat normal — a fill is the
        ///     normal fill, then the transform in place. Byte-identical to <c>default_rng(seed).lognormal</c>.
        /// </remarks>
        public NDArray lognormal(double mean = 0.0, double sigma = 1.0, Shape size = default)
        {
            RandomConstraints.Check(sigma, "sigma", ConstraintType.CONS_NON_NEGATIVE);

            lock (_bitGenerator.@lock)
            {
                if (IsNoSize(size))
                    return NDArray.Scalar(Math.Exp(mean + sigma * NextStandardNormal()));

                var ret = DistOutput(NPTypeCode.Double, size);
                unsafe
                {
                    var dst = (double*)ret.Address;
                    long n = ret.size;
                    FillStandardNormal(dst, n);
                    for (long i = 0; i < n; i++)
                        dst[i] = Math.Exp(mean + sigma * dst[i]);
                }
                return ret;
            }
        }

        /// <summary>
        ///     Draw samples from a Rayleigh distribution.
        /// </summary>
        /// <param name="scale">Scale, also equals the mode; non-negative (<c>-0.0</c> rejected; NaN accepted). Default 1.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) returns a single value.</param>
        /// <returns>The float64 draws.</returns>
        /// <exception cref="ValueError"><paramref name="scale"/> is negative (<c>scale &lt; 0</c>).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.rayleigh.html
        ///     <br/>NumPy's modern <c>random_rayleigh</c>: <c>mode * sqrt(2 * E)</c> over a ziggurat exponential (the legacy
        ///     sampler uses <c>-log1p(-U)</c> instead). Byte-identical to <c>default_rng(seed).rayleigh</c>.
        /// </remarks>
        public NDArray rayleigh(double scale = 1.0, Shape size = default)
        {
            RandomConstraints.Check(scale, "scale", ConstraintType.CONS_NON_NEGATIVE);

            lock (_bitGenerator.@lock)
            {
                if (IsNoSize(size))
                    return NDArray.Scalar(scale * Math.Sqrt(2.0 * NextStandardExponential()));

                var ret = DistOutput(NPTypeCode.Double, size);
                unsafe
                {
                    var dst = (double*)ret.Address;
                    long n = ret.size;
                    FillStandardExponential(dst, n);
                    for (long i = 0; i < n; i++)
                        dst[i] = scale * Math.Sqrt(2.0 * dst[i]);
                }
                return ret;
            }
        }

        /// <summary>
        ///     Draw samples from a Wald, or inverse Gaussian, distribution.
        /// </summary>
        /// <param name="mean">Distribution mean, positive (NaN accepted).</param>
        /// <param name="scale">Scale parameter, positive (NaN accepted).</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) returns a single value.</param>
        /// <returns>The float64 draws.</returns>
        /// <exception cref="ValueError"><paramref name="mean"/> (checked first) or <paramref name="scale"/> is not positive.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.wald.html
        ///     <br/>NumPy's modern <c>random_wald</c> (the cancellation-free form), one ziggurat normal and one uniform per
        ///     value. Byte-identical to <c>default_rng(seed).wald</c>.
        /// </remarks>
        public NDArray wald(double mean, double scale, Shape size = default)
        {
            RandomConstraints.Check(mean, "mean", ConstraintType.CONS_POSITIVE);
            RandomConstraints.Check(scale, "scale", ConstraintType.CONS_POSITIVE);

            if (IsNoSize(size))
            {
                unsafe
                {
                    ulong word;
                    var one = new DrawBuffer64(_bitGenerator, &word, 1);
                    lock (_bitGenerator.@lock)
                        return NDArray.Scalar(Wald(ref one, mean, scale));
                }
            }

            var ret = DistOutput(NPTypeCode.Double, size);
            unsafe
            {
                var dst = (double*)ret.Address;
                long n = ret.size;
                ulong* storage = stackalloc ulong[DrawBuffer64.Capacity];
                var src = new DrawBuffer64(_bitGenerator, storage, DrawBuffer64.Capacity);
                lock (_bitGenerator.@lock)
                    for (long i = 0; i < n; i++)
                    {
                        src.Owed = n - i;
                        dst[i] = Wald(ref src, mean, scale);
                    }
            }
            return ret;
        }

        /// <summary>
        ///     Draw samples from the triangular distribution over the interval <c>[left, right]</c>.
        /// </summary>
        /// <param name="left">Lower limit.</param>
        /// <param name="mode">The peak; <c>left &lt;= mode &lt;= right</c>.</param>
        /// <param name="right">Upper limit, greater than <paramref name="left"/>.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) returns a single value.</param>
        /// <returns>The float64 draws.</returns>
        /// <exception cref="ValueError">In NumPy's order: <c>left &gt; mode</c>, <c>mode &gt; right</c>, <c>left == right</c>.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.triangular.html
        ///     <br/>NumPy's <c>random_triangular</c> (shared with RandomState), one uniform per value. A NaN limit passes
        ///     the checks (every comparison is false) and samples NaN, as in NumPy.
        /// </remarks>
        public NDArray triangular(double left, double mode, double right, Shape size = default)
        {
            if (left > mode)
                throw new ValueError("left > mode");
            if (mode > right)
                throw new ValueError("mode > right");
            if (left == right)
                throw new ValueError("left == right");

            if (IsNoSize(size))
            {
                unsafe
                {
                    double word;
                    var one = new DrawBufferDouble(_bitGenerator, &word, 1);
                    lock (_bitGenerator.@lock)
                        return NDArray.Scalar(Distributions.RandomTriangular(ref one, left, mode, right));
                }
            }

            var ret = DistOutput(NPTypeCode.Double, size);
            unsafe
            {
                var dst = (double*)ret.Address;
                long n = ret.size;
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                var src = new DrawBufferDouble(_bitGenerator, storage, DrawBufferDouble.Capacity);
                lock (_bitGenerator.@lock)
                    for (long i = 0; i < n; i++)
                    {
                        src.Owed = n - i;
                        dst[i] = Distributions.RandomTriangular(ref src, left, mode, right);
                    }
            }
            return ret;
        }
    }
}
