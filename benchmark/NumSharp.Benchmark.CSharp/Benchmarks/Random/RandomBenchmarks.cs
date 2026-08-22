using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.CSharp.Infrastructure;

namespace NumSharp.Benchmark.CSharp.Benchmarks.Random;

[BenchmarkCategory("Random", "Continuous")]
public class ContinuousRandomBenchmarks : BenchmarkBase
{
    [Params(ArraySizeSource.Small, ArraySizeSource.Medium)]
    public override int N { get; set; }

    private Shape Size => new(N);

    [GlobalSetup]
    public void Setup() => np.random.seed(Seed);

    [Benchmark(Description = "np.random.beta(a, b)")] public NDArray Beta() => np.random.beta(2.0, 5.0, Size);
    [Benchmark(Description = "np.random.chisquare(df)")] public NDArray ChiSquare() => np.random.chisquare(4.0, Size);
    [Benchmark(Description = "np.random.exponential(scale)")] public NDArray Exponential() => np.random.exponential(2.0, Size);
    [Benchmark(Description = "np.random.f(dfnum, dfden)")] public NDArray F() => np.random.f(5.0, 7.0, Size);
    [Benchmark(Description = "np.random.gamma(shape, scale)")] public NDArray Gamma() => np.random.gamma(2.0, 3.0, Size);
    [Benchmark(Description = "np.random.gumbel(loc, scale)")] public NDArray Gumbel() => np.random.gumbel(0.0, 1.0, Size);
    [Benchmark(Description = "np.random.laplace(loc, scale)")] public NDArray Laplace() => np.random.laplace(0.0, 1.0, Size);
    [Benchmark(Description = "np.random.logistic(loc, scale)")] public NDArray Logistic() => np.random.logistic(0.0, 1.0, Size);
    [Benchmark(Description = "np.random.lognormal(mean, sigma)")] public NDArray LogNormal() => np.random.lognormal(0.0, 1.0, Size);
    [Benchmark(Description = "np.random.noncentral_chisquare(df, nonc)")] public NDArray NoncentralChiSquare() => np.random.noncentral_chisquare(4.0, 1.5, Size);
    [Benchmark(Description = "np.random.noncentral_f(dfnum, dfden, nonc)")] public NDArray NoncentralF() => np.random.noncentral_f(5.0, 7.0, 1.5, Size);
    [Benchmark(Description = "np.random.normal(loc, scale)")] public NDArray Normal() => np.random.normal(0.0, 1.0, Size);
    [Benchmark(Description = "np.random.pareto(a)")] public NDArray Pareto() => np.random.pareto(3.0, Size);
    [Benchmark(Description = "np.random.power(a)")] public NDArray Power() => np.random.power(3.0, Size);
    [Benchmark(Description = "np.random.rand(n)")] public NDArray Rand() => np.random.rand(N);
    [Benchmark(Description = "np.random.randn(n)")] public NDArray RandN() => np.random.randn(N);
    [Benchmark(Description = "np.random.random(n)")] public NDArray Random() => np.random.random(N);
    [Benchmark(Description = "np.random.random_sample(n)")] public NDArray RandomSample() => np.random.random_sample(N);
    [Benchmark(Description = "np.random.rayleigh(scale)")] public NDArray Rayleigh() => np.random.rayleigh(2.0, Size);
    [Benchmark(Description = "np.random.standard_cauchy(n)")] public NDArray StandardCauchy() => np.random.standard_cauchy(Size);
    [Benchmark(Description = "np.random.standard_exponential(n)")] public NDArray StandardExponential() => np.random.standard_exponential(Size);
    [Benchmark(Description = "np.random.standard_gamma(shape)")] public NDArray StandardGamma() => np.random.standard_gamma(2.0, Size);
    [Benchmark(Description = "np.random.standard_normal(n)")] public NDArray StandardNormal() => np.random.standard_normal(Size);
    [Benchmark(Description = "np.random.standard_t(df)")] public NDArray StandardT() => np.random.standard_t(5.0, Size);
    [Benchmark(Description = "np.random.triangular(left, mode, right)")] public NDArray Triangular() => np.random.triangular(-1.0, 0.0, 2.0, Size);
    [Benchmark(Description = "np.random.uniform(low, high)")] public NDArray Uniform() => np.random.uniform(-1.0, 2.0, Size);
    [Benchmark(Description = "np.random.vonmises(mu, kappa)")] public NDArray VonMises() => np.random.vonmises(0.0, 2.0, Size);
    [Benchmark(Description = "np.random.wald(mean, scale)")] public NDArray Wald() => np.random.wald(2.0, 1.0, Size);
    [Benchmark(Description = "np.random.weibull(a)")] public NDArray Weibull() => np.random.weibull(2.0, Size);
}

[BenchmarkCategory("Random", "Discrete")]
public class DiscreteRandomBenchmarks : TypedBenchmarkBase
{
    [Params(ArraySizeSource.Small, ArraySizeSource.Medium)]
    public override int N { get; set; }

    [ParamsSource(nameof(Types))]
    public new NPTypeCode DType { get; set; }

    public static IEnumerable<NPTypeCode> Types => new[] { NPTypeCode.Int64 };
    private Shape Size => new(N);

    [GlobalSetup]
    public void Setup() => np.random.seed(Seed);

    [Benchmark(Description = "np.random.binomial(n, p)")] public NDArray Binomial() => np.random.binomial(10, 0.4, Size);
    [Benchmark(Description = "np.random.geometric(p)")] public NDArray Geometric() => np.random.geometric(0.4, Size);
    [Benchmark(Description = "np.random.hypergeometric(ngood, nbad, nsample)")] public NDArray Hypergeometric() => np.random.hypergeometric(20, 30, 10, Size);
    [Benchmark(Description = "np.random.logseries(p)")] public NDArray LogSeries() => np.random.logseries(0.6, Size);
    [Benchmark(Description = "np.random.negative_binomial(n, p)")] public NDArray NegativeBinomial() => np.random.negative_binomial(5.0, 0.4, Size);
    [Benchmark(Description = "np.random.poisson(lam)")] public NDArray Poisson() => np.random.poisson(3.0, Size);
    [Benchmark(Description = "np.random.randint(low, high)")] public NDArray RandInt() => np.random.randint(0, 100, Size, np.int64);
    [Benchmark(Description = "np.random.zipf(a)")] public NDArray Zipf() => np.random.zipf(2.0, Size);
}

[BenchmarkCategory("Random", "Structured")]
public class StructuredRandomBenchmarks : BenchmarkBase
{
    private NDArray _source = null!;
    private NDArray _shuffleTarget = null!;
    private readonly double[] _alpha = { 1.0, 2.0, 3.0 };
    private readonly double[] _probabilities = { 0.2, 0.3, 0.5 };
    private readonly double[] _mean = { 0.0, 1.0, -1.0 };
    private readonly double[,] _covariance = { { 1.0, 0.2, 0.1 }, { 0.2, 1.5, 0.3 }, { 0.1, 0.3, 2.0 } };

    [Params(ArraySizeSource.Small, ArraySizeSource.Medium)]
    public override int N { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        np.random.seed(Seed);
        _source = np.arange(N).astype(np.float64);
        _shuffleTarget = _source.copy();
    }

    [GlobalCleanup]
    public void Cleanup() { _source = null!; _shuffleTarget = null!; GC.Collect(); }

    [Benchmark(Description = "np.random.choice(a, size)")] public NDArray Choice() => np.random.choice(_source, new Shape(N));
    [Benchmark(Description = "np.random.dirichlet(alpha, size)")] public NDArray Dirichlet() => np.random.dirichlet(_alpha, N);
    [Benchmark(Description = "np.random.multivariate_normal(mean, cov, size)")] public NDArray MultivariateNormal() => np.random.multivariate_normal(_mean, _covariance, N);
    [Benchmark(Description = "np.random.permutation(a)")] public NDArray Permutation() => np.random.permutation(_source);
    [Benchmark(Description = "np.random.shuffle(a)")] public void Shuffle() => np.random.shuffle(_shuffleTarget);
}

[BenchmarkCategory("Random", "StructuredDiscrete")]
public class MultinomialRandomBenchmarks : TypedBenchmarkBase
{
    private readonly double[] _probabilities = { 0.2, 0.3, 0.5 };

    [Params(ArraySizeSource.Small, ArraySizeSource.Medium)]
    public override int N { get; set; }

    [ParamsSource(nameof(Types))]
    public new NPTypeCode DType { get; set; }

    public static IEnumerable<NPTypeCode> Types => new[] { NPTypeCode.Int64 };

    [GlobalSetup]
    public void Setup() => np.random.seed(Seed);

    [Benchmark(Description = "np.random.multinomial(n, pvals, size)")] public NDArray Multinomial() => np.random.multinomial(10, _probabilities, N);
}
