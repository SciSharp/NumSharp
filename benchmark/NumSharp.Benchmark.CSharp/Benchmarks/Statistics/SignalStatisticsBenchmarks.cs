using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.CSharp.Infrastructure;

namespace NumSharp.Benchmark.CSharp.Benchmarks.Statistics;

/// <summary>Signal, difference, covariance, histogram, and small product compositions.</summary>
[BenchmarkCategory("Statistics", "Signal")]
public class SignalStatisticsBenchmarks : BenchmarkBase
{
    private NDArray _a = null!;
    private NDArray _b = null!;
    private NDArray _kernel = null!;
    private NDArray _vectors = null!;
    private NDArray _otherVectors = null!;
    private NDArray _kronA = null!;
    private NDArray _kronB = null!;
    private NDArray _matrix = null!;
    private NDArray _bins = null!;

    [Params(ArraySizeSource.Small, ArraySizeSource.Medium)]
    public override int N { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        np.random.seed(Seed);
        _a = np.random.rand(N) * 100 - 50;
        _b = np.random.rand(N) * 100 - 50;
        _kernel = np.random.rand(31) * 2 - 1;
        int vectorRows = N / 3;
        _vectors = np.random.rand(vectorRows * 3).reshape(vectorRows, 3);
        _otherVectors = np.random.rand(vectorRows * 3).reshape(vectorRows, 3);
        int side = (int)Math.Sqrt(N);
        _kronA = np.random.rand(side);
        _kronB = np.random.rand(side);
        _matrix = np.random.rand(side * side).reshape(side, side);
        _bins = np.linspace(-50, 50, 257);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _a = null!;
        _b = null!;
        _kernel = null!;
        _vectors = null!;
        _otherVectors = null!;
        _kronA = null!;
        _kronB = null!;
        _matrix = null!;
        _bins = null!;
        GC.Collect();
    }

    [Benchmark(Description = "np.convolve(a, kernel)")] public NDArray Convolve() => np.convolve(_a, _kernel, "same");
    [Benchmark(Description = "np.correlate(a, kernel)")] public NDArray Correlate() => np.correlate(_a, _kernel, "same");
    [Benchmark(Description = "np.diff(a)")] public NDArray Diff() => np.diff(_a);
    [Benchmark(Description = "np.ediff1d(a)")] public NDArray EDiff1D() => np.ediff1d(_a);
    [Benchmark(Description = "np.cross(a, b)")] public NDArray Cross() => np.cross(_vectors, _otherVectors);
    [Benchmark(Description = "np.inner(a, b)")] public NDArray Inner() => np.inner(_a, _b);
    [Benchmark(Description = "np.kron(a, b)")] public NDArray Kron() => np.kron(_kronA, _kronB);
    [Benchmark(Description = "np.trace(a)")] public NDArray Trace() => np.trace(_matrix);
    [Benchmark(Description = "np.cov(a, b)")] public NDArray Cov() => np.cov(_a, _b);
    [Benchmark(Description = "np.corrcoef(a, b)")] public NDArray CorrCoef() => np.corrcoef(_a, _b);
    [Benchmark(Description = "np.digitize(a, bins)")] public NDArray Digitize() => np.digitize(_a, _bins);
}

/// <summary>Bincount uses a non-negative int32 input and reports that input dtype explicitly.</summary>
[BenchmarkCategory("Statistics", "Histogram")]
public class BincountBenchmarks : TypedBenchmarkBase
{
    private NDArray _values = null!;

    [Params(ArraySizeSource.Small, ArraySizeSource.Medium)]
    public override int N { get; set; }

    [ParamsSource(nameof(Types))]
    public new NPTypeCode DType { get; set; }

    public static IEnumerable<NPTypeCode> Types => new[] { NPTypeCode.Int32 };

    [GlobalSetup]
    public void Setup()
    {
        np.random.seed(Seed);
        _values = np.random.randint(0, Math.Max(2, N / 10), new Shape(N));
    }

    [GlobalCleanup]
    public void Cleanup() { _values = null!; GC.Collect(); }

    [Benchmark(Description = "np.bincount(a)")] public NDArray Bincount() => np.bincount(_values);
}
