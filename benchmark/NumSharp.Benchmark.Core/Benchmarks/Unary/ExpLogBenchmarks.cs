using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.Core.Infrastructure;

namespace NumSharp.Benchmark.Core.Benchmarks.Unary;

/// <summary>
/// Benchmarks for exponential and logarithmic functions.
/// </summary>
[BenchmarkCategory("Unary", "ExpLog")]
public class ExpLogBenchmarks : TypedBenchmarkBase
{
    private NDArray _a = null!;
    private NDArray _positive = null!;
    private NDArray _smallPositive = null!;

    [Params(ArraySizeSource.Small, ArraySizeSource.Medium, ArraySizeSource.Large)]
    public override int N { get; set; }

    [ParamsSource(nameof(Types))]
    public new NPTypeCode DType { get; set; }

    public static IEnumerable<NPTypeCode> Types => TypeParameterSource.TranscendentalTypes;

    [GlobalSetup]
    public void Setup()
    {
        _a = CreateRandomArray(N, DType);
        _positive = CreatePositiveArray(N, DType);
        // Small positive for exp (avoid overflow)
        np.random.seed(Seed);
        _smallPositive = (np.random.rand(N) * 10).astype(DType);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _a = null!;
        _positive = null!;
        _smallPositive = null!;
        GC.Collect();
    }

    // ========================================================================
    // Exponential Functions
    // ========================================================================

    [Benchmark(Description = "np.exp(a)")]
    [BenchmarkCategory("Exp")]
    public NDArray Exp() => np.exp(_smallPositive);

    [Benchmark(Description = "np.exp2(a)")]
    [BenchmarkCategory("Exp")]
    public NDArray Exp2() => np.exp2(_smallPositive);

    [Benchmark(Description = "np.expm1(a)")]
    [BenchmarkCategory("Exp")]
    public NDArray Expm1() => np.expm1(_smallPositive);

    // ========================================================================
    // Logarithmic Functions
    // ========================================================================

    [Benchmark(Description = "np.log(a)")]
    [BenchmarkCategory("Log")]
    public NDArray Log() => np.log(_positive);

    [Benchmark(Description = "np.log2(a)")]
    [BenchmarkCategory("Log")]
    public NDArray Log2() => np.log2(_positive);

    [Benchmark(Description = "np.log10(a)")]
    [BenchmarkCategory("Log")]
    public NDArray Log10() => np.log10(_positive);

    [Benchmark(Description = "np.log1p(a)")]
    [BenchmarkCategory("Log")]
    public NDArray Log1p() => np.log1p(_positive);
}
