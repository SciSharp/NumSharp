using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Order;
using NumSharp;

namespace NumSharp.Benchmark.Core;

/// <summary>
/// Benchmarks NumSharp's actual current performance.
/// This measures the real-world performance of the library as users experience it.
/// </summary>
[MemoryDiagnoser]
[Orderer(SummaryOrderPolicy.FastestToSlowest)]
[RankColumn]
public class NumSharpBenchmarks
{
    private NDArray _aInt32 = null!;
    private NDArray _bInt32 = null!;
    private NDArray _aFloat64 = null!;
    private NDArray _bFloat64 = null!;

    [Params(1_000, 100_000, 10_000_000)]
    public int N { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        np.random.seed(42);
        _aInt32 = np.random.randint(0, 100, new Shape(N)).astype(np.int32);
        _bInt32 = np.random.randint(0, 100, new Shape(N)).astype(np.int32);
        _aFloat64 = np.random.rand(N) * 10;
        _bFloat64 = np.random.rand(N) * 10;
    }

    // ========================================================================
    // Binary Operations (int32)
    // ========================================================================

    [Benchmark(Description = "NumSharp: a + b (int32)")]
    [BenchmarkCategory("Binary_Int32")]
    public NDArray Add_Int32() => _aInt32 + _bInt32;

    [Benchmark(Description = "NumSharp: a - b (int32)")]
    [BenchmarkCategory("Binary_Int32")]
    public NDArray Subtract_Int32() => _aInt32 - _bInt32;

    [Benchmark(Description = "NumSharp: a * b (int32)")]
    [BenchmarkCategory("Binary_Int32")]
    public NDArray Multiply_Int32() => _aInt32 * _bInt32;

    // ========================================================================
    // Binary Operations (float64)
    // ========================================================================

    [Benchmark(Description = "NumSharp: a + b (float64)")]
    [BenchmarkCategory("Binary_Float64")]
    public NDArray Add_Float64() => _aFloat64 + _bFloat64;

    [Benchmark(Description = "NumSharp: a * a (float64)")]
    [BenchmarkCategory("Binary_Float64")]
    public NDArray Square_Float64() => _aFloat64 * _aFloat64;

    // ========================================================================
    // Compound Expressions (float64)
    // ========================================================================

    [Benchmark(Description = "NumSharp: a*a + 2*b")]
    [BenchmarkCategory("Compound")]
    public NDArray Compound_AaBb() => _aFloat64 * _aFloat64 + 2 * _bFloat64;

    [Benchmark(Description = "NumSharp: a*a*a + a*a + a")]
    [BenchmarkCategory("Compound")]
    public NDArray Compound_Polynomial() => _aFloat64 * _aFloat64 * _aFloat64 + _aFloat64 * _aFloat64 + _aFloat64;

    [Benchmark(Description = "NumSharp: sqrt(a*a + b*b)")]
    [BenchmarkCategory("Compound")]
    public NDArray Compound_Euclidean() => np.sqrt(_aFloat64 * _aFloat64 + _bFloat64 * _bFloat64);

    // ========================================================================
    // Reductions (float64)
    // ========================================================================

    [Benchmark(Description = "NumSharp: np.sum(a)")]
    [BenchmarkCategory("Reduction")]
    public NDArray Sum() => np.sum(_aFloat64);

    [Benchmark(Description = "NumSharp: np.mean(a)")]
    [BenchmarkCategory("Reduction")]
    public NDArray Mean() => np.mean(_aFloat64);

    [Benchmark(Description = "NumSharp: np.var(a)")]
    [BenchmarkCategory("Reduction")]
    public NDArray Variance() => np.var(_aFloat64);

    [Benchmark(Description = "NumSharp: np.std(a)")]
    [BenchmarkCategory("Reduction")]
    public NDArray StdDev() => np.std(_aFloat64);

    // ========================================================================
    // Unary Operations (float64)
    // ========================================================================

    [Benchmark(Description = "NumSharp: np.sqrt(a)")]
    [BenchmarkCategory("Unary")]
    public NDArray Sqrt() => np.sqrt(_aFloat64);

    [Benchmark(Description = "NumSharp: np.abs(a)")]
    [BenchmarkCategory("Unary")]
    public NDArray Abs() => np.abs(_aFloat64);

    [Benchmark(Description = "NumSharp: np.exp(a)")]
    [BenchmarkCategory("Unary")]
    public NDArray Exp() => np.exp(_aFloat64);

    [Benchmark(Description = "NumSharp: np.log(a)")]
    [BenchmarkCategory("Unary")]
    public NDArray Log() => np.log(_aFloat64);
}
