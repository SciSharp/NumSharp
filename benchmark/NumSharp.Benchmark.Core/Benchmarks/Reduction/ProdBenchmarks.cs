using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.Core.Infrastructure;

namespace NumSharp.Benchmark.Core.Benchmarks.Reduction;

/// <summary>
/// Benchmarks for product reduction operations.
/// </summary>
[BenchmarkCategory("Reduction", "Prod")]
public class ProdBenchmarks : TypedBenchmarkBase
{
    private NDArray _a1D = null!;
    private NDArray _a2D = null!;

    // Use smaller arrays for prod to avoid overflow
    [Params(100, 1000, 10000)]
    public override int N { get; set; }

    // Use types that can hold larger products
    [ParamsSource(nameof(Types))]
    public new NPTypeCode DType { get; set; }

    public static IEnumerable<NPTypeCode> Types => new[]
    {
        NPTypeCode.Int64,
        NPTypeCode.Double
    };

    [GlobalSetup]
    public void Setup()
    {
        // Use small values to avoid overflow
        np.random.seed(Seed);
        _a1D = (np.random.rand(N) * 0.5 + 0.5).astype(DType);  // Values between 0.5 and 1.0

        var rows = (int)Math.Sqrt(N);
        var cols = N / rows;
        np.random.seed(Seed);
        _a2D = (np.random.rand(rows * cols) * 0.5 + 0.5).astype(DType).reshape(rows, cols);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _a1D = null!;
        _a2D = null!;
        GC.Collect();
    }

    [Benchmark(Description = "a.prod() [full]")]
    [BenchmarkCategory("Full")]
    public NDArray Prod_Full() => _a1D.prod();

    [Benchmark(Description = "a.prod(axis=0)")]
    [BenchmarkCategory("Axis")]
    public NDArray Prod_Axis0() => _a2D.prod(axis: 0);

    [Benchmark(Description = "a.prod(axis=1)")]
    [BenchmarkCategory("Axis")]
    public NDArray Prod_Axis1() => _a2D.prod(axis: 1);
}
