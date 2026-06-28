using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.CSharp.Infrastructure;

namespace NumSharp.Benchmark.CSharp.Benchmarks.Reduction;

/// <summary>
/// Benchmarks for product reduction operations.
/// </summary>
[BenchmarkCategory("Reduction", "Prod")]
public class ProdBenchmarks : TypedBenchmarkBase
{
    private NDArray _a1D = null!;
    private NDArray _a2D = null!;

    // Standard cache-tier sizes (matches every other reduction so the NumPy join lines up).
    // Overflow is avoided by the value range, not by shrinking N: inputs are in [0.5, 1.0], so
    // the product stays finite (it underflows toward 0 at large N) at every size.
    [Params(ArraySizeSource.Small, ArraySizeSource.Medium, ArraySizeSource.Large)]
    public override int N { get; set; }

    // Int64 / Double hold the (bounded) product; matches the NumPy run_prod_benchmarks dtypes.
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

    [Benchmark(Description = "np.prod(a) [full]")]
    [BenchmarkCategory("Full")]
    public NDArray Prod_Full() => np.prod(_a1D);

    [Benchmark(Description = "np.prod(a, axis=0)")]
    [BenchmarkCategory("Axis")]
    public NDArray Prod_Axis0() => np.prod(_a2D, axis: 0);

    [Benchmark(Description = "np.prod(a, axis=1)")]
    [BenchmarkCategory("Axis")]
    public NDArray Prod_Axis1() => np.prod(_a2D, axis: 1);
}
