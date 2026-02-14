using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.GraphEngine.Infrastructure;

namespace NumSharp.Benchmark.GraphEngine.Benchmarks.Reduction;

/// <summary>
/// Benchmarks for sum reduction operations.
/// </summary>
[BenchmarkCategory("Reduction", "Sum")]
public class SumBenchmarks : TypedBenchmarkBase
{
    private NDArray _a1D = null!;
    private NDArray _a2D = null!;

    [Params(ArraySizeSource.Small, ArraySizeSource.Medium, ArraySizeSource.Large)]
    public override int N { get; set; }

    [ParamsSource(nameof(Types))]
    public new NPTypeCode DType { get; set; }

    public static IEnumerable<NPTypeCode> Types => TypeParameterSource.ArithmeticTypes;

    [GlobalSetup]
    public void Setup()
    {
        _a1D = CreateRandomArray(N, DType);

        // 2D array for axis tests
        var rows = (int)Math.Sqrt(N);
        var cols = N / rows;
        _a2D = CreateRandomArray(rows * cols, DType).reshape(rows, cols);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _a1D = null!;
        _a2D = null!;
        GC.Collect();
    }

    // ========================================================================
    // Full Reduction
    // ========================================================================

    [Benchmark(Description = "np.sum(a) [full]")]
    [BenchmarkCategory("Full")]
    public NDArray Sum_Full() => np.sum(_a1D);

    [Benchmark(Description = "a.sum() [method]")]
    [BenchmarkCategory("Full")]
    public NDArray Sum_Method() => _a1D.sum();

    // ========================================================================
    // Axis Reduction
    // ========================================================================

    [Benchmark(Description = "np.sum(a, axis=0) [columns]")]
    [BenchmarkCategory("Axis")]
    public NDArray Sum_Axis0() => np.sum(_a2D, axis: 0);

    [Benchmark(Description = "np.sum(a, axis=1) [rows]")]
    [BenchmarkCategory("Axis")]
    public NDArray Sum_Axis1() => np.sum(_a2D, axis: 1);

    // ========================================================================
    // Cumulative Sum
    // ========================================================================

    [Benchmark(Description = "np.cumsum(a)")]
    [BenchmarkCategory("Cumulative")]
    public NDArray CumSum() => np.cumsum(_a1D);
}
