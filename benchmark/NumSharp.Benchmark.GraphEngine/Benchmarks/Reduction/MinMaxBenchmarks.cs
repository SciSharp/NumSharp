using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.GraphEngine.Infrastructure;

namespace NumSharp.Benchmark.GraphEngine.Benchmarks.Reduction;

/// <summary>
/// Benchmarks for min/max reduction operations.
/// </summary>
[BenchmarkCategory("Reduction", "MinMax")]
public class MinMaxBenchmarks : TypedBenchmarkBase
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
    // Min
    // ========================================================================

    [Benchmark(Description = "np.amin(a) [full]")]
    [BenchmarkCategory("Min")]
    public NDArray AMin_Full() => np.amin(_a1D);

    [Benchmark(Description = "a.amin() [method]")]
    [BenchmarkCategory("Min")]
    public NDArray AMin_Method() => _a1D.amin();

    [Benchmark(Description = "np.amin(a, axis=0)")]
    [BenchmarkCategory("Min")]
    public NDArray AMin_Axis0() => np.amin(_a2D, axis: 0);

    // ========================================================================
    // Max
    // ========================================================================

    [Benchmark(Description = "np.amax(a) [full]")]
    [BenchmarkCategory("Max")]
    public NDArray AMax_Full() => np.amax(_a1D);

    [Benchmark(Description = "a.amax() [method]")]
    [BenchmarkCategory("Max")]
    public NDArray AMax_Method() => _a1D.amax();

    [Benchmark(Description = "np.amax(a, axis=0)")]
    [BenchmarkCategory("Max")]
    public NDArray AMax_Axis0() => np.amax(_a2D, axis: 0);

    // ========================================================================
    // ArgMin / ArgMax
    // ========================================================================

    [Benchmark(Description = "np.argmin(a)")]
    [BenchmarkCategory("ArgMin")]
    public int ArgMin() => np.argmin(_a1D);

    [Benchmark(Description = "np.argmax(a)")]
    [BenchmarkCategory("ArgMax")]
    public int ArgMax() => np.argmax(_a1D);
}
