using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.GraphEngine.Infrastructure;

namespace NumSharp.Benchmark.GraphEngine.Benchmarks.Reduction;

/// <summary>
/// Benchmarks for mean reduction operations.
/// </summary>
[BenchmarkCategory("Reduction", "Mean")]
public class MeanBenchmarks : TypedBenchmarkBase
{
    private NDArray _a1D = null!;
    private NDArray _a2D = null!;

    [Params(ArraySizeSource.Small, ArraySizeSource.Medium, ArraySizeSource.Large)]
    public override int N { get; set; }

    // Mean typically produces floating-point result
    [ParamsSource(nameof(Types))]
    public new NPTypeCode DType { get; set; }

    public static IEnumerable<NPTypeCode> Types => TypeParameterSource.CommonTypes;

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

    [Benchmark(Description = "np.mean(a) [full]")]
    [BenchmarkCategory("Full")]
    public NDArray Mean_Full() => np.mean(_a1D);

    [Benchmark(Description = "a.mean() [method]")]
    [BenchmarkCategory("Full")]
    public NDArray Mean_Method() => _a1D.mean();

    [Benchmark(Description = "np.mean(a, axis=0)")]
    [BenchmarkCategory("Axis")]
    public NDArray Mean_Axis0() => np.mean(_a2D, axis: 0);

    [Benchmark(Description = "np.mean(a, axis=1)")]
    [BenchmarkCategory("Axis")]
    public NDArray Mean_Axis1() => np.mean(_a2D, axis: 1);
}
