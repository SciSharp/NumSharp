using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.GraphEngine.Infrastructure;

namespace NumSharp.Benchmark.GraphEngine.Benchmarks.Arithmetic;

/// <summary>
/// Benchmarks for division operations.
/// </summary>
[BenchmarkCategory("Arithmetic", "Divide")]
public class DivideBenchmarks : TypedBenchmarkBase
{
    private NDArray _a = null!;
    private NDArray _b = null!;
    private NDArray _scalar = null!;

    [Params(ArraySizeSource.Small, ArraySizeSource.Medium, ArraySizeSource.Large)]
    public override int N { get; set; }

    // Division works best with floating types
    [ParamsSource(nameof(Types))]
    public new NPTypeCode DType { get; set; }

    public static IEnumerable<NPTypeCode> Types => TypeParameterSource.CommonTypes;

    [GlobalSetup]
    public void Setup()
    {
        _a = CreateRandomArray(N, DType);
        // Avoid division by zero - ensure positive values
        _b = CreatePositiveArray(N, DType, seed: 43);
        _scalar = NDArray.Scalar(GetScalar(DType, 2.0), DType);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _a = null!;
        _b = null!;
        _scalar = null!;
        GC.Collect();
    }

    [Benchmark(Description = "a / b (element-wise)")]
    [BenchmarkCategory("Elementwise")]
    public NDArray Divide_Elementwise() => _a / _b;

    [Benchmark(Description = "a / scalar")]
    [BenchmarkCategory("Scalar")]
    public NDArray Divide_Scalar() => _a / _scalar;

    [Benchmark(Description = "scalar / a")]
    [BenchmarkCategory("Scalar")]
    public NDArray Divide_ScalarLeft() => _scalar / _b;
}
