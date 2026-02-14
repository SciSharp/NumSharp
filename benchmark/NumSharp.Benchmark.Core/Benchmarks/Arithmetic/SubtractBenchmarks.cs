using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.Core.Infrastructure;

namespace NumSharp.Benchmark.Core.Benchmarks.Arithmetic;

/// <summary>
/// Benchmarks for subtraction operations.
/// </summary>
[BenchmarkCategory("Arithmetic", "Subtract")]
public class SubtractBenchmarks : TypedBenchmarkBase
{
    private NDArray _a = null!;
    private NDArray _b = null!;
    private NDArray _scalar = null!;

    [Params(ArraySizeSource.Small, ArraySizeSource.Medium, ArraySizeSource.Large)]
    public override int N { get; set; }

    [ParamsSource(nameof(Types))]
    public new NPTypeCode DType { get; set; }

    public static IEnumerable<NPTypeCode> Types => TypeParameterSource.ArithmeticTypes;

    [GlobalSetup]
    public void Setup()
    {
        _a = CreateRandomArray(N, DType);
        _b = CreateRandomArray(N, DType, seed: 43);
        _scalar = NDArray.Scalar(GetScalar(DType, 5.0), DType);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _a = null!;
        _b = null!;
        _scalar = null!;
        GC.Collect();
    }

    [Benchmark(Description = "a - b (element-wise)")]
    [BenchmarkCategory("Elementwise")]
    public NDArray Subtract_Elementwise() => _a - _b;

    [Benchmark(Description = "a - scalar")]
    [BenchmarkCategory("Scalar")]
    public NDArray Subtract_Scalar() => _a - _scalar;

    [Benchmark(Description = "scalar - a")]
    [BenchmarkCategory("Scalar")]
    public NDArray Subtract_ScalarLeft() => _scalar - _a;
}
