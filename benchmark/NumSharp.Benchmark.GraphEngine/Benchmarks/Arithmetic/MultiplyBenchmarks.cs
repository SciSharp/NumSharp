using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.GraphEngine.Infrastructure;

namespace NumSharp.Benchmark.GraphEngine.Benchmarks.Arithmetic;

/// <summary>
/// Benchmarks for multiplication operations.
/// </summary>
[BenchmarkCategory("Arithmetic", "Multiply")]
public class MultiplyBenchmarks : TypedBenchmarkBase
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

    [Benchmark(Description = "a * b (element-wise)")]
    [BenchmarkCategory("Elementwise")]
    public NDArray Multiply_Elementwise() => _a * _b;

    [Benchmark(Description = "a * a (square)")]
    [BenchmarkCategory("Elementwise")]
    public NDArray Multiply_Square() => _a * _a;

    [Benchmark(Description = "a * scalar")]
    [BenchmarkCategory("Scalar")]
    public NDArray Multiply_Scalar() => _a * _scalar;

    [Benchmark(Description = "a * 2 (literal)")]
    [BenchmarkCategory("Scalar")]
    public NDArray Multiply_ScalarLiteral() => _a * 2;
}
