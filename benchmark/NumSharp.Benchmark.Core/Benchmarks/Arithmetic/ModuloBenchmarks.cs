using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.Core.Infrastructure;

namespace NumSharp.Benchmark.Core.Benchmarks.Arithmetic;

/// <summary>
/// Benchmarks for modulo operations.
/// </summary>
[BenchmarkCategory("Arithmetic", "Modulo")]
public class ModuloBenchmarks : TypedBenchmarkBase
{
    private NDArray _a = null!;
    private NDArray _b = null!;

    [Params(ArraySizeSource.Small, ArraySizeSource.Medium, ArraySizeSource.Large)]
    public override int N { get; set; }

    // Modulo typically used with integers
    [ParamsSource(nameof(Types))]
    public new NPTypeCode DType { get; set; }

    public static IEnumerable<NPTypeCode> Types => new[]
    {
        NPTypeCode.Int32,
        NPTypeCode.Int64,
        NPTypeCode.Single,
        NPTypeCode.Double
    };

    [GlobalSetup]
    public void Setup()
    {
        _a = CreateRandomArray(N, DType);
        // Avoid modulo by zero
        _b = CreatePositiveArray(N, DType, seed: 43);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _a = null!;
        _b = null!;
        GC.Collect();
    }

    [Benchmark(Description = "a % b (element-wise)")]
    [BenchmarkCategory("Elementwise")]
    public NDArray Modulo_Elementwise() => _a % _b;

    [Benchmark(Description = "a % 7 (literal)")]
    [BenchmarkCategory("Scalar")]
    public NDArray Modulo_Scalar() => _a % 7;
}
