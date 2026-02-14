using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.Core.Infrastructure;

namespace NumSharp.Benchmark.Core.Benchmarks.Unary;

/// <summary>
/// Benchmarks for power operations.
/// Note: NumSharp only supports scalar exponents (ValueType), not element-wise NDArray exponents.
/// </summary>
[BenchmarkCategory("Unary", "Power")]
public class PowerBenchmarks : TypedBenchmarkBase
{
    private NDArray _a = null!;
    private NDArray _positive = null!;

    [Params(ArraySizeSource.Small, ArraySizeSource.Medium, ArraySizeSource.Large)]
    public override int N { get; set; }

    [ParamsSource(nameof(Types))]
    public new NPTypeCode DType { get; set; }

    public static IEnumerable<NPTypeCode> Types => TypeParameterSource.TranscendentalTypes;

    [GlobalSetup]
    public void Setup()
    {
        _a = CreateRandomArray(N, DType);
        _positive = CreatePositiveArray(N, DType);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _a = null!;
        _positive = null!;
        GC.Collect();
    }

    [Benchmark(Description = "np.power(a, 2)")]
    [BenchmarkCategory("Scalar")]
    public NDArray Power_Square() => np.power(_a, 2);

    [Benchmark(Description = "np.power(a, 3)")]
    [BenchmarkCategory("Scalar")]
    public NDArray Power_Cube() => np.power(_a, 3);

    [Benchmark(Description = "np.power(a, 0.5) [sqrt]")]
    [BenchmarkCategory("Scalar")]
    public NDArray Power_SqrtEquivalent() => np.power(_positive, 0.5);

    [Benchmark(Description = "a * a (square via multiply)")]
    [BenchmarkCategory("Alternative")]
    public NDArray Power_SquareViaMul() => _a * _a;

    [Benchmark(Description = "a * a * a (cube via multiply)")]
    [BenchmarkCategory("Alternative")]
    public NDArray Power_CubeViaMul() => _a * _a * _a;
}
