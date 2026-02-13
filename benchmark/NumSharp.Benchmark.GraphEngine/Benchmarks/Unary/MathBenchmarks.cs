using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.GraphEngine.Infrastructure;

namespace NumSharp.Benchmark.GraphEngine.Benchmarks.Unary;

/// <summary>
/// Benchmarks for basic unary math operations: sqrt, abs, sign, floor, ceil, round.
/// </summary>
[BenchmarkCategory("Unary", "Math")]
public class MathBenchmarks : TypedBenchmarkBase
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

    [Benchmark(Description = "np.sqrt(a)")]
    [BenchmarkCategory("Sqrt")]
    public NDArray Sqrt() => np.sqrt(_positive);

    [Benchmark(Description = "np.abs(a)")]
    [BenchmarkCategory("Abs")]
    public NDArray Abs() => np.abs(_a);

    [Benchmark(Description = "np.sign(a)")]
    [BenchmarkCategory("Sign")]
    public NDArray Sign() => np.sign(_a);

    [Benchmark(Description = "np.floor(a)")]
    [BenchmarkCategory("Rounding")]
    public NDArray Floor() => np.floor(_a);

    [Benchmark(Description = "np.ceil(a)")]
    [BenchmarkCategory("Rounding")]
    public NDArray Ceil() => np.ceil(_a);

    [Benchmark(Description = "np.around(a)")]
    [BenchmarkCategory("Rounding")]
    public NDArray Round() => np.around(_a);

    [Benchmark(Description = "np.clip(a, -10, 10)")]
    [BenchmarkCategory("Clip")]
    public NDArray Clip() => np.clip(_a, -10.0, 10.0);
}
