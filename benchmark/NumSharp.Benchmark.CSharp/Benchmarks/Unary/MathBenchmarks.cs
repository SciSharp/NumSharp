using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.CSharp.Infrastructure;

namespace NumSharp.Benchmark.CSharp.Benchmarks.Unary;

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

    public static IEnumerable<NPTypeCode> Types => TypeParameterSource.AllNumericTypes;

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
    public object Sign() => DType != NPTypeCode.Boolean
        ? np.sign(_a)
        : VerifyUnsupportedDtype(() => np.sign(_a));

    [Benchmark(Description = "np.floor(a)")]
    [BenchmarkCategory("Rounding")]
    public object Floor() => DType != NPTypeCode.Complex
        ? np.floor(_a)
        : VerifyUnsupportedDtype(() => np.floor(_a));

    [Benchmark(Description = "np.ceil(a)")]
    [BenchmarkCategory("Rounding")]
    public object Ceil() => DType != NPTypeCode.Complex
        ? np.ceil(_a)
        : VerifyUnsupportedDtype(() => np.ceil(_a));

    [Benchmark(Description = "np.around(a)")]
    [BenchmarkCategory("Rounding")]
    public NDArray Round() => np.around(_a);

    [Benchmark(Description = "np.clip(a, -10, 10)")]
    [BenchmarkCategory("Clip")]
    public NDArray Clip() => np.clip(_a, -10.0, 10.0);
}
