using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.CSharp.Infrastructure;

namespace NumSharp.Benchmark.CSharp.Benchmarks.Arithmetic;

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

    public static IEnumerable<NPTypeCode> Types => TypeParameterSource.AllNumericTypes;
    private bool SupportsSubtract => DType != NPTypeCode.Boolean;

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
    public object Subtract_Elementwise() => SupportsSubtract
        ? _a - _b
        : VerifyUnsupportedDtype(() => _a - _b);

    [Benchmark(Description = "np.subtract(a, b)")]
    [BenchmarkCategory("Elementwise", "ApiEntryPoint")]
    public object NpSubtract() => SupportsSubtract
        ? np.subtract(_a, _b)
        : VerifyUnsupportedDtype(() => np.subtract(_a, _b));

    [Benchmark(Description = "a - scalar")]
    [BenchmarkCategory("Scalar")]
    public object Subtract_Scalar() => SupportsSubtract
        ? _a - _scalar
        : VerifyUnsupportedDtype(() => _a - _scalar);

    [Benchmark(Description = "scalar - a")]
    [BenchmarkCategory("Scalar")]
    public object Subtract_ScalarLeft() => SupportsSubtract
        ? _scalar - _a
        : VerifyUnsupportedDtype(() => _scalar - _a);
}
