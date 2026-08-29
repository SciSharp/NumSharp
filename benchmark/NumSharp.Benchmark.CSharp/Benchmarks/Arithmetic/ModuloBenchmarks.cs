using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.CSharp.Infrastructure;

namespace NumSharp.Benchmark.CSharp.Benchmarks.Arithmetic;

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

    public static IEnumerable<NPTypeCode> Types => TypeParameterSource.AllNumericTypes;

    [GlobalSetup]
    public void Setup()
    {
        _a = CreateRandomArray(N, DType);
        // Avoid modulo by zero - use positive values only
        // CreatePositiveArray returns values >= 1
        _b = CreatePositiveArray(N, DType, seed: 43);
        // Ensure no zeros by clamping minimum to 1
        // This is needed because SIMD modulo will fail on any zero
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
    public object Modulo_Elementwise() => DType != NPTypeCode.Complex
        ? _a % _b
        : VerifyUnsupportedDtype(() => _a % _b);

    [Benchmark(Description = "np.mod(a, b)")]
    [BenchmarkCategory("Elementwise", "ApiEntryPoint")]
    public object NpMod() => DType != NPTypeCode.Complex
        ? np.mod(_a, _b)
        : VerifyUnsupportedDtype(() => np.mod(_a, _b));

    [Benchmark(Description = "a % 7 (literal)")]
    [BenchmarkCategory("Scalar")]
    public object Modulo_Scalar() => DType != NPTypeCode.Complex
        ? _a % 7
        : VerifyUnsupportedDtype(() => _a % 7);
}
