using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.Core.Infrastructure;

namespace NumSharp.Benchmark.Core.Benchmarks.Arithmetic;

/// <summary>
/// Benchmarks for addition operations: a + b, a + scalar.
/// Tests all numeric types and standard array sizes.
/// Note: 2D broadcasting tests are in BroadcastBenchmarks (float64 only).
/// </summary>
[BenchmarkCategory("Arithmetic", "Add")]
public class AddBenchmarks : TypedBenchmarkBase
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
        _b = CreateRandomArray(N, DType, seed: 43);  // Different seed for variety
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

    // ========================================================================
    // Element-wise Addition
    // ========================================================================

    [Benchmark(Description = "a + b (element-wise)")]
    [BenchmarkCategory("Elementwise")]
    public NDArray Add_Elementwise() => _a + _b;

    [Benchmark(Description = "np.add(a, b)")]
    [BenchmarkCategory("Elementwise")]
    public NDArray NpAdd() => np.add(_a, _b);

    // ========================================================================
    // Scalar Addition
    // ========================================================================

    [Benchmark(Description = "a + scalar")]
    [BenchmarkCategory("Scalar")]
    public NDArray Add_Scalar() => _a + _scalar;

    [Benchmark(Description = "a + 5 (literal)")]
    [BenchmarkCategory("Scalar")]
    public NDArray Add_ScalarLiteral() => _a + 5;
}
