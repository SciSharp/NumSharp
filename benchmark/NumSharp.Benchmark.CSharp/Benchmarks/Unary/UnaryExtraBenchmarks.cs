using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.CSharp.Infrastructure;

namespace NumSharp.Benchmark.CSharp.Benchmarks.Unary;

/// <summary>
/// Unary math functions not covered by MathBenchmarks/TrigBenchmarks/ExpLogBenchmarks:
/// cbrt, reciprocal, square, negative, positive, trunc. Float dtypes only.
/// </summary>
[BenchmarkCategory("Unary", "Extra")]
public class UnaryExtraBenchmarks : TypedBenchmarkBase
{
    private NDArray _a = null!;

    [Params(ArraySizeSource.Small, ArraySizeSource.Medium, ArraySizeSource.Large)]
    public override int N { get; set; }

    [ParamsSource(nameof(Types))]
    public new NPTypeCode DType { get; set; }

    public static IEnumerable<NPTypeCode> Types => TypeParameterSource.FloatingTypes;

    [GlobalSetup]
    public void Setup()
    {
        // Positive (and non-zero) so reciprocal/cbrt are well-behaved.
        _a = CreatePositiveArray(N, DType);
    }

    [GlobalCleanup]
    public void Cleanup() { _a = null!; GC.Collect(); }

    [Benchmark(Description = "np.cbrt(a)")] public NDArray Cbrt() => np.cbrt(_a);
    [Benchmark(Description = "np.reciprocal(a)")] public NDArray Reciprocal() => np.reciprocal(_a);
    [Benchmark(Description = "np.square(a)")] public NDArray Square() => np.square(_a);
    [Benchmark(Description = "np.negative(a)")] public NDArray Negative() => np.negative(_a);
    [Benchmark(Description = "np.positive(a)")] public NDArray Positive() => np.positive(_a);
    [Benchmark(Description = "np.trunc(a)")] public NDArray Trunc() => np.trunc(_a);
}
