using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.GraphEngine.Infrastructure;

namespace NumSharp.Benchmark.GraphEngine.Benchmarks.Reduction;

/// <summary>
/// Cumulative product (np.cumprod). Cumulative sum is already covered by SumBenchmarks.
/// Floating dtypes.
/// </summary>
[BenchmarkCategory("Reduction", "Cumulative")]
public class CumulativeBenchmarks : TypedBenchmarkBase
{
    private NDArray _a = null!;

    [Params(ArraySizeSource.Small, ArraySizeSource.Medium, ArraySizeSource.Large)]
    public override int N { get; set; }

    [ParamsSource(nameof(Types))]
    public new NPTypeCode DType { get; set; }

    public static IEnumerable<NPTypeCode> Types => TypeParameterSource.FloatingTypes;

    [GlobalSetup]
    public void Setup() => _a = CreateRandomArray(N, DType);

    [GlobalCleanup]
    public void Cleanup() { _a = null!; GC.Collect(); }

    [Benchmark(Description = "np.cumprod(a)")] public NDArray CumProd() => np.cumprod(_a);
}
