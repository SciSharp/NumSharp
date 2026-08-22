using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.CSharp.Infrastructure;

namespace NumSharp.Benchmark.CSharp.Benchmarks.Reduction;

/// <summary>
/// Cumulative product (np.cumprod) over the floating dtypes. Cumulative sum is already covered by
/// SumBenchmarks. Inputs are bounded to [0.5, 1.0] so the running product stays finite at every
/// size and dtype — see Setup.
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
    public void Setup()
    {
        // Values in [0.5, 1.0] keep the cumulative product FINITE at every size and dtype: the running
        // product monotonically shrinks toward 0 (underflow) instead of growing. Full-range data
        // ([-50, 50), from CreateRandomArray) makes it blow up — float/Half saturate to +/-inf, but
        // Decimal has no infinity, so np.cumprod(Decimal) threw OverflowException at every size.
        // Mirrors ProdBenchmarks, which bounds its inputs the same way (and matches the NumPy twin).
        np.random.seed(Seed);
        _a = (np.random.rand(N) * 0.5 + 0.5).astype(DType);
    }

    [GlobalCleanup]
    public void Cleanup() { _a = null!; GC.Collect(); }

    [Benchmark(Description = "np.cumprod(a)")] public NDArray CumProd() => np.cumprod(_a);
}
