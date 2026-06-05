using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.GraphEngine.Infrastructure;

namespace NumSharp.Benchmark.GraphEngine.Benchmarks.Reduction;

/// <summary>
/// NaN-aware reductions: nansum, nanmean, nanmax, nanmin, nanstd, nanvar, nanprod,
/// nanmedian, nanpercentile, nanquantile. Floating dtypes only.
/// </summary>
[BenchmarkCategory("Reduction", "NaN")]
public class NanReductionBenchmarks : TypedBenchmarkBase
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

    [Benchmark(Description = "np.nansum(a)")] public NDArray NanSum() => np.nansum(_a);
    [Benchmark(Description = "np.nanmean(a)")] public NDArray NanMean() => np.nanmean(_a);
    [Benchmark(Description = "np.nanmax(a)")] public NDArray NanMax() => np.nanmax(_a);
    [Benchmark(Description = "np.nanmin(a)")] public NDArray NanMin() => np.nanmin(_a);
    [Benchmark(Description = "np.nanstd(a)")] public NDArray NanStd() => np.nanstd(_a);
    [Benchmark(Description = "np.nanvar(a)")] public NDArray NanVar() => np.nanvar(_a);
    [Benchmark(Description = "np.nanprod(a)")] public NDArray NanProd() => np.nanprod(_a);
    [Benchmark(Description = "np.nanmedian(a)")] public NDArray NanMedian() => np.nanmedian(_a);
    [Benchmark(Description = "np.nanpercentile(a, 50)")] public NDArray NanPercentile() => np.nanpercentile(_a, 50.0);
    [Benchmark(Description = "np.nanquantile(a, 0.5)")] public NDArray NanQuantile() => np.nanquantile(_a, 0.5);
}
