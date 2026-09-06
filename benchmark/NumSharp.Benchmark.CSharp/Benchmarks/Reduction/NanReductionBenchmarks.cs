using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.CSharp.Infrastructure;

namespace NumSharp.Benchmark.CSharp.Benchmarks.Reduction;

/// <summary>
/// NaN-aware reductions: nansum, nanmean, nanmax, nanmin, nanstd, nanvar, nanprod,
/// nanmedian, nanpercentile, nanquantile. Floating dtypes only. nanprod (the one PRODUCT
/// reduction here) runs on a separately bounded input — see Setup — so it stays finite.
/// </summary>
[BenchmarkCategory("Reduction", "NaN")]
public class NanReductionBenchmarks : TypedBenchmarkBase
{
    private NDArray _a = null!;
    private NDArray _aProd = null!;

    [Params(ArraySizeSource.Small, ArraySizeSource.Medium, ArraySizeSource.Large)]
    public override int N { get; set; }

    [ParamsSource(nameof(Types))]
    public new NPTypeCode DType { get; set; }

    public static IEnumerable<NPTypeCode> Types => TypeParameterSource.AllNumericTypes;

    [GlobalSetup]
    public void Setup()
    {
        _a = CreateRandomArray(N, DType);
        // nanprod multiplies the WHOLE array, so full-range data ([-50, 50)) blows the product up:
        // float/Half saturate to +/-inf, but Decimal has no infinity, so np.nanprod(Decimal) threw
        // OverflowException at every size. Give nanprod its own [0.5, 1.0] input (product shrinks
        // toward 0, stays finite for every dtype) — the ExpLog._smallPositive / ProdBenchmarks pattern.
        // The other eleven reductions keep the representative full-range _a (all overflow-safe there).
        np.random.seed(Seed);
        _aProd = (np.random.rand(N) * 0.5 + 0.5).astype(DType);
    }

    [GlobalCleanup]
    public void Cleanup() { _a = null!; _aProd = null!; GC.Collect(); }

    [Benchmark(Description = "np.nansum(a)")] public NDArray NanSum() => np.nansum(_a);
    [Benchmark(Description = "np.nanmean(a)")] public NDArray NanMean() => np.nanmean(_a);
    [Benchmark(Description = "np.nanmax(a)")] public NDArray NanMax() => np.nanmax(_a);
    [Benchmark(Description = "np.nanmin(a)")] public NDArray NanMin() => np.nanmin(_a);
    [Benchmark(Description = "np.nanstd(a)")] public NDArray NanStd() => np.nanstd(_a);
    [Benchmark(Description = "np.nanvar(a)")] public NDArray NanVar() => np.nanvar(_a);
    [Benchmark(Description = "np.nanprod(a)")] public NDArray NanProd() => np.nanprod(_aProd);
    [Benchmark(Description = "np.nanmedian(a)")] public NDArray NanMedian() => np.nanmedian(_a);
    [Benchmark(Description = "np.nanpercentile(a, 50)")] public object NanPercentile() => DType != NPTypeCode.Boolean
        ? np.nanpercentile(_a, 50.0)
        : VerifyUnsupportedDtype(() => np.nanpercentile(_a, 50.0));
    [Benchmark(Description = "np.nanquantile(a, 0.5)")] public object NanQuantile() => DType != NPTypeCode.Boolean
        ? np.nanquantile(_a, 0.5)
        : VerifyUnsupportedDtype(() => np.nanquantile(_a, 0.5));
    [Benchmark(Description = "np.nanargmax(a)")] public long NanArgMax() => np.nanargmax(_a);
    [Benchmark(Description = "np.nanargmin(a)")] public long NanArgMin() => np.nanargmin(_a);
}
