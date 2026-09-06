using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.CSharp.Infrastructure;

namespace NumSharp.Benchmark.CSharp.Benchmarks.Statistics;

/// <summary>
/// Order statistics and summaries: median, percentile, quantile, average, ptp,
/// count_nonzero. Floating dtypes (count_nonzero also returns a count over the same input).
/// </summary>
[BenchmarkCategory("Statistics")]
public class StatisticsBenchmarks : TypedBenchmarkBase
{
    private NDArray _a = null!;

    [Params(ArraySizeSource.Small, ArraySizeSource.Medium, ArraySizeSource.Large)]
    public override int N { get; set; }

    [ParamsSource(nameof(Types))]
    public new NPTypeCode DType { get; set; }

    public static IEnumerable<NPTypeCode> Types => TypeParameterSource.AllNumericTypes;

    [GlobalSetup]
    public void Setup() => _a = CreateRandomArray(N, DType);

    [GlobalCleanup]
    public void Cleanup() { _a = null!; GC.Collect(); }

    [Benchmark(Description = "np.median(a)")] public NDArray Median() => np.median(_a);
    [Benchmark(Description = "np.percentile(a, 50)")] public object Percentile() => DType != NPTypeCode.Boolean
        ? np.percentile(_a, 50.0)
        : VerifyUnsupportedDtype(() => np.percentile(_a, 50.0));
    [Benchmark(Description = "np.quantile(a, 0.5)")] public object Quantile() => DType != NPTypeCode.Boolean
        ? np.quantile(_a, 0.5)
        : VerifyUnsupportedDtype(() => np.quantile(_a, 0.5));
    [Benchmark(Description = "np.average(a)")] public NDArray Average() => np.average(_a);
    [Benchmark(Description = "np.ptp(a)")] public object Ptp() => DType != NPTypeCode.Boolean
        ? np.ptp(_a)
        : VerifyUnsupportedDtype(() => np.ptp(_a));
    [Benchmark(Description = "np.count_nonzero(a)")] public long CountNonzero() => np.count_nonzero(_a);
}
