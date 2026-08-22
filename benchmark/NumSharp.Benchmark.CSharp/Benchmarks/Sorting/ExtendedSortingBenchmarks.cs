using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.CSharp.Infrastructure;

namespace NumSharp.Benchmark.CSharp.Benchmarks.Sorting;

/// <summary>
/// Sorting/searching APIs not covered by <see cref="SortingBenchmarks"/>. The 10M tier is omitted:
/// seven O(N log N) methods times 50 measured iterations would dominate the entire official run.
/// </summary>
[BenchmarkCategory("Sorting", "Extended")]
public class ExtendedSortingBenchmarks : TypedBenchmarkBase
{
    private NDArray _a = null!;
    private NDArray _b = null!;

    [Params(ArraySizeSource.Small, ArraySizeSource.Medium)]
    public override int N { get; set; }

    [ParamsSource(nameof(Types))]
    public new NPTypeCode DType { get; set; }

    public static IEnumerable<NPTypeCode> Types => TypeParameterSource.CommonTypes;

    [GlobalSetup]
    public void Setup()
    {
        _a = CreateRandomArray(N, DType);
        _b = CreateRandomArray(N, DType, seed: 43);
    }

    [GlobalCleanup]
    public void Cleanup() { _a = null!; _b = null!; GC.Collect(); }

    [Benchmark(Description = "np.sort(a)")] public NDArray Sort() => np.sort(_a);
    [Benchmark(Description = "np.sort_complex(a)")] public NDArray SortComplex() => np.sort_complex(_a);
    [Benchmark(Description = "np.partition(a, kth)")] public NDArray Partition() => np.partition(_a, N / 2);
    [Benchmark(Description = "np.argpartition(a, kth)")] public NDArray ArgPartition() => np.argpartition(_a, N / 2);
    [Benchmark(Description = "np.lexsort((b, a))")] public NDArray LexSort() => np.lexsort(new[] { _b, _a });
    [Benchmark(Description = "np.argwhere(a)")] public NDArray ArgWhere() => np.argwhere(_a);
    [Benchmark(Description = "np.flatnonzero(a)")] public NDArray FlatNonZero() => np.flatnonzero(_a);
}
