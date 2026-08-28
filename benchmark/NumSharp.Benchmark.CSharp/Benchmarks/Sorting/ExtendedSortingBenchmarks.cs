using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.CSharp.Infrastructure;

namespace NumSharp.Benchmark.CSharp.Benchmarks.Sorting;

/// <summary>
/// Sorting/searching APIs not covered by <see cref="SortingBenchmarks"/>, measured at all universal
/// workload tiers despite the expected cost of the 10M O(N log N) cases.
/// </summary>
[BenchmarkCategory("Sorting", "Extended")]
public class ExtendedSortingBenchmarks : TypedBenchmarkBase
{
    private NDArray _a = null!;
    private NDArray _b = null!;
    private int WorkN => ArraySizeSource.ResolveMemoryHeavyWorkload(N);

    [Params(ArraySizeSource.Small, ArraySizeSource.Medium, ArraySizeSource.Large)]
    public override int N { get; set; }

    [ParamsSource(nameof(Types))]
    public new NPTypeCode DType { get; set; }

    public static IEnumerable<NPTypeCode> Types => TypeParameterSource.CommonTypes;

    [GlobalSetup]
    public void Setup()
    {
        _a = CreateRandomArray(WorkN, DType);
        _b = CreateRandomArray(WorkN, DType, seed: 43);
    }

    [GlobalCleanup]
    public void Cleanup() { _a = null!; _b = null!; GC.Collect(); }

    [Benchmark(Description = "np.sort(a)")] public NDArray Sort() => np.sort(_a);
    [Benchmark(Description = "np.sort_complex(a)")] public NDArray SortComplex() => np.sort_complex(_a);
    [Benchmark(Description = "np.partition(a, kth)")] public NDArray Partition() => np.partition(_a, WorkN / 2);
    [Benchmark(Description = "np.argpartition(a, kth)")] public NDArray ArgPartition() => np.argpartition(_a, WorkN / 2);
    [Benchmark(Description = "np.lexsort((b, a))")] public NDArray LexSort() => np.lexsort(new[] { _b, _a });
    [Benchmark(Description = "np.argwhere(a)")] public NDArray ArgWhere() => np.argwhere(_a);
    [Benchmark(Description = "np.flatnonzero(a)")] public NDArray FlatNonZero() => np.flatnonzero(_a);
}
