using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.CSharp.Infrastructure;

namespace NumSharp.Benchmark.CSharp.Benchmarks.Sorting;

/// <summary>
/// Sorting / searching: argsort (indirect sort), nonzero (index discovery), searchsorted
/// (binary search into a sorted array). Mirrors NumPy's np.argsort/np.nonzero/np.searchsorted.
/// </summary>
[BenchmarkCategory("Sorting")]
public class SortingBenchmarks : TypedBenchmarkBase
{
    private NDArray _a = null!;
    private NDArray _sorted = null!;

    [Params(ArraySizeSource.Small, ArraySizeSource.Medium, ArraySizeSource.Large)]
    public override int N { get; set; }

    [ParamsSource(nameof(Types))]
    public new NPTypeCode DType { get; set; }

    public static IEnumerable<NPTypeCode> Types => TypeParameterSource.CommonTypes;

    [GlobalSetup]
    public void Setup()
    {
        _a = CreateRandomArray(N, DType);
        _sorted = np.arange(N).astype(DType);   // already ascending → valid searchsorted target
    }

    [GlobalCleanup]
    public void Cleanup() { _a = null!; _sorted = null!; GC.Collect(); }

    // argsort is generic over the element type; switch to the right closed form.
    [Benchmark(Description = "np.argsort(a)")]
    public NDArray ArgSort() => DType switch
    {
        NPTypeCode.Int32 => np.argsort<int>(_a),
        NPTypeCode.Int64 => np.argsort<long>(_a),
        NPTypeCode.Single => np.argsort<float>(_a),
        NPTypeCode.Double => np.argsort<double>(_a),
        _ => np.argsort<double>(_a)
    };

    [Benchmark(Description = "np.nonzero(a)")]
    public object NonZero() => np.nonzero(_a);

    // Query N points (the random array a) into the sorted target → N binary searches,
    // real O(N log N) work that scales with size. (Previously this issued a SINGLE
    // scalar lookup, ~18ns at every N — pure call overhead, not a throughput benchmark;
    // against NumPy's ~1µs Python overhead it manufactured a meaningless 50–1000x "win".)
    [Benchmark(Description = "np.searchsorted(a, v)")]
    public NDArray SearchSorted() => np.searchsorted(_sorted, _a);
}
