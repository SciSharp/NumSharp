using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.CSharp.Infrastructure;

namespace NumSharp.Benchmark.CSharp.Benchmarks.Manipulation;

/// <summary>Sorted/hashed one-dimensional set operations over duplicate-heavy int32 inputs.</summary>
[BenchmarkCategory("Manipulation", "SetOperations")]
public class SetOperationBenchmarks : TypedBenchmarkBase
{
    private NDArray _a = null!;
    private NDArray _b = null!;

    [Params(ArraySizeSource.Small, ArraySizeSource.Medium)]
    public override int N { get; set; }

    [ParamsSource(nameof(Types))]
    public new NPTypeCode DType { get; set; }

    public static IEnumerable<NPTypeCode> Types => new[] { NPTypeCode.Int32 };

    [GlobalSetup]
    public void Setup()
    {
        np.random.seed(Seed);
        _a = np.random.randint(0, Math.Max(2, N / 2), new Shape(N));
        _b = np.random.randint(0, Math.Max(2, N / 2), new Shape(N));
    }

    [GlobalCleanup]
    public void Cleanup() { _a = null!; _b = null!; GC.Collect(); }

    [Benchmark(Description = "np.intersect1d(a, b)")] public NDArray Intersect1D() => np.intersect1d(_a, _b);
    [Benchmark(Description = "np.isin(a, b)")] public NDArray IsIn() => np.isin(_a, _b);
    [Benchmark(Description = "np.setdiff1d(a, b)")] public NDArray SetDiff1D() => np.setdiff1d(_a, _b);
    [Benchmark(Description = "np.setxor1d(a, b)")] public NDArray SetXor1D() => np.setxor1d(_a, _b);
    [Benchmark(Description = "np.union1d(a, b)")] public NDArray Union1D() => np.union1d(_a, _b);
    [Benchmark(Description = "np.unique(a)")] public np.UniqueResult Unique() => np.unique(_a);
    [Benchmark(Description = "np.unique_all(a)")] public np.UniqueAllResult UniqueAll() => np.unique_all(_a);
    [Benchmark(Description = "np.unique_counts(a)")] public np.UniqueCountsResult UniqueCounts() => np.unique_counts(_a);
    [Benchmark(Description = "np.unique_inverse(a)")] public np.UniqueInverseResult UniqueInverse() => np.unique_inverse(_a);
    [Benchmark(Description = "np.unique_values(a)")] public NDArray UniqueValues() => np.unique_values(_a);
}
