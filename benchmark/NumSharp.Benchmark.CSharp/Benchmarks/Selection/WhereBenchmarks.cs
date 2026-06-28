using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.CSharp.Infrastructure;

namespace NumSharp.Benchmark.CSharp.Benchmarks.Selection;

/// <summary>
/// np.where: the ternary select form (cond ? x : y) and the single-argument index form
/// (np.where(cond) → indices of true elements). float64.
/// </summary>
[BenchmarkCategory("Selection")]
public class WhereBenchmarks : BenchmarkBase
{
    private NDArray _cond = null!;
    private NDArray _a = null!;
    private NDArray _b = null!;

    [Params(ArraySizeSource.Small, ArraySizeSource.Medium, ArraySizeSource.Large)]
    public override int N { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        np.random.seed(Seed);
        _a = np.random.rand(N) * 100 - 50;
        _b = np.random.rand(N) * 100 - 50;
        _cond = _a > 0.0;
    }

    [GlobalCleanup]
    public void Cleanup() { _cond = null!; _a = null!; _b = null!; GC.Collect(); }

    [Benchmark(Description = "np.where(cond, a, b)")] public NDArray WhereSelect() => np.where(_cond, _a, _b);
    [Benchmark(Description = "np.where(cond)")] public object WhereIndices() => np.where(_cond);
}
