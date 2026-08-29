using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.CSharp.Infrastructure;

namespace NumSharp.Benchmark.CSharp.Benchmarks.Selection;

/// <summary>
/// np.where: the ternary select form (cond ? x : y) and the single-argument index form
/// (np.where(cond) → indices of true elements), with typed select operands.
/// </summary>
[BenchmarkCategory("Selection")]
public class WhereBenchmarks : OfficialBenchmarkBase
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
        _a = CreateRandomArray(N, DType);
        _b = CreateRandomArray(N, DType, seed: 43);
        _cond = np.random.rand(N) > 0.5;
    }

    [GlobalCleanup]
    public void Cleanup() { _cond = null!; _a = null!; _b = null!; GC.Collect(); }

    [Benchmark(Description = "np.where(cond, a, b)")] public NDArray WhereSelect() => np.where(_cond, _a, _b);
    [Benchmark(Description = "np.where(cond)")] public object WhereIndices() => np.where(_cond);
}
