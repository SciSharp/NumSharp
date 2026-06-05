using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.GraphEngine.Infrastructure;

namespace NumSharp.Benchmark.GraphEngine.Benchmarks.Logic;

/// <summary>
/// Logic / predicate ufuncs on floating arrays: isnan, isinf, isfinite, maximum, minimum,
/// isclose, allclose, array_equal. (all / any live in <see cref="BoolLogicBenchmarks"/>.)
/// </summary>
[BenchmarkCategory("Logic")]
public class LogicBenchmarks : TypedBenchmarkBase
{
    private NDArray _a = null!;
    private NDArray _b = null!;

    [Params(ArraySizeSource.Small, ArraySizeSource.Medium, ArraySizeSource.Large)]
    public override int N { get; set; }

    [ParamsSource(nameof(Types))]
    public new NPTypeCode DType { get; set; }

    public static IEnumerable<NPTypeCode> Types => TypeParameterSource.FloatingTypes;

    [GlobalSetup]
    public void Setup()
    {
        _a = CreateRandomArray(N, DType);
        _b = CreateRandomArray(N, DType, seed: 43);
    }

    [GlobalCleanup]
    public void Cleanup() { _a = null!; _b = null!; GC.Collect(); }

    [Benchmark(Description = "np.isnan(a)")] public NDArray IsNan() => np.isnan(_a);
    [Benchmark(Description = "np.isinf(a)")] public NDArray IsInf() => np.isinf(_a);
    [Benchmark(Description = "np.isfinite(a)")] public NDArray IsFinite() => np.isfinite(_a);
    [Benchmark(Description = "np.maximum(a, b)")] public NDArray Maximum() => np.maximum(_a, _b);
    [Benchmark(Description = "np.minimum(a, b)")] public NDArray Minimum() => np.minimum(_a, _b);
    [Benchmark(Description = "np.isclose(a, b)")] public NDArray IsClose() => np.isclose(_a, _b);
    [Benchmark(Description = "np.allclose(a, b)")] public bool AllClose() => np.allclose(_a, _b);
    [Benchmark(Description = "np.array_equal(a, b)")] public bool ArrayEqual() => np.array_equal(_a, _b);
}

/// <summary>
/// Boolean reductions all / any. Input is a boolean array (~50% true), so dtype is bool.
/// </summary>
[BenchmarkCategory("Logic")]
public class BoolLogicBenchmarks : BenchmarkBase
{
    private NDArray _mask = null!;

    [Params(ArraySizeSource.Small, ArraySizeSource.Medium, ArraySizeSource.Large)]
    public override int N { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        np.random.seed(Seed);
        _mask = (np.random.rand(N) > 0.5).astype(np.@bool);
    }

    [GlobalCleanup]
    public void Cleanup() { _mask = null!; GC.Collect(); }

    [Benchmark(Description = "np.all(a)")] public bool All() => (bool)np.all(_mask);
    [Benchmark(Description = "np.any(a)")] public bool Any() => (bool)np.any(_mask);
}
