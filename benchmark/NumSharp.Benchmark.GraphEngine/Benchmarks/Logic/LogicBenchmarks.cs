using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.GraphEngine.Infrastructure;

namespace NumSharp.Benchmark.GraphEngine.Benchmarks.Logic;

/// <summary>
/// Logic / predicate ufuncs on floating arrays: isnan, isinf, isfinite, maximum, minimum,
/// array_equal. (all / any live in <see cref="BoolLogicBenchmarks"/>.) isclose/allclose are
/// disabled — they segfault NumSharp (see the note on the benchmark methods below).
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

    // Half/Single/Double only — these are the dtypes NumPy benchmarks logic on (float16/32/64).
    // Decimal is excluded: it has no NumPy peer (would be discarded anyway) AND its scalar
    // DecimalMath path reliably triggers the known unmanaged-storage AccessViolation under this
    // suite's load, which crashes the whole class before BenchmarkDotNet can export a report.
    public static IEnumerable<NPTypeCode> Types => new[]
    {
        NPTypeCode.Half,
        NPTypeCode.Single,
        NPTypeCode.Double
    };

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
    [Benchmark(Description = "np.array_equal(a, b)")] public bool ArrayEqual() => np.array_equal(_a, _b);

    // DISABLED — np.isclose / np.allclose deterministically segfault NumSharp with the
    // unmanaged-storage AccessViolation (each crashes even when run alone, not just under the
    // suite's cumulative load). Left in the class, the crash kills the whole LogicBenchmarks
    // process before BenchmarkDotNet can export ANY report — taking the six working predicates
    // above down with it. Re-enable once the NumSharp isclose/allclose lifetime bug is fixed.
    // [Benchmark(Description = "np.isclose(a, b)")] public NDArray IsClose() => np.isclose(_a, _b);
    // [Benchmark(Description = "np.allclose(a, b)")] public bool AllClose() => np.allclose(_a, _b);
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
