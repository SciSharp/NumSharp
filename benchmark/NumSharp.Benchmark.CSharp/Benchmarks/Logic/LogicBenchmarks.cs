using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.CSharp.Infrastructure;

namespace NumSharp.Benchmark.CSharp.Benchmarks.Logic;

/// <summary>
/// Logic / predicate ufuncs on floating arrays: isnan, isinf, isfinite, maximum, minimum,
/// array_equal. (all / any live in <see cref="BoolLogicBenchmarks"/>; isclose/allclose live in
/// <see cref="CloseBenchmarks"/> because their multi-intermediate formula gets a bounded size tier.)
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
    [Benchmark(Description = "np.fmax(a, b)")] public NDArray FMax() => np.fmax(_a, _b);
    [Benchmark(Description = "np.fmin(a, b)")] public NDArray FMin() => np.fmin(_a, _b);
    // For a real input (Half/Single/Double only here) these short-circuit to a FRESH, owned bool
    // array — np.iscomplex -> np.zeros(bool) (lazy), np.isreal -> np.ones(bool) — without touching
    // the input, so they are O(1)-to-fast. BenchmarkDotNet then runs thousands of invocations per
    // iteration (iscomplex's lazy zeros pilots at ~2.8us, so BDN scaled to ~9000 ops), and on Windows
    // each committed buffer charges commit, OOMing the 10M case before finalizers reclaim them (only
    // iscomplex tipped over; isreal's np.ones fills memory, so it throttled BDN and merely leaked).
    // Dispose per invocation to bound resident memory (matches CreationBenchmarks, which disposes
    // zeros AND ones, np.imag above, and the NumPy twin's discard-each-result loop). Safe because the
    // result is a fresh owned array that never aliases _a (unlike np.real on a real input).
    [Benchmark(Description = "np.iscomplex(a)")] public void IsComplex() { using var _ = np.iscomplex(_a); }
    [Benchmark(Description = "np.isreal(a)")] public void IsReal() { using var _ = np.isreal(_a); }
    [Benchmark(Description = "np.iscomplexobj(a)")] public bool IsComplexObject() => np.iscomplexobj(_a);
    [Benchmark(Description = "np.isrealobj(a)")] public bool IsRealObject() => np.isrealobj(_a);
    [Benchmark(Description = "np.isscalar(a)")] public bool IsScalar() => np.isscalar(_a);
    [Benchmark(Description = "np.isfortran(a)")] public bool IsFortran() => np.isfortran(_a);
    [Benchmark(Description = "np.array_equal(a, b)")] public bool ArrayEqual() => np.array_equal(_a, _b);

}

/// <summary>
/// Tolerance comparisons allocate several simultaneous intermediates by definition. They run at
/// all universal tiers; the former two-buffer deferred-release slope is regression-gated by
/// BufferReleaseSweepTests.
/// </summary>
[BenchmarkCategory("Logic", "Close")]
public class CloseBenchmarks : TypedBenchmarkBase
{
    private NDArray _a = null!;
    private NDArray _b = null!;
    private int WorkN => ArraySizeSource.ResolveMemoryHeavyWorkload(N);

    [Params(ArraySizeSource.Small, ArraySizeSource.Medium, ArraySizeSource.Large)]
    public override int N { get; set; }

    [ParamsSource(nameof(Types))]
    public new NPTypeCode DType { get; set; }

    public static IEnumerable<NPTypeCode> Types => new[]
    {
        NPTypeCode.Half,
        NPTypeCode.Single,
        NPTypeCode.Double
    };

    [GlobalSetup]
    public void Setup()
    {
        _a = CreateRandomArray(WorkN, DType);
        _b = CreateRandomArray(WorkN, DType, seed: 43);
    }

    [GlobalCleanup]
    public void Cleanup() { _a = null!; _b = null!; GC.Collect(); }

    [Benchmark(Description = "np.isclose(a, b)")] public NDArray IsClose() => np.isclose(_a, _b);
    [Benchmark(Description = "np.allclose(a, b)")] public bool AllClose() => np.allclose(_a, _b);
}

/// <summary>
/// Boolean reductions all / any. Input is a boolean array (~50% true), so dtype is bool.
/// </summary>
[BenchmarkCategory("Logic")]
public class BoolLogicBenchmarks : BenchmarkBase
{
    private NDArray _mask = null!;
    private NDArray _other = null!;

    [Params(ArraySizeSource.Small, ArraySizeSource.Medium, ArraySizeSource.Large)]
    public override int N { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        np.random.seed(Seed);
        _mask = (np.random.rand(N) > 0.5).astype(np.@bool);
        _other = (np.random.rand(N) > 0.5).astype(np.@bool);
    }

    [GlobalCleanup]
    public void Cleanup() { _mask = null!; _other = null!; GC.Collect(); }

    [Benchmark(Description = "np.all(a)")] public bool All() => (bool)np.all(_mask);
    [Benchmark(Description = "np.any(a)")] public bool Any() => (bool)np.any(_mask);
    [Benchmark(Description = "np.logical_and(a, b)")] public NDArray LogicalAnd() => np.logical_and(_mask, _other);
    [Benchmark(Description = "np.logical_or(a, b)")] public NDArray LogicalOr() => np.logical_or(_mask, _other);
    [Benchmark(Description = "np.logical_xor(a, b)")] public NDArray LogicalXor() => np.logical_xor(_mask, _other);
    [Benchmark(Description = "np.logical_not(a)")] public NDArray LogicalNot() => np.logical_not(_mask);
}
