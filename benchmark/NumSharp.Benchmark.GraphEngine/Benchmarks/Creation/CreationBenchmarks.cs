using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.GraphEngine.Infrastructure;

namespace NumSharp.Benchmark.GraphEngine.Benchmarks.Creation;

/// <summary>
/// Benchmarks for array creation functions.
///
/// <para>
/// LIFETIME / FAIRNESS NOTE: every benchmark disposes the array it creates,
/// per invocation. This mirrors the NumPy harness (numpy_benchmark.py), whose
/// timed loop discards each <c>np.zeros(...)</c> result so CPython's refcount
/// frees it inside the timed region — i.e. NumPy measures alloc+free, not
/// alloc-only. Disposing here makes the comparison apples-to-apples AND bounds
/// resident memory: without it, a fast allocator (e.g. the calloc/VirtualAlloc
/// np.zeros fast path, or np.empty) leaks one buffer per op, and BenchmarkDotNet
/// runs thousands of ops per iteration — on Windows every untouched-but-committed
/// buffer still charges commit, so the 10M (80 MB) cases hit OutOfMemoryException
/// before finalizers can reclaim them. (Pre-existing: np.empty(10M) already
/// OOM'd this way; the old np.zeros only escaped by being ~14 ms/op, which
/// throttled BDN to a couple of ops per iteration.)
/// </para>
/// </summary>
[BenchmarkCategory("Creation")]
public class CreationBenchmarks : TypedBenchmarkBase
{
    private NDArray _source = null!;

    [Params(ArraySizeSource.Small, ArraySizeSource.Medium, ArraySizeSource.Large)]
    public override int N { get; set; }

    [ParamsSource(nameof(Types))]
    public new NPTypeCode DType { get; set; }

    public static IEnumerable<NPTypeCode> Types => TypeParameterSource.CommonTypes;

    [GlobalSetup]
    public void Setup()
    {
        _source = CreateRandomArray(N, DType);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _source = null!;
        GC.Collect();
    }

    // ========================================================================
    // Initialized Arrays
    // ========================================================================

    [Benchmark(Description = "np.zeros(N)")]
    [BenchmarkCategory("Initialized")]
    public void Zeros() { using var _ = np.zeros(new Shape(N), DType); }

    [Benchmark(Description = "np.ones(N)")]
    [BenchmarkCategory("Initialized")]
    public void Ones() { using var _ = np.ones(new Shape(N), DType); }

    [Benchmark(Description = "np.full(N, value)")]
    [BenchmarkCategory("Initialized")]
    public void Full() { using var _ = np.full(new Shape(N), 42, DType); }

    [Benchmark(Description = "np.empty(N)")]
    [BenchmarkCategory("Uninitialized")]
    public void Empty() { using var _ = np.empty(new Shape(N), DType); }

    // ========================================================================
    // Range-based
    // ========================================================================

    [Benchmark(Description = "np.arange(N)")]
    [BenchmarkCategory("Range")]
    public void Arange() { using var _ = np.arange(N); }

    [Benchmark(Description = "np.linspace(0, N, N)")]
    [BenchmarkCategory("Range")]
    public void Linspace() { using var _ = np.linspace(0, N, N); }

    // ========================================================================
    // Copy / Conversion
    // ========================================================================

    [Benchmark(Description = "np.copy(a)")]
    [BenchmarkCategory("Copy")]
    public void Copy() { using var _ = np.copy(_source); }

    [Benchmark(Description = "a.copy()")]
    [BenchmarkCategory("Copy")]
    public void CopyMethod() { using var _ = _source.copy(); }

    [Benchmark(Description = "np.copy(a) [asarray equivalent]")]
    [BenchmarkCategory("Convert")]
    public void AsArray() { using var _ = np.copy(_source); }

    // ========================================================================
    // Like-based
    // ========================================================================

    [Benchmark(Description = "np.zeros_like(a)")]
    [BenchmarkCategory("Like")]
    public void ZerosLike() { using var _ = np.zeros_like(_source); }

    [Benchmark(Description = "np.ones_like(a)")]
    [BenchmarkCategory("Like")]
    public void OnesLike() { using var _ = np.ones_like(_source); }

    [Benchmark(Description = "np.empty_like(a)")]
    [BenchmarkCategory("Like")]
    public void EmptyLike() { using var _ = np.empty_like(_source); }

    [Benchmark(Description = "np.full_like(a, 42)")]
    [BenchmarkCategory("Like")]
    public void FullLike() { using var _ = np.full_like(_source, 42); }
}
