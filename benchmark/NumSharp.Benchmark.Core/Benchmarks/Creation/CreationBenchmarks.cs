using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.Core.Infrastructure;

namespace NumSharp.Benchmark.Core.Benchmarks.Creation;

/// <summary>
/// Benchmarks for array creation functions.
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
    public NDArray Zeros() => np.zeros(new Shape(N), DType);

    [Benchmark(Description = "np.ones(N)")]
    [BenchmarkCategory("Initialized")]
    public NDArray Ones() => np.ones(new Shape(N), DType);

    [Benchmark(Description = "np.full(N, value)")]
    [BenchmarkCategory("Initialized")]
    public NDArray Full() => np.full(new Shape(N), 42, DType);

    [Benchmark(Description = "np.empty(N)")]
    [BenchmarkCategory("Uninitialized")]
    public NDArray Empty() => np.empty(new Shape(N), DType);

    // ========================================================================
    // Range-based
    // ========================================================================

    [Benchmark(Description = "np.arange(N)")]
    [BenchmarkCategory("Range")]
    public NDArray Arange() => np.arange(N);

    [Benchmark(Description = "np.linspace(0, N, N)")]
    [BenchmarkCategory("Range")]
    public NDArray Linspace() => np.linspace(0, N, N);

    // ========================================================================
    // Copy / Conversion
    // ========================================================================

    [Benchmark(Description = "np.copy(a)")]
    [BenchmarkCategory("Copy")]
    public NDArray Copy() => np.copy(_source);

    [Benchmark(Description = "a.copy()")]
    [BenchmarkCategory("Copy")]
    public NDArray CopyMethod() => _source.copy();

    [Benchmark(Description = "np.copy(a) [asarray equivalent]")]
    [BenchmarkCategory("Convert")]
    public NDArray AsArray() => np.copy(_source);

    // ========================================================================
    // Like-based
    // ========================================================================

    [Benchmark(Description = "np.zeros_like(a)")]
    [BenchmarkCategory("Like")]
    public NDArray ZerosLike() => np.zeros_like(_source);

    [Benchmark(Description = "np.ones_like(a)")]
    [BenchmarkCategory("Like")]
    public NDArray OnesLike() => np.ones_like(_source);

    [Benchmark(Description = "np.empty_like(a)")]
    [BenchmarkCategory("Like")]
    public NDArray EmptyLike() => np.empty_like(_source);

    [Benchmark(Description = "np.full_like(a, 42)")]
    [BenchmarkCategory("Like")]
    public NDArray FullLike() => np.full_like(_source, 42);
}
