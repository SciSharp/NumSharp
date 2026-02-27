using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.GraphEngine.Infrastructure;

namespace NumSharp.Benchmark.GraphEngine.Benchmarks.Allocation;

/// <summary>
/// End-to-end NumSharp benchmarks measuring allocation impact:
/// - Array creation functions that allocate memory
/// - Operations that allocate output arrays
///
/// These complement the micro-benchmarks by showing real-world impact.
/// </summary>
[BenchmarkCategory("Allocation", "NumSharp")]
public class NumSharpAllocationBenchmarks : TypedBenchmarkBase
{
    [Params(ArraySizeSource.Small, ArraySizeSource.Medium, ArraySizeSource.Large)]
    public override int N { get; set; }

    [ParamsSource(nameof(Types))]
    public new NPTypeCode DType { get; set; }

    /// <summary>
    /// Types to benchmark. Focus on common types to keep benchmark time reasonable.
    /// </summary>
    public static IEnumerable<NPTypeCode> Types => TypeParameterSource.MinimalTypes;

    private NDArray _a = null!;
    private NDArray _b = null!;

    [GlobalSetup]
    public void Setup()
    {
        _a = CreateRandomArray(N, DType, Seed);
        _b = CreateRandomArray(N, DType, Seed + 1);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _a = null!;
        _b = null!;
        GC.Collect();
    }

    // ========================================================================
    // Array Creation - Uninitialized
    // ========================================================================

    [Benchmark(Description = "np.empty(N) - uninitialized")]
    [BenchmarkCategory("Creation", "Uninitialized")]
    public NDArray Empty() => np.empty(new Shape(N), DType);

    // ========================================================================
    // Array Creation - Zero-Initialized
    // This is the target for AllocZeroed optimization
    // ========================================================================

    [Benchmark(Description = "np.zeros(N) - zero-initialized")]
    [BenchmarkCategory("Creation", "ZeroInit")]
    public NDArray Zeros() => np.zeros(new Shape(N), DType);

    // ========================================================================
    // Array Creation - Value-Initialized
    // ========================================================================

    [Benchmark(Description = "np.ones(N) - ones")]
    [BenchmarkCategory("Creation", "ValueInit")]
    public NDArray Ones() => np.ones(new Shape(N), DType);

    [Benchmark(Description = "np.full(N, 42) - fill value")]
    [BenchmarkCategory("Creation", "ValueInit")]
    public NDArray Full() => np.full(new Shape(N), 42, DType);

    // ========================================================================
    // Array Creation - Sequential
    // ========================================================================

    [Benchmark(Description = "np.arange(N) - sequential")]
    [BenchmarkCategory("Creation", "Sequential")]
    public NDArray Arange() => np.arange(N);

    // ========================================================================
    // Operations That Allocate Output
    // Shows allocation overhead in computation context
    // ========================================================================

    [Benchmark(Description = "a + b - binary op output alloc")]
    [BenchmarkCategory("Operation", "Binary")]
    public NDArray Add() => _a + _b;

    [Benchmark(Description = "a * b - binary op output alloc")]
    [BenchmarkCategory("Operation", "Binary")]
    public NDArray Multiply() => _a * _b;

    [Benchmark(Description = "np.sqrt(a) - unary op output alloc")]
    [BenchmarkCategory("Operation", "Unary")]
    public NDArray Sqrt() => np.sqrt(_a.astype(np.float64));

    // ========================================================================
    // Copy Operations
    // ========================================================================

    [Benchmark(Description = "np.copy(a) - explicit copy")]
    [BenchmarkCategory("Copy")]
    public NDArray Copy() => np.copy(_a);

    [Benchmark(Description = "a.copy() - method copy")]
    [BenchmarkCategory("Copy")]
    public NDArray CopyMethod() => _a.copy();
}
