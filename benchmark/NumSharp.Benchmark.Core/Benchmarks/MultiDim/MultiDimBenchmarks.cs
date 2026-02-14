using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.Core.Infrastructure;

namespace NumSharp.Benchmark.Core.Benchmarks.MultiDim;

/// <summary>
/// Benchmarks comparing 1D, 2D, and 3D array operations.
/// Tests overhead of multi-dimensional indexing vs contiguous memory access.
/// </summary>
[BenchmarkCategory("MultiDim")]
public class MultiDimBenchmarks : BenchmarkBase
{
    private NDArray _arr1D = null!;
    private NDArray _arr2D = null!;
    private NDArray _arr3D = null!;
    private NDArray _arr1D_b = null!;
    private NDArray _arr2D_b = null!;
    private NDArray _arr3D_b = null!;

    // Total elements constant across all dimensions
    [Params(1_000_000, 10_000_000)]
    public override int N { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        np.random.seed(Seed);

        // 1D: N elements
        _arr1D = np.random.rand(N) * 100;
        _arr1D_b = np.random.rand(N) * 100;

        // 2D: sqrt(N) x sqrt(N) ≈ N elements
        var dim2d = (int)Math.Sqrt(N);
        _arr2D = np.random.rand(dim2d, dim2d) * 100;
        _arr2D_b = np.random.rand(dim2d, dim2d) * 100;

        // 3D: cbrt(N) x cbrt(N) x cbrt(N) ≈ N elements
        var dim3d = (int)Math.Pow(N, 1.0 / 3);
        _arr3D = np.random.rand(dim3d, dim3d, dim3d) * 100;
        _arr3D_b = np.random.rand(dim3d, dim3d, dim3d) * 100;
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _arr1D = null!;
        _arr2D = null!;
        _arr3D = null!;
        _arr1D_b = null!;
        _arr2D_b = null!;
        _arr3D_b = null!;
        GC.Collect();
    }

    // ========================================================================
    // Element-wise Addition
    // ========================================================================

    [Benchmark(Description = "1D: a + b")]
    [BenchmarkCategory("Add")]
    public NDArray Add_1D() => _arr1D + _arr1D_b;

    [Benchmark(Description = "2D: a + b")]
    [BenchmarkCategory("Add")]
    public NDArray Add_2D() => _arr2D + _arr2D_b;

    [Benchmark(Description = "3D: a + b")]
    [BenchmarkCategory("Add")]
    public NDArray Add_3D() => _arr3D + _arr3D_b;

    // ========================================================================
    // Reductions
    // ========================================================================

    [Benchmark(Description = "1D: np.sum(a)")]
    [BenchmarkCategory("Sum")]
    public NDArray Sum_1D() => np.sum(_arr1D);

    [Benchmark(Description = "2D: np.sum(a)")]
    [BenchmarkCategory("Sum")]
    public NDArray Sum_2D() => np.sum(_arr2D);

    [Benchmark(Description = "3D: np.sum(a)")]
    [BenchmarkCategory("Sum")]
    public NDArray Sum_3D() => np.sum(_arr3D);

    // ========================================================================
    // Axis Reductions
    // ========================================================================

    [Benchmark(Description = "2D: np.sum(a, axis=0)")]
    [BenchmarkCategory("SumAxis")]
    public NDArray Sum_2D_Axis0() => np.sum(_arr2D, axis: 0);

    [Benchmark(Description = "2D: np.sum(a, axis=1)")]
    [BenchmarkCategory("SumAxis")]
    public NDArray Sum_2D_Axis1() => np.sum(_arr2D, axis: 1);

    [Benchmark(Description = "3D: np.sum(a, axis=0)")]
    [BenchmarkCategory("SumAxis")]
    public NDArray Sum_3D_Axis0() => np.sum(_arr3D, axis: 0);

    [Benchmark(Description = "3D: np.sum(a, axis=2)")]
    [BenchmarkCategory("SumAxis")]
    public NDArray Sum_3D_Axis2() => np.sum(_arr3D, axis: 2);

    // ========================================================================
    // Unary Operations
    // ========================================================================

    [Benchmark(Description = "1D: np.sqrt(a)")]
    [BenchmarkCategory("Sqrt")]
    public NDArray Sqrt_1D() => np.sqrt(_arr1D);

    [Benchmark(Description = "2D: np.sqrt(a)")]
    [BenchmarkCategory("Sqrt")]
    public NDArray Sqrt_2D() => np.sqrt(_arr2D);

    [Benchmark(Description = "3D: np.sqrt(a)")]
    [BenchmarkCategory("Sqrt")]
    public NDArray Sqrt_3D() => np.sqrt(_arr3D);
}
