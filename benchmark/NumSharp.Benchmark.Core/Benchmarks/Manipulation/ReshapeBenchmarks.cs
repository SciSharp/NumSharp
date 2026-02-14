using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.Core.Infrastructure;

namespace NumSharp.Benchmark.Core.Benchmarks.Manipulation;

/// <summary>
/// Benchmarks for reshape and transpose operations.
/// </summary>
[BenchmarkCategory("Manipulation", "Reshape")]
public class ReshapeBenchmarks : BenchmarkBase
{
    private NDArray _arr1D = null!;
    private NDArray _arr2D = null!;
    private NDArray _arr3D = null!;

    [Params(ArraySizeSource.Medium, ArraySizeSource.Large)]
    public override int N { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        np.random.seed(Seed);
        _arr1D = np.random.rand(N) * 100;

        var rows = (int)Math.Sqrt(N);
        var cols = N / rows;
        _arr2D = np.random.rand(rows, cols) * 100;

        var d = (int)Math.Pow(N, 1.0 / 3);
        _arr3D = np.random.rand(d, d, d) * 100;
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _arr1D = null!;
        _arr2D = null!;
        _arr3D = null!;
        GC.Collect();
    }

    // ========================================================================
    // Reshape
    // ========================================================================

    [Benchmark(Description = "reshape 1D -> 2D")]
    [BenchmarkCategory("Reshape")]
    public NDArray Reshape_1D_to_2D()
    {
        var rows = (int)Math.Sqrt(N);
        var cols = N / rows;
        return _arr1D.reshape(rows, cols);
    }

    [Benchmark(Description = "reshape 2D -> 1D")]
    [BenchmarkCategory("Reshape")]
    public NDArray Reshape_2D_to_1D() => _arr2D.reshape(-1);

    [Benchmark(Description = "reshape 1D -> 3D")]
    [BenchmarkCategory("Reshape")]
    public NDArray Reshape_1D_to_3D()
    {
        var d = (int)Math.Pow(N, 1.0 / 3);
        return _arr1D.reshape(d, d, -1);
    }

    [Benchmark(Description = "np.reshape(a, shape)")]
    [BenchmarkCategory("Reshape")]
    public NDArray NpReshape()
    {
        var rows = (int)Math.Sqrt(N);
        var cols = N / rows;
        return np.reshape(_arr1D, new Shape(rows, cols));
    }

    // ========================================================================
    // Transpose
    // ========================================================================

    [Benchmark(Description = "a.T (transpose 2D)")]
    [BenchmarkCategory("Transpose")]
    public NDArray Transpose_2D() => _arr2D.T;

    [Benchmark(Description = "np.transpose(a)")]
    [BenchmarkCategory("Transpose")]
    public NDArray NpTranspose_2D() => np.transpose(_arr2D);

    [Benchmark(Description = "np.transpose(a, axes)")]
    [BenchmarkCategory("Transpose")]
    public NDArray NpTranspose_3D_Axes() => np.transpose(_arr3D, new[] { 2, 0, 1 });

    // ========================================================================
    // Ravel / Flatten
    // ========================================================================

    [Benchmark(Description = "a.ravel() (view)")]
    [BenchmarkCategory("Flatten")]
    public NDArray Ravel() => _arr2D.ravel();

    [Benchmark(Description = "np.ravel(a)")]
    [BenchmarkCategory("Flatten")]
    public NDArray NpRavel() => np.ravel(_arr2D);

    [Benchmark(Description = "a.flatten() (copy)")]
    [BenchmarkCategory("Flatten")]
    public NDArray Flatten() => _arr2D.flatten();
}
