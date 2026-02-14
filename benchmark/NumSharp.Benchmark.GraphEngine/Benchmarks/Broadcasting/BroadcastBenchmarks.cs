using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.GraphEngine.Infrastructure;

namespace NumSharp.Benchmark.GraphEngine.Benchmarks.Broadcasting;

/// <summary>
/// Benchmarks for broadcasting operations.
/// Tests various broadcasting patterns: scalar, row, column, and general.
/// </summary>
[BenchmarkCategory("Broadcasting")]
public class BroadcastBenchmarks : BenchmarkBase
{
    private NDArray _matrix = null!;
    private NDArray _rowVector = null!;
    private NDArray _colVector = null!;
    private NDArray _scalar = null!;
    private NDArray _tensor3D = null!;
    private NDArray _broadcast2D = null!;

    [Params(1000, 3162)]  // 1M and ~10M elements when squared
    public int MatrixSize { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        np.random.seed(Seed);
        var n = MatrixSize;

        _matrix = (np.random.rand(n, n) * 100).astype(np.float64);
        _rowVector = (np.random.rand(n) * 100).astype(np.float64);
        _colVector = (np.random.rand(n, 1) * 100).astype(np.float64);
        _scalar = np.array(42.0);

        // 3D tensor for more complex broadcasting
        var d = (int)Math.Pow(n * n, 1.0 / 3);  // ~same total elements
        _tensor3D = (np.random.rand(d, d, d) * 100).astype(np.float64);
        _broadcast2D = (np.random.rand(d, d) * 100).astype(np.float64);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _matrix = null!;
        _rowVector = null!;
        _colVector = null!;
        _scalar = null!;
        _tensor3D = null!;
        _broadcast2D = null!;
        GC.Collect();
    }

    // ========================================================================
    // Scalar Broadcasting
    // ========================================================================

    [Benchmark(Description = "matrix + scalar")]
    [BenchmarkCategory("Scalar")]
    public NDArray Broadcast_Scalar() => _matrix + _scalar;

    [Benchmark(Description = "matrix * 2.0")]
    [BenchmarkCategory("Scalar")]
    public NDArray Broadcast_ScalarLiteral() => _matrix * 2.0;

    // ========================================================================
    // Row Broadcasting (N,M) + (M,) -> (N,M)
    // ========================================================================

    [Benchmark(Description = "matrix + row_vector (N,M)+(M,)")]
    [BenchmarkCategory("Row")]
    public NDArray Broadcast_Row_Add() => _matrix + _rowVector;

    [Benchmark(Description = "matrix * row_vector (N,M)*(M,)")]
    [BenchmarkCategory("Row")]
    public NDArray Broadcast_Row_Mul() => _matrix * _rowVector;

    // ========================================================================
    // Column Broadcasting (N,M) + (N,1) -> (N,M)
    // ========================================================================

    [Benchmark(Description = "matrix + col_vector (N,M)+(N,1)")]
    [BenchmarkCategory("Column")]
    public NDArray Broadcast_Col_Add() => _matrix + _colVector;

    [Benchmark(Description = "matrix * col_vector (N,M)*(N,1)")]
    [BenchmarkCategory("Column")]
    public NDArray Broadcast_Col_Mul() => _matrix * _colVector;

    // ========================================================================
    // 3D Broadcasting
    // ========================================================================

    [Benchmark(Description = "tensor3D + matrix2D (D,D,D)+(D,D)")]
    [BenchmarkCategory("3D")]
    public NDArray Broadcast_3D_2D() => _tensor3D + _broadcast2D;

    // ========================================================================
    // np.broadcast_to
    // ========================================================================

    [Benchmark(Description = "np.broadcast_to(row, (N,M))")]
    [BenchmarkCategory("BroadcastTo")]
    public NDArray BroadcastTo_Row()
    {
        var n = MatrixSize;
        return np.broadcast_to(_rowVector, new Shape(n, n));
    }

    [Benchmark(Description = "np.broadcast_to(col, (N,M))")]
    [BenchmarkCategory("BroadcastTo")]
    public NDArray BroadcastTo_Col()
    {
        var n = MatrixSize;
        return np.broadcast_to(_colVector, new Shape(n, n));
    }
}
