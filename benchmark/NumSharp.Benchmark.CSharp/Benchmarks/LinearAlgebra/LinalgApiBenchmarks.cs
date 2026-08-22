using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.CSharp.Infrastructure;

namespace NumSharp.Benchmark.CSharp.Benchmarks.LinearAlgebra;

/// <summary>
/// Managed np.linalg compositions. Factorisations and factorisation-dependent modes live in the
/// isolated OpenBLAS subsystem because NumSharp.Core intentionally has no LAPACK fallback.
/// </summary>
[BenchmarkCategory("LinearAlgebra", "LinalgApi")]
public class LinalgApiBenchmarks : BenchmarkBase
{
    private NDArray _vector = null!;
    private NDArray _shortVector = null!;
    private NDArray _vectorsA = null!;
    private NDArray _vectorsB = null!;
    private NDArray _matrixA = null!;
    private NDArray _matrixB = null!;

    [Params(ArraySizeSource.Small, ArraySizeSource.Medium)]
    public override int N { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        np.random.seed(Seed);
        _vector = np.random.rand(N);
        _shortVector = np.random.rand((int)Math.Sqrt(N));
        int vectorRows = N / 3;
        _vectorsA = np.random.rand(vectorRows * 3).reshape(vectorRows, 3);
        _vectorsB = np.random.rand(vectorRows * 3).reshape(vectorRows, 3);
        int side = Math.Min((int)Math.Sqrt(N), 128);
        _matrixA = np.random.rand(side * side).reshape(side, side);
        _matrixB = np.random.rand(side * side).reshape(side, side);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _vector = null!;
        _shortVector = null!;
        _vectorsA = null!;
        _vectorsB = null!;
        _matrixA = null!;
        _matrixB = null!;
        GC.Collect();
    }

    [Benchmark(Description = "np.linalg.cross(a, b)")] public NDArray Cross() => np.linalg.cross(_vectorsA, _vectorsB);
    [Benchmark(Description = "np.linalg.diagonal(a)")] public NDArray Diagonal() => np.linalg.diagonal(_matrixA);
    [Benchmark(Description = "np.linalg.matmul(a, b)")] public NDArray MatMul() => np.linalg.matmul(_matrixA, _matrixB);
    [Benchmark(Description = "np.linalg.matrix_norm(a)")] public NDArray MatrixNorm() => np.linalg.matrix_norm(_matrixA);
    [Benchmark(Description = "np.linalg.matrix_power(a, 3)")] public NDArray MatrixPower() => np.linalg.matrix_power(_matrixA, 3);
    [Benchmark(Description = "np.linalg.matrix_transpose(a)")] public NDArray MatrixTranspose() => np.linalg.matrix_transpose(_matrixA);
    [Benchmark(Description = "np.linalg.multi_dot(arrays)")] public NDArray MultiDot() => np.linalg.multi_dot(new[] { _matrixA, _matrixB, _matrixA });
    [Benchmark(Description = "np.linalg.norm(a)")] public NDArray Norm() => np.linalg.norm(_vector);
    [Benchmark(Description = "np.linalg.outer(a, b)")] public NDArray Outer() => np.linalg.outer(_shortVector, _shortVector);
    [Benchmark(Description = "np.linalg.tensordot(a, b)")] public NDArray TensorDot() => np.linalg.tensordot(_matrixA, _matrixB, 1);
    [Benchmark(Description = "np.linalg.trace(a)")] public NDArray Trace() => np.linalg.trace(_matrixA);
    [Benchmark(Description = "np.linalg.vecdot(a, b)")] public NDArray VecDot() => np.linalg.vecdot(_vector, _vector);
    [Benchmark(Description = "np.linalg.vector_norm(a)")] public NDArray VectorNorm() => np.linalg.vector_norm(_vector);
}
