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
    private NDArray _matmulA = null!;
    private NDArray _matmulB = null!;

    [Params(ArraySizeSource.Small, ArraySizeSource.Medium, ArraySizeSource.Large)]
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
        // The O(M^3) linalg ops below (matrix_power(3), multi_dot, ...) stay on a 128-capped matrix
        // so the benchmark finishes; matmul, being O(M^2)-cheap per element and the one cell users
        // compare against np.matmul, uses the SAME min(sqrt(N),384) side as LinAlgBenchmarks.MatMul
        // and the backend-profile bench. Sharing a shape with those keeps the merge key from
        // cross-pairing a 128x128 timing with a 316x316 one at the same N.
        int side = Math.Min((int)Math.Sqrt(N), 128);
        _matrixA = np.random.rand(side * side).reshape(side, side);
        _matrixB = np.random.rand(side * side).reshape(side, side);
        int mc = Math.Min((int)Math.Sqrt(N), 384);
        _matmulA = np.random.rand(mc * mc).reshape(mc, mc);
        _matmulB = np.random.rand(mc * mc).reshape(mc, mc);
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
        _matmulA = null!;
        _matmulB = null!;
        GC.Collect();
    }

    [Benchmark(Description = "np.linalg.cross(a, b)")] public NDArray Cross() => np.linalg.cross(_vectorsA, _vectorsB);
    [Benchmark(Description = "np.linalg.diagonal(a)")] public NDArray Diagonal() => np.linalg.diagonal(_matrixA);
    [Benchmark(Description = "np.linalg.matmul(a, b)")] public NDArray MatMul() => np.linalg.matmul(_matmulA, _matmulB);
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
