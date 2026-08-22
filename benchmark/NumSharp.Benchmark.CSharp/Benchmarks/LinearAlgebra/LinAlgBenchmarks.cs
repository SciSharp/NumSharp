using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.CSharp.Infrastructure;

namespace NumSharp.Benchmark.CSharp.Benchmarks.LinearAlgebra;

/// <summary>
/// Linear algebra: dot (1-D inner product), outer (1-D outer product), matmul (2-D).
/// float64 only — the conventional linear-algebra dtype.
///
/// N is the vector length. dot uses length-N vectors (O(N)); outer uses length-isqrt(N)
/// vectors so the result is ~N elements (O(N)); matmul squares isqrt(N) capped at 384 so
/// the O(M^3) cost stays bounded (a true M=3162 matmul at N=10M would be tens of seconds).
/// The Python side mirrors these exact dimensions.
/// </summary>
[BenchmarkCategory("LinearAlgebra")]
public class LinAlgBenchmarks : BenchmarkBase
{
    private NDArray _v = null!;        // length-N vector (dot)
    private NDArray _vM = null!;       // length-M vector (outer)
    private NDArray _matA = null!;     // Mc x Mc matrix (matmul)
    private NDArray _matB = null!;
    private NDArray _vMc = null!;

    [Params(ArraySizeSource.Small, ArraySizeSource.Medium, ArraySizeSource.Large)]
    public override int N { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        np.random.seed(Seed);
        int m = (int)System.Math.Sqrt(N);
        int mc = System.Math.Min(m, 384);
        _v = np.random.rand(N);
        _vM = np.random.rand(m);
        _matA = np.random.rand(mc * mc).reshape(mc, mc);
        _matB = np.random.rand(mc * mc).reshape(mc, mc);
        _vMc = np.random.rand(mc);
    }

    [GlobalCleanup]
    public void Cleanup() { _v = null!; _vM = null!; _matA = null!; _matB = null!; _vMc = null!; GC.Collect(); }

    [Benchmark(Description = "np.dot(a, b)")] public NDArray Dot() => np.dot(_v, _v);
    [Benchmark(Description = "np.outer(a, b)")] public NDArray Outer() => np.outer(_vM, _vM);
    [Benchmark(Description = "np.matmul(A, B)")] public NDArray MatMul() => np.matmul(_matA, _matB);
    [Benchmark(Description = "np.diagonal(a)")] public NDArray Diagonal() => np.diagonal(_matA);
    [Benchmark(Description = "np.einsum(subscripts, operands)")] public NDArray Einsum() => np.einsum("i,i->", _v, _v);
    [Benchmark(Description = "np.einsum_path(subscripts, operands)")] public object EinsumPath() => np.einsum_path("ij,jk->ik", new[] { _matA, _matB });
    [Benchmark(Description = "np.matvec(a, b)")] public NDArray MatVec() => np.matvec(_matA, _vMc);
    [Benchmark(Description = "np.tensordot(a, b)")] public NDArray TensorDot() => np.tensordot(_v, _v, 1);
    [Benchmark(Description = "np.vdot(a, b)")] public NDArray VDot() => np.vdot(_v, _v);
    [Benchmark(Description = "np.vecdot(a, b)")] public NDArray VecDot() => np.vecdot(_v, _v);
    [Benchmark(Description = "np.vecmat(a, b)")] public NDArray VecMat() => np.vecmat(_vMc, _matA);
}
