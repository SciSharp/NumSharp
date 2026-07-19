using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.CSharp.Infrastructure;

namespace NumSharp.Benchmark.CSharp.Benchmarks.Manipulation;

/// <summary>
/// Benchmarks for the reversal / rotation / transpose-alias view ops merged into journey2, plus the
/// value-dependent trim_zeros crop: flip, fliplr, flipud, rot90, permute_dims, matrix_transpose,
/// trim_zeros. The first six are O(1)/O(ndim) pure views (stride negation / axis permutation via
/// Storage.Alias — no data movement), so their timing tracks view-construction overhead; trim_zeros
/// is the one doing an O(N) nonzero bounding-box scan (the op with the standout NumPy-vs-NumSharp
/// speedup). NumPy twins live in numpy_benchmark.py::run_manipulation_benchmarks (float64), and the
/// [Benchmark(Description)] labels normalize 1:1 onto those names for the (op, dtype, N) merge join.
/// </summary>
[BenchmarkCategory("Manipulation", "FlipRot")]
public class FlipRotBenchmarks : BenchmarkBase
{
    private NDArray _arr2D = null!;

    [Params(ArraySizeSource.Medium, ArraySizeSource.Large)]
    public override int N { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        np.random.seed(Seed);
        var rows = (int)Math.Sqrt(N);
        var cols = N / rows;
        _arr2D = np.random.rand(rows, cols) * 100;
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _arr2D = null!;
        GC.Collect();
    }

    // ------------------------------------------------------------------------
    // Reversal views (stride negation)
    // ------------------------------------------------------------------------

    [Benchmark(Description = "np.flip(a)")]
    [BenchmarkCategory("Flip")]
    public NDArray Flip() => np.flip(_arr2D);

    [Benchmark(Description = "np.fliplr(a)")]
    [BenchmarkCategory("Flip")]
    public NDArray FlipLr() => np.fliplr(_arr2D);

    [Benchmark(Description = "np.flipud(a)")]
    [BenchmarkCategory("Flip")]
    public NDArray FlipUd() => np.flipud(_arr2D);

    // ------------------------------------------------------------------------
    // Rotation (flip + transpose composition)
    // ------------------------------------------------------------------------

    [Benchmark(Description = "np.rot90(a)")]
    [BenchmarkCategory("Rot90")]
    public NDArray Rot90() => np.rot90(_arr2D);

    // ------------------------------------------------------------------------
    // Transpose aliases (NumPy 2.x / Array-API)
    // ------------------------------------------------------------------------

    [Benchmark(Description = "np.permute_dims(a)")]
    [BenchmarkCategory("Transpose")]
    public NDArray PermuteDims() => np.permute_dims(_arr2D);

    [Benchmark(Description = "np.matrix_transpose(a)")]
    [BenchmarkCategory("Transpose")]
    public NDArray MatrixTranspose() => np.matrix_transpose(_arr2D);

    // ------------------------------------------------------------------------
    // Value-dependent crop (O(N) nonzero bounding-box scan)
    // ------------------------------------------------------------------------

    [Benchmark(Description = "np.trim_zeros(a)")]
    [BenchmarkCategory("TrimZeros")]
    public NDArray TrimZeros() => np.trim_zeros(_arr2D, "fb");
}
