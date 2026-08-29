using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.CSharp.Infrastructure;

namespace NumSharp.Benchmark.CSharp.Benchmarks.Manipulation;

/// <summary>
/// Benchmarks for NumPy's two-dimensional base family: diag, diagflat, tri, tril, triu,
/// fill_diagonal and the tril/triu index generators.
///
/// <para>
/// The three cost classes here are deliberately separated, because a single geomean over them would
/// be meaningless:
/// </para>
/// <list type="bullet">
///   <item><b>O(1) view</b> — <c>np.diag(2-D)</c> extracts a read-only diagonal by stride addition,
///         so its timing tracks view-construction overhead, not throughput.</item>
///   <item><b>O(N) fill/copy</b> — <c>tri</c>, <c>tril</c>, <c>triu</c>. NumPy builds an N×M boolean
///         mask (<c>greater_equal.outer</c>) and runs a full <c>where</c> select; NumSharp blits the
///         retained contiguous column run of each row over a zero-allocated result. Half the memory
///         traffic and no mask temporary, so this is where the family should win.</item>
///   <item><b>O(sqrt N) sparse</b> — <c>fill_diagonal</c> touches only min(rows, cols) elements, and
///         <c>diag(1-D)</c>/<c>diagflat</c> are dominated by allocating the n² zero matrix.</item>
/// </list>
///
/// <para>
/// <c>tril_indices</c>/<c>triu_indices</c> are the algorithmic outlier: NumPy materialises an N×M
/// bool mask plus two broadcast coordinate grids and boolean-gathers them, while NumSharp emits the
/// (separable) row/column runs directly in O(output). Expect the largest ratio here.
/// </para>
///
/// NumPy twins live in numpy_benchmark.py::run_manipulation_benchmarks (float64); the
/// [Benchmark(Description)] labels normalize 1:1 onto those names for the (op, dtype, N) merge join.
/// </summary>
[BenchmarkCategory("Manipulation", "DiagTri")]
public class DiagTriBenchmarks : OfficialBenchmarkBase
{
    private NDArray _arr2D = null!;
    private NDArray _arr1D = null!;
    private NDArray _fillTarget = null!;
    private int _rows;
    private int _cols;

    [Params(ArraySizeSource.Small, ArraySizeSource.Medium, ArraySizeSource.Large)]
    public override int N { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        np.random.seed(Seed);
        _rows = (int)Math.Sqrt(N);
        _cols = N / _rows;
        _arr2D = CreateRandomArray2D(_rows, _cols, DType);
        // diag/diagflat on a 1-D input build an n x n matrix, so the vector is sized to the
        // square root of N — keeping the CONSTRUCTED array at ~N elements like every other case.
        _arr1D = CreateRandomArray(_rows, DType);
        _fillTarget = CreateRandomArray2D(_rows, _cols, DType, seed: 43);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _arr2D = null!;
        _arr1D = null!;
        _fillTarget = null!;
        GC.Collect();
    }

    // ------------------------------------------------------------------------
    // Diagonal extraction (O(1) view) vs construction (O(n^2) alloc + strided scatter)
    // ------------------------------------------------------------------------

    [Benchmark(Description = "np.diag(a2d)")]
    [BenchmarkCategory("Diag")]
    public NDArray DiagExtract() => np.diag(_arr2D);

    [Benchmark(Description = "np.diag(a1d)")]
    [BenchmarkCategory("Diag")]
    public NDArray DiagConstruct() => np.diag(_arr1D);

    [Benchmark(Description = "np.diagflat(a1d)")]
    [BenchmarkCategory("Diag")]
    public NDArray Diagflat() => np.diagflat(_arr1D);

    // ------------------------------------------------------------------------
    // Triangular fill / mask (the O(N) row-run blit)
    // ------------------------------------------------------------------------

    [Benchmark(Description = "np.tri(n)")]
    [BenchmarkCategory("Tri")]
    public NDArray Tri() => np.tri(_rows, _cols, 0, DType);

    [Benchmark(Description = "np.tril(a2d)")]
    [BenchmarkCategory("Tri")]
    public NDArray Tril() => np.tril(_arr2D);

    [Benchmark(Description = "np.triu(a2d)")]
    [BenchmarkCategory("Tri")]
    public NDArray Triu() => np.triu(_arr2D);

    // ------------------------------------------------------------------------
    // Sparse in-place write (O(min(rows, cols)))
    // ------------------------------------------------------------------------

    [Benchmark(Description = "np.fill_diagonal(a2d)")]
    [BenchmarkCategory("FillDiagonal")]
    public NDArray FillDiagonal()
    {
        np.fill_diagonal(_fillTarget, GetScalar(DType, 1));
        return _fillTarget;
    }

    // ------------------------------------------------------------------------
    // Index generators (NumPy: mask + broadcast + boolean gather; NumSharp: direct emit)
    // ------------------------------------------------------------------------

    [Benchmark(Description = "np.tril_indices(n)")]
    [BenchmarkCategory("TriIndices")]
    public NDArray TrilIndices() => np.tril_indices(_rows, 0, _cols)[0];

    [Benchmark(Description = "np.triu_indices(n)")]
    [BenchmarkCategory("TriIndices")]
    public NDArray TriuIndices() => np.triu_indices(_rows, 0, _cols)[0];
}
