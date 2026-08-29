using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.CSharp.Infrastructure;

namespace NumSharp.Benchmark.CSharp.Benchmarks.Slicing;

/// <summary>
/// Benchmarks for slicing and view operations.
/// Tests contiguous vs strided slices and their impact on subsequent operations.
/// </summary>
[BenchmarkCategory("Slicing")]
public class SliceBenchmarks : OfficialBenchmarkBase
{
    private NDArray _arr1D = null!;
    private NDArray _arr2D = null!;
    private NDArray _contiguousSlice = null!;
    private NDArray _stridedSlice = null!;
    private NDArray _rowSlice = null!;
    private NDArray _colSlice = null!;
    // Proportional slice range so copy/multiply work scales with N (see below).
    private string _sliceRange = null!;

    [Params(ArraySizeSource.Small, ArraySizeSource.Medium, ArraySizeSource.Large)]
    public override int N { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        np.random.seed(Seed);

        _arr1D = CreateRandomArray(N, DType);

        // Middle 80% of the array ([N/10 : 9N/10]) — a real O(N) span, unlike the fixed
        // "100:1000" that did the same ~900 elements at every N (non-comparable across sizes).
        _sliceRange = $"{N / 10}:{N / 10 * 9}";

        var rows = (int)Math.Sqrt(N);
        var cols = N / rows;
        _arr2D = CreateRandomArray2D(rows, cols, DType);

        // Pre-slice for operation benchmarks
        _contiguousSlice = _arr1D["100:1000"];           // Contiguous view
        _stridedSlice = _arr1D["::2"];                    // Every other element (strided)
        _rowSlice = _arr2D["10:100, :"];                  // Row slice
        _colSlice = _arr2D[":, 10:100"];                  // Column slice (strided)
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _arr1D = null!;
        _arr2D = null!;
        _contiguousSlice = null!;
        _stridedSlice = null!;
        _rowSlice = null!;
        _colSlice = null!;
        GC.Collect();
    }

    // ========================================================================
    // Slice Creation (view creation time)
    // ========================================================================

    [Benchmark(Description = "a[100:1000] (contiguous slice)")]
    [BenchmarkCategory("Create")]
    public NDArray Slice_Contiguous() => _arr1D["100:1000"];

    [Benchmark(Description = "a[::2] (strided slice)")]
    [BenchmarkCategory("Create")]
    public NDArray Slice_Strided() => _arr1D["::2"];

    [Benchmark(Description = "a[::-1] (reversed)")]
    [BenchmarkCategory("Create")]
    public NDArray Slice_Reversed() => _arr1D["::-1"];

    [Benchmark(Description = "a[10:100, :] (row slice 2D)")]
    [BenchmarkCategory("Create")]
    public NDArray Slice_Row2D() => _arr2D["10:100, :"];

    [Benchmark(Description = "a[:, 10:100] (col slice 2D)")]
    [BenchmarkCategory("Create")]
    public NDArray Slice_Col2D() => _arr2D[":, 10:100"];

    // ========================================================================
    // Operations on Slices (measure view overhead)
    // ========================================================================

    [Benchmark(Description = "np.sum(contiguous_slice)")]
    [BenchmarkCategory("SumSlice")]
    public NDArray Sum_ContiguousSlice() => np.sum(_contiguousSlice);

    [Benchmark(Description = "np.sum(strided_slice)")]
    [BenchmarkCategory("SumSlice")]
    public NDArray Sum_StridedSlice() => np.sum(_stridedSlice);

    [Benchmark(Description = "np.sum(row_slice)")]
    [BenchmarkCategory("SumSlice")]
    public NDArray Sum_RowSlice() => np.sum(_rowSlice);

    [Benchmark(Description = "np.sum(col_slice)")]
    [BenchmarkCategory("SumSlice")]
    public NDArray Sum_ColSlice() => np.sum(_colSlice);

    // ========================================================================
    // Slice + Operation (combined)
    // ========================================================================

    [Benchmark(Description = "a[N/10:9N/10] * 2")]
    [BenchmarkCategory("SliceOp")]
    public NDArray SliceAndOp_Contiguous() => _arr1D[_sliceRange] * 2;

    [Benchmark(Description = "a[::2] * 2")]
    [BenchmarkCategory("SliceOp")]
    public NDArray SliceAndOp_Strided() => _arr1D["::2"] * 2;

    // ========================================================================
    // Copy vs View
    // ========================================================================

    [Benchmark(Description = "a[N/10:9N/10].copy()")]
    [BenchmarkCategory("Copy")]
    public NDArray SliceCopy() => _arr1D[_sliceRange].copy();

    [Benchmark(Description = "np.copy(a[N/10:9N/10])")]
    [BenchmarkCategory("Copy")]
    public NDArray NpCopy() => np.copy(_arr1D[_sliceRange]);
}
