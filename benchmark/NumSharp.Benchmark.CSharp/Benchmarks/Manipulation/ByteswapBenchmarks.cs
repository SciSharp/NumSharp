using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.CSharp.Infrastructure;

namespace NumSharp.Benchmark.CSharp.Benchmarks.Manipulation;

/// <summary>
/// Benchmark for ndarray.byteswap — reverse the bytes within each element (toggle the endian
/// representation) into a fresh copy. Unlike the O(1) view ops elsewhere in this suite it does real
/// O(N) work: a fused read -> VPSHUFB byte-reversal -> write pass (non-temporal stores above an
/// L2-scale threshold, skipping the read-for-ownership a fresh destination would otherwise pay).
/// The NumPy twin lives in numpy_benchmark.py::run_manipulation_benchmarks (float64, arr_2d), and
/// the [Benchmark(Description)] label "np.byteswap(a)" normalizes onto "np.byteswap" for the
/// (op, dtype, N) merge join. Measured at float64 to match the manipulation suite convention;
/// per-dtype correctness across all 15 dtypes is covered by the differential-fuzz oracle.
/// </summary>
[BenchmarkCategory("Manipulation", "Byteswap")]
public class ByteswapBenchmarks : BenchmarkBase
{
    private NDArray _arr2D = null!;

    [Params(ArraySizeSource.Small, ArraySizeSource.Medium, ArraySizeSource.Large)]
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

    [Benchmark(Description = "np.byteswap(a)")]
    [BenchmarkCategory("Byteswap")]
    public NDArray Byteswap() => _arr2D.byteswap();
}
