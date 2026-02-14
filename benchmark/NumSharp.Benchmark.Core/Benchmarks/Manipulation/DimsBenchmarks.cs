using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.Core.Infrastructure;

namespace NumSharp.Benchmark.Core.Benchmarks.Manipulation;

/// <summary>
/// Benchmarks for dimension manipulation: squeeze, expand_dims, swapaxes, moveaxis.
/// </summary>
[BenchmarkCategory("Manipulation", "Dims")]
public class DimsBenchmarks : BenchmarkBase
{
    private NDArray _arr1D = null!;
    private NDArray _arr2D = null!;
    private NDArray _arr3D = null!;
    private NDArray _arrWithSingleton = null!;

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

        // Array with singleton dimensions for squeeze tests
        _arrWithSingleton = np.random.rand(rows, 1, cols) * 100;
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _arr1D = null!;
        _arr2D = null!;
        _arr3D = null!;
        _arrWithSingleton = null!;
        GC.Collect();
    }

    // ========================================================================
    // Expand Dims
    // ========================================================================

    [Benchmark(Description = "np.expand_dims(a, axis=0)")]
    [BenchmarkCategory("ExpandDims")]
    public NDArray ExpandDims_Axis0() => np.expand_dims(_arr1D, axis: 0);

    [Benchmark(Description = "np.expand_dims(a, axis=-1)")]
    [BenchmarkCategory("ExpandDims")]
    public NDArray ExpandDims_AxisNeg1() => np.expand_dims(_arr1D, axis: -1);

    [Benchmark(Description = "np.expand_dims(a2D, axis=1)")]
    [BenchmarkCategory("ExpandDims")]
    public NDArray ExpandDims_2D_Axis1() => np.expand_dims(_arr2D, axis: 1);

    // ========================================================================
    // Squeeze
    // ========================================================================

    [Benchmark(Description = "np.squeeze(a)")]
    [BenchmarkCategory("Squeeze")]
    public NDArray Squeeze() => np.squeeze(_arrWithSingleton);

    [Benchmark(Description = "np.squeeze(a, axis=1)")]
    [BenchmarkCategory("Squeeze")]
    public NDArray Squeeze_Axis1() => np.squeeze(_arrWithSingleton, axis: 1);

    // ========================================================================
    // Swap / Move Axes
    // ========================================================================

    [Benchmark(Description = "np.swapaxes(a, 0, 1)")]
    [BenchmarkCategory("SwapAxes")]
    public NDArray SwapAxes_01() => np.swapaxes(_arr2D, 0, 1);

    [Benchmark(Description = "np.swapaxes(a3D, 0, 2)")]
    [BenchmarkCategory("SwapAxes")]
    public NDArray SwapAxes_3D_02() => np.swapaxes(_arr3D, 0, 2);

    [Benchmark(Description = "np.moveaxis(a, 0, -1)")]
    [BenchmarkCategory("MoveAxis")]
    public NDArray MoveAxis_0_to_Neg1() => np.moveaxis(_arr3D, 0, -1);

    [Benchmark(Description = "np.rollaxis(a, 2)")]
    [BenchmarkCategory("RollAxis")]
    public NDArray RollAxis() => np.rollaxis(_arr3D, 2);
}
