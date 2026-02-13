using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.GraphEngine.Infrastructure;

namespace NumSharp.Benchmark.GraphEngine.Benchmarks.Manipulation;

/// <summary>
/// Benchmarks for stacking and concatenation operations.
/// </summary>
[BenchmarkCategory("Manipulation", "Stack")]
public class StackBenchmarks : BenchmarkBase
{
    private NDArray _arr1D_a = null!;
    private NDArray _arr1D_b = null!;
    private NDArray _arr1D_c = null!;
    private NDArray _arr2D_a = null!;
    private NDArray _arr2D_b = null!;

    [Params(ArraySizeSource.Small, ArraySizeSource.Medium)]
    public override int N { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        np.random.seed(Seed);
        _arr1D_a = np.random.rand(N) * 100;
        _arr1D_b = np.random.rand(N) * 100;
        _arr1D_c = np.random.rand(N) * 100;

        var rows = (int)Math.Sqrt(N);
        var cols = N / rows;
        _arr2D_a = np.random.rand(rows, cols) * 100;
        _arr2D_b = np.random.rand(rows, cols) * 100;
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _arr1D_a = null!;
        _arr1D_b = null!;
        _arr1D_c = null!;
        _arr2D_a = null!;
        _arr2D_b = null!;
        GC.Collect();
    }

    // ========================================================================
    // Concatenate
    // ========================================================================

    [Benchmark(Description = "np.concatenate([a, b])")]
    [BenchmarkCategory("Concatenate")]
    public NDArray Concatenate_2() => np.concatenate(new[] { _arr1D_a, _arr1D_b });

    [Benchmark(Description = "np.concatenate([a, b, c])")]
    [BenchmarkCategory("Concatenate")]
    public NDArray Concatenate_3() => np.concatenate(new[] { _arr1D_a, _arr1D_b, _arr1D_c });

    [Benchmark(Description = "np.concatenate([a, b], axis=0)")]
    [BenchmarkCategory("Concatenate")]
    public NDArray Concatenate_2D_Axis0() => np.concatenate(new[] { _arr2D_a, _arr2D_b }, axis: 0);

    [Benchmark(Description = "np.concatenate([a, b], axis=1)")]
    [BenchmarkCategory("Concatenate")]
    public NDArray Concatenate_2D_Axis1() => np.concatenate(new[] { _arr2D_a, _arr2D_b }, axis: 1);

    // ========================================================================
    // Stack
    // ========================================================================

    [Benchmark(Description = "np.stack([a, b])")]
    [BenchmarkCategory("Stack")]
    public NDArray Stack_2() => np.stack(new[] { _arr1D_a, _arr1D_b });

    [Benchmark(Description = "np.stack([a, b], axis=1)")]
    [BenchmarkCategory("Stack")]
    public NDArray Stack_2_Axis1() => np.stack(new[] { _arr1D_a, _arr1D_b }, axis: 1);

    // ========================================================================
    // HStack / VStack / DStack
    // ========================================================================

    [Benchmark(Description = "np.hstack([a, b])")]
    [BenchmarkCategory("HVDStack")]
    public NDArray HStack() => np.hstack(_arr1D_a, _arr1D_b);

    [Benchmark(Description = "np.vstack([a, b])")]
    [BenchmarkCategory("HVDStack")]
    public NDArray VStack() => np.vstack(_arr1D_a, _arr1D_b);

    [Benchmark(Description = "np.dstack([a, b])")]
    [BenchmarkCategory("HVDStack")]
    public NDArray DStack() => np.dstack(_arr2D_a, _arr2D_b);
}
