using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.GraphEngine.Infrastructure;

namespace NumSharp.Benchmark.GraphEngine.Benchmarks.Reduction;

/// <summary>
/// Benchmarks for variance and standard deviation.
/// </summary>
[BenchmarkCategory("Reduction", "VarStd")]
public class VarStdBenchmarks : TypedBenchmarkBase
{
    private NDArray _a1D = null!;
    private NDArray _a2D = null!;

    [Params(ArraySizeSource.Small, ArraySizeSource.Medium, ArraySizeSource.Large)]
    public override int N { get; set; }

    // Variance/StdDev produce floating-point
    [ParamsSource(nameof(Types))]
    public new NPTypeCode DType { get; set; }

    public static IEnumerable<NPTypeCode> Types => TypeParameterSource.TranscendentalTypes;

    [GlobalSetup]
    public void Setup()
    {
        _a1D = CreateRandomArray(N, DType);

        var rows = (int)Math.Sqrt(N);
        var cols = N / rows;
        _a2D = CreateRandomArray(rows * cols, DType).reshape(rows, cols);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _a1D = null!;
        _a2D = null!;
        GC.Collect();
    }

    // ========================================================================
    // Variance
    // ========================================================================

    [Benchmark(Description = "np.var(a) [full]")]
    [BenchmarkCategory("Variance")]
    public NDArray Var_Full() => np.var(_a1D);

    [Benchmark(Description = "a.var() [method]")]
    [BenchmarkCategory("Variance")]
    public NDArray Var_Method() => _a1D.var();

    [Benchmark(Description = "np.var(a, axis=0)")]
    [BenchmarkCategory("Variance")]
    public NDArray Var_Axis0() => np.var(_a2D, axis: 0);

    // ========================================================================
    // Standard Deviation
    // ========================================================================

    [Benchmark(Description = "np.std(a) [full]")]
    [BenchmarkCategory("StdDev")]
    public NDArray Std_Full() => np.std(_a1D);

    [Benchmark(Description = "a.std() [method]")]
    [BenchmarkCategory("StdDev")]
    public NDArray Std_Method() => _a1D.std();

    [Benchmark(Description = "np.std(a, axis=0)")]
    [BenchmarkCategory("StdDev")]
    public NDArray Std_Axis0() => np.std(_a2D, axis: 0);
}
