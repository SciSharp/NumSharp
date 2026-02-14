using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.GraphEngine.Infrastructure;

namespace NumSharp.Benchmark.GraphEngine.Benchmarks.Unary;

/// <summary>
/// Benchmarks for trigonometric functions.
/// </summary>
[BenchmarkCategory("Unary", "Trig")]
public class TrigBenchmarks : TypedBenchmarkBase
{
    private NDArray _angles = null!;

    [Params(ArraySizeSource.Small, ArraySizeSource.Medium, ArraySizeSource.Large)]
    public override int N { get; set; }

    [ParamsSource(nameof(Types))]
    public new NPTypeCode DType { get; set; }

    public static IEnumerable<NPTypeCode> Types => TypeParameterSource.TranscendentalTypes;

    [GlobalSetup]
    public void Setup()
    {
        np.random.seed(Seed);
        // Angles between -2π and 2π
        _angles = ((np.random.rand(N) * 4 - 2) * Math.PI).astype(DType);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _angles = null!;
        GC.Collect();
    }

    [Benchmark(Description = "np.sin(a)")]
    [BenchmarkCategory("Basic")]
    public NDArray Sin() => np.sin(_angles);

    [Benchmark(Description = "np.cos(a)")]
    [BenchmarkCategory("Basic")]
    public NDArray Cos() => np.cos(_angles);

    [Benchmark(Description = "np.tan(a)")]
    [BenchmarkCategory("Basic")]
    public NDArray Tan() => np.tan(_angles);
}
