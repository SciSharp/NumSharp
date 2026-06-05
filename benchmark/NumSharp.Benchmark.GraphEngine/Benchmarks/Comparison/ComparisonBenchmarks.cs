using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.GraphEngine.Infrastructure;

namespace NumSharp.Benchmark.GraphEngine.Benchmarks.Comparison;

/// <summary>
/// Element-wise comparison operators (==, !=, &lt;, &gt;, &lt;=, &gt;=). Each returns a
/// boolean array. Mirrors NumPy's comparison ufuncs.
/// </summary>
[BenchmarkCategory("Comparison")]
public class ComparisonBenchmarks : TypedBenchmarkBase
{
    private NDArray _a = null!;
    private NDArray _b = null!;

    [Params(ArraySizeSource.Small, ArraySizeSource.Medium, ArraySizeSource.Large)]
    public override int N { get; set; }

    [ParamsSource(nameof(Types))]
    public new NPTypeCode DType { get; set; }

    public static IEnumerable<NPTypeCode> Types => TypeParameterSource.CommonTypes;

    [GlobalSetup]
    public void Setup()
    {
        _a = CreateRandomArray(N, DType);
        _b = CreateRandomArray(N, DType, seed: 43);
    }

    [GlobalCleanup]
    public void Cleanup() { _a = null!; _b = null!; GC.Collect(); }

    [Benchmark(Description = "a == b")] public NDArray Equal() => _a == _b;
    [Benchmark(Description = "a != b")] public NDArray NotEqual() => _a != _b;
    [Benchmark(Description = "a < b")] public NDArray Less() => _a < _b;
    [Benchmark(Description = "a > b")] public NDArray Greater() => _a > _b;
    [Benchmark(Description = "a <= b")] public NDArray LessEqual() => _a <= _b;
    [Benchmark(Description = "a >= b")] public NDArray GreaterEqual() => _a >= _b;
}
