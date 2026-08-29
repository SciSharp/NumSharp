using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.CSharp.Infrastructure;

namespace NumSharp.Benchmark.CSharp.Benchmarks.Comparison;

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

    public static IEnumerable<NPTypeCode> Types => TypeParameterSource.AllNumericTypes;

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

    [Benchmark(Description = "np.equal(a, b)")] public NDArray NpEqual() => np.equal(_a, _b);
    [Benchmark(Description = "np.not_equal(a, b)")] public NDArray NpNotEqual() => np.not_equal(_a, _b);
    [Benchmark(Description = "np.less(a, b)")] public NDArray NpLess() => np.less(_a, _b);
    [Benchmark(Description = "np.greater(a, b)")] public NDArray NpGreater() => np.greater(_a, _b);
    [Benchmark(Description = "np.less_equal(a, b)")] public NDArray NpLessEqual() => np.less_equal(_a, _b);
    [Benchmark(Description = "np.greater_equal(a, b)")] public NDArray NpGreaterEqual() => np.greater_equal(_a, _b);
}
