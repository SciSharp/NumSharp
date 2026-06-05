using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.GraphEngine.Infrastructure;

namespace NumSharp.Benchmark.GraphEngine.Benchmarks.Bitwise;

/// <summary>
/// Bitwise operations on integer arrays: and / or / xor / invert and the bit shifts.
/// Mirrors NumPy's np.bitwise_and/or/xor, np.invert, np.left_shift, np.right_shift.
/// </summary>
[BenchmarkCategory("Bitwise")]
public class BitwiseBenchmarks : TypedBenchmarkBase
{
    private NDArray _a = null!;
    private NDArray _b = null!;

    [Params(ArraySizeSource.Small, ArraySizeSource.Medium, ArraySizeSource.Large)]
    public override int N { get; set; }

    [ParamsSource(nameof(Types))]
    public new NPTypeCode DType { get; set; }

    // Bitwise ops only make sense for integer dtypes.
    public static IEnumerable<NPTypeCode> Types => TypeParameterSource.IntegerTypes;

    [GlobalSetup]
    public void Setup()
    {
        _a = CreateRandomArray(N, DType);
        _b = CreateRandomArray(N, DType, seed: 43);
    }

    [GlobalCleanup]
    public void Cleanup() { _a = null!; _b = null!; GC.Collect(); }

    [Benchmark(Description = "a & b")] public NDArray And() => _a & _b;
    [Benchmark(Description = "a | b")] public NDArray Or() => _a | _b;
    [Benchmark(Description = "a ^ b")] public NDArray Xor() => _a ^ _b;
    [Benchmark(Description = "np.invert(a)")] public NDArray Invert() => np.invert(_a);
    [Benchmark(Description = "np.left_shift(a, 2)")] public NDArray LeftShift() => np.left_shift(_a, 2);
    [Benchmark(Description = "np.right_shift(a, 2)")] public NDArray RightShift() => np.right_shift(_a, 2);
}
