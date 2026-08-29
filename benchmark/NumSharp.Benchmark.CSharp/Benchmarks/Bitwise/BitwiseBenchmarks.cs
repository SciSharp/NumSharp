using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.CSharp.Infrastructure;

namespace NumSharp.Benchmark.CSharp.Benchmarks.Bitwise;

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
    // The shifts on a (bool, int-scalar) pair widen to a FRESH int64 result (8x the input; 80 MB at
    // the 10M tier), and right_shift(bool, 2) short-circuits to LAZY zeros (shift >= bit width), so
    // its pilot is ~1.5us size-invariant. BenchmarkDotNet then batches thousands of invocations per
    // iteration, each lazily committing 80 MB that only finalizers reclaim — commit-OOMing the 10M
    // case into "Statistics": null (it crashed in two consecutive official runs; the computed
    // left_shift merely allocates hard). Dispose per invocation to bound resident memory — the
    // LogicBenchmarks.IsComplex lazy-zeros precedent — matching the NumPy twin's
    // discard-each-result loop. Safe: the result is fresh and owned, never aliasing _a.
    [Benchmark(Description = "np.left_shift(a, 2)")] public void LeftShift() { using var _ = np.left_shift(_a, 2); }
    [Benchmark(Description = "np.right_shift(a, 2)")] public void RightShift() { using var _ = np.right_shift(_a, 2); }
    [Benchmark(Description = "np.bitwise_and(a, b)")] public NDArray NpAnd() => np.bitwise_and(_a, _b);
    [Benchmark(Description = "np.bitwise_or(a, b)")] public NDArray NpOr() => np.bitwise_or(_a, _b);
    [Benchmark(Description = "np.bitwise_xor(a, b)")] public NDArray NpXor() => np.bitwise_xor(_a, _b);
    [Benchmark(Description = "np.bitwise_not(a)")] public NDArray NpNot() => np.bitwise_not(_a);
}
