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
    public static IEnumerable<NPTypeCode> Types => TypeParameterSource.AllNumericTypes;
    private bool SupportsBitwise => DType is
        NPTypeCode.Boolean or NPTypeCode.Byte or NPTypeCode.SByte or NPTypeCode.Int16 or
        NPTypeCode.UInt16 or NPTypeCode.Int32 or NPTypeCode.UInt32 or NPTypeCode.Int64 or
        NPTypeCode.UInt64 or NPTypeCode.Char;

    [GlobalSetup]
    public void Setup()
    {
        _a = CreateRandomArray(N, DType);
        _b = CreateRandomArray(N, DType, seed: 43);
    }

    [GlobalCleanup]
    public void Cleanup() { _a = null!; _b = null!; GC.Collect(); }

    [Benchmark(Description = "a & b")] public object And() => SupportsBitwise ? _a & _b : VerifyUnsupportedDtype(() => _a & _b);
    [Benchmark(Description = "a | b")] public object Or() => SupportsBitwise ? _a | _b : VerifyUnsupportedDtype(() => _a | _b);
    [Benchmark(Description = "a ^ b")] public object Xor() => SupportsBitwise ? _a ^ _b : VerifyUnsupportedDtype(() => _a ^ _b);
    [Benchmark(Description = "np.invert(a)")] public object Invert() => SupportsBitwise ? np.invert(_a) : VerifyUnsupportedDtype(() => np.invert(_a));
    // The shifts on a (bool, int-scalar) pair widen to a FRESH int64 result (8x the input; 80 MB at
    // the 10M tier), and right_shift(bool, 2) short-circuits to LAZY zeros (shift >= bit width), so
    // its pilot is ~1.5us size-invariant. BenchmarkDotNet then batches thousands of invocations per
    // iteration, each lazily committing 80 MB that only finalizers reclaim — commit-OOMing the 10M
    // case into "Statistics": null (it crashed in two consecutive official runs; the computed
    // left_shift merely allocates hard). Dispose per invocation to bound resident memory — the
    // LogicBenchmarks.IsComplex lazy-zeros precedent — matching the NumPy twin's
    // discard-each-result loop. Safe: the result is fresh and owned, never aliasing _a.
    [Benchmark(Description = "np.left_shift(a, 2)")]
    public object LeftShift()
    {
        if (!SupportsBitwise)
            return VerifyUnsupportedDtype(() => np.left_shift(_a, 2));
        using var result = np.left_shift(_a, 2);
        return DType;
    }

    [Benchmark(Description = "np.right_shift(a, 2)")]
    public object RightShift()
    {
        if (!SupportsBitwise)
            return VerifyUnsupportedDtype(() => np.right_shift(_a, 2));
        using var result = np.right_shift(_a, 2);
        return DType;
    }

    [Benchmark(Description = "np.bitwise_and(a, b)")] public object NpAnd() => SupportsBitwise ? np.bitwise_and(_a, _b) : VerifyUnsupportedDtype(() => np.bitwise_and(_a, _b));
    [Benchmark(Description = "np.bitwise_or(a, b)")] public object NpOr() => SupportsBitwise ? np.bitwise_or(_a, _b) : VerifyUnsupportedDtype(() => np.bitwise_or(_a, _b));
    [Benchmark(Description = "np.bitwise_xor(a, b)")] public object NpXor() => SupportsBitwise ? np.bitwise_xor(_a, _b) : VerifyUnsupportedDtype(() => np.bitwise_xor(_a, _b));
    [Benchmark(Description = "np.bitwise_not(a)")] public object NpNot() => SupportsBitwise ? np.bitwise_not(_a) : VerifyUnsupportedDtype(() => np.bitwise_not(_a));
}
