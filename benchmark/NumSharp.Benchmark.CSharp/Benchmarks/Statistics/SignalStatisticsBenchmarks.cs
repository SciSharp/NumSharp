using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.CSharp.Infrastructure;

namespace NumSharp.Benchmark.CSharp.Benchmarks.Statistics;

/// <summary>Signal, difference, covariance, histogram, and small product compositions.</summary>
[BenchmarkCategory("Statistics", "Signal")]
public class SignalStatisticsBenchmarks : OfficialBenchmarkBase
{
    private NDArray _a = null!;
    private NDArray _b = null!;
    private NDArray _kernel = null!;
    private NDArray _vectors = null!;
    private NDArray _otherVectors = null!;
    private NDArray _kronA = null!;
    private NDArray _kronB = null!;
    private NDArray _matrix = null!;
    private NDArray _bins = null!;
    private int WorkN => ArraySizeSource.ResolveMemoryHeavyWorkload(N);

    [Params(ArraySizeSource.Small, ArraySizeSource.Medium, ArraySizeSource.Large)]
    public override int N { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        np.random.seed(Seed);
        _a = CreateRandomArray(WorkN, DType);
        _b = CreateRandomArray(WorkN, DType, seed: 43);
        _kernel = CreateRandomArray(31, DType, seed: 44);
        int vectorRows = WorkN / 3;
        _vectors = CreateRandomArray(vectorRows * 3, DType).reshape(vectorRows, 3);
        _otherVectors = CreateRandomArray(vectorRows * 3, DType, seed: 43).reshape(vectorRows, 3);
        int side = (int)Math.Sqrt(WorkN);
        _kronA = CreateRandomArray(side, DType);
        _kronB = CreateRandomArray(side, DType, seed: 43);
        _matrix = CreateRandomArray2D(side, side, DType);
        _bins = DType == NPTypeCode.Boolean
            ? np.array(new[] { false, true })
            : np.linspace(0, 100, 101).astype(DType);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _a = null!;
        _b = null!;
        _kernel = null!;
        _vectors = null!;
        _otherVectors = null!;
        _kronA = null!;
        _kronB = null!;
        _matrix = null!;
        _bins = null!;
        GC.Collect();
    }

    [Benchmark(Description = "np.convolve(a, kernel)")] public NDArray Convolve() => np.convolve(_a, _kernel, "same");
    [Benchmark(Description = "np.correlate(a, kernel)")] public NDArray Correlate() => np.correlate(_a, _kernel, "same");
    [Benchmark(Description = "np.diff(a)")] public NDArray Diff() => np.diff(_a);
    [Benchmark(Description = "np.ediff1d(a)")] public object EDiff1D() => DType != NPTypeCode.Boolean
        ? np.ediff1d(_a)
        : VerifyUnsupportedDtype(() => np.ediff1d(_a));
    [Benchmark(Description = "np.cross(a, b)")] public object Cross() => DType != NPTypeCode.Boolean
        ? np.cross(_vectors, _otherVectors)
        : VerifyUnsupportedDtype(() => np.cross(_vectors, _otherVectors));
    [Benchmark(Description = "np.inner(a, b)")] public object Inner() => DType != NPTypeCode.Char
        ? np.inner(_a, _b)
        : VerifyUnsupportedDtype(() => np.inner(_a, _b));
    [Benchmark(Description = "np.kron(a, b)")] public NDArray Kron() => np.kron(_kronA, _kronB);
    [Benchmark(Description = "np.trace(a)")] public NDArray Trace() => np.trace(_matrix);
    [Benchmark(Description = "np.cov(a, b)")] public NDArray Cov() => np.cov(_a, _b);
    [Benchmark(Description = "np.corrcoef(a, b)")] public NDArray CorrCoef() => np.corrcoef(_a, _b);
    [Benchmark(Description = "np.digitize(a, bins)")] public object Digitize() => DType != NPTypeCode.Complex
        ? np.digitize(_a, _bins)
        : VerifyUnsupportedDtype(() => np.digitize(_a, _bins));
}

/// <summary>Bincount uses a non-negative int32 input and reports that input dtype explicitly.</summary>
[BenchmarkCategory("Statistics", "Histogram")]
public class BincountBenchmarks : TypedBenchmarkBase
{
    private NDArray _values = null!;
    private int WorkN => ArraySizeSource.ResolveMemoryHeavyWorkload(N);

    [Params(ArraySizeSource.Small, ArraySizeSource.Medium, ArraySizeSource.Large)]
    public override int N { get; set; }

    [ParamsSource(nameof(Types))]
    public new NPTypeCode DType { get; set; }

    public static IEnumerable<NPTypeCode> Types => TypeParameterSource.AllNumericTypes;

    [GlobalSetup]
    public void Setup()
    {
        np.random.seed(Seed);
        // np.bincount requires NON-NEGATIVE integers. randint(0, WorkN/10) can exceed a narrow
        // signed dtype's positive max (SByte 127, Int16 32767); .astype(DType) would then wrap to
        // negative values and np.bincount would (correctly, matching NumPy) reject them, leaving the
        // cell with zero measurements. Cap the exclusive upper bound to what the dtype represents
        // non-negatively so every swept dtype gets valid input, while the wide dtypes (Int32+) keep
        // the NumPy twin's full WorkN/10 range for a comparable join.
        int upper = Math.Min(Math.Max(2, WorkN / 10), NonNegativeUpperBound(DType));
        _values = np.random.randint(0, upper, new Shape(WorkN)).astype(DType);
    }

    [GlobalCleanup]
    public void Cleanup() { _values = null!; GC.Collect(); }

    [Benchmark(Description = "np.bincount(a)")] public object Bincount() => SupportsBincount
        ? np.bincount(_values)
        : VerifyUnsupportedDtype(() => np.bincount(_values));

    private bool SupportsBincount => DType is
        NPTypeCode.Boolean or NPTypeCode.Byte or NPTypeCode.SByte or NPTypeCode.Int16 or
        NPTypeCode.UInt16 or NPTypeCode.Int32 or NPTypeCode.UInt32 or NPTypeCode.Int64 or
        NPTypeCode.UInt64 or NPTypeCode.Char;

    /// <summary>Exclusive randint upper bound that keeps every generated value non-negative and
    /// representable in <paramref name="dtype"/> after astype — bincount rejects negatives, and a
    /// value above the dtype's positive max wraps to a negative on the narrow signed types.</summary>
    private static int NonNegativeUpperBound(NPTypeCode dtype) => dtype switch
    {
        NPTypeCode.Boolean => 2,     // {0, 1}
        NPTypeCode.SByte   => 128,   // 0..127
        NPTypeCode.Byte    => 256,   // 0..255
        NPTypeCode.Int16   => 32768, // 0..32767
        NPTypeCode.UInt16  => 65536, // 0..65535
        NPTypeCode.Char    => 65536, // 0..65535 (uint16 proxy)
        _ => int.MaxValue,           // Int32/UInt32/Int64/UInt64: WorkN/10 (<=100000) always fits
    };
}
