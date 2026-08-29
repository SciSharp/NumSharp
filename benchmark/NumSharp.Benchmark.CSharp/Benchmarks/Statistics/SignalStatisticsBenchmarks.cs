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
        _values = np.random.randint(0, Math.Max(2, WorkN / 10), new Shape(WorkN)).astype(DType);
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
}
