using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.CSharp.Infrastructure;

namespace NumSharp.Benchmark.CSharp.Benchmarks.Fourier;

[BenchmarkCategory("Fourier", "Complex")]
public class ComplexFftBenchmarks : TypedBenchmarkBase
{
    private NDArray _vector = null!;
    private NDArray _spectrum = null!;
    private NDArray _matrix = null!;
    private NDArray _matrixSpectrum = null!;
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
        _vector = CreateRandomArray(WorkN, DType);
        int side = (int)Math.Sqrt(WorkN);
        _matrix = CreateRandomArray2D(side, side, DType);
        _spectrum = np.fft.fft(_vector);
        _matrixSpectrum = np.fft.fft2(_matrix);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _vector = null!;
        _spectrum = null!;
        _matrix = null!;
        _matrixSpectrum = null!;
        GC.Collect();
    }

    [Benchmark(Description = "np.fft.fft(a)")] public NDArray Fft() => np.fft.fft(_vector);
    [Benchmark(Description = "np.fft.ifft(a)")] public NDArray IFft() => np.fft.ifft(_spectrum);
    [Benchmark(Description = "np.fft.fft2(a)")] public NDArray Fft2() => np.fft.fft2(_matrix);
    [Benchmark(Description = "np.fft.ifft2(a)")] public NDArray IFft2() => np.fft.ifft2(_matrixSpectrum);
    [Benchmark(Description = "np.fft.fftn(a)")] public NDArray FftN() => np.fft.fftn(_matrix);
    [Benchmark(Description = "np.fft.ifftn(a)")] public NDArray IFftN() => np.fft.ifftn(_matrixSpectrum);
    [Benchmark(Description = "np.fft.fftshift(a)")] public NDArray FftShift() => np.fft.fftshift(_vector);
    [Benchmark(Description = "np.fft.ifftshift(a)")] public NDArray IFftShift() => np.fft.ifftshift(_vector);
}

[BenchmarkCategory("Fourier", "Real")]
public class RealFftBenchmarks : TypedBenchmarkBase
{
    private NDArray _vector = null!;
    private NDArray _spectrum = null!;
    private NDArray _matrix = null!;
    private NDArray _matrixSpectrum = null!;
    private int _side;
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
        _vector = CreateRandomArray(WorkN, DType);
        _side = (int)Math.Sqrt(WorkN);
        _matrix = CreateRandomArray2D(_side, _side, DType);
        if (DType == NPTypeCode.Complex)
        {
            // The real-transform family intentionally rejects complex-domain inputs.
            _spectrum = null!;
            _matrixSpectrum = null!;
            return;
        }
        _spectrum = np.fft.rfft(_vector);
        _matrixSpectrum = np.fft.rfft2(_matrix);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _vector = null!;
        _spectrum = null!;
        _matrix = null!;
        _matrixSpectrum = null!;
        GC.Collect();
    }

    [Benchmark(Description = "np.fft.rfft(a)")] public object RFft() => RealInput(() => np.fft.rfft(_vector));
    [Benchmark(Description = "np.fft.irfft(a)")] public object IRFft() => RealInput(() => np.fft.irfft(_spectrum, n: WorkN));
    [Benchmark(Description = "np.fft.hfft(a)")] public object HFft() => RealInput(() => np.fft.hfft(_spectrum, n: WorkN));
    [Benchmark(Description = "np.fft.ihfft(a)")] public object IHFft() => RealInput(() => np.fft.ihfft(_vector));
    [Benchmark(Description = "np.fft.rfft2(a)")] public object RFft2() => RealInput(() => np.fft.rfft2(_matrix));
    [Benchmark(Description = "np.fft.irfft2(a)")] public object IRFft2() => RealInput(() => np.fft.irfft2(_matrixSpectrum, s: new[] { _side, _side }));
    [Benchmark(Description = "np.fft.rfftn(a)")] public object RFftN() => RealInput(() => np.fft.rfftn(_matrix));
    [Benchmark(Description = "np.fft.irfftn(a)")] public object IRFftN() => RealInput(() => np.fft.irfftn(_matrixSpectrum, s: new[] { _side, _side }));

    private object RealInput(Func<NDArray> operation) =>
        DType == NPTypeCode.Complex ? RecordUnsafeUnsupportedDtype() : operation();
}

[BenchmarkCategory("Fourier", "Helpers")]
public class FftHelperBenchmarks : OfficialBenchmarkBase
{
    [Params(ArraySizeSource.Small, ArraySizeSource.Medium, ArraySizeSource.Large)]
    public override int N { get; set; }

    private int WorkN => ArraySizeSource.ResolveMemoryHeavyWorkload(N);

    [Benchmark(Description = "np.fft.fftfreq(n)")] public NDArray FftFreq() => np.fft.fftfreq(WorkN);
    [Benchmark(Description = "np.fft.rfftfreq(n)")] public NDArray RFftFreq() => np.fft.rfftfreq(WorkN);
}
