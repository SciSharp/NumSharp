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

    [Params(ArraySizeSource.Small, ArraySizeSource.Medium)]
    public override int N { get; set; }

    [ParamsSource(nameof(Types))]
    public new NPTypeCode DType { get; set; }

    public static IEnumerable<NPTypeCode> Types => new[] { NPTypeCode.Complex };

    [GlobalSetup]
    public void Setup()
    {
        np.random.seed(Seed);
        _vector = np.random.rand(N).astype(np.complex128);
        int side = (int)Math.Sqrt(N);
        _matrix = np.random.rand(side * side).reshape(side, side).astype(np.complex128);
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

    [Params(ArraySizeSource.Small, ArraySizeSource.Medium)]
    public override int N { get; set; }

    [ParamsSource(nameof(Types))]
    public new NPTypeCode DType { get; set; }

    public static IEnumerable<NPTypeCode> Types => new[] { NPTypeCode.Double };

    [GlobalSetup]
    public void Setup()
    {
        np.random.seed(Seed);
        _vector = np.random.rand(N);
        _side = (int)Math.Sqrt(N);
        _matrix = np.random.rand(_side * _side).reshape(_side, _side);
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

    [Benchmark(Description = "np.fft.rfft(a)")] public NDArray RFft() => np.fft.rfft(_vector);
    [Benchmark(Description = "np.fft.irfft(a)")] public NDArray IRFft() => np.fft.irfft(_spectrum, n: N);
    [Benchmark(Description = "np.fft.hfft(a)")] public NDArray HFft() => np.fft.hfft(_spectrum, n: N);
    [Benchmark(Description = "np.fft.ihfft(a)")] public NDArray IHFft() => np.fft.ihfft(_vector);
    [Benchmark(Description = "np.fft.rfft2(a)")] public NDArray RFft2() => np.fft.rfft2(_matrix);
    [Benchmark(Description = "np.fft.irfft2(a)")] public NDArray IRFft2() => np.fft.irfft2(_matrixSpectrum, s: new[] { _side, _side });
    [Benchmark(Description = "np.fft.rfftn(a)")] public NDArray RFftN() => np.fft.rfftn(_matrix);
    [Benchmark(Description = "np.fft.irfftn(a)")] public NDArray IRFftN() => np.fft.irfftn(_matrixSpectrum, s: new[] { _side, _side });
}

[BenchmarkCategory("Fourier", "Helpers")]
public class FftHelperBenchmarks : BenchmarkBase
{
    [Params(ArraySizeSource.Small, ArraySizeSource.Medium)]
    public override int N { get; set; }

    [Benchmark(Description = "np.fft.fftfreq(n)")] public NDArray FftFreq() => np.fft.fftfreq(N);
    [Benchmark(Description = "np.fft.rfftfreq(n)")] public NDArray RFftFreq() => np.fft.rfftfreq(N);
}
