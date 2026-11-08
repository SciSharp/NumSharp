using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Examples.Gist;

namespace NumSharp.Tests.Examples;

// Source-derived demonstrations and separately labelled synthetic experiments.
// Exact source locations and unrepeatable historical claims: examples/gist/coverage-signal.json.
[TestClass]
public class GistSignalSourceDemonstrationsTests
{
    [TestMethod]
    public void ParabolicMain_PlotMarkersHaveThePublishedInputGeometry()
    {
        using var scope = NDScope.Open();
        // parabolic.py:47-55: silver is the sampled maximum; blue is the fitted vertex.
        var values = np.array(new double[] { 2, 1, 4, 8, 11, 10, 7, 3, 1, 1 });
        long sampled = np.argmax(values);
        var interpolated = FrequencyEstimation.Parabolic(values, sampled);
        Assert.AreEqual(4L, sampled);
        Assert.AreEqual(11.0, values.item<double>(sampled));
        Assert.AreEqual(4.25, interpolated.Position);
        Assert.AreEqual(11.125, interpolated.Value);
        Assert.IsTrue(interpolated.Value > values.item<double>(sampled));
    }

    /// <summary>
    /// parabolic.md's three-point README example: <c>np.polyfit</c> over x=[2,3,4] must recover exactly
    /// the vertex the closed-form <see cref="FrequencyEstimation.Parabolic"/> computes. polyfit rides
    /// LAPACK <c>lstsq</c>, which Core ships no managed fallback for, so this is inconclusive on a host
    /// that cannot load the OpenBLAS LAPACK backend.
    /// </summary>
    [TestMethod]
    [DoNotParallelize]
    public void ParabolicReadme_ThreePointPolyfitMatchesTheSameQuadratic()
    {
        using var scope = NDScope.Open();
        // parabolic.md:15-27 uses points x=[2,3,4], y=[1,6,4], NOT a five-point fit.
        var values = np.array(new double[] { 2, 3, 1, 6, 4, 2, 3, 1 });
        var originalBackend = values.TensorEngine.Blas;
        try
        {
            ExampleBlasBackend.EnableOrInconclusive(requireLapack: true);
            var direct = FrequencyEstimation.Parabolic(values, 3);
            var fit = FrequencyEstimation.ParabolicPolyfit(values, 3, 3);
            Assert.AreEqual(45.0 / 14.0, direct.Position);
            Assert.AreEqual(direct.Position, fit.Position, 2e-12);
            Assert.AreEqual(direct.Value, fit.Value, 2e-12);
            // The old README prints a different final rounding for np.polyfit.
            // Contemporary bytes are checked against live NumPy, not that historical display.
        }
        finally { values.TensorEngine.Blas = originalBackend; }
    }

    [TestMethod]
    public void PeakMain_ScatterColumnsAreTheActualDemonstrationMarkers()
    {
        using var scope = NDScope.Open();
        // peakdetect.py:77-81; the source plots these columns independently.
        var values = np.array(new double[] { 0, 0, 0, 2, 0, 0, 0, -2, 0, 0, 0, 2, 0, 0, 0, -2, 0 });
        var (maxima, minima) = PeakDetection.Detect(values, .3);
        CollectionAssert.AreEqual(new[] { 3.0, 11.0 }, maxima[":,0"].ToArray<double>());
        CollectionAssert.AreEqual(new[] { 2.0, 2.0 }, maxima[":,1"].ToArray<double>());
        CollectionAssert.AreEqual(new[] { 7.0, 15.0 }, minima[":,0"].ToArray<double>());
        CollectionAssert.AreEqual(new[] { -2.0, -2.0 }, minima[":,1"].ToArray<double>());
    }

    [TestMethod]
    public void SyntheticCrossings_InterpolationImprovesThisSpecifiedSine()
    {
        using var scope = NDScope.Open();
        // A NEW reproducible experiment, not the unspecified recording behind
        // the source's 1000.185 / 1000.000129 Hz comments.
        const double sampleRate = 44100, frequency = 1000;
        var signal = Sine(2048, sampleRate, frequency, .4);
        var indices = np.nonzero((signal["1:"] >= 0.0) & (signal[":-1"] < 0.0))[0];
        double naive = sampleRate / np.mean(np.diff(indices)).item<double>();
        double interpolated = FrequencyEstimation.FromCrossings(signal, sampleRate);
        Assert.AreEqual(1000.2520161290323, naive);
        Assert.IsTrue(System.Math.Abs(interpolated - frequency) < .001);
        Assert.IsTrue(System.Math.Abs(interpolated - frequency) * 100 < System.Math.Abs(naive - frequency));
    }

    [TestMethod]
    public void SyntheticFft_InterpolationAndLongerRecordImproveThisSpecifiedSine()
    {
        using var scope = NDScope.Open();
        // Selected deterministic experiment; not a universal monotonic-accuracy assertion.
        var shortSignal = Sine(1024, 48000, 1000, .4);
        var longSignal = Sine(4096, 48000, 1000, .4);
        var magnitude = np.abs(np.fft.rfft(shortSignal * FrequencyEstimation.BlackmanHarris(1024)));
        double binOnly = 48000.0 * np.argmax(magnitude) / 1024;
        double shortEstimate = FrequencyEstimation.FromFft(shortSignal, 48000);
        double longEstimate = FrequencyEstimation.FromFft(longSignal, 48000);
        Assert.AreEqual(984.375, binOnly);
        Assert.IsTrue(System.Math.Abs(shortEstimate - 1000) < .15);
        Assert.IsTrue(System.Math.Abs(longEstimate - 1000) < .04);
        Assert.IsTrue(System.Math.Abs(longEstimate - 1000) < System.Math.Abs(shortEstimate - 1000));
        Assert.IsTrue(System.Math.Abs(shortEstimate - 1000) < System.Math.Abs(binOnly - 1000));
    }

    [DataTestMethod]
    [DataRow("sine")]
    [DataRow("square")]
    [DataRow("triangle")]
    public void SyntheticPeriodicWaveforms_CrossingsRecoverTheirKnownPeriod(string waveform)
    {
        using var scope = NDScope.Open();
        double[] period = waveform switch
        {
            "square" => new double[] { 1, 1, 1, 1, -1, -1, -1, -1 },
            "triangle" => new double[] { 0, .5, 1, .5, 0, -.5, -1, -.5 },
            _ => new double[8]
        };
        if (waveform == "sine")
            for (int i = 0; i < 8; i++) period[i] = System.Math.Sin(2 * System.Math.PI * i / 8);
        var samples = new double[512];
        for (int i = 0; i < samples.Length; i++) samples[i] = period[i % 8];
        var signal = np.array(samples);
        Assert.AreEqual(8.0, FrequencyEstimation.FromCrossings(signal, 64));
        // Specific noiseless, repeated periods; not a promise for arbitrary/noisy waveforms.
    }

    [TestMethod]
    public void SyntheticLogGaussianQuadratic_HasAnExactInterpolatedVertex()
    {
        using var scope = NDScope.Open();
        // parabolic.md:3 motivates an exact quadratic in log magnitude. This checks
        // that ALGEBRA, not a finite sampled FFT or the external Gaussian theorem.
        var coordinates = np.arange(8).astype(NPTypeCode.Double);
        var logMagnitude = 5 - 2 * np.square(coordinates - 3.25);
        var (position, value) = FrequencyEstimation.Parabolic(logMagnitude, np.argmax(logMagnitude));
        Assert.AreEqual(3.25, position);
        Assert.AreEqual(5.0, value);
    }

    [TestMethod]
    public void SyntheticZeroPadding_RefinesThisInterpolatedFftPeak()
    {
        using var scope = NDScope.Open();
        // parabolic.md:6-9 suggests a hybrid, but supplies no signal or padding factor.
        // This is an explicit NEW 8x-padding experiment using the port's interpolation.
        var signal = Sine(1024, 48000, 1000, .4);
        double unpadded = FrequencyEstimation.FromFft(signal, 48000);
        var magnitude = np.abs(np.fft.rfft(signal * FrequencyEstimation.BlackmanHarris(1024), n: 8192));
        var (position, _) = FrequencyEstimation.Parabolic(np.log(magnitude), np.argmax(magnitude));
        double padded = 48000 * position / 8192;
        Assert.IsTrue(System.Math.Abs(padded - 1000) < .001);
        Assert.IsTrue(System.Math.Abs(padded - 1000) * 100 < System.Math.Abs(unpadded - 1000));
    }

    [TestMethod]
    public void SyntheticMissingFundamental_ExposesPeakPickingLimits()
    {
        using var scope = NDScope.Open();
        // New example for readme.md:22,27,31,38. No 128-Hz component exists;
        // harmonics 2..7 still repeat every 1/128 second.
        var samples = new double[2048];
        for (int i = 0; i < samples.Length; i++)
            for (int h = 2; h <= 7; h++)
                samples[i] += System.Math.Sin(2 * System.Math.PI * 128 * h * (i / 8192.0)) / h;
        var signal = np.array(samples);
        double crossings = FrequencyEstimation.FromCrossings(signal, 8192);
        double fft = FrequencyEstimation.FromFft(signal, 8192);
        double autocorrelation = FrequencyEstimation.FromAutocorrelation(signal, 8192);
        double repairedHps = FrequencyEstimation.FromHarmonicProductSpectrum(signal, 8192);
        Assert.AreEqual(256, crossings, 1e-8);
        Assert.AreEqual(256, fft, 1e-5);
        Assert.AreEqual(128, autocorrelation, 1e-8);
        Assert.AreEqual(128, repairedHps, .01);
        // This is the explicitly repaired return-valued HPS, not an output claimed
        // by the original plot-only recursively truncated routine.
    }

    [TestMethod]
    public void SyntheticDemoHarmonics_AllFourPrintedEstimatesHaveHonestErrorBounds()
    {
        using var scope = NDScope.Open();
        var time = np.arange(1024).astype(NPTypeCode.Double) / 8192.0;
        var signal = np.zeros(new Shape(1024), NPTypeCode.Double);
        for (int h = 1; h <= 7; h++) signal = signal + np.sin(2 * System.Math.PI * 384 * h * time) / h;
        double crossings = FrequencyEstimation.FromCrossings(signal, 8192);
        double fft = FrequencyEstimation.FromFft(signal, 8192);
        double autocorrelation = FrequencyEstimation.FromAutocorrelation(signal, 8192);
        double hps = FrequencyEstimation.FromHarmonicProductSpectrum(signal, 8192);
        Assert.AreEqual(384.0170102059301, crossings, 1e-9);
        Assert.AreEqual(383.99999998553875, fft, 1e-9);
        Assert.AreEqual(384.5035347504078, autocorrelation, 1e-9);
        Assert.AreEqual(383.99534699242423, hps, 1e-9);
        Assert.IsTrue(autocorrelation - 384 > .5 && autocorrelation - 384 < .51,
            "Algorithm agreement is not estimation accuracy: this autocorrelation estimate is biased by about half a Hz.");
    }

    [TestMethod]
    public void SyntheticThreshold_RequiresStrictReversalAndCustomPositions()
    {
        using var scope = NDScope.Open();
        // Explicit fixture for the source's strict < and > conditions and X override.
        var values = np.array(new double[] { 0, 1, .5, 0, .5, 1 });
        var x = np.arange(1, 7).astype(NPTypeCode.Double);
        var (maxima, minima) = PeakDetection.Detect(values, .5, x);
        CollectionAssert.AreEqual(new[] { 2.0, 1.0 }, maxima.ToArray<double>());
        CollectionAssert.AreEqual(new[] { 4.0, 0.0 }, minima.ToArray<double>());
        // Supplying 1..N also reproduces the MATLAB companion's coordinate convention;
        // the Python port's default remains 0..N-1. No MATLAB engine is claimed here.
    }

    private static NDArray Sine(int length, double sampleRate, double frequency, double phase)
    {
        var values = new double[length];
        for (int i = 0; i < length; i++)
            values[i] = System.Math.Sin(2 * System.Math.PI * frequency * i / sampleRate + phase);
        return np.array(values);
    }
}
