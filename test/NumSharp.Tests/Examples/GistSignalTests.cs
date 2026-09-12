using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Examples.Gist;
using NumSharp.Interop.OpenBLAS;

namespace NumSharp.Tests.Examples;

[TestClass]
public class GistSignalTests
{
    [TestMethod]
    public void PeakDetection_ReproducesPublishedDemonstration()
    {
        using var scope = NDScope.Open();
        var input = np.array(new double[] { 0, 0, 0, 2, 0, 0, 0, -2, 0, 0, 0, 2, 0, 0, 0, -2, 0 });
        var (maxima, minima) = PeakDetection.Detect(input, 0.3);
        maxima.shape.Should().Equal(2L, 2L);
        minima.shape.Should().Equal(2L, 2L);
        maxima.ToArray<double>().Should().Equal(3, 2, 11, 2);
        minima.ToArray<double>().Should().Equal(7, -2, 15, -2);
    }

    [TestMethod]
    public void PeakDetection_CustomCoordinatesPlateausAndTrailingPeak()
    {
        using var scope = NDScope.Open();
        var input = np.array(new double[] { 0, 2, 2, 0, -2, -2, 0, 3 });
        var coordinates = np.arange(8).astype(NPTypeCode.Double) * 0.25;
        var (maxima, minima) = PeakDetection.Detect(input, 0.5, coordinates);
        maxima.ToArray<double>().Should().Equal(0.25, 2);
        minima.ToArray<double>().Should().Equal(1, -2);
        input.item<double>(7).Should().Be(3, "detection does not mutate the input");
    }

    [TestMethod]
    public void PeakDetection_EmptyAndFlatHaveOriginalEmptyShape()
    {
        using var scope = NDScope.Open();
        foreach (var input in new[] { np.array(Array.Empty<double>()), np.zeros(new Shape(8), NPTypeCode.Double) })
        {
            var (maxima, minima) = PeakDetection.Detect(input, 0.3);
            maxima.shape.Should().Equal(0L);
            minima.shape.Should().Equal(0L);
        }
    }

    [TestMethod]
    public void PeakDetection_RejectsInvalidThresholdAndCoordinates()
    {
        using var scope = NDScope.Open();
        var input = np.arange(5);
        foreach (double delta in new[] { 0.0, -1.0, double.NaN, double.PositiveInfinity })
            new Action(() => PeakDetection.Detect(input, delta)).Should().Throw<ArgumentOutOfRangeException>();
        new Action(() => PeakDetection.Detect(input, 1, np.arange(3))).Should().Throw<ArgumentException>();
    }

    [TestMethod]
    public void Frequency_CrossingsAndAutocorrelationRecoverExactPeriod()
    {
        using var scope = NDScope.Open();
        var samples = new double[32];
        double[] period = { 0, 1, 0, -1 };
        for (int i = 0; i < samples.Length; i++) samples[i] = period[i % 4];
        var signal = np.array(samples);
        FrequencyEstimation.FromCrossings(signal, 32).Should().Be(8);
        FrequencyEstimation.FromAutocorrelation(signal, 32).Should().Be(8);
    }

    [TestMethod]
    public void Frequency_FftAndHpsRecoverHarmonicFundamental()
    {
        using var scope = NDScope.Open();
        var time = np.arange(1024).astype(NPTypeCode.Double) / 8192.0;
        var signal = np.zeros(new Shape(1024), NPTypeCode.Double);
        for (int h = 1; h <= 7; h++) signal = signal + np.sin(2 * System.Math.PI * 384 * h * time) / h;
        FrequencyEstimation.FromFft(signal, 8192).Should().BeApproximately(384, 0.01);
        FrequencyEstimation.FromHarmonicProductSpectrum(signal, 8192).Should().BeApproximately(383.99534699242423, 1e-9);
        FrequencyEstimation.LegacyHarmonicProductPasses(signal, 8192).size.Should().Be(6);
    }

    [TestMethod]
    [DoNotParallelize]
    public void Frequency_ParabolicAndPolynomialCompanionAreFunctional()
    {
        using var scope = NDScope.Open();
        var values = np.array(new double[] { 2, 3, 1, 6, 4, 2, 3, 1 });
        var vertex = FrequencyEstimation.Parabolic(values, 3);
        vertex.Position.Should().Be(3.2142857142857144);
        vertex.Value.Should().Be(6.160714285714286);
        WithLapack(() =>
        {
            var fit = FrequencyEstimation.ParabolicPolyfit(values, 3, 5);
            fit.Position.Should().BeApproximately(3.1, 1e-12);
            fit.Value.Should().BeApproximately(4.205, 1e-12);
        });
    }

    [TestMethod]
    public void Frequency_WindowHasCorrectSymmetryAndEndpoints()
    {
        using var scope = NDScope.Open();
        var window = FrequencyEstimation.BlackmanHarris(9);
        window.item<double>(4).Should().Be(1);
        window.item<double>(0).Should().BeApproximately(0.00006, 1e-16);
        window.ToArray<double>().Should().Equal(window["::-1"].ToArray<double>());
        FrequencyEstimation.BlackmanHarris(0).size.Should().Be(0);
        FrequencyEstimation.BlackmanHarris(1).item<double>().Should().Be(1);
    }

    [TestMethod]
    public void Frequency_UndefinedEstimatesAreNaNAndInvalidArgumentsThrow()
    {
        using var scope = NDScope.Open();
        var flat = np.zeros(new Shape(32), NPTypeCode.Double);
        double.IsNaN(FrequencyEstimation.FromCrossings(flat, 32)).Should().BeTrue();
        double.IsNaN(FrequencyEstimation.FromFft(flat, 32)).Should().BeTrue();
        double.IsNaN(FrequencyEstimation.FromAutocorrelation(flat, 32)).Should().BeTrue();
        double.IsNaN(FrequencyEstimation.FromHarmonicProductSpectrum(flat, 32)).Should().BeTrue();
        new Action(() => FrequencyEstimation.FromFft(flat, 0)).Should().Throw<ArgumentOutOfRangeException>();
        new Action(() => FrequencyEstimation.FromFft(np.array(new[] { 0.0, double.NaN }), 32)).Should().Throw<ArgumentException>();
        new Action(() => FrequencyEstimation.ParabolicPolyfit(flat, 3, 4)).Should().Throw<ArgumentOutOfRangeException>();
        new Action(() => FrequencyEstimation.ParabolicPolyfit(flat, 0, 5)).Should().Throw<ArgumentOutOfRangeException>();
        new Action(() => FrequencyEstimation.FromHarmonicProductSpectrum(flat, 32, 1)).Should().Throw<ArgumentOutOfRangeException>();
    }

    [TestMethod]
    [DoNotParallelize]
    public void SignalDemosRunWithoutExternalApplicationDependencies()
    {
        WithLapack(() =>
        {
            FrequencyEstimation.Demo();
            PeakDetection.Demo();
        });
    }

    private static void WithLapack(Action action)
    {
        // This test assembly intentionally suppresses backend autoinstall. The
        // example's optional polynomial fit needs LAPACK; enable it explicitly,
        // fail if unavailable, and restore the exact prior engine binding.
        using var probe = np.array(0.0);
        var engine = probe.TensorEngine;
        var previousBackend = engine.Blas;
        try
        {
            OpenBlasEngine.Enable(threads: 1);
            Assert.IsTrue(OpenBlasEngine.LapackAvailable, "The polynomial companion requires the bundled LAPACK backend.");
            action();
        }
        finally
        {
            engine.Blas = previousBackend;
        }
    }
}
