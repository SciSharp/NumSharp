using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Examples.Gist;
using NumSharp.Interop.OpenBLAS;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Tests.Interop;

/// <summary>
/// Live NumPy/SciPy evaluation over the example inputs' zero-copy buffers.
/// Mathematical references below are independently written from the reviewed pinned
/// gists, with current SciPy window imports and the port's documented NaN guards.
/// The source scripts' audio-loading and plotting top-level code is never executed.
/// </summary>
[TestClass]
public class GistSignalLiveParityTests : InteropTestBase
{
    public TestContext TestContext { get; set; }

    private const string Reference = """
        from scipy.signal.windows import blackmanharris
        from scipy.signal import correlate
        def vertex(y, k):
            if k <= 0 or k >= len(y)-1 or not np.all(np.isfinite(y[k-1:k+2])):
                return (np.nan, np.nan)
            a, b, c = y[k-1:k+2]
            d = a - 2*b + c
            if d == 0: return (np.nan, np.nan)
            p = .5*(a-c)/d + k
            return p, b - .25*(a-c)*(p-k)
        def crossing(s, fs):
            i = np.flatnonzero((s[1:] >= 0) & (s[:-1] < 0))
            if len(i) < 2: return np.nan
            t = np.array([j-s[j]/(s[j+1]-s[j]) for j in i])
            return fs / np.mean(np.diff(t))
        def fft_frequency(s, fs):
            m = abs(np.fft.rfft(s*blackmanharris(len(s))))
            return fs*vertex(np.log(m), np.argmax(m))[0]/len(s)
        def autocorrelation_frequency(s, fs):
            # SciPy 'auto' uses its direct method for these small fixtures.
            a = correlate(s, s, mode='full')
            a = a[len(a)//2:]
            rising = np.flatnonzero(np.diff(a) > 0)
            if len(rising) == 0: return np.nan
            k = rising[0] + np.argmax(a[rising[0]:])
            return fs/vertex(a, k)[0]
        def hps_frequency(s, fs, harmonics=7):
            m = abs(np.fft.rfft(s*blackmanharris(len(s))))
            n = len(m)//harmonics
            if n < 3: return np.nan
            p = m[:n].copy()
            for h in range(2, harmonics+1): p *= m[::h][:n]
            k = 1 + np.argmax(p[1:])
            return fs*vertex(np.log(p), k)[0]/len(s)
        def legacy_hps(s, fs):
            c = abs(np.fft.rfft(s*blackmanharris(len(s))))
            estimates = []
            for h in range(2, 8):
                decimated = c[::h].copy()
                c = c[:len(decimated)]
                estimates.append(fs*vertex(abs(c), np.argmax(abs(c)))[0]/len(s))
                c *= decimated
            return np.array(estimates)
        def polynomial_vertex(y, k, n):
            x = np.arange(k-n//2, k+n//2+1, dtype=np.float64)
            a,b,c = np.polyfit(x, y[k-n//2:k+n//2+1], 2)
            p = -.5*b/a
            return np.array([p, a*p**2+b*p+c])
        def detect_peaks(y, delta, x):
            hi = -np.inf
            lo = np.inf
            hx = lx = np.nan
            waiting = True
            maxima, minima = [], []
            for value, coordinate in zip(y, x):
                if value > hi: hi, hx = value, coordinate
                if value < lo: lo, lx = value, coordinate
                if waiting:
                    if value < hi-delta:
                        maxima.append((hx, hi))
                        lo,lx,waiting = value,coordinate,False
                elif value > lo+delta:
                    minima.append((lx,lo))
                    hi,hx,waiting = value,coordinate,True
            return np.asarray(maxima, dtype=np.float64), np.asarray(minima, dtype=np.float64)
        """;

    /// <summary>
    /// PeakDetection vs the reference <c>detect_peaks</c>, byte-exact over views, ties, NaNs, empty input
    /// and custom coordinates. The shared <see cref="Reference"/> imports SciPy, so this is inconclusive
    /// where SciPy is not installed (CI's interop job installs only NumPy).
    /// </summary>
    [TestMethod]
    public void Peaks_ByteExactIncludingViewsTiesNaNsEmptyAndCustomCoordinates()
    {
        // Checked BEFORE any ExportTo: a later failure would leave the exports pinned, and the leak
        // then cascades into every interop test that asserts an absolute LiveExports count.
        SkipUnless("scipy");
        using var scope = NDScope.Open();
        var parent = np.array(new double[] { 0, 99, 2, 99, 2, 99, 0, 99, -2, 99, double.NaN, 99, 0, 99, 3, 99 });
        foreach (var values in new[] { parent["::2"], parent["::2"]["::-1"], np.zeros(new Shape(8), NPTypeCode.Double), np.array(Array.Empty<double>()) })
        {
            var coordinates = np.arange(values.size).astype(NPTypeCode.Double) * 0.25;
            var (maxima, minima) = PeakDetection.Detect(values, 0.5, coordinates);
            ExportTo("s", values);
            ExportTo("x", coordinates);
            PyExec(Reference + "\nhi, lo = detect_peaks(s, .5, x)");
            using (Gil())
            {
                using PyObject hi = Scope.Eval("hi");
                using PyObject lo = Scope.Eval("lo");
                maxima.shape.Should().Equal(hi.shape_dims());
                minima.shape.Should().Equal(lo.shape_dims());
                hi.dtype_str().Should().Be("<f8");
                lo.dtype_str().Should().Be("<f8");
                ByteContract.AssertSameBytes(maxima, hi, "peak positions/values are copied unchanged");
                ByteContract.AssertSameBytes(minima, lo, "peak positions/values are copied unchanged");
            }
        }
    }

    /// <summary>
    /// Zero-crossing frequency estimates vs the reference, byte-exact on non-integer crossings and
    /// reversed views. Inconclusive without SciPy (the shared <see cref="Reference"/> imports it).
    /// </summary>
    [TestMethod]
    public void Crossings_ByteExactOnNonIntegerCrossingsAndReversedViews()
    {
        SkipUnless("scipy");   // before any export — see Peaks_ByteExactIncludingViewsTiesNaNsEmptyAndCustomCoordinates
        using var scope = NDScope.Open();
        var source = np.array(new double[] { -2, 1, 3, -1, -3, 2, 4, -4, -2, 5, 2, -3, -1, 3, 2, -5 });
        foreach (var signal in new[] { source, source["::-1"] })
        {
            var result = np.array(new[] { FrequencyEstimation.FromCrossings(signal, 48000) });
            ExportTo("s", signal);
            PyExec(Reference);
            using (Gil())
            {
                using PyObject expected = Scope.Eval("np.array([crossing(s, 48000)])");
                ByteContract.AssertSameBytes(result, expected, "linear crossing interpolation and mean");
            }
        }
    }

    /// <summary>
    /// The three-point parabolic vertex vs the reference <c>vertex</c>, byte-exact. Inconclusive without
    /// SciPy (the shared <see cref="Reference"/> imports it).
    /// </summary>
    [TestMethod]
    public void QuadraticVertex_ByteExact()
    {
        SkipUnless("scipy");   // before any export — see Peaks_ByteExactIncludingViewsTiesNaNsEmptyAndCustomCoordinates
        using var scope = NDScope.Open();
        var values = np.array(new double[] { 2, 3, 1, 6, 4, 2, 3, 1 });
        var (x, y) = FrequencyEstimation.Parabolic(values, 3);
        var result = np.array(new[] { x, y });
        ExportTo("s", values);
        PyExec(Reference);
        using (Gil())
        {
            using PyObject expected = Scope.Eval("np.array(vertex(s, 3))");
            ByteContract.AssertSameBytes(result, expected);
        }
    }

    /// <summary>
    /// The polyfit-based vertex vs the reference <c>polynomial_vertex</c>, byte-exact on the pinned LAPACK.
    /// Inconclusive without SciPy (the shared <see cref="Reference"/> imports it) or without LAPACK.
    /// </summary>
    [TestMethod]
    public void PolynomialVertex_ByteExactWithPinnedLapack()
    {
        SkipUnless("scipy");   // before any export — see Peaks_ByteExactIncludingViewsTiesNaNsEmptyAndCustomCoordinates
        if (!OpenBlasEngine.LapackAvailable) Assert.Inconclusive("This exact polyfit gate requires the pinned LAPACK backend.");
        using var scope = NDScope.Open();
        var values = np.array(new double[] { 2, 3, 1, 6, 4, 2, 3, 1 });
        var (x, y) = FrequencyEstimation.ParabolicPolyfit(values, 3, 5);
        var result = np.array(new[] { x, y });
        ExportTo("s", values);
        PyExec(Reference);
        using (Gil())
        {
            using PyObject expected = Scope.Eval("polynomial_vertex(s, 3, 5)");
            ByteContract.AssertSameBytes(result, expected);
        }
    }

    /// <summary>
    /// The Blackman-Harris window vs live <c>scipy.signal.windows.blackmanharris</c>, byte-exact.
    /// Inconclusive where SciPy is not installed.
    /// </summary>
    [TestMethod]
    public void Window_ByteExactVersusLiveScipy()
    {
        SkipUnless("scipy");   // before any export — see Peaks_ByteExactIncludingViewsTiesNaNsEmptyAndCustomCoordinates
        using var scope = NDScope.Open();
        PyExec(Reference);
        foreach (int length in new[] { 0, 1, 9, 32, 255, 1024 })
        {
            var result = FrequencyEstimation.BlackmanHarris(length);
            using (Gil())
            {
                using PyObject expected = Scope.Eval($"blackmanharris({length})");
                AssertMeasured(result, expected, 0, $"Blackman-Harris N={length}");
            }
        }
    }

    /// <summary>
    /// The four frequency pipelines (crossings, FFT, autocorrelation, HPS) vs live NumPy/SciPy within their
    /// measured ULP budgets. Inconclusive where SciPy is not installed.
    /// </summary>
    [TestMethod]
    public void FrequencyPipelines_MeasuredUlpVersusLiveNumpyAndScipy()
    {
        SkipUnless("scipy");   // before any export — see Peaks_ByteExactIncludingViewsTiesNaNsEmptyAndCustomCoordinates
        using var scope = NDScope.Open();
        var samples = new double[1024];
        for (int i = 0; i < samples.Length; i++)
            for (int h = 1; h <= 7; h++) samples[i] += Math.Sin(2 * Math.PI * 384 * h * i / 8192.0) / h;
        var signal = np.array(samples);
        var result = np.array(new[] {
            FrequencyEstimation.FromFft(signal, 8192),
            FrequencyEstimation.FromAutocorrelation(signal, 8192),
            FrequencyEstimation.FromHarmonicProductSpectrum(signal, 8192)
        });
        ExportTo("s", signal);
        PyExec(Reference);
        using (Gil())
        {
            using PyObject expected = Scope.Eval("np.array([fft_frequency(s,8192), autocorrelation_frequency(s,8192), hps_frequency(s,8192)])");
            AssertMeasured(result, expected, 0, "FFT/autocorrelation/HPS Hz");
        }
    }

    /// <summary>
    /// Every printed pass of the legacy harmonic-product-spectrum estimator, and its undefined-peak cases,
    /// vs the reference <c>legacy_hps</c>. Inconclusive where SciPy is not installed.
    /// </summary>
    [TestMethod]
    public void LegacyHps_AllPrintedPassesAndUndefinedPeaksAreChecked()
    {
        SkipUnless("scipy");   // before any export — see Peaks_ByteExactIncludingViewsTiesNaNsEmptyAndCustomCoordinates
        using var scope = NDScope.Open();
        var samples = new double[16384];
        for (int i = 0; i < samples.Length; i++)
            for (int h = 1; h <= 7; h++) samples[i] += Math.Sin(2 * Math.PI * 32 * h * i / samples.Length) / h;
        var signal = np.array(samples);
        var result = FrequencyEstimation.LegacyHarmonicProductPasses(signal, 16384);
        ExportTo("s", signal);
        PyExec(Reference);
        using (Gil())
        {
            using PyObject expected = Scope.Eval("legacy_hps(s, 16384)");
            AssertMeasured(result, expected, 0, "original recursive HPS pass diagnostics");
        }
    }

    // Fail first on any unexpected dtype/shape/special value. Report exact matching
    // elements and the actual finite ULP maximum; do not replace a byte gate by allclose.
    private void AssertMeasured(NDArray result, PyObject expected, ulong maximumUlp, string label)
    {
        result.typecode.Should().Be(NPTypeCode.Double, label + " actual dtype");
        result.shape.Should().Equal(expected.shape_dims());
        expected.dtype_str().Should().Be("<f8");
        byte[] actualBytes = ByteContract.NsBytes(result), expectedBytes = expected.bytes_c();
        actualBytes.Length.Should().Be(expectedBytes.Length);
        ulong worst = 0;
        int exact = 0, nanPairs = 0;
        for (int i = 0; i < actualBytes.Length; i += 8)
        {
            ulong a = BitConverter.ToUInt64(actualBytes, i), b = BitConverter.ToUInt64(expectedBytes, i);
            if (a == b) { exact++; continue; }
            double x = BitConverter.ToDouble(actualBytes, i), y = BitConverter.ToDouble(expectedBytes, i);
            if (double.IsNaN(x) && double.IsNaN(y)) { nanPairs++; continue; }
            double.IsFinite(x).Should().BeTrue($"{label}: actual[{i/8}]={x:R}");
            double.IsFinite(y).Should().BeTrue($"{label}: expected[{i/8}]={y:R}");
            ulong oa = (a >> 63) != 0 ? ~a : a | (1UL << 63);
            ulong ob = (b >> 63) != 0 ? ~b : b | (1UL << 63);
            ulong distance = oa > ob ? oa - ob : ob - oa;
            worst = Math.Max(worst, distance);
        }
        TestContext.WriteLine($"{label}: {exact}/{actualBytes.Length/8} exact float64 bit patterns; {nanPairs} semantic NaN pairs; maximum finite ULP={worst} (budget {maximumUlp}).");
        worst.Should().BeLessThanOrEqualTo(maximumUlp, label);
        if (maximumUlp == 0 && nanPairs == 0) ByteContract.AssertSameBytes(result, expected, label);
    }
}
