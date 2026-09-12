// Gist: https://gist.github.com/endolith/255291
// Pinned revision: f94aa1d3d63a8ec3c0e80f30c6f9f4e375ceea0f
// Source files: frequency_estimator.py and parabolic.py; author: endolith.
// No explicit license was present in these pinned files. This is an independent
// implementation of the numerical methods, not a transcription of their prose.
using System;

namespace NumSharp.Examples.Gist;

/// <summary>Four frequency estimators and the quadratic interpolation they share.</summary>
/// <remarks>
/// Inputs are finite, real, one-dimensional samples; computations use float64.
/// These are estimators, not general pitch trackers: harmonics, noise, DC and short
/// records can fool them. Undefined estimates return NaN rather than an invented Hz.
/// Audio decoding and plots are intentionally outside this numerical example.
/// </remarks>
public static class FrequencyEstimation
{
    public static double FromCrossings(NDArray signal, double sampleRate)
    {
        using var scope = NDScope.Open();
        var s = Samples(signal, sampleRate);
        if (s.size < 2) return double.NaN;
        var indices = np.nonzero((s["1:"] >= 0.0) & (s[":-1"] < 0.0))[0];
        if (indices.size < 2) return double.NaN;
        var crossings = np.empty(new Shape(indices.size), NPTypeCode.Double);
        for (long k = 0; k < indices.size; k++)
        {
            long i = indices.item<long>(k);
            double left = s.item<double>(i), right = s.item<double>(i + 1);
            crossings[k] = i - left / (right - left);
        }
        // The differences form a 1-D vector, so axis=0 is the same mathematical
        // reduction as NumPy's omitted axis. NumSharp's explicit-axis route uses
        // NumPy's pairwise fold; its flat fast reduction can differ by one ULP here.
        return sampleRate / np.mean(np.diff(crossings), axis: 0).item<double>();
    }

    public static double FromFft(NDArray signal, double sampleRate)
    {
        using var scope = NDScope.Open();
        var s = Samples(signal, sampleRate);
        if (s.size < 3) return double.NaN;
        var magnitude = np.abs(np.fft.rfft(s * BlackmanHarris(checked((int)s.size))));
        long peak = np.argmax(magnitude);
        var (position, _) = Parabolic(np.log(magnitude), peak);
        return sampleRate * position / s.size;
    }

    public static double FromAutocorrelation(NDArray signal, double sampleRate)
    {
        using var scope = NDScope.Open();
        var s = Samples(signal, sampleRate);
        if (s.size < 3) return double.NaN;
        var full = np.correlate(s, s, "full");
        var correlation = full[$"{full.size / 2}:"];
        var rising = np.nonzero(np.diff(correlation) > 0.0)[0];
        if (rising.size == 0) return double.NaN;
        long start = rising.item<long>(0);
        long peak = start + np.argmax(correlation[$"{start}:"]);
        var (position, _) = Parabolic(correlation, peak);
        return sampleRate / position;
    }

    /// <summary>
    /// A usable HPS estimate: multiply decimations of the ORIGINAL magnitude spectrum,
    /// excluding DC and interpolating its log. Unlike the gist's plotting routine this
    /// returns Hz. It does not recursively decimate an already multiplied spectrum.
    /// </summary>
    public static double FromHarmonicProductSpectrum(NDArray signal, double sampleRate, int harmonics = 7)
    {
        using var scope = NDScope.Open();
        var s = Samples(signal, sampleRate);
        if (harmonics < 2) throw new ArgumentOutOfRangeException(nameof(harmonics));
        if (s.size < 3) return double.NaN;
        var magnitude = np.abs(np.fft.rfft(s * BlackmanHarris(checked((int)s.size))));
        long length = magnitude.size / harmonics;
        if (length < 3) return double.NaN;
        var product = magnitude[$":{length}"].copy();
        for (int h = 2; h <= harmonics; h++)
            product = product * magnitude[$"::{h}"][$":{length}"];
        long peak = 1 + np.argmax(product["1:"]);
        var (position, _) = Parabolic(np.log(product), peak);
        return sampleRate * position / s.size;
    }

    /// <summary>
    /// The gist's six PRINTED HPS pass estimates, in their original order, before each
    /// multiplication. Preserves the recursive truncation for reproducibility. The
    /// original has no return value and can index outside its shrunken spectrum;
    /// this API returns six values, using NaN for an undefined interior peak.
    /// </summary>
    public static NDArray LegacyHarmonicProductPasses(NDArray signal, double sampleRate)
    {
        using var scope = NDScope.Open();
        var s = Samples(signal, sampleRate);
        var results = np.full(new Shape(6), double.NaN);
        if (s.size == 0) return scope.Returns(results);
        var spectrum = np.abs(np.fft.rfft(s * BlackmanHarris(checked((int)s.size))));
        for (int h = 2; h < 8; h++)
        {
            var decimated = spectrum[$"::{h}"].copy();
            spectrum = spectrum[$":{decimated.size}"];
            long peak = np.argmax(np.abs(spectrum));
            var (position, _) = Parabolic(np.abs(spectrum), peak);
            results[h - 2] = sampleRate * position / s.size;
            spectrum = spectrum * decimated;
        }
        return scope.Returns(results);
    }

    /// <summary>Symmetric four-term Blackman–Harris window (SciPy's default convention).</summary>
    public static NDArray BlackmanHarris(int length)
    {
        if (length < 0) throw new ArgumentOutOfRangeException(nameof(length));
        using var scope = NDScope.Open();
        if (length <= 1) return scope.Returns(np.ones(new Shape(length), NPTypeCode.Double));
        var phase = np.linspace(-Math.PI, Math.PI, length);
        var window = np.zeros(new Shape(length), NPTypeCode.Double);
        double[] coefficients = { 0.35875, 0.48829, 0.14128, 0.01168 };
        for (int k = 0; k < coefficients.Length; k++)
            window = window + coefficients[k] * np.cos(k * phase);
        return scope.Returns(window);
    }

    /// <summary>Vertex of the quadratic through an interior sample and its two neighbours.</summary>
    public static (double Position, double Value) Parabolic(NDArray values, long index)
    {
        using var scope = NDScope.Open();
        if (values is null || values.ndim != 1) throw new ArgumentException("Expected a vector.", nameof(values));
        if (index <= 0 || index >= values.size - 1) return (double.NaN, double.NaN);
        var f = values.astype(NPTypeCode.Double);
        double left = f.item<double>(index - 1), middle = f.item<double>(index), right = f.item<double>(index + 1);
        double curvature = left - 2 * middle + right;
        if (curvature == 0 || !double.IsFinite(left) || !double.IsFinite(middle) || !double.IsFinite(right))
            return (double.NaN, double.NaN);
        double position = 0.5 * (left - right) / curvature + index;
        double value = middle - 0.25 * (left - right) * (position - index);
        return (position, value);
    }

    /// <summary>Least-squares quadratic interpolation over an odd, centred sample count.</summary>
    public static (double Position, double Value) ParabolicPolyfit(NDArray values, long index, int samples = 5)
    {
        using var scope = NDScope.Open();
        if (values is null || values.ndim != 1) throw new ArgumentException("Expected a vector.", nameof(values));
        if (samples < 3 || samples % 2 == 0) throw new ArgumentOutOfRangeException(nameof(samples), "Use an odd sample count of at least three.");
        long half = samples / 2;
        if (index < half || index + half >= values.size) throw new ArgumentOutOfRangeException(nameof(index));
        var x = np.arange(index - half, index + half + 1).astype(NPTypeCode.Double);
        NDArray fit = np.polyfit(x, values[$"{index - half}:{index + half + 1}"].astype(NPTypeCode.Double), 2);
        double a = fit.item<double>(0), b = fit.item<double>(1), c = fit.item<double>(2);
        if (a == 0) return (double.NaN, double.NaN);
        double position = -0.5 * b / a;
        return (position, a * (position * position) + b * position + c);
    }

    private static NDArray Samples(NDArray signal, double sampleRate)
    {
        if (signal is null || signal.ndim != 1) throw new ArgumentException("Expected a one-dimensional signal.", nameof(signal));
        if (!(sampleRate > 0) || !double.IsFinite(sampleRate)) throw new ArgumentOutOfRangeException(nameof(sampleRate));
        if (signal.typecode == NPTypeCode.Complex) throw new ArgumentException("Expected real-valued samples.", nameof(signal));
        using var scope = NDScope.Open();
        var s = signal.astype(NPTypeCode.Double);
        if (!np.all(np.isfinite(s))) throw new ArgumentException("Expected finite samples.", nameof(signal));
        return scope.Returns(s);
    }

    public static void Demo()
    {
        using var scope = NDScope.Open();
        const double sampleRate = 8192;
        var time = np.arange(1024).astype(NPTypeCode.Double) / sampleRate;
        var signal = np.zeros(new Shape(1024), NPTypeCode.Double);
        for (int h = 1; h <= 7; h++) signal = signal + np.sin(2 * Math.PI * 384 * h * time) / h;
        Console.WriteLine($"384 Hz harmonic signal: crossings {FromCrossings(signal, sampleRate):F3}, FFT {FromFft(signal, sampleRate):F3}, autocorrelation {FromAutocorrelation(signal, sampleRate):F3}, HPS {FromHarmonicProductSpectrum(signal, sampleRate):F3} Hz");
        var localPeak = np.array(new double[] { 2, 3, 1, 6, 4, 2, 3, 1 });
        var polynomial = ParabolicPolyfit(localPeak, 3, 5);
        Console.WriteLine($"Five-sample polynomial peak: position={polynomial.Position:F3}, value={polynomial.Value:F3}");
    }
}
