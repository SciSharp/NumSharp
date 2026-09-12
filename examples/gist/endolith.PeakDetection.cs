// Gist: https://gist.github.com/endolith/250860
// Pinned revision: bd0936f3983d43b37472f7b59991fdd4bcbb35c5; source: peakdetect.py.
// Python translation: endolith. Original algorithm: Eli Billauer, 3 April 2005.
// The pinned source explicitly releases the algorithm to the public domain.
using System;
using System.Collections.Generic;

namespace NumSharp.Examples.Gist;

/// <summary>Hysteresis peak detection: a delta-sized reversal confirms a pending extremum.</summary>
/// <remarks>
/// Real vectors are converted to float64, including the returned coordinates. NaN
/// samples retain the source algorithm's comparison semantics (they do not update
/// extrema). Unlike the script's process exit, invalid arguments throw exceptions;
/// NaN/infinite delta is rejected explicitly rather than silently producing no peaks.
/// </remarks>
public static class PeakDetection
{
    /// <summary>
    /// Returns (positions, values) rows for maxima and minima. Like the gist, an
    /// empty result has shape (0,), ties keep their FIRST position, and a trailing
    /// unconfirmed extremum is not emitted. Caller owns both returned arrays.
    /// </summary>
    public static (NDArray Maxima, NDArray Minima) Detect(NDArray values, double delta, NDArray? positions = null)
    {
        if (values is null || values.ndim != 1) throw new ArgumentException("Expected a vector.", nameof(values));
        if (positions is not null && (positions.ndim != 1 || values.size != positions.size))
            throw new ArgumentException("Input vectors v and x must have same length", nameof(positions));
        if (!(delta > 0) || !double.IsFinite(delta)) throw new ArgumentOutOfRangeException(nameof(delta), "Input argument delta must be positive and finite");
        if (values.typecode == NPTypeCode.Complex || positions?.typecode == NPTypeCode.Complex)
            throw new ArgumentException("Expected real-valued vectors.");
        using var scope = NDScope.Open();
        var v = values.astype(NPTypeCode.Double);
        var x = positions?.astype(NPTypeCode.Double) ?? np.arange(v.size).astype(NPTypeCode.Double);
        var maxima = new List<(double Position, double Value)>();
        var minima = new List<(double Position, double Value)>();
        double minimum = double.PositiveInfinity, maximum = double.NegativeInfinity;
        double minimumPosition = double.NaN, maximumPosition = double.NaN;
        bool seekingMaximum = true;
        for (long i = 0; i < v.size; i++)
        {
            double sample = v.item<double>(i), position = x.item<double>(i);
            if (sample > maximum) { maximum = sample; maximumPosition = position; }
            if (sample < minimum) { minimum = sample; minimumPosition = position; }
            if (seekingMaximum && sample < maximum - delta)
            {
                maxima.Add((maximumPosition, maximum));
                minimum = sample;
                minimumPosition = position;
                seekingMaximum = false;
            }
            else if (!seekingMaximum && sample > minimum + delta)
            {
                minima.Add((minimumPosition, minimum));
                maximum = sample;
                maximumPosition = position;
                seekingMaximum = true;
            }
        }
        return scope.Returns((Table(maxima), Table(minima)));
    }

    private static NDArray Table(List<(double Position, double Value)> peaks)
    {
        if (peaks.Count == 0) return np.array(Array.Empty<double>());
        var rows = new double[peaks.Count, 2];
        for (int i = 0; i < peaks.Count; i++) { rows[i, 0] = peaks[i].Position; rows[i, 1] = peaks[i].Value; }
        return np.array(rows);
    }

    public static void Demo()
    {
        using var scope = NDScope.Open();
        var samples = np.array(new double[] { 0, 0, 0, 2, 0, 0, 0, -2, 0, 0, 0, 2, 0, 0, 0, -2, 0 });
        var (maxima, minima) = Detect(samples, 0.3);
        Console.WriteLine($"Confirmed peaks (position, value): maxima {maxima}; minima {minima}");
    }
}
