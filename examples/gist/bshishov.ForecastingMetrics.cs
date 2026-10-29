// Inspired by bshishov's forecasting-metrics gist:
// https://gist.github.com/bshishov/5dc237f59f019b26145648e2124ca1c9
// Pinned revision: 824c2f332d919185289ded2709752c67bd41244d / forecasting_metrics.py
// Source SHA256: df7a6d0709b228cd8ae9b44a416da200e7f988ef2ef3e0479b89f89bc80a17be
// No explicit license is recorded in that pinned file. Independent C# implementation of
// its mathematical behavior; attribution does not imply a license grant.

namespace NumSharp.Examples.Gist;

/// <summary>
/// Thirty forecast scores over equally sized float64 time-series vectors. Inputs may be
/// strided or reversed and are never modified. Scalar results use item&lt;double&gt;(); returned
/// arrays belong to the caller. Percentages are fractions, not multiplied by 100.
/// </summary>
public static class ForecastingMetrics
{
    public const double Epsilon = 1e-10;
    public static IReadOnlyList<string> MetricNames { get; } = Array.AsReadOnly(new[]
    {
        "mse", "rmse", "nrmse", "me", "mae", "mad", "gmae", "mdae", "mpe", "mape",
        "mdape", "smape", "smdape", "maape", "mase", "std_ae", "std_ape", "rmspe",
        "rmdspe", "rmsse", "inrse", "rrse", "mre", "rae", "mrae", "mdrae", "gmrae",
        "mbrae", "umbrae", "mda"
    });

    internal static void RequireVector(NDArray value, string name)
    {
        ArgumentNullException.ThrowIfNull(value, name);
        if (value.ndim != 1 || value.typecode != NPTypeCode.Double)
            throw new ArgumentException("Expected a one-dimensional float64 array.", name);
    }

    private static void Pair(NDArray actual, NDArray predicted)
    {
        RequireVector(actual, nameof(actual));
        RequireVector(predicted, nameof(predicted));
        if (actual.size != predicted.size)
            throw new ArgumentException("Time-series lengths must match.", nameof(predicted));
    }

    public static NDArray Error(NDArray actual, NDArray predicted)
    {
        Pair(actual, predicted);
        return actual - predicted;
    }

    public static NDArray PercentageError(NDArray actual, NDArray predicted)
    {
        using var scope = NDScope.Open();
        return scope.Returns(Error(actual, predicted) / (actual + Epsilon));
    }

    /// <summary>A caller-owned view of the history used by a seasonal persistence forecast.</summary>
    public static NDArray NaiveForecasting(NDArray actual, int seasonality = 1)
    {
        RequireVector(actual, nameof(actual));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(seasonality);
        return actual[$":-{seasonality}"];
    }

    public static NDArray RelativeError(NDArray actual, NDArray predicted,
        NDArray? benchmark = null, int seasonality = 1)
    {
        using var scope = NDScope.Open();
        Pair(actual, predicted);
        if (benchmark is not null)
        {
            Pair(actual, benchmark);
            return scope.Returns((actual - predicted) / (actual - benchmark + Epsilon));
        }
        var history = NaiveForecasting(actual, seasonality);
        var current = actual[$"{seasonality}:"];
        return scope.Returns((current - predicted[$"{seasonality}:"]) / (current - history + Epsilon));
    }

    public static NDArray BoundedRelativeError(NDArray actual, NDArray predicted,
        NDArray? benchmark = null, int seasonality = 1)
    {
        using var scope = NDScope.Open();
        Pair(actual, predicted);
        NDArray forecastError, baselineError;
        if (benchmark is not null)
        {
            Pair(actual, benchmark);
            forecastError = np.abs(actual - predicted);
            baselineError = np.abs(actual - benchmark);
        }
        else
        {
            var history = NaiveForecasting(actual, seasonality);
            var current = actual[$"{seasonality}:"];
            forecastError = np.abs(current - predicted[$"{seasonality}:"]);
            baselineError = np.abs(current - history);
        }
        return scope.Returns(forecastError / (forecastError + baselineError + Epsilon));
    }

    public static double GeometricMean(NDArray values)
    {
        RequireVector(values, nameof(values));
        using var scope = NDScope.Open();
        return np.exp(np.mean(np.log(values))).item<double>();
    }

    /// <summary>
    /// Compute by the original short name. An explicit benchmark overrides seasonality for
    /// relative scores; scaled scores always use seasonal persistence. Empty means and
    /// zero denominators retain IEEE NaN/infinity. Seasonality must be positive.
    /// </summary>
    public static double Compute(string metric, NDArray actual, NDArray predicted,
        NDArray? benchmark = null, int seasonality = 1)
    {
        Pair(actual, predicted);
        using var scope = NDScope.Open();
        var error = actual - predicted;
        double Mean(NDArray x) => np.mean(x).item<double>();
        double Median(NDArray x) => np.median(x).item<double>();
        double Sum(NDArray x) => np.sum(x).item<double>();
        NDArray Percentage() => error / (actual + Epsilon);
        NDArray Relative() => RelativeError(actual, predicted, benchmark, seasonality);
        NDArray Symmetric() => 2.0 * np.abs(error) / (np.abs(actual) + np.abs(predicted) + Epsilon);
        double NaiveMae() => Mae(actual[$"{seasonality}:"], NaiveForecasting(actual, seasonality));
        double Root(double x) => np.sqrt(np.array(x)).item<double>();

        return metric switch
        {
            "mse" => Mean(np.square(error)),
            "rmse" => Root(Mean(np.square(error))),
            "nrmse" => Root(Mean(np.square(error))) / (np.max(actual).item<double>() - np.min(actual).item<double>()),
            "me" => Mean(error),
            "mae" => Mean(np.abs(error)),
            "mad" => Mean(np.abs(error - Mean(error))),
            "gmae" => GeometricMean(np.abs(error)),
            "mdae" => Median(np.abs(error)),
            "mpe" => Mean(Percentage()),
            "mape" => Mean(np.abs(Percentage())),
            "mdape" => Median(np.abs(Percentage())),
            "smape" => Mean(Symmetric()),
            "smdape" => Median(Symmetric()),
            "maape" => Mean(np.arctan(np.abs(Percentage()))),
            "mase" => Mean(np.abs(error)) / NaiveMae(),
            // These intentionally retain the gist's signed-error-minus-MAE definition.
            "std_ae" => Root(Sum(np.square(error - Mean(np.abs(error)))) / (actual.size - 1)),
            "std_ape" => Root(Sum(np.square(Percentage() - Mean(np.abs(Percentage())))) / (actual.size - 1)),
            "rmspe" => Root(Mean(np.square(Percentage()))),
            "rmdspe" => Root(Median(np.square(Percentage()))),
            "rmsse" => Root(Mean(np.square(np.abs(error) / NaiveMae()))),
            "inrse" or "rrse" => Root(Sum(np.square(error)) / Sum(np.square(actual - Mean(actual)))),
            "mre" => Mean(Relative()),
            "rae" => Sum(np.abs(error)) / (Sum(np.abs(actual - Mean(actual))) + Epsilon),
            "mrae" => Mean(np.abs(Relative())),
            "mdrae" => Median(np.abs(Relative())),
            "gmrae" => GeometricMean(np.abs(Relative())),
            "mbrae" => Mean(BoundedRelativeError(actual, predicted, benchmark, seasonality)),
            "umbrae" => Unscale(Mean(BoundedRelativeError(actual, predicted, benchmark, seasonality))),
            // The gist compares next prediction with previous ACTUAL, not previous prediction.
            "mda" => Mean(np.equal(np.sign(actual["1:"] - actual[":-1"]),
                np.sign(predicted["1:"] - actual[":-1"])).astype(np.float64)),
            _ => throw new ArgumentException($"Unknown forecast metric '{metric}'.", nameof(metric))
        };
    }

    private static double Unscale(double bounded) => bounded / (1.0 - bounded);

    /// <summary>Like the gist's evaluator, an uncomputable or unknown score is recorded as NaN.</summary>
    public static IReadOnlyDictionary<string, double> Evaluate(NDArray actual, NDArray predicted,
        IEnumerable<string>? metrics = null)
    {
        var results = new Dictionary<string, double>(StringComparer.Ordinal);
        foreach (var metric in metrics ?? new[] { "mae", "mse", "smape", "umbrae" })
        {
            try { results[metric] = Compute(metric, actual, predicted); }
            catch (ArgumentException) { results[metric] = double.NaN; }
            catch (InvalidOperationException) { results[metric] = double.NaN; }
        }
        return results;
    }

    public static IReadOnlyDictionary<string, double> EvaluateAll(NDArray actual, NDArray predicted)
        => Evaluate(actual, predicted, MetricNames);

    public static double Mse(NDArray a, NDArray p) => Compute("mse", a, p);
    public static double Rmse(NDArray a, NDArray p) => Compute("rmse", a, p);
    public static double Nrmse(NDArray a, NDArray p) => Compute("nrmse", a, p);
    public static double Me(NDArray a, NDArray p) => Compute("me", a, p);
    public static double Mae(NDArray a, NDArray p) => Compute("mae", a, p);
    public static double Mad(NDArray a, NDArray p) => Compute("mad", a, p);
    public static double Gmae(NDArray a, NDArray p) => Compute("gmae", a, p);
    public static double Mdae(NDArray a, NDArray p) => Compute("mdae", a, p);
    public static double Mpe(NDArray a, NDArray p) => Compute("mpe", a, p);
    public static double Mape(NDArray a, NDArray p) => Compute("mape", a, p);
    public static double Mdape(NDArray a, NDArray p) => Compute("mdape", a, p);
    public static double Smape(NDArray a, NDArray p) => Compute("smape", a, p);
    public static double Smdape(NDArray a, NDArray p) => Compute("smdape", a, p);
    public static double Maape(NDArray a, NDArray p) => Compute("maape", a, p);
    public static double Mase(NDArray a, NDArray p, int seasonality = 1) => Compute("mase", a, p, seasonality: seasonality);
    public static double StdAe(NDArray a, NDArray p) => Compute("std_ae", a, p);
    public static double StdApe(NDArray a, NDArray p) => Compute("std_ape", a, p);
    public static double Rmspe(NDArray a, NDArray p) => Compute("rmspe", a, p);
    public static double Rmdspe(NDArray a, NDArray p) => Compute("rmdspe", a, p);
    public static double Rmsse(NDArray a, NDArray p, int seasonality = 1) => Compute("rmsse", a, p, seasonality: seasonality);
    public static double Inrse(NDArray a, NDArray p) => Compute("inrse", a, p);
    public static double Rrse(NDArray a, NDArray p) => Compute("rrse", a, p);
    public static double Rae(NDArray a, NDArray p) => Compute("rae", a, p);
    public static double Mre(NDArray a, NDArray p, NDArray? b = null, int seasonality = 1) => Compute("mre", a, p, b, seasonality);
    public static double Mrae(NDArray a, NDArray p, NDArray? b = null, int seasonality = 1) => Compute("mrae", a, p, b, seasonality);
    public static double Mdrae(NDArray a, NDArray p, NDArray? b = null, int seasonality = 1) => Compute("mdrae", a, p, b, seasonality);
    public static double Gmrae(NDArray a, NDArray p, NDArray? b = null, int seasonality = 1) => Compute("gmrae", a, p, b, seasonality);
    public static double Mbrae(NDArray a, NDArray p, NDArray? b = null, int seasonality = 1) => Compute("mbrae", a, p, b, seasonality);
    public static double Umbrae(NDArray a, NDArray p, NDArray? b = null, int seasonality = 1) => Compute("umbrae", a, p, b, seasonality);
    public static double Mda(NDArray a, NDArray p) => Compute("mda", a, p);

    public static void Demo()
    {
        using var actual = np.array(new[] { 2.0, 4.0, 8.0, 16.0, 32.0, 64.0 });
        using var predicted = np.array(new[] { 1.0, 5.0, 6.0, 20.0, 30.0, 60.0 });
        foreach (var (name, value) in EvaluateAll(actual, predicted))
            Console.WriteLine($"{name,-8} {value:G17}");
    }
}
