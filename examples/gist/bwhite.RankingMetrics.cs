// Inspired by bwhite's ranking-metrics gist:
// https://gist.github.com/bwhite/3726239
// Pinned revision: 2c92e90259b01b4a657d20c0ad8390caadd59c8b / rank_metrics.py
// Source SHA256: b9abd30faa5fe1d4ff968769888bd774ad2220ce1d64365c66aa9b0a42055a1c
// No explicit license is recorded in that pinned file. Independent C# implementation of
// its numerical behavior; attribution does not imply a license grant.

namespace NumSharp.Examples.Gist;

/// <summary>
/// Retrieval scores on float64 relevance vectors in best-first order. Nonzero means relevant
/// for binary metrics. DCG uses relevance itself as gain, NOT 2^relevance - 1. Empty binary
/// queries score zero; empty collections have a NaN mean. Inputs are never mutated.
/// </summary>
public static class RankingMetrics
{
    public static double MeanReciprocalRank(IEnumerable<NDArray> queries)
    {
        ArgumentNullException.ThrowIfNull(queries);
        var values = new List<double>();
        foreach (var relevance in queries)
        {
            ForecastingMetrics.RequireVector(relevance, nameof(queries));
            using var scope = NDScope.Open();
            var indices = np.flatnonzero(relevance);
            values.Add(indices.size == 0 ? 0.0 : 1.0 / (indices.item<long>(0) + 1.0));
        }
        using var outer = NDScope.Open();
        return np.mean(np.array(values.ToArray())).item<double>();
    }

    /// <summary>
    /// Retains this gist's definition: precision at the LAST relevant position. This is not
    /// the alternative R-precision definition whose cutoff is the number of relevant items.
    /// </summary>
    public static double RPrecision(NDArray relevance)
    {
        ForecastingMetrics.RequireVector(relevance, nameof(relevance));
        using var scope = NDScope.Open();
        var indices = np.flatnonzero(relevance);
        return indices.size == 0 ? 0.0 : PrecisionAtK(relevance, checked((int)indices.item<long>(indices.size - 1) + 1));
    }

    public static double PrecisionAtK(NDArray relevance, int k)
    {
        ForecastingMetrics.RequireVector(relevance, nameof(relevance));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(k);
        if (relevance.size < k)
            throw new ArgumentException("Relevance score length < k", nameof(k));
        using var scope = NDScope.Open();
        return np.mean(np.not_equal(relevance[$":{k}"], 0.0).astype(np.float64)).item<double>();
    }

    public static double AveragePrecision(NDArray relevance)
    {
        ForecastingMetrics.RequireVector(relevance, nameof(relevance));
        using var scope = NDScope.Open();
        var indices = np.flatnonzero(relevance);
        if (indices.size == 0) return 0.0;
        var values = new double[checked((int)indices.size)];
        for (var i = 0; i < values.Length; ++i)
            values[i] = PrecisionAtK(relevance, checked((int)indices.item<long>(i) + 1));
        return np.mean(np.array(values)).item<double>();
    }

    public static double MeanAveragePrecision(IEnumerable<NDArray> queries)
    {
        ArgumentNullException.ThrowIfNull(queries);
        using var scope = NDScope.Open();
        return np.mean(np.array(queries.Select(AveragePrecision).ToArray())).item<double>();
    }

    /// <summary>
    /// Method 0 leaves the first two ranks undiscounted; method 1 starts logarithmic
    /// discounting at rank two. k follows Python prefix-slice semantics, including negative k.
    /// The gist accepts an invalid method when the selected prefix is empty; so does this port.
    /// </summary>
    public static double DcgAtK(NDArray relevance, int k, int method = 0)
    {
        ForecastingMetrics.RequireVector(relevance, nameof(relevance));
        using var scope = NDScope.Open();
        var prefix = relevance[$":{k}"];
        if (prefix.size == 0) return 0.0;
        return method switch
        {
            0 => prefix.item<double>(0) + np.sum(prefix["1:"] /
                np.log2(np.arange(2.0, prefix.size + 1.0))).item<double>(),
            1 => np.sum(prefix / np.log2(np.arange(2.0, prefix.size + 2.0))).item<double>(),
            _ => throw new ArgumentException("method must be 0 or 1.", nameof(method))
        };
    }

    public static double NdcgAtK(NDArray relevance, int k, int method = 0)
    {
        ForecastingMetrics.RequireVector(relevance, nameof(relevance));
        using var scope = NDScope.Open();
        var ideal = np.sort(relevance)["::-1"];
        var denominator = DcgAtK(ideal, k, method);
        return denominator == 0.0 ? 0.0 : DcgAtK(relevance, k, method) / denominator;
    }

    public static void Demo()
    {
        using var relevance = np.array(new[] { 3.0, 2.0, 3.0, 0.0, 0.0, 1.0, 2.0, 2.0, 3.0, 0.0 });
        Console.WriteLine($"Average precision: {AveragePrecision(relevance):G17}");
        Console.WriteLine($"NDCG@10: {NdcgAtK(relevance, 10):G17}");
    }
}
