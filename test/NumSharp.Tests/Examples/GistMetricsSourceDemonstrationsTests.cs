using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Examples.Gist;

namespace NumSharp.UnitTest.Examples;

// Case IDs are original pinned rank_metrics.py expression lines; see coverage-metrics.json.
// These fixtures exercise the compiled ports, not another C# implementation of the metrics.
[TestClass]
public class GistMetricsSourceDemonstrationsTests
{
    [TestMethod]
    public void RankingSource_MeanReciprocalRank_AllThreeDemonstrations()
    {
        using var scope = NDScope.Open();
        var right = np.array(new[] { 0.0, 0.0, 1.0 });
        var middle = np.array(new[] { 0.0, 1.0, 0.0 });
        var left = np.array(new[] { 1.0, 0.0, 0.0 });
        Assert.AreEqual(0.61111111111111105, RankingMetrics.MeanReciprocalRank(new[] { right, middle, left }), "rank-L20");

        var matrix = np.array(new double[,] { { 0, 0, 0 }, { 0, 1, 0 }, { 1, 0, 0 } });
        Assert.AreEqual(0.5, RankingMetrics.MeanReciprocalRank(new[] { matrix["0, :"], matrix["1, :"], matrix["2, :"] }), "rank-L23: rows of an ndarray");

        var fourth = np.array(new[] { 0.0, 0.0, 0.0, 1.0 });
        Assert.AreEqual(0.75, RankingMetrics.MeanReciprocalRank(new[] { fourth, left, left }), "rank-L26: unequal query lengths");
    }

    [TestMethod]
    public void RankingSource_RPrecision_AllThreeDemonstrations()
    {
        using var scope = NDScope.Open();
        Assert.AreEqual(0.33333333333333331, RankingMetrics.RPrecision(np.array(new[] { 0.0, 0.0, 1.0 })), "rank-L46");
        Assert.AreEqual(0.5, RankingMetrics.RPrecision(np.array(new[] { 0.0, 1.0, 0.0 })), "rank-L49");
        Assert.AreEqual(1.0, RankingMetrics.RPrecision(np.array(new[] { 1.0, 0.0, 0.0 })), "rank-L52");
    }

    [TestMethod]
    public void RankingSource_PrecisionAtK_AllValuesAndOriginalError()
    {
        using var r = np.array(new[] { 0.0, 0.0, 1.0 });
        Assert.AreEqual(0.0, RankingMetrics.PrecisionAtK(r, 1), "rank-L75");
        Assert.AreEqual(0.0, RankingMetrics.PrecisionAtK(r, 2), "rank-L77");
        Assert.AreEqual(0.33333333333333331, RankingMetrics.PrecisionAtK(r, 3), "rank-L79");
        var error = Assert.ThrowsException<ArgumentException>(() => RankingMetrics.PrecisionAtK(r, 4), "rank-L81");
        StringAssert.StartsWith(error.Message, "Relevance score length < k");
    }

    [TestMethod]
    public void RankingSource_AveragePrecision_AndIndependentAreaMeasurement()
    {
        using var r = np.array(new[] { 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 });
        using var totalRelevant = np.sum(r);
        Assert.AreEqual(5.0, totalRelevant.item<double>());
        var deltaRecall = 1.0 / totalRelevant.item<double>();
        Assert.AreEqual(0.2, deltaRecall, "rank-L110 setup");
        var actual = RankingMetrics.AveragePrecision(r);
        Assert.AreEqual(0.78333333333333333, actual, "rank-L113 metric demonstration");

        // rank-L111 is an independent Python-list sum, not average_precision's np.mean.
        // Re-running its exact arithmetic on Python 3.12 produces the next float64 number;
        // the historical printed result is therefore not claimed as today's exact result.
        const double observedPython312Area = 0.7833333333333334;
        Assert.AreEqual(1L, BitConverter.DoubleToInt64Bits(observedPython312Area) - BitConverter.DoubleToInt64Bits(actual));
        foreach (var k in new[] { 1, 2, 4, 6, 10 })
            Assert.IsTrue(RankingMetrics.PrecisionAtK(r, k) > 0.0, $"rank-L111 contributing rank {k}");
    }

    [TestMethod]
    public void RankingSource_MeanAveragePrecision_BothDemonstrations()
    {
        using var r = np.array(new[] { 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 });
        using var zero = np.array(new[] { 0.0 });
        Assert.AreEqual(0.78333333333333333, RankingMetrics.MeanAveragePrecision(new[] { r }), "rank-L136");
        Assert.AreEqual(0.39166666666666666, RankingMetrics.MeanAveragePrecision(new[] { r, zero }), "rank-L139");
    }

    [DataTestMethod]
    [DataRow(1, 0, 3.0, "rank-L161")]
    [DataRow(1, 1, 3.0, "rank-L163")]
    [DataRow(2, 0, 5.0, "rank-L165")]
    [DataRow(2, 1, 4.2618595071429155, "rank-L167")]
    [DataRow(10, 0, 9.6051177391888114, "rank-L169")]
    [DataRow(11, 0, 9.6051177391888114, "rank-L171")]
    public void RankingSource_Dcg_AllSixDemonstrations(int k, int method, double expected, string caseId)
    {
        using var r = np.array(new[] { 3.0, 2.0, 3.0, 0.0, 0.0, 1.0, 2.0, 2.0, 3.0, 0.0 });
        Assert.AreEqual(expected, RankingMetrics.DcgAtK(r, k, method), 2e-15, caseId);
    }

    [TestMethod]
    public void RankingSource_Ndcg_AllFiveDemonstrations()
    {
        using var scope = NDScope.Open();
        var example = np.array(new[] { 3.0, 2.0, 3.0, 0.0, 0.0, 1.0, 2.0, 2.0, 3.0, 0.0 });
        var mixed = np.array(new[] { 2.0, 1.0, 2.0, 0.0 });
        Assert.AreEqual(1.0, RankingMetrics.NdcgAtK(example, 1), "rank-L204");
        Assert.AreEqual(0.9203032077642922, RankingMetrics.NdcgAtK(mixed, 4), 2e-16, "rank-L207");
        Assert.AreEqual(0.96519546960144276, RankingMetrics.NdcgAtK(mixed, 4, 1), 2e-16, "rank-L209");
        Assert.AreEqual(0.0, RankingMetrics.NdcgAtK(np.array(new[] { 0.0 }), 1), "rank-L211");
        Assert.AreEqual(1.0, RankingMetrics.NdcgAtK(np.array(new[] { 1.0 }), 2), "rank-L213");
    }

    [TestMethod]
    public void ForecastSource_DefaultEvaluator_AndNamedSurface_AreMeasurable()
    {
        using var a = np.array(new[] { 2.0, 4.0, 8.0, 16.0, 32.0, 64.0 });
        using var p = np.array(new[] { 1.0, 5.0, 6.0, 20.0, 30.0, 60.0 });
        var defaults = ForecastingMetrics.Evaluate(a, p);
        CollectionAssert.AreEquivalent(new[] { "mae", "mse", "smape", "umbrae" }, defaults.Keys.ToArray());
        Assert.AreEqual(7.0, defaults["mse"]);
        Assert.AreEqual(2.3333333333333335, defaults["mae"]);
        Assert.AreEqual(0.2543096091437346, defaults["smape"], 6e-17);
        Assert.AreEqual(0.3235294117575692, defaults["umbrae"], 6e-17);
        var all = ForecastingMetrics.EvaluateAll(a, p);
        var named = new Dictionary<string, Func<double>>
        {
            ["mse"] = () => ForecastingMetrics.Mse(a,p), ["rmse"] = () => ForecastingMetrics.Rmse(a,p),
            ["nrmse"] = () => ForecastingMetrics.Nrmse(a,p), ["me"] = () => ForecastingMetrics.Me(a,p),
            ["mae"] = () => ForecastingMetrics.Mae(a,p), ["mad"] = () => ForecastingMetrics.Mad(a,p),
            ["gmae"] = () => ForecastingMetrics.Gmae(a,p), ["mdae"] = () => ForecastingMetrics.Mdae(a,p),
            ["mpe"] = () => ForecastingMetrics.Mpe(a,p), ["mape"] = () => ForecastingMetrics.Mape(a,p),
            ["mdape"] = () => ForecastingMetrics.Mdape(a,p), ["smape"] = () => ForecastingMetrics.Smape(a,p),
            ["smdape"] = () => ForecastingMetrics.Smdape(a,p), ["maape"] = () => ForecastingMetrics.Maape(a,p),
            ["mase"] = () => ForecastingMetrics.Mase(a,p), ["std_ae"] = () => ForecastingMetrics.StdAe(a,p),
            ["std_ape"] = () => ForecastingMetrics.StdApe(a,p), ["rmspe"] = () => ForecastingMetrics.Rmspe(a,p),
            ["rmdspe"] = () => ForecastingMetrics.Rmdspe(a,p), ["rmsse"] = () => ForecastingMetrics.Rmsse(a,p),
            ["inrse"] = () => ForecastingMetrics.Inrse(a,p), ["rrse"] = () => ForecastingMetrics.Rrse(a,p),
            ["mre"] = () => ForecastingMetrics.Mre(a,p), ["rae"] = () => ForecastingMetrics.Rae(a,p),
            ["mrae"] = () => ForecastingMetrics.Mrae(a,p), ["mdrae"] = () => ForecastingMetrics.Mdrae(a,p),
            ["gmrae"] = () => ForecastingMetrics.Gmrae(a,p), ["mbrae"] = () => ForecastingMetrics.Mbrae(a,p),
            ["umbrae"] = () => ForecastingMetrics.Umbrae(a,p), ["mda"] = () => ForecastingMetrics.Mda(a,p)
        };
        CollectionAssert.AreEquivalent(ForecastingMetrics.MetricNames.ToArray(), named.Keys.ToArray());
        foreach (var (name, call) in named)
            Assert.AreEqual(BitConverter.DoubleToInt64Bits(all[name]), BitConverter.DoubleToInt64Bits(call()), name);
        // Independent values for all thirty are pinned by GistMetricsTests, not this dispatch check.
    }

    [TestMethod]
    public void ForecastSource_SeasonalAndExplicitBenchmarks_AllRelativeScores()
    {
        using var a = np.array(new[] { 2.0, 4.0, 8.0, 16.0, 32.0, 64.0 });
        using var p = np.array(new[] { 1.0, 5.0, 6.0, 20.0, 30.0, 60.0 });
        using var benchmark = np.ones(6, dtype: np.float64);
        string[] names = { "mre", "mrae", "mdrae", "gmrae", "mbrae", "umbrae" };
        double[] seasonal = { 0.04166666666584201, 0.20833333333111978, 0.20833333333046875, 0.16666666666536462, 0.16346153846025566, 0.1954022988487416 };
        double[] explicitScores = { 0.13562041302451763, 0.33562041302022133, 0.27619047618754644, 0.21686752817714736, 0.21717601518666166, 0.2774263683788975 };
        for (var i = 0; i < names.Length; ++i)
        {
            Assert.AreEqual(seasonal[i], ForecastingMetrics.Compute(names[i], a, p, seasonality: 2), 4e-16, names[i] + " seasonal");
            Assert.AreEqual(explicitScores[i], ForecastingMetrics.Compute(names[i], a, p, benchmark), 4e-16, names[i] + " explicit benchmark");
        }
        Assert.AreEqual(0.10370370370370371, ForecastingMetrics.Mase(a, p, 2), 2e-16);
        Assert.AreEqual(0.11758894715842624, ForecastingMetrics.Rmsse(a, p, 2), 2e-16);
        using var geometric = np.array(new[] { 1.0, 4.0, 16.0 });
        Assert.AreEqual(4.0, ForecastingMetrics.GeometricMean(geometric));
    }
}
