using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Examples.Gist;

namespace NumSharp.UnitTest.Examples;

[TestClass]
public class GistMetricsTests
{
    // Observed BEFORE the C# port, using reviewed pinned forecasting formulas on NumPy 2.4.2.
    // These immutable values complement the live oracle; this project needs no Python runtime.
    private static readonly double[] ForecastExpected =
    {
        7.0, 2.6457513110645907, 0.042673408242977266, 0.6666666666666666,
        2.3333333333333335, 2.111111111111111, 2.0, 2.0, 0.062499999996565754,
        0.22916666666062824, 0.2499999999953125, 0.2543096091437346,
        0.222222222220679, 0.2205702030571212, 0.1881720430107527,
        3.3466401061363027, 0.34308648082765625, 0.27243118396196087,
        0.2499999999953125, 0.21336704121488634, 0.12216944435630522,
        0.12216944435630522, -0.04999999999648438, 0.1296296296295096,
        0.34999999999101566, 0.499999999975, 0.2871745887436948,
        0.2444444444403704, 0.3235294117575692, 1.0
    };

    [TestMethod]
    public void AllThirtyForecastMetrics_MatchRecordedNumpy()
    {
        using var a = np.array(new[] { 2.0, 4.0, 8.0, 16.0, 32.0, 64.0 });
        using var p = np.array(new[] { 1.0, 5.0, 6.0, 20.0, 30.0, 60.0 });
        var scores = ForecastingMetrics.EvaluateAll(a, p);
        Assert.AreEqual(30, scores.Count);
        for (var i = 0; i < ForecastExpected.Length; ++i)
        {
            var name = ForecastingMetrics.MetricNames[i];
            Assert.AreEqual(ForecastExpected[i], scores[name], 2e-14 * System.Math.Max(1.0, System.Math.Abs(ForecastExpected[i])), name);
        }
        Assert.AreEqual(2.0, a.item<double>(0), "The input remains alive and unchanged.");
    }

    [TestMethod]
    public void ForecastHelpers_ReturnOwnedResults_AndSeasonalViews()
    {
        using var a = np.array(new[] { 2.0, 4.0, 8.0, 16.0, 32.0, 64.0 });
        using var p = np.array(new[] { 1.0, 5.0, 6.0, 20.0, 30.0, 60.0 });
        using var benchmark = np.ones(6, dtype: np.float64);
        using var error = ForecastingMetrics.Error(a, p);
        using var percentage = ForecastingMetrics.PercentageError(a, p);
        using var seasonal = ForecastingMetrics.NaiveForecasting(a, 2);
        using var relative = ForecastingMetrics.RelativeError(a, p, seasonality: 2);
        using var bounded = ForecastingMetrics.BoundedRelativeError(a, p, benchmark);
        Assert.AreEqual(-4.0, error.item<double>(3));
        Assert.AreEqual(1.0 / (2.0 + 1e-10), percentage.item<double>(0));
        Assert.AreEqual(4L, seasonal.size);
        Assert.AreEqual(16.0, seasonal.item<double>(3));
        Assert.AreEqual(2.0 / (6.0 + 1e-10), relative.item<double>(0));
        Assert.AreEqual(1.0 / (2.0 + 1e-10), bounded.item<double>(0));
        seasonal[0] = 9.0;
        Assert.AreEqual(9.0, a.item<double>(0), "NaiveForecasting deliberately returns a shared view.");
    }

    [TestMethod]
    public void Forecasts_RespectNoncontiguousInputs()
    {
        using var backingA = np.array(new[] { 99.0, 2.0, 99.0, 4.0, 99.0, 8.0, 99.0, 16.0 });
        using var backingP = np.array(new[] { 99.0, 1.0, 99.0, 5.0, 99.0, 6.0, 99.0, 20.0 });
        using var a = backingA["1::2"];
        using var p = backingP["1::2"];
        Assert.AreEqual(5.5, ForecastingMetrics.Mse(a, p));
        using var reverseA = a["::-1"];
        using var reverseP = p["::-1"];
        Assert.AreEqual(5.5, ForecastingMetrics.Mse(reverseA, reverseP));
        Assert.AreEqual(99.0, backingA.item<double>(0));
    }

    [TestMethod]
    public void ForecastEdgeCases_AreExplicit()
    {
        using var empty = np.array(Array.Empty<double>());
        using var constant = np.array(new[] { 1.0, 1.0 });
        using var zero = np.array(new[] { 0.0, 0.0 });
        using var shortArray = np.array(new[] { 1.0 });
        using var integers = np.array(new[] { 1, 2 });
        using var matrix = np.array(new double[,] { { 1.0, 2.0 } });
        Assert.IsTrue(double.IsNaN(ForecastingMetrics.Mse(empty, empty)));
        Assert.IsTrue(double.IsPositiveInfinity(ForecastingMetrics.Nrmse(constant, zero)));
        Assert.AreEqual(0.0, ForecastingMetrics.Mape(zero, zero));
        Assert.AreEqual(0.0, ForecastingMetrics.Gmae(zero, zero));
        Assert.ThrowsException<ArgumentException>(() => ForecastingMetrics.Mse(constant, shortArray));
        Assert.ThrowsException<ArgumentException>(() => ForecastingMetrics.Mse(integers, integers));
        Assert.ThrowsException<ArgumentException>(() => ForecastingMetrics.Mse(matrix, matrix));
        Assert.ThrowsException<ArgumentOutOfRangeException>(() => ForecastingMetrics.NaiveForecasting(constant, 0));
        Assert.IsTrue(double.IsNaN(ForecastingMetrics.Evaluate(constant, zero, new[] { "not-a-metric" })["not-a-metric"]));
    }

    [TestMethod]
    public void ForecastMda_UsesPreviousActual_AndStdAeRetainsOriginalDefinition()
    {
        using var a = np.array(new[] { 10.0, 11.0, 12.0 });
        using var p = np.array(new[] { 20.0, 15.0, 14.0 });
        Assert.AreEqual(1.0, ForecastingMetrics.Mda(a, p)); // predictions themselves decrease
        using var b = np.array(new[] { 2.0, 4.0 });
        using var q = np.array(new[] { 3.0, 5.0 });
        Assert.AreEqual(System.Math.Sqrt(8.0), ForecastingMetrics.StdAe(b, q));
    }

    [TestMethod]
    public void AllSevenRankingMetrics_MatchRecordedNumpy()
    {
        using var relevance = np.array(new[] { 3.0, 2.0, 3.0, 0.0, 0.0, 1.0, 2.0, 2.0, 3.0, 0.0 });
        using var q1 = np.array(new[] { 0.0, 0.0, 1.0 });
        using var q2 = np.array(new[] { 0.0, 1.0, 0.0 });
        using var q3 = np.array(new[] { 1.0, 0.0, 0.0 });
        var queries = new[] { q1, q2, q3 };
        Assert.AreEqual(0.611111111111111, RankingMetrics.MeanReciprocalRank(queries), 2e-16);
        Assert.AreEqual(0.7777777777777778, RankingMetrics.RPrecision(relevance), 2e-16);
        Assert.AreEqual(1.0, RankingMetrics.PrecisionAtK(relevance, 3));
        Assert.AreEqual(0.8441043083900226, RankingMetrics.AveragePrecision(relevance), 2e-16);
        Assert.AreEqual(0.611111111111111, RankingMetrics.MeanAveragePrecision(queries), 2e-16);
        Assert.AreEqual(9.605117739188811, RankingMetrics.DcgAtK(relevance, 10), 2e-14);
        Assert.AreEqual(8.318753101481006, RankingMetrics.DcgAtK(relevance, 10, 1), 2e-14);
        Assert.AreEqual(0.8824943995338175, RankingMetrics.NdcgAtK(relevance, 10), 2e-15);
        Assert.AreEqual(0.9168088790321769, RankingMetrics.NdcgAtK(relevance, 10, 1), 2e-15);
    }

    [TestMethod]
    public void RankingEdgeCases_PreserveGistCutoffs()
    {
        using var empty = np.array(Array.Empty<double>());
        using var zeros = np.array(new[] { 0.0, 0.0, 0.0 });
        using var relevance = np.array(new[] { 1.0, 0.0, 1.0 });
        Assert.AreEqual(0.0, RankingMetrics.RPrecision(empty));
        Assert.AreEqual(0.0, RankingMetrics.AveragePrecision(zeros));
        Assert.AreEqual(0.0, RankingMetrics.NdcgAtK(zeros, 3));
        Assert.AreEqual(0.0, RankingMetrics.DcgAtK(relevance, 0, 99));
        Assert.AreEqual(1.0, RankingMetrics.DcgAtK(relevance, -1));
        Assert.AreEqual(2.0 / 3.0, RankingMetrics.RPrecision(relevance));
        Assert.IsTrue(double.IsNaN(RankingMetrics.MeanAveragePrecision(Array.Empty<NDArray>())));
        Assert.IsTrue(double.IsNaN(RankingMetrics.MeanReciprocalRank(Array.Empty<NDArray>())));
        Assert.ThrowsException<ArgumentException>(() => RankingMetrics.PrecisionAtK(relevance, 4));
        Assert.ThrowsException<ArgumentOutOfRangeException>(() => RankingMetrics.PrecisionAtK(relevance, 0));
        Assert.ThrowsException<ArgumentException>(() => RankingMetrics.DcgAtK(relevance, 2, 99));
    }

    [TestMethod]
    public void RankingViews_AreNotSortedInPlace()
    {
        using var storage = np.array(new[] { 3.0, 99.0, 1.0, 99.0, 2.0, 99.0 });
        using var relevance = storage["::2"];
        _ = RankingMetrics.NdcgAtK(relevance, 3, 1);
        Assert.AreEqual(1.0, relevance.item<double>(1));
        Assert.AreEqual(99.0, storage.item<double>(1));
    }
}
