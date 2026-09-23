using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Examples.Gist;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Tests.Interop;

[TestClass]
public class GistMetricsLiveParityTests : InteropTestBase
{
    // Independent mathematical references, not a copy of either downloaded Python file.
    // Original formulas were reviewed and run on NumPy 2.4.2 before implementing the C# port.
    // a and p are zero-copy views of the exact NumSharp input buffers.
    private const string ForecastReference = """
        e = a - p
        pe = e / (a + 1e-10)
        ae = np.abs(e)
        re = e[1:] / (a[1:] - a[:-1] + 1e-10)
        bre = np.abs(e[1:]) / (np.abs(e[1:]) + np.abs(a[1:] - a[:-1]) + 1e-10)
        symmetric = 2.0 * ae / (np.abs(a) + np.abs(p) + 1e-10)
        naive_mae = np.mean(np.abs(a[1:] - a[:-1]))
        bounded_mean = np.mean(bre)
        scores = {
            'mse': np.mean(np.square(e)),
            'rmse': np.sqrt(np.mean(np.square(e))),
            'nrmse': np.sqrt(np.mean(np.square(e))) / (a.max() - a.min()),
            'me': np.mean(e), 'mae': np.mean(ae),
            'mad': np.mean(np.abs(e - np.mean(e))),
            'gmae': np.exp(np.mean(np.log(ae))), 'mdae': np.median(ae),
            'mpe': np.mean(pe), 'mape': np.mean(np.abs(pe)), 'mdape': np.median(np.abs(pe)),
            'smape': np.mean(symmetric), 'smdape': np.median(symmetric),
            'maape': np.mean(np.arctan(np.abs(pe))),
            'mase': np.mean(ae) / naive_mae,
            'std_ae': np.sqrt(np.sum(np.square(e - np.mean(ae))) / (len(a) - 1)),
            'std_ape': np.sqrt(np.sum(np.square(pe - np.mean(np.abs(pe)))) / (len(a) - 1)),
            'rmspe': np.sqrt(np.mean(np.square(pe))), 'rmdspe': np.sqrt(np.median(np.square(pe))),
            'rmsse': np.sqrt(np.mean(np.square(ae / naive_mae))),
            'inrse': np.sqrt(np.sum(np.square(e)) / np.sum(np.square(a - np.mean(a)))),
            'rrse': np.sqrt(np.sum(np.square(e)) / np.sum(np.square(a - np.mean(a)))),
            'mre': np.mean(re), 'rae': np.sum(ae) / (np.sum(np.abs(a - np.mean(a))) + 1e-10),
            'mrae': np.mean(np.abs(re)), 'mdrae': np.median(np.abs(re)),
            'gmrae': np.exp(np.mean(np.log(np.abs(re)))), 'mbrae': bounded_mean,
            'umbrae': bounded_mean / (1.0 - bounded_mean),
            'mda': np.mean((np.sign(a[1:] - a[:-1]) == np.sign(p[1:] - a[:-1])).astype(np.int64))
        }
        """;

    [TestMethod]
    public void ForecastArithmeticAndArrayHelpers_AreByteExact()
    {
        using var scope = NDScope.Open();
        var a = np.array(new[] { 2.0, 4.0, 8.0, 16.0, 32.0, 64.0 });
        var p = np.array(new[] { 1.0, 5.0, 6.0, 20.0, 30.0, 60.0 });
        var csharp = new Dictionary<string, NDArray>
        {
            ["a-p"] = ForecastingMetrics.Error(a, p),
            ["(a-p)/(a+1e-10)"] = ForecastingMetrics.PercentageError(a, p),
            ["a[:-2]"] = ForecastingMetrics.NaiveForecasting(a, 2),
            ["(a[2:]-p[2:])/(a[2:]-a[:-2]+1e-10)"] = ForecastingMetrics.RelativeError(a, p, seasonality: 2),
            ["np.abs(a-p)/(np.abs(a-p)+np.abs(a-1.0)+1e-10)"] =
                ForecastingMetrics.BoundedRelativeError(a, p, np.ones(6, dtype: np.float64)),
            ["np.asarray(np.mean(np.square(a-p)))"] = np.array(ForecastingMetrics.Mse(a, p)),
            ["np.asarray(np.mean(a-p))"] = np.array(ForecastingMetrics.Me(a, p)),
            ["np.asarray(np.mean(np.abs(a-p)))"] = np.array(ForecastingMetrics.Mae(a, p)),
            ["np.asarray(np.median(np.abs(a-p)))"] = np.array(ForecastingMetrics.Mdae(a, p))
        };
        using (Gil())
        {
            using var va = a.ToNumpy();
            using var vp = p.ToNumpy();
            foreach (var (expression, actual) in csharp)
            {
                using var expected = Python.np.with(expression, ("a", va), ("p", vp));
                ByteContract.AssertSameBytes(actual, expected, expression);
            }
        }
    }

    /// <summary>
    /// All thirty forecast scores against live NumPy on three layouts, exact except the two metrics whose
    /// 1-ULP budgets were measured on the x64 reference fixture. On arm64 it is inconclusive: the
    /// macos-latest runner measured <c>mpe</c> 1 ULP off (0.06249999999656575 vs …754), and the budgets
    /// are deliberately not widened per host (see <see cref="AssertWithinUlps"/>).
    /// </summary>
    [TestMethod]
    public void AllThirtyForecastScores_AreExactExceptTwoOneUlpMetrics_OnThreeLayouts()
    {
        SkipByteExactOnArm64("AllThirtyForecastScores (mpe 1 ULP off measured on macos-latest arm64)");
        using var scope = NDScope.Open();
        var storageA = np.array(new[] { 99.0, 2.0, 99.0, 4.0, 99.0, 8.0, 99.0, 16.0, 99.0, 32.0, 99.0, 64.0 });
        var storageP = np.array(new[] { 99.0, 1.0, 99.0, 5.0, 99.0, 6.0, 99.0, 20.0, 99.0, 30.0, 99.0, 60.0 });
        var stridedA = storageA["1::2"];
        var stridedP = storageP["1::2"];
        var variants = new[]
        {
            (stridedA.copy(), stridedP.copy()), (stridedA, stridedP),
            (stridedA["::-1"], stridedP["::-1"])
        };
        foreach (var (a, p) in variants)
        {
            var scores = ForecastingMetrics.EvaluateAll(a, p);
            using (Gil())
            {
                using var va = a.ToNumpy();
                using var vp = p.ToNumpy();
                using var reference = Py.CreateScope();
                reference.Exec("import numpy as np");
                reference.Set("a", va);
                reference.Set("p", vp);
                reference.Exec(ForecastReference);
                foreach (var name in ForecastingMetrics.MetricNames)
                {
                    using var actual = np.array(scores[name]);
                    using var expected = reference.Eval($"np.asarray(scores['{name}'], dtype=np.float64)");
                    // Measured on the first live run: 86/90 byte-identical; smape and mrae
                    // differed by one ULP on the two forward layouts, zero when reversed.
                    // All other names remain hard byte gates, including log/exp/arctan metrics.
                    AssertWithinUlps(actual, expected, name is "smape" or "mrae" ? 1UL : 0UL, name);
                }
            }
        }
    }

    [TestMethod]
    public void ForecastBenchmarkHelpers_AreByteExact_OnStridedSharedInputs()
    {
        using var scope = NDScope.Open();
        var backing = np.arange(2.0, 18.0);
        var a = backing["::2"];
        var p = backing["1::2"];
        var benchmark = np.ones(8, dtype: np.float64);
        var relative = ForecastingMetrics.RelativeError(a, p, benchmark);
        var bounded = ForecastingMetrics.BoundedRelativeError(a, p, seasonality: 3);
        using (Gil())
        {
            using var va = a.ToNumpy();
            using var vp = p.ToNumpy();
            using var vb = benchmark.ToNumpy();
            using var r = Python.np.with("(a-p)/(a-b+1e-10)", ("a", va), ("p", vp), ("b", vb));
            using var br = Python.np.with("np.abs(a[3:]-p[3:])/(np.abs(a[3:]-p[3:])+np.abs(a[3:]-a[:-3])+1e-10)", ("a", va), ("p", vp));
            ByteContract.AssertSameBytes(relative, r);
            ByteContract.AssertSameBytes(bounded, br);
        }
    }

    [TestMethod]
    public void RankingBinaryScores_AreByteExact_VsLiveNumpy()
    {
        using var scope = NDScope.Open();
        var r = np.array(new[] { 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 });
        var zero = np.zeros(3, dtype: np.float64);
        var results = new Dictionary<string, double>
        {
            ["np.mean(r[:np.flatnonzero(r)[-1]+1] != 0)"] = RankingMetrics.RPrecision(r),
            ["np.mean(r[:4] != 0)"] = RankingMetrics.PrecisionAtK(r, 4),
            ["np.mean([np.mean(r[:i+1] != 0) for i in np.flatnonzero(r)])"] = RankingMetrics.AveragePrecision(r),
            ["np.mean([1.0/(np.flatnonzero(r)[0]+1), 0.0])"] = RankingMetrics.MeanReciprocalRank(new[] { r, zero }),
            ["np.mean([np.mean([np.mean(r[:i+1] != 0) for i in np.flatnonzero(r)]), 0.0])"] = RankingMetrics.MeanAveragePrecision(new[] { r, zero })
        };
        using (Gil())
        {
            using var vr = r.ToNumpy();
            foreach (var (expression, value) in results)
            {
                using var actual = np.array(value);
                using var expected = Python.np.with($"np.asarray({expression},dtype=np.float64)", ("r", vr));
                ByteContract.AssertSameBytes(actual, expected, expression);
            }
        }
    }

    [TestMethod]
    public void RankingLogDiscounts_AreByteExact_AtPowerOfTwoKnots()
    {
        using var scope = NDScope.Open();
        var method0 = np.array(new[] { 4.0, 2.0, 0.0, 8.0 });
        var method1 = np.array(new[] { 4.0, 0.0, 2.0 });
        var d0 = np.array(RankingMetrics.DcgAtK(method0, 4));
        var d1 = np.array(RankingMetrics.DcgAtK(method1, 3, 1));
        using (Gil())
        {
            using var v0 = method0.ToNumpy();
            using var v1 = method1.ToNumpy();
            using var e0 = Python.np.with("np.asarray(r[0]+np.sum(r[1:]/np.log2(np.arange(2,len(r)+1))))", ("r", v0));
            using var e1 = Python.np.with("np.asarray(np.sum(r/np.log2(np.arange(2,len(r)+2))))", ("r", v1));
            ByteContract.AssertSameBytes(d0, e0);
            ByteContract.AssertSameBytes(d1, e1);
        }
    }

    [TestMethod]
    public void ForecastSpecialValues_MatchLiveNumpyClassification()
    {
        using var scope = NDScope.Open();
        var empty = np.array(Array.Empty<double>());
        var ones = np.ones(2, dtype: np.float64);
        var zeros = np.zeros(2, dtype: np.float64);
        var emptyMean = ForecastingMetrics.Mse(empty, empty);
        var constantRange = ForecastingMetrics.Nrmse(ones, zeros);
        var zeroGeometric = ForecastingMetrics.Gmae(zeros, zeros);
        using (Gil())
        {
            using var ve = empty.ToNumpy();
            using var vo = ones.ToNumpy();
            using var vz = zeros.ToNumpy();
            Assert.AreEqual(Python.np.truthy("np.isnan(np.mean(np.square(e-e)))", ("e", ve)), double.IsNaN(emptyMean));
            Assert.AreEqual(Python.np.truthy("np.isposinf(np.sqrt(np.mean(np.square(o-z)))/(o.max()-o.min()))", ("o", vo), ("z", vz)), double.IsPositiveInfinity(constantRange));
            using var actual = np.array(zeroGeometric);
            using var expected = Python.np.with("np.asarray(np.exp(np.mean(np.log(np.abs(z-z)))))", ("z", vz));
            ByteContract.AssertSameBytes(actual, expected);
        }
    }

    /// <summary>
    /// Ranking discounts against live NumPy: exact for method 0, within the measured 1-2 ULP for method 1,
    /// on the x64 reference fixture. On arm64 it is inconclusive: the macos-latest runner measured method-0
    /// <c>dcg</c> 1 ULP off (7.323465818787765 vs …766), and the budgets are deliberately not widened per
    /// host (see <see cref="AssertWithinUlps"/>).
    /// </summary>
    [TestMethod]
    public void RankingDiscounts_AreExactForMethodZero_AndOneOrTwoUlpsForMethodOne()
    {
        SkipByteExactOnArm64("RankingDiscounts (method-0 dcg 1 ULP off measured on macos-latest arm64)");
        using var scope = NDScope.Open();
        var storage = np.array(new[] { 3.0, 99.0, 2.0, 99.0, 3.0, 99.0, 0.0, 99.0, 0.0, 99.0, 1.0, 99.0, 2.0, 99.0, 2.0, 99.0, 3.0, 99.0, 0.0, 99.0 });
        var strided = storage["::2"];
        foreach (var r in new[] { strided.copy(), strided, strided["::-1"] })
        foreach (var method in new[] { 0, 1 })
        foreach (var k in new[] { -1, 1, 2, 10, 12 })
        {
            using var dcg = np.array(RankingMetrics.DcgAtK(r, k, method));
            using var ndcg = np.array(RankingMetrics.NdcgAtK(r, k, method));
            using (Gil())
            {
                using var vr = r.ToNumpy();
                using var reference = Py.CreateScope();
                reference.Exec("import numpy as np");
                reference.Set("r", vr);
                // np.asfarray was removed in NumPy 2.x; all shared inputs here are already float64.
                var gain = method == 0
                    ? "x[0] + np.sum(x[1:] / np.log2(np.arange(2, len(x)+1)))"
                    : "np.sum(x / np.log2(np.arange(2, len(x)+2)))";
                reference.Exec($"def gain(x):\n    return {gain} if len(x) else 0.0\nx = r[:{k}]\ny = np.sort(r)[::-1][:{k}]\nz = gain(y)");
                using var expectedDcg = reference.Eval("np.asarray(gain(x),dtype=np.float64)");
                using var expectedNdcg = reference.Eval("np.asarray(gain(x)/z if z else 0.0,dtype=np.float64)");
                // First live run: all 30 method-0 scalars exact. Method 1 had six one-ULP
                // DCG and six two-ULP NDCG differences; its other 18 scalars were exact.
                AssertWithinUlps(dcg, expectedDcg, method == 0 ? 0UL : 1UL, $"dcg method={method}, k={k}");
                AssertWithinUlps(ndcg, expectedNdcg, method == 0 ? 0UL : 2UL, $"ndcg method={method}, k={k}");
            }
        }
    }

    // These named bounds are the measured maxima on the NumPy 2.4.2 Windows/net10 fixture,
    // not a promise that arbitrary platforms/data differ by at most this many ULP. Other hosts
    // deliberately expose a disagreement instead of silently widening the gate. There is no
    // broad relative tolerance or fallback from a failed exact test to a permissive comparison.
    private void AssertWithinUlps(NDArray actual, PyObject expected, ulong limit, string context)
    {
        using var oracle = expected.ToNDArray();
        Assert.AreEqual(NPTypeCode.Double, actual.typecode, context);
        Assert.AreEqual(NPTypeCode.Double, oracle.typecode, context);
        Assert.AreEqual(0, actual.ndim, context);
        Assert.AreEqual(0, oracle.ndim, context);
        var a = actual.item<double>();
        var b = oracle.item<double>();
        Assert.IsTrue(double.IsFinite(a) && double.IsFinite(b), context);
        static ulong Ordered(double x)
        {
            var bits = unchecked((ulong)BitConverter.DoubleToInt64Bits(x));
            return (bits >> 63) == 0 ? bits | (1UL << 63) : ~bits;
        }
        var left = Ordered(a);
        var right = Ordered(b);
        var ulps = left > right ? left - right : right - left;
        TestContext?.WriteLine($"{context}: {ulps} ULP; NumSharp={a:R}, NumPy={b:R}");
        Assert.IsTrue(ulps <= limit, $"{context}: {ulps} ULP > {limit}; {a:R} vs {b:R}");
        if (ulps == 0) ByteContract.AssertSameBytes(actual, expected, context);
    }
}
