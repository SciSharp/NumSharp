using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Examples.Gist;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Tests.Interop;

// The original doctest inputs, independently expressed with live NumPy over C# buffers.
// This deliberately does not execute/import the downloaded gist or its doctest module.
[TestClass]
public class GistMetricsSourceDemonstrationsLiveTests : InteropTestBase
{
    [TestMethod]
    public void RankingSource_Mrr_OriginalThreeInputs_ByteExact()
    {
        using var scope = NDScope.Open();
        var one = np.array(new[] { 1.0, 0.0, 0.0 });
        var two = np.array(new[] { 0.0, 1.0, 0.0 });
        var three = np.array(new[] { 0.0, 0.0, 1.0 });
        var four = np.array(new[] { 0.0, 0.0, 0.0, 1.0 });
        var matrix = np.array(new double[,] { { 0, 0, 0 }, { 0, 1, 0 }, { 1, 0, 0 } });
        CheckMrr(new[] { three, two, one }, "rank-L20");
        CheckMrr(new[] { matrix["0,:"], matrix["1,:"], matrix["2,:"] }, "rank-L23");
        CheckMrr(new[] { four, one, one }, "rank-L26");
    }

    private void CheckMrr(NDArray[] queries, string caseId)
    {
        using var actual = np.array(RankingMetrics.MeanReciprocalRank(queries));
        using (Gil())
        {
            using var reference = Py.CreateScope();
            reference.Exec("import numpy as np");
            for (var i = 0; i < queries.Length; ++i)
            {
                using var view = queries[i].ToNumpy();
                reference.Set($"q{i}", view);
            }
            var names = string.Join(",", Enumerable.Range(0, queries.Length).Select(i => $"q{i}"));
            using var expected = reference.Eval($"np.asarray(np.mean([1.0/(np.flatnonzero(q)[0]+1) if np.any(q) else 0.0 for q in ({names})]),dtype=np.float64)");
            GistParity.AssertExact(actual, expected, caseId);
        }
    }

    [TestMethod]
    public void RankingSource_BinaryDoctests_AndAreaMeasurement()
    {
        using var scope = NDScope.Open();
        var right = np.array(new[] { 0.0, 0.0, 1.0 });
        var middle = np.array(new[] { 0.0, 1.0, 0.0 });
        var left = np.array(new[] { 1.0, 0.0, 0.0 });
        foreach (var (r, id) in new[] { (right, "rank-L46"), (middle, "rank-L49"), (left, "rank-L52") })
            CheckScalar(RankingMetrics.RPrecision(r), r, "np.mean(r[:np.flatnonzero(r)[-1]+1]!=0)", id);
        foreach (var (k, id) in new[] { (1, "rank-L75"), (2, "rank-L77"), (3, "rank-L79") })
            CheckScalar(RankingMetrics.PrecisionAtK(right, k), right, $"np.mean(r[:{k}]!=0)", id);

        var relevance = np.array(new[] { 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 });
        var zero = np.array(new[] { 0.0 });
        const string averagePrecision = "np.mean([np.mean(r[:i+1]!=0) for i in np.flatnonzero(r)])";
        CheckScalar(RankingMetrics.AveragePrecision(relevance), relevance, averagePrecision, "rank-L113");
        CheckScalar(RankingMetrics.MeanAveragePrecision(new[] { relevance }), relevance, $"np.mean([{averagePrecision}])", "rank-L136");
        CheckScalar(RankingMetrics.MeanAveragePrecision(new[] { relevance, zero }), relevance, $"np.mean([{averagePrecision},0.0])", "rank-L139");

        using var actual = np.array(RankingMetrics.AveragePrecision(relevance));
        using (Gil())
        {
            using var vr = relevance.ToNumpy();
            // The source demonstration uses a Python list, so retain built-in Python float
            // summation, not NumPy scalar summation. Python 3.12 improved built-in sum accuracy.
            using var reference = Py.CreateScope();
            reference.Exec("import numpy as np");
            reference.Set("r", vr);
            reference.Exec("values=r.tolist()\ndelta=1.0/sum(values)\narea=sum([sum(values[:i+1])/(i+1.0)*delta for i,v in enumerate(values) if v])");
            using var area = reference.Eval("np.asarray(area,dtype=np.float64)");
            GistParity.AssertUlps(actual, area, 1, "rank-L111 independent PR-area versus average_precision; measured 1 ULP on Python 3.12");
        }
    }

    [TestMethod]
    public void RankingSource_PrecisionAtK_OriginalValueError()
    {
        using var r = np.array(new[] { 0.0, 0.0, 1.0 });
        var csharpError = Assert.ThrowsException<ArgumentException>(() => RankingMetrics.PrecisionAtK(r, 4));
        StringAssert.StartsWith(csharpError.Message, "Relevance score length < k");
        using (Gil())
        {
            using var vr = r.ToNumpy();
            using var reference = Py.CreateScope();
            reference.Exec("import numpy as np");
            reference.Set("r", vr);
            reference.Exec("def precision_at_four(r):\n    selected=np.asarray(r)[:4]!=0\n    if selected.size!=4:\n        raise ValueError('Relevance score length < k')\n    return np.mean(selected)");
            var pythonError = Assert.ThrowsException<PythonException>(() =>
            {
                using var unexpected = reference.Eval("precision_at_four(r)");
            });
            StringAssert.Contains(pythonError.Message, "Relevance score length < k");
        }
    }

    [TestMethod]
    public void RankingSource_AllDcgAndNdcgDemonstrations_ByteExact()
    {
        using var scope = NDScope.Open();
        var g = np.array(new[] { 3.0, 2.0, 3.0, 0.0, 0.0, 1.0, 2.0, 2.0, 3.0, 0.0 });
        foreach (var (k, method, id) in new[]
        {
            (1,0,"rank-L161"), (1,1,"rank-L163"), (2,0,"rank-L165"),
            (2,1,"rank-L167"), (10,0,"rank-L169"), (11,0,"rank-L171")
        })
            CheckGain(g, k, method, false, id);
        var mixed = np.array(new[] { 2.0, 1.0, 2.0, 0.0 });
        CheckGain(g, 1, 0, true, "rank-L204");
        CheckGain(mixed, 4, 0, true, "rank-L207");
        CheckGain(mixed, 4, 1, true, "rank-L209");
        CheckGain(np.array(new[] { 0.0 }), 1, 0, true, "rank-L211");
        CheckGain(np.array(new[] { 1.0 }), 2, 0, true, "rank-L213");
    }

    private void CheckGain(NDArray r, int k, int method, bool normalized, string caseId)
    {
        using var actual = np.array(normalized ? RankingMetrics.NdcgAtK(r, k, method) : RankingMetrics.DcgAtK(r, k, method));
        using (Gil())
        {
            using var vr = r.ToNumpy();
            using var reference = Py.CreateScope();
            reference.Exec("import numpy as np");
            reference.Set("r", vr);
            var formula = method == 0 ? "x[0]+np.sum(x[1:]/np.log2(np.arange(2,len(x)+1)))"
                : "np.sum(x/np.log2(np.arange(2,len(x)+2)))";
            reference.Exec($"def gain(x):\n    return {formula} if len(x) else 0.0\nx=r[:{k}]\ny=np.sort(r)[::-1][:{k}]\nz=gain(y)");
            using var expected = reference.Eval(normalized
                ? "np.asarray(gain(x)/z if z else 0.0,dtype=np.float64)"
                : "np.asarray(gain(x),dtype=np.float64)");
            GistParity.AssertExact(actual, expected, caseId);
        }
    }

    private void CheckScalar(double value, NDArray input, string expression, string caseId)
    {
        using var actual = np.array(value);
        using (Gil())
        {
            using var view = input.ToNumpy();
            using var expected = Python.np.with($"np.asarray({expression},dtype=np.float64)", ("r", view));
            GistParity.AssertExact(actual, expected, caseId);
        }
    }
}
