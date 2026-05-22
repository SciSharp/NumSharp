using System;
using AwesomeAssertions;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.UnitTest.Statistics;

/// <summary>
/// Battle tests for np.percentile and np.quantile, comparing against NumPy 2.4.2.
/// Validates: scalar/array q, all 11 NumPy methods, axis variants, dtype rules,
/// NaN propagation, q-range errors. The 'inverted_cdf' / 'closest_observation'
/// edge cases are explicitly covered because their gamma formulas diverge from
/// the standard linear/midpoint patterns.
/// </summary>
[TestClass]
public class np_percentile_BattleTests
{
    private static double At(NDArray nd, int i) => Convert.ToDouble(nd.GetAtIndex(i));
    private static bool Near(double got, double want, double tol = 1e-9)
        => Math.Abs(got - want) < tol || (double.IsNaN(got) && double.IsNaN(want));

    // ── basic ───────────────────────────────────────────────────────────

    [TestMethod]
    public void Percentile_q50_IsMedian()
    {
        var a = np.array(new long[] { 3, 1, 4, 1, 5, 9, 2, 6 });
        At(np.percentile(a, 50), 0).Should().Be(3.5);
    }

    [TestMethod]
    public void Percentile_q25()
    {
        var a = np.array(new long[] { 3, 1, 4, 1, 5, 9, 2, 6 });
        At(np.percentile(a, 25), 0).Should().BeApproximately(1.75, 1e-9);
    }

    [TestMethod]
    public void Percentile_q75()
    {
        var a = np.array(new long[] { 3, 1, 4, 1, 5, 9, 2, 6 });
        At(np.percentile(a, 75), 0).Should().BeApproximately(5.25, 1e-9);
    }

    [TestMethod]
    public void Percentile_q0_ReturnsMin()
    {
        var a = np.array(new long[] { 3, 1, 4, 1, 5, 9, 2, 6 });
        At(np.percentile(a, 0), 0).Should().Be(1.0);
    }

    [TestMethod]
    public void Percentile_q100_ReturnsMax()
    {
        var a = np.array(new long[] { 3, 1, 4, 1, 5, 9, 2, 6 });
        At(np.percentile(a, 100), 0).Should().Be(9.0);
    }

    [TestMethod]
    public void Quantile_q05_EqualsPercentile_q50()
    {
        var a = np.array(new long[] { 3, 1, 4, 1, 5, 9, 2, 6 });
        At(np.quantile(a, 0.5), 0).Should().Be(At(np.percentile(a, 50), 0));
    }

    // ── array q ─────────────────────────────────────────────────────────

    [TestMethod]
    public void Percentile_ArrayQ_PrependsAxis()
    {
        var a = np.array(new long[] { 3, 1, 4, 1, 5, 9, 2, 6 });
        var p = np.percentile(a, new double[] { 25, 50, 75 });
        p.shape.Should().Equal(new long[] { 3 });
        At(p, 0).Should().BeApproximately(1.75, 1e-9);
        At(p, 1).Should().BeApproximately(3.5, 1e-9);
        At(p, 2).Should().BeApproximately(5.25, 1e-9);
    }

    [TestMethod]
    public void Percentile_ArrayQ_With2D_ShapeIsQ_plus_remaining()
    {
        var a = np.array(new long[,] { { 10, 7, 4 }, { 3, 2, 1 } });
        // Expected: shape (3, 3) — q axis prepended.
        var p = np.percentile(a, new double[] { 25, 50, 75 }, axis: 0);
        p.shape.Should().Equal(new long[] { 3, 3 });
        // NumPy: [[4.75,3.25,1.75],[6.5,4.5,2.5],[8.25,5.75,3.25]]
        At(p, 0).Should().BeApproximately(4.75, 1e-9);
        At(p, 1).Should().BeApproximately(3.25, 1e-9);
        At(p, 2).Should().BeApproximately(1.75, 1e-9);
        At(p, 3).Should().BeApproximately(6.5, 1e-9);
        At(p, 6).Should().BeApproximately(8.25, 1e-9);
    }

    [TestMethod]
    public void Percentile_NDArrayQ_ScalarShape()
    {
        var a = np.array(new long[] { 3, 1, 4, 1, 5, 9, 2, 6 });
        var p = np.percentile(a, NDArray.Scalar(50.0));
        p.shape.Should().BeEmpty(); // 0-D q stays 0-D
        At(p, 0).Should().Be(3.5);
    }

    [TestMethod]
    public void Percentile_NDArrayQ_1DPreservesQAxis()
    {
        var a = np.array(new long[] { 3, 1, 4, 1, 5, 9, 2, 6 });
        var p = np.percentile(a, np.array(new double[] { 25, 50, 75 }));
        p.shape.Should().Equal(new long[] { 3 });
    }

    [TestMethod]
    public void Percentile_NDArrayQ_HigherRank_Throws()
    {
        var a = np.array(new long[] { 1, 2, 3 });
        Action act = () => np.percentile(a, np.array(new double[,] { { 25, 50 }, { 75, 100 } }));
        act.Should().Throw<ArgumentException>();
    }

    // ── methods ─────────────────────────────────────────────────────────

    // All 13 methods, reference values from numpy 2.4.2 with q=0.5 on arange(11).
    [DataTestMethod]
    [DataRow("linear", 5.0)]
    [DataRow("lower", 5.0)]
    [DataRow("higher", 5.0)]
    [DataRow("nearest", 5.0)]
    [DataRow("midpoint", 5.0)]
    [DataRow("inverted_cdf", 5.0)]
    [DataRow("averaged_inverted_cdf", 5.0)]
    [DataRow("closest_observation", 5.0)]
    [DataRow("interpolated_inverted_cdf", 4.5)]
    [DataRow("hazen", 5.0)]
    [DataRow("weibull", 5.0)]
    [DataRow("median_unbiased", 5.0)]
    [DataRow("normal_unbiased", 5.0)]
    public void Quantile_AllMethods_q05_OnArange11(string method, double expected)
    {
        var a = np.arange(11);
        var v = np.quantile(a, 0.5, method: method);
        Near(At(v, 0), expected).Should().BeTrue($"method={method} got={At(v,0)} want={expected}");
    }

    // q=0.1 on arr_int = [0,5,...,60] (13 elements). Reference values verified
    // against numpy 2.4.2 — each method's distinct formula manifests here.
    [DataTestMethod]
    [DataRow("linear", 6.000000000000001)]
    [DataRow("lower", 5.0)]
    [DataRow("higher", 10.0)]
    [DataRow("nearest", 5.0)]
    [DataRow("midpoint", 7.5)]
    [DataRow("inverted_cdf", 5.0)]
    [DataRow("averaged_inverted_cdf", 5.0)]
    [DataRow("closest_observation", 0.0)]
    [DataRow("interpolated_inverted_cdf", 1.5)]
    [DataRow("hazen", 4.0)]
    [DataRow("weibull", 2.0)]
    [DataRow("median_unbiased", 3.333333333333334)]
    [DataRow("normal_unbiased", 3.500000000000001)]
    public void Quantile_AllMethods_q010_OnArrInt13(string method, double expected)
    {
        var arr = new int[] { 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60 };
        var a = np.array(arr);
        var v = np.quantile(a, 0.1, method: method);
        Near(At(v, 0), expected).Should().BeTrue($"method={method} got={At(v,0)} want={expected}");
    }

    [TestMethod]
    public void Quantile_UnknownMethod_Throws()
    {
        var a = np.arange(11);
        Action act = () => np.quantile(a, 0.5, method: "bogus");
        act.Should().Throw<ArgumentException>();
    }

    // ── dtype rules ─────────────────────────────────────────────────────

    [TestMethod]
    public void Percentile_Int32_Linear_PromotesToFloat64()
    {
        var a = np.array(new int[] { 1, 2, 3, 4, 5 });
        var p = np.percentile(a, 50);
        p.typecode.Should().Be(NPTypeCode.Double);
    }

    [TestMethod]
    public void Percentile_Int32_LowerMethod_PreservesIntegerDtype()
    {
        var a = np.array(new int[] { 1, 2, 3, 4, 5 });
        var p = np.percentile(a, 50, method: "lower");
        p.typecode.Should().Be(NPTypeCode.Int32);
        p.GetAtIndex<int>(0).Should().Be(3);
    }

    [TestMethod]
    public void Percentile_Int32_HigherMethod_PreservesIntegerDtype()
    {
        var a = np.array(new int[] { 1, 2, 3, 4, 5 });
        var p = np.percentile(a, 50, method: "higher");
        p.typecode.Should().Be(NPTypeCode.Int32);
        p.GetAtIndex<int>(0).Should().Be(3);
    }

    [TestMethod]
    public void Percentile_Float32_Linear_PreservesFloat32()
    {
        var a = np.array(new float[] { 1, 2, 3, 4, 5 });
        var p = np.percentile(a, 50);
        p.typecode.Should().Be(NPTypeCode.Single);
        p.GetAtIndex<float>(0).Should().Be(3.0f);
    }

    // ── range validation ────────────────────────────────────────────────

    [TestMethod]
    public void Percentile_Q_OutOfRange_Throws()
    {
        var a = np.array(new int[] { 1, 2, 3 });
        Action lo = () => np.percentile(a, -1);
        Action hi = () => np.percentile(a, 150);
        lo.Should().Throw<ArgumentException>();
        hi.Should().Throw<ArgumentException>();
    }

    [TestMethod]
    public void Quantile_Q_OutOfRange_Throws()
    {
        var a = np.array(new int[] { 1, 2, 3 });
        Action lo = () => np.quantile(a, -0.1);
        Action hi = () => np.quantile(a, 1.5);
        lo.Should().Throw<ArgumentException>();
        hi.Should().Throw<ArgumentException>();
    }

    [TestMethod]
    public void Percentile_ArrayQ_OneOutOfRange_Throws()
    {
        var a = np.array(new int[] { 1, 2, 3 });
        Action act = () => np.percentile(a, new double[] { 25, 50, 150 });
        act.Should().Throw<ArgumentException>();
    }

    // ── NaN propagation (any NaN in slice → NaN result) ─────────────────

    [TestMethod]
    public void Percentile_NaN_Propagates_AcrossAllQ()
    {
        var a = np.array(new double[] { 1, 2, double.NaN, 4 });
        foreach (double q in new[] { 0.0, 25.0, 50.0, 75.0, 100.0 })
        {
            double v = At(np.percentile(a, q), 0);
            double.IsNaN(v).Should().BeTrue($"q={q}");
        }
    }

    // ── axis variants ───────────────────────────────────────────────────

    [TestMethod]
    public void Percentile_3D_TupleAxis()
    {
        var a = np.arange(24).reshape(2, 3, 4);
        // np.percentile(a, 50, axis=(0,2)) -> [7.5, 11.5, 15.5]
        var p = np.percentile(a, 50, axis: new[] { 0, 2 });
        p.shape.Should().Equal(new long[] { 3 });
        At(p, 0).Should().BeApproximately(7.5, 1e-9);
        At(p, 1).Should().BeApproximately(11.5, 1e-9);
        At(p, 2).Should().BeApproximately(15.5, 1e-9);
    }

    [TestMethod]
    public void Percentile_ArrayQ_Keepdims()
    {
        var a = np.array(new long[,] { { 10, 7, 4 }, { 3, 2, 1 } });
        var p = np.percentile(a, new double[] { 25, 50 }, axis: 0, keepdims: true);
        // shape: (2, 1, 3) — q prepended, reduced axis kept as 1
        p.shape.Should().Equal(new long[] { 2, 1, 3 });
    }

    // ── 0-D + 1-element edge cases ──────────────────────────────────────

    [TestMethod]
    public void Percentile_ZeroD_Scalar_ReturnsValue()
    {
        var s = NDArray.Scalar(5.0);
        var p = np.percentile(s, 50);
        At(p, 0).Should().Be(5.0);
    }

    [TestMethod]
    public void Percentile_SingleElement_ReturnsThatElement()
    {
        var a = np.array(new double[] { 42.0 });
        var p = np.percentile(a, 50);
        At(p, 0).Should().Be(42.0);
    }
}
