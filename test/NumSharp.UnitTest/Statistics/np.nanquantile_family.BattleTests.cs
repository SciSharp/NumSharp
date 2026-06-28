using System;
using AwesomeAssertions;
using NumSharp;

namespace NumSharp.UnitTest.Statistics;

/// <summary>
/// Battle tests for np.nanmedian / np.nanquantile / np.nanpercentile and the
/// quantile-family output-dtype rules, all verified against NumPy 2.4.2.
///
/// Covers (all values from actual NumPy 2.4.2 output):
///   * NaN-ignoring median/quantile/percentile (per-slice NaN removal).
///   * All-NaN slice -> NaN.
///   * Integer input (no NaN) == plain variant.
///   * Empty input -> nanmean short-circuit (q dimension dropped, nanmean dtype).
///   * Output-dtype rule: float32/float16 + continuous + array-q -> float64,
///     but scalar-q preserves the float width (NEP50 weak promotion); discrete
///     methods preserve the input dtype; median keeps the float width.
///   * Boolean input: continuous methods raise (bool subtract unsupported),
///     discrete methods return bool, median coerces to float64.
/// </summary>
[TestClass]
public class np_nanquantile_family_BattleTests
{
    private static double At(NDArray nd, int i) => nd.GetAtIndex(i) is Half h ? (double)h : Convert.ToDouble(nd.GetAtIndex(i));
    private static bool Near(double got, double want, double tol = 1e-9)
        => Math.Abs(got - want) < tol || (double.IsNaN(got) && double.IsNaN(want));

    // ── nanmedian ─────────────────────────────────────────────────────────

    [TestMethod]
    public void NanMedian_1D_IgnoresNaN()
    {
        // NumPy: np.nanmedian([10,nan,4,3,2,1]) -> 3.0  (median of [10,4,3,2,1])
        var a = np.array(new[] { 10.0, double.NaN, 4, 3, 2, 1 });
        var m = np.nanmedian(a);
        m.typecode.Should().Be(NPTypeCode.Double);
        At(m, 0).Should().Be(3.0);
    }

    [TestMethod]
    public void NanMedian_2D_Axis0()
    {
        // NumPy: np.nanmedian([[10,nan,4],[3,2,1]], axis=0) -> [6.5, 2.0, 2.5]
        var a = np.array(new[,] { { 10.0, double.NaN, 4 }, { 3, 2, 1 } });
        var m = np.nanmedian(a, axis: 0);
        m.shape.Should().Equal(new long[] { 3 });
        At(m, 0).Should().Be(6.5);
        At(m, 1).Should().Be(2.0);
        At(m, 2).Should().Be(2.5);
    }

    [TestMethod]
    public void NanMedian_2D_Axis1()
    {
        // NumPy: np.nanmedian([[10,nan,4],[3,2,1]], axis=1) -> [7.0, 2.0]
        var a = np.array(new[,] { { 10.0, double.NaN, 4 }, { 3, 2, 1 } });
        var m = np.nanmedian(a, axis: 1);
        m.shape.Should().Equal(new long[] { 2 });
        At(m, 0).Should().Be(7.0);
        At(m, 1).Should().Be(2.0);
    }

    [TestMethod]
    public void NanMedian_AllNaNSlice_ReturnsNaN()
    {
        // NumPy: np.nanmedian([[nan,nan],[1,3]], axis=1) -> [nan, 2.0]
        var a = np.array(new[,] { { double.NaN, double.NaN }, { 1.0, 3.0 } });
        var m = np.nanmedian(a, axis: 1);
        double.IsNaN(At(m, 0)).Should().BeTrue();
        At(m, 1).Should().Be(2.0);
    }

    [TestMethod]
    public void NanMedian_Keepdims()
    {
        // NumPy: np.nanmedian([[10,nan,4],[3,2,1]], axis=1, keepdims=True) -> [[7.0],[2.0]]
        var a = np.array(new[,] { { 10.0, double.NaN, 4 }, { 3, 2, 1 } });
        var m = np.nanmedian(a, axis: 1, keepdims: true);
        m.shape.Should().Equal(new long[] { 2, 1 });
        At(m, 0).Should().Be(7.0);
        At(m, 1).Should().Be(2.0);
    }

    [TestMethod]
    public void NanMedian_IntInput_NoNaN_EqualsMedian()
    {
        // Integer input can't carry NaN -> identical to np.median, float64.
        var a = np.array(new[] { 3, 1, 4, 1, 5, 9, 2 });
        var m = np.nanmedian(a);
        m.typecode.Should().Be(NPTypeCode.Double);
        At(m, 0).Should().Be(3.0);
    }

    // ── nanquantile / nanpercentile ─────────────────────────────────────────

    [TestMethod]
    public void NanQuantile_2D_Axis0_ArrayQ()
    {
        // NumPy: np.nanquantile([[10,nan,4],[3,2,1]], [.25,.5,.75], axis=0)
        //   -> [[4.75,2,1.75],[6.5,2,2.5],[8.25,2,3.25]]
        var a = np.array(new[,] { { 10.0, double.NaN, 4 }, { 3, 2, 1 } });
        var r = np.nanquantile(a, new[] { 0.25, 0.5, 0.75 }, axis: 0);
        r.shape.Should().Equal(new long[] { 3, 3 });
        double[] want = { 4.75, 2.0, 1.75, 6.5, 2.0, 2.5, 8.25, 2.0, 3.25 };
        for (int i = 0; i < want.Length; i++) Near(At(r, i), want[i]).Should().BeTrue($"idx {i}");
    }

    [TestMethod]
    public void NanQuantile_MethodVariants()
    {
        var a = np.array(new[] { 10.0, double.NaN, 4, 3, 2, 1 });
        // both reduce to median of [10,4,3,2,1] = 3.0
        At(np.nanquantile(a, 0.5, method: "median_unbiased"), 0).Should().Be(3.0);
        At(np.nanquantile(a, 0.5, method: "lower"), 0).Should().Be(3.0);
    }

    [TestMethod]
    public void NanPercentile_1D()
    {
        // NumPy: np.nanpercentile([10,nan,4,3,2,1], [25,75]) -> [2.0, 4.0]
        var a = np.array(new[] { 10.0, double.NaN, 4, 3, 2, 1 });
        var r = np.nanpercentile(a, new[] { 25.0, 75.0 });
        r.shape.Should().Equal(new long[] { 2 });
        At(r, 0).Should().Be(2.0);
        At(r, 1).Should().Be(4.0);
    }

    // ── empty input -> nanmean short-circuit ────────────────────────────────

    [TestMethod]
    public void NanMedian_Empty_ReturnsNaNScalar()
    {
        // NumPy: np.nanmedian([]) -> nan (float64 scalar)
        var r = np.nanmedian(np.array(new double[0]));
        r.typecode.Should().Be(NPTypeCode.Double);
        double.IsNaN(At(r, 0)).Should().BeTrue();
    }

    [TestMethod]
    public void NanQuantile_Empty_ArrayQ_DropsQDimension()
    {
        // NumPy artifact: empty short-circuits to nanmean, which DROPS the q axis.
        // np.nanquantile([], [.25,.75]) -> nan (scalar), NOT shape (2,).
        var r = np.nanquantile(np.array(new double[0]), new[] { 0.25, 0.75 });
        r.size.Should().Be(1);
        double.IsNaN(At(r, 0)).Should().BeTrue();
    }

    [TestMethod]
    public void NanQuantile_EmptyInt_PromotesToFloat64()
    {
        // NumPy: np.nanquantile(np.array([],int32), 0.5, method='lower') -> nan float64
        // (empty -> nanmean(int)->float64, even for a discrete method).
        var r = np.nanquantile(np.array(new int[0]), 0.5, method: "lower");
        r.typecode.Should().Be(NPTypeCode.Double);
        double.IsNaN(At(r, 0)).Should().BeTrue();
    }

    // ── output-dtype rule (quantile/percentile) ─────────────────────────────

    [TestMethod]
    public void Quantile_Float32_ArrayQ_PromotesToFloat64()
    {
        // NumPy: np.quantile(float32, [0.5]) -> float64 (strong gamma upcasts).
        var a = np.array(new[] { 3f, 1f, 4f, 1f, 5f });
        var r = np.quantile(a, new[] { 0.5 });
        r.typecode.Should().Be(NPTypeCode.Double);
        At(r, 0).Should().Be(3.0);
    }

    [TestMethod]
    public void Quantile_Float32_ScalarQ_PreservesFloat32()
    {
        // NumPy: np.quantile(float32, 0.5) -> float32 (NEP50 weak gamma).
        var a = np.array(new[] { 3f, 1f, 4f, 1f, 5f });
        var r = np.quantile(a, 0.5);
        r.typecode.Should().Be(NPTypeCode.Single);
        At(r, 0).Should().Be(3.0);
    }

    [TestMethod]
    public void Quantile_Float32_Discrete_PreservesFloat32()
    {
        // NumPy: np.quantile(float32, 0.5, method='lower') -> float32.
        var a = np.array(new[] { 3f, 1f, 4f, 1f, 5f });
        var r = np.quantile(a, 0.5, method: "lower");
        r.typecode.Should().Be(NPTypeCode.Single);
    }

    [TestMethod]
    public void Quantile_Int32_Linear_ToFloat64_Discrete_PreservesInt()
    {
        var a = np.array(new[] { 3, 1, 4, 1, 5 });
        np.quantile(a, 0.5).typecode.Should().Be(NPTypeCode.Double);            // linear -> float64
        np.quantile(a, 0.5, method: "lower").typecode.Should().Be(NPTypeCode.Int32);  // discrete -> int32
    }

    [TestMethod]
    public void Median_Float32_PreservesFloat32()
    {
        // NumPy: np.median(float32) -> float32 (coerces via mean, not lerp).
        var a = np.array(new[] { 3f, 1f, 4f, 1f, 5f });
        var m = np.median(a);
        m.typecode.Should().Be(NPTypeCode.Single);
        At(m, 0).Should().Be(3.0);
    }

    // ── boolean input ───────────────────────────────────────────────────────

    [TestMethod]
    public void Quantile_Bool_Continuous_Throws()
    {
        // NumPy raises TypeError (boolean subtract unsupported in the lerp).
        var b = np.array(new[] { true, false, true, true, false });
        Action act = () => np.quantile(b, 0.5);
        act.Should().Throw<Exception>();
    }

    [TestMethod]
    public void Quantile_Bool_Discrete_ReturnsBool()
    {
        // NumPy: np.quantile(bool, 0.5, method='lower') -> bool (True here).
        var b = np.array(new[] { true, false, true, true, false });
        var r = np.quantile(b, 0.5, method: "lower");
        r.typecode.Should().Be(NPTypeCode.Boolean);
    }

    [TestMethod]
    public void Median_Bool_CoercesToFloat64()
    {
        // NumPy: np.median(bool) -> float64 (1.0 here). median uses mean, not lerp.
        var b = np.array(new[] { true, false, true, true, false });
        var m = np.median(b);
        m.typecode.Should().Be(NPTypeCode.Double);
        At(m, 0).Should().Be(1.0);
    }

    // ── strided / transposed input layout ───────────────────────────────────

    [TestMethod]
    public void NanQuantile_TransposedInput()
    {
        // Transposed (non-contiguous) input must stage correctly.
        // base [[10,nan,4],[3,2,1]] ; .T is (3,2); nanmedian over axis=1 of .T
        // equals nanmedian of base over axis=0 -> [6.5, 2.0, 2.5].
        var a = np.array(new[,] { { 10.0, double.NaN, 4 }, { 3, 2, 1 } });
        var t = a.T;                          // shape (3,2), strided
        var m = np.nanmedian(t, axis: 1);
        m.shape.Should().Equal(new long[] { 3 });
        At(m, 0).Should().Be(6.5);
        At(m, 1).Should().Be(2.0);
        At(m, 2).Should().Be(2.5);
    }
}
