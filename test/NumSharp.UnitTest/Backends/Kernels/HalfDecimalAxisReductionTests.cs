using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.Backends.Kernels;

/// <summary>
/// Pins the NpyIter-routed Half and Decimal axis reductions:
///   - Half MEAN (accumulates in Double, then casts back to Half).
///   - Decimal sum/prod/min/max/mean (full-precision, on contiguous stripes).
/// Half sum/prod/min/max stay on the legacy path (their f16 sequential accumulator
/// can't be SIMD'd) and are covered elsewhere; a couple are checked here for sanity.
/// Half expected values are from NumPy 2.4.2 float16.
/// </summary>
[TestClass]
public class HalfDecimalAxisReductionTests
{
    // b[r,c] = (r*4+c) % 7, as float16.
    private static NDArray HalfB() =>
        (np.arange(12).astype(NPTypeCode.Double).reshape(3, 4) % 7).astype(NPTypeCode.Half);

    private static float HF(NDArray a, long i) => (float)(Half)a.GetAtIndex(i);

    private static void AssertHalf(NDArray r, float[] expected, string ctx)
    {
        Assert.AreEqual(expected.Length, (int)r.size, $"{ctx}: size");
        for (long i = 0; i < expected.Length; i++)
        {
            float g = HF(r, i);
            Assert.IsTrue(Math.Abs(g - expected[i]) <= 0.02f * (1 + Math.Abs(expected[i])),
                $"{ctx}[{i}]: expected {expected[i]} got {g}");
        }
    }

    [TestMethod]
    public void Half_Mean_Axis0_Axis1_NpyIterPath()
    {
        var b = HalfB();
        // NumPy: mean axis0 = [1.667, 2.666, 3.666, 2.334]; axis1 = [1.5, 3.75, 2.5]
        AssertHalf(np.mean(b, axis: 0), new[] { 1.667f, 2.666f, 3.666f, 2.334f }, "half mean axis0");
        AssertHalf(np.mean(b, axis: 1), new[] { 1.5f, 3.75f, 2.5f }, "half mean axis1");
    }

    [TestMethod]
    public void Half_Mean_Keepdims()
    {
        var b = HalfB();
        var r = np.mean(b, axis: 1, keepdims: true);
        Assert.AreEqual(2, r.ndim);
        Assert.AreEqual(3, (int)r.shape[0]);
        Assert.AreEqual(1, (int)r.shape[1]);
        AssertHalf(r, new[] { 1.5f, 3.75f, 2.5f }, "half mean axis1 keepdims");
    }

    [TestMethod]
    public void Half_Sum_AccumulatesInFloat32_NotFloat16()
    {
        // NumPy accumulates float16 reductions in float32, NOT float16. np.sum(ones(4096,f16))
        // == 4096 — an f16 accumulator would saturate at ~2048 (2048 + 1 == 2048 in float16).
        // NumSharp routes Half sum through a Double accumulator and casts back, matching NumPy.
        var b = HalfB();
        AssertHalf(np.sum(b, axis: 0), new[] { 5f, 8f, 11f, 7f }, "half sum axis0");
        var ones = np.ones(new Shape(4096, 2), NPTypeCode.Half);
        AssertHalf(np.sum(ones, axis: 0), new[] { 4096f, 4096f }, "half sum 4096 ones (float32 accumulate)");
    }

    // ---- Decimal (full-precision; no NumPy reference type) ----

    private static NDArray DecB()
    {
        // 3x4 decimals with a fractional part that a double-bridge would round.
        var d = new decimal[12];
        for (int i = 0; i < 12; i++) d[i] = i + 0.001m * i;
        return np.array(d).reshape(3, 4);
    }

    private static decimal DF(NDArray a, long i) => (decimal)a.GetAtIndex(i);

    [TestMethod]
    public void Decimal_Sum_Axis0_Axis1()
    {
        var b = DecB();
        // column sums of [[0,1.001,2.002,3.003],[4.004,5.005,6.006,7.007],[8.008,9.009,10.01,11.011]]
        Assert.AreEqual(12.012m, DF(np.sum(b, axis: 0), 0));
        Assert.AreEqual(15.015m, DF(np.sum(b, axis: 0), 1));
        Assert.AreEqual(6.006m, DF(np.sum(b, axis: 1), 0));   // 0+1.001+2.002+3.003
        Assert.AreEqual(22.022m, DF(np.sum(b, axis: 1), 1));  // 4.004+5.005+6.006+7.007
    }

    [TestMethod]
    public void Decimal_Mean_Min_Max_Prod()
    {
        var b = DecB();
        Assert.AreEqual(12.012m / 3m, DF(np.mean(b, axis: 0), 0));
        Assert.AreEqual(0m, DF(np.amin(b, axis: 0), 0));
        Assert.AreEqual(8.008m, DF(np.amax(b, axis: 0), 0));
        // prod axis0 col1: 1.001 * 5.005 * 9.009
        Assert.AreEqual(1.001m * 5.005m * 9.009m, DF(np.prod(b, axis: 0), 1));
    }

    [TestMethod]
    public void Decimal_Sum_FullPrecision_NoDoubleBridgeLoss()
    {
        // Values whose exact decimal sum is NOT representable through a double bridge.
        // 0.1 is inexact in binary; 30 * 0.1 exact in decimal = 3.0.
        var d = new decimal[30];
        for (int i = 0; i < 30; i++) d[i] = 0.1m;
        var a = np.array(d).reshape(30, 1);
        Assert.AreEqual(3.0m, DF(np.sum(a, axis: 0), 0), "decimal sum should be exact 3.0");
    }

    [TestMethod]
    public void Decimal_LayoutInvariance()
    {
        var b = DecB();
        Assert.AreEqual(12.012m, DF(np.sum(b.copy(order: 'F'), axis: 0), 0), "F-order");
        Assert.AreEqual(6.006m, DF(np.sum(b.T, axis: 0), 0), "transpose: sum axis0 of b.T == sum axis1 of b");
    }
}
