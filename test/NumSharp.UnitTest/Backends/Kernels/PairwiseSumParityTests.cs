using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.Backends.Kernels;

/// <summary>
/// Pins float32/float64 <c>sum</c> &amp; <c>mean</c> axis reductions to NumPy 2.4.2
/// EXACTLY. The per-chunk PINNED path uses <c>PairwiseFold</c>, ported 1:1 from
/// NumPy's <c>pairwise_sum</c> (loops_utils.h.src), so a contiguous reduced axis sums
/// bit-for-bit like NumPy — which is what made float32 safe to route through the
/// per-chunk path (a flat accumulator diverged; see Float32_LargeValues_*).
///
/// Expected values produced by NumPy 2.4.2:
///   a64 = np.arange(400).reshape(2,200)*0.5 ;  a32 = a64.astype(np.float32)
///     np.sum(a32, axis=1)  = [9950., 29950.]
///     np.mean(a32, axis=1) = [49.75, 149.75]
///     np.sum(a32, axis=0)[:4] = [100., 101., 102., 103.]
///   b32 = (np.arange(2000).reshape(2,1000)*0.0009+0.7).astype(np.float32)
///     np.sum(b32, axis=1)  = [1149.5499267578125, 2049.550048828125]
/// </summary>
[TestClass]
public class PairwiseSumParityTests
{
    private static NDArray A64() => np.arange(400).astype(NPTypeCode.Double).reshape(2, 200) * 0.5;
    private static NDArray A32() => A64().astype(NPTypeCode.Single);

    [TestMethod]
    public void Float64_Sum_Mean_Axis1_ExactNumPy()
    {
        var a = A64(); // axis1 reduces 200 (>128 → pairwise recursion)
        var s = np.sum(a, axis: 1);
        Assert.AreEqual(9950.0, (double)s.GetAtIndex(0));
        Assert.AreEqual(29950.0, (double)s.GetAtIndex(1));
        var m = np.mean(a, axis: 1);
        Assert.AreEqual(49.75, (double)m.GetAtIndex(0));
        Assert.AreEqual(149.75, (double)m.GetAtIndex(1));
    }

    [TestMethod]
    public void Float32_Sum_Mean_Axis1_ExactNumPy()
    {
        var a = A32();
        var s = np.sum(a, axis: 1);
        Assert.AreEqual(9950.0f, (float)s.GetAtIndex(0));
        Assert.AreEqual(29950.0f, (float)s.GetAtIndex(1));
        var m = np.mean(a, axis: 1);
        Assert.AreEqual(49.75f, (float)m.GetAtIndex(0));
        Assert.AreEqual(149.75f, (float)m.GetAtIndex(1));
    }

    [TestMethod]
    public void Float32_Sum_Axis0_SlabExactNumPy()
    {
        var a = A32(); // axis0 reduces 2 (SLAB streaming add)
        var s = np.sum(a, axis: 0);
        Assert.AreEqual(100.0f, (float)s.GetAtIndex(0));
        Assert.AreEqual(101.0f, (float)s.GetAtIndex(1));
        Assert.AreEqual(102.0f, (float)s.GetAtIndex(2));
        Assert.AreEqual(103.0f, (float)s.GetAtIndex(3));
    }

    [TestMethod]
    public void Float32_LargeValues_Sum_Axis1_BitExact()
    {
        // The regression case: with a flat 8-accumulator this float32 sum diverged from
        // NumPy by more than float tolerance; pairwise makes it bit-exact. (1000 > 128.)
        var b = (np.arange(2000).astype(NPTypeCode.Double).reshape(2, 1000) * 0.0009 + 0.7).astype(NPTypeCode.Single);
        var s = np.sum(b, axis: 1);
        Assert.AreEqual(1149.5499267578125f, (float)s.GetAtIndex(0));
        Assert.AreEqual(2049.550048828125f, (float)s.GetAtIndex(1));
    }
}
