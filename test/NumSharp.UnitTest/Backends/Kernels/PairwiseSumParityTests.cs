using System.Numerics;
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

    // The IL-emitted SIMD pairwise kernel folds a CONTIGUOUS reduced axis with the
    // vector leaf and a STRIDED reduced axis with the scalar 8-accumulator leaf. Both
    // must reproduce NumPy's pairwise_sum, which depends only on (values, order, n) —
    // so a strided view and its contiguous .copy() (identical values, identical order)
    // must sum BIT-FOR-BIT identically. This pins the strided IL leaf against the SIMD leaf.
    [TestMethod]
    public void Float64_StridedReducedAxis_BitEqualsContiguousLeaf()
    {
        var b = (np.arange(4 * 512).astype(NPTypeCode.Double) % 31) * 0.125 + 0.3;
        var full = b.reshape(4, 512);
        var strided = full[":, ::2"]; // (4,256) reduced axis stride 2
        var s_strided = np.sum(strided, axis: 1);       // scalar 8-acc pairwise leaf (strided)
        var s_contig = np.sum(strided.copy(), axis: 1); // SIMD pairwise leaf (contiguous)
        for (int i = 0; i < 4; i++)
            Assert.AreEqual((double)s_contig.GetAtIndex(i), (double)s_strided.GetAtIndex(i),
                $"row {i}: strided leaf must bit-match the contiguous SIMD leaf");
    }

    [TestMethod]
    public void Float32_StridedReducedAxis_BitEqualsContiguousLeaf()
    {
        var b = (np.arange(4 * 600).astype(NPTypeCode.Single) % 17) * 0.3f + 0.5f;
        var full = b.reshape(4, 600);
        var strided = full[":, ::3"];                    // (4,200) reduced axis stride 3
        var s_strided = np.sum(strided, axis: 1);
        var s_contig = np.sum(strided.copy(), axis: 1);
        for (int i = 0; i < 4; i++)
            Assert.AreEqual((float)s_contig.GetAtIndex(i), (float)s_strided.GetAtIndex(i),
                $"row {i}: strided leaf must bit-match the contiguous SIMD leaf");
    }

    // Complex128 sum is the IL-emitted Vector128-per-complex pairwise (NumPy CDOUBLE
    // pairwise_sum). Expected values produced by NumPy 2.4.2 for:
    //   k=arange(N); a=((k%17)*0.25-2 + 1j*((k%13)*0.5-3)).reshape(R,C)
    //     np.sum(a,axis=1):  (2,200)->[(-6.5,-10),(-2.5,2.5)]  (2,1000)->[(-5.25,-3),(-3,-2.5)]
    private static NDArray MakeComplex(int R, int C)
    {
        long N = (long)R * C;
        var re = (np.arange(N) % 17).astype(NPTypeCode.Double) * 0.25 - 2.0;
        var im = (np.arange(N) % 13).astype(NPTypeCode.Double) * 0.5 - 3.0;
        return (re.astype(NPTypeCode.Complex) + im.astype(NPTypeCode.Complex) * new Complex(0, 1)).reshape(R, C);
    }

    [TestMethod]
    public void Complex128_Sum_Axis1_ExactNumPy()
    {
        var a = MakeComplex(2, 200); // 200 > 64 → recursion
        var s = np.sum(a, axis: 1);
        Assert.AreEqual(new Complex(-6.5, -10.0), (Complex)s.GetAtIndex(0));
        Assert.AreEqual(new Complex(-2.5, 2.5), (Complex)s.GetAtIndex(1));

        var b = MakeComplex(2, 1000);
        var s2 = np.sum(b, axis: 1);
        Assert.AreEqual(new Complex(-5.25, -3.0), (Complex)s2.GetAtIndex(0));
        Assert.AreEqual(new Complex(-3.0, -2.5), (Complex)s2.GetAtIndex(1));
    }

    [TestMethod]
    public void Complex128_StridedReducedAxis_BitEqualsContiguous()
    {
        var a = MakeComplex(20, 30);
        var t = a.T;                       // (30,20) axis1 strided
        var sStrided = np.sum(t, axis: 1);
        var sContig = np.sum(t.copy(), axis: 1);
        for (int i = 0; i < 30; i++)
            Assert.AreEqual((Complex)sContig.GetAtIndex(i), (Complex)sStrided.GetAtIndex(i), $"row {i}");
    }
}
