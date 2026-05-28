using System;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.APIs;

/// <summary>
/// IL-kernel-specific coverage for np.searchsorted.
/// Every test verifies against NumPy 2.4.2 expected values (commented).
/// </summary>
[TestClass]
public class NpSearchsortedIlKernelTests
{
    private static void AssertArr(NDArray r, long[] expected)
    {
        Assert.AreEqual(expected.Length, (int)r.size, "size mismatch");
        for (int i = 0; i < expected.Length; i++)
            Assert.AreEqual(expected[i], r.GetInt64(i), $"index {i}");
    }

    #region Per-dtype contract (a=[1,3,5,7,9], v=[0,1,2,5,8,10])

    // NumPy: left -> [0,0,1,2,4,5]; right -> [0,1,1,3,4,5]
    private static readonly long[] ExpectedLeft  = { 0, 0, 1, 2, 4, 5 };
    private static readonly long[] ExpectedRight = { 0, 1, 1, 3, 4, 5 };

    [TestMethod] public void Dtype_SByte()
    {
        var a = np.array(new sbyte[] { 1, 3, 5, 7, 9 });
        var v = np.array(new sbyte[] { 0, 1, 2, 5, 8, 10 });
        AssertArr(np.searchsorted(a, v), ExpectedLeft);
        AssertArr(np.searchsorted(a, v, side: "right"), ExpectedRight);
    }

    [TestMethod] public void Dtype_Byte()
    {
        var a = np.array(new byte[] { 1, 3, 5, 7, 9 });
        var v = np.array(new byte[] { 0, 1, 2, 5, 8, 10 });
        AssertArr(np.searchsorted(a, v), ExpectedLeft);
        AssertArr(np.searchsorted(a, v, side: "right"), ExpectedRight);
    }

    [TestMethod] public void Dtype_Int16()
    {
        var a = np.array(new short[] { 1, 3, 5, 7, 9 });
        var v = np.array(new short[] { 0, 1, 2, 5, 8, 10 });
        AssertArr(np.searchsorted(a, v), ExpectedLeft);
        AssertArr(np.searchsorted(a, v, side: "right"), ExpectedRight);
    }

    [TestMethod] public void Dtype_UInt16()
    {
        var a = np.array(new ushort[] { 1, 3, 5, 7, 9 });
        var v = np.array(new ushort[] { 0, 1, 2, 5, 8, 10 });
        AssertArr(np.searchsorted(a, v), ExpectedLeft);
        AssertArr(np.searchsorted(a, v, side: "right"), ExpectedRight);
    }

    [TestMethod] public void Dtype_Int32()
    {
        var a = np.array(new[] { 1, 3, 5, 7, 9 });
        var v = np.array(new[] { 0, 1, 2, 5, 8, 10 });
        AssertArr(np.searchsorted(a, v), ExpectedLeft);
        AssertArr(np.searchsorted(a, v, side: "right"), ExpectedRight);
    }

    [TestMethod] public void Dtype_UInt32()
    {
        var a = np.array(new uint[] { 1, 3, 5, 7, 9 });
        var v = np.array(new uint[] { 0, 1, 2, 5, 8, 10 });
        AssertArr(np.searchsorted(a, v), ExpectedLeft);
        AssertArr(np.searchsorted(a, v, side: "right"), ExpectedRight);
    }

    [TestMethod] public void Dtype_Int64()
    {
        var a = np.array(new[] { 1L, 3L, 5L, 7L, 9L });
        var v = np.array(new[] { 0L, 1L, 2L, 5L, 8L, 10L });
        AssertArr(np.searchsorted(a, v), ExpectedLeft);
        AssertArr(np.searchsorted(a, v, side: "right"), ExpectedRight);
    }

    [TestMethod] public void Dtype_UInt64()
    {
        var a = np.array(new[] { 1UL, 3UL, 5UL, 7UL, 9UL });
        var v = np.array(new[] { 0UL, 1UL, 2UL, 5UL, 8UL, 10UL });
        AssertArr(np.searchsorted(a, v), ExpectedLeft);
        AssertArr(np.searchsorted(a, v, side: "right"), ExpectedRight);
    }

    [TestMethod] public void Dtype_Single()
    {
        var a = np.array(new[] { 1f, 3f, 5f, 7f, 9f });
        var v = np.array(new[] { 0f, 1f, 2f, 5f, 8f, 10f });
        AssertArr(np.searchsorted(a, v), ExpectedLeft);
        AssertArr(np.searchsorted(a, v, side: "right"), ExpectedRight);
    }

    [TestMethod] public void Dtype_Double()
    {
        var a = np.array(new[] { 1.0, 3.0, 5.0, 7.0, 9.0 });
        var v = np.array(new[] { 0.0, 1.0, 2.0, 5.0, 8.0, 10.0 });
        AssertArr(np.searchsorted(a, v), ExpectedLeft);
        AssertArr(np.searchsorted(a, v, side: "right"), ExpectedRight);
    }

    [TestMethod] public void Dtype_Half()
    {
        var a = np.array(new[] { (Half)1f, (Half)3f, (Half)5f, (Half)7f, (Half)9f });
        var v = np.array(new[] { (Half)0f, (Half)1f, (Half)2f, (Half)5f, (Half)8f, (Half)10f });
        AssertArr(np.searchsorted(a, v), ExpectedLeft);
        AssertArr(np.searchsorted(a, v, side: "right"), ExpectedRight);
    }

    [TestMethod] public void Dtype_Decimal()
    {
        var a = np.array(new[] { 1m, 3m, 5m, 7m, 9m });
        var v = np.array(new[] { 0m, 1m, 2m, 5m, 8m, 10m });
        AssertArr(np.searchsorted(a, v), ExpectedLeft);
        AssertArr(np.searchsorted(a, v, side: "right"), ExpectedRight);
    }

    [TestMethod] public void Dtype_Complex_RealPartCompare()
    {
        // NumSharp compares Complex by real component (matches legacy + the IL kernel intentional shim).
        var a = np.array(new[] { new Complex(1, 0), new Complex(3, 0), new Complex(5, 0), new Complex(7, 0), new Complex(9, 0) });
        var v = np.array(new[] { new Complex(0, 0), new Complex(1, 0), new Complex(2, 0), new Complex(5, 0), new Complex(8, 0), new Complex(10, 0) });
        AssertArr(np.searchsorted(a, v), ExpectedLeft);
        AssertArr(np.searchsorted(a, v, side: "right"), ExpectedRight);
    }

    #endregion

    #region Boolean / negative / large

    [TestMethod] public void Dtype_Boolean()
    {
        // np.searchsorted([F,F,T,T], [F,T])         -> [0, 2]
        // np.searchsorted([F,F,T,T], [F,T], 'right') -> [2, 4]
        var a = np.array(new[] { false, false, true, true });
        var v = np.array(new[] { false, true });
        AssertArr(np.searchsorted(a, v), new long[] { 0, 2 });
        AssertArr(np.searchsorted(a, v, side: "right"), new long[] { 2, 4 });
    }

    [TestMethod] public void Signed_NegativesAndMix()
    {
        // np.searchsorted([-10,-5,0,5,10], [-7,0,3,15])         -> [1,2,3,5]
        // np.searchsorted([-10,-5,0,5,10], [-7,0,3,15], 'right') -> [1,3,3,5]
        var a = np.array(new[] { -10, -5, 0, 5, 10 });
        var v = np.array(new[] { -7, 0, 3, 15 });
        AssertArr(np.searchsorted(a, v), new long[] { 1, 2, 3, 5 });
        AssertArr(np.searchsorted(a, v, side: "right"), new long[] { 1, 3, 3, 5 });
    }

    [TestMethod] public void Large_Arange1MStep7()
    {
        // np.searchsorted(arange(0,1_000_000,7), [0,100,1000,100000,999993])
        //   left  -> [0,15,143,14286,142857]
        //   right -> [1,15,143,14286,142857]
        var a = np.arange(0, 1_000_000, 7);
        var v = np.array(new[] { 0, 100, 1000, 100_000, 999_993 });
        AssertArr(np.searchsorted(a, v), new long[] { 0, 15, 143, 14_286, 142_857 });
        AssertArr(np.searchsorted(a, v, side: "right"), new long[] { 1, 15, 143, 14_286, 142_857 });
    }

    #endregion

    #region Sorter + side + multidim — combined paths

    [TestMethod] public void Sorter_Right_Side_Duplicates()
    {
        // a = [3,1,2,1,3], argsort = [1,3,2,0,4] -> sorted = [1,1,2,3,3]
        // np.searchsorted(a, [1,3], sorter=..., side='right') -> [2, 5]
        var a = np.array(new[] { 3, 1, 2, 1, 3 });
        var sorter = np.array(new[] { 1L, 3L, 2L, 0L, 4L });
        var v = np.array(new[] { 1, 3 });
        AssertArr(np.searchsorted(a, v, side: "right", sorter: sorter), new long[] { 2, 5 });
    }

    [TestMethod] public void Sorter_Float_Left()
    {
        var a = np.array(new[] { 3.5, 1.5, 2.5, 1.5, 3.5 });
        var sorter = np.array(new[] { 1L, 3L, 2L, 0L, 4L });
        var v = np.array(new[] { 1.5, 2.0, 3.5 });
        // sorted view = [1.5, 1.5, 2.5, 3.5, 3.5]
        // np.searchsorted([1.5,1.5,2.5,3.5,3.5], [1.5,2.0,3.5], side='left') -> [0, 2, 3]
        AssertArr(np.searchsorted(a, v, sorter: sorter), new long[] { 0, 2, 3 });
    }

    [TestMethod] public void Multidim_V_With_Sorter()
    {
        // a = [3,1,2,1,3], sorter = [1,3,2,0,4] -> sorted = [1,1,2,3,3]
        // v = [[1,2],[3,4]], left: [[0,2],[3,5]]
        var a = np.array(new[] { 3, 1, 2, 1, 3 });
        var sorter = np.array(new[] { 1L, 3L, 2L, 0L, 4L });
        var v = np.array(new[,] { { 1, 2 }, { 3, 4 } });
        var r = np.searchsorted(a, v, sorter: sorter);
        Assert.AreEqual(2, r.ndim);
        Assert.AreEqual(0L, r.GetInt64(0, 0));
        Assert.AreEqual(2L, r.GetInt64(0, 1));
        Assert.AreEqual(3L, r.GetInt64(1, 0));
        Assert.AreEqual(5L, r.GetInt64(1, 1));
    }

    #endregion

    #region Scalar fast path

    [TestMethod] public void Scalar_Int_Hits_IL_Path()
    {
        // The scalar overloads still use the IL kernel internally (single-key buffer).
        var a = np.array(new[] { 1, 3, 5, 7, 9 });
        Assert.AreEqual(0L, np.searchsorted(a, 0));
        Assert.AreEqual(0L, np.searchsorted(a, 1));
        Assert.AreEqual(1L, np.searchsorted(a, 1, side: "right"));
        Assert.AreEqual(5L, np.searchsorted(a, 10));
    }

    [TestMethod] public void Scalar_CrossType_DoubleArrayIntKey()
    {
        // Key int promotes to a's double dtype before search.
        var a = np.array(new[] { 1.5, 2.5, 3.5, 4.5 });
        Assert.AreEqual(2L, np.searchsorted(a, 3));
        Assert.AreEqual(3L, np.searchsorted(a, 4));
    }

    #endregion

    #region Empty / NaN edge cases

    [TestMethod] public void Empty_A_BothSides()
    {
        var a = np.array(Array.Empty<int>());
        Assert.AreEqual(0L, np.searchsorted(a, 5));
        Assert.AreEqual(0L, np.searchsorted(a, 5, side: "right"));
    }

    [TestMethod] public void Empty_V_PreservesShape()
    {
        var a = np.array(new[] { 1, 2, 3 });
        var v = np.array(Array.Empty<int>());
        var r = np.searchsorted(a, v);
        Assert.AreEqual(0, r.size);
    }

    [TestMethod] public void Empty_A_With_NDArray_V()
    {
        var a = np.array(Array.Empty<int>());
        var v = np.array(new[] { 1, 2, 3 });
        var r = np.searchsorted(a, v);
        AssertArr(r, new long[] { 0, 0, 0 });
    }

    [TestMethod] public void Single_Element_A()
    {
        var a = np.array(new[] { 5 });
        var v = np.array(new[] { 1, 5, 10 });
        // NumPy: left  -> [0, 0, 1];  right -> [0, 1, 1]
        AssertArr(np.searchsorted(a, v), new long[] { 0, 0, 1 });
        AssertArr(np.searchsorted(a, v, side: "right"), new long[] { 0, 1, 1 });
    }

    #endregion
}
