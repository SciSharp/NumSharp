using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.APIs;

/// <summary>
/// Tests for np.searchsorted side / sorter / multidim parameters added per audit T1.27 follow-up.
/// All expected values produced from NumPy 2.x and pasted in the comment above each assertion.
/// </summary>
[TestClass]
public class NpSearchsortedSideSorterTests
{
    #region side='left' vs side='right' on duplicates

    [TestMethod]
    public void Side_Left_DefaultIsLeft()
    {
        // np.searchsorted([1,2,2,2,3], 2) -> 1
        var a = np.array(new[] { 1, 2, 2, 2, 3 });
        Assert.AreEqual(1L, np.searchsorted(a, 2));
        Assert.AreEqual(1L, np.searchsorted(a, 2, side: "left"));
    }

    [TestMethod]
    public void Side_Right_Duplicates()
    {
        // np.searchsorted([1,2,2,2,3], 2, side='right') -> 4
        var a = np.array(new[] { 1, 2, 2, 2, 3 });
        Assert.AreEqual(4L, np.searchsorted(a, 2, side: "right"));
    }

    [TestMethod]
    public void Side_Left_AllDuplicatesMatch()
    {
        // np.searchsorted([5,5,5], 5, side='left') -> 0
        var a = np.array(new[] { 5, 5, 5 });
        Assert.AreEqual(0L, np.searchsorted(a, 5, side: "left"));
    }

    [TestMethod]
    public void Side_Right_AllDuplicatesMatch()
    {
        // np.searchsorted([5,5,5], 5, side='right') -> 3
        var a = np.array(new[] { 5, 5, 5 });
        Assert.AreEqual(3L, np.searchsorted(a, 5, side: "right"));
    }

    [TestMethod]
    public void Side_Right_NoMatch()
    {
        // np.searchsorted([1,3,5], 3, side='right') -> 2
        var a = np.array(new[] { 1, 3, 5 });
        Assert.AreEqual(2L, np.searchsorted(a, 3, side: "right"));
    }

    [TestMethod]
    public void Side_Right_FloatDuplicates()
    {
        // np.searchsorted([1.0, 2.0, 2.0, 3.0], 2.0, side='right') -> 3
        var a = np.array(new[] { 1.0, 2.0, 2.0, 3.0 });
        Assert.AreEqual(3L, np.searchsorted(a, 2.0, side: "right"));
        Assert.AreEqual(1L, np.searchsorted(a, 2.0, side: "left"));
    }

    [TestMethod]
    public void Side_NDArrayValues_Right()
    {
        // np.searchsorted([1,2,2,2,3], [2,3], side='right') -> [4, 5]
        var a = np.array(new[] { 1, 2, 2, 2, 3 });
        var v = np.array(new[] { 2, 3 });
        var r = np.searchsorted(a, v, side: "right");
        Assert.AreEqual(2, r.size);
        Assert.AreEqual(4L, r.GetInt64(0));
        Assert.AreEqual(5L, r.GetInt64(1));
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void Side_InvalidValue_Throws()
    {
        // NumPy raises ValueError for invalid side
        var a = np.array(new[] { 1, 2, 3 });
        np.searchsorted(a, 2, side: "middle");
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void Side_EmptyString_Throws()
    {
        var a = np.array(new[] { 1, 2, 3 });
        np.searchsorted(a, 2, side: "");
    }

    #endregion

    #region sorter parameter

    [TestMethod]
    public void Sorter_BasicScalar()
    {
        // a = [40,10,20,30], sorter=argsort(a)=[1,2,3,0]
        // np.searchsorted(a, 25, sorter=sorter) -> 2
        var a = np.array(new[] { 40, 10, 20, 30 });
        var sorter = np.array(new[] { 1L, 2L, 3L, 0L });
        Assert.AreEqual(2L, np.searchsorted(a, 25, sorter: sorter));
    }

    [TestMethod]
    public void Sorter_ArrayValues()
    {
        // np.searchsorted([40,10,20,30], [25,40,15], sorter=[1,2,3,0]) -> [2,3,1]
        var a = np.array(new[] { 40, 10, 20, 30 });
        var sorter = np.array(new[] { 1L, 2L, 3L, 0L });
        var v = np.array(new[] { 25, 40, 15 });
        var r = np.searchsorted(a, v, sorter: sorter);
        Assert.AreEqual(3, r.size);
        Assert.AreEqual(2L, r.GetInt64(0));
        Assert.AreEqual(3L, r.GetInt64(1));
        Assert.AreEqual(1L, r.GetInt64(2));
    }

    [TestMethod]
    public void Sorter_WithSideLeft()
    {
        // a=[40,10,20,30] (sorted=[10,20,30,40]), sorter=[1,2,3,0]
        // np.searchsorted(a, 30, sorter=sorter, side='left') -> 2
        var a = np.array(new[] { 40, 10, 20, 30 });
        var sorter = np.array(new[] { 1L, 2L, 3L, 0L });
        Assert.AreEqual(2L, np.searchsorted(a, 30, side: "left", sorter: sorter));
    }

    [TestMethod]
    public void Sorter_WithSideRight()
    {
        // a=[40,10,20,30] (sorted=[10,20,30,40]), sorter=[1,2,3,0]
        // np.searchsorted(a, 30, sorter=sorter, side='right') -> 3
        var a = np.array(new[] { 40, 10, 20, 30 });
        var sorter = np.array(new[] { 1L, 2L, 3L, 0L });
        Assert.AreEqual(3L, np.searchsorted(a, 30, side: "right", sorter: sorter));
    }

    [TestMethod]
    public void Sorter_DoubleValue()
    {
        // a=[40,10,20,30], sorter=[1,2,3,0]
        // np.searchsorted(a, 25.5, sorter=sorter) -> 2
        var a = np.array(new[] { 40, 10, 20, 30 });
        var sorter = np.array(new[] { 1L, 2L, 3L, 0L });
        Assert.AreEqual(2L, np.searchsorted(a, 25.5, sorter: sorter));
    }

    [TestMethod]
    public void Sorter_NDArrayValues_DuplicatesRight()
    {
        // a = [3,1,2,1,3] (sorter = [1,3,2,0,4] -> sorted [1,1,2,3,3])
        // np.searchsorted(a, [1,3], sorter=sorter, side='right') -> [2, 5]
        var a = np.array(new[] { 3, 1, 2, 1, 3 });
        var sorter = np.array(new[] { 1L, 3L, 2L, 0L, 4L });
        var v = np.array(new[] { 1, 3 });
        var r = np.searchsorted(a, v, side: "right", sorter: sorter);
        Assert.AreEqual(2, r.size);
        Assert.AreEqual(2L, r.GetInt64(0));
        Assert.AreEqual(5L, r.GetInt64(1));
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void Sorter_WrongSize_Throws()
    {
        // NumPy: ValueError "sorter.size must equal a.size"
        var a = np.array(new[] { 1, 2, 3 });
        var sorter = np.array(new[] { 0L, 1L });
        np.searchsorted(a, 2, sorter: sorter);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void Sorter_EmptyForNonEmptyA_Throws()
    {
        var a = np.array(new[] { 1, 2, 3 });
        var sorter = np.array(Array.Empty<long>());
        np.searchsorted(a, 2, sorter: sorter);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void Sorter_MultidimSorter_Throws()
    {
        var a = np.array(new[] { 1, 2, 3, 4 });
        var sorter = np.array(new long[,] { { 0, 1 }, { 2, 3 } });
        np.searchsorted(a, 2, sorter: sorter);
    }

    #endregion

    #region Multidim 'a' validation

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void Multidim_A_2D_Throws()
    {
        // NumPy: ValueError "object too deep for desired array"
        var a = np.arange(20).reshape(4, 5);
        np.searchsorted(a, 5);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void Multidim_A_3D_Throws()
    {
        var a = np.arange(24).reshape(2, 3, 4);
        np.searchsorted(a, 5);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void Multidim_A_2D_WithNDArrayV_Throws()
    {
        var a = np.arange(20).reshape(4, 5);
        var v = np.array(new[] { 5, 10 });
        np.searchsorted(a, v);
    }

    #endregion

    #region Multidim 'v' preserves shape

    [TestMethod]
    public void Multidim_V_2D_PreservesShape()
    {
        // np.searchsorted([1,2,3,4,5], [[1,2],[3,4]]) -> [[0,1],[2,3]] shape (2,2)
        var a = np.array(new[] { 1, 2, 3, 4, 5 });
        var v = np.array(new[,] { { 1, 2 }, { 3, 4 } });
        var r = np.searchsorted(a, v);
        CollectionAssert.AreEqual(new long[] { 2, 2 }, r.shape);
        Assert.AreEqual(0L, r.GetInt64(0, 0));
        Assert.AreEqual(1L, r.GetInt64(0, 1));
        Assert.AreEqual(2L, r.GetInt64(1, 0));
        Assert.AreEqual(3L, r.GetInt64(1, 1));
    }

    [TestMethod]
    public void Multidim_V_2D_RightSide()
    {
        // np.searchsorted([1,2,3,4,5], [[1,2],[3,4]], side='right') -> [[1,2],[3,4]]
        var a = np.array(new[] { 1, 2, 3, 4, 5 });
        var v = np.array(new[,] { { 1, 2 }, { 3, 4 } });
        var r = np.searchsorted(a, v, side: "right");
        CollectionAssert.AreEqual(new long[] { 2, 2 }, r.shape);
        Assert.AreEqual(1L, r.GetInt64(0, 0));
        Assert.AreEqual(2L, r.GetInt64(0, 1));
        Assert.AreEqual(3L, r.GetInt64(1, 0));
        Assert.AreEqual(4L, r.GetInt64(1, 1));
    }

    [TestMethod]
    public void Multidim_V_3D_PreservesShape()
    {
        // a = [10,20,30,40], v = arange(24).reshape(2,3,4)
        // np.searchsorted(a, v) reshape (2,3,4):
        //   [[[0,0,0,0],[0,0,0,0],[0,0,0,1]],
        //    [[1,1,1,1],[1,1,1,1],[1,2,2,2]]]
        var a = np.array(new[] { 10, 20, 30, 40 });
        var v = np.arange(24).reshape(2, 3, 4);
        var r = np.searchsorted(a, v);
        CollectionAssert.AreEqual(new long[] { 2, 3, 4 }, r.shape);

        // First plane: values 0..11
        Assert.AreEqual(0L, r.GetInt64(0, 0, 0));   // v=0  -> insert at 0
        Assert.AreEqual(0L, r.GetInt64(0, 2, 0));   // v=8  -> insert at 0 (8 < 10)
        Assert.AreEqual(0L, r.GetInt64(0, 2, 1));   // v=9  -> 0
        Assert.AreEqual(1L, r.GetInt64(0, 2, 3));   // v=11 -> 1 (between 10 and 20)
        // Second plane: values 12..23
        Assert.AreEqual(1L, r.GetInt64(1, 0, 0));   // v=12 -> 1
        Assert.AreEqual(1L, r.GetInt64(1, 2, 0));   // v=20 -> 1 (left side, 20 found at idx 1)
        Assert.AreEqual(2L, r.GetInt64(1, 2, 1));   // v=21 -> 2
        Assert.AreEqual(2L, r.GetInt64(1, 2, 2));   // v=22 -> 2
        Assert.AreEqual(2L, r.GetInt64(1, 2, 3));   // v=23 -> 2
    }

    [TestMethod]
    public void Multidim_V_Empty_PreservesShape()
    {
        // np.searchsorted([1,2,3], np.array([], dtype=int).reshape(0,3)) -> shape (0,3)
        var a = np.array(new[] { 1, 2, 3 });
        var v = np.zeros(new Shape(0, 3), NPTypeCode.Int32);
        var r = np.searchsorted(a, v);
        CollectionAssert.AreEqual(new long[] { 0, 3 }, r.shape);
        Assert.AreEqual(0, r.size);
    }

    #endregion

    #region Empty 'a'

    [TestMethod]
    public void EmptyA_ScalarV_ReturnsZero_BothSides()
    {
        // np.searchsorted([], 5) -> 0 for both sides
        var a = np.array(Array.Empty<int>());
        Assert.AreEqual(0L, np.searchsorted(a, 5));
        Assert.AreEqual(0L, np.searchsorted(a, 5, side: "right"));
    }

    [TestMethod]
    public void EmptyA_ArrayV_AllZeros()
    {
        // np.searchsorted([], [1, 2, 3]) -> [0, 0, 0]
        var a = np.array(Array.Empty<int>());
        var v = np.array(new[] { 1, 2, 3 });
        var r = np.searchsorted(a, v);
        Assert.AreEqual(3, r.size);
        Assert.AreEqual(0L, r.GetInt64(0));
        Assert.AreEqual(0L, r.GetInt64(1));
        Assert.AreEqual(0L, r.GetInt64(2));
    }

    #endregion

    #region Dtype coverage

    [TestMethod]
    public void Dtype_Float_Right()
    {
        // np.searchsorted([1.0f, 2.0f, 2.0f, 3.0f], 2.0, side='right') -> 3
        var a = np.array(new float[] { 1.0f, 2.0f, 2.0f, 3.0f });
        Assert.AreEqual(3L, np.searchsorted(a, 2.0, side: "right"));
    }

    [TestMethod]
    public void Dtype_Int64_Right()
    {
        // np.searchsorted(int64[1,2,2,3], 2, 'right') -> 3
        var a = np.array(new[] { 1L, 2L, 2L, 3L });
        Assert.AreEqual(3L, np.searchsorted(a, 2, side: "right"));
    }

    [TestMethod]
    public void Dtype_UInt32_Right()
    {
        var a = np.array(new uint[] { 1, 2, 2, 3 });
        Assert.AreEqual(3L, np.searchsorted(a, 2, side: "right"));
    }

    [TestMethod]
    public void Dtype_Int16_Right()
    {
        var a = np.array(new short[] { 1, 2, 2, 3 });
        Assert.AreEqual(3L, np.searchsorted(a, 2, side: "right"));
    }

    [TestMethod]
    public void Dtype_Byte_Right()
    {
        var a = np.array(new byte[] { 1, 2, 2, 3 });
        Assert.AreEqual(3L, np.searchsorted(a, 2, side: "right"));
    }

    [TestMethod]
    public void Dtype_Half_Right()
    {
        var a = np.array(new[] { (Half)1.0f, (Half)2.0f, (Half)2.0f, (Half)3.0f });
        Assert.AreEqual(3L, np.searchsorted(a, 2.0, side: "right"));
        Assert.AreEqual(1L, np.searchsorted(a, 2.0, side: "left"));
    }

    #endregion

    #region View / strided 'a'

    [TestMethod]
    public void Sliced_A_Works()
    {
        // a = [0,1,2,3,4,5,6,7,8,9][2:8] = [2,3,4,5,6,7]
        // searchsorted slice for 5 -> 3 (insert at index 3 of the slice)
        var a = np.arange(10);
        var slice = a["2:8"];
        Assert.AreEqual(3L, np.searchsorted(slice, 5));
        Assert.AreEqual(4L, np.searchsorted(slice, 5, side: "right"));
    }

    [TestMethod]
    public void Strided_A_StepWorks()
    {
        // a = arange(10)[::2] = [0,2,4,6,8]
        // searchsorted for 5 -> 3 (between 4 and 6)
        var a = np.arange(10)["::2"];
        Assert.AreEqual(3L, np.searchsorted(a, 5));
        Assert.AreEqual(3L, np.searchsorted(a, 5, side: "right"));
    }

    [TestMethod]
    public void Sliced_V_PreservesValues()
    {
        // a=[10,20,30], v=arange(10)[1:6:2]=[1,3,5]
        // np.searchsorted([10,20,30], [1,3,5]) -> [0,0,0]
        var a = np.array(new[] { 10, 20, 30 });
        var v = np.arange(10)["1:6:2"];
        var r = np.searchsorted(a, v);
        Assert.AreEqual(3, r.size);
        Assert.AreEqual(0L, r.GetInt64(0));
        Assert.AreEqual(0L, r.GetInt64(1));
        Assert.AreEqual(0L, r.GetInt64(2));
    }

    #endregion
}
