using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.APIs;

/// <summary>
/// Edge case tests for np.searchsorted to verify fix from commit 40a5c831.
/// Tests validated against NumPy 2.4.2.
/// </summary>
public class NpSearchsortedEdgeCaseTests
{
    #region Scalar Input Tests (Main Fix)

    [TestMethod]
    public void ScalarInt_ExactMatch()
    {
        // NumPy 2.4.2: np.searchsorted([1, 2, 3, 4, 5], 3) = 2
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });
        var result = np.searchsorted(arr, 3);
        Assert.AreEqual(2, result);
    }

    [TestMethod]
    public void ScalarInt_NotInArray()
    {
        // NumPy 2.4.2: np.searchsorted([1, 3, 5], 4) = 2
        var arr = np.array(new[] { 1, 3, 5 });
        var result = np.searchsorted(arr, 4);
        Assert.AreEqual(2, result);
    }

    [TestMethod]
    public void ScalarDouble_ExactMatch()
    {
        // NumPy 2.4.2: np.searchsorted([1.0, 2.0, 3.0], 2.0) = 1
        var arr = np.array(new[] { 1.0, 2.0, 3.0 });
        var result = np.searchsorted(arr, 2.0);
        Assert.AreEqual(1, result);
    }

    [TestMethod]
    public void ScalarDouble_Between()
    {
        // NumPy 2.4.2: np.searchsorted([1.0, 2.0, 3.0], 2.5) = 2
        var arr = np.array(new[] { 1.0, 2.0, 3.0 });
        var result = np.searchsorted(arr, 2.5);
        Assert.AreEqual(2, result);
    }

    #endregion

    #region Value Not In Array Tests

    [TestMethod]
    public void Value_BeforeAllElements()
    {
        // NumPy 2.4.2: np.searchsorted([10, 20, 30], 5) = 0
        var arr = np.array(new[] { 10, 20, 30 });
        var result = np.searchsorted(arr, 5);
        Assert.AreEqual(0, result);
    }

    [TestMethod]
    public void Value_AfterAllElements()
    {
        // NumPy 2.4.2: np.searchsorted([10, 20, 30], 100) = 3
        var arr = np.array(new[] { 10, 20, 30 });
        var result = np.searchsorted(arr, 100);
        Assert.AreEqual(3, result);
    }

    [TestMethod]
    public void Value_InMiddleGap()
    {
        // NumPy 2.4.2: np.searchsorted([1, 5, 10, 50], 7) = 2
        var arr = np.array(new[] { 1, 5, 10, 50 });
        var result = np.searchsorted(arr, 7);
        Assert.AreEqual(2, result);
    }

    #endregion

    #region Empty Array Edge Cases

    [TestMethod]
    public void EmptySearchArray_ReturnZero()
    {
        // NumPy 2.4.2: np.searchsorted([], 5) = 0
        var arr = np.array(Array.Empty<int>());
        var result = np.searchsorted(arr, 5);
        Assert.AreEqual(0, result);
    }

    [TestMethod]
    public void EmptySearchArray_ReturnZeroForDouble()
    {
        // NumPy 2.4.2: np.searchsorted([], 5.0) = 0
        var arr = np.array(Array.Empty<double>());
        var result = np.searchsorted(arr, 5.0);
        Assert.AreEqual(0, result);
    }

    [TestMethod]
    public void EmptyValuesArray_ReturnEmptyArray()
    {
        // NumPy 2.4.2: np.searchsorted([1, 2, 3], []) returns array([], dtype=int64)
        var arr = np.array(new[] { 1, 2, 3 });
        var values = np.array(Array.Empty<int>());
        var result = np.searchsorted(arr, values);
        Assert.AreEqual(0, result.size);
    }

    #endregion

    #region Duplicate Values Tests

    [TestMethod]
    public void DuplicateValues_ReturnsLeftmostIndex()
    {
        // NumPy 2.4.2 (side='left' default): np.searchsorted([1, 2, 2, 2, 3], 2) = 1
        var arr = np.array(new[] { 1, 2, 2, 2, 3 });
        var result = np.searchsorted(arr, 2);
        Assert.AreEqual(1, result);
    }

    [TestMethod]
    public void AllDuplicates_ReturnsZero()
    {
        // NumPy 2.4.2: np.searchsorted([5, 5, 5, 5], 5) = 0
        var arr = np.array(new[] { 5, 5, 5, 5 });
        var result = np.searchsorted(arr, 5);
        Assert.AreEqual(0, result);
    }

    [TestMethod]
    public void AllDuplicates_ValueGreater()
    {
        // NumPy 2.4.2: np.searchsorted([5, 5, 5, 5], 10) = 4
        var arr = np.array(new[] { 5, 5, 5, 5 });
        var result = np.searchsorted(arr, 10);
        Assert.AreEqual(4, result);
    }

    [TestMethod]
    public void AllDuplicates_ValueLess()
    {
        // NumPy 2.4.2: np.searchsorted([5, 5, 5, 5], 1) = 0
        var arr = np.array(new[] { 5, 5, 5, 5 });
        var result = np.searchsorted(arr, 1);
        Assert.AreEqual(0, result);
    }

    #endregion

    #region Different dtypes Tests

    [TestMethod]
    public void Int32Array_IntSearch()
    {
        var arr = np.array(new long[] { 1, 2, 3, 4, 5 });
        var result = np.searchsorted(arr, 3);
        Assert.AreEqual(2, result);
    }

    [TestMethod]
    public void Int64Array_IntSearch()
    {
        var arr = np.array(new long[] { 1L, 2L, 3L, 4L, 5L });
        var result = np.searchsorted(arr, 3);
        Assert.AreEqual(2, result);
    }

    [TestMethod]
    public void FloatArray_DoubleSearch()
    {
        var arr = np.array(new float[] { 1.0f, 2.0f, 3.0f });
        var result = np.searchsorted(arr, 2.5);
        Assert.AreEqual(2, result);
    }

    [TestMethod]
    public void DoubleArray_IntSearch()
    {
        // Cross-type search: int value in double array
        var arr = np.array(new double[] { 1.0, 2.0, 3.0, 4.0 });
        var result = np.searchsorted(arr, 3);
        Assert.AreEqual(2, result);
    }

    [TestMethod]
    public void ByteArray_Search()
    {
        var arr = np.array(new byte[] { 10, 20, 30, 40 });
        var result = np.searchsorted(arr, 25);
        Assert.AreEqual(2, result);
    }

    #endregion

    #region NDArray Input Tests

    [TestMethod]
    public void NDArrayInput_SingleElement()
    {
        // NumPy 2.4.2: np.searchsorted([1, 2, 3], [2]) = array([1])
        var arr = np.array(new[] { 1, 2, 3 });
        var values = np.array(new[] { 2 });
        var result = np.searchsorted(arr, values);
        Assert.AreEqual(1, result.size);
        Assert.AreEqual(1L, result.GetInt64(0));
    }

    [TestMethod]
    public void NDArrayInput_MultipleElements()
    {
        // NumPy 2.4.2: np.searchsorted([1, 2, 3, 4, 5], [2, 4]) = array([1, 3])
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });
        var values = np.array(new[] { 2, 4 });
        var result = np.searchsorted(arr, values);
        Assert.AreEqual(2, result.size);
        Assert.AreEqual(1L, result.GetInt64(0));
        Assert.AreEqual(3L, result.GetInt64(1));
    }

    [TestMethod]
    public void NDArrayInput_AllOutOfRange()
    {
        // NumPy 2.4.2: np.searchsorted([10, 20, 30], [-5, 100]) = array([0, 3])
        var arr = np.array(new[] { 10, 20, 30 });
        var values = np.array(new[] { -5, 100 });
        var result = np.searchsorted(arr, values);
        Assert.AreEqual(2, result.size);
        Assert.AreEqual(0L, result.GetInt64(0));
        Assert.AreEqual(3L, result.GetInt64(1));
    }

    [TestMethod]
    public void NDArrayInput_MixedInAndOut()
    {
        // NumPy 2.4.2: np.searchsorted([11, 12, 13, 14, 15], [-10, 20, 12, 13]) = array([0, 5, 1, 2])
        var arr = np.array(new[] { 11, 12, 13, 14, 15 });
        var values = np.array(new[] { -10, 20, 12, 13 });
        var result = np.searchsorted(arr, values);
        Assert.AreEqual(4, result.size);
        Assert.AreEqual(0L, result.GetInt64(0));
        Assert.AreEqual(5L, result.GetInt64(1));
        Assert.AreEqual(1L, result.GetInt64(2));
        Assert.AreEqual(2L, result.GetInt64(3));
    }

    [TestMethod]
    public void NDArrayInput_DoubleValues()
    {
        // NumPy 2.4.2: np.searchsorted([1.0, 2.0, 3.0], [1.5, 2.5]) = array([1, 2])
        var arr = np.array(new[] { 1.0, 2.0, 3.0 });
        var values = np.array(new[] { 1.5, 2.5 });
        var result = np.searchsorted(arr, values);
        Assert.AreEqual(2, result.size);
        Assert.AreEqual(1L, result.GetInt64(0));
        Assert.AreEqual(2L, result.GetInt64(1));
    }

    #endregion

    #region Scalar NDArray Tests (Key Fix Area)

    [TestMethod]
    public void ScalarNDArray_ReturnsScalar()
    {
        // NumPy 2.4.2: np.searchsorted([1, 2, 3], np.array(2)) returns scalar 1
        var arr = np.array(new[] { 1, 2, 3 });
        var scalar = NDArray.Scalar(2);
        var result = np.searchsorted(arr, scalar);

        // Result should be a scalar NDArray
        Assert.IsTrue(result.Shape.IsScalar, "Result should be scalar");
        Assert.AreEqual(1L, result.GetInt64());
    }

    [TestMethod]
    public void ScalarNDArray_Double_ReturnsScalar()
    {
        // np.searchsorted([1.0, 2.0, 3.0], np.array(2.5)) returns scalar 2
        var arr = np.array(new[] { 1.0, 2.0, 3.0 });
        var scalar = NDArray.Scalar(2.5);
        var result = np.searchsorted(arr, scalar);

        Assert.IsTrue(result.Shape.IsScalar, "Result should be scalar");
        Assert.AreEqual(2L, result.GetInt64());
    }

    #endregion

    #region Single Element Array Tests

    [TestMethod]
    public void SingleElementArray_ValueLess()
    {
        // NumPy 2.4.2: np.searchsorted([5], 1) = 0
        var arr = np.array(new[] { 5 });
        var result = np.searchsorted(arr, 1);
        Assert.AreEqual(0, result);
    }

    [TestMethod]
    public void SingleElementArray_ValueEqual()
    {
        // NumPy 2.4.2: np.searchsorted([5], 5) = 0
        var arr = np.array(new[] { 5 });
        var result = np.searchsorted(arr, 5);
        Assert.AreEqual(0, result);
    }

    [TestMethod]
    public void SingleElementArray_ValueGreater()
    {
        // NumPy 2.4.2: np.searchsorted([5], 10) = 1
        var arr = np.array(new[] { 5 });
        var result = np.searchsorted(arr, 10);
        Assert.AreEqual(1, result);
    }

    #endregion

    #region Boundary Value Tests

    [TestMethod]
    public void FirstElement_Match()
    {
        // NumPy 2.4.2: np.searchsorted([1, 2, 3], 1) = 0
        var arr = np.array(new[] { 1, 2, 3 });
        var result = np.searchsorted(arr, 1);
        Assert.AreEqual(0, result);
    }

    [TestMethod]
    public void LastElement_Match()
    {
        // NumPy 2.4.2: np.searchsorted([1, 2, 3], 3) = 2
        var arr = np.array(new[] { 1, 2, 3 });
        var result = np.searchsorted(arr, 3);
        Assert.AreEqual(2, result);
    }

    [TestMethod]
    public void JustBeforeFirst()
    {
        // NumPy 2.4.2: np.searchsorted([10, 20, 30], 9) = 0
        var arr = np.array(new[] { 10, 20, 30 });
        var result = np.searchsorted(arr, 9);
        Assert.AreEqual(0, result);
    }

    [TestMethod]
    public void JustAfterLast()
    {
        // NumPy 2.4.2: np.searchsorted([10, 20, 30], 31) = 3
        var arr = np.array(new[] { 10, 20, 30 });
        var result = np.searchsorted(arr, 31);
        Assert.AreEqual(3, result);
    }

    #endregion

    #region Large Array Tests

    [TestMethod]
    public void LargeArray_FindMiddle()
    {
        // Create array 0..999
        var arr = np.arange(1000);
        var result = np.searchsorted(arr, 500);
        Assert.AreEqual(500, result);
    }

    [TestMethod]
    public void LargeArray_FindNearEnd()
    {
        // Create array 0..999
        var arr = np.arange(1000);
        var result = np.searchsorted(arr, 999);
        Assert.AreEqual(999, result);
    }

    [TestMethod]
    public void LargeArray_FindBetweenElements()
    {
        // Create array [0, 10, 20, ..., 990]
        var arr = np.arange(0, 1000, 10);
        var result = np.searchsorted(arr, 55);  // Between 50 and 60
        Assert.AreEqual(6, result);  // Index of 60
    }

    #endregion

    #region Negative Value Tests

    [TestMethod]
    public void NegativeValues_InArray()
    {
        // NumPy 2.4.2: np.searchsorted([-3, -1, 0, 2, 5], -2) = 1
        var arr = np.array(new[] { -3, -1, 0, 2, 5 });
        var result = np.searchsorted(arr, -2);
        Assert.AreEqual(1, result);
    }

    [TestMethod]
    public void NegativeValues_SearchNegative()
    {
        // NumPy 2.4.2: np.searchsorted([-10, -5, 0, 5, 10], -7) = 1
        var arr = np.array(new[] { -10, -5, 0, 5, 10 });
        var result = np.searchsorted(arr, -7);
        Assert.AreEqual(1, result);
    }

    [TestMethod]
    public void NegativeValues_SearchZero()
    {
        // NumPy 2.4.2: np.searchsorted([-10, -5, 0, 5, 10], 0) = 2
        var arr = np.array(new[] { -10, -5, 0, 5, 10 });
        var result = np.searchsorted(arr, 0);
        Assert.AreEqual(2, result);
    }

    #endregion

    #region Floating Point Precision Tests

    [TestMethod]
    public void FloatPrecision_VeryClose()
    {
        // Test with very close floating point values
        var arr = np.array(new[] { 1.0, 1.0000001, 1.0000002 });
        var result = np.searchsorted(arr, 1.00000015);
        Assert.AreEqual(2, result);
    }

    [TestMethod]
    public void FloatPrecision_SmallValues()
    {
        // Test with small values
        var arr = np.array(new[] { 0.001, 0.002, 0.003 });
        var result = np.searchsorted(arr, 0.0025);
        Assert.AreEqual(2, result);
    }

    #endregion
}
