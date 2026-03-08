using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using TUnit.Core;

namespace NumSharp.UnitTest.Backends.Kernels;

/// <summary>
/// Tests for NaN-aware reduction operations (nansum, nanprod, nanmin, nanmax).
/// All expected values are verified against NumPy 2.4.2.
/// </summary>
public class NanReductionTests
{
    #region np.nansum Tests

    [Test]
    public void NanSum_BasicFloat_IgnoresNaN()
    {
        // NumPy: np.nansum([1.0, np.nan, 3.0]) == 4.0
        var arr = np.array(new float[] { 1.0f, float.NaN, 3.0f });
        var result = np.nansum(arr);
        Assert.AreEqual(4.0f, (float)result.GetAtIndex(0), 1e-6f);
    }

    [Test]
    public void NanSum_BasicDouble_IgnoresNaN()
    {
        // NumPy: np.nansum([1.0, np.nan, 3.0]) == 4.0
        var arr = np.array(new double[] { 1.0, double.NaN, 3.0 });
        var result = np.nansum(arr);
        Assert.AreEqual(4.0, (double)result.GetAtIndex(0), 1e-10);
    }

    [Test]
    public void NanSum_AllNaN_ReturnsZero()
    {
        // NumPy: np.nansum([np.nan, np.nan]) == 0.0
        var arr = np.array(new double[] { double.NaN, double.NaN });
        var result = np.nansum(arr);
        Assert.AreEqual(0.0, (double)result.GetAtIndex(0), 1e-10);
    }

    [Test]
    public void NanSum_NoNaN_SameAsSum()
    {
        // NumPy: np.nansum([1.0, 2.0, 3.0]) == 6.0
        var arr = np.array(new double[] { 1.0, 2.0, 3.0 });
        var result = np.nansum(arr);
        Assert.AreEqual(6.0, (double)result.GetAtIndex(0), 1e-10);
    }

    [Test]
    public void NanSum_MultipleNaN_IgnoresAll()
    {
        // NumPy: np.nansum([np.nan, 1.0, np.nan, 2.0, np.nan]) == 3.0
        var arr = np.array(new double[] { double.NaN, 1.0, double.NaN, 2.0, double.NaN });
        var result = np.nansum(arr);
        Assert.AreEqual(3.0, (double)result.GetAtIndex(0), 1e-10);
    }

    [Test]
    public void NanSum_LargeArray_SIMD()
    {
        // Test SIMD path with large array (> Vector256 count)
        var data = new double[1000];
        for (int i = 0; i < 1000; i++)
        {
            data[i] = i % 10 == 0 ? double.NaN : 1.0;
        }
        var arr = np.array(data);
        var result = np.nansum(arr);
        // 900 non-NaN values of 1.0 each
        Assert.AreEqual(900.0, (double)result.GetAtIndex(0), 1e-10);
    }

    #endregion

    #region np.nanprod Tests

    [Test]
    public void NanProd_BasicDouble_IgnoresNaN()
    {
        // NumPy: np.nanprod([2.0, np.nan, 3.0]) == 6.0
        var arr = np.array(new double[] { 2.0, double.NaN, 3.0 });
        var result = np.nanprod(arr);
        Assert.AreEqual(6.0, (double)result.GetAtIndex(0), 1e-10);
    }

    [Test]
    public void NanProd_AllNaN_ReturnsOne()
    {
        // NumPy: np.nanprod([np.nan, np.nan]) == 1.0
        var arr = np.array(new double[] { double.NaN, double.NaN });
        var result = np.nanprod(arr);
        Assert.AreEqual(1.0, (double)result.GetAtIndex(0), 1e-10);
    }

    [Test]
    public void NanProd_NoNaN_SameAsProd()
    {
        // NumPy: np.nanprod([2.0, 3.0, 4.0]) == 24.0
        var arr = np.array(new double[] { 2.0, 3.0, 4.0 });
        var result = np.nanprod(arr);
        Assert.AreEqual(24.0, (double)result.GetAtIndex(0), 1e-10);
    }

    [Test]
    public void NanProd_WithZero_ReturnsZero()
    {
        // NumPy: np.nanprod([2.0, 0.0, np.nan, 3.0]) == 0.0
        var arr = np.array(new double[] { 2.0, 0.0, double.NaN, 3.0 });
        var result = np.nanprod(arr);
        Assert.AreEqual(0.0, (double)result.GetAtIndex(0), 1e-10);
    }

    [Test]
    public void NanProd_Float_IgnoresNaN()
    {
        // NumPy: np.nanprod(np.array([2.0, np.nan, 3.0], dtype=np.float32)) == 6.0
        var arr = np.array(new float[] { 2.0f, float.NaN, 3.0f });
        var result = np.nanprod(arr);
        Assert.AreEqual(6.0f, (float)result.GetAtIndex(0), 1e-6f);
    }

    #endregion

    #region np.nanmin Tests

    [Test]
    public void NanMin_BasicDouble_IgnoresNaN()
    {
        // NumPy: np.nanmin([1.0, np.nan, 3.0]) == 1.0
        var arr = np.array(new double[] { 1.0, double.NaN, 3.0 });
        var result = np.nanmin(arr);
        Assert.AreEqual(1.0, (double)result.GetAtIndex(0), 1e-10);
    }

    [Test]
    public void NanMin_AllNaN_ReturnsNaN()
    {
        // NumPy: np.nanmin([np.nan, np.nan]) == nan (with RuntimeWarning)
        var arr = np.array(new double[] { double.NaN, double.NaN });
        var result = np.nanmin(arr);
        Assert.IsTrue(double.IsNaN((double)result.GetAtIndex(0)));
    }

    [Test]
    public void NanMin_NoNaN_SameAsMin()
    {
        // NumPy: np.nanmin([5.0, 2.0, 8.0]) == 2.0
        var arr = np.array(new double[] { 5.0, 2.0, 8.0 });
        var result = np.nanmin(arr);
        Assert.AreEqual(2.0, (double)result.GetAtIndex(0), 1e-10);
    }

    [Test]
    public void NanMin_NegativeValues_IgnoresNaN()
    {
        // NumPy: np.nanmin([-5.0, np.nan, -2.0]) == -5.0
        var arr = np.array(new double[] { -5.0, double.NaN, -2.0 });
        var result = np.nanmin(arr);
        Assert.AreEqual(-5.0, (double)result.GetAtIndex(0), 1e-10);
    }

    [Test]
    public void NanMin_Float_IgnoresNaN()
    {
        // NumPy: np.nanmin(np.array([1.0, np.nan, 3.0], dtype=np.float32)) == 1.0
        var arr = np.array(new float[] { 1.0f, float.NaN, 3.0f });
        var result = np.nanmin(arr);
        Assert.AreEqual(1.0f, (float)result.GetAtIndex(0), 1e-6f);
    }

    [Test]
    public void NanMin_NaNAtStart_IgnoresIt()
    {
        // NumPy: np.nanmin([np.nan, 5.0, 2.0]) == 2.0
        var arr = np.array(new double[] { double.NaN, 5.0, 2.0 });
        var result = np.nanmin(arr);
        Assert.AreEqual(2.0, (double)result.GetAtIndex(0), 1e-10);
    }

    [Test]
    public void NanMin_NaNAtEnd_IgnoresIt()
    {
        // NumPy: np.nanmin([5.0, 2.0, np.nan]) == 2.0
        var arr = np.array(new double[] { 5.0, 2.0, double.NaN });
        var result = np.nanmin(arr);
        Assert.AreEqual(2.0, (double)result.GetAtIndex(0), 1e-10);
    }

    #endregion

    #region np.nanmax Tests

    [Test]
    public void NanMax_BasicDouble_IgnoresNaN()
    {
        // NumPy: np.nanmax([1.0, np.nan, 3.0]) == 3.0
        var arr = np.array(new double[] { 1.0, double.NaN, 3.0 });
        var result = np.nanmax(arr);
        Assert.AreEqual(3.0, (double)result.GetAtIndex(0), 1e-10);
    }

    [Test]
    public void NanMax_AllNaN_ReturnsNaN()
    {
        // NumPy: np.nanmax([np.nan, np.nan]) == nan (with RuntimeWarning)
        var arr = np.array(new double[] { double.NaN, double.NaN });
        var result = np.nanmax(arr);
        Assert.IsTrue(double.IsNaN((double)result.GetAtIndex(0)));
    }

    [Test]
    public void NanMax_NoNaN_SameAsMax()
    {
        // NumPy: np.nanmax([5.0, 2.0, 8.0]) == 8.0
        var arr = np.array(new double[] { 5.0, 2.0, 8.0 });
        var result = np.nanmax(arr);
        Assert.AreEqual(8.0, (double)result.GetAtIndex(0), 1e-10);
    }

    [Test]
    public void NanMax_NegativeValues_IgnoresNaN()
    {
        // NumPy: np.nanmax([-5.0, np.nan, -2.0]) == -2.0
        var arr = np.array(new double[] { -5.0, double.NaN, -2.0 });
        var result = np.nanmax(arr);
        Assert.AreEqual(-2.0, (double)result.GetAtIndex(0), 1e-10);
    }

    [Test]
    public void NanMax_Float_IgnoresNaN()
    {
        // NumPy: np.nanmax(np.array([1.0, np.nan, 3.0], dtype=np.float32)) == 3.0
        var arr = np.array(new float[] { 1.0f, float.NaN, 3.0f });
        var result = np.nanmax(arr);
        Assert.AreEqual(3.0f, (float)result.GetAtIndex(0), 1e-6f);
    }

    [Test]
    public void NanMax_NaNAtStart_IgnoresIt()
    {
        // NumPy: np.nanmax([np.nan, 5.0, 2.0]) == 5.0
        var arr = np.array(new double[] { double.NaN, 5.0, 2.0 });
        var result = np.nanmax(arr);
        Assert.AreEqual(5.0, (double)result.GetAtIndex(0), 1e-10);
    }

    [Test]
    public void NanMax_NaNAtEnd_IgnoresIt()
    {
        // NumPy: np.nanmax([5.0, 2.0, np.nan]) == 5.0
        var arr = np.array(new double[] { 5.0, 2.0, double.NaN });
        var result = np.nanmax(arr);
        Assert.AreEqual(5.0, (double)result.GetAtIndex(0), 1e-10);
    }

    #endregion

    #region Large Array SIMD Tests

    [Test]
    public void NanSum_LargeFloatArray_SIMD()
    {
        // Test SIMD path with large float array
        var data = new float[1024];
        float expectedSum = 0;
        for (int i = 0; i < 1024; i++)
        {
            if (i % 7 == 0)
            {
                data[i] = float.NaN;
            }
            else
            {
                data[i] = 1.0f;
                expectedSum += 1.0f;
            }
        }
        var arr = np.array(data);
        var result = np.nansum(arr);
        Assert.AreEqual(expectedSum, (float)result.GetAtIndex(0), 1e-3f);
    }

    [Test]
    public void NanMin_LargeDoubleArray_SIMD()
    {
        // Test SIMD path for nanmin
        var data = new double[1024];
        for (int i = 0; i < 1024; i++)
        {
            data[i] = i % 5 == 0 ? double.NaN : (double)(i + 100);
        }
        data[500] = 0.5; // This should be the minimum
        var arr = np.array(data);
        var result = np.nanmin(arr);
        Assert.AreEqual(0.5, (double)result.GetAtIndex(0), 1e-10);
    }

    [Test]
    public void NanMax_LargeDoubleArray_SIMD()
    {
        // Test SIMD path for nanmax
        var data = new double[1024];
        for (int i = 0; i < 1024; i++)
        {
            data[i] = i % 5 == 0 ? double.NaN : (double)i;
        }
        data[999] = 9999.0; // This should be the maximum
        var arr = np.array(data);
        var result = np.nanmax(arr);
        Assert.AreEqual(9999.0, (double)result.GetAtIndex(0), 1e-10);
    }

    [Test]
    public void NanProd_LargeFloatArray_SIMD()
    {
        // Test SIMD path for nanprod with small values to avoid overflow
        var data = new float[64];
        float expectedProd = 1.0f;
        for (int i = 0; i < 64; i++)
        {
            if (i % 4 == 0)
            {
                data[i] = float.NaN;
            }
            else
            {
                data[i] = 1.1f;
                expectedProd *= 1.1f;
            }
        }
        var arr = np.array(data);
        var result = np.nanprod(arr);
        // Use tolerance for floating point comparison
        var resultVal = (float)result.GetAtIndex(0);
        Assert.IsTrue(Math.Abs(resultVal - expectedProd) < 1e-3f);
    }

    #endregion

    #region Edge Cases

    [Test]
    public void NanSum_SingleNaN_ReturnsZero()
    {
        var arr = np.array(new double[] { double.NaN });
        var result = np.nansum(arr);
        Assert.AreEqual(0.0, (double)result.GetAtIndex(0), 1e-10);
    }

    [Test]
    public void NanProd_SingleNaN_ReturnsOne()
    {
        var arr = np.array(new double[] { double.NaN });
        var result = np.nanprod(arr);
        Assert.AreEqual(1.0, (double)result.GetAtIndex(0), 1e-10);
    }

    [Test]
    public void NanMin_SingleNaN_ReturnsNaN()
    {
        var arr = np.array(new double[] { double.NaN });
        var result = np.nanmin(arr);
        Assert.IsTrue(double.IsNaN((double)result.GetAtIndex(0)));
    }

    [Test]
    public void NanMax_SingleNaN_ReturnsNaN()
    {
        var arr = np.array(new double[] { double.NaN });
        var result = np.nanmax(arr);
        Assert.IsTrue(double.IsNaN((double)result.GetAtIndex(0)));
    }

    [Test]
    public void NanSum_SingleValue_ReturnsThatValue()
    {
        var arr = np.array(new double[] { 42.0 });
        var result = np.nansum(arr);
        Assert.AreEqual(42.0, (double)result.GetAtIndex(0), 1e-10);
    }

    [Test]
    public void NanMin_WithInfinity_IgnoresNaN()
    {
        // NumPy: np.nanmin([np.nan, np.inf, 1.0]) == 1.0
        var arr = np.array(new double[] { double.NaN, double.PositiveInfinity, 1.0 });
        var result = np.nanmin(arr);
        Assert.AreEqual(1.0, (double)result.GetAtIndex(0), 1e-10);
    }

    [Test]
    public void NanMax_WithNegInfinity_IgnoresNaN()
    {
        // NumPy: np.nanmax([np.nan, -np.inf, 1.0]) == 1.0
        var arr = np.array(new double[] { double.NaN, double.NegativeInfinity, 1.0 });
        var result = np.nanmax(arr);
        Assert.AreEqual(1.0, (double)result.GetAtIndex(0), 1e-10);
    }

    #endregion

    #region Keepdims Tests

    [Test]
    public void NanSum_Keepdims_PreservesShape()
    {
        var arr = np.array(new double[] { 1.0, double.NaN, 3.0 }).reshape(1, 3);
        var result = np.nansum(arr, keepdims: true);
        Assert.AreEqual(2, result.ndim);
        Assert.AreEqual(1, result.shape[0]);
        Assert.AreEqual(1, result.shape[1]);
    }

    [Test]
    public void NanProd_Keepdims_PreservesShape()
    {
        var arr = np.array(new double[] { 2.0, double.NaN, 3.0 }).reshape(1, 3);
        var result = np.nanprod(arr, keepdims: true);
        Assert.AreEqual(2, result.ndim);
        Assert.AreEqual(1, result.shape[0]);
        Assert.AreEqual(1, result.shape[1]);
    }

    #endregion
}
