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

    #region Axis Reduction Tests

    [Test]
    public void NanSum_Axis0_2D_IgnoresNaN()
    {
        // NumPy: np.nansum([[1, np.nan], [np.nan, 4]], axis=0) == [1, 4]
        var arr = np.array(new double[,] { { 1.0, double.NaN }, { double.NaN, 4.0 } });
        var result = np.nansum(arr, axis: 0);
        Assert.AreEqual(1, result.ndim);
        Assert.AreEqual(2, result.size);
        Assert.AreEqual(1.0, (double)result.GetAtIndex(0), 1e-10);
        Assert.AreEqual(4.0, (double)result.GetAtIndex(1), 1e-10);
    }

    [Test]
    public void NanSum_Axis1_2D_IgnoresNaN()
    {
        // NumPy: np.nansum([[1, np.nan], [np.nan, 4]], axis=1) == [1, 4]
        var arr = np.array(new double[,] { { 1.0, double.NaN }, { double.NaN, 4.0 } });
        var result = np.nansum(arr, axis: 1);
        Assert.AreEqual(1, result.ndim);
        Assert.AreEqual(2, result.size);
        Assert.AreEqual(1.0, (double)result.GetAtIndex(0), 1e-10);
        Assert.AreEqual(4.0, (double)result.GetAtIndex(1), 1e-10);
    }

    [Test]
    public void NanSum_Axis_AllNaNRow_ReturnsZero()
    {
        // NumPy: np.nansum([[np.nan, np.nan], [1.0, 2.0]], axis=1) == [0.0, 3.0]
        var arr = np.array(new double[,] { { double.NaN, double.NaN }, { 1.0, 2.0 } });
        var result = np.nansum(arr, axis: 1);
        Assert.AreEqual(0.0, (double)result.GetAtIndex(0), 1e-10);
        Assert.AreEqual(3.0, (double)result.GetAtIndex(1), 1e-10);
    }

    [Test]
    public void NanProd_Axis0_2D_IgnoresNaN()
    {
        // NumPy: np.nanprod([[1, np.nan], [np.nan, 4]], axis=0) == [1, 4]
        var arr = np.array(new double[,] { { 1.0, double.NaN }, { double.NaN, 4.0 } });
        var result = np.nanprod(arr, axis: 0);
        Assert.AreEqual(1, result.ndim);
        Assert.AreEqual(2, result.size);
        Assert.AreEqual(1.0, (double)result.GetAtIndex(0), 1e-10);
        Assert.AreEqual(4.0, (double)result.GetAtIndex(1), 1e-10);
    }

    [Test]
    public void NanProd_Axis1_2D_IgnoresNaN()
    {
        // NumPy: np.nanprod([[2, np.nan], [np.nan, 3]], axis=1) == [2, 3]
        var arr = np.array(new double[,] { { 2.0, double.NaN }, { double.NaN, 3.0 } });
        var result = np.nanprod(arr, axis: 1);
        Assert.AreEqual(2.0, (double)result.GetAtIndex(0), 1e-10);
        Assert.AreEqual(3.0, (double)result.GetAtIndex(1), 1e-10);
    }

    [Test]
    public void NanProd_Axis_AllNaNRow_ReturnsOne()
    {
        // NumPy: np.nanprod([[np.nan, np.nan], [1.0, 2.0]], axis=1) == [1.0, 2.0]
        var arr = np.array(new double[,] { { double.NaN, double.NaN }, { 1.0, 2.0 } });
        var result = np.nanprod(arr, axis: 1);
        Assert.AreEqual(1.0, (double)result.GetAtIndex(0), 1e-10);
        Assert.AreEqual(2.0, (double)result.GetAtIndex(1), 1e-10);
    }

    [Test]
    public void NanMin_Axis0_2D_IgnoresNaN()
    {
        // NumPy: np.nanmin([[1, np.nan], [np.nan, 4]], axis=0) == [1, 4]
        var arr = np.array(new double[,] { { 1.0, double.NaN }, { double.NaN, 4.0 } });
        var result = np.nanmin(arr, axis: 0);
        Assert.AreEqual(1, result.ndim);
        Assert.AreEqual(2, result.size);
        Assert.AreEqual(1.0, (double)result.GetAtIndex(0), 1e-10);
        Assert.AreEqual(4.0, (double)result.GetAtIndex(1), 1e-10);
    }

    [Test]
    public void NanMin_Axis1_2D_IgnoresNaN()
    {
        // NumPy: np.nanmin([[5, np.nan, 2], [np.nan, 3, 1]], axis=1) == [2, 1]
        var arr = np.array(new double[,] { { 5.0, double.NaN, 2.0 }, { double.NaN, 3.0, 1.0 } });
        var result = np.nanmin(arr, axis: 1);
        Assert.AreEqual(2.0, (double)result.GetAtIndex(0), 1e-10);
        Assert.AreEqual(1.0, (double)result.GetAtIndex(1), 1e-10);
    }

    [Test]
    public void NanMin_Axis_AllNaNRow_ReturnsNaN()
    {
        // NumPy: np.nanmin([[np.nan, np.nan], [1.0, 2.0]], axis=1) == [nan, 1.0]
        var arr = np.array(new double[,] { { double.NaN, double.NaN }, { 1.0, 2.0 } });
        var result = np.nanmin(arr, axis: 1);
        Assert.IsTrue(double.IsNaN((double)result.GetAtIndex(0)));
        Assert.AreEqual(1.0, (double)result.GetAtIndex(1), 1e-10);
    }

    [Test]
    public void NanMax_Axis0_2D_IgnoresNaN()
    {
        // NumPy: np.nanmax([[1, np.nan], [np.nan, 4]], axis=0) == [1, 4]
        var arr = np.array(new double[,] { { 1.0, double.NaN }, { double.NaN, 4.0 } });
        var result = np.nanmax(arr, axis: 0);
        Assert.AreEqual(1, result.ndim);
        Assert.AreEqual(2, result.size);
        Assert.AreEqual(1.0, (double)result.GetAtIndex(0), 1e-10);
        Assert.AreEqual(4.0, (double)result.GetAtIndex(1), 1e-10);
    }

    [Test]
    public void NanMax_Axis1_2D_IgnoresNaN()
    {
        // NumPy: np.nanmax([[5, np.nan, 2], [np.nan, 3, 1]], axis=1) == [5, 3]
        var arr = np.array(new double[,] { { 5.0, double.NaN, 2.0 }, { double.NaN, 3.0, 1.0 } });
        var result = np.nanmax(arr, axis: 1);
        Assert.AreEqual(5.0, (double)result.GetAtIndex(0), 1e-10);
        Assert.AreEqual(3.0, (double)result.GetAtIndex(1), 1e-10);
    }

    [Test]
    public void NanMax_Axis_AllNaNRow_ReturnsNaN()
    {
        // NumPy: np.nanmax([[np.nan, np.nan], [1.0, 2.0]], axis=1) == [nan, 2.0]
        var arr = np.array(new double[,] { { double.NaN, double.NaN }, { 1.0, 2.0 } });
        var result = np.nanmax(arr, axis: 1);
        Assert.IsTrue(double.IsNaN((double)result.GetAtIndex(0)));
        Assert.AreEqual(2.0, (double)result.GetAtIndex(1), 1e-10);
    }

    [Test]
    public void NanSum_Axis_Keepdims_PreservesShape()
    {
        // NumPy: np.nansum([[1, np.nan], [np.nan, 4]], axis=0, keepdims=True).shape == (1, 2)
        var arr = np.array(new double[,] { { 1.0, double.NaN }, { double.NaN, 4.0 } });
        var result = np.nansum(arr, axis: 0, keepdims: true);
        Assert.AreEqual(2, result.ndim);
        Assert.AreEqual(1, result.shape[0]);
        Assert.AreEqual(2, result.shape[1]);
    }

    [Test]
    public void NanSum_Axis_Float_IgnoresNaN()
    {
        // Same as double but with float32
        var arr = np.array(new float[,] { { 1.0f, float.NaN }, { float.NaN, 4.0f } });
        var result = np.nansum(arr, axis: 0);
        Assert.AreEqual(1.0f, (float)result.GetAtIndex(0), 1e-6f);
        Assert.AreEqual(4.0f, (float)result.GetAtIndex(1), 1e-6f);
    }

    [Test]
    public void NanMin_Axis_Float_IgnoresNaN()
    {
        var arr = np.array(new float[,] { { 5.0f, float.NaN }, { float.NaN, 2.0f } });
        var result = np.nanmin(arr, axis: 0);
        Assert.AreEqual(5.0f, (float)result.GetAtIndex(0), 1e-6f);
        Assert.AreEqual(2.0f, (float)result.GetAtIndex(1), 1e-6f);
    }

    [Test]
    public void NanMax_Axis_Float_IgnoresNaN()
    {
        var arr = np.array(new float[,] { { 5.0f, float.NaN }, { float.NaN, 2.0f } });
        var result = np.nanmax(arr, axis: 0);
        Assert.AreEqual(5.0f, (float)result.GetAtIndex(0), 1e-6f);
        Assert.AreEqual(2.0f, (float)result.GetAtIndex(1), 1e-6f);
    }

    [Test]
    public void NanSum_3D_Axis1()
    {
        // 3D array with NaN, reduce along middle axis
        // Shape: (2, 2, 2) -> (2, 2)
        var data = new double[2, 2, 2];
        data[0, 0, 0] = 1.0; data[0, 0, 1] = 2.0;
        data[0, 1, 0] = double.NaN; data[0, 1, 1] = 3.0;
        data[1, 0, 0] = 4.0; data[1, 0, 1] = double.NaN;
        data[1, 1, 0] = 5.0; data[1, 1, 1] = 6.0;

        var arr = np.array(data);
        var result = np.nansum(arr, axis: 1);

        Assert.AreEqual(2, result.ndim);
        Assert.AreEqual(2, result.shape[0]);
        Assert.AreEqual(2, result.shape[1]);

        // [0, :, 0]: 1 + 0(NaN) = 1
        // [0, :, 1]: 2 + 3 = 5
        // [1, :, 0]: 4 + 5 = 9
        // [1, :, 1]: 0(NaN) + 6 = 6
        Assert.AreEqual(1.0, (double)result[0, 0], 1e-10);
        Assert.AreEqual(5.0, (double)result[0, 1], 1e-10);
        Assert.AreEqual(9.0, (double)result[1, 0], 1e-10);
        Assert.AreEqual(6.0, (double)result[1, 1], 1e-10);
    }

    [Test]
    public void NanSum_NegativeAxis()
    {
        // NumPy: np.nansum([[1, np.nan], [np.nan, 4]], axis=-1) == [1, 4]
        var arr = np.array(new double[,] { { 1.0, double.NaN }, { double.NaN, 4.0 } });
        var result = np.nansum(arr, axis: -1);
        Assert.AreEqual(1, result.ndim);
        Assert.AreEqual(2, result.size);
        Assert.AreEqual(1.0, (double)result.GetAtIndex(0), 1e-10);
        Assert.AreEqual(4.0, (double)result.GetAtIndex(1), 1e-10);
    }

    #endregion
}
