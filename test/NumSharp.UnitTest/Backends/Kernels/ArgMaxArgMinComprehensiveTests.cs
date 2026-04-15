using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Backends.Kernels;

/// <summary>
/// Comprehensive tests for np.argmax and np.argmin operations.
/// All expected values are verified against NumPy 2.4.2 output.
///
/// Test coverage:
/// - 1D, 2D, 3D arrays
/// - Axis variations (None, 0, 1, 2, -1)
/// - Data types (int32, int64, float32, float64)
/// - Edge cases (single element, identical values, NaN handling)
/// - Shape variations (square, rectangular, higher dimensional)
/// </summary>
public class ArgMaxArgMinComprehensiveTests
{
    #region ArgMax 1D Tests

    [TestMethod]
    public void ArgMax_1D_Int32()
    {
        // NumPy: np.argmax(np.array([1, 3, 2, 5, 4])) = 3
        var arr = np.array(new int[] { 1, 3, 2, 5, 4 });
        var result = np.argmax(arr);
        Assert.AreEqual(3L, (long)result);
    }

    [TestMethod]
    public void ArgMax_1D_Int64()
    {
        // NumPy: np.argmax(np.array([1, 3, 2, 5, 4], dtype=np.int64)) = 3
        var arr = np.array(new long[] { 1L, 3L, 2L, 5L, 4L });
        var result = np.argmax(arr);
        Assert.AreEqual(3L, (long)result);
    }

    [TestMethod]
    public void ArgMax_1D_Float32()
    {
        // NumPy: np.argmax(np.array([1.1, 3.3, 2.2, 5.5, 4.4], dtype=np.float32)) = 3
        var arr = np.array(new float[] { 1.1f, 3.3f, 2.2f, 5.5f, 4.4f });
        var result = np.argmax(arr);
        Assert.AreEqual(3L, (long)result);
    }

    [TestMethod]
    public void ArgMax_1D_Float64()
    {
        // NumPy: np.argmax(np.array([1.1, 3.3, 2.2, 5.5, 4.4], dtype=np.float64)) = 3
        var arr = np.array(new double[] { 1.1, 3.3, 2.2, 5.5, 4.4 });
        var result = np.argmax(arr);
        Assert.AreEqual(3L, (long)result);
    }

    #endregion

    #region ArgMax 2D Tests

    [TestMethod]
    public void ArgMax_2D_NoAxis()
    {
        // NumPy: np.argmax([[1, 5, 3], [7, 2, 8], [4, 6, 0]]) = 5
        // Flattened array: [1, 5, 3, 7, 2, 8, 4, 6, 0], max is 8 at index 5
        var arr = np.array(new int[,] { { 1, 5, 3 }, { 7, 2, 8 }, { 4, 6, 0 } });
        var result = np.argmax(arr);
        Assert.AreEqual(5L, (long)result);
    }

    [TestMethod]
    public void ArgMax_2D_Axis0()
    {
        // NumPy: np.argmax([[1, 5, 3], [7, 2, 8], [4, 6, 0]], axis=0) = [1, 2, 1]
        var arr = np.array(new int[,] { { 1, 5, 3 }, { 7, 2, 8 }, { 4, 6, 0 } });
        var result = np.argmax(arr, axis: 0);
        result.Should().BeShaped(3);
        result.Should().BeOfValues(1L, 2L, 1L);
    }

    [TestMethod]
    public void ArgMax_2D_Axis1()
    {
        // NumPy: np.argmax([[1, 5, 3], [7, 2, 8], [4, 6, 0]], axis=1) = [1, 2, 1]
        var arr = np.array(new int[,] { { 1, 5, 3 }, { 7, 2, 8 }, { 4, 6, 0 } });
        var result = np.argmax(arr, axis: 1);
        result.Should().BeShaped(3);
        result.Should().BeOfValues(1L, 2L, 1L);
    }

    [TestMethod]
    public void ArgMax_2D_AxisNeg1()
    {
        // NumPy: np.argmax([[1, 5, 3], [7, 2, 8], [4, 6, 0]], axis=-1) = [1, 2, 1]
        // axis=-1 is equivalent to axis=1 for 2D
        var arr = np.array(new int[,] { { 1, 5, 3 }, { 7, 2, 8 }, { 4, 6, 0 } });
        var result = np.argmax(arr, axis: -1);
        result.Should().BeShaped(3);
        result.Should().BeOfValues(1L, 2L, 1L);
    }

    #endregion

    #region ArgMax 3D Tests

    [TestMethod]
    public void ArgMax_3D_NoAxis()
    {
        // NumPy: np.argmax([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]) = 11
        var arr = np.array(new int[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } }, { { 9, 10 }, { 11, 12 } } });
        var result = np.argmax(arr);
        Assert.AreEqual(11L, (long)result);
    }

    [TestMethod]
    public void ArgMax_3D_Axis0()
    {
        // NumPy: np.argmax([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], axis=0) = [[2, 2], [2, 2]]
        var arr = np.array(new int[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } }, { { 9, 10 }, { 11, 12 } } });
        var result = np.argmax(arr, axis: 0);
        result.Should().BeShaped(2, 2);
        result.Should().BeOfValues(2L, 2L, 2L, 2L);
    }

    [TestMethod]
    public void ArgMax_3D_Axis1()
    {
        // NumPy: np.argmax([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], axis=1) = [[1, 1], [1, 1], [1, 1]]
        var arr = np.array(new int[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } }, { { 9, 10 }, { 11, 12 } } });
        var result = np.argmax(arr, axis: 1);
        result.Should().BeShaped(3, 2);
        result.Should().BeOfValues(1L, 1L, 1L, 1L, 1L, 1L);
    }

    [TestMethod]
    public void ArgMax_3D_Axis2()
    {
        // NumPy: np.argmax([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], axis=2) = [[1, 1], [1, 1], [1, 1]]
        var arr = np.array(new int[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } }, { { 9, 10 }, { 11, 12 } } });
        var result = np.argmax(arr, axis: 2);
        result.Should().BeShaped(3, 2);
        result.Should().BeOfValues(1L, 1L, 1L, 1L, 1L, 1L);
    }

    #endregion

    #region ArgMax Edge Cases

    [TestMethod]
    public void ArgMax_SingleElement()
    {
        // NumPy: np.argmax([42]) = 0
        var arr = np.array(new int[] { 42 });
        var result = np.argmax(arr);
        Assert.AreEqual(0L, (long)result);
    }

    [TestMethod]
    public void ArgMax_IdenticalValues()
    {
        // NumPy: np.argmax([5, 5, 5, 5]) = 0 (returns first occurrence)
        var arr = np.array(new int[] { 5, 5, 5, 5 });
        var result = np.argmax(arr);
        Assert.AreEqual(0L, (long)result);
    }

    [TestMethod]
    public void ArgMax_WithNaN()
    {
        // NumPy: np.argmax([1.0, nan, 3.0, 2.0]) = 1 (NaN returns its index)
        var arr = np.array(new double[] { 1.0, double.NaN, 3.0, 2.0 });
        var result = np.argmax(arr);
        Assert.AreEqual(1L, (long)result);
    }

    #endregion

    #region ArgMax Rectangular Arrays

    [TestMethod]
    public void ArgMax_Rectangular_2x5_Axis0()
    {
        // NumPy: np.argmax([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], axis=0) = [1, 1, 1, 1, 1]
        var arr = np.array(new int[,] { { 1, 2, 3, 4, 5 }, { 6, 7, 8, 9, 10 } });
        var result = np.argmax(arr, axis: 0);
        result.Should().BeShaped(5);
        result.Should().BeOfValues(1L, 1L, 1L, 1L, 1L);
    }

    [TestMethod]
    public void ArgMax_Rectangular_2x5_Axis1()
    {
        // NumPy: np.argmax([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], axis=1) = [4, 4]
        var arr = np.array(new int[,] { { 1, 2, 3, 4, 5 }, { 6, 7, 8, 9, 10 } });
        var result = np.argmax(arr, axis: 1);
        result.Should().BeShaped(2);
        result.Should().BeOfValues(4L, 4L);
    }

    #endregion

    #region ArgMax Square Arrays

    [TestMethod]
    public void ArgMax_Square_4x4_Axis0()
    {
        // NumPy: np.argmax([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], axis=0) = [3, 3, 3, 3]
        var arr = np.array(new int[,] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 }, { 9, 10, 11, 12 }, { 13, 14, 15, 16 } });
        var result = np.argmax(arr, axis: 0);
        result.Should().BeShaped(4);
        result.Should().BeOfValues(3L, 3L, 3L, 3L);
    }

    [TestMethod]
    public void ArgMax_Square_4x4_Axis1()
    {
        // NumPy: np.argmax([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], axis=1) = [3, 3, 3, 3]
        var arr = np.array(new int[,] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 }, { 9, 10, 11, 12 }, { 13, 14, 15, 16 } });
        var result = np.argmax(arr, axis: 1);
        result.Should().BeShaped(4);
        result.Should().BeOfValues(3L, 3L, 3L, 3L);
    }

    #endregion

    #region ArgMax Higher Dimensional

    [TestMethod]
    public void ArgMax_3D_2x3x4_AxisNeg1()
    {
        // NumPy: np.argmax(np.arange(24).reshape(2, 3, 4), axis=-1) = [[3, 3, 3], [3, 3, 3]]
        var arr = np.arange(24).reshape(2, 3, 4);
        var result = np.argmax(arr, axis: -1);
        result.Should().BeShaped(2, 3);
        result.Should().BeOfValues(3L, 3L, 3L, 3L, 3L, 3L);
    }

    [TestMethod]
    public void ArgMax_3D_2x3x4_AxisNeg2()
    {
        // NumPy: np.argmax(np.arange(24).reshape(2, 3, 4), axis=-2) = [[2, 2, 2, 2], [2, 2, 2, 2]]
        var arr = np.arange(24).reshape(2, 3, 4);
        var result = np.argmax(arr, axis: -2);
        result.Should().BeShaped(2, 4);
        result.Should().BeOfValues(2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L);
    }

    [TestMethod]
    public void ArgMax_3D_2x3x4_AxisNeg3()
    {
        // NumPy: np.argmax(np.arange(24).reshape(2, 3, 4), axis=-3) = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        var arr = np.arange(24).reshape(2, 3, 4);
        var result = np.argmax(arr, axis: -3);
        result.Should().BeShaped(3, 4);
        result.Should().BeOfValues(1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L);
    }

    #endregion

    #region ArgMin 1D Tests

    [TestMethod]
    public void ArgMin_1D_Int32()
    {
        // NumPy: np.argmin(np.array([1, 3, 2, 5, 4])) = 0
        var arr = np.array(new int[] { 1, 3, 2, 5, 4 });
        var result = np.argmin(arr);
        Assert.AreEqual(0L, (long)result);
    }

    [TestMethod]
    public void ArgMin_1D_Int64()
    {
        // NumPy: np.argmin(np.array([1, 3, 2, 5, 4], dtype=np.int64)) = 0
        var arr = np.array(new long[] { 1L, 3L, 2L, 5L, 4L });
        var result = np.argmin(arr);
        Assert.AreEqual(0L, (long)result);
    }

    [TestMethod]
    public void ArgMin_1D_Float32()
    {
        // NumPy: np.argmin(np.array([1.1, 3.3, 2.2, 5.5, 4.4], dtype=np.float32)) = 0
        var arr = np.array(new float[] { 1.1f, 3.3f, 2.2f, 5.5f, 4.4f });
        var result = np.argmin(arr);
        Assert.AreEqual(0L, (long)result);
    }

    [TestMethod]
    public void ArgMin_1D_Float64()
    {
        // NumPy: np.argmin(np.array([1.1, 3.3, 2.2, 5.5, 4.4], dtype=np.float64)) = 0
        var arr = np.array(new double[] { 1.1, 3.3, 2.2, 5.5, 4.4 });
        var result = np.argmin(arr);
        Assert.AreEqual(0L, (long)result);
    }

    #endregion

    #region ArgMin 2D Tests

    [TestMethod]
    public void ArgMin_2D_NoAxis()
    {
        // NumPy: np.argmin([[1, 5, 3], [7, 2, 8], [4, 6, 0]]) = 8
        // Flattened array: [1, 5, 3, 7, 2, 8, 4, 6, 0], min is 0 at index 8
        var arr = np.array(new int[,] { { 1, 5, 3 }, { 7, 2, 8 }, { 4, 6, 0 } });
        var result = np.argmin(arr);
        Assert.AreEqual(8L, (long)result);
    }

    [TestMethod]
    public void ArgMin_2D_Axis0()
    {
        // NumPy: np.argmin([[1, 5, 3], [7, 2, 8], [4, 6, 0]], axis=0) = [0, 1, 2]
        var arr = np.array(new int[,] { { 1, 5, 3 }, { 7, 2, 8 }, { 4, 6, 0 } });
        var result = np.argmin(arr, axis: 0);
        result.Should().BeShaped(3);
        result.Should().BeOfValues(0L, 1L, 2L);
    }

    [TestMethod]
    public void ArgMin_2D_Axis1()
    {
        // NumPy: np.argmin([[1, 5, 3], [7, 2, 8], [4, 6, 0]], axis=1) = [0, 1, 2]
        var arr = np.array(new int[,] { { 1, 5, 3 }, { 7, 2, 8 }, { 4, 6, 0 } });
        var result = np.argmin(arr, axis: 1);
        result.Should().BeShaped(3);
        result.Should().BeOfValues(0L, 1L, 2L);
    }

    [TestMethod]
    public void ArgMin_2D_AxisNeg1()
    {
        // NumPy: np.argmin([[1, 5, 3], [7, 2, 8], [4, 6, 0]], axis=-1) = [0, 1, 2]
        var arr = np.array(new int[,] { { 1, 5, 3 }, { 7, 2, 8 }, { 4, 6, 0 } });
        var result = np.argmin(arr, axis: -1);
        result.Should().BeShaped(3);
        result.Should().BeOfValues(0L, 1L, 2L);
    }

    #endregion

    #region ArgMin 3D Tests

    [TestMethod]
    public void ArgMin_3D_NoAxis()
    {
        // NumPy: np.argmin([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]) = 0
        var arr = np.array(new int[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } }, { { 9, 10 }, { 11, 12 } } });
        var result = np.argmin(arr);
        Assert.AreEqual(0L, (long)result);
    }

    [TestMethod]
    public void ArgMin_3D_Axis0()
    {
        // NumPy: np.argmin([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], axis=0) = [[0, 0], [0, 0]]
        var arr = np.array(new int[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } }, { { 9, 10 }, { 11, 12 } } });
        var result = np.argmin(arr, axis: 0);
        result.Should().BeShaped(2, 2);
        result.Should().BeOfValues(0L, 0L, 0L, 0L);
    }

    [TestMethod]
    public void ArgMin_3D_Axis1()
    {
        // NumPy: np.argmin([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], axis=1) = [[0, 0], [0, 0], [0, 0]]
        var arr = np.array(new int[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } }, { { 9, 10 }, { 11, 12 } } });
        var result = np.argmin(arr, axis: 1);
        result.Should().BeShaped(3, 2);
        result.Should().BeOfValues(0L, 0L, 0L, 0L, 0L, 0L);
    }

    [TestMethod]
    public void ArgMin_3D_Axis2()
    {
        // NumPy: np.argmin([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], axis=2) = [[0, 0], [0, 0], [0, 0]]
        var arr = np.array(new int[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } }, { { 9, 10 }, { 11, 12 } } });
        var result = np.argmin(arr, axis: 2);
        result.Should().BeShaped(3, 2);
        result.Should().BeOfValues(0L, 0L, 0L, 0L, 0L, 0L);
    }

    #endregion

    #region ArgMin Edge Cases

    [TestMethod]
    public void ArgMin_SingleElement()
    {
        // NumPy: np.argmin([42]) = 0
        var arr = np.array(new int[] { 42 });
        var result = np.argmin(arr);
        Assert.AreEqual(0L, (long)result);
    }

    [TestMethod]
    public void ArgMin_IdenticalValues()
    {
        // NumPy: np.argmin([5, 5, 5, 5]) = 0 (returns first occurrence)
        var arr = np.array(new int[] { 5, 5, 5, 5 });
        var result = np.argmin(arr);
        Assert.AreEqual(0L, (long)result);
    }

    [TestMethod]
    public void ArgMin_WithNaN()
    {
        // NumPy: np.argmin([1.0, nan, 3.0, 2.0]) = 1 (NaN returns its index)
        var arr = np.array(new double[] { 1.0, double.NaN, 3.0, 2.0 });
        var result = np.argmin(arr);
        Assert.AreEqual(1L, (long)result);
    }

    #endregion

    #region ArgMin Rectangular Arrays

    [TestMethod]
    public void ArgMin_Rectangular_2x5_Axis0()
    {
        // NumPy: np.argmin([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], axis=0) = [0, 0, 0, 0, 0]
        var arr = np.array(new int[,] { { 1, 2, 3, 4, 5 }, { 6, 7, 8, 9, 10 } });
        var result = np.argmin(arr, axis: 0);
        result.Should().BeShaped(5);
        result.Should().BeOfValues(0L, 0L, 0L, 0L, 0L);
    }

    [TestMethod]
    public void ArgMin_Rectangular_2x5_Axis1()
    {
        // NumPy: np.argmin([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], axis=1) = [0, 0]
        var arr = np.array(new int[,] { { 1, 2, 3, 4, 5 }, { 6, 7, 8, 9, 10 } });
        var result = np.argmin(arr, axis: 1);
        result.Should().BeShaped(2);
        result.Should().BeOfValues(0L, 0L);
    }

    #endregion

    #region ArgMin Higher Dimensional

    [TestMethod]
    public void ArgMin_3D_2x3x4_AxisNeg1()
    {
        // NumPy: np.argmin(np.arange(24).reshape(2, 3, 4), axis=-1) = [[0, 0, 0], [0, 0, 0]]
        var arr = np.arange(24).reshape(2, 3, 4);
        var result = np.argmin(arr, axis: -1);
        result.Should().BeShaped(2, 3);
        result.Should().BeOfValues(0L, 0L, 0L, 0L, 0L, 0L);
    }

    #endregion

    #region keepdims Tests

    [TestMethod]
    public void ArgMax_2D_Axis0_Keepdims()
    {
        // NumPy: np.argmax([[1, 2, 3], [4, 5, 6]], axis=0, keepdims=True).shape = (1, 3)
        var arr = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        var result = np.argmax(arr, axis: 0, keepdims: true);
        result.Should().BeShaped(1, 3);
    }

    [TestMethod]
    public void ArgMax_2D_Axis1_Keepdims()
    {
        // NumPy: np.argmax([[1, 2, 3], [4, 5, 6]], axis=1, keepdims=True).shape = (2, 1)
        var arr = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        var result = np.argmax(arr, axis: 1, keepdims: true);
        result.Should().BeShaped(2, 1);
    }

    #endregion
}
