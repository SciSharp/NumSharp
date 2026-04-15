using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Backends.Kernels;

/// <summary>
/// Comprehensive tests for np.cumsum operation.
/// All expected values are verified against NumPy 2.4.2 output.
///
/// Test coverage:
/// - 1D, 2D, 3D arrays
/// - Axis variations (None, 0, 1, 2, -1)
/// - Data types (int32, int64, float32, float64)
/// - Edge cases (single element, empty arrays)
/// - Shape variations (square, rectangular, higher dimensional)
/// - Type promotion (int32 -> int64)
/// </summary>
public class CumSumComprehensiveTests
{
    #region CumSum 1D Tests

    [TestMethod]
    public void CumSum_1D_Int32()
    {
        // NumPy: np.cumsum([1, 2, 3, 4, 5]) = [1, 3, 6, 10, 15]
        // Note: NumPy returns int64 for int32 input
        var arr = np.array(new int[] { 1, 2, 3, 4, 5 });
        var result = np.cumsum(arr);
        result.Should().BeShaped(5);
        result.Should().BeOfValues(1L, 3L, 6L, 10L, 15L);
    }

    [TestMethod]
    public void CumSum_1D_Int64()
    {
        // NumPy: np.cumsum([1, 2, 3, 4, 5], dtype=int64) = [1, 3, 6, 10, 15]
        var arr = np.array(new long[] { 1L, 2L, 3L, 4L, 5L });
        var result = np.cumsum(arr);
        result.Should().BeShaped(5);
        result.Should().BeOfValues(1L, 3L, 6L, 10L, 15L);
    }

    [TestMethod]
    public void CumSum_1D_Float32()
    {
        // NumPy: np.cumsum([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float32) = [1.0, 3.0, 6.0, 10.0, 15.0]
        var arr = np.array(new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f });
        var result = np.cumsum(arr);
        result.Should().BeShaped(5);
        result.Should().BeOfValues(1.0f, 3.0f, 6.0f, 10.0f, 15.0f);
    }

    [TestMethod]
    public void CumSum_1D_Float64()
    {
        // NumPy: np.cumsum([1.0, 2.0, 3.0, 4.0, 5.0]) = [1.0, 3.0, 6.0, 10.0, 15.0]
        var arr = np.array(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var result = np.cumsum(arr);
        result.Should().BeShaped(5);
        result.Should().BeOfValues(1.0, 3.0, 6.0, 10.0, 15.0);
    }

    #endregion

    #region CumSum 2D Tests - No Axis (Flatten)

    [TestMethod]
    public void CumSum_2D_NoAxis()
    {
        // NumPy: np.cumsum([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) = [1, 3, 6, 10, 15, 21, 28, 36, 45]
        // Returns flattened result
        var arr = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } });
        var result = np.cumsum(arr);
        result.Should().BeShaped(9);
        result.Should().BeOfValues(1L, 3L, 6L, 10L, 15L, 21L, 28L, 36L, 45L);
    }

    #endregion

    #region CumSum 2D Tests - With Axis

    [TestMethod]
    public void CumSum_2D_Axis0()
    {
        // NumPy: np.cumsum([[1, 2, 3], [4, 5, 6], [7, 8, 9]], axis=0) = [[1, 2, 3], [5, 7, 9], [12, 15, 18]]
        var arr = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } });
        var result = np.cumsum(arr, axis: 0);
        result.Should().BeShaped(3, 3);
        result.Should().BeOfValues(1L, 2L, 3L, 5L, 7L, 9L, 12L, 15L, 18L);
    }

    [TestMethod]
    public void CumSum_2D_Axis1()
    {
        // NumPy: np.cumsum([[1, 2, 3], [4, 5, 6], [7, 8, 9]], axis=1) = [[1, 3, 6], [4, 9, 15], [7, 15, 24]]
        var arr = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } });
        var result = np.cumsum(arr, axis: 1);
        result.Should().BeShaped(3, 3);
        result.Should().BeOfValues(1L, 3L, 6L, 4L, 9L, 15L, 7L, 15L, 24L);
    }

    [TestMethod]
    public void CumSum_2D_AxisNeg1()
    {
        // NumPy: np.cumsum([[1, 2, 3], [4, 5, 6], [7, 8, 9]], axis=-1) = [[1, 3, 6], [4, 9, 15], [7, 15, 24]]
        // axis=-1 is equivalent to axis=1 for 2D
        var arr = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } });
        var result = np.cumsum(arr, axis: -1);
        result.Should().BeShaped(3, 3);
        result.Should().BeOfValues(1L, 3L, 6L, 4L, 9L, 15L, 7L, 15L, 24L);
    }

    #endregion

    #region CumSum 3D Tests

    [TestMethod]
    public void CumSum_3D_NoAxis()
    {
        // NumPy: np.cumsum([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]) = [1, 3, 6, 10, 15, 21, 28, 36]
        var arr = np.array(new int[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } });
        var result = np.cumsum(arr);
        result.Should().BeShaped(8);
        result.Should().BeOfValues(1L, 3L, 6L, 10L, 15L, 21L, 28L, 36L);
    }

    [TestMethod]
    public void CumSum_3D_Axis0()
    {
        // NumPy: np.cumsum([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], axis=0) = [[[1, 2], [3, 4]], [[6, 8], [10, 12]]]
        var arr = np.array(new int[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } });
        var result = np.cumsum(arr, axis: 0);
        result.Should().BeShaped(2, 2, 2);
        result.Should().BeOfValues(1L, 2L, 3L, 4L, 6L, 8L, 10L, 12L);
    }

    [TestMethod]
    public void CumSum_3D_Axis1()
    {
        // NumPy: np.cumsum([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], axis=1) = [[[1, 2], [4, 6]], [[5, 6], [12, 14]]]
        var arr = np.array(new int[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } });
        var result = np.cumsum(arr, axis: 1);
        result.Should().BeShaped(2, 2, 2);
        result.Should().BeOfValues(1L, 2L, 4L, 6L, 5L, 6L, 12L, 14L);
    }

    [TestMethod]
    public void CumSum_3D_Axis2()
    {
        // NumPy: np.cumsum([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], axis=2) = [[[1, 3], [3, 7]], [[5, 11], [7, 15]]]
        var arr = np.array(new int[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } });
        var result = np.cumsum(arr, axis: 2);
        result.Should().BeShaped(2, 2, 2);
        result.Should().BeOfValues(1L, 3L, 3L, 7L, 5L, 11L, 7L, 15L);
    }

    #endregion

    #region CumSum Edge Cases

    [TestMethod]
    public void CumSum_SingleElement()
    {
        // NumPy: np.cumsum([42]) = [42]
        var arr = np.array(new int[] { 42 });
        var result = np.cumsum(arr);
        result.Should().BeShaped(1);
        result.Should().BeOfValues(42L);
    }

    [TestMethod]
    public void CumSum_EmptyArray()
    {
        // NumPy: np.cumsum([]) = []
        var arr = np.array(Array.Empty<double>());
        var result = np.cumsum(arr);
        result.Should().BeShaped(0);
    }

    #endregion

    #region CumSum Rectangular Arrays

    [TestMethod]
    public void CumSum_Rectangular_2x5_Axis0()
    {
        // NumPy: np.cumsum([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], axis=0) = [[1, 2, 3, 4, 5], [7, 9, 11, 13, 15]]
        var arr = np.array(new int[,] { { 1, 2, 3, 4, 5 }, { 6, 7, 8, 9, 10 } });
        var result = np.cumsum(arr, axis: 0);
        result.Should().BeShaped(2, 5);
        result.Should().BeOfValues(1L, 2L, 3L, 4L, 5L, 7L, 9L, 11L, 13L, 15L);
    }

    [TestMethod]
    public void CumSum_Rectangular_2x5_Axis1()
    {
        // NumPy: np.cumsum([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], axis=1) = [[1, 3, 6, 10, 15], [6, 13, 21, 30, 40]]
        var arr = np.array(new int[,] { { 1, 2, 3, 4, 5 }, { 6, 7, 8, 9, 10 } });
        var result = np.cumsum(arr, axis: 1);
        result.Should().BeShaped(2, 5);
        result.Should().BeOfValues(1L, 3L, 6L, 10L, 15L, 6L, 13L, 21L, 30L, 40L);
    }

    #endregion

    #region CumSum Square Arrays

    [TestMethod]
    public void CumSum_Square_4x4_Axis0()
    {
        // NumPy: np.cumsum([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], axis=0)
        // = [[1, 2, 3, 4], [6, 8, 10, 12], [15, 18, 21, 24], [28, 32, 36, 40]]
        var arr = np.array(new int[,] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 }, { 9, 10, 11, 12 }, { 13, 14, 15, 16 } });
        var result = np.cumsum(arr, axis: 0);
        result.Should().BeShaped(4, 4);
        result.Should().BeOfValues(1L, 2L, 3L, 4L, 6L, 8L, 10L, 12L, 15L, 18L, 21L, 24L, 28L, 32L, 36L, 40L);
    }

    [TestMethod]
    public void CumSum_Square_4x4_Axis1()
    {
        // NumPy: np.cumsum([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], axis=1)
        // = [[1, 3, 6, 10], [5, 11, 18, 26], [9, 19, 30, 42], [13, 27, 42, 58]]
        var arr = np.array(new int[,] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 }, { 9, 10, 11, 12 }, { 13, 14, 15, 16 } });
        var result = np.cumsum(arr, axis: 1);
        result.Should().BeShaped(4, 4);
        result.Should().BeOfValues(1L, 3L, 6L, 10L, 5L, 11L, 18L, 26L, 9L, 19L, 30L, 42L, 13L, 27L, 42L, 58L);
    }

    #endregion

    #region CumSum Higher Dimensional (2x3x4)

    [TestMethod]
    public void CumSum_HigherDim_2x3x4_NoAxis()
    {
        // NumPy: np.cumsum(np.arange(24).reshape(2, 3, 4))
        // = [0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136, 153, 171, 190, 210, 231, 253, 276]
        var arr = np.arange(24).reshape(2, 3, 4);
        var result = np.cumsum(arr);
        result.Should().BeShaped(24);
        result.Should().BeOfValues(
            0L, 1L, 3L, 6L, 10L, 15L, 21L, 28L, 36L, 45L, 55L, 66L,
            78L, 91L, 105L, 120L, 136L, 153L, 171L, 190L, 210L, 231L, 253L, 276L
        );
    }

    [TestMethod]
    public void CumSum_HigherDim_2x3x4_Axis0_Shape()
    {
        // NumPy: np.cumsum(np.arange(24).reshape(2, 3, 4), axis=0).shape = (2, 3, 4)
        var arr = np.arange(24).reshape(2, 3, 4);
        var result = np.cumsum(arr, axis: 0);
        result.Should().BeShaped(2, 3, 4);
    }

    [TestMethod]
    public void CumSum_HigherDim_2x3x4_Axis1_Shape()
    {
        // NumPy: np.cumsum(np.arange(24).reshape(2, 3, 4), axis=1).shape = (2, 3, 4)
        var arr = np.arange(24).reshape(2, 3, 4);
        var result = np.cumsum(arr, axis: 1);
        result.Should().BeShaped(2, 3, 4);
    }

    [TestMethod]
    public void CumSum_HigherDim_2x3x4_Axis2_Shape()
    {
        // NumPy: np.cumsum(np.arange(24).reshape(2, 3, 4), axis=2).shape = (2, 3, 4)
        var arr = np.arange(24).reshape(2, 3, 4);
        var result = np.cumsum(arr, axis: 2);
        result.Should().BeShaped(2, 3, 4);
    }

    #endregion

    #region CumSum Type Promotion

    [TestMethod]
    public void CumSum_Int32_ReturnsInt64()
    {
        // NumPy 2.x (NEP50): cumsum of int32 returns int64
        var arr = np.array(new int[] { 1, 2, 3, 4, 5 });
        var result = np.cumsum(arr);
        Assert.AreEqual(NPTypeCode.Int64, result.typecode);
    }

    [TestMethod]
    public void CumSum_Int64_ReturnsInt64()
    {
        // NumPy: cumsum of int64 returns int64
        var arr = np.array(new long[] { 1L, 2L, 3L, 4L, 5L });
        var result = np.cumsum(arr);
        Assert.AreEqual(NPTypeCode.Int64, result.typecode);
    }

    [TestMethod]
    public void CumSum_Float32_ReturnsFloat32()
    {
        // NumPy: cumsum of float32 returns float32
        var arr = np.array(new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f });
        var result = np.cumsum(arr);
        Assert.AreEqual(NPTypeCode.Single, result.typecode);
    }

    [TestMethod]
    public void CumSum_Float64_ReturnsFloat64()
    {
        // NumPy: cumsum of float64 returns float64
        var arr = np.array(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var result = np.cumsum(arr);
        Assert.AreEqual(NPTypeCode.Double, result.typecode);
    }

    #endregion

    #region CumSum Float Values

    [TestMethod]
    public void CumSum_Float64_2D_NoAxis()
    {
        // NumPy: np.cumsum([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) = [1.0, 3.0, 6.0, 10.0, 15.0, 21.0]
        var arr = np.array(new double[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } });
        var result = np.cumsum(arr);
        result.Should().BeShaped(6);
        result.Should().BeOfValues(1.0, 3.0, 6.0, 10.0, 15.0, 21.0);
    }

    [TestMethod]
    public void CumSum_Float64_2D_Axis0()
    {
        // NumPy: np.cumsum([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], axis=0) = [[1.0, 2.0, 3.0], [5.0, 7.0, 9.0]]
        var arr = np.array(new double[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } });
        var result = np.cumsum(arr, axis: 0);
        result.Should().BeShaped(2, 3);
        result.Should().BeOfValues(1.0, 2.0, 3.0, 5.0, 7.0, 9.0);
    }

    [TestMethod]
    public void CumSum_Float64_2D_Axis1()
    {
        // NumPy: np.cumsum([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], axis=1) = [[1.0, 3.0, 6.0], [4.0, 9.0, 15.0]]
        var arr = np.array(new double[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } });
        var result = np.cumsum(arr, axis: 1);
        result.Should().BeShaped(2, 3);
        result.Should().BeOfValues(1.0, 3.0, 6.0, 4.0, 9.0, 15.0);
    }

    #endregion

    #region CumSum Negative Axis

    [TestMethod]
    public void CumSum_3D_AxisNegative1()
    {
        // NumPy: np.cumsum([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], axis=-1)
        // axis=-1 for 3D is axis=2
        var arr = np.array(new int[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } });
        var result = np.cumsum(arr, axis: -1);
        result.Should().BeShaped(2, 2, 2);
        // Same as axis=2: [[[1, 3], [3, 7]], [[5, 11], [7, 15]]]
        result.Should().BeOfValues(1L, 3L, 3L, 7L, 5L, 11L, 7L, 15L);
    }

    [TestMethod]
    public void CumSum_3D_AxisNegative2()
    {
        // NumPy: np.cumsum([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], axis=-2)
        // axis=-2 for 3D is axis=1
        var arr = np.array(new int[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } });
        var result = np.cumsum(arr, axis: -2);
        result.Should().BeShaped(2, 2, 2);
        // Same as axis=1: [[[1, 2], [4, 6]], [[5, 6], [12, 14]]]
        result.Should().BeOfValues(1L, 2L, 4L, 6L, 5L, 6L, 12L, 14L);
    }

    #endregion
}
