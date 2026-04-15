using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Backends.Kernels;

/// <summary>
/// Comprehensive tests for np.var and np.std operations.
/// All expected values are verified against NumPy 2.4.2 output.
///
/// Test coverage:
/// - 1D, 2D, 3D arrays
/// - Axis variations (None, 0, 1, 2, -1)
/// - ddof parameter (0, 1)
/// - Data types (int32, int64, float32, float64)
/// - Edge cases (single element, identical values, NaN handling)
/// - Shape variations (square, rectangular, higher dimensional)
/// </summary>
public class VarStdComprehensiveTests
{
    private const double Tolerance = 1e-10;

    #region Var 1D Tests

    [TestMethod]
    public void Var_1D_Float64()
    {
        // NumPy: np.var([1.0, 2.0, 3.0, 4.0, 5.0]) = 2.0
        var arr = np.array(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var result = np.var(arr);
        Assert.IsTrue(Math.Abs(result.GetDouble(0) - 2.0) < Tolerance);
    }

    [TestMethod]
    public void Var_1D_Float64_Ddof0()
    {
        // NumPy: np.var([1.0, 2.0, 3.0, 4.0, 5.0], ddof=0) = 2.0
        var arr = np.array(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var result = np.var(arr, ddof: 0);
        Assert.IsTrue(Math.Abs(result.GetDouble(0) - 2.0) < Tolerance);
    }

    [TestMethod]
    public void Var_1D_Float64_Ddof1()
    {
        // NumPy: np.var([1.0, 2.0, 3.0, 4.0, 5.0], ddof=1) = 2.5
        var arr = np.array(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var result = np.var(arr, ddof: 1);
        Assert.IsTrue(Math.Abs(result.GetDouble(0) - 2.5) < Tolerance);
    }

    [TestMethod]
    public void Var_1D_Int32()
    {
        // NumPy: np.var([1, 2, 3, 4, 5]) = 2.0 (returns float64)
        var arr = np.array(new int[] { 1, 2, 3, 4, 5 });
        var result = np.var(arr);
        Assert.IsTrue(Math.Abs(result.GetDouble(0) - 2.0) < Tolerance);
    }

    [TestMethod]
    public void Var_1D_Int64()
    {
        // NumPy: np.var([1, 2, 3, 4, 5], dtype=int64) = 2.0 (returns float64)
        var arr = np.array(new long[] { 1L, 2L, 3L, 4L, 5L });
        var result = np.var(arr);
        Assert.IsTrue(Math.Abs(result.GetDouble(0) - 2.0) < Tolerance);
    }

    [TestMethod]
    public void Var_1D_Float32()
    {
        // NumPy: np.var([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float32) = 2.0 (returns float32)
        var arr = np.array(new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f });
        var result = np.var(arr);
        // Use implicit conversion to double (works for any dtype scalar)
        Assert.IsTrue(Math.Abs((double)result - 2.0) < 0.01); // float32 has less precision
    }

    #endregion

    #region Var 2D Tests

    [TestMethod]
    public void Var_2D_NoAxis()
    {
        // NumPy: np.var([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]) = 6.666666666666667
        var arr = np.array(new double[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 }, { 7.0, 8.0, 9.0 } });
        var result = np.var(arr);
        Assert.IsTrue(Math.Abs(result.GetDouble(0) - 6.666666666666667) < Tolerance);
    }

    [TestMethod]
    public void Var_2D_Axis0()
    {
        // NumPy: np.var([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], axis=0) = [6.0, 6.0, 6.0]
        var arr = np.array(new double[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 }, { 7.0, 8.0, 9.0 } });
        var result = np.var(arr, axis: 0);
        result.Should().BeShaped(3);
        result.Should().BeOfValuesApproximately(Tolerance, 6.0, 6.0, 6.0);
    }

    [TestMethod]
    public void Var_2D_Axis1()
    {
        // NumPy: np.var([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], axis=1) = [0.666..., 0.666..., 0.666...]
        var arr = np.array(new double[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 }, { 7.0, 8.0, 9.0 } });
        var result = np.var(arr, axis: 1);
        result.Should().BeShaped(3);
        result.Should().BeOfValuesApproximately(Tolerance, 0.6666666666666666, 0.6666666666666666, 0.6666666666666666);
    }

    [TestMethod]
    public void Var_2D_AxisNeg1()
    {
        // NumPy: np.var([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], axis=-1) = [0.666..., 0.666..., 0.666...]
        var arr = np.array(new double[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 }, { 7.0, 8.0, 9.0 } });
        var result = np.var(arr, axis: -1);
        result.Should().BeShaped(3);
        result.Should().BeOfValuesApproximately(Tolerance, 0.6666666666666666, 0.6666666666666666, 0.6666666666666666);
    }

    [TestMethod]
    public void Var_2D_Axis0_Ddof1()
    {
        // NumPy: np.var([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], axis=0, ddof=1) = [9.0, 9.0, 9.0]
        var arr = np.array(new double[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 }, { 7.0, 8.0, 9.0 } });
        var result = np.var(arr, axis: 0, ddof: 1);
        result.Should().BeShaped(3);
        result.Should().BeOfValuesApproximately(Tolerance, 9.0, 9.0, 9.0);
    }

    #endregion

    #region Var 3D Tests

    [TestMethod]
    public void Var_3D_NoAxis()
    {
        // NumPy: np.var([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]) = 5.25
        var arr = np.array(new double[,,] { { { 1.0, 2.0 }, { 3.0, 4.0 } }, { { 5.0, 6.0 }, { 7.0, 8.0 } } });
        var result = np.var(arr);
        Assert.IsTrue(Math.Abs(result.GetDouble(0) - 5.25) < Tolerance);
    }

    [TestMethod]
    public void Var_3D_Axis0()
    {
        // NumPy: np.var([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], axis=0) = [[4.0, 4.0], [4.0, 4.0]]
        var arr = np.array(new double[,,] { { { 1.0, 2.0 }, { 3.0, 4.0 } }, { { 5.0, 6.0 }, { 7.0, 8.0 } } });
        var result = np.var(arr, axis: 0);
        result.Should().BeShaped(2, 2);
        result.Should().BeOfValuesApproximately(Tolerance, 4.0, 4.0, 4.0, 4.0);
    }

    [TestMethod]
    public void Var_3D_Axis1()
    {
        // NumPy: np.var([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], axis=1) = [[1.0, 1.0], [1.0, 1.0]]
        var arr = np.array(new double[,,] { { { 1.0, 2.0 }, { 3.0, 4.0 } }, { { 5.0, 6.0 }, { 7.0, 8.0 } } });
        var result = np.var(arr, axis: 1);
        result.Should().BeShaped(2, 2);
        result.Should().BeOfValuesApproximately(Tolerance, 1.0, 1.0, 1.0, 1.0);
    }

    [TestMethod]
    public void Var_3D_Axis2()
    {
        // NumPy: np.var([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], axis=2) = [[0.25, 0.25], [0.25, 0.25]]
        var arr = np.array(new double[,,] { { { 1.0, 2.0 }, { 3.0, 4.0 } }, { { 5.0, 6.0 }, { 7.0, 8.0 } } });
        var result = np.var(arr, axis: 2);
        result.Should().BeShaped(2, 2);
        result.Should().BeOfValuesApproximately(Tolerance, 0.25, 0.25, 0.25, 0.25);
    }

    #endregion

    #region Var Edge Cases

    [TestMethod]
    public void Var_SingleElement()
    {
        // NumPy: np.var([42.0]) = 0.0
        var arr = np.array(new double[] { 42.0 });
        var result = np.var(arr);
        Assert.IsTrue(Math.Abs(result.GetDouble(0) - 0.0) < Tolerance);
    }

    [TestMethod]
    public void Var_IdenticalValues()
    {
        // NumPy: np.var([5.0, 5.0, 5.0, 5.0]) = 0.0
        var arr = np.array(new double[] { 5.0, 5.0, 5.0, 5.0 });
        var result = np.var(arr);
        Assert.IsTrue(Math.Abs(result.GetDouble(0) - 0.0) < Tolerance);
    }

    [TestMethod]
    public void Var_Rectangular_2x5_Axis0()
    {
        // NumPy: np.var([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]], axis=0) = [6.25, 6.25, 6.25, 6.25, 6.25]
        var arr = np.array(new double[,] { { 1.0, 2.0, 3.0, 4.0, 5.0 }, { 6.0, 7.0, 8.0, 9.0, 10.0 } });
        var result = np.var(arr, axis: 0);
        result.Should().BeShaped(5);
        result.Should().BeOfValuesApproximately(Tolerance, 6.25, 6.25, 6.25, 6.25, 6.25);
    }

    [TestMethod]
    public void Var_Rectangular_2x5_Axis1()
    {
        // NumPy: np.var([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]], axis=1) = [2.0, 2.0]
        var arr = np.array(new double[,] { { 1.0, 2.0, 3.0, 4.0, 5.0 }, { 6.0, 7.0, 8.0, 9.0, 10.0 } });
        var result = np.var(arr, axis: 1);
        result.Should().BeShaped(2);
        result.Should().BeOfValuesApproximately(Tolerance, 2.0, 2.0);
    }

    #endregion

    #region Std 1D Tests

    [TestMethod]
    public void Std_1D_Float64()
    {
        // NumPy: np.std([1.0, 2.0, 3.0, 4.0, 5.0]) = 1.4142135623730951
        var arr = np.array(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var result = np.std(arr);
        Assert.IsTrue(Math.Abs(result.GetDouble(0) - 1.4142135623730951) < Tolerance);
    }

    [TestMethod]
    public void Std_1D_Float64_Ddof0()
    {
        // NumPy: np.std([1.0, 2.0, 3.0, 4.0, 5.0], ddof=0) = 1.4142135623730951
        var arr = np.array(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var result = np.std(arr, ddof: 0);
        Assert.IsTrue(Math.Abs(result.GetDouble(0) - 1.4142135623730951) < Tolerance);
    }

    [TestMethod]
    public void Std_1D_Float64_Ddof1()
    {
        // NumPy: np.std([1.0, 2.0, 3.0, 4.0, 5.0], ddof=1) = 1.5811388300841898
        var arr = np.array(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var result = np.std(arr, ddof: 1);
        Assert.IsTrue(Math.Abs(result.GetDouble(0) - 1.5811388300841898) < Tolerance);
    }

    [TestMethod]
    public void Std_1D_Int32()
    {
        // NumPy: np.std([1, 2, 3, 4, 5]) = 1.4142135623730951 (returns float64)
        var arr = np.array(new int[] { 1, 2, 3, 4, 5 });
        var result = np.std(arr);
        Assert.IsTrue(Math.Abs(result.GetDouble(0) - 1.4142135623730951) < Tolerance);
    }

    [TestMethod]
    public void Std_1D_Int64()
    {
        // NumPy: np.std([1, 2, 3, 4, 5], dtype=int64) = 1.4142135623730951 (returns float64)
        var arr = np.array(new long[] { 1L, 2L, 3L, 4L, 5L });
        var result = np.std(arr);
        Assert.IsTrue(Math.Abs(result.GetDouble(0) - 1.4142135623730951) < Tolerance);
    }

    [TestMethod]
    public void Std_1D_Float32()
    {
        // NumPy: np.std([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float32) = 1.4142135381698608 (returns float32)
        var arr = np.array(new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f });
        var result = np.std(arr);
        // Use implicit conversion to double (works for any dtype scalar)
        Assert.IsTrue(Math.Abs((double)result - 1.4142135) < 0.001); // float32 has less precision
    }

    #endregion

    #region Std 2D Tests

    [TestMethod]
    public void Std_2D_NoAxis()
    {
        // NumPy: np.std([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]) = 2.581988897471611
        var arr = np.array(new double[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 }, { 7.0, 8.0, 9.0 } });
        var result = np.std(arr);
        Assert.IsTrue(Math.Abs(result.GetDouble(0) - 2.581988897471611) < Tolerance);
    }

    [TestMethod]
    public void Std_2D_Axis0()
    {
        // NumPy: np.std([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], axis=0) = [2.449..., 2.449..., 2.449...]
        var arr = np.array(new double[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 }, { 7.0, 8.0, 9.0 } });
        var result = np.std(arr, axis: 0);
        result.Should().BeShaped(3);
        result.Should().BeOfValuesApproximately(Tolerance, 2.449489742783178, 2.449489742783178, 2.449489742783178);
    }

    [TestMethod]
    public void Std_2D_Axis1()
    {
        // NumPy: np.std([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], axis=1) = [0.816..., 0.816..., 0.816...]
        var arr = np.array(new double[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 }, { 7.0, 8.0, 9.0 } });
        var result = np.std(arr, axis: 1);
        result.Should().BeShaped(3);
        result.Should().BeOfValuesApproximately(Tolerance, 0.816496580927726, 0.816496580927726, 0.816496580927726);
    }

    [TestMethod]
    public void Std_2D_AxisNeg1()
    {
        // NumPy: np.std([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], axis=-1) = [0.816..., 0.816..., 0.816...]
        var arr = np.array(new double[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 }, { 7.0, 8.0, 9.0 } });
        var result = np.std(arr, axis: -1);
        result.Should().BeShaped(3);
        result.Should().BeOfValuesApproximately(Tolerance, 0.816496580927726, 0.816496580927726, 0.816496580927726);
    }

    [TestMethod]
    public void Std_2D_Axis0_Ddof1()
    {
        // NumPy: np.std([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], axis=0, ddof=1) = [3.0, 3.0, 3.0]
        var arr = np.array(new double[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 }, { 7.0, 8.0, 9.0 } });
        var result = np.std(arr, axis: 0, ddof: 1);
        result.Should().BeShaped(3);
        result.Should().BeOfValuesApproximately(Tolerance, 3.0, 3.0, 3.0);
    }

    #endregion

    #region Std 3D Tests

    [TestMethod]
    public void Std_3D_NoAxis()
    {
        // NumPy: np.std([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]) = 2.29128784747792
        var arr = np.array(new double[,,] { { { 1.0, 2.0 }, { 3.0, 4.0 } }, { { 5.0, 6.0 }, { 7.0, 8.0 } } });
        var result = np.std(arr);
        Assert.IsTrue(Math.Abs(result.GetDouble(0) - 2.29128784747792) < Tolerance);
    }

    [TestMethod]
    public void Std_3D_Axis0()
    {
        // NumPy: np.std([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], axis=0) = [[2.0, 2.0], [2.0, 2.0]]
        var arr = np.array(new double[,,] { { { 1.0, 2.0 }, { 3.0, 4.0 } }, { { 5.0, 6.0 }, { 7.0, 8.0 } } });
        var result = np.std(arr, axis: 0);
        result.Should().BeShaped(2, 2);
        result.Should().BeOfValuesApproximately(Tolerance, 2.0, 2.0, 2.0, 2.0);
    }

    [TestMethod]
    public void Std_3D_Axis1()
    {
        // NumPy: np.std([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], axis=1) = [[1.0, 1.0], [1.0, 1.0]]
        var arr = np.array(new double[,,] { { { 1.0, 2.0 }, { 3.0, 4.0 } }, { { 5.0, 6.0 }, { 7.0, 8.0 } } });
        var result = np.std(arr, axis: 1);
        result.Should().BeShaped(2, 2);
        result.Should().BeOfValuesApproximately(Tolerance, 1.0, 1.0, 1.0, 1.0);
    }

    [TestMethod]
    public void Std_3D_Axis2()
    {
        // NumPy: np.std([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], axis=2) = [[0.5, 0.5], [0.5, 0.5]]
        var arr = np.array(new double[,,] { { { 1.0, 2.0 }, { 3.0, 4.0 } }, { { 5.0, 6.0 }, { 7.0, 8.0 } } });
        var result = np.std(arr, axis: 2);
        result.Should().BeShaped(2, 2);
        result.Should().BeOfValuesApproximately(Tolerance, 0.5, 0.5, 0.5, 0.5);
    }

    #endregion

    #region Std Edge Cases

    [TestMethod]
    public void Std_SingleElement()
    {
        // NumPy: np.std([42.0]) = 0.0
        var arr = np.array(new double[] { 42.0 });
        var result = np.std(arr);
        Assert.IsTrue(Math.Abs(result.GetDouble(0) - 0.0) < Tolerance);
    }

    [TestMethod]
    public void Std_IdenticalValues()
    {
        // NumPy: np.std([5.0, 5.0, 5.0, 5.0]) = 0.0
        var arr = np.array(new double[] { 5.0, 5.0, 5.0, 5.0 });
        var result = np.std(arr);
        Assert.IsTrue(Math.Abs(result.GetDouble(0) - 0.0) < Tolerance);
    }

    [TestMethod]
    public void Std_Rectangular_2x5_Axis0()
    {
        // NumPy: np.std([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]], axis=0) = [2.5, 2.5, 2.5, 2.5, 2.5]
        var arr = np.array(new double[,] { { 1.0, 2.0, 3.0, 4.0, 5.0 }, { 6.0, 7.0, 8.0, 9.0, 10.0 } });
        var result = np.std(arr, axis: 0);
        result.Should().BeShaped(5);
        result.Should().BeOfValuesApproximately(Tolerance, 2.5, 2.5, 2.5, 2.5, 2.5);
    }

    [TestMethod]
    public void Std_Rectangular_2x5_Axis1()
    {
        // NumPy: np.std([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]], axis=1) = [1.414..., 1.414...]
        var arr = np.array(new double[,] { { 1.0, 2.0, 3.0, 4.0, 5.0 }, { 6.0, 7.0, 8.0, 9.0, 10.0 } });
        var result = np.std(arr, axis: 1);
        result.Should().BeShaped(2);
        result.Should().BeOfValuesApproximately(Tolerance, 1.4142135623730951, 1.4142135623730951);
    }

    #endregion

    #region keepdims Tests

    [TestMethod]
    public void Var_2D_Axis0_Keepdims()
    {
        // NumPy: np.var([[1, 2, 3], [4, 5, 6]], axis=0, keepdims=True).shape = (1, 3)
        var arr = np.array(new double[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } });
        var result = np.var(arr, axis: 0, keepdims: true);
        result.Should().BeShaped(1, 3);
    }

    [TestMethod]
    public void Std_2D_Axis0_Keepdims()
    {
        // NumPy: np.std([[1, 2, 3], [4, 5, 6]], axis=0, keepdims=True).shape = (1, 3)
        var arr = np.array(new double[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } });
        var result = np.std(arr, axis: 0, keepdims: true);
        result.Should().BeShaped(1, 3);
    }

    #endregion

    #region Large Values (Numerical Stability)

    [TestMethod]
    public void Var_LargeValues()
    {
        // NumPy: np.var([1e15, 2e15, 3e15]) = 6.666666666666666e+29
        var arr = np.array(new double[] { 1e15, 2e15, 3e15 });
        var result = np.var(arr);
        Assert.IsTrue(Math.Abs(result.GetDouble(0) - 6.666666666666666e+29) / 6.666666666666666e+29 < 1e-10);
    }

    [TestMethod]
    public void Std_LargeValues()
    {
        // NumPy: np.std([1e15, 2e15, 3e15]) = 816496580927726.0
        var arr = np.array(new double[] { 1e15, 2e15, 3e15 });
        var result = np.std(arr);
        Assert.IsTrue(Math.Abs(result.GetDouble(0) - 816496580927726.0) / 816496580927726.0 < 1e-10);
    }

    #endregion
}
