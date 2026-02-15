using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.UnitTest.Utilities;
using TUnit.Core;

namespace NumSharp.UnitTest.Backends.Kernels;

/// <summary>
/// Comprehensive tests for reduction operations (Sum, Prod, Max, Min, ArgMax, ArgMin, Mean, Std, Var, CumSum).
/// All expected values are verified against NumPy 2.x output.
/// These tests validate existing reduction functionality and serve as
/// regression tests for future IL kernel Phase 5 implementation.
/// </summary>
public class ReductionOpTests
{
    private const double Tolerance = 1e-10;

    #region Basic 1D Sum Tests

    [Test]
    public void Sum_Float64_1D()
    {
        // NumPy: sum([1.0, 2.0, 3.0, 4.0, 5.0]) = 15.0
        var a = np.array(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var result = np.sum(a);

        Assert.AreEqual(15.0, result.GetDouble(0));
    }

    [Test]
    [OpenBugs] // Sum may have memory corruption issues
    public void Sum_Int32_1D()
    {
        var a = np.array(new[] { 1, 2, 3, 4, 5 });
        var result = np.sum(a);

        Assert.AreEqual(15L, result.GetInt64(0));
    }

    #endregion

    #region Basic 1D Prod Tests

    [Test]
    public void Prod_Float64_1D()
    {
        // NumPy: prod([1.0, 2.0, 3.0, 4.0, 5.0]) = 120.0
        var a = np.array(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var result = np.prod(a);

        Assert.AreEqual(120.0, result.GetDouble(0));
    }

    #endregion

    #region Basic 1D Max/Min Tests

    [Test]
    public void Max_Float64_1D()
    {
        // NumPy: max([1.0, 2.0, 3.0, 4.0, 5.0]) = 5.0
        var a = np.array(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var result = np.amax(a);

        Assert.AreEqual(5.0, result.GetDouble(0));
    }

    [Test]
    public void Min_Float64_1D()
    {
        // NumPy: min([1.0, 2.0, 3.0, 4.0, 5.0]) = 1.0
        var a = np.array(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var result = np.amin(a);

        Assert.AreEqual(1.0, result.GetDouble(0));
    }

    #endregion

    #region Basic 1D ArgMax/ArgMin Tests

    [Test]
    public void ArgMax_Float64_1D()
    {
        // NumPy: argmax([1.0, 2.0, 3.0, 4.0, 5.0]) = 4
        var a = np.array(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var result = np.argmax(a);

        Assert.AreEqual(4, result);
    }

    [Test]
    public void ArgMin_Float64_1D()
    {
        // NumPy: argmin([1.0, 2.0, 3.0, 4.0, 5.0]) = 0
        var a = np.array(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var result = np.argmin(a);

        Assert.AreEqual(0, result);
    }

    #endregion

    #region Basic 1D Mean/Std/Var Tests

    [Test]
    public void Mean_Float64_1D()
    {
        // NumPy: mean([1.0, 2.0, 3.0, 4.0, 5.0]) = 3.0
        var a = np.array(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var result = np.mean(a);

        Assert.AreEqual(3.0, result.GetDouble(0));
    }

    [Test]
    public void Std_Float64_1D()
    {
        // NumPy: std([1.0, 2.0, 3.0, 4.0, 5.0]) = 1.4142135623730951
        var a = np.array(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var result = np.std(a);

        Assert.IsTrue(Math.Abs(result.GetDouble(0) - 1.4142135623730951) < Tolerance);
    }

    [Test]
    public void Var_Float64_1D()
    {
        // NumPy: var([1.0, 2.0, 3.0, 4.0, 5.0]) = 2.0
        var a = np.array(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var result = np.var(a);

        Assert.IsTrue(Math.Abs(result.GetDouble(0) - 2.0) < Tolerance);
    }

    #endregion

    #region Basic 1D CumSum Tests

    [Test]
    public void CumSum_Float64_1D()
    {
        // NumPy: cumsum([1.0, 2.0, 3.0, 4.0, 5.0]) = [1.0, 3.0, 6.0, 10.0, 15.0]
        var a = np.array(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var result = np.cumsum(a);

        result.Should().BeOfValues(1.0, 3.0, 6.0, 10.0, 15.0);
    }

    #endregion

    #region 2D Sum with Axis Tests

    [Test]
    public void Sum_2D_NoAxis()
    {
        // NumPy: sum([[1,2,3],[4,5,6]]) = 21.0
        var a = np.array(new[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } });
        var result = np.sum(a);

        Assert.AreEqual(21.0, result.GetDouble(0));
    }

    [Test]
    public void Sum_2D_Axis0()
    {
        // NumPy: sum([[1,2,3],[4,5,6]], axis=0) = [5, 7, 9]
        var a = np.array(new[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } });
        var result = np.sum(a, axis: 0);

        result.Should().BeShaped(3);
        result.Should().BeOfValues(5.0, 7.0, 9.0);
    }

    [Test]
    public void Sum_2D_Axis1()
    {
        // NumPy: sum([[1,2,3],[4,5,6]], axis=1) = [6, 15]
        var a = np.array(new[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } });
        var result = np.sum(a, axis: 1);

        result.Should().BeShaped(2);
        result.Should().BeOfValues(6.0, 15.0);
    }

    #endregion

    #region 2D Mean with Axis Tests

    [Test]
    public void Mean_2D_Axis0()
    {
        // NumPy: mean([[1,2,3],[4,5,6]], axis=0) = [2.5, 3.5, 4.5]
        var a = np.array(new[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } });
        var result = np.mean(a, axis: 0);

        result.Should().BeShaped(3);
        result.Should().BeOfValues(2.5, 3.5, 4.5);
    }

    [Test]
    public void Mean_2D_Axis1()
    {
        // NumPy: mean([[1,2,3],[4,5,6]], axis=1) = [2.0, 5.0]
        var a = np.array(new[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } });
        var result = np.mean(a, axis: 1);

        result.Should().BeShaped(2);
        result.Should().BeOfValues(2.0, 5.0);
    }

    #endregion

    #region 2D Max/Min with Axis Tests

    [Test]
    public void Max_2D_Axis0()
    {
        // NumPy: max([[1,2,3],[4,5,6]], axis=0) = [4, 5, 6]
        var a = np.array(new[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } });
        var result = np.amax(a, axis: 0);

        result.Should().BeShaped(3);
        result.Should().BeOfValues(4.0, 5.0, 6.0);
    }

    [Test]
    public void Max_2D_Axis1()
    {
        // NumPy: max([[1,2,3],[4,5,6]], axis=1) = [3, 6]
        var a = np.array(new[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } });
        var result = np.amax(a, axis: 1);

        result.Should().BeShaped(2);
        result.Should().BeOfValues(3.0, 6.0);
    }

    [Test]
    public void Min_2D_Axis0()
    {
        // NumPy: min([[1,2,3],[4,5,6]], axis=0) = [1, 2, 3]
        var a = np.array(new[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } });
        var result = np.amin(a, axis: 0);

        result.Should().BeShaped(3);
        result.Should().BeOfValues(1.0, 2.0, 3.0);
    }

    [Test]
    public void Min_2D_Axis1()
    {
        // NumPy: min([[1,2,3],[4,5,6]], axis=1) = [1, 4]
        var a = np.array(new[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } });
        var result = np.amin(a, axis: 1);

        result.Should().BeShaped(2);
        result.Should().BeOfValues(1.0, 4.0);
    }

    #endregion

    #region 2D ArgMax/ArgMin with Axis Tests

    [Test]
    public void ArgMax_2D_Axis0()
    {
        // NumPy: argmax([[1,2,3],[4,5,6]], axis=0) = [1, 1, 1]
        var a = np.array(new[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } });
        var result = np.argmax(a, axis: 0);

        result.Should().BeShaped(3);
        result.Should().BeOfValues(1L, 1L, 1L);
    }

    [Test]
    public void ArgMax_2D_Axis1()
    {
        // NumPy: argmax([[1,2,3],[4,5,6]], axis=1) = [2, 2]
        var a = np.array(new[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } });
        var result = np.argmax(a, axis: 1);

        result.Should().BeShaped(2);
        result.Should().BeOfValues(2L, 2L);
    }

    #endregion

    #region keepdims Tests

    [Test]
    public void Sum_2D_Axis0_Keepdims()
    {
        // NumPy: sum([[1,2,3],[4,5,6]], axis=0, keepdims=True) = [[5, 7, 9]], shape=(1,3)
        var a = np.array(new[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } });
        var result = np.sum(a, axis: 0, keepdims: true);

        result.Should().BeShaped(1, 3);
        result.Should().BeOfValues(5.0, 7.0, 9.0);
    }

    [Test]
    public void Sum_2D_Axis1_Keepdims()
    {
        // NumPy: sum([[1,2,3],[4,5,6]], axis=1, keepdims=True) = [[6], [15]], shape=(2,1)
        var a = np.array(new[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } });
        var result = np.sum(a, axis: 1, keepdims: true);

        result.Should().BeShaped(2, 1);
        result.Should().BeOfValues(6.0, 15.0);
    }

    [Test]
    public void Mean_2D_Axis0_Keepdims()
    {
        var a = np.array(new[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } });
        var result = np.mean(a, axis: 0, keepdims: true);

        result.Should().BeShaped(1, 3);
        result.Should().BeOfValues(2.5, 3.5, 4.5);
    }

    #endregion

    #region Edge Cases - Infinity

    [Test]
    public void Sum_WithInfinity()
    {
        // NumPy: sum([1.0, inf, 3.0]) = inf
        var a = np.array(new double[] { 1.0, double.PositiveInfinity, 3.0 });
        var result = np.sum(a);

        Assert.IsTrue(double.IsPositiveInfinity(result.GetDouble(0)));
    }

    [Test]
    public void Mean_WithInfinity()
    {
        // NumPy: mean([1.0, inf, 3.0]) = inf
        var a = np.array(new double[] { 1.0, double.PositiveInfinity, 3.0 });
        var result = np.mean(a);

        Assert.IsTrue(double.IsPositiveInfinity(result.GetDouble(0)));
    }

    #endregion

    #region Edge Cases - NaN

    [Test]
    public void Sum_WithNaN()
    {
        // NumPy: sum([1.0, nan, 3.0]) = nan
        var a = np.array(new double[] { 1.0, double.NaN, 3.0 });
        var result = np.sum(a);

        Assert.IsTrue(double.IsNaN(result.GetDouble(0)));
    }

    [Test]
    public void Mean_WithNaN()
    {
        // NumPy: mean([1.0, nan, 3.0]) = nan
        var a = np.array(new double[] { 1.0, double.NaN, 3.0 });
        var result = np.mean(a);

        Assert.IsTrue(double.IsNaN(result.GetDouble(0)));
    }

    [Test]
    public void Max_WithNaN()
    {
        // NumPy: max([1.0, nan, 3.0]) = nan
        var a = np.array(new double[] { 1.0, double.NaN, 3.0 });
        var result = np.amax(a);

        Assert.IsTrue(double.IsNaN(result.GetDouble(0)));
    }

    #endregion

    #region Edge Cases - Empty Array

    [Test]
    public void Sum_EmptyArray()
    {
        // NumPy: sum([]) = 0.0
        var a = np.array(Array.Empty<double>());
        var result = np.sum(a);

        Assert.AreEqual(0.0, result.GetDouble(0));
    }

    [Test]
    public void Prod_EmptyArray()
    {
        // NumPy: prod([]) = 1.0
        var a = np.array(Array.Empty<double>());
        var result = np.prod(a);

        Assert.AreEqual(1.0, result.GetDouble(0));
    }

    #endregion

    #region Edge Cases - 0D Scalar

    [Test]
    public void Sum_0DScalar()
    {
        // NumPy: sum(np.array(5.0)) = 5.0, shape=()
        var a = np.array(5.0);
        var result = np.sum(a);

        Assert.AreEqual(0, result.ndim);
        Assert.AreEqual(5.0, result.GetDouble(0));
    }

    #endregion

    #region CumSum with Axis Tests

    [Test]
    public void CumSum_2D_NoAxis()
    {
        // NumPy: cumsum([[1,2,3],[4,5,6]]) = [1, 3, 6, 10, 15, 21]
        var a = np.array(new[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        var result = np.cumsum(a);

        result.Should().BeOfValues(1, 3, 6, 10, 15, 21);
    }

    [Test]
    public void CumSum_2D_Axis0()
    {
        // NumPy: cumsum([[1,2,3],[4,5,6]], axis=0) = [[1, 2, 3], [5, 7, 9]]
        var a = np.array(new[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        var result = np.cumsum(a, axis: 0);

        result.Should().BeShaped(2, 3);
        result.Should().BeOfValues(1, 2, 3, 5, 7, 9);
    }

    [Test]
    public void CumSum_2D_Axis1()
    {
        // NumPy: cumsum([[1,2,3],[4,5,6]], axis=1) = [[1, 3, 6], [4, 9, 15]]
        var a = np.array(new[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        var result = np.cumsum(a, axis: 1);

        result.Should().BeShaped(2, 3);
        result.Should().BeOfValues(1, 3, 6, 4, 9, 15);
    }

    #endregion

    #region Boolean Reduction Tests

    [Test]
    [OpenBugs] // Bool sum may have issues
    public void Sum_Bool()
    {
        // NumPy: sum([True, False, True, True]) = 3
        var a = np.array(new[] { true, false, true, true });
        var result = np.sum(a);

        Assert.AreEqual(3L, result.GetInt64(0));
    }

    [Test]
    public void All_Bool()
    {
        // NumPy: all([True, False, True, True]) = False
        var a = np.array(new[] { true, false, true, true });
        var result = np.all(a);

        Assert.IsFalse(result);
    }

    [Test]
    public void All_Bool_AllTrue()
    {
        var a = np.array(new[] { true, true, true, true });
        var result = np.all(a);

        Assert.IsTrue(result);
    }

    #endregion

    #region Integer Type Tests

    [Test]
    [OpenBugs] // Sum may have memory corruption issues
    public void Sum_Int32_PromotesToInt64()
    {
        // NumPy: int32 sum returns int64
        var a = np.array(new[] { 1, 2, 3, 4, 5 });
        var result = np.sum(a);

        Assert.AreEqual(15L, result.GetInt64(0));
        Assert.AreEqual(NPTypeCode.Int64, result.typecode);
    }

    [Test]
    public void Mean_Int32_ReturnsFloat64()
    {
        // NumPy: int32 mean returns float64
        var a = np.array(new[] { 1, 2, 3, 4, 5 });
        var result = np.mean(a);

        Assert.AreEqual(3.0, result.GetDouble(0));
        Assert.AreEqual(NPTypeCode.Double, result.typecode);
    }

    [Test]
    public void Max_Int32_PreservesType()
    {
        // NumPy: int32 max returns int32
        var a = np.array(new[] { 1, 2, 3, 4, 5 });
        var result = np.amax(a);

        Assert.AreEqual(5, result.GetInt32(0));
        Assert.AreEqual(NPTypeCode.Int32, result.typecode);
    }

    #endregion

    #region Sliced Array Tests

    [Test]
    [OpenBugs] // Sum may have memory corruption issues
    public void Sum_SlicedArray()
    {
        var a = np.array(new[] { 1, 2, 3, 4, 5, 6 });
        var sliced = a["::2"];  // [1, 3, 5]
        var result = np.sum(sliced);

        Assert.AreEqual(9L, result.GetInt64(0));
    }

    [Test]
    public void Mean_SlicedArray()
    {
        var a = np.array(new double[] { 1, 2, 3, 4, 5, 6 });
        var sliced = a["::2"];  // [1, 3, 5]
        var result = np.mean(sliced);

        Assert.AreEqual(3.0, result.GetDouble(0));
    }

    #endregion
}
