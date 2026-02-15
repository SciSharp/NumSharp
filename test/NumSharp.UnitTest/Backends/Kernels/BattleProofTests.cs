using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Backends.Kernels;

/// <summary>
/// Battle-proof tests verifying IL kernel fixes against NumPy behavior.
/// Each test documents the NumPy output and verifies NumSharp matches.
/// </summary>
public class BattleProofTests : TestClass
{
    #region Fix 1: Sliced Array × Scalar (ClassifyPath contiguity check)

    [Test]
    public void SlicedColumn_MultiplyByScalar_MatchesNumPy()
    {
        // NumPy: x = np.arange(4).reshape(2,2); y = x[:,1]; z = y * 2
        // y = [1, 3], z = [2, 6]
        var x = np.arange(4).reshape(2, 2);
        var y = x[":,1"];  // column slice, stride=2

        Assert.IsFalse(y.Shape.IsContiguous, "Column slice should not be contiguous");

        var z = y * 2;

        Assert.AreEqual(2, z.GetInt32(0), "y[0]*2 = 1*2 = 2");
        Assert.AreEqual(6, z.GetInt32(1), "y[1]*2 = 3*2 = 6");
    }

    [Test]
    public void StepSlice_MultiplyByScalar_MatchesNumPy()
    {
        // NumPy: np.arange(10)[::2] * 3 = [0, 6, 12, 18, 24]
        var arr = np.arange(10);
        var step = arr["::2"];

        Assert.IsFalse(step.Shape.IsContiguous, "Step slice should not be contiguous");

        var result = step * 3;

        result.Should().BeOfValues(0, 6, 12, 18, 24);
    }

    [Test]
    public void ReverseSlice_MultiplyByScalar_MatchesNumPy()
    {
        // NumPy: np.arange(10)[::-1] * 2 = [18, 16, 14, 12, 10, 8, 6, 4, 2, 0]
        var arr = np.arange(10);
        var rev = arr["::-1"];

        var result = rev * 2;

        result.Should().BeOfValues(18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
    }

    [Test]
    public void SlicedRow_MultiplyByScalar_MatchesNumPy()
    {
        // NumPy: np.arange(12).reshape(3,4)[1,:] * 3 = [12, 15, 18, 21]
        var arr = np.arange(12).reshape(3, 4);
        var row = arr["1,:"];

        // Row slice IS contiguous (strides match)
        Assert.IsTrue(row.Shape.IsContiguous, "Row slice should be contiguous");

        var result = row * 3;

        result.Should().BeOfValues(12, 15, 18, 21);
    }

    [Test]
    public void ScalarMultiplySlice_BothDirections_MatchesNumPy()
    {
        // NumPy: 2 * x[:,1] and x[:,1] * 2 should give same result
        var x = np.arange(4).reshape(2, 2);
        var y = x[":,1"];

        var left = 2 * y;
        var right = y * 2;

        left.Should().BeOfValues(2, 6);
        right.Should().BeOfValues(2, 6);
    }

    #endregion

    #region Fix 2: Division Type Promotion (True Division → float64)

    [Test]
    public void IntDivInt_ReturnsFloat64_MatchesNumPy()
    {
        // NumPy: np.array([1,2,3,4], dtype=int32) / np.array([2,2,2,2], dtype=int32)
        // Result: [0.5, 1.0, 1.5, 2.0], dtype=float64
        var a = np.array(new[] { 1, 2, 3, 4 });
        var b = np.array(new[] { 2, 2, 2, 2 });

        var result = a / b;

        Assert.AreEqual(typeof(double), result.dtype, "int/int should return float64");
        result.Should().BeOfValues(0.5, 1.0, 1.5, 2.0);
    }

    [Test]
    public void UInt8DivScalar_ReturnsFloat64_MatchesNumPy()
    {
        // NumPy: np.array([10, 20, 30], dtype=uint8) / 5
        // Result: [2.0, 4.0, 6.0], dtype=float64
        var a = np.array(new byte[] { 10, 20, 30 });

        var result = a / 5;

        Assert.AreEqual(typeof(double), result.dtype, "uint8/scalar should return float64");
        result.Should().BeOfValues(2.0, 4.0, 6.0);
    }

    [Test]
    public void IntDivInt_FractionalResult_MatchesNumPy()
    {
        // NumPy: 3 / 2 = 1.5 (not 1 like integer division)
        var a = np.array(new[] { 3 });
        var b = np.array(new[] { 2 });

        var result = a / b;

        Assert.AreEqual(typeof(double), result.dtype);
        Assert.AreEqual(1.5, result.GetDouble(0), 0.001, "3/2 should be 1.5, not 1");
    }

    [Test]
    public void Float32DivFloat32_StaysFloat32_MatchesNumPy()
    {
        // NumPy: float32 / float32 = float32 (not promoted to float64)
        var a = np.array(new[] { 3.0f, 6.0f });
        var b = np.array(new[] { 2.0f, 2.0f });

        var result = a / b;

        Assert.AreEqual(typeof(float), result.dtype, "float32/float32 should stay float32");
    }

    [Test]
    public void Float64DivFloat64_StaysFloat64_MatchesNumPy()
    {
        // NumPy: float64 / float64 = float64
        var a = np.array(new[] { 3.0, 6.0 });
        var b = np.array(new[] { 2.0, 2.0 });

        var result = a / b;

        Assert.AreEqual(typeof(double), result.dtype, "float64/float64 should stay float64");
    }

    #endregion

    #region Fix 3: Sign(NaN) Returns NaN (not exception)

    [Test]
    public void SignNaN_ReturnsNaN_MatchesNumPy()
    {
        // NumPy: np.sign(np.nan) = nan
        // .NET Math.Sign(NaN) throws ArithmeticException - we fixed this
        var arr = np.array(new[] { double.NaN });

        var result = np.sign(arr);

        Assert.IsTrue(double.IsNaN(result.GetDouble(0)), "sign(NaN) should return NaN, not throw");
    }

    [Test]
    public void SignFloat32NaN_ReturnsNaN_MatchesNumPy()
    {
        // NumPy: np.sign(np.float32('nan')) = nan
        var arr = np.array(new[] { float.NaN });

        var result = np.sign(arr);

        Assert.IsTrue(float.IsNaN(result.GetSingle(0)), "sign(float32 NaN) should return NaN");
    }

    [Test]
    public void SignInfinity_ReturnsOne_MatchesNumPy()
    {
        // NumPy: np.sign([inf, -inf]) = [1, -1]
        var arr = np.array(new[] { double.PositiveInfinity, double.NegativeInfinity });

        var result = np.sign(arr);

        Assert.AreEqual(1.0, result.GetDouble(0), "sign(+inf) = 1");
        Assert.AreEqual(-1.0, result.GetDouble(1), "sign(-inf) = -1");
    }

    [Test]
    public void SignMixedWithNaN_MatchesNumPy()
    {
        // NumPy: np.sign([nan, 1, -1, 0, nan]) = [nan, 1, -1, 0, nan]
        var arr = np.array(new[] { double.NaN, 1.0, -1.0, 0.0, double.NaN });

        var result = np.sign(arr);

        Assert.IsTrue(double.IsNaN(result.GetDouble(0)), "[0] NaN -> NaN");
        Assert.AreEqual(1.0, result.GetDouble(1), "[1] 1 -> 1");
        Assert.AreEqual(-1.0, result.GetDouble(2), "[2] -1 -> -1");
        Assert.AreEqual(0.0, result.GetDouble(3), "[3] 0 -> 0");
        Assert.IsTrue(double.IsNaN(result.GetDouble(4)), "[4] NaN -> NaN");
    }

    [Test]
    public void SignBasicValues_MatchesNumPy()
    {
        // NumPy: np.sign([1, -1, 0, 5, -5]) = [1, -1, 0, 1, -1]
        var arr = np.array(new[] { 1.0, -1.0, 0.0, 5.0, -5.0 });

        var result = np.sign(arr);

        result.Should().BeOfValues(1.0, -1.0, 0.0, 1.0, -1.0);
    }

    #endregion
}
