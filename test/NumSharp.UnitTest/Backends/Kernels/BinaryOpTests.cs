using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.UnitTest.Utilities;
using TUnit.Core;

namespace NumSharp.UnitTest.Backends.Kernels;

/// <summary>
/// Comprehensive tests for binary operations (Add, Subtract, Multiply, Divide, Mod).
/// All expected values are verified against NumPy 2.x output.
/// </summary>
public class BinaryOpTests
{
    #region Same-Type Add Tests

    [Test]
    public void Add_Bool_SameType()
    {
        // NumPy: np.add([True, False, True, False], [True, True, False, False]) = [True, True, True, False]
        var a = np.array(new[] { true, false, true, false });
        var b = np.array(new[] { true, true, false, false });
        var result = a + b;

        Assert.IsTrue(result.GetBoolean(0));
        Assert.IsTrue(result.GetBoolean(1));
        Assert.IsTrue(result.GetBoolean(2));
        Assert.IsFalse(result.GetBoolean(3));
    }

    [Test]
    public void Add_Byte_SameType()
    {
        // NumPy: [1, 2, 3, 4] + [2, 2, 2, 2] = [3, 4, 5, 6]
        var a = np.array(new byte[] { 1, 2, 3, 4 });
        var b = np.array(new byte[] { 2, 2, 2, 2 });
        var result = a + b;

        result.Should().BeOfValues(3, 4, 5, 6).And.BeOfType(NPTypeCode.Byte);
    }

    [Test]
    public void Add_Int16_SameType()
    {
        var a = np.array(new short[] { 1, 2, 3, 4 });
        var b = np.array(new short[] { 2, 2, 2, 2 });
        var result = a + b;

        result.Should().BeOfValues(3, 4, 5, 6).And.BeOfType(NPTypeCode.Int16);
    }

    [Test]
    public void Add_UInt16_SameType()
    {
        var a = np.array(new ushort[] { 1, 2, 3, 4 });
        var b = np.array(new ushort[] { 2, 2, 2, 2 });
        var result = a + b;

        result.Should().BeOfValues(3, 4, 5, 6).And.BeOfType(NPTypeCode.UInt16);
    }

    [Test]
    public void Add_Int32_SameType()
    {
        var a = np.array(new[] { 1, 2, 3, 4 });
        var b = np.array(new[] { 2, 2, 2, 2 });
        var result = a + b;

        result.Should().BeOfValues(3, 4, 5, 6).And.BeOfType(NPTypeCode.Int32);
    }

    [Test]
    public void Add_UInt32_SameType()
    {
        var a = np.array(new uint[] { 1, 2, 3, 4 });
        var b = np.array(new uint[] { 2, 2, 2, 2 });
        var result = a + b;

        result.Should().BeOfValues(3, 4, 5, 6).And.BeOfType(NPTypeCode.UInt32);
    }

    [Test]
    public void Add_Int64_SameType()
    {
        var a = np.array(new long[] { 1, 2, 3, 4 });
        var b = np.array(new long[] { 2, 2, 2, 2 });
        var result = a + b;

        result.Should().BeOfValues(3L, 4L, 5L, 6L).And.BeOfType(NPTypeCode.Int64);
    }

    [Test]
    public void Add_UInt64_SameType()
    {
        var a = np.array(new ulong[] { 1, 2, 3, 4 });
        var b = np.array(new ulong[] { 2, 2, 2, 2 });
        var result = a + b;

        result.Should().BeOfValues(3UL, 4UL, 5UL, 6UL).And.BeOfType(NPTypeCode.UInt64);
    }

    [Test]
    public void Add_Float32_SameType()
    {
        var a = np.array(new float[] { 1f, 2f, 3f, 4f });
        var b = np.array(new float[] { 2f, 2f, 2f, 2f });
        var result = a + b;

        result.Should().BeOfValues(3f, 4f, 5f, 6f).And.BeOfType(NPTypeCode.Single);
    }

    [Test]
    public void Add_Float64_SameType()
    {
        var a = np.array(new double[] { 1.0, 2.0, 3.0, 4.0 });
        var b = np.array(new double[] { 2.0, 2.0, 2.0, 2.0 });
        var result = a + b;

        result.Should().BeOfValues(3.0, 4.0, 5.0, 6.0).And.BeOfType(NPTypeCode.Double);
    }

    #endregion

    #region Same-Type Subtract Tests

    [Test]
    public void Subtract_Int32_SameType()
    {
        // NumPy: [1, 2, 3, 4] - [2, 2, 2, 2] = [-1, 0, 1, 2]
        var a = np.array(new[] { 1, 2, 3, 4 });
        var b = np.array(new[] { 2, 2, 2, 2 });
        var result = a - b;

        result.Should().BeOfValues(-1, 0, 1, 2).And.BeOfType(NPTypeCode.Int32);
    }

    [Test]
    public void Subtract_Byte_Underflow()
    {
        // NumPy: uint8 [1, 2, 3, 4] - [2, 2, 2, 2] = [255, 0, 1, 2] (wraps)
        var a = np.array(new byte[] { 1, 2, 3, 4 });
        var b = np.array(new byte[] { 2, 2, 2, 2 });
        var result = a - b;

        result.Should().BeOfValues(255, 0, 1, 2).And.BeOfType(NPTypeCode.Byte);
    }

    [Test]
    public void Subtract_UInt16_Underflow()
    {
        // NumPy: uint16 [1, 2, 3, 4] - [2, 2, 2, 2] = [65535, 0, 1, 2]
        var a = np.array(new ushort[] { 1, 2, 3, 4 });
        var b = np.array(new ushort[] { 2, 2, 2, 2 });
        var result = a - b;

        result.Should().BeOfValues(65535, 0, 1, 2).And.BeOfType(NPTypeCode.UInt16);
    }

    [Test]
    public void Subtract_UInt32_Underflow()
    {
        // NumPy: uint32 [1, 2, 3, 4] - [2, 2, 2, 2] = [4294967295, 0, 1, 2]
        var a = np.array(new uint[] { 1, 2, 3, 4 });
        var b = np.array(new uint[] { 2, 2, 2, 2 });
        var result = a - b;

        result.Should().BeOfValues(4294967295u, 0u, 1u, 2u).And.BeOfType(NPTypeCode.UInt32);
    }

    [Test]
    public void Subtract_Float64_SameType()
    {
        var a = np.array(new double[] { 1.0, 2.0, 3.0, 4.0 });
        var b = np.array(new double[] { 2.0, 2.0, 2.0, 2.0 });
        var result = a - b;

        result.Should().BeOfValues(-1.0, 0.0, 1.0, 2.0).And.BeOfType(NPTypeCode.Double);
    }

    #endregion

    #region Same-Type Multiply Tests

    [Test]
    public void Multiply_Bool_SameType()
    {
        // NumPy: True * True = True, True * False = False (logical AND behavior)
        var a = np.array(new[] { true, false, true, false });
        var b = np.array(new[] { true, true, false, false });
        var result = a * b;

        Assert.IsTrue(result.GetBoolean(0));
        Assert.IsFalse(result.GetBoolean(1));
        Assert.IsFalse(result.GetBoolean(2));
        Assert.IsFalse(result.GetBoolean(3));
    }

    [Test]
    public void Multiply_Int32_SameType()
    {
        var a = np.array(new[] { 1, 2, 3, 4 });
        var b = np.array(new[] { 2, 2, 2, 2 });
        var result = a * b;

        result.Should().BeOfValues(2, 4, 6, 8).And.BeOfType(NPTypeCode.Int32);
    }

    [Test]
    public void Multiply_Float64_SameType()
    {
        var a = np.array(new double[] { 1.0, 2.0, 3.0, 4.0 });
        var b = np.array(new double[] { 2.0, 2.0, 2.0, 2.0 });
        var result = a * b;

        result.Should().BeOfValues(2.0, 4.0, 6.0, 8.0).And.BeOfType(NPTypeCode.Double);
    }

    #endregion

    #region Same-Type Divide Tests

    [Test]
    public void Divide_Float32_SameType()
    {
        var a = np.array(new float[] { 1f, 2f, 3f, 4f });
        var b = np.array(new float[] { 2f, 2f, 2f, 2f });
        var result = a / b;

        result.Should().BeOfValues(0.5f, 1.0f, 1.5f, 2.0f).And.BeOfType(NPTypeCode.Single);
    }

    [Test]
    public void Divide_Float64_SameType()
    {
        var a = np.array(new double[] { 1.0, 2.0, 3.0, 4.0 });
        var b = np.array(new double[] { 2.0, 2.0, 2.0, 2.0 });
        var result = a / b;

        result.Should().BeOfValues(0.5, 1.0, 1.5, 2.0).And.BeOfType(NPTypeCode.Double);
    }

    [Test]
    [OpenBugs] // NumSharp does integer division, NumPy promotes to float64
    public void Divide_Int32_ReturnsDouble()
    {
        // NumPy: int32 / int32 returns float64
        var a = np.array(new[] { 1, 2, 3, 4 });
        var b = np.array(new[] { 2, 2, 2, 2 });
        var result = a / b;

        result.Should().BeOfValues(0.5, 1.0, 1.5, 2.0).And.BeOfType(NPTypeCode.Double);
    }

    #endregion

    #region Same-Type Mod Tests

    [Test]
    public void Mod_Int32_SameType()
    {
        // NumPy: [1, 2, 3, 4] % [2, 3, 2, 3] = [1, 2, 1, 1]
        var a = np.array(new[] { 1, 2, 3, 4 });
        var b = np.array(new[] { 2, 3, 2, 3 });
        var result = a % b;

        result.Should().BeOfValues(1, 2, 1, 1).And.BeOfType(NPTypeCode.Int32);
    }

    [Test]
    public void Mod_Float64_SameType()
    {
        var a = np.array(new double[] { 1.0, 2.0, 3.0, 4.0 });
        var b = np.array(new double[] { 2.0, 3.0, 2.0, 3.0 });
        var result = a % b;

        result.Should().BeOfValues(1.0, 2.0, 1.0, 1.0).And.BeOfType(NPTypeCode.Double);
    }

    #endregion

    #region Scalar Broadcasting Tests

    [Test]
    public void Add_ArrayPlusScalar_Int32()
    {
        // NumPy: [1, 2, 3, 4] + 2 = [3, 4, 5, 6]
        var a = np.array(new[] { 1, 2, 3, 4 });
        var result = a + 2;

        result.Should().BeOfValues(3, 4, 5, 6).And.BeOfType(NPTypeCode.Int32);
    }

    [Test]
    public void Add_ScalarPlusArray_Int32()
    {
        // NumPy: 2 + [1, 2, 3, 4] = [3, 4, 5, 6]
        var a = np.array(new[] { 1, 2, 3, 4 });
        var result = 2 + a;

        result.Should().BeOfValues(3, 4, 5, 6).And.BeOfType(NPTypeCode.Int32);
    }

    [Test]
    public void Subtract_ArrayMinusScalar_Int32()
    {
        // NumPy: [1, 2, 3, 4] - 2 = [-1, 0, 1, 2]
        var a = np.array(new[] { 1, 2, 3, 4 });
        var result = a - 2;

        result.Should().BeOfValues(-1, 0, 1, 2).And.BeOfType(NPTypeCode.Int32);
    }

    [Test]
    public void Subtract_ScalarMinusArray_Int32()
    {
        // NumPy: 2 - [1, 2, 3, 4] = [1, 0, -1, -2]
        var a = np.array(new[] { 1, 2, 3, 4 });
        var result = 2 - a;

        result.Should().BeOfValues(1, 0, -1, -2).And.BeOfType(NPTypeCode.Int32);
    }

    [Test]
    public void Multiply_ArrayTimesScalar_Float64()
    {
        var a = np.array(new double[] { 1.0, 2.0, 3.0, 4.0 });
        var result = a * 2.0;

        result.Should().BeOfValues(2.0, 4.0, 6.0, 8.0).And.BeOfType(NPTypeCode.Double);
    }

    [Test]
    public void Divide_ArrayDividedByScalar_Float64()
    {
        var a = np.array(new double[] { 1.0, 2.0, 3.0, 4.0 });
        var result = a / 2.0;

        result.Should().BeOfValues(0.5, 1.0, 1.5, 2.0).And.BeOfType(NPTypeCode.Double);
    }

    [Test]
    public void Divide_ScalarDividedByArray_Float64()
    {
        // NumPy: 2 / [1, 2, 3, 4] = [2.0, 1.0, 0.666..., 0.5]
        var a = np.array(new double[] { 1.0, 2.0, 3.0, 4.0 });
        var result = 2.0 / a;

        Assert.AreEqual(2.0, result.GetDouble(0));
        Assert.AreEqual(1.0, result.GetDouble(1));
        Assert.IsTrue(Math.Abs(result.GetDouble(2) - 0.6666666666666666) < 1e-10);
        Assert.AreEqual(0.5, result.GetDouble(3));
    }

    [Test]
    public void Mod_ArrayModScalar_Int32()
    {
        // NumPy: [1, 2, 3, 4] % 2 = [1, 0, 1, 0]
        var a = np.array(new[] { 1, 2, 3, 4 });
        var result = a % 2;

        result.Should().BeOfValues(1, 0, 1, 0).And.BeOfType(NPTypeCode.Int32);
    }

    [Test]
    public void Mod_ScalarModArray_Int32()
    {
        // NumPy: 2 % [1, 2, 3, 4] = [0, 0, 2, 2]
        var a = np.array(new[] { 1, 2, 3, 4 });
        var result = 2 % a;

        result.Should().BeOfValues(0, 0, 2, 2).And.BeOfType(NPTypeCode.Int32);
    }

    #endregion

    #region Broadcasting Shape Tests

    [Test]
    public void Add_2D_Plus_1D_Broadcasting()
    {
        // NumPy: (3,4) + (4,) = [[2,4,6,8],[6,8,10,12],[10,12,14,16]]
        var a = np.array(new[,] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 }, { 9, 10, 11, 12 } });
        var b = np.array(new[] { 1, 2, 3, 4 });
        var result = a + b;

        result.Should().BeShaped(3, 4);
        result.Should().BeOfValues(2, 4, 6, 8, 6, 8, 10, 12, 10, 12, 14, 16);
    }

    [Test]
    public void Add_Column_Plus_Row_Broadcasting()
    {
        // NumPy: (3,1) + (1,4) = [[2,3,4,5],[3,4,5,6],[4,5,6,7]]
        var a = np.array(new[,] { { 1 }, { 2 }, { 3 } });
        var b = np.array(new[,] { { 1, 2, 3, 4 } });
        var result = a + b;

        result.Should().BeShaped(3, 4);
        result.Should().BeOfValues(2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 7);
    }

    [Test]
    public void Add_2D_Plus_1D_Float64()
    {
        // NumPy: [[1,2],[3,4]] + [10,20] = [[11,22],[13,24]]
        var a = np.array(new[,] { { 1.0, 2.0 }, { 3.0, 4.0 } });
        var b = np.array(new[] { 10.0, 20.0 });
        var result = a + b;

        result.Should().BeShaped(2, 2);
        result.Should().BeOfValues(11.0, 22.0, 13.0, 24.0);
    }

    #endregion

    #region Edge Cases - Division by Zero

    [Test]
    public void Divide_Float64_DivisionByZero()
    {
        // NumPy: [1.0, -1.0, 0.0] / [0.0, 0.0, 0.0] = [inf, -inf, nan]
        var a = np.array(new double[] { 1.0, -1.0, 0.0 });
        var b = np.array(new double[] { 0.0, 0.0, 0.0 });
        var result = a / b;

        Assert.IsTrue(double.IsPositiveInfinity(result.GetDouble(0)));
        Assert.IsTrue(double.IsNegativeInfinity(result.GetDouble(1)));
        Assert.IsTrue(double.IsNaN(result.GetDouble(2)));
    }

    #endregion

    #region Edge Cases - Infinity Arithmetic

    [Test]
    public void Add_InfinityArithmetic()
    {
        // NumPy: [inf, -inf, inf, 1.0] + [1.0, 1.0, inf, inf] = [inf, -inf, inf, inf]
        var a = np.array(new double[] { double.PositiveInfinity, double.NegativeInfinity, double.PositiveInfinity, 1.0 });
        var b = np.array(new double[] { 1.0, 1.0, double.PositiveInfinity, double.PositiveInfinity });
        var result = a + b;

        Assert.IsTrue(double.IsPositiveInfinity(result.GetDouble(0)));
        Assert.IsTrue(double.IsNegativeInfinity(result.GetDouble(1)));
        Assert.IsTrue(double.IsPositiveInfinity(result.GetDouble(2)));
        Assert.IsTrue(double.IsPositiveInfinity(result.GetDouble(3)));
    }

    [Test]
    public void Subtract_InfMinusInf_IsNaN()
    {
        // NumPy: inf - inf = nan
        var a = np.array(new double[] { double.PositiveInfinity });
        var b = np.array(new double[] { double.PositiveInfinity });
        var result = a - b;

        Assert.IsTrue(double.IsNaN(result.GetDouble(0)));
    }

    [Test]
    public void Multiply_InfTimesZero_IsNaN()
    {
        // NumPy: inf * 0 = nan
        var a = np.array(new double[] { double.PositiveInfinity });
        var b = np.array(new double[] { 0.0 });
        var result = a * b;

        Assert.IsTrue(double.IsNaN(result.GetDouble(0)));
    }

    #endregion

    #region Edge Cases - NaN Propagation

    [Test]
    public void Add_NaNPropagation()
    {
        // NumPy: [1.0, nan, 3.0] + [2.0, 2.0, nan] = [3.0, nan, nan]
        var a = np.array(new double[] { 1.0, double.NaN, 3.0 });
        var b = np.array(new double[] { 2.0, 2.0, double.NaN });
        var result = a + b;

        Assert.AreEqual(3.0, result.GetDouble(0));
        Assert.IsTrue(double.IsNaN(result.GetDouble(1)));
        Assert.IsTrue(double.IsNaN(result.GetDouble(2)));
    }

    [Test]
    public void Multiply_NaNPropagation()
    {
        var a = np.array(new double[] { 1.0, double.NaN, 3.0 });
        var b = np.array(new double[] { 2.0, 2.0, double.NaN });
        var result = a * b;

        Assert.AreEqual(2.0, result.GetDouble(0));
        Assert.IsTrue(double.IsNaN(result.GetDouble(1)));
        Assert.IsTrue(double.IsNaN(result.GetDouble(2)));
    }

    #endregion

    #region Edge Cases - 0D Scalars

    [Test]
    [OpenBugs] // NumSharp doesn't properly handle 0D scalars (returns 1D instead)
    public void Add_0DScalars()
    {
        // NumPy: np.array(5) + np.array(3) = 8, shape=()
        var a = np.array(5);
        var b = np.array(3);
        var result = a + b;

        Assert.AreEqual(0, result.ndim);
        Assert.AreEqual(8, result.GetInt32(0));
    }

    #endregion

    #region Edge Cases - Empty Arrays

    [Test]
    public void Add_EmptyArrays()
    {
        // NumPy: [] + [] = [], shape=(0,)
        var a = np.array(Array.Empty<double>());
        var b = np.array(Array.Empty<double>());
        var result = a + b;

        // Note: NumSharp may return IsEmpty=true, size=0 but with different shape representation
        Assert.AreEqual(0, result.size);
    }

    #endregion

    #region Type Promotion Tests

    [Test]
    public void Add_Int32_Float64_Promotion()
    {
        // NumPy: int32 + float64 → float64: [1, 2, 3] + [0.5, 0.5, 0.5] = [1.5, 2.5, 3.5]
        var a = np.array(new[] { 1, 2, 3 });
        var b = np.array(new double[] { 0.5, 0.5, 0.5 });
        var result = a + b;

        result.Should().BeOfValues(1.5, 2.5, 3.5).And.BeOfType(NPTypeCode.Double);
    }

    [Test]
    public void Add_Int32_Int64_Promotion()
    {
        // NumPy: int32 + int64 → int64
        var a = np.array(new[] { 1, 2, 3 });
        var b = np.array(new long[] { 1, 2, 3 });
        var result = a + b;

        result.Should().BeOfValues(2L, 4L, 6L).And.BeOfType(NPTypeCode.Int64);
    }

    [Test]
    public void Add_Byte_Int32_Promotion()
    {
        // NumPy: uint8 + int32 → int32
        var a = np.array(new byte[] { 1, 2, 3 });
        var b = np.array(new[] { 1, 2, 3 });
        var result = a + b;

        result.Should().BeOfValues(2, 4, 6).And.BeOfType(NPTypeCode.Int32);
    }

    [Test]
    public void Add_Bool_Int32_Promotion()
    {
        // NumPy: bool + int32 → int32: [True, False, True] + [1, 2, 3] = [2, 2, 4]
        var a = np.array(new[] { true, false, true });
        var b = np.array(new[] { 1, 2, 3 });
        var result = a + b;

        result.Should().BeOfValues(2, 2, 4).And.BeOfType(NPTypeCode.Int32);
    }

    #endregion

    #region Overflow Behavior Tests

    [Test]
    public void Add_Byte_Overflow()
    {
        // NumPy: uint8 255 + 1 = 0 (wraps)
        var a = np.array(new byte[] { 255 });
        var b = np.array(new byte[] { 1 });
        var result = a + b;

        result.Should().BeOfValues(0).And.BeOfType(NPTypeCode.Byte);
    }

    [Test]
    public void Add_Int32_Overflow()
    {
        // NumPy: int32 2147483647 + 1 = -2147483648 (wraps)
        var a = np.array(new[] { int.MaxValue });
        var b = np.array(new[] { 1 });
        var result = a + b;

        result.Should().BeOfValues(int.MinValue).And.BeOfType(NPTypeCode.Int32);
    }

    #endregion

    #region ATan2 Tests (Bug Fix Verification)

    [Test]
    public void ATan2_Float64()
    {
        // NumPy: np.arctan2([1, 1, -1, -1], [1, -1, 1, -1]) = [π/4, 3π/4, -π/4, -3π/4]
        var y = np.array(new double[] { 1.0, 1.0, -1.0, -1.0 });
        var x = np.array(new double[] { 1.0, -1.0, 1.0, -1.0 });
        var result = np.arctan2(y, x);

        Assert.IsTrue(Math.Abs(result.GetDouble(0) - Math.PI / 4) < 1e-10);
        Assert.IsTrue(Math.Abs(result.GetDouble(1) - 3 * Math.PI / 4) < 1e-10);
        Assert.IsTrue(Math.Abs(result.GetDouble(2) - (-Math.PI / 4)) < 1e-10);
        Assert.IsTrue(Math.Abs(result.GetDouble(3) - (-3 * Math.PI / 4)) < 1e-10);
    }

    [Test]
    public void ATan2_Float32()
    {
        // NumPy: np.arctan2([1, 0, -1], [0, 1, 0]) = [π/2, 0, -π/2]
        var y = np.array(new float[] { 1.0f, 0.0f, -1.0f });
        var x = np.array(new float[] { 0.0f, 1.0f, 0.0f });
        var result = np.arctan2(y, x);

        Assert.IsTrue(Math.Abs(result.GetSingle(0) - (float)(Math.PI / 2)) < 1e-6f);
        Assert.IsTrue(Math.Abs(result.GetSingle(1) - 0.0f) < 1e-6f);
        Assert.IsTrue(Math.Abs(result.GetSingle(2) - (float)(-Math.PI / 2)) < 1e-6f);
    }

    [Test]
    public void ATan2_Int32()
    {
        // NumPy: np.arctan2([1000, 0, -1000], [0, 1000, 0]) returns float64
        // Integer inputs are promoted to float64 for arctan2 (same as NumPy)
        var y = np.array(new int[] { 1000, 0, -1000 });
        var x = np.array(new int[] { 0, 1000, 0 });
        var result = np.arctan2(y, x);

        // arctan2 always returns float64, even for integer inputs
        Assert.AreEqual(typeof(double), result.dtype);
        Assert.IsTrue(Math.Abs(result.GetDouble(0) - Math.PI / 2) < 1e-10);  // atan2(1000, 0) = π/2
        Assert.AreEqual(0.0, result.GetDouble(1));                           // atan2(0, 1000) = 0
        Assert.IsTrue(Math.Abs(result.GetDouble(2) + Math.PI / 2) < 1e-10);  // atan2(-1000, 0) = -π/2
    }

    [Test]
    public void ATan2_Int64()
    {
        // NumPy: np.arctan2(int64, int64) returns float64
        // Integer inputs are promoted to float64 for arctan2 (same as NumPy)
        var y = np.array(new long[] { 1000000L, 0L, -1000000L });
        var x = np.array(new long[] { 0L, 1000000L, 0L });
        var result = np.arctan2(y, x);

        // arctan2 always returns float64, even for integer inputs
        Assert.AreEqual(typeof(double), result.dtype);
        Assert.IsTrue(Math.Abs(result.GetDouble(0) - Math.PI / 2) < 1e-10);  // atan2(1M, 0) = π/2
        Assert.AreEqual(0.0, result.GetDouble(1));                           // atan2(0, 1M) = 0
        Assert.IsTrue(Math.Abs(result.GetDouble(2) + Math.PI / 2) < 1e-10);  // atan2(-1M, 0) = -π/2
    }

    [Test]
    public void ATan2_SpecialValues()
    {
        // NumPy: arctan2(0, 0) = 0, arctan2(inf, 1) = π/2, arctan2(1, inf) = 0
        var y = np.array(new double[] { 0.0, double.PositiveInfinity, 1.0, double.NaN });
        var x = np.array(new double[] { 0.0, 1.0, double.PositiveInfinity, 1.0 });
        var result = np.arctan2(y, x);

        Assert.AreEqual(0.0, result.GetDouble(0));
        Assert.IsTrue(Math.Abs(result.GetDouble(1) - Math.PI / 2) < 1e-10);
        Assert.AreEqual(0.0, result.GetDouble(2));
        Assert.IsTrue(double.IsNaN(result.GetDouble(3)));
    }

    [Test]
    public void ATan2_SlicedArray()
    {
        // Test with sliced (non-contiguous) input: arr[::2]
        // NumPy: np.arctan2(y[::2], x[::2]) should work correctly
        var y_full = np.array(new double[] { 1.0, 999.0, 1.0, 999.0, -1.0, 999.0 });
        var x_full = np.array(new double[] { 1.0, 999.0, -1.0, 999.0, 1.0, 999.0 });

        // Slice to get every other element (non-contiguous view)
        var y = y_full["::2"];
        var x = x_full["::2"];

        // Verify inputs are non-contiguous
        Assert.IsFalse(y.Shape.IsContiguous, "y should be non-contiguous slice");
        Assert.IsFalse(x.Shape.IsContiguous, "x should be non-contiguous slice");

        var result = np.arctan2(y, x);

        // Expected: arctan2([1, 1, -1], [1, -1, 1]) = [π/4, 3π/4, -π/4]
        Assert.IsTrue(Math.Abs(result.GetDouble(0) - Math.PI / 4) < 1e-10);
        Assert.IsTrue(Math.Abs(result.GetDouble(1) - 3 * Math.PI / 4) < 1e-10);
        Assert.IsTrue(Math.Abs(result.GetDouble(2) - (-Math.PI / 4)) < 1e-10);
    }

    [Test]
    public void ATan2_BroadcastScalar()
    {
        // Test broadcasting: arctan2(array, scalar)
        // NumPy: np.arctan2([1, 0, -1], 1) = [π/4, 0, -π/4]
        var y = np.array(new double[] { 1.0, 0.0, -1.0 });
        var x = np.array(1.0);  // Scalar (will be broadcast)

        var result = np.arctan2(y, x);

        Assert.AreEqual(3, result.size);
        Assert.IsTrue(Math.Abs(result.GetDouble(0) - Math.PI / 4) < 1e-10);
        Assert.AreEqual(0.0, result.GetDouble(1));
        Assert.IsTrue(Math.Abs(result.GetDouble(2) - (-Math.PI / 4)) < 1e-10);
    }

    [Test]
    public void ATan2_Broadcast2D()
    {
        // Test 2D broadcasting: arctan2(column, row)
        // NumPy: np.arctan2([[1], [0], [-1]], [1, -1]) broadcasts to (3, 2)
        var y = np.array(new double[,] { { 1.0 }, { 0.0 }, { -1.0 } });  // Shape (3, 1)
        var x = np.array(new double[] { 1.0, -1.0 });                    // Shape (2,)

        var result = np.arctan2(y, x);

        // Result shape should be (3, 2)
        Assert.IsTrue(result.shape.SequenceEqual(new long[] { 3, 2 }));

        // Row 0: arctan2(1, 1) = π/4, arctan2(1, -1) = 3π/4
        Assert.IsTrue(Math.Abs(result.GetDouble(0, 0) - Math.PI / 4) < 1e-10);
        Assert.IsTrue(Math.Abs(result.GetDouble(0, 1) - 3 * Math.PI / 4) < 1e-10);

        // Row 1: arctan2(0, 1) = 0, arctan2(0, -1) = π
        Assert.AreEqual(0.0, result.GetDouble(1, 0));
        Assert.IsTrue(Math.Abs(result.GetDouble(1, 1) - Math.PI) < 1e-10);

        // Row 2: arctan2(-1, 1) = -π/4, arctan2(-1, -1) = -3π/4
        Assert.IsTrue(Math.Abs(result.GetDouble(2, 0) - (-Math.PI / 4)) < 1e-10);
        Assert.IsTrue(Math.Abs(result.GetDouble(2, 1) - (-3 * Math.PI / 4)) < 1e-10);
    }

    #endregion
}
