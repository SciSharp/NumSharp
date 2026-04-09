using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using AwesomeAssertions;
using NumSharp;
using NumSharp.Backends;
using NumSharp.UnitTest;

namespace NumSharp.UnitTest.NpApiOverloads;

/// <summary>
/// Tests verifying that binary math operations and reduction overloads compile and work correctly
/// after removing the `in` parameter modifier from method signatures.
/// </summary>
public class NpApiOverloadTests_BinaryReductions
{
    #region Basic Binary Operations

    [Test]
    public void Add_TwoArrays_Compiles()
    {
        var a = np.array(new double[] { 1, 2, 3 });
        var b = np.array(new double[] { 4, 5, 6 });
        var result = np.add(a, b);
        result.Should().NotBeNull();
        result.GetDouble(0).Should().Be(5.0);
        result.GetDouble(1).Should().Be(7.0);
        result.GetDouble(2).Should().Be(9.0);
    }

    [Test]
    public void Subtract_TwoArrays_Compiles()
    {
        var a = np.array(new double[] { 5, 7, 9 });
        var b = np.array(new double[] { 1, 2, 3 });
        var result = np.subtract(a, b);
        result.Should().NotBeNull();
        result.GetDouble(0).Should().Be(4.0);
        result.GetDouble(1).Should().Be(5.0);
        result.GetDouble(2).Should().Be(6.0);
    }

    [Test]
    public void Multiply_TwoArrays_Compiles()
    {
        var a = np.array(new double[] { 2, 3, 4 });
        var b = np.array(new double[] { 5, 6, 7 });
        var result = np.multiply(a, b);
        result.Should().NotBeNull();
        result.GetDouble(0).Should().Be(10.0);
        result.GetDouble(1).Should().Be(18.0);
        result.GetDouble(2).Should().Be(28.0);
    }

    [Test]
    public void Divide_TwoArrays_Compiles()
    {
        var a = np.array(new double[] { 10, 20, 30 });
        var b = np.array(new double[] { 2, 4, 5 });
        var result = np.divide(a, b);
        result.Should().NotBeNull();
        result.GetDouble(0).Should().Be(5.0);
        result.GetDouble(1).Should().Be(5.0);
        result.GetDouble(2).Should().Be(6.0);
    }

    [Test]
    public void TrueDivide_TwoArrays_Compiles()
    {
        var a = np.array(new double[] { 10, 20, 30 });
        var b = np.array(new double[] { 3, 4, 6 });
        var result = np.true_divide(a, b);
        result.Should().NotBeNull();
        result.GetDouble(0).Should().BeApproximately(10.0 / 3.0, 1e-10);
        result.GetDouble(1).Should().Be(5.0);
        result.GetDouble(2).Should().Be(5.0);
    }

    [Test]
    public void Mod_TwoArrays_Compiles()
    {
        var a = np.array(new double[] { 10, 20, 30 });
        var b = np.array(new double[] { 3, 7, 8 });
        var result = np.mod(a, b);
        result.Should().NotBeNull();
        result.GetDouble(0).Should().Be(1.0);
        result.GetDouble(1).Should().Be(6.0);
        result.GetDouble(2).Should().Be(6.0);
    }

    [Test]
    public void Mod_ArrayAndFloatScalar_Compiles()
    {
        var a = np.array(new double[] { 10, 20, 30 });
        var result = np.mod(a, 7.0f);
        result.Should().NotBeNull();
        result.GetDouble(0).Should().Be(3.0);
        result.GetDouble(1).Should().Be(6.0);
        result.GetDouble(2).Should().Be(2.0);
    }

    #endregion

    #region Power Operations

    [Test]
    public void Power_ArrayAndValueTypeScalar_Compiles()
    {
        var a = np.array(new double[] { 2, 3, 4 });
        ValueType exp = 2.0;
        var result = np.power(a, exp);
        result.Should().NotBeNull();
        result.GetDouble(0).Should().Be(4.0);
        result.GetDouble(1).Should().Be(9.0);
        result.GetDouble(2).Should().Be(16.0);
    }

    [Test]
    public void Power_ArrayAndValueTypeScalarWithDtype_Compiles()
    {
        var a = np.array(new double[] { 2, 3, 4 });
        ValueType exp = 2.0;
        var result = np.power(a, exp, typeof(double));
        result.Should().NotBeNull();
        result.dtype.Should().Be(typeof(double));
    }

    [Test]
    public void Power_ArrayAndValueTypeScalarWithNPTypeCode_Compiles()
    {
        var a = np.array(new double[] { 2, 3, 4 });
        ValueType exp = 2.0;
        var result = np.power(a, exp, NPTypeCode.Double);
        result.Should().NotBeNull();
        result.typecode.Should().Be(NPTypeCode.Double);
    }

    [Test]
    public void Power_TwoArrays_Compiles()
    {
        var a = np.array(new double[] { 2, 3, 4 });
        var b = np.array(new double[] { 2, 2, 2 });
        var result = np.power(a, b);
        result.Should().NotBeNull();
        result.GetDouble(0).Should().Be(4.0);
        result.GetDouble(1).Should().Be(9.0);
        result.GetDouble(2).Should().Be(16.0);
    }

    [Test]
    public void Power_TwoArraysWithDtype_Compiles()
    {
        var a = np.array(new double[] { 2, 3, 4 });
        var b = np.array(new double[] { 2, 2, 2 });
        var result = np.power(a, b, typeof(double));
        result.Should().NotBeNull();
        result.dtype.Should().Be(typeof(double));
    }

    [Test]
    public void Power_TwoArraysWithNPTypeCode_Compiles()
    {
        var a = np.array(new double[] { 2, 3, 4 });
        var b = np.array(new double[] { 2, 2, 2 });
        var result = np.power(a, b, NPTypeCode.Double);
        result.Should().NotBeNull();
        result.typecode.Should().Be(NPTypeCode.Double);
    }

    #endregion

    #region Floor Division Operations

    [Test]
    public void FloorDivide_TwoArrays_Compiles()
    {
        var a = np.array(new double[] { 7, 8, 9 });
        var b = np.array(new double[] { 2, 3, 4 });
        var result = np.floor_divide(a, b);
        result.Should().NotBeNull();
        result.GetDouble(0).Should().Be(3.0);
        result.GetDouble(1).Should().Be(2.0);
        result.GetDouble(2).Should().Be(2.0);
    }

    [Test]
    public void FloorDivide_TwoArraysWithDtype_Compiles()
    {
        var a = np.array(new double[] { 7, 8, 9 });
        var b = np.array(new double[] { 2, 3, 4 });
        var result = np.floor_divide(a, b, typeof(double));
        result.Should().NotBeNull();
        result.dtype.Should().Be(typeof(double));
    }

    [Test]
    public void FloorDivide_TwoArraysWithNPTypeCode_Compiles()
    {
        var a = np.array(new double[] { 7, 8, 9 });
        var b = np.array(new double[] { 2, 3, 4 });
        var result = np.floor_divide(a, b, NPTypeCode.Double);
        result.Should().NotBeNull();
        result.typecode.Should().Be(NPTypeCode.Double);
    }

    [Test]
    public void FloorDivide_ArrayAndValueTypeScalar_Compiles()
    {
        var a = np.array(new double[] { 7, 8, 9 });
        ValueType divisor = 2.0;
        var result = np.floor_divide(a, divisor);
        result.Should().NotBeNull();
        result.GetDouble(0).Should().Be(3.0);
        result.GetDouble(1).Should().Be(4.0);
        result.GetDouble(2).Should().Be(4.0);
    }

    [Test]
    public void FloorDivide_ArrayAndValueTypeScalarWithDtype_Compiles()
    {
        var a = np.array(new double[] { 7, 8, 9 });
        ValueType divisor = 2.0;
        var result = np.floor_divide(a, divisor, typeof(double));
        result.Should().NotBeNull();
        result.dtype.Should().Be(typeof(double));
    }

    [Test]
    public void FloorDivide_ArrayAndValueTypeScalarWithNPTypeCode_Compiles()
    {
        var a = np.array(new double[] { 7, 8, 9 });
        ValueType divisor = 2.0;
        var result = np.floor_divide(a, divisor, NPTypeCode.Double);
        result.Should().NotBeNull();
        result.typecode.Should().Be(NPTypeCode.Double);
    }

    #endregion

    #region Maximum Operations

    [Test]
    public void Maximum_TwoArrays_Compiles()
    {
        var a = np.array(new double[] { 1, 5, 3 });
        var b = np.array(new double[] { 4, 2, 6 });
        var result = np.maximum(a, b);
        result.Should().NotBeNull();
        result.GetDouble(0).Should().Be(4.0);
        result.GetDouble(1).Should().Be(5.0);
        result.GetDouble(2).Should().Be(6.0);
    }

    [Test]
    public void Maximum_TwoArraysWithDtype_Compiles()
    {
        var a = np.array(new double[] { 1, 5, 3 });
        var b = np.array(new double[] { 4, 2, 6 });
        var result = np.maximum(a, b, typeof(double));
        result.Should().NotBeNull();
        result.dtype.Should().Be(typeof(double));
    }

    [Test]
    public void Maximum_TwoArraysWithOut_Compiles()
    {
        var a = np.array(new double[] { 1, 5, 3 });
        var b = np.array(new double[] { 4, 2, 6 });
        var @out = np.empty(new int[] { 3 }, np.float64);
        var result = np.maximum(a, b, @out);
        result.Should().NotBeNull();
        @out.GetDouble(0).Should().Be(4.0);
    }

    #endregion

    #region Minimum Operations

    [Test]
    public void Minimum_TwoArrays_Compiles()
    {
        var a = np.array(new double[] { 1, 5, 3 });
        var b = np.array(new double[] { 4, 2, 6 });
        var result = np.minimum(a, b);
        result.Should().NotBeNull();
        result.GetDouble(0).Should().Be(1.0);
        result.GetDouble(1).Should().Be(2.0);
        result.GetDouble(2).Should().Be(3.0);
    }

    [Test]
    public void Minimum_TwoArraysWithDtype_Compiles()
    {
        var a = np.array(new double[] { 1, 5, 3 });
        var b = np.array(new double[] { 4, 2, 6 });
        var result = np.minimum(a, b, typeof(double));
        result.Should().NotBeNull();
        result.dtype.Should().Be(typeof(double));
    }

    [Test]
    public void Minimum_TwoArraysWithOut_Compiles()
    {
        var a = np.array(new double[] { 1, 5, 3 });
        var b = np.array(new double[] { 4, 2, 6 });
        var @out = np.empty(new int[] { 3 }, np.float64);
        var result = np.minimum(a, b, @out);
        result.Should().NotBeNull();
        @out.GetDouble(0).Should().Be(1.0);
    }

    #endregion

    #region Fmax Operations

    [Test]
    public void Fmax_TwoArrays_Compiles()
    {
        var a = np.array(new double[] { 1, 5, 3 });
        var b = np.array(new double[] { 4, 2, 6 });
        var result = np.fmax(a, b);
        result.Should().NotBeNull();
        result.GetDouble(0).Should().Be(4.0);
        result.GetDouble(1).Should().Be(5.0);
        result.GetDouble(2).Should().Be(6.0);
    }

    [Test]
    public void Fmax_TwoArraysWithDtype_Compiles()
    {
        var a = np.array(new double[] { 1, 5, 3 });
        var b = np.array(new double[] { 4, 2, 6 });
        var result = np.fmax(a, b, typeof(double));
        result.Should().NotBeNull();
        result.dtype.Should().Be(typeof(double));
    }

    [Test]
    public void Fmax_TwoArraysWithOut_Compiles()
    {
        var a = np.array(new double[] { 1, 5, 3 });
        var b = np.array(new double[] { 4, 2, 6 });
        var @out = np.empty(new int[] { 3 }, np.float64);
        var result = np.fmax(a, b, @out);
        result.Should().NotBeNull();
        @out.GetDouble(0).Should().Be(4.0);
    }

    #endregion

    #region Fmin Operations

    [Test]
    public void Fmin_TwoArrays_Compiles()
    {
        var a = np.array(new double[] { 1, 5, 3 });
        var b = np.array(new double[] { 4, 2, 6 });
        var result = np.fmin(a, b);
        result.Should().NotBeNull();
        result.GetDouble(0).Should().Be(1.0);
        result.GetDouble(1).Should().Be(2.0);
        result.GetDouble(2).Should().Be(3.0);
    }

    [Test]
    public void Fmin_TwoArraysWithDtype_Compiles()
    {
        var a = np.array(new double[] { 1, 5, 3 });
        var b = np.array(new double[] { 4, 2, 6 });
        var result = np.fmin(a, b, typeof(double));
        result.Should().NotBeNull();
        result.dtype.Should().Be(typeof(double));
    }

    [Test]
    public void Fmin_TwoArraysWithOut_Compiles()
    {
        var a = np.array(new double[] { 1, 5, 3 });
        var b = np.array(new double[] { 4, 2, 6 });
        var @out = np.empty(new int[] { 3 }, np.float64);
        var result = np.fmin(a, b, @out);
        result.Should().NotBeNull();
        @out.GetDouble(0).Should().Be(1.0);
    }

    #endregion

    #region Clip Operations

    [Test]
    public void Clip_MinMax_Compiles()
    {
        var a = np.array(new double[] { 1, 5, 3, 8, 2 });
        var a_min = np.array(new double[] { 2, 2, 2, 2, 2 });
        var a_max = np.array(new double[] { 6, 6, 6, 6, 6 });
        var result = np.clip(a, a_min, a_max);
        result.Should().NotBeNull();
        result.GetDouble(0).Should().Be(2.0);  // clipped to min
        result.GetDouble(1).Should().Be(5.0);  // unchanged
        result.GetDouble(2).Should().Be(3.0);  // unchanged
        result.GetDouble(3).Should().Be(6.0);  // clipped to max
        result.GetDouble(4).Should().Be(2.0);  // clipped to min
    }

    [Test]
    public void Clip_MinMaxWithDtype_Compiles()
    {
        var a = np.array(new double[] { 1, 5, 3, 8, 2 });
        var a_min = np.array(new double[] { 2, 2, 2, 2, 2 });
        var a_max = np.array(new double[] { 6, 6, 6, 6, 6 });
        var result = np.clip(a, a_min, a_max, typeof(double));
        result.Should().NotBeNull();
        result.dtype.Should().Be(typeof(double));
    }

    [Test]
    public void Clip_MinMaxWithOut_Compiles()
    {
        var a = np.array(new double[] { 1, 5, 3, 8, 2 });
        var a_min = np.array(new double[] { 2, 2, 2, 2, 2 });
        var a_max = np.array(new double[] { 6, 6, 6, 6, 6 });
        var @out = np.empty(new int[] { 5 }, np.float64);
        var result = np.clip(a, a_min, a_max, @out);
        result.Should().NotBeNull();
        @out.GetDouble(0).Should().Be(2.0);
    }

    #endregion

    #region Bitwise Operations

    [Test]
    public void LeftShift_TwoArrays_Compiles()
    {
        var a = np.array(new int[] { 1, 2, 4 });
        var b = np.array(new int[] { 1, 2, 3 });
        var result = np.left_shift(a, b);
        result.Should().NotBeNull();
        result.GetInt32(0).Should().Be(2);   // 1 << 1 = 2
        result.GetInt32(1).Should().Be(8);   // 2 << 2 = 8
        result.GetInt32(2).Should().Be(32);  // 4 << 3 = 32
    }

    [Test]
    public void LeftShift_ArrayAndIntScalar_Compiles()
    {
        var a = np.array(new int[] { 1, 2, 4 });
        var result = np.left_shift(a, 2);
        result.Should().NotBeNull();
        result.GetInt32(0).Should().Be(4);   // 1 << 2 = 4
        result.GetInt32(1).Should().Be(8);   // 2 << 2 = 8
        result.GetInt32(2).Should().Be(16);  // 4 << 2 = 16
    }

    [Test]
    public void RightShift_TwoArrays_Compiles()
    {
        var a = np.array(new int[] { 4, 8, 32 });
        var b = np.array(new int[] { 1, 2, 3 });
        var result = np.right_shift(a, b);
        result.Should().NotBeNull();
        result.GetInt32(0).Should().Be(2);   // 4 >> 1 = 2
        result.GetInt32(1).Should().Be(2);   // 8 >> 2 = 2
        result.GetInt32(2).Should().Be(4);   // 32 >> 3 = 4
    }

    [Test]
    public void RightShift_ArrayAndIntScalar_Compiles()
    {
        var a = np.array(new int[] { 4, 8, 16 });
        var result = np.right_shift(a, 2);
        result.Should().NotBeNull();
        result.GetInt32(0).Should().Be(1);   // 4 >> 2 = 1
        result.GetInt32(1).Should().Be(2);   // 8 >> 2 = 2
        result.GetInt32(2).Should().Be(4);   // 16 >> 2 = 4
    }

    [Test]
    public void Invert_Array_Compiles()
    {
        var a = np.array(new int[] { 0, 1, -1 });
        var result = np.invert(a);
        result.Should().NotBeNull();
        result.GetInt32(0).Should().Be(-1);  // ~0 = -1
        result.GetInt32(1).Should().Be(-2);  // ~1 = -2
        result.GetInt32(2).Should().Be(0);   // ~-1 = 0
    }

    [Test]
    public void Invert_ArrayWithDtype_Compiles()
    {
        var a = np.array(new int[] { 0, 1, -1 });
        var result = np.invert(a, typeof(int));
        result.Should().NotBeNull();
    }

    [Test]
    public void BitwiseNot_Array_Compiles()
    {
        var a = np.array(new int[] { 0, 1, -1 });
        var result = np.bitwise_not(a);
        result.Should().NotBeNull();
        result.GetInt32(0).Should().Be(-1);  // ~0 = -1
        result.GetInt32(1).Should().Be(-2);  // ~1 = -2
        result.GetInt32(2).Should().Be(0);   // ~-1 = 0
    }

    [Test]
    public void BitwiseNot_ArrayWithDtype_Compiles()
    {
        var a = np.array(new int[] { 0, 1, -1 });
        var result = np.bitwise_not(a, typeof(int));
        result.Should().NotBeNull();
    }

    #endregion

    #region Arctan2 Operations

    [Test]
    public void Arctan2_TwoArrays_Compiles()
    {
        var y = np.array(new double[] { 1, 0, -1 });
        var x = np.array(new double[] { 0, 1, 0 });
        var result = np.arctan2(y, x);
        result.Should().NotBeNull();
        result.GetDouble(0).Should().BeApproximately(Math.PI / 2, 1e-10);   // arctan2(1, 0) = pi/2
        result.GetDouble(1).Should().BeApproximately(0.0, 1e-10);            // arctan2(0, 1) = 0
        result.GetDouble(2).Should().BeApproximately(-Math.PI / 2, 1e-10);  // arctan2(-1, 0) = -pi/2
    }

    [Test]
    public void Arctan2_TwoArraysWithDtype_Compiles()
    {
        var y = np.array(new double[] { 1, 0, -1 });
        var x = np.array(new double[] { 0, 1, 0 });
        var result = np.arctan2(y, x, typeof(double));
        result.Should().NotBeNull();
        result.dtype.Should().Be(typeof(double));
    }

    #endregion

    #region Sum Reductions

    [Test]
    public void Sum_NoParams_Compiles()
    {
        var a = np.array(new double[] { 1, 2, 3 });
        var result = np.sum(a);
        result.GetDouble(0).Should().Be(6.0);
    }

    [Test]
    public void Sum_WithAxis_Compiles()
    {
        var a = np.array(new double[,] { { 1, 2 }, { 3, 4 } });
        var result = np.sum(a, 0);
        result.Should().NotBeNull();
        result.GetDouble(0).Should().Be(4.0);  // 1 + 3
        result.GetDouble(1).Should().Be(6.0);  // 2 + 4
    }

    [Test]
    public void Sum_WithKeepdims_Compiles()
    {
        var a = np.array(new double[] { 1, 2, 3 });
        var result = np.sum(a, keepdims: true);
        result.Should().NotBeNull();
        result.GetDouble(0).Should().Be(6.0);
    }

    [Test]
    public void Sum_WithAxisAndKeepdims_Compiles()
    {
        var a = np.array(new double[,] { { 1, 2 }, { 3, 4 } });
        var result = np.sum(a, axis: 0, keepdims: true);
        result.Should().NotBeNull();
        result.ndim.Should().Be(2);
    }

    [Test]
    public void Sum_WithAxisKeepdimsAndTypeDtype_Compiles()
    {
        var a = np.array(new int[] { 1, 2, 3 });
        var result = np.sum(a, axis: null, keepdims: false, dtype: typeof(double));
        result.Should().NotBeNull();
    }

    [Test]
    public void Sum_WithAxisKeepdimsAndNPTypeCode_Compiles()
    {
        var a = np.array(new int[] { 1, 2, 3 });
        var result = np.sum(a, axis: null, keepdims: false, typeCode: NPTypeCode.Int64);
        result.Should().NotBeNull();
    }

    [Test]
    public void Sum_WithAxisAndTypeDtype_Compiles()
    {
        var a = np.array(new int[] { 1, 2, 3 });
        var result = np.sum(a, axis: null, dtype: typeof(double));
        result.Should().NotBeNull();
    }

    [Test]
    public void Sum_WithAxisAndNPTypeCode_Compiles()
    {
        var a = np.array(new int[] { 1, 2, 3 });
        var result = np.sum(a, axis: null, typeCode: NPTypeCode.Int64);
        result.Should().NotBeNull();
    }

    [Test]
    public void Sum_WithTypeDtype_Compiles()
    {
        var a = np.array(new int[] { 1, 2, 3 });
        var result = np.sum(a, typeof(double));
        result.Should().NotBeNull();
    }

    [Test]
    public void Sum_WithNPTypeCode_Compiles()
    {
        var a = np.array(new int[] { 1, 2, 3 });
        var result = np.sum(a, NPTypeCode.Int64);
        result.Should().NotBeNull();
    }

    #endregion

    #region Prod Reduction

    [Test]
    public void Prod_WithAxisDtypeKeepdims_Compiles()
    {
        var a = np.array(new int[] { 1, 2, 3, 4 });
        var result = np.prod(a, axis: null, dtype: typeof(double), keepdims: false);
        result.Should().NotBeNull();
        // 1 * 2 * 3 * 4 = 24
        result.GetDouble(0).Should().Be(24.0);
    }

    #endregion

    #region Mean Reductions

    [Test]
    public void Mean_NoParams_Compiles()
    {
        var a = np.array(new double[] { 1, 2, 3, 4 });
        var result = np.mean(a);
        result.Should().NotBeNull();
        result.GetDouble(0).Should().Be(2.5);
    }

    [Test]
    public void Mean_WithAxis_Compiles()
    {
        var a = np.array(new double[,] { { 1, 2 }, { 3, 4 } });
        var result = np.mean(a, 0);
        result.Should().NotBeNull();
        result.GetDouble(0).Should().Be(2.0);  // (1+3)/2
        result.GetDouble(1).Should().Be(3.0);  // (2+4)/2
    }

    [Test]
    public void Mean_WithKeepdims_Compiles()
    {
        var a = np.array(new double[] { 1, 2, 3, 4 });
        var result = np.mean(a, keepdims: true);
        result.Should().NotBeNull();
        result.GetDouble(0).Should().Be(2.5);
    }

    [Test]
    public void Mean_WithAxisDtypeKeepdims_Compiles()
    {
        var a = np.array(new double[,] { { 1, 2 }, { 3, 4 } });
        var result = np.mean(a, axis: 0, dtype: typeof(double), keepdims: false);
        result.Should().NotBeNull();
    }

    [Test]
    public void Mean_WithAxisNPTypeCodeKeepdims_Compiles()
    {
        var a = np.array(new double[,] { { 1, 2 }, { 3, 4 } });
        var result = np.mean(a, axis: 0, type: NPTypeCode.Double, keepdims: false);
        result.Should().NotBeNull();
    }

    [Test]
    public void Mean_WithAxisAndKeepdims_Compiles()
    {
        var a = np.array(new double[,] { { 1, 2 }, { 3, 4 } });
        var result = np.mean(a, axis: 0, keepdims: true);
        result.Should().NotBeNull();
        result.ndim.Should().Be(2);
    }

    #endregion

    #region Nan Reductions

    [Test]
    public void NanSum_WithAxisAndKeepdims_Compiles()
    {
        var a = np.array(new double[] { 1, double.NaN, 3, 4 });
        var result = np.nansum(a, axis: null, keepdims: false);
        result.Should().NotBeNull();
        result.GetDouble(0).Should().Be(8.0);  // 1 + 3 + 4 = 8 (NaN ignored)
    }

    [Test]
    public void NanProd_WithAxisAndKeepdims_Compiles()
    {
        var a = np.array(new double[] { 2, double.NaN, 3, 4 });
        var result = np.nanprod(a, axis: null, keepdims: false);
        result.Should().NotBeNull();
        result.GetDouble(0).Should().Be(24.0);  // 2 * 3 * 4 = 24 (NaN ignored)
    }

    [Test]
    public void NanMean_WithAxisAndKeepdims_Compiles()
    {
        var a = np.array(new double[] { 1, double.NaN, 3, 4 });
        var result = np.nanmean(a, axis: null, keepdims: false);
        result.Should().NotBeNull();
        // (1 + 3 + 4) / 3 = 8/3 = 2.666...
        result.GetDouble(0).Should().BeApproximately(8.0 / 3.0, 1e-10);
    }

    [Test]
    public void NanMin_WithAxisAndKeepdims_Compiles()
    {
        var a = np.array(new double[] { 5, double.NaN, 3, 4 });
        var result = np.nanmin(a, axis: null, keepdims: false);
        result.Should().NotBeNull();
        result.GetDouble(0).Should().Be(3.0);  // min ignoring NaN
    }

    [Test]
    public void NanMax_WithAxisAndKeepdims_Compiles()
    {
        var a = np.array(new double[] { 1, double.NaN, 3, 4 });
        var result = np.nanmax(a, axis: null, keepdims: false);
        result.Should().NotBeNull();
        result.GetDouble(0).Should().Be(4.0);  // max ignoring NaN
    }

    #endregion

    #region Std/Var Reductions

    [Test]
    public void Std_WithKeepdimsDdofDtype_Compiles()
    {
        var a = np.array(new double[] { 1, 2, 3, 4, 5 });
        var result = np.std(a, keepdims: false, ddof: null, dtype: null);
        result.Should().NotBeNull();
        // std of [1,2,3,4,5] with ddof=0 is sqrt(2) approximately 1.4142
        result.GetDouble(0).Should().BeApproximately(Math.Sqrt(2.0), 1e-10);
    }

    [Test]
    public void Std_WithAxisTypeDtypeKeepdimsDdof_Compiles()
    {
        var a = np.array(new double[,] { { 1, 2 }, { 3, 4 } });
        var result = np.std(a, axis: 0, dtype: typeof(double), keepdims: false, ddof: null);
        result.Should().NotBeNull();
    }

    [Test]
    public void Std_WithAxisNPTypeCodeKeepdimsDdof_Compiles()
    {
        var a = np.array(new double[,] { { 1, 2 }, { 3, 4 } });
        var result = np.std(a, axis: 0, type: NPTypeCode.Double, keepdims: false, ddof: null);
        result.Should().NotBeNull();
    }

    [Test]
    public void Std_WithAxisKeepdimsDdofDtype_Compiles()
    {
        var a = np.array(new double[,] { { 1, 2 }, { 3, 4 } });
        // Explicitly pass NPTypeCode? to avoid ambiguity with Type overload
        var result = np.std(a, axis: 0, keepdims: false, ddof: null, dtype: (NPTypeCode?)null);
        result.Should().NotBeNull();
    }

    [Test]
    public void Var_WithKeepdimsDdofDtype_Compiles()
    {
        var a = np.array(new double[] { 1, 2, 3, 4, 5 });
        var result = np.var(a, keepdims: false, ddof: null, dtype: null);
        result.Should().NotBeNull();
        // var of [1,2,3,4,5] with ddof=0 is 2.0
        result.GetDouble(0).Should().Be(2.0);
    }

    [Test]
    public void Var_WithAxisTypeDtypeKeepdimsDdof_Compiles()
    {
        var a = np.array(new double[,] { { 1, 2 }, { 3, 4 } });
        var result = np.var(a, axis: 0, dtype: typeof(double), keepdims: false, ddof: null);
        result.Should().NotBeNull();
    }

    [Test]
    public void Var_WithAxisNPTypeCodeKeepdimsDdof_Compiles()
    {
        var a = np.array(new double[,] { { 1, 2 }, { 3, 4 } });
        var result = np.var(a, axis: 0, type: NPTypeCode.Double, keepdims: false, ddof: null);
        result.Should().NotBeNull();
    }

    [Test]
    public void Var_WithAxisKeepdimsDdofDtype_Compiles()
    {
        var a = np.array(new double[,] { { 1, 2 }, { 3, 4 } });
        // Explicitly pass NPTypeCode? to avoid ambiguity with Type overload
        var result = np.var(a, axis: 0, keepdims: false, ddof: null, dtype: (NPTypeCode?)null);
        result.Should().NotBeNull();
    }

    [Test]
    public void NanStd_WithAxisKeepdimsDdof_Compiles()
    {
        var a = np.array(new double[] { 1, double.NaN, 3, 4, 5 });
        var result = np.nanstd(a, axis: null, keepdims: false, ddof: 0);
        result.Should().NotBeNull();
        // std of [1,3,4,5] ignoring NaN
    }

    [Test]
    public void NanVar_WithAxisKeepdimsDdof_Compiles()
    {
        var a = np.array(new double[] { 1, double.NaN, 3, 4, 5 });
        var result = np.nanvar(a, axis: null, keepdims: false, ddof: 0);
        result.Should().NotBeNull();
        // var of [1,3,4,5] ignoring NaN
    }

    #endregion
}
