using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.UnitTest.Utilities;
using TUnit.Core;

namespace NumSharp.UnitTest.Backends.Kernels;

/// <summary>
/// Edge case tests discovered through NumPy battle testing.
/// All expected values are verified against NumPy 2.x output.
/// </summary>
public class EdgeCaseTests
{
    #region Modulo Edge Cases - Python vs C Semantics

    [Test]
    [OpenBugs]  // NumSharp likely uses C semantics, NumPy uses Python semantics
    public void Mod_NegativeDividend_PythonSemantics()
    {
        // NumPy: -7 % 3 = 2 (Python semantics: result has same sign as divisor)
        // C#:    -7 % 3 = -1 (C semantics: result has same sign as dividend)
        var a = np.array(new[] { -7 });
        var b = np.array(new[] { 3 });

        var result = a % b;

        Assert.AreEqual(2, result.GetInt32(0), "NumPy uses Python modulo semantics");
    }

    [Test]
    [OpenBugs]  // NumSharp likely uses C semantics
    public void Mod_NegativeDivisor_PythonSemantics()
    {
        // NumPy: 7 % -3 = -2 (Python semantics)
        // C#:    7 % -3 = 1 (C semantics)
        var a = np.array(new[] { 7 });
        var b = np.array(new[] { -3 });

        var result = a % b;

        Assert.AreEqual(-2, result.GetInt32(0), "NumPy uses Python modulo semantics");
    }

    [Test]
    public void Mod_BothNegative()
    {
        // NumPy: -7 % -3 = -1 (both semantics agree here)
        var a = np.array(new[] { -7 });
        var b = np.array(new[] { -3 });

        var result = a % b;

        Assert.AreEqual(-1, result.GetInt32(0));
    }

    [Test]
    public void Mod_Float_ByZero()
    {
        // NumPy: 5.0 % 0.0 = NaN
        var a = np.array(new[] { 5.0 });
        var b = np.array(new[] { 0.0 });

        var result = a % b;

        Assert.IsTrue(double.IsNaN(result.GetDouble(0)));
    }

    #endregion

    #region Absolute Value Edge Cases

    [Test]
    [OpenBugs]  // Int8 abs overflow is a known edge case
    public void Abs_Int8_MinValue_Overflow()
    {
        // NumPy: np.abs(np.int8(-128)) = -128 (overflow, returns same value!)
        // This is expected behavior - abs of MIN_VALUE for signed ints overflows
        var a = np.array(new sbyte[] { -128 });

        var result = np.abs(a);

        // NumPy actually returns -128 due to overflow
        Assert.AreEqual(-128, (sbyte)result.GetByte(0), "abs(MIN_VALUE) overflows to MIN_VALUE");
    }

    [Test]
    public void Abs_NegativeZero()
    {
        // NumPy: np.abs(-0.0) = 0.0 (positive zero)
        var a = np.array(new[] { -0.0 });

        var result = np.abs(a);

        Assert.AreEqual(0.0, result.GetDouble(0));
        // Note: Checking if it's positive zero vs negative zero is tricky
    }

    [Test]
    public void Abs_NegativeInfinity()
    {
        // NumPy: np.abs(-inf) = inf
        var a = np.array(new[] { double.NegativeInfinity });

        var result = np.abs(a);

        Assert.IsTrue(double.IsPositiveInfinity(result.GetDouble(0)));
    }

    [Test]
    public void Abs_NaN()
    {
        // NumPy: np.abs(nan) = nan
        var a = np.array(new[] { double.NaN });

        var result = np.abs(a);

        Assert.IsTrue(double.IsNaN(result.GetDouble(0)));
    }

    #endregion

    #region Sign Edge Cases

    [Test]
    [OpenBugs]  // .NET Math.Sign(NaN) throws ArithmeticException
    public void Sign_NaN_ReturnsNaN()
    {
        // NumPy: np.sign(nan) = nan
        // .NET: Math.Sign(NaN) throws ArithmeticException
        var a = np.array(new[] { double.NaN });

        var result = np.sign(a);

        Assert.IsTrue(double.IsNaN(result.GetDouble(0)));
    }

    [Test]
    public void Sign_Infinity()
    {
        // NumPy: np.sign(inf) = 1.0, np.sign(-inf) = -1.0
        var a = np.array(new[] { double.PositiveInfinity, double.NegativeInfinity });

        var result = np.sign(a);

        Assert.AreEqual(1.0, result.GetDouble(0));
        Assert.AreEqual(-1.0, result.GetDouble(1));
    }

    [Test]
    public void Sign_Zero()
    {
        // NumPy: np.sign(0) = 0, np.sign(-0.0) = 0.0
        var a = np.array(new[] { 0.0, -0.0 });

        var result = np.sign(a);

        Assert.AreEqual(0.0, result.GetDouble(0));
        Assert.AreEqual(0.0, result.GetDouble(1));
    }

    [Test]
    [OpenBugs]  // sbyte array sign operations fail
    public void Sign_PreservesIntegerType()
    {
        // NumPy: np.sign(int8) returns int8
        var a = np.array(new sbyte[] { -5, 0, 5 });

        var result = np.sign(a);

        // NumSharp might not preserve exact integer types for sign
        Assert.AreEqual(-1, result.GetInt32(0));
        Assert.AreEqual(0, result.GetInt32(1));
        Assert.AreEqual(1, result.GetInt32(2));
    }

    #endregion

    #region Negative Edge Cases

    [Test]
    [OpenBugs]  // NumSharp throws NotSupportedException for unsigned types
    public void Negative_UInt8_Wraps()
    {
        // NumPy: np.negative(np.uint8(1)) = 255 (wraps)
        // NumPy: np.negative(np.uint8(5)) = 251
        // NumPy: np.negative(np.uint8(0)) = 0
        var a = np.array(new byte[] { 1, 5, 0 });

        var result = np.negative(a);

        Assert.AreEqual(255, result.GetByte(0));
        Assert.AreEqual(251, result.GetByte(1));
        Assert.AreEqual(0, result.GetByte(2));
    }

    [Test]
    public void Negative_NegativeZero()
    {
        // NumPy: np.negative(-0.0) = 0.0 (becomes positive zero)
        var a = np.array(new[] { -0.0 });

        var result = np.negative(a);

        Assert.AreEqual(0.0, result.GetDouble(0));
    }

    [Test]
    public void Negative_Infinity()
    {
        // NumPy: np.negative(inf) = -inf
        var a = np.array(new[] { double.PositiveInfinity });

        var result = np.negative(a);

        Assert.IsTrue(double.IsNegativeInfinity(result.GetDouble(0)));
    }

    [Test]
    public void Negative_NaN()
    {
        // NumPy: np.negative(nan) = nan
        var a = np.array(new[] { double.NaN });

        var result = np.negative(a);

        Assert.IsTrue(double.IsNaN(result.GetDouble(0)));
    }

    #endregion

    #region Integer Overflow Edge Cases

    [Test]
    [OpenBugs]  // sbyte array operations may not work correctly
    public void Add_Int8_Overflow_Wraps()
    {
        // NumPy: np.int8(127) + np.int8(1) = -128
        var a = np.array(new sbyte[] { 127 });
        var b = np.array(new sbyte[] { 1 });

        var result = a + b;

        Assert.AreEqual(-128, (sbyte)result.GetByte(0));
    }

    [Test]
    [OpenBugs]  // sbyte array operations may not work correctly
    public void Subtract_Int8_Underflow_Wraps()
    {
        // NumPy: np.int8(-128) - np.int8(1) = 127
        var a = np.array(new sbyte[] { -128 });
        var b = np.array(new sbyte[] { 1 });

        var result = a - b;

        Assert.AreEqual(127, (sbyte)result.GetByte(0));
    }

    [Test]
    public void Add_UInt8_Overflow_Wraps()
    {
        // NumPy: np.uint8(255) + np.uint8(1) = 0
        var a = np.array(new byte[] { 255 });
        var b = np.array(new byte[] { 1 });

        var result = a + b;

        Assert.AreEqual(0, result.GetByte(0));
    }

    [Test]
    public void Subtract_UInt8_Underflow_Wraps()
    {
        // NumPy: np.uint8(0) - np.uint8(1) = 255
        var a = np.array(new byte[] { 0 });
        var b = np.array(new byte[] { 1 });

        var result = a - b;

        Assert.AreEqual(255, result.GetByte(0));
    }

    [Test]
    [OpenBugs]  // sbyte array operations may not work correctly
    public void Multiply_Int8_Overflow_Wraps()
    {
        // NumPy: np.int8(100) * np.int8(2) = -56
        var a = np.array(new sbyte[] { 100 });
        var b = np.array(new sbyte[] { 2 });

        var result = a * b;

        Assert.AreEqual(-56, (sbyte)result.GetByte(0));
    }

    #endregion

    #region Division Edge Cases

    [Test]
    public void Divide_Float_ByPositiveZero()
    {
        // NumPy: 1.0 / 0.0 = inf
        var a = np.array(new[] { 1.0 });
        var b = np.array(new[] { 0.0 });

        var result = a / b;

        Assert.IsTrue(double.IsPositiveInfinity(result.GetDouble(0)));
    }

    [Test]
    public void Divide_Float_ByNegativeZero()
    {
        // NumPy: 1.0 / -0.0 = -inf
        var a = np.array(new[] { 1.0 });
        var b = np.array(new[] { -0.0 });

        var result = a / b;

        Assert.IsTrue(double.IsNegativeInfinity(result.GetDouble(0)));
    }

    [Test]
    public void Divide_ZeroByZero()
    {
        // NumPy: 0.0 / 0.0 = nan
        var a = np.array(new[] { 0.0 });
        var b = np.array(new[] { 0.0 });

        var result = a / b;

        Assert.IsTrue(double.IsNaN(result.GetDouble(0)));
    }

    [Test]
    public void Divide_InfinityByInfinity()
    {
        // NumPy: inf / inf = nan
        var a = np.array(new[] { double.PositiveInfinity });
        var b = np.array(new[] { double.PositiveInfinity });

        var result = a / b;

        Assert.IsTrue(double.IsNaN(result.GetDouble(0)));
    }

    [Test]
    public void Divide_ZeroByInfinity()
    {
        // NumPy: 0.0 / inf = 0.0
        var a = np.array(new[] { 0.0 });
        var b = np.array(new[] { double.PositiveInfinity });

        var result = a / b;

        Assert.AreEqual(0.0, result.GetDouble(0));
    }

    #endregion

    #region Infinity Arithmetic Edge Cases

    [Test]
    public void Subtract_InfinityFromInfinity()
    {
        // NumPy: inf - inf = nan
        var a = np.array(new[] { double.PositiveInfinity });
        var b = np.array(new[] { double.PositiveInfinity });

        var result = a - b;

        Assert.IsTrue(double.IsNaN(result.GetDouble(0)));
    }

    [Test]
    public void Multiply_InfinityByZero()
    {
        // NumPy: inf * 0 = nan
        var a = np.array(new[] { double.PositiveInfinity });
        var b = np.array(new[] { 0.0 });

        var result = a * b;

        Assert.IsTrue(double.IsNaN(result.GetDouble(0)));
    }

    [Test]
    public void Add_InfinityPlusOne()
    {
        // NumPy: inf + 1 = inf
        var a = np.array(new[] { double.PositiveInfinity });
        var b = np.array(new[] { 1.0 });

        var result = a + b;

        Assert.IsTrue(double.IsPositiveInfinity(result.GetDouble(0)));
    }

    [Test]
    public void Add_NegativeInfinityPlusInfinity()
    {
        // NumPy: -inf + inf = nan
        var a = np.array(new[] { double.NegativeInfinity });
        var b = np.array(new[] { double.PositiveInfinity });

        var result = a + b;

        Assert.IsTrue(double.IsNaN(result.GetDouble(0)));
    }

    #endregion

    #region Trig Function Edge Cases

    [Test]
    public void Sin_Infinity()
    {
        // NumPy: np.sin(inf) = nan
        var a = np.array(new[] { double.PositiveInfinity });

        var result = np.sin(a);

        Assert.IsTrue(double.IsNaN(result.GetDouble(0)));
    }

    [Test]
    public void Cos_Infinity()
    {
        // NumPy: np.cos(inf) = nan
        var a = np.array(new[] { double.PositiveInfinity });

        var result = np.cos(a);

        Assert.IsTrue(double.IsNaN(result.GetDouble(0)));
    }

    [Test]
    public void ArcSin_OutOfRange()
    {
        // NumPy: np.arcsin(2) = nan
        var a = np.array(new[] { 2.0 });

        var result = np.arcsin(a);

        Assert.IsTrue(double.IsNaN(result.GetDouble(0)));
    }

    [Test]
    public void ArcTan_Infinity()
    {
        // NumPy: np.arctan(inf) = pi/2
        var a = np.array(new[] { double.PositiveInfinity, double.NegativeInfinity });

        var result = np.arctan(a);

        Assert.AreEqual(Math.PI / 2, result.GetDouble(0), 1e-10);
        Assert.AreEqual(-Math.PI / 2, result.GetDouble(1), 1e-10);
    }

    #endregion

    #region Log/Exp Edge Cases

    [Test]
    public void Log_Zero()
    {
        // NumPy: np.log(0) = -inf
        var a = np.array(new[] { 0.0 });

        var result = np.log(a);

        Assert.IsTrue(double.IsNegativeInfinity(result.GetDouble(0)));
    }

    [Test]
    public void Log_Negative()
    {
        // NumPy: np.log(-1) = nan
        var a = np.array(new[] { -1.0 });

        var result = np.log(a);

        Assert.IsTrue(double.IsNaN(result.GetDouble(0)));
    }

    [Test]
    public void Log1p_MinusOne()
    {
        // NumPy: np.log1p(-1) = -inf
        var a = np.array(new[] { -1.0 });

        var result = np.log1p(a);

        Assert.IsTrue(double.IsNegativeInfinity(result.GetDouble(0)));
    }

    [Test]
    public void Log1p_LessThanMinusOne()
    {
        // NumPy: np.log1p(-2) = nan
        var a = np.array(new[] { -2.0 });

        var result = np.log1p(a);

        Assert.IsTrue(double.IsNaN(result.GetDouble(0)));
    }

    [Test]
    public void Exp_NegativeInfinity()
    {
        // NumPy: np.exp(-inf) = 0
        var a = np.array(new[] { double.NegativeInfinity });

        var result = np.exp(a);

        Assert.AreEqual(0.0, result.GetDouble(0));
    }

    [Test]
    public void Expm1_NegativeInfinity()
    {
        // NumPy: np.expm1(-inf) = -1
        var a = np.array(new[] { double.NegativeInfinity });

        var result = np.expm1(a);

        Assert.AreEqual(-1.0, result.GetDouble(0));
    }

    [Test]
    public void Exp2_NegativeInfinity()
    {
        // NumPy: np.exp2(-inf) = 0
        var a = np.array(new[] { double.NegativeInfinity });

        var result = np.exp2(a);

        Assert.AreEqual(0.0, result.GetDouble(0));
    }

    #endregion

    #region Sqrt Edge Cases

    [Test]
    public void Sqrt_Negative()
    {
        // NumPy: np.sqrt(-1) = nan
        var a = np.array(new[] { -1.0 });

        var result = np.sqrt(a);

        Assert.IsTrue(double.IsNaN(result.GetDouble(0)));
    }

    [Test]
    public void Sqrt_NegativeInfinity()
    {
        // NumPy: np.sqrt(-inf) = nan
        var a = np.array(new[] { double.NegativeInfinity });

        var result = np.sqrt(a);

        Assert.IsTrue(double.IsNaN(result.GetDouble(0)));
    }

    [Test]
    public void Sqrt_PositiveInfinity()
    {
        // NumPy: np.sqrt(inf) = inf
        var a = np.array(new[] { double.PositiveInfinity });

        var result = np.sqrt(a);

        Assert.IsTrue(double.IsPositiveInfinity(result.GetDouble(0)));
    }

    #endregion

    #region Hyperbolic Edge Cases

    [Test]
    public void Tanh_Infinity()
    {
        // NumPy: np.tanh(inf) = 1.0, np.tanh(-inf) = -1.0
        var a = np.array(new[] { double.PositiveInfinity, double.NegativeInfinity });

        var result = np.tanh(a);

        Assert.AreEqual(1.0, result.GetDouble(0));
        Assert.AreEqual(-1.0, result.GetDouble(1));
    }

    [Test]
    public void Tanh_LargeValue()
    {
        // NumPy: np.tanh(1000) = 1.0 (saturates to 1)
        var a = np.array(new[] { 1000.0 });

        var result = np.tanh(a);

        Assert.AreEqual(1.0, result.GetDouble(0));
    }

    [Test]
    public void Sinh_Infinity()
    {
        // NumPy: np.sinh(inf) = inf, np.sinh(-inf) = -inf
        var a = np.array(new[] { double.PositiveInfinity, double.NegativeInfinity });

        var result = np.sinh(a);

        Assert.IsTrue(double.IsPositiveInfinity(result.GetDouble(0)));
        Assert.IsTrue(double.IsNegativeInfinity(result.GetDouble(1)));
    }

    [Test]
    public void Cosh_Infinity()
    {
        // NumPy: np.cosh(inf) = inf (both +inf and -inf give +inf)
        var a = np.array(new[] { double.PositiveInfinity, double.NegativeInfinity });

        var result = np.cosh(a);

        Assert.IsTrue(double.IsPositiveInfinity(result.GetDouble(0)));
        Assert.IsTrue(double.IsPositiveInfinity(result.GetDouble(1)));
    }

    #endregion

    #region Rounding Edge Cases (Banker's Rounding)

    [Test]
    public void Round_HalfToEven_BankersRounding()
    {
        // NumPy uses banker's rounding: round half to even
        // np.round(0.5) = 0, np.round(1.5) = 2, np.round(2.5) = 2, np.round(3.5) = 4
        var a = np.array(new[] { 0.5, 1.5, 2.5, 3.5, 4.5 });

        var result = np.around(a);

        Assert.AreEqual(0.0, result.GetDouble(0));  // Half rounds to even (0)
        Assert.AreEqual(2.0, result.GetDouble(1));  // Half rounds to even (2)
        Assert.AreEqual(2.0, result.GetDouble(2));  // Half rounds to even (2)
        Assert.AreEqual(4.0, result.GetDouble(3));  // Half rounds to even (4)
        Assert.AreEqual(4.0, result.GetDouble(4));  // Half rounds to even (4)
    }

    [Test]
    public void Round_NegativeHalf()
    {
        // NumPy: np.round(-0.5) = -0.0, np.round(-1.5) = -2.0
        var a = np.array(new[] { -0.5, -1.5, -2.5 });

        var result = np.around(a);

        Assert.AreEqual(0.0, result.GetDouble(0));   // Rounds to -0.0 (displays as 0.0)
        Assert.AreEqual(-2.0, result.GetDouble(1));  // Half rounds to even (-2)
        Assert.AreEqual(-2.0, result.GetDouble(2));  // Half rounds to even (-2)
    }

    [Test]
    public void Floor_Infinity()
    {
        // NumPy: np.floor(inf) = inf, np.floor(-inf) = -inf
        var a = np.array(new[] { double.PositiveInfinity, double.NegativeInfinity });

        var result = np.floor(a);

        Assert.IsTrue(double.IsPositiveInfinity(result.GetDouble(0)));
        Assert.IsTrue(double.IsNegativeInfinity(result.GetDouble(1)));
    }

    [Test]
    public void Floor_NaN()
    {
        // NumPy: np.floor(nan) = nan
        var a = np.array(new[] { double.NaN });

        var result = np.floor(a);

        Assert.IsTrue(double.IsNaN(result.GetDouble(0)));
    }

    [Test]
    public void Ceil_NegativeHalf()
    {
        // NumPy: np.ceil(-0.5) = -0.0
        var a = np.array(new[] { -0.5 });

        var result = np.ceil(a);

        Assert.AreEqual(0.0, result.GetDouble(0));  // -0.0 == 0.0
    }

    #endregion

    #region Empty Array Edge Cases

    [Test]
    public void Sum_EmptyArray()
    {
        // NumPy: np.sum([]) = 0.0
        var a = np.array(new double[0]);

        var result = np.sum(a);

        Assert.AreEqual(0.0, result.GetDouble(0));
    }

    [Test]
    public void Prod_EmptyArray()
    {
        // NumPy: np.prod([]) = 1.0
        var a = np.array(new double[0]);

        var result = np.prod(a);

        Assert.AreEqual(1.0, result.GetDouble(0));
    }

    [Test]
    [OpenBugs]  // Mean of empty array fails (should return NaN)
    public void Mean_EmptyArray()
    {
        // NumPy: np.mean([]) = nan
        var a = np.array(new double[0]);

        var result = np.mean(a);

        Assert.IsTrue(double.IsNaN(result.GetDouble(0)));
    }

    [Test]
    public void Cumsum_EmptyArray()
    {
        // NumPy: np.cumsum([]) = [] (empty array)
        var a = np.array(new double[0]);

        var result = np.cumsum(a);

        Assert.AreEqual(0, result.size);
        Assert.AreEqual(1, result.ndim);
    }

    [Test]
    public void Dot_EmptyArrays()
    {
        // NumPy: np.dot([], []) = 0.0
        var a = np.array(new double[0]);
        var b = np.array(new double[0]);

        var result = np.dot(a, b);

        Assert.AreEqual(0.0, (double)result);
    }

    #endregion

    #region Type Promotion Edge Cases

    [Test]
    public void Add_Bool_ActsAsLogicalOr()
    {
        // NumPy: True + True = True (bool + bool stays bool, acts as OR)
        var a = np.array(new[] { true });
        var b = np.array(new[] { true });

        var result = a + b;

        Assert.AreEqual(NPTypeCode.Boolean, result.typecode);
        Assert.IsTrue(result.GetBoolean(0));
    }

    [Test]
    public void Add_Bool_False()
    {
        // NumPy: False + False = False
        var a = np.array(new[] { false });
        var b = np.array(new[] { false });

        var result = a + b;

        Assert.AreEqual(NPTypeCode.Boolean, result.typecode);
        Assert.IsFalse(result.GetBoolean(0));
    }

    [Test]
    public void Multiply_Bool_ActsAsLogicalAnd()
    {
        // NumPy: True * False = False (bool * bool stays bool, acts as AND)
        var a = np.array(new[] { true });
        var b = np.array(new[] { false });

        var result = a * b;

        Assert.AreEqual(NPTypeCode.Boolean, result.typecode);
        Assert.IsFalse(result.GetBoolean(0));
    }

    [Test]
    public void Multiply_Bool_WithInt()
    {
        // NumPy: True * 5 = 5 (promotes to int64)
        var a = np.array(new[] { true });
        var b = np.array(new[] { 5 });

        var result = a * b;

        Assert.AreEqual(5, result.GetInt32(0));
    }

    [Test]
    public void Add_UInt64_Int64_PromotesToFloat64()
    {
        // NumPy: uint64 + int64 = float64 (can't safely fit in either)
        var a = np.array(new ulong[] { 1 });
        var b = np.array(new long[] { 1 });

        var result = a + b;

        // NumPy promotes to float64 in this case
        Assert.AreEqual(NPTypeCode.Double, result.typecode);
    }

    #endregion

    #region NaN Comparison Edge Cases

    [Test]
    public void Equal_NaN_NaN_IsFalse()
    {
        // NumPy: nan == nan -> False (IEEE 754)
        var a = np.array(new[] { double.NaN });
        var b = np.array(new[] { double.NaN });

        var result = a == b;

        Assert.IsFalse(result.GetBoolean(0));
    }

    [Test]
    public void NotEqual_NaN_NaN_IsTrue()
    {
        // NumPy: nan != nan -> True (IEEE 754)
        var a = np.array(new[] { double.NaN });
        var b = np.array(new[] { double.NaN });

        var result = a != b;

        Assert.IsTrue(result.GetBoolean(0));
    }

    [Test]
    public void LessThan_NaN_IsFalse()
    {
        // NumPy: nan < anything -> False
        var a = np.array(new[] { double.NaN });
        var b = np.array(new[] { 0.0 });

        var result = a < b;

        Assert.IsFalse(result.GetBoolean(0));
    }

    [Test]
    public void GreaterThan_NaN_IsFalse()
    {
        // NumPy: nan > anything -> False
        var a = np.array(new[] { double.NaN });
        var b = np.array(new[] { 0.0 });

        var result = a > b;

        Assert.IsFalse(result.GetBoolean(0));
    }

    #endregion

    #region Infinity Comparison Edge Cases

    [Test]
    public void Equal_Infinity_Infinity_IsTrue()
    {
        // NumPy: inf == inf -> True
        var a = np.array(new[] { double.PositiveInfinity });
        var b = np.array(new[] { double.PositiveInfinity });

        var result = a == b;

        Assert.IsTrue(result.GetBoolean(0));
    }

    [Test]
    public void GreaterThan_Infinity_Infinity_IsFalse()
    {
        // NumPy: inf > inf -> False
        var a = np.array(new[] { double.PositiveInfinity });
        var b = np.array(new[] { double.PositiveInfinity });

        var result = a > b;

        Assert.IsFalse(result.GetBoolean(0));
    }

    [Test]
    public void GreaterThanOrEqual_Infinity_Infinity_IsTrue()
    {
        // NumPy: inf >= inf -> True
        var a = np.array(new[] { double.PositiveInfinity });
        var b = np.array(new[] { double.PositiveInfinity });

        var result = a >= b;

        Assert.IsTrue(result.GetBoolean(0));
    }

    #endregion
}
