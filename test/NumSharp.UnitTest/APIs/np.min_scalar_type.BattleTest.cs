using System;

namespace NumSharp.UnitTest.APIs;

/// <summary>
/// Battle tests for np.min_scalar_type - comprehensive coverage of scalar type inference.
/// </summary>
[TestClass]
public class NpMinScalarTypeBattleTests
{
    #region Unsigned Integer Boundaries

    [TestMethod]
    public void MinScalarType_Zero()
    {
        np.min_scalar_type(0).Should().Be(NPTypeCode.Byte);
    }

    [TestMethod]
    public void MinScalarType_ByteMax()
    {
        np.min_scalar_type(255).Should().Be(NPTypeCode.Byte);
    }

    [TestMethod]
    public void MinScalarType_ByteMaxPlus1()
    {
        np.min_scalar_type(256).Should().Be(NPTypeCode.UInt16);
    }

    [TestMethod]
    public void MinScalarType_UInt16Max()
    {
        np.min_scalar_type(65535).Should().Be(NPTypeCode.UInt16);
    }

    [TestMethod]
    public void MinScalarType_UInt16MaxPlus1()
    {
        np.min_scalar_type(65536).Should().Be(NPTypeCode.UInt32);
    }

    [TestMethod]
    public void MinScalarType_UInt32Max()
    {
        np.min_scalar_type(uint.MaxValue).Should().Be(NPTypeCode.UInt32);
    }

    #endregion

    #region Signed Integer Boundaries

    // NumPy 2.4.2: min_scalar_type demotes negatives down to int8 — NumSharp HAS int8 (SByte).
    //   np.min_scalar_type(-1)   -> int8
    //   np.min_scalar_type(-128) -> int8
    //   np.min_scalar_type(-129) -> int16
    [TestMethod]
    public void MinScalarType_MinusOne()
    {
        np.min_scalar_type(-1).Should().Be(NPTypeCode.SByte);
    }

    [TestMethod]
    public void MinScalarType_Int8Min()
    {
        np.min_scalar_type(-128).Should().Be(NPTypeCode.SByte);
    }

    [TestMethod]
    public void MinScalarType_Int8MinMinus1()
    {
        np.min_scalar_type(-129).Should().Be(NPTypeCode.Int16);
    }

    [TestMethod]
    public void MinScalarType_NegativeSByteInput()
    {
        // sbyte -5 must resolve to int8, not int16.
        np.min_scalar_type((sbyte)-5).Should().Be(NPTypeCode.SByte);
        // non-negative signed demotes to the smallest unsigned (uint8).
        np.min_scalar_type((sbyte)5).Should().Be(NPTypeCode.Byte);
    }

    [TestMethod]
    public void MinScalarType_Int16Min()
    {
        np.min_scalar_type(short.MinValue).Should().Be(NPTypeCode.Int16);
    }

    [TestMethod]
    public void MinScalarType_Int16MinMinus1()
    {
        np.min_scalar_type((int)short.MinValue - 1).Should().Be(NPTypeCode.Int32);
    }

    [TestMethod]
    public void MinScalarType_Int32Min()
    {
        np.min_scalar_type(int.MinValue).Should().Be(NPTypeCode.Int32);
    }

    [TestMethod]
    public void MinScalarType_NegativeLong_DemotesToInt8()
    {
        // Value-based, not type-based: int64 -10 still demotes to int8 in NumPy.
        np.min_scalar_type(-10L).Should().Be(NPTypeCode.SByte);
    }

    #endregion

    #region Float Values

    // NumPy 2.4.2 min_scalar_type demotes floats by RANGE (magnitude), not exact
    // representability — NumSharp HAS float16 (Half):
    //   |v| < 65000 (or non-finite)  -> float16
    //   |v| < 3.4e38                 -> float32
    //   else                         -> float64
    [TestMethod]
    public void MinScalarType_FloatValue()
    {
        np.min_scalar_type(1.0f).Should().Be(NPTypeCode.Half);
    }

    [TestMethod]
    public void MinScalarType_DoubleSmall_IsHalf()
    {
        // Even a value that would lose precision (0.1) or underflow float16 (1e-40)
        // reports float16 — NumPy demotes purely by magnitude.
        np.min_scalar_type(1.0).Should().Be(NPTypeCode.Half);
        np.min_scalar_type(0.1).Should().Be(NPTypeCode.Half);
        np.min_scalar_type(1e-40).Should().Be(NPTypeCode.Half);
    }

    [TestMethod]
    public void MinScalarType_Float16RangeBoundary()
    {
        // Bounds are EXCLUSIVE at 65000.
        np.min_scalar_type(64999.0).Should().Be(NPTypeCode.Half);
        np.min_scalar_type(65000.0).Should().Be(NPTypeCode.Single);
        np.min_scalar_type(-65000.0).Should().Be(NPTypeCode.Single);
        np.min_scalar_type(64999.0f).Should().Be(NPTypeCode.Half);
        np.min_scalar_type(65000.0f).Should().Be(NPTypeCode.Single);
    }

    [TestMethod]
    public void MinScalarType_Float32RangeBoundary()
    {
        // 70000 exceeds float16 range -> float32; 3.4e38 is the exclusive float32 cutoff.
        np.min_scalar_type(70000.0).Should().Be(NPTypeCode.Single);
        np.min_scalar_type(3.39e38).Should().Be(NPTypeCode.Single);
        np.min_scalar_type(3.4e38).Should().Be(NPTypeCode.Double);
    }

    [TestMethod]
    public void MinScalarType_DoubleLarge()
    {
        np.min_scalar_type(1e100).Should().Be(NPTypeCode.Double);
    }

    [TestMethod]
    public void MinScalarType_FloatNaN()
    {
        // Non-finite values demote to float16 in NumPy.
        np.min_scalar_type(float.NaN).Should().Be(NPTypeCode.Half);
        np.min_scalar_type(double.NaN).Should().Be(NPTypeCode.Half);
    }

    [TestMethod]
    public void MinScalarType_FloatInfinity()
    {
        np.min_scalar_type(float.PositiveInfinity).Should().Be(NPTypeCode.Half);
        np.min_scalar_type(double.NegativeInfinity).Should().Be(NPTypeCode.Half);
    }

    [TestMethod]
    public void MinScalarType_HalfInput_StaysHalf()
    {
        // NumPy NPY_HALF: a float16 scalar always reports float16.
        np.min_scalar_type((Half)70000f).Should().Be(NPTypeCode.Half);
        np.min_scalar_type((Half)1f).Should().Be(NPTypeCode.Half);
    }

    #endregion

    #region Boolean

    [TestMethod]
    public void MinScalarType_True()
    {
        np.min_scalar_type(true).Should().Be(NPTypeCode.Boolean);
    }

    [TestMethod]
    public void MinScalarType_False()
    {
        np.min_scalar_type(false).Should().Be(NPTypeCode.Boolean);
    }

    #endregion

    #region Decimal

    [TestMethod]
    public void MinScalarType_Decimal()
    {
        np.min_scalar_type(1.0m).Should().Be(NPTypeCode.Decimal);
    }

    #endregion

    #region Error Cases

    [TestMethod]
    public void MinScalarType_Null_Throws()
    {
        new Action(() => np.min_scalar_type(null!)).Should().Throw<Exception>();
    }

    #endregion
}
