using System;

namespace NumSharp.Tests.APIs;

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

    #region Arrays

    /// <summary>
    ///     NumPy's <c>PyArray_MinScalarType</c>: an array of one or more dimensions answers its own dtype, whatever it
    ///     holds (<c>np.min_scalar_type(np.array([1, 2], 'f8'))</c> is float64, not float16). It used to fall through to
    ///     <c>value.GetType().GetTypeCode()</c> and answer null for every array.
    /// </summary>
    [TestMethod]
    public void MinScalarType_Array_KeepsItsDtype()
    {
        foreach (var dt in new DType[] { np.bool_, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64,
                     np.uint64, np.float16, np.float32, np.float64, np.complex128, DType.Decimal, DType.Char })
        {
            var a = np.zeros(new Shape(3), dt);
            np.min_scalar_type(a).Should().BeSameAs(dt, dt.name);
            np.min_scalar_type(np.zeros(new Shape(2, 0), dt)).Should().BeSameAs(dt, "an empty array is no scalar either");
        }
    }

    /// <summary>
    ///     A 0-d array is a scalar to NumPy: its ONE value decides, by the rules of its own dtype (probed against 2.4.2:
    ///     <c>np.array(1000, 'i8')</c> → uint16, <c>np.array(-5, 'i2')</c> → int8, <c>np.array(1e10, 'f8')</c> → float32,
    ///     <c>np.array(1.0, 'f4')</c> → float16, <c>np.array(True)</c> → bool). A 0-d VIEW of a larger array reads its own
    ///     element.
    /// </summary>
    [TestMethod]
    public void MinScalarType_ZeroDimensionalArray_IsDemotedByItsValue()
    {
        np.min_scalar_type(np.array(1000L)).Should().Be(NPTypeCode.UInt16);
        np.min_scalar_type(np.array((short)-5)).Should().Be(NPTypeCode.SByte);
        np.min_scalar_type(np.array(1e10)).Should().Be(NPTypeCode.Single);
        np.min_scalar_type(np.array(1.0f)).Should().Be(NPTypeCode.Half);
        np.min_scalar_type(np.array(true)).Should().Be(NPTypeCode.Boolean);
        np.min_scalar_type(np.array(70000.0)).Should().Be(NPTypeCode.Single);
        var row = np.array(new long[] { 1, -300, 70000 });
        np.min_scalar_type(row[1]).Should().Be(NPTypeCode.Int16, "row[1] is a 0-d view of -300");
        np.min_scalar_type(row[2]).Should().Be(NPTypeCode.UInt32, "row[2] is a 0-d view of 70000");
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
