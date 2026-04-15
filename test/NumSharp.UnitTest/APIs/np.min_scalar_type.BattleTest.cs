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

    [TestMethod]
    public void MinScalarType_MinusOne()
    {
        np.min_scalar_type(-1).Should().Be(NPTypeCode.Int16);
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

    #endregion

    #region Float Values

    [TestMethod]
    public void MinScalarType_FloatValue()
    {
        np.min_scalar_type(1.0f).Should().Be(NPTypeCode.Single);
    }

    [TestMethod]
    public void MinScalarType_DoubleLarge()
    {
        np.min_scalar_type(1e100).Should().Be(NPTypeCode.Double);
    }

    [TestMethod]
    public void MinScalarType_FloatNaN()
    {
        np.min_scalar_type(float.NaN).Should().Be(NPTypeCode.Single);
    }

    [TestMethod]
    public void MinScalarType_FloatInfinity()
    {
        np.min_scalar_type(float.PositiveInfinity).Should().Be(NPTypeCode.Single);
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
