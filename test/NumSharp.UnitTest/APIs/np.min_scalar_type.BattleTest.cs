using System;
using System.Threading.Tasks;

namespace NumSharp.UnitTest.APIs;

/// <summary>
/// Battle tests for np.min_scalar_type - comprehensive coverage of scalar type inference.
/// </summary>
public class NpMinScalarTypeBattleTests
{
    #region Unsigned Integer Boundaries

    [TestMethod]
    public async Task MinScalarType_Zero()
    {
        await Assert.That(np.min_scalar_type(0)).IsEqualTo(NPTypeCode.Byte);
    }

    [TestMethod]
    public async Task MinScalarType_ByteMax()
    {
        await Assert.That(np.min_scalar_type(255)).IsEqualTo(NPTypeCode.Byte);
    }

    [TestMethod]
    public async Task MinScalarType_ByteMaxPlus1()
    {
        await Assert.That(np.min_scalar_type(256)).IsEqualTo(NPTypeCode.UInt16);
    }

    [TestMethod]
    public async Task MinScalarType_UInt16Max()
    {
        await Assert.That(np.min_scalar_type(65535)).IsEqualTo(NPTypeCode.UInt16);
    }

    [TestMethod]
    public async Task MinScalarType_UInt16MaxPlus1()
    {
        await Assert.That(np.min_scalar_type(65536)).IsEqualTo(NPTypeCode.UInt32);
    }

    [TestMethod]
    public async Task MinScalarType_UInt32Max()
    {
        await Assert.That(np.min_scalar_type(uint.MaxValue)).IsEqualTo(NPTypeCode.UInt32);
    }

    #endregion

    #region Signed Integer Boundaries

    [TestMethod]
    public async Task MinScalarType_MinusOne()
    {
        await Assert.That(np.min_scalar_type(-1)).IsEqualTo(NPTypeCode.Int16);
    }

    [TestMethod]
    public async Task MinScalarType_Int16Min()
    {
        await Assert.That(np.min_scalar_type(short.MinValue)).IsEqualTo(NPTypeCode.Int16);
    }

    [TestMethod]
    public async Task MinScalarType_Int16MinMinus1()
    {
        await Assert.That(np.min_scalar_type((int)short.MinValue - 1)).IsEqualTo(NPTypeCode.Int32);
    }

    [TestMethod]
    public async Task MinScalarType_Int32Min()
    {
        await Assert.That(np.min_scalar_type(int.MinValue)).IsEqualTo(NPTypeCode.Int32);
    }

    #endregion

    #region Float Values

    [TestMethod]
    public async Task MinScalarType_FloatValue()
    {
        await Assert.That(np.min_scalar_type(1.0f)).IsEqualTo(NPTypeCode.Single);
    }

    [TestMethod]
    public async Task MinScalarType_DoubleLarge()
    {
        await Assert.That(np.min_scalar_type(1e100)).IsEqualTo(NPTypeCode.Double);
    }

    [TestMethod]
    public async Task MinScalarType_FloatNaN()
    {
        await Assert.That(np.min_scalar_type(float.NaN)).IsEqualTo(NPTypeCode.Single);
    }

    [TestMethod]
    public async Task MinScalarType_FloatInfinity()
    {
        await Assert.That(np.min_scalar_type(float.PositiveInfinity)).IsEqualTo(NPTypeCode.Single);
    }

    #endregion

    #region Boolean

    [TestMethod]
    public async Task MinScalarType_True()
    {
        await Assert.That(np.min_scalar_type(true)).IsEqualTo(NPTypeCode.Boolean);
    }

    [TestMethod]
    public async Task MinScalarType_False()
    {
        await Assert.That(np.min_scalar_type(false)).IsEqualTo(NPTypeCode.Boolean);
    }

    #endregion

    #region Decimal

    [TestMethod]
    public async Task MinScalarType_Decimal()
    {
        await Assert.That(np.min_scalar_type(1.0m)).IsEqualTo(NPTypeCode.Decimal);
    }

    #endregion

    #region Error Cases

    [TestMethod]
    public async Task MinScalarType_Null_Throws()
    {
        await Assert.That(() => np.min_scalar_type(null!)).ThrowsException();
    }

    #endregion
}
