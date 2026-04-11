using System;
using System.Threading.Tasks;
using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace NumSharp.UnitTest.APIs;

/// <summary>
/// Battle tests for np.common_type - comprehensive coverage.
/// </summary>
public class NpCommonTypeBattleTests
{
    #region Integer Arrays - Always Return Double

    [Test]
    public async Task CommonType_Int32Array_ReturnsDouble()
    {
        var arr = np.array(new int[] { 1, 2, 3 });
        await Assert.That(np.common_type_code(arr)).IsEqualTo(NPTypeCode.Double);
    }

    [Test]
    public async Task CommonType_ByteArray_ReturnsDouble()
    {
        var arr = np.array(new byte[] { 1, 2, 3 });
        await Assert.That(np.common_type_code(arr)).IsEqualTo(NPTypeCode.Double);
    }

    [Test]
    public async Task CommonType_BoolArray_ReturnsDouble()
    {
        var arr = np.array(new bool[] { true, false });
        await Assert.That(np.common_type_code(arr)).IsEqualTo(NPTypeCode.Double);
    }

    #endregion

    #region Float Arrays

    [Test]
    public async Task CommonType_Float32Array_ReturnsSingle()
    {
        var arr = np.array(new float[] { 1.0f, 2.0f });
        await Assert.That(np.common_type_code(arr)).IsEqualTo(NPTypeCode.Single);
    }

    [Test]
    public async Task CommonType_Float64Array_ReturnsDouble()
    {
        var arr = np.array(new double[] { 1.0, 2.0 });
        await Assert.That(np.common_type_code(arr)).IsEqualTo(NPTypeCode.Double);
    }

    #endregion

    #region Multiple Arrays

    [Test]
    public async Task CommonType_Float32AndFloat64_ReturnsDouble()
    {
        var a = np.array(new float[] { 1.0f });
        var b = np.array(new double[] { 1.0 });
        await Assert.That(np.common_type_code(a, b)).IsEqualTo(NPTypeCode.Double);
    }

    [Test]
    public async Task CommonType_AllFloat32_ReturnsSingle()
    {
        var a = np.array(new float[] { 1.0f });
        var b = np.array(new float[] { 2.0f });
        await Assert.That(np.common_type_code(a, b)).IsEqualTo(NPTypeCode.Single);
    }

    #endregion

    #region NPTypeCode Overload

    [Test]
    public async Task CommonTypeCode_SingleInt_ReturnsDouble()
    {
        await Assert.That(np.common_type_code(NPTypeCode.Int32)).IsEqualTo(NPTypeCode.Double);
    }

    [Test]
    public async Task CommonTypeCode_SingleFloat_ReturnsSingle()
    {
        await Assert.That(np.common_type_code(NPTypeCode.Single)).IsEqualTo(NPTypeCode.Single);
    }

    #endregion

    #region Type Overload

    [Test]
    public async Task CommonType_Type_Int32_ReturnsDouble()
    {
        var arr = np.array(new int[] { 1, 2 });
        var result = np.common_type(arr);
        await Assert.That(result).IsEqualTo(typeof(double));
    }

    [Test]
    public async Task CommonType_Type_Float32_ReturnsSingle()
    {
        var arr = np.array(new float[] { 1.0f, 2.0f });
        var result = np.common_type(arr);
        await Assert.That(result).IsEqualTo(typeof(float));
    }

    #endregion

    #region Error Cases

    [Test]
    public async Task CommonType_Empty_Throws()
    {
        await Assert.That(() => np.common_type_code(Array.Empty<NDArray>())).ThrowsException();
    }

    [Test]
    public async Task CommonType_Null_Throws()
    {
        await Assert.That(() => np.common_type_code((NDArray[])null!)).ThrowsException();
    }

    [Test]
    public async Task CommonTypeCode_Empty_Throws()
    {
        await Assert.That(() => np.common_type_code(Array.Empty<NPTypeCode>())).ThrowsException();
    }

    #endregion
}
