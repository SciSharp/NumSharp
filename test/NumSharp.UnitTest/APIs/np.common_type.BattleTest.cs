using System;
using System.Threading.Tasks;

namespace NumSharp.UnitTest.APIs;

/// <summary>
/// Battle tests for np.common_type - comprehensive coverage.
/// </summary>
[TestClass]
public class NpCommonTypeBattleTests
{
    #region Integer Arrays - Always Return Double

    [TestMethod]
    public async Task CommonType_Int32Array_ReturnsDouble()
    {
        var arr = np.array(new int[] { 1, 2, 3 });
        np.common_type_code(arr).Should().Be(NPTypeCode.Double);
    }

    [TestMethod]
    public async Task CommonType_ByteArray_ReturnsDouble()
    {
        var arr = np.array(new byte[] { 1, 2, 3 });
        np.common_type_code(arr).Should().Be(NPTypeCode.Double);
    }

    [TestMethod]
    public async Task CommonType_BoolArray_ReturnsDouble()
    {
        var arr = np.array(new bool[] { true, false });
        np.common_type_code(arr).Should().Be(NPTypeCode.Double);
    }

    #endregion

    #region Float Arrays

    [TestMethod]
    public async Task CommonType_Float32Array_ReturnsSingle()
    {
        var arr = np.array(new float[] { 1.0f, 2.0f });
        np.common_type_code(arr).Should().Be(NPTypeCode.Single);
    }

    [TestMethod]
    public async Task CommonType_Float64Array_ReturnsDouble()
    {
        var arr = np.array(new double[] { 1.0, 2.0 });
        np.common_type_code(arr).Should().Be(NPTypeCode.Double);
    }

    #endregion

    #region Multiple Arrays

    [TestMethod]
    public async Task CommonType_Float32AndFloat64_ReturnsDouble()
    {
        var a = np.array(new float[] { 1.0f });
        var b = np.array(new double[] { 1.0 });
        np.common_type_code(a, b).Should().Be(NPTypeCode.Double);
    }

    [TestMethod]
    public async Task CommonType_AllFloat32_ReturnsSingle()
    {
        var a = np.array(new float[] { 1.0f });
        var b = np.array(new float[] { 2.0f });
        np.common_type_code(a, b).Should().Be(NPTypeCode.Single);
    }

    #endregion

    #region NPTypeCode Overload

    [TestMethod]
    public async Task CommonTypeCode_SingleInt_ReturnsDouble()
    {
        np.common_type_code(NPTypeCode.Int32).Should().Be(NPTypeCode.Double);
    }

    [TestMethod]
    public async Task CommonTypeCode_SingleFloat_ReturnsSingle()
    {
        np.common_type_code(NPTypeCode.Single).Should().Be(NPTypeCode.Single);
    }

    #endregion

    #region Type Overload

    [TestMethod]
    public async Task CommonType_Type_Int32_ReturnsDouble()
    {
        var arr = np.array(new int[] { 1, 2 });
        var result = np.common_type(arr);
        result.Should().Be(typeof(double));
    }

    [TestMethod]
    public async Task CommonType_Type_Float32_ReturnsSingle()
    {
        var arr = np.array(new float[] { 1.0f, 2.0f });
        var result = np.common_type(arr);
        result.Should().Be(typeof(float));
    }

    #endregion

    #region Error Cases

    [TestMethod]
    public async Task CommonType_Empty_Throws()
    {
        new Action(() => np.common_type_code(Array.Empty<NDArray>())).Should().Throw<Exception>();
    }

    [TestMethod]
    public async Task CommonType_Null_Throws()
    {
        new Action(() => np.common_type_code((NDArray[])null!)).Should().Throw<Exception>();
    }

    [TestMethod]
    public async Task CommonTypeCode_Empty_Throws()
    {
        new Action(() => np.common_type_code(Array.Empty<NPTypeCode>())).Should().Throw<Exception>();
    }

    #endregion
}
