using System;
using System.Threading.Tasks;

namespace NumSharp.UnitTest.APIs;

/// <summary>
/// Battle tests for np.finfo - comprehensive coverage of all floating point types.
/// </summary>
public class NpFInfoBattleTests
{
    #region All Float Types Coverage

    [TestMethod]
    public async Task FInfo_Single_AllProperties()
    {
        var info = np.finfo(NPTypeCode.Single);
        await Assert.That(info.bits).IsEqualTo(32);
        await Assert.That(info.precision).IsEqualTo(6);
        await Assert.That(info.maxexp).IsEqualTo(128);
        await Assert.That(info.minexp).IsEqualTo(-125);
        // max/min are stored as double, so compare with tolerance
        await Assert.That(info.max).IsGreaterThan(3.4e38);
        await Assert.That(info.min).IsLessThan(-3.4e38);
        await Assert.That(info.tiny).IsEqualTo(info.smallest_normal);
        // smallest_subnormal for float is ~1.4e-45
        await Assert.That(info.smallest_subnormal).IsLessThan(1e-44);
        await Assert.That(info.resolution).IsEqualTo(1e-6);
        // eps for float32 ~ 1.19e-07
        await Assert.That(info.eps).IsGreaterThan(1e-7);
        await Assert.That(info.eps).IsLessThan(2e-7);
    }

    [TestMethod]
    public async Task FInfo_Double_AllProperties()
    {
        var info = np.finfo(NPTypeCode.Double);
        await Assert.That(info.bits).IsEqualTo(64);
        await Assert.That(info.precision).IsEqualTo(15);
        await Assert.That(info.maxexp).IsEqualTo(1024);
        await Assert.That(info.minexp).IsEqualTo(-1021);
        await Assert.That(info.max).IsEqualTo(double.MaxValue);
        await Assert.That(info.min).IsEqualTo(-double.MaxValue);
        await Assert.That(info.smallest_subnormal).IsEqualTo(double.Epsilon);
        await Assert.That(info.resolution).IsEqualTo(1e-15);
        // eps ~ 2.22e-16
        await Assert.That(info.eps).IsGreaterThan(2e-16);
        await Assert.That(info.eps).IsLessThan(3e-16);
    }

    [TestMethod]
    public async Task FInfo_Decimal_AllProperties()
    {
        var info = np.finfo(NPTypeCode.Decimal);
        await Assert.That(info.bits).IsEqualTo(128);
        await Assert.That(info.precision).IsEqualTo(28);
        await Assert.That(info.resolution).IsEqualTo(1e-28);
    }

    #endregion

    #region Error Cases

    [TestMethod]
    public async Task FInfo_Int32_Throws()
    {
        await Assert.That(() => np.finfo(NPTypeCode.Int32)).ThrowsException();
    }

    [TestMethod]
    public async Task FInfo_Boolean_Throws()
    {
        await Assert.That(() => np.finfo(NPTypeCode.Boolean)).ThrowsException();
    }

    [TestMethod]
    public async Task FInfo_Byte_Throws()
    {
        await Assert.That(() => np.finfo(NPTypeCode.Byte)).ThrowsException();
    }

    [TestMethod]
    public async Task FInfo_Empty_Throws()
    {
        await Assert.That(() => np.finfo(NPTypeCode.Empty)).ThrowsException();
    }

    [TestMethod]
    public async Task FInfo_NullType_Throws()
    {
        await Assert.That(() => np.finfo((Type)null!)).ThrowsException();
    }

    [TestMethod]
    public async Task FInfo_NullArray_Throws()
    {
        await Assert.That(() => np.finfo((NDArray)null!)).ThrowsException();
    }

    [TestMethod]
    public async Task FInfo_EmptyDtypeString_Throws()
    {
        await Assert.That(() => np.finfo("")).ThrowsException();
    }

    [TestMethod]
    public async Task FInfo_InvalidDtypeString_Throws()
    {
        await Assert.That(() => np.finfo("int32")).ThrowsException();
    }

    #endregion

    #region Generic Overload Tests

    [TestMethod]
    public async Task FInfo_Generic_Float()
    {
        var info = np.finfo<float>();
        await Assert.That(info.bits).IsEqualTo(32);
        await Assert.That(info.precision).IsEqualTo(6);
    }

    [TestMethod]
    public async Task FInfo_Generic_Double()
    {
        var info = np.finfo<double>();
        await Assert.That(info.bits).IsEqualTo(64);
        await Assert.That(info.precision).IsEqualTo(15);
    }

    [TestMethod]
    public async Task FInfo_Generic_Decimal()
    {
        var info = np.finfo<decimal>();
        await Assert.That(info.bits).IsEqualTo(128);
    }

    [TestMethod]
    public async Task FInfo_Generic_Int_Throws()
    {
        await Assert.That(() => np.finfo<int>()).ThrowsException();
    }

    #endregion

    #region NDArray Overload Tests

    [TestMethod]
    public async Task FInfo_NDArray_Float32()
    {
        var arr = np.array(new float[] { 1.0f, 2.0f, 3.0f });
        var info = np.finfo(arr);
        await Assert.That(info.bits).IsEqualTo(32);
    }

    [TestMethod]
    public async Task FInfo_NDArray_Float64()
    {
        var arr = np.array(new double[] { 1.0, 2.0, 3.0 });
        var info = np.finfo(arr);
        await Assert.That(info.bits).IsEqualTo(64);
    }

    [TestMethod]
    public async Task FInfo_NDArray_Int_Throws()
    {
        var arr = np.array(new int[] { 1, 2, 3 });
        await Assert.That(() => np.finfo(arr)).ThrowsException();
    }

    #endregion

    #region String dtype Overload Tests

    // Note: np.dtype() uses type names like "float", "double", "single"
    // NumPy-style names like "float32", "float64" are not fully supported yet

    [TestMethod]
    public async Task FInfo_String_Float()
    {
        var info = np.finfo("float");  // defaults to float (single)
        await Assert.That(info.bits).IsEqualTo(32);
    }

    [TestMethod]
    public async Task FInfo_String_Single()
    {
        var info = np.finfo("single");
        await Assert.That(info.bits).IsEqualTo(32);
    }

    [TestMethod]
    public async Task FInfo_String_Double()
    {
        var info = np.finfo("double");
        await Assert.That(info.bits).IsEqualTo(64);
    }

    #endregion

    #region Epsilon Verification

    [TestMethod]
    public async Task FInfo_Double_EpsIsNextFloat()
    {
        var info = np.finfo(NPTypeCode.Double);
        double expected = Math.BitIncrement(1.0) - 1.0;
        await Assert.That(info.eps).IsEqualTo(expected);
    }

    [TestMethod]
    public async Task FInfo_Single_EpsIsNextFloat()
    {
        var info = np.finfo(NPTypeCode.Single);
        // Must use MathF for float operations
        double expected = MathF.BitIncrement(1.0f) - 1.0f;
        await Assert.That(info.eps).IsEqualTo(expected);
    }

    #endregion

    #region ToString Tests

    [TestMethod]
    public async Task FInfo_ToString_ContainsExpectedInfo()
    {
        var info = np.finfo(NPTypeCode.Double);
        var str = info.ToString();
        await Assert.That(str).Contains("resolution=");
        await Assert.That(str).Contains("float64");
    }

    #endregion
}
