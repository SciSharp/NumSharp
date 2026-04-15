using System;
using System.Threading.Tasks;

namespace NumSharp.UnitTest.APIs;

/// <summary>
/// Battle tests for np.finfo - comprehensive coverage of all floating point types.
/// </summary>
[TestClass]
public class NpFInfoBattleTests
{
    #region All Float Types Coverage

    [TestMethod]
    public async Task FInfo_Single_AllProperties()
    {
        var info = np.finfo(NPTypeCode.Single);
        info.bits.Should().Be(32);
        info.precision.Should().Be(6);
        info.maxexp.Should().Be(128);
        info.minexp.Should().Be(-125);
        // max/min are stored as double, so compare with tolerance
        info.max.Should().BeGreaterThan(3.4e38);
        info.min.Should().BeLessThan(-3.4e38);
        info.tiny.Should().Be(info.smallest_normal);
        // smallest_subnormal for float is ~1.4e-45
        info.smallest_subnormal.Should().BeLessThan(1e-44);
        info.resolution.Should().Be(1e-6);
        // eps for float32 ~ 1.19e-07
        info.eps.Should().BeGreaterThan(1e-7);
        info.eps.Should().BeLessThan(2e-7);
    }

    [TestMethod]
    public async Task FInfo_Double_AllProperties()
    {
        var info = np.finfo(NPTypeCode.Double);
        info.bits.Should().Be(64);
        info.precision.Should().Be(15);
        info.maxexp.Should().Be(1024);
        info.minexp.Should().Be(-1021);
        info.max.Should().Be(double.MaxValue);
        info.min.Should().Be(-double.MaxValue);
        info.smallest_subnormal.Should().Be(double.Epsilon);
        info.resolution.Should().Be(1e-15);
        // eps ~ 2.22e-16
        info.eps.Should().BeGreaterThan(2e-16);
        info.eps.Should().BeLessThan(3e-16);
    }

    [TestMethod]
    public async Task FInfo_Decimal_AllProperties()
    {
        var info = np.finfo(NPTypeCode.Decimal);
        info.bits.Should().Be(128);
        info.precision.Should().Be(28);
        info.resolution.Should().Be(1e-28);
    }

    #endregion

    #region Error Cases

    [TestMethod]
    public async Task FInfo_Int32_Throws()
    {
        Microsoft.VisualStudio.TestTools.UnitTesting.Assert.ThrowsException<Exception>(() => np.finfo(NPTypeCode.Int32));
    }

    [TestMethod]
    public async Task FInfo_Boolean_Throws()
    {
        Microsoft.VisualStudio.TestTools.UnitTesting.Assert.ThrowsException<Exception>(() => np.finfo(NPTypeCode.Boolean));
    }

    [TestMethod]
    public async Task FInfo_Byte_Throws()
    {
        Microsoft.VisualStudio.TestTools.UnitTesting.Assert.ThrowsException<Exception>(() => np.finfo(NPTypeCode.Byte));
    }

    [TestMethod]
    public async Task FInfo_Empty_Throws()
    {
        Microsoft.VisualStudio.TestTools.UnitTesting.Assert.ThrowsException<Exception>(() => np.finfo(NPTypeCode.Empty));
    }

    [TestMethod]
    public async Task FInfo_NullType_Throws()
    {
        Microsoft.VisualStudio.TestTools.UnitTesting.Assert.ThrowsException<Exception>(() => np.finfo((Type)null!));
    }

    [TestMethod]
    public async Task FInfo_NullArray_Throws()
    {
        Microsoft.VisualStudio.TestTools.UnitTesting.Assert.ThrowsException<Exception>(() => np.finfo((NDArray)null!));
    }

    [TestMethod]
    public async Task FInfo_EmptyDtypeString_Throws()
    {
        Microsoft.VisualStudio.TestTools.UnitTesting.Assert.ThrowsException<Exception>(() => np.finfo(""));
    }

    [TestMethod]
    public async Task FInfo_InvalidDtypeString_Throws()
    {
        Microsoft.VisualStudio.TestTools.UnitTesting.Assert.ThrowsException<Exception>(() => np.finfo("int32"));
    }

    #endregion

    #region Generic Overload Tests

    [TestMethod]
    public async Task FInfo_Generic_Float()
    {
        var info = np.finfo<float>();
        info.bits.Should().Be(32);
        info.precision.Should().Be(6);
    }

    [TestMethod]
    public async Task FInfo_Generic_Double()
    {
        var info = np.finfo<double>();
        info.bits.Should().Be(64);
        info.precision.Should().Be(15);
    }

    [TestMethod]
    public async Task FInfo_Generic_Decimal()
    {
        var info = np.finfo<decimal>();
        info.bits.Should().Be(128);
    }

    [TestMethod]
    public async Task FInfo_Generic_Int_Throws()
    {
        Microsoft.VisualStudio.TestTools.UnitTesting.Assert.ThrowsException<Exception>(() => np.finfo<int>());
    }

    #endregion

    #region NDArray Overload Tests

    [TestMethod]
    public async Task FInfo_NDArray_Float32()
    {
        var arr = np.array(new float[] { 1.0f, 2.0f, 3.0f });
        var info = np.finfo(arr);
        info.bits.Should().Be(32);
    }

    [TestMethod]
    public async Task FInfo_NDArray_Float64()
    {
        var arr = np.array(new double[] { 1.0, 2.0, 3.0 });
        var info = np.finfo(arr);
        info.bits.Should().Be(64);
    }

    [TestMethod]
    public async Task FInfo_NDArray_Int_Throws()
    {
        var arr = np.array(new int[] { 1, 2, 3 });
        Microsoft.VisualStudio.TestTools.UnitTesting.Assert.ThrowsException<Exception>(() => np.finfo(arr));
    }

    #endregion

    #region String dtype Overload Tests

    // Note: np.dtype() uses type names like "float", "double", "single"
    // NumPy-style names like "float32", "float64" are not fully supported yet

    [TestMethod]
    public async Task FInfo_String_Float()
    {
        var info = np.finfo("float");  // defaults to float (single)
        info.bits.Should().Be(32);
    }

    [TestMethod]
    public async Task FInfo_String_Single()
    {
        var info = np.finfo("single");
        info.bits.Should().Be(32);
    }

    [TestMethod]
    public async Task FInfo_String_Double()
    {
        var info = np.finfo("double");
        info.bits.Should().Be(64);
    }

    #endregion

    #region Epsilon Verification

    [TestMethod]
    public async Task FInfo_Double_EpsIsNextFloat()
    {
        var info = np.finfo(NPTypeCode.Double);
        double expected = Math.BitIncrement(1.0) - 1.0;
        info.eps.Should().Be(expected);
    }

    [TestMethod]
    public async Task FInfo_Single_EpsIsNextFloat()
    {
        var info = np.finfo(NPTypeCode.Single);
        // Must use MathF for float operations
        double expected = MathF.BitIncrement(1.0f) - 1.0f;
        info.eps.Should().Be(expected);
    }

    #endregion

    #region ToString Tests

    [TestMethod]
    public async Task FInfo_ToString_ContainsExpectedInfo()
    {
        var info = np.finfo(NPTypeCode.Double);
        var str = info.ToString();
        str.Should().Contain("resolution=");
        str.Should().Contain("float64");
    }

    #endregion
}
