using System;
using System.Threading.Tasks;

namespace NumSharp.UnitTest.APIs;

/// <summary>
/// Battle tests for np.iinfo - comprehensive coverage of all integer types and edge cases.
/// </summary>
[TestClass]
public class NpIInfoBattleTests
{
    #region All Integer Types Coverage

    [TestMethod]
    public async Task IInfo_Boolean_AllProperties()
    {
        var info = np.iinfo(NPTypeCode.Boolean);
        info.bits.Should().Be(8);
        info.min.Should().Be(0);
        info.max.Should().Be(1);
        info.maxUnsigned.Should().Be(1UL);
        info.kind.Should().Be('b');
        info.dtype.Should().Be(NPTypeCode.Boolean);
    }

    [TestMethod]
    public async Task IInfo_Byte_AllProperties()
    {
        var info = np.iinfo(NPTypeCode.Byte);
        info.bits.Should().Be(8);
        info.min.Should().Be(0);
        info.max.Should().Be(255);
        info.maxUnsigned.Should().Be(255UL);
        info.kind.Should().Be('u');
    }

    [TestMethod]
    public async Task IInfo_Int16_AllProperties()
    {
        var info = np.iinfo(NPTypeCode.Int16);
        info.bits.Should().Be(16);
        info.min.Should().Be(short.MinValue);
        info.max.Should().Be(short.MaxValue);
        info.kind.Should().Be('i');
    }

    [TestMethod]
    public async Task IInfo_UInt16_AllProperties()
    {
        var info = np.iinfo(NPTypeCode.UInt16);
        info.bits.Should().Be(16);
        info.min.Should().Be(0);
        info.max.Should().Be(ushort.MaxValue);
        info.kind.Should().Be('u');
    }

    [TestMethod]
    public async Task IInfo_Int32_AllProperties()
    {
        var info = np.iinfo(NPTypeCode.Int32);
        info.bits.Should().Be(32);
        info.min.Should().Be(int.MinValue);
        info.max.Should().Be(int.MaxValue);
        info.kind.Should().Be('i');
    }

    [TestMethod]
    public async Task IInfo_UInt32_AllProperties()
    {
        var info = np.iinfo(NPTypeCode.UInt32);
        info.bits.Should().Be(32);
        info.min.Should().Be(0);
        info.max.Should().Be(uint.MaxValue);
        info.kind.Should().Be('u');
    }

    [TestMethod]
    public async Task IInfo_Int64_AllProperties()
    {
        var info = np.iinfo(NPTypeCode.Int64);
        info.bits.Should().Be(64);
        info.min.Should().Be(long.MinValue);
        info.max.Should().Be(long.MaxValue);
        info.kind.Should().Be('i');
    }

    [TestMethod]
    public async Task IInfo_UInt64_AllProperties()
    {
        var info = np.iinfo(NPTypeCode.UInt64);
        info.bits.Should().Be(64);
        info.min.Should().Be(0);
        info.max.Should().Be(long.MaxValue); // clamped
        info.maxUnsigned.Should().Be(ulong.MaxValue);
        info.kind.Should().Be('u');
    }

    [TestMethod]
    public async Task IInfo_Char_AllProperties()
    {
        var info = np.iinfo(NPTypeCode.Char);
        info.bits.Should().Be(16);
        info.min.Should().Be(0);
        info.max.Should().Be(char.MaxValue);
        info.kind.Should().Be('u');
    }

    #endregion

    #region Error Cases

    [TestMethod]
    public async Task IInfo_Single_Throws()
    {
        new Action(() => np.iinfo(NPTypeCode.Single)).Should().Throw<Exception>();
    }

    [TestMethod]
    public async Task IInfo_Double_Throws()
    {
        new Action(() => np.iinfo(NPTypeCode.Double)).Should().Throw<Exception>();
    }

    [TestMethod]
    public async Task IInfo_Decimal_Throws()
    {
        new Action(() => np.iinfo(NPTypeCode.Decimal)).Should().Throw<Exception>();
    }

    [TestMethod]
    public async Task IInfo_Empty_Throws()
    {
        new Action(() => np.iinfo(NPTypeCode.Empty)).Should().Throw<Exception>();
    }

    [TestMethod]
    public async Task IInfo_NullType_Throws()
    {
        new Action(() => np.iinfo((Type)null!)).Should().Throw<Exception>();
    }

    [TestMethod]
    public async Task IInfo_NullArray_Throws()
    {
        new Action(() => np.iinfo((NDArray)null!)).Should().Throw<Exception>();
    }

    [TestMethod]
    public async Task IInfo_EmptyDtypeString_Throws()
    {
        new Action(() => np.iinfo("")).Should().Throw<Exception>();
    }

    [TestMethod]
    public async Task IInfo_InvalidDtypeString_Throws()
    {
        new Action(() => np.iinfo("float32")).Should().Throw<Exception>();
    }

    #endregion

    #region Generic Overload Tests

    [TestMethod]
    public async Task IInfo_Generic_Int()
    {
        var info = np.iinfo<int>();
        info.bits.Should().Be(32);
        info.min.Should().Be(int.MinValue);
        info.max.Should().Be(int.MaxValue);
    }

    [TestMethod]
    public async Task IInfo_Generic_Byte()
    {
        var info = np.iinfo<byte>();
        info.bits.Should().Be(8);
        info.max.Should().Be(255);
    }

    [TestMethod]
    public async Task IInfo_Generic_Long()
    {
        var info = np.iinfo<long>();
        info.bits.Should().Be(64);
    }

    [TestMethod]
    public async Task IInfo_Generic_Bool()
    {
        var info = np.iinfo<bool>();
        info.bits.Should().Be(8);
        info.max.Should().Be(1);
    }

    [TestMethod]
    public async Task IInfo_Generic_Float_Throws()
    {
        new Action(() => np.iinfo<float>()).Should().Throw<Exception>();
    }

    #endregion

    #region NDArray Overload Tests

    [TestMethod]
    public async Task IInfo_NDArray_Int32()
    {
        var arr = np.array(new int[] { 1, 2, 3 });
        var info = np.iinfo(arr);
        info.bits.Should().Be(32);
        info.dtype.Should().Be(NPTypeCode.Int32);
    }

    [TestMethod]
    public async Task IInfo_NDArray_Byte()
    {
        var arr = np.array(new byte[] { 1, 2, 3 });
        var info = np.iinfo(arr);
        info.bits.Should().Be(8);
    }

    [TestMethod]
    public async Task IInfo_NDArray_Float_Throws()
    {
        var arr = np.array(new float[] { 1.0f, 2.0f });
        new Action(() => np.iinfo(arr)).Should().Throw<Exception>();
    }

    #endregion

    #region String dtype Overload Tests

    // Note: np.dtype() uses size+type format (e.g., "i4" for int32)
    // NumPy-style names like "int32" are not fully supported yet

    [TestMethod]
    public async Task IInfo_String_Int()
    {
        var info = np.iinfo("int");  // defaults to int32
        info.bits.Should().Be(32);
    }

    [TestMethod]
    public async Task IInfo_String_I4()
    {
        var info = np.iinfo("i4");  // int32
        info.bits.Should().Be(32);
    }

    [TestMethod]
    public async Task IInfo_String_I8()
    {
        var info = np.iinfo("i8");  // int64
        info.bits.Should().Be(64);
    }

    [TestMethod]
    public async Task IInfo_String_Bool()
    {
        var info = np.iinfo("bool");
        info.bits.Should().Be(8);
    }

    #endregion

    #region ToString Tests

    [TestMethod]
    public async Task IInfo_ToString_ContainsExpectedInfo()
    {
        var info = np.iinfo(NPTypeCode.Int32);
        var str = info.ToString();
        str.Should().Contain("min=");
        str.Should().Contain("max=");
        str.Should().Contain("int32");
    }

    #endregion

    #region UInt64 Max Value Edge Case

    [TestMethod]
    public async Task IInfo_UInt64_MaxExceedsLongMaxValue()
    {
        var info = np.iinfo(NPTypeCode.UInt64);
        info.maxUnsigned.Should().Be(ulong.MaxValue);
        info.maxUnsigned.Should().BeGreaterThan((ulong)info.max);
    }

    #endregion
}
