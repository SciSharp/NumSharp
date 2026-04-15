using System;
using System.Threading.Tasks;

namespace NumSharp.UnitTest.APIs;

/// <summary>
/// Battle tests for np.iinfo - comprehensive coverage of all integer types and edge cases.
/// </summary>
public class NpIInfoBattleTests
{
    #region All Integer Types Coverage

    [TestMethod]
    public async Task IInfo_Boolean_AllProperties()
    {
        var info = np.iinfo(NPTypeCode.Boolean);
        await Assert.That(info.bits).IsEqualTo(8);
        await Assert.That(info.min).IsEqualTo(0);
        await Assert.That(info.max).IsEqualTo(1);
        await Assert.That(info.maxUnsigned).IsEqualTo(1UL);
        await Assert.That(info.kind).IsEqualTo('b');
        await Assert.That(info.dtype).IsEqualTo(NPTypeCode.Boolean);
    }

    [TestMethod]
    public async Task IInfo_Byte_AllProperties()
    {
        var info = np.iinfo(NPTypeCode.Byte);
        await Assert.That(info.bits).IsEqualTo(8);
        await Assert.That(info.min).IsEqualTo(0);
        await Assert.That(info.max).IsEqualTo(255);
        await Assert.That(info.maxUnsigned).IsEqualTo(255UL);
        await Assert.That(info.kind).IsEqualTo('u');
    }

    [TestMethod]
    public async Task IInfo_Int16_AllProperties()
    {
        var info = np.iinfo(NPTypeCode.Int16);
        await Assert.That(info.bits).IsEqualTo(16);
        await Assert.That(info.min).IsEqualTo(short.MinValue);
        await Assert.That(info.max).IsEqualTo(short.MaxValue);
        await Assert.That(info.kind).IsEqualTo('i');
    }

    [TestMethod]
    public async Task IInfo_UInt16_AllProperties()
    {
        var info = np.iinfo(NPTypeCode.UInt16);
        await Assert.That(info.bits).IsEqualTo(16);
        await Assert.That(info.min).IsEqualTo(0);
        await Assert.That(info.max).IsEqualTo(ushort.MaxValue);
        await Assert.That(info.kind).IsEqualTo('u');
    }

    [TestMethod]
    public async Task IInfo_Int32_AllProperties()
    {
        var info = np.iinfo(NPTypeCode.Int32);
        await Assert.That(info.bits).IsEqualTo(32);
        await Assert.That(info.min).IsEqualTo(int.MinValue);
        await Assert.That(info.max).IsEqualTo(int.MaxValue);
        await Assert.That(info.kind).IsEqualTo('i');
    }

    [TestMethod]
    public async Task IInfo_UInt32_AllProperties()
    {
        var info = np.iinfo(NPTypeCode.UInt32);
        await Assert.That(info.bits).IsEqualTo(32);
        await Assert.That(info.min).IsEqualTo(0);
        await Assert.That(info.max).IsEqualTo(uint.MaxValue);
        await Assert.That(info.kind).IsEqualTo('u');
    }

    [TestMethod]
    public async Task IInfo_Int64_AllProperties()
    {
        var info = np.iinfo(NPTypeCode.Int64);
        await Assert.That(info.bits).IsEqualTo(64);
        await Assert.That(info.min).IsEqualTo(long.MinValue);
        await Assert.That(info.max).IsEqualTo(long.MaxValue);
        await Assert.That(info.kind).IsEqualTo('i');
    }

    [TestMethod]
    public async Task IInfo_UInt64_AllProperties()
    {
        var info = np.iinfo(NPTypeCode.UInt64);
        await Assert.That(info.bits).IsEqualTo(64);
        await Assert.That(info.min).IsEqualTo(0);
        await Assert.That(info.max).IsEqualTo(long.MaxValue); // clamped
        await Assert.That(info.maxUnsigned).IsEqualTo(ulong.MaxValue);
        await Assert.That(info.kind).IsEqualTo('u');
    }

    [TestMethod]
    public async Task IInfo_Char_AllProperties()
    {
        var info = np.iinfo(NPTypeCode.Char);
        await Assert.That(info.bits).IsEqualTo(16);
        await Assert.That(info.min).IsEqualTo(0);
        await Assert.That(info.max).IsEqualTo(char.MaxValue);
        await Assert.That(info.kind).IsEqualTo('u');
    }

    #endregion

    #region Error Cases

    [TestMethod]
    public async Task IInfo_Single_Throws()
    {
        await Assert.That(() => np.iinfo(NPTypeCode.Single)).ThrowsException();
    }

    [TestMethod]
    public async Task IInfo_Double_Throws()
    {
        await Assert.That(() => np.iinfo(NPTypeCode.Double)).ThrowsException();
    }

    [TestMethod]
    public async Task IInfo_Decimal_Throws()
    {
        await Assert.That(() => np.iinfo(NPTypeCode.Decimal)).ThrowsException();
    }

    [TestMethod]
    public async Task IInfo_Empty_Throws()
    {
        await Assert.That(() => np.iinfo(NPTypeCode.Empty)).ThrowsException();
    }

    [TestMethod]
    public async Task IInfo_NullType_Throws()
    {
        await Assert.That(() => np.iinfo((Type)null!)).ThrowsException();
    }

    [TestMethod]
    public async Task IInfo_NullArray_Throws()
    {
        await Assert.That(() => np.iinfo((NDArray)null!)).ThrowsException();
    }

    [TestMethod]
    public async Task IInfo_EmptyDtypeString_Throws()
    {
        await Assert.That(() => np.iinfo("")).ThrowsException();
    }

    [TestMethod]
    public async Task IInfo_InvalidDtypeString_Throws()
    {
        await Assert.That(() => np.iinfo("float32")).ThrowsException();
    }

    #endregion

    #region Generic Overload Tests

    [TestMethod]
    public async Task IInfo_Generic_Int()
    {
        var info = np.iinfo<int>();
        await Assert.That(info.bits).IsEqualTo(32);
        await Assert.That(info.min).IsEqualTo(int.MinValue);
        await Assert.That(info.max).IsEqualTo(int.MaxValue);
    }

    [TestMethod]
    public async Task IInfo_Generic_Byte()
    {
        var info = np.iinfo<byte>();
        await Assert.That(info.bits).IsEqualTo(8);
        await Assert.That(info.max).IsEqualTo(255);
    }

    [TestMethod]
    public async Task IInfo_Generic_Long()
    {
        var info = np.iinfo<long>();
        await Assert.That(info.bits).IsEqualTo(64);
    }

    [TestMethod]
    public async Task IInfo_Generic_Bool()
    {
        var info = np.iinfo<bool>();
        await Assert.That(info.bits).IsEqualTo(8);
        await Assert.That(info.max).IsEqualTo(1);
    }

    [TestMethod]
    public async Task IInfo_Generic_Float_Throws()
    {
        await Assert.That(() => np.iinfo<float>()).ThrowsException();
    }

    #endregion

    #region NDArray Overload Tests

    [TestMethod]
    public async Task IInfo_NDArray_Int32()
    {
        var arr = np.array(new int[] { 1, 2, 3 });
        var info = np.iinfo(arr);
        await Assert.That(info.bits).IsEqualTo(32);
        await Assert.That(info.dtype).IsEqualTo(NPTypeCode.Int32);
    }

    [TestMethod]
    public async Task IInfo_NDArray_Byte()
    {
        var arr = np.array(new byte[] { 1, 2, 3 });
        var info = np.iinfo(arr);
        await Assert.That(info.bits).IsEqualTo(8);
    }

    [TestMethod]
    public async Task IInfo_NDArray_Float_Throws()
    {
        var arr = np.array(new float[] { 1.0f, 2.0f });
        await Assert.That(() => np.iinfo(arr)).ThrowsException();
    }

    #endregion

    #region String dtype Overload Tests

    // Note: np.dtype() uses size+type format (e.g., "i4" for int32)
    // NumPy-style names like "int32" are not fully supported yet

    [TestMethod]
    public async Task IInfo_String_Int()
    {
        var info = np.iinfo("int");  // defaults to int32
        await Assert.That(info.bits).IsEqualTo(32);
    }

    [TestMethod]
    public async Task IInfo_String_I4()
    {
        var info = np.iinfo("i4");  // int32
        await Assert.That(info.bits).IsEqualTo(32);
    }

    [TestMethod]
    public async Task IInfo_String_I8()
    {
        var info = np.iinfo("i8");  // int64
        await Assert.That(info.bits).IsEqualTo(64);
    }

    [TestMethod]
    public async Task IInfo_String_Bool()
    {
        var info = np.iinfo("bool");
        await Assert.That(info.bits).IsEqualTo(8);
    }

    #endregion

    #region ToString Tests

    [TestMethod]
    public async Task IInfo_ToString_ContainsExpectedInfo()
    {
        var info = np.iinfo(NPTypeCode.Int32);
        var str = info.ToString();
        await Assert.That(str).Contains("min=");
        await Assert.That(str).Contains("max=");
        await Assert.That(str).Contains("int32");
    }

    #endregion

    #region UInt64 Max Value Edge Case

    [TestMethod]
    public async Task IInfo_UInt64_MaxExceedsLongMaxValue()
    {
        var info = np.iinfo(NPTypeCode.UInt64);
        await Assert.That(info.maxUnsigned).IsEqualTo(ulong.MaxValue);
        await Assert.That(info.maxUnsigned).IsGreaterThan((ulong)info.max);
    }

    #endregion
}
