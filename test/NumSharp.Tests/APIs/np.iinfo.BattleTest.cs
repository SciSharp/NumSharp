using System;
using System.Linq;

namespace NumSharp.Tests.APIs;

/// <summary>
/// Battle tests for np.iinfo - comprehensive coverage of all integer types and edge cases.
/// </summary>
[TestClass]
public class NpIInfoBattleTests
{
    #region All Integer Types Coverage

    [TestMethod]
    public void IInfo_Boolean_AllProperties()
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
    public void IInfo_Byte_AllProperties()
    {
        var info = np.iinfo(NPTypeCode.Byte);
        info.bits.Should().Be(8);
        info.min.Should().Be(0);
        info.max.Should().Be(255);
        info.maxUnsigned.Should().Be(255UL);
        info.kind.Should().Be('u');
    }

    [TestMethod]
    public void IInfo_Int16_AllProperties()
    {
        var info = np.iinfo(NPTypeCode.Int16);
        info.bits.Should().Be(16);
        info.min.Should().Be(short.MinValue);
        info.max.Should().Be(short.MaxValue);
        info.kind.Should().Be('i');
    }

    [TestMethod]
    public void IInfo_UInt16_AllProperties()
    {
        var info = np.iinfo(NPTypeCode.UInt16);
        info.bits.Should().Be(16);
        info.min.Should().Be(0);
        info.max.Should().Be(ushort.MaxValue);
        info.kind.Should().Be('u');
    }

    [TestMethod]
    public void IInfo_Int32_AllProperties()
    {
        var info = np.iinfo(NPTypeCode.Int32);
        info.bits.Should().Be(32);
        info.min.Should().Be(int.MinValue);
        info.max.Should().Be(int.MaxValue);
        info.kind.Should().Be('i');
    }

    [TestMethod]
    public void IInfo_UInt32_AllProperties()
    {
        var info = np.iinfo(NPTypeCode.UInt32);
        info.bits.Should().Be(32);
        info.min.Should().Be(0);
        info.max.Should().Be(uint.MaxValue);
        info.kind.Should().Be('u');
    }

    [TestMethod]
    public void IInfo_Int64_AllProperties()
    {
        var info = np.iinfo(NPTypeCode.Int64);
        info.bits.Should().Be(64);
        info.min.Should().Be(long.MinValue);
        info.max.Should().Be(long.MaxValue);
        info.kind.Should().Be('i');
    }

    [TestMethod]
    public void IInfo_UInt64_AllProperties()
    {
        var info = np.iinfo(NPTypeCode.UInt64);
        info.bits.Should().Be(64);
        info.min.Should().Be(0);
        info.max.Should().Be(long.MaxValue); // clamped
        info.maxUnsigned.Should().Be(ulong.MaxValue);
        info.kind.Should().Be('u');
    }

    [TestMethod]
    public void IInfo_Char_AllProperties()
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
    public void IInfo_Single_Throws()
    {
        new Action(() => np.iinfo(NPTypeCode.Single)).Should().Throw<Exception>();
    }

    [TestMethod]
    public void IInfo_Double_Throws()
    {
        new Action(() => np.iinfo(NPTypeCode.Double)).Should().Throw<Exception>();
    }

    [TestMethod]
    public void IInfo_Decimal_Throws()
    {
        new Action(() => np.iinfo(NPTypeCode.Decimal)).Should().Throw<Exception>();
    }

    [TestMethod]
    public void IInfo_Empty_Throws()
    {
        new Action(() => np.iinfo(NPTypeCode.Empty)).Should().Throw<Exception>();
    }

    [TestMethod]
    public void IInfo_NullType_Throws()
    {
        new Action(() => np.iinfo((Type)null!)).Should().Throw<Exception>();
    }

    [TestMethod]
    public void IInfo_HasNoArrayOverload_AsNumPyRefusesAnArray()
    {
        // NumPy 2.4.2: np.iinfo(np.arange(3)) is "Invalid integer data type 'O'." (an array is not a dtype); API-parity
        // batch 7 removed the NDArray form, so a C# array argument does not compile: pass arr.dtype.
        typeof(np).GetMethods().Where(m => m.Name == "iinfo")
            .Any(m => m.GetParameters().Any(p => p.ParameterType == typeof(NDArray))).Should().BeFalse();
    }

    [TestMethod]
    public void IInfo_EmptyDtypeString_Throws()
    {
        new Action(() => np.iinfo("")).Should().Throw<Exception>();
    }

    [TestMethod]
    public void IInfo_InvalidDtypeString_Throws()
    {
        new Action(() => np.iinfo("float32")).Should().Throw<Exception>();
    }

    #endregion

    #region C# type spellings (the generic iinfo<T> went in API-parity batch 9, OD7)

    /// <summary>
    ///     The generic <c>iinfo&lt;int&gt;()</c> went (OD7): <c>np.iinfo(typeof(int))</c> is NumPy's
    ///     <c>np.iinfo(np.int32)</c> (32 bits, int.MinValue..int.MaxValue).
    /// </summary>
    [TestMethod]
    public void IInfo_TypeofInt()
    {
        // NumPy: np.iinfo(np.int32) bits 32, min -2147483648, max 2147483647.
        var info = np.iinfo(typeof(int));
        info.bits.Should().Be(32);
        info.min.Should().Be(int.MinValue);
        info.max.Should().Be(int.MaxValue);
    }

    /// <summary>
    ///     <c>np.iinfo(typeof(byte))</c> is NumPy's <c>np.iinfo(np.uint8)</c> (8 bits, max 255).
    /// </summary>
    [TestMethod]
    public void IInfo_TypeofByte()
    {
        // NumPy: np.iinfo(np.uint8).max == 255.
        var info = np.iinfo(typeof(byte));
        info.bits.Should().Be(8);
        info.max.Should().Be(255);
    }

    /// <summary>
    ///     <c>np.iinfo(typeof(long))</c> is NumPy's <c>np.iinfo(np.int64)</c> (64 bits).
    /// </summary>
    [TestMethod]
    public void IInfo_TypeofLong()
    {
        var info = np.iinfo(typeof(long));
        info.bits.Should().Be(64);
    }

    /// <summary>
    ///     <c>np.iinfo(typeof(bool))</c>, the C# type spelling the removed generic <c>iinfo&lt;bool&gt;()</c> migrated to:
    ///     NumSharp's bool extension answers it (bits 8, max 1) where NumPy raises <c>ValueError: Invalid integer data type
    ///     'b'.</c>; that divergence is the open pin in OpenBugs.Surface, so this test keeps the old assertion.
    /// </summary>
    [TestMethod]
    public void IInfo_TypeofBool()
    {
        // The removed generic iinfo<bool>() answered this, and the dtype form still does: NumSharp's bool extension (bits 8,
        // max 1). NumPy raises ValueError: Invalid integer data type 'b'. — pinned in OpenBugs.Surface (np.iinfo(np.@bool)).
        var info = np.iinfo(typeof(bool));
        info.bits.Should().Be(8);
        info.max.Should().Be(1);
    }

    /// <summary>
    ///     A float type is NumPy's <c>ValueError: Invalid integer data type 'f'.</c>, the kind character.
    /// </summary>
    [TestMethod]
    public void IInfo_TypeofFloat_IsNumPysValueError()
    {
        // NumPy: np.iinfo(np.float32) -> ValueError: Invalid integer data type 'f'.
        new Action(() => np.iinfo(typeof(float))).Should().Throw<ValueError>().WithMessage("Invalid integer data type 'f'.");
    }

    #endregion

    #region NDArray Overload Tests

    [TestMethod]
    public void IInfo_ArrayDtype_Int32()
    {
        var arr = np.array(new int[] { 1, 2, 3 });
        var info = np.iinfo(arr.dtype);
        info.bits.Should().Be(32);
        info.dtype.Should().Be(NPTypeCode.Int32);
    }

    [TestMethod]
    public void IInfo_ArrayDtype_Byte()
    {
        var arr = np.array(new byte[] { 1, 2, 3 });
        var info = np.iinfo(arr.dtype);
        info.bits.Should().Be(8);
    }

    [TestMethod]
    public void IInfo_ArrayDtype_Float_Throws()
    {
        // NumPy 2.4.2: np.iinfo(np.float32) names the KIND character.
        var arr = np.array(new float[] { 1.0f, 2.0f });
        new Action(() => np.iinfo(arr.dtype)).Should().Throw<ValueError>().WithMessage("Invalid integer data type 'f'.");
    }

    #endregion

    #region String dtype Overload Tests

    [TestMethod]
    public void IInfo_String_Int()
    {
        // NumPy 2.x parity: 'int' aliases to intp (pointer-sized) = int64 on 64-bit platforms.
        var info = np.iinfo("int");
        var expected = IntPtr.Size == 8 ? 64 : 32;
        info.bits.Should().Be(expected);
    }

    [TestMethod]
    public void IInfo_String_I4()
    {
        var info = np.iinfo("i4");  // int32
        info.bits.Should().Be(32);
    }

    [TestMethod]
    public void IInfo_String_I8()
    {
        var info = np.iinfo("i8");  // int64
        info.bits.Should().Be(64);
    }

    [TestMethod]
    public void IInfo_String_Bool()
    {
        var info = np.iinfo("bool");
        info.bits.Should().Be(8);
    }

    #endregion

    #region ToString Tests

    [TestMethod]
    public void IInfo_ToString_ContainsExpectedInfo()
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
    public void IInfo_UInt64_MaxExceedsLongMaxValue()
    {
        var info = np.iinfo(NPTypeCode.UInt64);
        info.maxUnsigned.Should().Be(ulong.MaxValue);
        info.maxUnsigned.Should().BeGreaterThan((ulong)info.max);
    }

    #endregion
}
