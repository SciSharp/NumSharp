using System;
using System.Threading.Tasks;

namespace NumSharp.UnitTest.APIs;

/// <summary>
/// Battle tests for np.can_cast - comprehensive coverage of type casting rules.
/// </summary>
public class NpCanCastBattleTests
{
    #region Safe Casting - Type to Type

    [TestMethod]
    public async Task CanCast_SameType_AlwaysTrue()
    {
        np.can_cast(NPTypeCode.Int32, NPTypeCode.Int32).Should().BeTrue();
        np.can_cast(NPTypeCode.Double, NPTypeCode.Double).Should().BeTrue();
        np.can_cast(NPTypeCode.Boolean, NPTypeCode.Boolean).Should().BeTrue();
    }

    [TestMethod]
    public async Task CanCast_BoolToIntegers_Safe()
    {
        np.can_cast(NPTypeCode.Boolean, NPTypeCode.Byte).Should().BeTrue();
        np.can_cast(NPTypeCode.Boolean, NPTypeCode.Int32).Should().BeTrue();
        np.can_cast(NPTypeCode.Boolean, NPTypeCode.Int64).Should().BeTrue();
    }

    [TestMethod]
    public async Task CanCast_BoolToFloats_Safe()
    {
        np.can_cast(NPTypeCode.Boolean, NPTypeCode.Single).Should().BeTrue();
        np.can_cast(NPTypeCode.Boolean, NPTypeCode.Double).Should().BeTrue();
    }

    [TestMethod]
    public async Task CanCast_ByteUpcast_Safe()
    {
        np.can_cast(NPTypeCode.Byte, NPTypeCode.Int16).Should().BeTrue();
        np.can_cast(NPTypeCode.Byte, NPTypeCode.Int32).Should().BeTrue();
        np.can_cast(NPTypeCode.Byte, NPTypeCode.Double).Should().BeTrue();
    }

    [TestMethod]
    public async Task CanCast_Int32ToInt64_Safe()
    {
        np.can_cast(NPTypeCode.Int32, NPTypeCode.Int64).Should().BeTrue();
    }

    [TestMethod]
    public async Task CanCast_Int64ToInt32_NotSafe()
    {
        np.can_cast(NPTypeCode.Int64, NPTypeCode.Int32).Should().BeFalse();
    }

    [TestMethod]
    public async Task CanCast_Float32ToFloat64_Safe()
    {
        np.can_cast(NPTypeCode.Single, NPTypeCode.Double).Should().BeTrue();
    }

    [TestMethod]
    public async Task CanCast_Float64ToFloat32_NotSafe()
    {
        np.can_cast(NPTypeCode.Double, NPTypeCode.Single).Should().BeFalse();
    }

    [TestMethod]
    public async Task CanCast_FloatToInt_NotSafe()
    {
        np.can_cast(NPTypeCode.Double, NPTypeCode.Int32).Should().BeFalse();
    }

    #endregion

    #region Casting Modes

    [TestMethod]
    public async Task CanCast_NoMode_OnlySameType()
    {
        np.can_cast(NPTypeCode.Int32, NPTypeCode.Int32, "no").Should().BeTrue();
        np.can_cast(NPTypeCode.Int32, NPTypeCode.Int64, "no").Should().BeFalse();
    }

    [TestMethod]
    public async Task CanCast_SameKindMode_AllowsDowncastWithinKind()
    {
        np.can_cast(NPTypeCode.Int32, NPTypeCode.Int16, "same_kind").Should().BeTrue();
        np.can_cast(NPTypeCode.Double, NPTypeCode.Single, "same_kind").Should().BeTrue();
    }

    [TestMethod]
    public async Task CanCast_SameKindMode_RejectsCrossKind()
    {
        np.can_cast(NPTypeCode.Int32, NPTypeCode.Single, "same_kind").Should().BeFalse();
    }

    [TestMethod]
    public async Task CanCast_UnsafeMode_AllowsAnything()
    {
        np.can_cast(NPTypeCode.Double, NPTypeCode.Int32, "unsafe").Should().BeTrue();
        np.can_cast(NPTypeCode.Int64, NPTypeCode.Byte, "unsafe").Should().BeTrue();
    }

    [TestMethod]
    public async Task CanCast_InvalidMode_Throws()
    {
        Microsoft.VisualStudio.TestTools.UnitTesting.Assert.ThrowsException<Exception>(() => np.can_cast(NPTypeCode.Int32, NPTypeCode.Int64, "invalid"));
    }

    #endregion

    #region Scalar Value Tests

    [TestMethod]
    public async Task CanCast_IntScalar_FitsInByte()
    {
        np.can_cast(0, NPTypeCode.Byte).Should().BeTrue();
        np.can_cast(255, NPTypeCode.Byte).Should().BeTrue();
    }

    [TestMethod]
    public async Task CanCast_IntScalar_ExceedsByte()
    {
        np.can_cast(256, NPTypeCode.Byte).Should().BeFalse();
        np.can_cast(-1, NPTypeCode.Byte).Should().BeFalse();
    }

    [TestMethod]
    public async Task CanCast_BooleanBoundaries()
    {
        np.can_cast(0, NPTypeCode.Boolean).Should().BeTrue();
        np.can_cast(1, NPTypeCode.Boolean).Should().BeTrue();
        np.can_cast(2, NPTypeCode.Boolean).Should().BeFalse();
    }

    [TestMethod]
    public async Task CanCast_LongScalar_FitsInInt32()
    {
        np.can_cast((long)int.MaxValue, NPTypeCode.Int32).Should().BeTrue();
    }

    [TestMethod]
    public async Task CanCast_LongScalar_ExceedsInt32()
    {
        np.can_cast((long)int.MaxValue + 1, NPTypeCode.Int32).Should().BeFalse();
    }

    #endregion

    #region Generic Overload Tests

    [TestMethod]
    public async Task CanCast_Generic_IntToLong()
    {
        np.can_cast<int, long>().Should().BeTrue();
    }

    [TestMethod]
    public async Task CanCast_Generic_LongToInt()
    {
        np.can_cast<long, int>().Should().BeFalse();
    }

    [TestMethod]
    public async Task CanCast_Generic_FloatToDouble()
    {
        np.can_cast<float, double>().Should().BeTrue();
    }

    [TestMethod]
    public async Task CanCast_Generic_SameKind()
    {
        np.can_cast<int, short>("same_kind").Should().BeTrue();
    }

    #endregion

    #region Primitive Type Overloads

    [TestMethod]
    public async Task CanCast_ByteOverload()
    {
        np.can_cast((byte)100, NPTypeCode.Int32).Should().BeTrue();
    }

    [TestMethod]
    public async Task CanCast_ShortOverload()
    {
        np.can_cast((short)-100, NPTypeCode.Byte).Should().BeFalse();
    }

    [TestMethod]
    public async Task CanCast_ULongOverload()
    {
        np.can_cast(ulong.MaxValue, NPTypeCode.Int64).Should().BeFalse();
    }

    [TestMethod]
    public async Task CanCast_FloatOverload()
    {
        np.can_cast(1.0f, NPTypeCode.Double).Should().BeTrue();
    }

    [TestMethod]
    public async Task CanCast_BoolOverload()
    {
        np.can_cast(true, NPTypeCode.Int32).Should().BeTrue();
    }

    #endregion

    #region NDArray Tests

    [TestMethod]
    public async Task CanCast_NDArray_ToLargerType()
    {
        var arr = np.array(new int[] { 1, 2, 3 });
        np.can_cast(arr, NPTypeCode.Int64).Should().BeTrue();
    }

    [TestMethod]
    public async Task CanCast_NDArray_ToSmallerType()
    {
        var arr = np.array(new int[] { 1, 2, 3 });
        np.can_cast(arr, NPTypeCode.Int16).Should().BeFalse();
    }

    #endregion

    #region Type Overload Tests

    [TestMethod]
    public async Task CanCast_Type_IntToLong()
    {
        np.can_cast(typeof(int), typeof(long)).Should().BeTrue();
    }

    [TestMethod]
    public async Task CanCast_Type_LongToInt()
    {
        np.can_cast(typeof(long), typeof(int)).Should().BeFalse();
    }

    #endregion
}
