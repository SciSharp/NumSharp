using System;
using System.Threading.Tasks;
using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace NumSharp.UnitTest.APIs;

/// <summary>
/// Battle tests for np.can_cast - comprehensive coverage of type casting rules.
/// </summary>
public class NpCanCastBattleTests
{
    #region Safe Casting - Type to Type

    [Test]
    public async Task CanCast_SameType_AlwaysTrue()
    {
        await Assert.That(np.can_cast(NPTypeCode.Int32, NPTypeCode.Int32)).IsTrue();
        await Assert.That(np.can_cast(NPTypeCode.Double, NPTypeCode.Double)).IsTrue();
        await Assert.That(np.can_cast(NPTypeCode.Boolean, NPTypeCode.Boolean)).IsTrue();
    }

    [Test]
    public async Task CanCast_BoolToIntegers_Safe()
    {
        await Assert.That(np.can_cast(NPTypeCode.Boolean, NPTypeCode.Byte)).IsTrue();
        await Assert.That(np.can_cast(NPTypeCode.Boolean, NPTypeCode.Int32)).IsTrue();
        await Assert.That(np.can_cast(NPTypeCode.Boolean, NPTypeCode.Int64)).IsTrue();
    }

    [Test]
    public async Task CanCast_BoolToFloats_Safe()
    {
        await Assert.That(np.can_cast(NPTypeCode.Boolean, NPTypeCode.Single)).IsTrue();
        await Assert.That(np.can_cast(NPTypeCode.Boolean, NPTypeCode.Double)).IsTrue();
    }

    [Test]
    public async Task CanCast_ByteUpcast_Safe()
    {
        await Assert.That(np.can_cast(NPTypeCode.Byte, NPTypeCode.Int16)).IsTrue();
        await Assert.That(np.can_cast(NPTypeCode.Byte, NPTypeCode.Int32)).IsTrue();
        await Assert.That(np.can_cast(NPTypeCode.Byte, NPTypeCode.Double)).IsTrue();
    }

    [Test]
    public async Task CanCast_Int32ToInt64_Safe()
    {
        await Assert.That(np.can_cast(NPTypeCode.Int32, NPTypeCode.Int64)).IsTrue();
    }

    [Test]
    public async Task CanCast_Int64ToInt32_NotSafe()
    {
        await Assert.That(np.can_cast(NPTypeCode.Int64, NPTypeCode.Int32)).IsFalse();
    }

    [Test]
    public async Task CanCast_Float32ToFloat64_Safe()
    {
        await Assert.That(np.can_cast(NPTypeCode.Single, NPTypeCode.Double)).IsTrue();
    }

    [Test]
    public async Task CanCast_Float64ToFloat32_NotSafe()
    {
        await Assert.That(np.can_cast(NPTypeCode.Double, NPTypeCode.Single)).IsFalse();
    }

    [Test]
    public async Task CanCast_FloatToInt_NotSafe()
    {
        await Assert.That(np.can_cast(NPTypeCode.Double, NPTypeCode.Int32)).IsFalse();
    }

    #endregion

    #region Casting Modes

    [Test]
    public async Task CanCast_NoMode_OnlySameType()
    {
        await Assert.That(np.can_cast(NPTypeCode.Int32, NPTypeCode.Int32, "no")).IsTrue();
        await Assert.That(np.can_cast(NPTypeCode.Int32, NPTypeCode.Int64, "no")).IsFalse();
    }

    [Test]
    public async Task CanCast_SameKindMode_AllowsDowncastWithinKind()
    {
        await Assert.That(np.can_cast(NPTypeCode.Int32, NPTypeCode.Int16, "same_kind")).IsTrue();
        await Assert.That(np.can_cast(NPTypeCode.Double, NPTypeCode.Single, "same_kind")).IsTrue();
    }

    [Test]
    public async Task CanCast_SameKindMode_RejectsCrossKind()
    {
        await Assert.That(np.can_cast(NPTypeCode.Int32, NPTypeCode.Single, "same_kind")).IsFalse();
    }

    [Test]
    public async Task CanCast_UnsafeMode_AllowsAnything()
    {
        await Assert.That(np.can_cast(NPTypeCode.Double, NPTypeCode.Int32, "unsafe")).IsTrue();
        await Assert.That(np.can_cast(NPTypeCode.Int64, NPTypeCode.Byte, "unsafe")).IsTrue();
    }

    [Test]
    public async Task CanCast_InvalidMode_Throws()
    {
        await Assert.That(() => np.can_cast(NPTypeCode.Int32, NPTypeCode.Int64, "invalid")).ThrowsException();
    }

    #endregion

    #region Scalar Value Tests

    [Test]
    public async Task CanCast_IntScalar_FitsInByte()
    {
        await Assert.That(np.can_cast(0, NPTypeCode.Byte)).IsTrue();
        await Assert.That(np.can_cast(255, NPTypeCode.Byte)).IsTrue();
    }

    [Test]
    public async Task CanCast_IntScalar_ExceedsByte()
    {
        await Assert.That(np.can_cast(256, NPTypeCode.Byte)).IsFalse();
        await Assert.That(np.can_cast(-1, NPTypeCode.Byte)).IsFalse();
    }

    [Test]
    public async Task CanCast_BooleanBoundaries()
    {
        await Assert.That(np.can_cast(0, NPTypeCode.Boolean)).IsTrue();
        await Assert.That(np.can_cast(1, NPTypeCode.Boolean)).IsTrue();
        await Assert.That(np.can_cast(2, NPTypeCode.Boolean)).IsFalse();
    }

    [Test]
    public async Task CanCast_LongScalar_FitsInInt32()
    {
        await Assert.That(np.can_cast((long)int.MaxValue, NPTypeCode.Int32)).IsTrue();
    }

    [Test]
    public async Task CanCast_LongScalar_ExceedsInt32()
    {
        await Assert.That(np.can_cast((long)int.MaxValue + 1, NPTypeCode.Int32)).IsFalse();
    }

    #endregion

    #region Generic Overload Tests

    [Test]
    public async Task CanCast_Generic_IntToLong()
    {
        await Assert.That(np.can_cast<int, long>()).IsTrue();
    }

    [Test]
    public async Task CanCast_Generic_LongToInt()
    {
        await Assert.That(np.can_cast<long, int>()).IsFalse();
    }

    [Test]
    public async Task CanCast_Generic_FloatToDouble()
    {
        await Assert.That(np.can_cast<float, double>()).IsTrue();
    }

    [Test]
    public async Task CanCast_Generic_SameKind()
    {
        await Assert.That(np.can_cast<int, short>("same_kind")).IsTrue();
    }

    #endregion

    #region Primitive Type Overloads

    [Test]
    public async Task CanCast_ByteOverload()
    {
        await Assert.That(np.can_cast((byte)100, NPTypeCode.Int32)).IsTrue();
    }

    [Test]
    public async Task CanCast_ShortOverload()
    {
        await Assert.That(np.can_cast((short)-100, NPTypeCode.Byte)).IsFalse();
    }

    [Test]
    public async Task CanCast_ULongOverload()
    {
        await Assert.That(np.can_cast(ulong.MaxValue, NPTypeCode.Int64)).IsFalse();
    }

    [Test]
    public async Task CanCast_FloatOverload()
    {
        await Assert.That(np.can_cast(1.0f, NPTypeCode.Double)).IsTrue();
    }

    [Test]
    public async Task CanCast_BoolOverload()
    {
        await Assert.That(np.can_cast(true, NPTypeCode.Int32)).IsTrue();
    }

    #endregion

    #region NDArray Tests

    [Test]
    public async Task CanCast_NDArray_ToLargerType()
    {
        var arr = np.array(new int[] { 1, 2, 3 });
        await Assert.That(np.can_cast(arr, NPTypeCode.Int64)).IsTrue();
    }

    [Test]
    public async Task CanCast_NDArray_ToSmallerType()
    {
        var arr = np.array(new int[] { 1, 2, 3 });
        await Assert.That(np.can_cast(arr, NPTypeCode.Int16)).IsFalse();
    }

    #endregion

    #region Type Overload Tests

    [Test]
    public async Task CanCast_Type_IntToLong()
    {
        await Assert.That(np.can_cast(typeof(int), typeof(long))).IsTrue();
    }

    [Test]
    public async Task CanCast_Type_LongToInt()
    {
        await Assert.That(np.can_cast(typeof(long), typeof(int))).IsFalse();
    }

    #endregion
}
