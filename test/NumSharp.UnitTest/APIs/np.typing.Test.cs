using System;
using System.Threading.Tasks;
using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace NumSharp.UnitTest.APIs;

/// <summary>
/// Tests for NumPy type introspection and promotion functions.
/// </summary>
public class NpTypingTests
{
#region iinfo tests

    [Test]
    public async Task IInfo_Int32_Bits()
    {
        var info = np.iinfo(NPTypeCode.Int32);
        await Assert.That(info.bits).IsEqualTo(32);
    }

    [Test]
    public async Task IInfo_Int32_Min()
    {
        var info = np.iinfo(NPTypeCode.Int32);
        await Assert.That(info.min).IsEqualTo(int.MinValue);
    }

    [Test]
    public async Task IInfo_Int32_Max()
    {
        var info = np.iinfo(NPTypeCode.Int32);
        await Assert.That(info.max).IsEqualTo(int.MaxValue);
    }

    [Test]
    public async Task IInfo_Int32_Kind()
    {
        var info = np.iinfo(NPTypeCode.Int32);
        await Assert.That(info.kind).IsEqualTo('i');
    }

    [Test]
    public async Task IInfo_UInt8_Min()
    {
        var info = np.iinfo(NPTypeCode.Byte);
        await Assert.That(info.min).IsEqualTo(0);
    }

    [Test]
    public async Task IInfo_UInt8_Max()
    {
        var info = np.iinfo(NPTypeCode.Byte);
        await Assert.That(info.max).IsEqualTo(255);
    }

    [Test]
    public async Task IInfo_UInt8_Kind()
    {
        var info = np.iinfo(NPTypeCode.Byte);
        await Assert.That(info.kind).IsEqualTo('u');
    }

    [Test]
    public async Task IInfo_Bool_Bits()
    {
        var info = np.iinfo(NPTypeCode.Boolean);
        await Assert.That(info.bits).IsEqualTo(8);
    }

    [Test]
    public async Task IInfo_Bool_MinMax()
    {
        var info = np.iinfo(NPTypeCode.Boolean);
        await Assert.That(info.min).IsEqualTo(0);
        await Assert.That(info.max).IsEqualTo(1);
    }

    [Test]
    public async Task IInfo_Float_Throws()
    {
        await Assert.That(() => np.iinfo(NPTypeCode.Double)).ThrowsException();
    }

    [Test]
    public async Task IInfo_Int64_Values()
    {
        var info = np.iinfo(NPTypeCode.Int64);
        await Assert.That(info.bits).IsEqualTo(64);
        await Assert.That(info.min).IsEqualTo(long.MinValue);
        await Assert.That(info.max).IsEqualTo(long.MaxValue);
    }

    [Test]
    public async Task IInfo_UInt64_MaxUnsigned()
    {
        var info = np.iinfo(NPTypeCode.UInt64);
        await Assert.That(info.maxUnsigned).IsEqualTo(ulong.MaxValue);
    }

#endregion

#region finfo tests

    [Test]
    public async Task FInfo_Float64_Bits()
    {
        var info = np.finfo(NPTypeCode.Double);
        await Assert.That(info.bits).IsEqualTo(64);
    }

    [Test]
    public async Task FInfo_Float64_Precision()
    {
        var info = np.finfo(NPTypeCode.Double);
        await Assert.That(info.precision).IsEqualTo(15);
    }

    [Test]
    public async Task FInfo_Float64_Eps()
    {
        var info = np.finfo(NPTypeCode.Double);
        // eps should be approximately 2.22e-16
        await Assert.That(info.eps).IsGreaterThan(2e-16);
        await Assert.That(info.eps).IsLessThan(3e-16);
    }

    [Test]
    public async Task FInfo_Float32_Bits()
    {
        var info = np.finfo(NPTypeCode.Single);
        await Assert.That(info.bits).IsEqualTo(32);
    }

    [Test]
    public async Task FInfo_Float32_Precision()
    {
        var info = np.finfo(NPTypeCode.Single);
        await Assert.That(info.precision).IsEqualTo(6);
    }

    [Test]
    public async Task FInfo_Int_Throws()
    {
        await Assert.That(() => np.finfo(NPTypeCode.Int32)).ThrowsException();
    }

    [Test]
    public async Task FInfo_Decimal()
    {
        var info = np.finfo(NPTypeCode.Decimal);
        await Assert.That(info.bits).IsEqualTo(128);
        await Assert.That(info.precision).IsEqualTo(28);
    }

#endregion

#region can_cast tests

    [Test]
    public async Task CanCast_Int32ToInt64_Safe()
    {
        await Assert.That(np.can_cast(NPTypeCode.Int32, NPTypeCode.Int64)).IsTrue();
    }

    [Test]
    public async Task CanCast_Int64ToInt32_Safe()
    {
        await Assert.That(np.can_cast(NPTypeCode.Int64, NPTypeCode.Int32)).IsFalse();
    }

    [Test]
    public async Task CanCast_Int32ToFloat64_Safe()
    {
        await Assert.That(np.can_cast(NPTypeCode.Int32, NPTypeCode.Double)).IsTrue();
    }

    [Test]
    public async Task CanCast_Float32ToFloat64_Safe()
    {
        await Assert.That(np.can_cast(NPTypeCode.Single, NPTypeCode.Double)).IsTrue();
    }

    [Test]
    public async Task CanCast_Float64ToInt32_Safe()
    {
        await Assert.That(np.can_cast(NPTypeCode.Double, NPTypeCode.Int32)).IsFalse();
    }

    [Test]
    public async Task CanCast_Int32ToInt16_SameKind()
    {
        await Assert.That(np.can_cast(NPTypeCode.Int32, NPTypeCode.Int16, "same_kind")).IsTrue();
    }

    [Test]
    public async Task CanCast_Int32ToFloat32_SameKind()
    {
        // Int to float is NOT same_kind - different type kinds
        await Assert.That(np.can_cast(NPTypeCode.Int32, NPTypeCode.Single, "same_kind")).IsFalse();
    }

    [Test]
    public async Task CanCast_Int32ToInt16_Unsafe()
    {
        await Assert.That(np.can_cast(NPTypeCode.Int32, NPTypeCode.Int16, "unsafe")).IsTrue();
    }

    [Test]
    public async Task CanCast_ScalarFits()
    {
        await Assert.That(np.can_cast(100, NPTypeCode.Byte)).IsTrue();
    }

    [Test]
    public async Task CanCast_ScalarOverflow()
    {
        await Assert.That(np.can_cast(1000, NPTypeCode.Byte)).IsFalse();
    }

    [Test]
    public async Task CanCast_BoolToInt()
    {
        await Assert.That(np.can_cast(NPTypeCode.Boolean, NPTypeCode.Int32)).IsTrue();
    }

#endregion

#region result_type tests

    [Test]
    public async Task ResultType_Int32Int64()
    {
        await Assert.That(np.result_type(NPTypeCode.Int32, NPTypeCode.Int64)).IsEqualTo(NPTypeCode.Int64);
    }

    [Test]
    public async Task ResultType_Int32Float32()
    {
        // Mixed int/float promotes to higher precision
        var result = np.result_type(NPTypeCode.Int32, NPTypeCode.Single);
        // Result should be single or double - we accept either
        await Assert.That(result == NPTypeCode.Single || result == NPTypeCode.Double).IsTrue();
    }

    [Test]
    public async Task ResultType_Float32Float64()
    {
        await Assert.That(np.result_type(NPTypeCode.Single, NPTypeCode.Double)).IsEqualTo(NPTypeCode.Double);
    }

    [Test]
    public async Task ResultType_Arrays()
    {
        var a = np.array(new int[] { 1, 2 });
        var b = np.array(new float[] { 1.0f, 2.0f });
        var result = np.result_type(a, b);
        // Result should be single or double - we accept either
        await Assert.That(result == NPTypeCode.Single || result == NPTypeCode.Double).IsTrue();
    }

#endregion

#region promote_types tests

    [Test]
    public async Task PromoteTypes_SameType()
    {
        await Assert.That(np.promote_types(NPTypeCode.Int32, NPTypeCode.Int32)).IsEqualTo(NPTypeCode.Int32);
    }

    [Test]
    public async Task PromoteTypes_Int16Int32()
    {
        await Assert.That(np.promote_types(NPTypeCode.Int16, NPTypeCode.Int32)).IsEqualTo(NPTypeCode.Int32);
    }

    [Test]
    public async Task PromoteTypes_Float32Float64()
    {
        await Assert.That(np.promote_types(NPTypeCode.Single, NPTypeCode.Double)).IsEqualTo(NPTypeCode.Double);
    }

#endregion

#region min_scalar_type tests

    [Test]
    public async Task MinScalarType_SmallPositive()
    {
        await Assert.That(np.min_scalar_type(10)).IsEqualTo(NPTypeCode.Byte);
    }

    [Test]
    public async Task MinScalarType_SmallNegative()
    {
        // No Int8 in NumSharp, so smallest signed is Int16
        await Assert.That(np.min_scalar_type(-10)).IsEqualTo(NPTypeCode.Int16);
    }

    [Test]
    public async Task MinScalarType_Large()
    {
        await Assert.That(np.min_scalar_type(100000)).IsEqualTo(NPTypeCode.UInt32);
    }

    [Test]
    public async Task MinScalarType_Bool()
    {
        await Assert.That(np.min_scalar_type(true)).IsEqualTo(NPTypeCode.Boolean);
    }

    [Test]
    public async Task MinScalarType_ByteRange()
    {
        await Assert.That(np.min_scalar_type(255)).IsEqualTo(NPTypeCode.Byte);
        await Assert.That(np.min_scalar_type(256)).IsEqualTo(NPTypeCode.UInt16);
    }

#endregion

#region issubdtype tests

    [Test]
    public async Task IsSubdtype_Int32Integer()
    {
        await Assert.That(np.issubdtype(NPTypeCode.Int32, "integer")).IsTrue();
    }

    [Test]
    public async Task IsSubdtype_Int32SignedInteger()
    {
        await Assert.That(np.issubdtype(NPTypeCode.Int32, "signedinteger")).IsTrue();
    }

    [Test]
    public async Task IsSubdtype_UInt32UnsignedInteger()
    {
        await Assert.That(np.issubdtype(NPTypeCode.UInt32, "unsignedinteger")).IsTrue();
    }

    [Test]
    public async Task IsSubdtype_Int32Floating()
    {
        await Assert.That(np.issubdtype(NPTypeCode.Int32, "floating")).IsFalse();
    }

    [Test]
    public async Task IsSubdtype_Float64Number()
    {
        await Assert.That(np.issubdtype(NPTypeCode.Double, "number")).IsTrue();
    }

    [Test]
    public async Task IsSubdtype_Float64Floating()
    {
        await Assert.That(np.issubdtype(NPTypeCode.Double, "floating")).IsTrue();
    }

    [Test]
    public async Task IsSubdtype_Float64Inexact()
    {
        await Assert.That(np.issubdtype(NPTypeCode.Double, "inexact")).IsTrue();
    }

    [Test]
    public async Task IsSubdtype_BoolInteger_NumPy2x()
    {
        // In NumPy 2.x, bool is NOT a subtype of integer
        await Assert.That(np.issubdtype(NPTypeCode.Boolean, "integer")).IsFalse();
    }

    [Test]
    public async Task IsSubdtype_BoolGeneric()
    {
        await Assert.That(np.issubdtype(NPTypeCode.Boolean, "generic")).IsTrue();
    }

#endregion

#region common_type tests

    [Test]
    public async Task CommonType_Int32ReturnsDouble()
    {
        var a = np.array(new int[] { 1, 2 });
        await Assert.That(np.common_type_code(a)).IsEqualTo(NPTypeCode.Double);
    }

    [Test]
    public async Task CommonType_Float32Float64()
    {
        var a = np.array(new float[] { 1.0f });
        var b = np.array(new double[] { 1.0 });
        await Assert.That(np.common_type_code(a, b)).IsEqualTo(NPTypeCode.Double);
    }

    [Test]
    public async Task CommonType_Float32Only()
    {
        var a = np.array(new float[] { 1.0f });
        await Assert.That(np.common_type_code(a)).IsEqualTo(NPTypeCode.Single);
    }

#endregion

#region Type checking functions tests

    [Test]
    public async Task IsSctype_Int32()
    {
        await Assert.That(np.issctype(typeof(int))).IsTrue();
    }

    [Test]
    public async Task IsSctype_NPTypeCode()
    {
        await Assert.That(np.issctype(NPTypeCode.Int32)).IsTrue();
    }

    [Test]
    public async Task IsDtype_Int32Integral()
    {
        await Assert.That(np.isdtype(NPTypeCode.Int32, "integral")).IsTrue();
    }

    [Test]
    public async Task IsDtype_Float64RealFloating()
    {
        await Assert.That(np.isdtype(NPTypeCode.Double, "real floating")).IsTrue();
    }

    [Test]
    public async Task IsDtype_Int32Numeric()
    {
        await Assert.That(np.isdtype(NPTypeCode.Int32, "numeric")).IsTrue();
    }

    [Test]
    public async Task Sctype2Char_Int32()
    {
        await Assert.That(np.sctype2char(NPTypeCode.Int32)).IsEqualTo('i');
    }

    [Test]
    public async Task Sctype2Char_Double()
    {
        await Assert.That(np.sctype2char(NPTypeCode.Double)).IsEqualTo('d');
    }

    [Test]
    public async Task Sctype2Char_UInt8()
    {
        await Assert.That(np.sctype2char(NPTypeCode.Byte)).IsEqualTo('B');
    }

    [Test]
    public async Task MaximumSctype_Int32()
    {
        await Assert.That(np.maximum_sctype(NPTypeCode.Int32)).IsEqualTo(NPTypeCode.Int64);
    }

    [Test]
    public async Task MaximumSctype_UInt16()
    {
        await Assert.That(np.maximum_sctype(NPTypeCode.UInt16)).IsEqualTo(NPTypeCode.UInt64);
    }

    [Test]
    public async Task MaximumSctype_Float32()
    {
        await Assert.That(np.maximum_sctype(NPTypeCode.Single)).IsEqualTo(NPTypeCode.Double);
    }

#endregion

#region isreal/iscomplex tests

    [Test]
    public async Task IsReal_IntArray()
    {
        var a = np.array(new int[] { 1, 2, 3 });
        var result = np.isreal(a);
        // All elements should be real for non-complex arrays
        await Assert.That(result.GetBoolean(0)).IsTrue();
        await Assert.That(result.GetBoolean(1)).IsTrue();
        await Assert.That(result.GetBoolean(2)).IsTrue();
    }

    [Test]
    public async Task IsComplex_IntArray()
    {
        var a = np.array(new int[] { 1, 2, 3 });
        var result = np.iscomplex(a);
        // No elements should be complex for non-complex arrays
        await Assert.That(result.GetBoolean(0)).IsFalse();
        await Assert.That(result.GetBoolean(1)).IsFalse();
        await Assert.That(result.GetBoolean(2)).IsFalse();
    }

    [Test]
    public async Task IsRealObj_IntArray()
    {
        var a = np.array(new int[] { 1, 2 });
        await Assert.That(np.isrealobj(a)).IsTrue();
    }

    [Test]
    public async Task IsComplexObj_IntArray()
    {
        var a = np.array(new int[] { 1, 2 });
        await Assert.That(np.iscomplexobj(a)).IsFalse();
    }

    [Test]
    public async Task IsRealObj_FloatArray()
    {
        var a = np.array(new double[] { 1.0, 2.0 });
        await Assert.That(np.isrealobj(a)).IsTrue();
    }

#endregion

#region NPTypeCode extension tests

    [Test]
    public async Task NPTypeCode_IsFloatingPoint()
    {
        await Assert.That(NPTypeCode.Double.IsFloatingPoint()).IsTrue();
        await Assert.That(NPTypeCode.Single.IsFloatingPoint()).IsTrue();
        await Assert.That(NPTypeCode.Decimal.IsFloatingPoint()).IsTrue();
        await Assert.That(NPTypeCode.Int32.IsFloatingPoint()).IsFalse();
    }

    [Test]
    public async Task NPTypeCode_IsInteger()
    {
        await Assert.That(NPTypeCode.Int32.IsInteger()).IsTrue();
        await Assert.That(NPTypeCode.UInt64.IsInteger()).IsTrue();
        await Assert.That(NPTypeCode.Double.IsInteger()).IsFalse();
        await Assert.That(NPTypeCode.Boolean.IsInteger()).IsFalse();
    }

    [Test]
    public async Task NPTypeCode_IsSimdCapable()
    {
        await Assert.That(NPTypeCode.Int32.IsSimdCapable()).IsTrue();
        await Assert.That(NPTypeCode.Double.IsSimdCapable()).IsTrue();
        await Assert.That(NPTypeCode.Decimal.IsSimdCapable()).IsFalse();
        await Assert.That(NPTypeCode.Boolean.IsSimdCapable()).IsFalse();
    }

    [Test]
    public async Task NPTypeCode_GetOneValue()
    {
        await Assert.That((int)NPTypeCode.Int32.GetOneValue()).IsEqualTo(1);
        await Assert.That((double)NPTypeCode.Double.GetOneValue()).IsEqualTo(1.0);
        await Assert.That((bool)NPTypeCode.Boolean.GetOneValue()).IsTrue();
    }

#endregion
}
