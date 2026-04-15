using System;

namespace NumSharp.UnitTest.APIs;

/// <summary>
/// Tests for NumPy type introspection and promotion functions.
/// </summary>
[TestClass]
public class NpTypingTests
{
#region iinfo tests

    [TestMethod]
    public void IInfo_Int32_Bits()
    {
        var info = np.iinfo(NPTypeCode.Int32);
        info.bits.Should().Be(32);
    }

    [TestMethod]
    public void IInfo_Int32_Min()
    {
        var info = np.iinfo(NPTypeCode.Int32);
        info.min.Should().Be(int.MinValue);
    }

    [TestMethod]
    public void IInfo_Int32_Max()
    {
        var info = np.iinfo(NPTypeCode.Int32);
        info.max.Should().Be(int.MaxValue);
    }

    [TestMethod]
    public void IInfo_Int32_Kind()
    {
        var info = np.iinfo(NPTypeCode.Int32);
        info.kind.Should().Be('i');
    }

    [TestMethod]
    public void IInfo_UInt8_Min()
    {
        var info = np.iinfo(NPTypeCode.Byte);
        info.min.Should().Be(0);
    }

    [TestMethod]
    public void IInfo_UInt8_Max()
    {
        var info = np.iinfo(NPTypeCode.Byte);
        info.max.Should().Be(255);
    }

    [TestMethod]
    public void IInfo_UInt8_Kind()
    {
        var info = np.iinfo(NPTypeCode.Byte);
        info.kind.Should().Be('u');
    }

    [TestMethod]
    public void IInfo_Bool_Bits()
    {
        var info = np.iinfo(NPTypeCode.Boolean);
        info.bits.Should().Be(8);
    }

    [TestMethod]
    public void IInfo_Bool_MinMax()
    {
        var info = np.iinfo(NPTypeCode.Boolean);
        info.min.Should().Be(0);
        info.max.Should().Be(1);
    }

    [TestMethod]
    public void IInfo_Float_Throws()
    {
        Assert.Throws<Exception>(() => np.iinfo(NPTypeCode.Double));
    }

    [TestMethod]
    public void IInfo_Int64_Values()
    {
        var info = np.iinfo(NPTypeCode.Int64);
        info.bits.Should().Be(64);
        info.min.Should().Be(long.MinValue);
        info.max.Should().Be(long.MaxValue);
    }

    [TestMethod]
    public void IInfo_UInt64_MaxUnsigned()
    {
        var info = np.iinfo(NPTypeCode.UInt64);
        info.maxUnsigned.Should().Be(ulong.MaxValue);
    }

#endregion

#region finfo tests

    [TestMethod]
    public void FInfo_Float64_Bits()
    {
        var info = np.finfo(NPTypeCode.Double);
        info.bits.Should().Be(64);
    }

    [TestMethod]
    public void FInfo_Float64_Precision()
    {
        var info = np.finfo(NPTypeCode.Double);
        info.precision.Should().Be(15);
    }

    [TestMethod]
    public void FInfo_Float64_Eps()
    {
        var info = np.finfo(NPTypeCode.Double);
        // eps should be approximately 2.22e-16
        info.eps.Should().BeGreaterThan(2e-16);
        info.eps.Should().BeLessThan(3e-16);
    }

    [TestMethod]
    public void FInfo_Float32_Bits()
    {
        var info = np.finfo(NPTypeCode.Single);
        info.bits.Should().Be(32);
    }

    [TestMethod]
    public void FInfo_Float32_Precision()
    {
        var info = np.finfo(NPTypeCode.Single);
        info.precision.Should().Be(6);
    }

    [TestMethod]
    public void FInfo_Int_Throws()
    {
        Assert.Throws<Exception>(() => np.finfo(NPTypeCode.Int32));
    }

    [TestMethod]
    public void FInfo_Decimal()
    {
        var info = np.finfo(NPTypeCode.Decimal);
        info.bits.Should().Be(128);
        info.precision.Should().Be(28);
    }

#endregion

#region can_cast tests

    [TestMethod]
    public void CanCast_Int32ToInt64_Safe()
    {
        np.can_cast(NPTypeCode.Int32, NPTypeCode.Int64).Should().BeTrue();
    }

    [TestMethod]
    public void CanCast_Int64ToInt32_Safe()
    {
        np.can_cast(NPTypeCode.Int64, NPTypeCode.Int32).Should().BeFalse();
    }

    [TestMethod]
    public void CanCast_Int32ToFloat64_Safe()
    {
        np.can_cast(NPTypeCode.Int32, NPTypeCode.Double).Should().BeTrue();
    }

    [TestMethod]
    public void CanCast_Float32ToFloat64_Safe()
    {
        np.can_cast(NPTypeCode.Single, NPTypeCode.Double).Should().BeTrue();
    }

    [TestMethod]
    public void CanCast_Float64ToInt32_Safe()
    {
        np.can_cast(NPTypeCode.Double, NPTypeCode.Int32).Should().BeFalse();
    }

    [TestMethod]
    public void CanCast_Int32ToInt16_SameKind()
    {
        np.can_cast(NPTypeCode.Int32, NPTypeCode.Int16, "same_kind").Should().BeTrue();
    }

    [TestMethod]
    public void CanCast_Int32ToFloat32_SameKind()
    {
        // Int to float is NOT same_kind - different type kinds
        np.can_cast(NPTypeCode.Int32, NPTypeCode.Single, "same_kind").Should().BeFalse();
    }

    [TestMethod]
    public void CanCast_Int32ToInt16_Unsafe()
    {
        np.can_cast(NPTypeCode.Int32, NPTypeCode.Int16, "unsafe").Should().BeTrue();
    }

    [TestMethod]
    public void CanCast_ScalarFits()
    {
        np.can_cast(100, NPTypeCode.Byte).Should().BeTrue();
    }

    [TestMethod]
    public void CanCast_ScalarOverflow()
    {
        np.can_cast(1000, NPTypeCode.Byte).Should().BeFalse();
    }

    [TestMethod]
    public void CanCast_BoolToInt()
    {
        np.can_cast(NPTypeCode.Boolean, NPTypeCode.Int32).Should().BeTrue();
    }

#endregion

#region result_type tests

    [TestMethod]
    public void ResultType_Int32Int64()
    {
        np.result_type(NPTypeCode.Int32, NPTypeCode.Int64).Should().Be(NPTypeCode.Int64);
    }

    [TestMethod]
    public void ResultType_Int32Float32()
    {
        // Mixed int/float promotes to higher precision
        var result = np.result_type(NPTypeCode.Int32, NPTypeCode.Single);
        // Result should be single or double - we accept either
        (result == NPTypeCode.Single || result == NPTypeCode.Double).Should().BeTrue();
    }

    [TestMethod]
    public void ResultType_Float32Float64()
    {
        np.result_type(NPTypeCode.Single, NPTypeCode.Double).Should().Be(NPTypeCode.Double);
    }

    [TestMethod]
    public void ResultType_Arrays()
    {
        var a = np.array(new int[] { 1, 2 });
        var b = np.array(new float[] { 1.0f, 2.0f });
        var result = np.result_type(a, b);
        // Result should be single or double - we accept either
        (result == NPTypeCode.Single || result == NPTypeCode.Double).Should().BeTrue();
    }

#endregion

#region promote_types tests

    [TestMethod]
    public void PromoteTypes_SameType()
    {
        np.promote_types(NPTypeCode.Int32, NPTypeCode.Int32).Should().Be(NPTypeCode.Int32);
    }

    [TestMethod]
    public void PromoteTypes_Int16Int32()
    {
        np.promote_types(NPTypeCode.Int16, NPTypeCode.Int32).Should().Be(NPTypeCode.Int32);
    }

    [TestMethod]
    public void PromoteTypes_Float32Float64()
    {
        np.promote_types(NPTypeCode.Single, NPTypeCode.Double).Should().Be(NPTypeCode.Double);
    }

#endregion

#region min_scalar_type tests

    [TestMethod]
    public void MinScalarType_SmallPositive()
    {
        np.min_scalar_type(10).Should().Be(NPTypeCode.Byte);
    }

    [TestMethod]
    public void MinScalarType_SmallNegative()
    {
        // No Int8 in NumSharp, so smallest signed is Int16
        np.min_scalar_type(-10).Should().Be(NPTypeCode.Int16);
    }

    [TestMethod]
    public void MinScalarType_Large()
    {
        np.min_scalar_type(100000).Should().Be(NPTypeCode.UInt32);
    }

    [TestMethod]
    public void MinScalarType_Bool()
    {
        np.min_scalar_type(true).Should().Be(NPTypeCode.Boolean);
    }

    [TestMethod]
    public void MinScalarType_ByteRange()
    {
        np.min_scalar_type(255).Should().Be(NPTypeCode.Byte);
        np.min_scalar_type(256).Should().Be(NPTypeCode.UInt16);
    }

#endregion

#region issubdtype tests

    [TestMethod]
    public void IsSubdtype_Int32Integer()
    {
        np.issubdtype(NPTypeCode.Int32, "integer").Should().BeTrue();
    }

    [TestMethod]
    public void IsSubdtype_Int32SignedInteger()
    {
        np.issubdtype(NPTypeCode.Int32, "signedinteger").Should().BeTrue();
    }

    [TestMethod]
    public void IsSubdtype_UInt32UnsignedInteger()
    {
        np.issubdtype(NPTypeCode.UInt32, "unsignedinteger").Should().BeTrue();
    }

    [TestMethod]
    public void IsSubdtype_Int32Floating()
    {
        np.issubdtype(NPTypeCode.Int32, "floating").Should().BeFalse();
    }

    [TestMethod]
    public void IsSubdtype_Float64Number()
    {
        np.issubdtype(NPTypeCode.Double, "number").Should().BeTrue();
    }

    [TestMethod]
    public void IsSubdtype_Float64Floating()
    {
        np.issubdtype(NPTypeCode.Double, "floating").Should().BeTrue();
    }

    [TestMethod]
    public void IsSubdtype_Float64Inexact()
    {
        np.issubdtype(NPTypeCode.Double, "inexact").Should().BeTrue();
    }

    [TestMethod]
    public void IsSubdtype_BoolInteger_NumPy2x()
    {
        // In NumPy 2.x, bool is NOT a subtype of integer
        np.issubdtype(NPTypeCode.Boolean, "integer").Should().BeFalse();
    }

    [TestMethod]
    public void IsSubdtype_BoolGeneric()
    {
        np.issubdtype(NPTypeCode.Boolean, "generic").Should().BeTrue();
    }

#endregion

#region common_type tests

    [TestMethod]
    public void CommonType_Int32ReturnsDouble()
    {
        var a = np.array(new int[] { 1, 2 });
        np.common_type_code(a).Should().Be(NPTypeCode.Double);
    }

    [TestMethod]
    public void CommonType_Float32Float64()
    {
        var a = np.array(new float[] { 1.0f });
        var b = np.array(new double[] { 1.0 });
        np.common_type_code(a, b).Should().Be(NPTypeCode.Double);
    }

    [TestMethod]
    public void CommonType_Float32Only()
    {
        var a = np.array(new float[] { 1.0f });
        np.common_type_code(a).Should().Be(NPTypeCode.Single);
    }

#endregion

#region Type checking functions tests

    [TestMethod]
    public void IsSctype_Int32()
    {
        np.issctype(typeof(int)).Should().BeTrue();
    }

    [TestMethod]
    public void IsSctype_NPTypeCode()
    {
        np.issctype(NPTypeCode.Int32).Should().BeTrue();
    }

    [TestMethod]
    public void IsDtype_Int32Integral()
    {
        np.isdtype(NPTypeCode.Int32, "integral").Should().BeTrue();
    }

    [TestMethod]
    public void IsDtype_Float64RealFloating()
    {
        np.isdtype(NPTypeCode.Double, "real floating").Should().BeTrue();
    }

    [TestMethod]
    public void IsDtype_Int32Numeric()
    {
        np.isdtype(NPTypeCode.Int32, "numeric").Should().BeTrue();
    }

    [TestMethod]
    public void Sctype2Char_Int32()
    {
        np.sctype2char(NPTypeCode.Int32).Should().Be('i');
    }

    [TestMethod]
    public void Sctype2Char_Double()
    {
        np.sctype2char(NPTypeCode.Double).Should().Be('d');
    }

    [TestMethod]
    public void Sctype2Char_UInt8()
    {
        np.sctype2char(NPTypeCode.Byte).Should().Be('B');
    }

    [TestMethod]
    public void MaximumSctype_Int32()
    {
        np.maximum_sctype(NPTypeCode.Int32).Should().Be(NPTypeCode.Int64);
    }

    [TestMethod]
    public void MaximumSctype_UInt16()
    {
        np.maximum_sctype(NPTypeCode.UInt16).Should().Be(NPTypeCode.UInt64);
    }

    [TestMethod]
    public void MaximumSctype_Float32()
    {
        np.maximum_sctype(NPTypeCode.Single).Should().Be(NPTypeCode.Double);
    }

#endregion

#region isreal/iscomplex tests

    [TestMethod]
    public void IsReal_IntArray()
    {
        var a = np.array(new int[] { 1, 2, 3 });
        var result = np.isreal(a);
        // All elements should be real for non-complex arrays
        result.GetBoolean(0).Should().BeTrue();
        result.GetBoolean(1).Should().BeTrue();
        result.GetBoolean(2).Should().BeTrue();
    }

    [TestMethod]
    public void IsComplex_IntArray()
    {
        var a = np.array(new int[] { 1, 2, 3 });
        var result = np.iscomplex(a);
        // No elements should be complex for non-complex arrays
        result.GetBoolean(0).Should().BeFalse();
        result.GetBoolean(1).Should().BeFalse();
        result.GetBoolean(2).Should().BeFalse();
    }

    [TestMethod]
    public void IsRealObj_IntArray()
    {
        var a = np.array(new int[] { 1, 2 });
        np.isrealobj(a).Should().BeTrue();
    }

    [TestMethod]
    public void IsComplexObj_IntArray()
    {
        var a = np.array(new int[] { 1, 2 });
        np.iscomplexobj(a).Should().BeFalse();
    }

    [TestMethod]
    public void IsRealObj_FloatArray()
    {
        var a = np.array(new double[] { 1.0, 2.0 });
        np.isrealobj(a).Should().BeTrue();
    }

#endregion

#region NPTypeCode extension tests

    [TestMethod]
    public void NPTypeCode_IsFloatingPoint()
    {
        NPTypeCode.Double.IsFloatingPoint().Should().BeTrue();
        NPTypeCode.Single.IsFloatingPoint().Should().BeTrue();
        NPTypeCode.Decimal.IsFloatingPoint().Should().BeTrue();
        NPTypeCode.Int32.IsFloatingPoint().Should().BeFalse();
    }

    [TestMethod]
    public void NPTypeCode_IsInteger()
    {
        NPTypeCode.Int32.IsInteger().Should().BeTrue();
        NPTypeCode.UInt64.IsInteger().Should().BeTrue();
        NPTypeCode.Double.IsInteger().Should().BeFalse();
        NPTypeCode.Boolean.IsInteger().Should().BeFalse();
    }

    [TestMethod]
    public void NPTypeCode_IsSimdCapable()
    {
        NPTypeCode.Int32.IsSimdCapable().Should().BeTrue();
        NPTypeCode.Double.IsSimdCapable().Should().BeTrue();
        NPTypeCode.Decimal.IsSimdCapable().Should().BeFalse();
        NPTypeCode.Boolean.IsSimdCapable().Should().BeFalse();
    }

    [TestMethod]
    public void NPTypeCode_GetOneValue()
    {
        ((int)NPTypeCode.Int32.GetOneValue()).Should().Be(1);
        ((double)NPTypeCode.Double.GetOneValue()).Should().Be(1.0);
        ((bool)NPTypeCode.Boolean.GetOneValue()).Should().BeTrue();
    }

#endregion
}
