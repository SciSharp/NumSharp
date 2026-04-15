using System.Threading.Tasks;
using NumSharp;

namespace NumSharp.UnitTest.Logic;

/// <summary>
/// Battle tests for NPTypeHierarchy - validates against actual NumPy behavior.
///
/// Ground truth generated from NumPy 2.x:
/// - issubdtype uses issubclass() on the type hierarchy
/// - Bool is NOT under Number in NumPy 2.x (directly under Generic)
/// - Concrete type comparison: only same type returns True
///
/// NumPy hierarchy (from multiarraymodule.c):
///   Generic
///   ├── Bool (NOT under Number!)
///   └── Number
///       ├── Integer
///       │   ├── SignedInteger (int8, int16, int32, int64)
///       │   └── UnsignedInteger (uint8, uint16, uint32, uint64)
///       └── Inexact
///           ├── Floating (float16, float32, float64)
///           └── ComplexFloating (complex64, complex128)
/// </summary>
[TestClass]
public class NPTypeHierarchyBattleTest
{
    #region issubdtype - Type Hierarchy Tests (NumPy verified)

    // ==========================================================================
    // SIGNED INTEGERS: belong to generic, number, integer, signedinteger
    // ==========================================================================

    [TestMethod]
    public async Task Int16_BelongsTo_Generic() =>
        np.issubdtype(NPTypeCode.Int16, "generic").Should().BeTrue();

    [TestMethod]
    public async Task Int16_BelongsTo_Number() =>
        np.issubdtype(NPTypeCode.Int16, "number").Should().BeTrue();

    [TestMethod]
    public async Task Int16_BelongsTo_Integer() =>
        np.issubdtype(NPTypeCode.Int16, "integer").Should().BeTrue();

    [TestMethod]
    public async Task Int16_BelongsTo_SignedInteger() =>
        np.issubdtype(NPTypeCode.Int16, "signedinteger").Should().BeTrue();

    [TestMethod]
    public async Task Int16_NotBelongsTo_UnsignedInteger() =>
        np.issubdtype(NPTypeCode.Int16, "unsignedinteger").Should().BeFalse();

    [TestMethod]
    public async Task Int16_NotBelongsTo_Inexact() =>
        np.issubdtype(NPTypeCode.Int16, "inexact").Should().BeFalse();

    [TestMethod]
    public async Task Int16_NotBelongsTo_Floating() =>
        np.issubdtype(NPTypeCode.Int16, "floating").Should().BeFalse();

    [TestMethod]
    public async Task Int32_BelongsTo_SignedInteger() =>
        np.issubdtype(NPTypeCode.Int32, "signedinteger").Should().BeTrue();

    [TestMethod]
    public async Task Int64_BelongsTo_SignedInteger() =>
        np.issubdtype(NPTypeCode.Int64, "signedinteger").Should().BeTrue();

    // ==========================================================================
    // UNSIGNED INTEGERS: belong to generic, number, integer, unsignedinteger
    // ==========================================================================

    [TestMethod]
    public async Task Byte_BelongsTo_Generic() =>
        np.issubdtype(NPTypeCode.Byte, "generic").Should().BeTrue();

    [TestMethod]
    public async Task Byte_BelongsTo_Number() =>
        np.issubdtype(NPTypeCode.Byte, "number").Should().BeTrue();

    [TestMethod]
    public async Task Byte_BelongsTo_Integer() =>
        np.issubdtype(NPTypeCode.Byte, "integer").Should().BeTrue();

    [TestMethod]
    public async Task Byte_BelongsTo_UnsignedInteger() =>
        np.issubdtype(NPTypeCode.Byte, "unsignedinteger").Should().BeTrue();

    [TestMethod]
    public async Task Byte_NotBelongsTo_SignedInteger() =>
        np.issubdtype(NPTypeCode.Byte, "signedinteger").Should().BeFalse();

    [TestMethod]
    public async Task UInt16_BelongsTo_UnsignedInteger() =>
        np.issubdtype(NPTypeCode.UInt16, "unsignedinteger").Should().BeTrue();

    [TestMethod]
    public async Task UInt32_BelongsTo_UnsignedInteger() =>
        np.issubdtype(NPTypeCode.UInt32, "unsignedinteger").Should().BeTrue();

    [TestMethod]
    public async Task UInt64_BelongsTo_UnsignedInteger() =>
        np.issubdtype(NPTypeCode.UInt64, "unsignedinteger").Should().BeTrue();

    // ==========================================================================
    // FLOATING: belong to generic, number, inexact, floating
    // ==========================================================================

    [TestMethod]
    public async Task Single_BelongsTo_Generic() =>
        np.issubdtype(NPTypeCode.Single, "generic").Should().BeTrue();

    [TestMethod]
    public async Task Single_BelongsTo_Number() =>
        np.issubdtype(NPTypeCode.Single, "number").Should().BeTrue();

    [TestMethod]
    public async Task Single_BelongsTo_Inexact() =>
        np.issubdtype(NPTypeCode.Single, "inexact").Should().BeTrue();

    [TestMethod]
    public async Task Single_BelongsTo_Floating() =>
        np.issubdtype(NPTypeCode.Single, "floating").Should().BeTrue();

    [TestMethod]
    public async Task Single_NotBelongsTo_Integer() =>
        np.issubdtype(NPTypeCode.Single, "integer").Should().BeFalse();

    [TestMethod]
    public async Task Single_NotBelongsTo_ComplexFloating() =>
        np.issubdtype(NPTypeCode.Single, "complexfloating").Should().BeFalse();

    [TestMethod]
    public async Task Double_BelongsTo_Floating() =>
        np.issubdtype(NPTypeCode.Double, "floating").Should().BeTrue();

    [TestMethod]
    public async Task Decimal_BelongsTo_Floating() =>
        np.issubdtype(NPTypeCode.Decimal, "floating").Should().BeTrue();

    // ==========================================================================
    // COMPLEX: belong to generic, number, inexact, complexfloating
    // ==========================================================================

    [TestMethod]
    public async Task Complex_BelongsTo_Generic() =>
        np.issubdtype(NPTypeCode.Complex, "generic").Should().BeTrue();

    [TestMethod]
    public async Task Complex_BelongsTo_Number() =>
        np.issubdtype(NPTypeCode.Complex, "number").Should().BeTrue();

    [TestMethod]
    public async Task Complex_BelongsTo_Inexact() =>
        np.issubdtype(NPTypeCode.Complex, "inexact").Should().BeTrue();

    [TestMethod]
    public async Task Complex_BelongsTo_ComplexFloating() =>
        np.issubdtype(NPTypeCode.Complex, "complexfloating").Should().BeTrue();

    [TestMethod]
    public async Task Complex_NotBelongsTo_Floating() =>
        np.issubdtype(NPTypeCode.Complex, "floating").Should().BeFalse();

    [TestMethod]
    public async Task Complex_NotBelongsTo_Integer() =>
        np.issubdtype(NPTypeCode.Complex, "integer").Should().BeFalse();

    // ==========================================================================
    // BOOLEAN: belongs to generic ONLY (NumPy 2.x critical behavior!)
    // ==========================================================================

    [TestMethod]
    public async Task Bool_BelongsTo_Generic() =>
        np.issubdtype(NPTypeCode.Boolean, "generic").Should().BeTrue();

    [TestMethod]
    public async Task Bool_BelongsTo_Boolean() =>
        np.issubdtype(NPTypeCode.Boolean, "boolean").Should().BeTrue();

    [TestMethod]
    public async Task Bool_NotBelongsTo_Number() =>
        np.issubdtype(NPTypeCode.Boolean, "number").Should().BeFalse();

    [TestMethod]
    public async Task Bool_NotBelongsTo_Integer() =>
        np.issubdtype(NPTypeCode.Boolean, "integer").Should().BeFalse();

    [TestMethod]
    public async Task Bool_NotBelongsTo_SignedInteger() =>
        np.issubdtype(NPTypeCode.Boolean, "signedinteger").Should().BeFalse();

    [TestMethod]
    public async Task Bool_NotBelongsTo_UnsignedInteger() =>
        np.issubdtype(NPTypeCode.Boolean, "unsignedinteger").Should().BeFalse();

    #endregion

    #region issubdtype - Concrete Type Comparison (NumPy verified)

    // NumPy: issubdtype(int32, int64) = False (different concrete types)
    // NumPy: issubdtype(int32, int32) = True (same type)

    [TestMethod]
    public async Task ConcreteType_SameType_ReturnsTrue_Int32()
    {
        np.issubdtype(NPTypeCode.Int32, NPTypeCode.Int32).Should().BeTrue();
    }

    [TestMethod]
    public async Task ConcreteType_SameType_ReturnsTrue_Double()
    {
        np.issubdtype(NPTypeCode.Double, NPTypeCode.Double).Should().BeTrue();
    }

    [TestMethod]
    public async Task ConcreteType_DifferentType_SameKind_ReturnsFalse()
    {
        np.issubdtype(NPTypeCode.Int32, NPTypeCode.Int64).Should().BeFalse();
    }

    [TestMethod]
    public async Task ConcreteType_DifferentType_SameKind_ReturnsFalse_Reverse()
    {
        np.issubdtype(NPTypeCode.Int64, NPTypeCode.Int32).Should().BeFalse();
    }

    [TestMethod]
    public async Task ConcreteType_Float32_Float64_ReturnsFalse()
    {
        np.issubdtype(NPTypeCode.Single, NPTypeCode.Double).Should().BeFalse();
    }

    [TestMethod]
    public async Task ConcreteType_Uint8_Uint64_ReturnsFalse()
    {
        np.issubdtype(NPTypeCode.Byte, NPTypeCode.UInt64).Should().BeFalse();
    }

    [TestMethod]
    public async Task ConcreteType_DifferentKind_ReturnsFalse()
    {
        np.issubdtype(NPTypeCode.Int32, NPTypeCode.Double).Should().BeFalse();
    }

    [TestMethod]
    public async Task ConcreteType_Bool_Int32_ReturnsFalse()
    {
        np.issubdtype(NPTypeCode.Boolean, NPTypeCode.Int32).Should().BeFalse();
    }

    #endregion

    #region isdtype - Category Checks (NumPy 2.0+ verified)

    [TestMethod]
    public async Task Isdtype_Int32_Integral()
    {
        np.isdtype(NPTypeCode.Int32, "integral").Should().BeTrue();
    }

    [TestMethod]
    public async Task Isdtype_Float64_RealFloating()
    {
        np.isdtype(NPTypeCode.Double, "real floating").Should().BeTrue();
    }

    [TestMethod]
    public async Task Isdtype_Complex_ComplexFloating()
    {
        np.isdtype(NPTypeCode.Complex, "complex floating").Should().BeTrue();
    }

    [TestMethod]
    public async Task Isdtype_Bool_Bool()
    {
        np.isdtype(NPTypeCode.Boolean, "bool").Should().BeTrue();
    }

    [TestMethod]
    public async Task Isdtype_Int32_Numeric()
    {
        np.isdtype(NPTypeCode.Int32, "numeric").Should().BeTrue();
    }

    [TestMethod]
    public async Task Isdtype_Bool_Numeric_IsFalse()
    {
        // Bool is excluded from 'numeric' in NumPy's isdtype
        np.isdtype(NPTypeCode.Boolean, "numeric").Should().BeFalse();
    }

    [TestMethod]
    public async Task Isdtype_AllIntegerTypes_AreIntegral()
    {
        np.isdtype(NPTypeCode.Byte, "integral").Should().BeTrue();
        np.isdtype(NPTypeCode.Int16, "integral").Should().BeTrue();
        np.isdtype(NPTypeCode.UInt16, "integral").Should().BeTrue();
        np.isdtype(NPTypeCode.Int32, "integral").Should().BeTrue();
        np.isdtype(NPTypeCode.UInt32, "integral").Should().BeTrue();
        np.isdtype(NPTypeCode.Int64, "integral").Should().BeTrue();
        np.isdtype(NPTypeCode.UInt64, "integral").Should().BeTrue();
    }

    [TestMethod]
    public async Task Isdtype_AllFloatTypes_AreRealFloating()
    {
        np.isdtype(NPTypeCode.Single, "real floating").Should().BeTrue();
        np.isdtype(NPTypeCode.Double, "real floating").Should().BeTrue();
        np.isdtype(NPTypeCode.Decimal, "real floating").Should().BeTrue();
    }

    #endregion

    #region maximum_sctype (NumPy verified)

    [TestMethod]
    public async Task MaximumSctype_Int16_ReturnsInt64()
    {
        np.maximum_sctype(NPTypeCode.Int16).Should().Be(NPTypeCode.Int64);
    }

    [TestMethod]
    public async Task MaximumSctype_Int32_ReturnsInt64()
    {
        np.maximum_sctype(NPTypeCode.Int32).Should().Be(NPTypeCode.Int64);
    }

    [TestMethod]
    public async Task MaximumSctype_Int64_ReturnsInt64()
    {
        np.maximum_sctype(NPTypeCode.Int64).Should().Be(NPTypeCode.Int64);
    }

    [TestMethod]
    public async Task MaximumSctype_Byte_ReturnsUInt64()
    {
        np.maximum_sctype(NPTypeCode.Byte).Should().Be(NPTypeCode.UInt64);
    }

    [TestMethod]
    public async Task MaximumSctype_UInt32_ReturnsUInt64()
    {
        np.maximum_sctype(NPTypeCode.UInt32).Should().Be(NPTypeCode.UInt64);
    }

    [TestMethod]
    public async Task MaximumSctype_Single_ReturnsDouble()
    {
        // NumPy returns longdouble, NumSharp doesn't have longdouble so we return Double
        np.maximum_sctype(NPTypeCode.Single).Should().Be(NPTypeCode.Double);
    }

    [TestMethod]
    public async Task MaximumSctype_Double_ReturnsDouble()
    {
        np.maximum_sctype(NPTypeCode.Double).Should().Be(NPTypeCode.Double);
    }

    [TestMethod]
    public async Task MaximumSctype_Bool_ReturnsBool()
    {
        np.maximum_sctype(NPTypeCode.Boolean).Should().Be(NPTypeCode.Boolean);
    }

    [TestMethod]
    public async Task MaximumSctype_Decimal_ReturnsDecimal()
    {
        // NumSharp-specific: Decimal stays Decimal
        np.maximum_sctype(NPTypeCode.Decimal).Should().Be(NPTypeCode.Decimal);
    }

    [TestMethod]
    public async Task MaximumSctype_Complex_ReturnsComplex()
    {
        np.maximum_sctype(NPTypeCode.Complex).Should().Be(NPTypeCode.Complex);
    }

    #endregion

    #region NPTypeHierarchy.IsSameKind - Used by can_cast same_kind

    [TestMethod]
    public async Task IsSameKind_SignedIntegers()
    {
        NPTypeHierarchy.IsSameKind(NPTypeCode.Int16, NPTypeCode.Int32).Should().BeTrue();
        NPTypeHierarchy.IsSameKind(NPTypeCode.Int32, NPTypeCode.Int64).Should().BeTrue();
    }

    [TestMethod]
    public async Task IsSameKind_UnsignedIntegers()
    {
        NPTypeHierarchy.IsSameKind(NPTypeCode.Byte, NPTypeCode.UInt16).Should().BeTrue();
        NPTypeHierarchy.IsSameKind(NPTypeCode.UInt32, NPTypeCode.UInt64).Should().BeTrue();
    }

    [TestMethod]
    public async Task IsSameKind_SignedAndUnsigned_AreInSameKind()
    {
        // In NumPy, same_kind casting allows int <-> uint (they're both integers)
        NPTypeHierarchy.IsSameKind(NPTypeCode.Int32, NPTypeCode.UInt32).Should().BeTrue();
        NPTypeHierarchy.IsSameKind(NPTypeCode.Int64, NPTypeCode.Byte).Should().BeTrue();
    }

    [TestMethod]
    public async Task IsSameKind_Floats()
    {
        NPTypeHierarchy.IsSameKind(NPTypeCode.Single, NPTypeCode.Double).Should().BeTrue();
        NPTypeHierarchy.IsSameKind(NPTypeCode.Double, NPTypeCode.Decimal).Should().BeTrue();
    }

    [TestMethod]
    public async Task IsSameKind_DifferentKinds_ReturnsFalse()
    {
        NPTypeHierarchy.IsSameKind(NPTypeCode.Int32, NPTypeCode.Double).Should().BeFalse();
        NPTypeHierarchy.IsSameKind(NPTypeCode.Single, NPTypeCode.Complex).Should().BeFalse();
        NPTypeHierarchy.IsSameKind(NPTypeCode.Boolean, NPTypeCode.Int32).Should().BeFalse();
    }

    #endregion

    #region Category Alias Tests

    [TestMethod]
    public async Task CategoryAlias_Signed_EqualsSignedInteger()
    {
        np.issubdtype(NPTypeCode.Int32, "signed").Should().BeTrue();
        np.issubdtype(NPTypeCode.Byte, "signed").Should().BeFalse();
    }

    [TestMethod]
    public async Task CategoryAlias_Unsigned_EqualsUnsignedInteger()
    {
        np.issubdtype(NPTypeCode.Byte, "unsigned").Should().BeTrue();
        np.issubdtype(NPTypeCode.Int32, "unsigned").Should().BeFalse();
    }

    [TestMethod]
    public async Task CategoryAlias_Float_EqualsFloating()
    {
        np.issubdtype(NPTypeCode.Double, "float").Should().BeTrue();
        np.issubdtype(NPTypeCode.Int32, "float").Should().BeFalse();
    }

    [TestMethod]
    public async Task CategoryAlias_Complex_EqualsComplexFloating()
    {
        np.issubdtype(NPTypeCode.Complex, "complex").Should().BeTrue();
        np.issubdtype(NPTypeCode.Double, "complex").Should().BeFalse();
    }

    #endregion

    #region NumSharp-specific: Char type handling

    [TestMethod]
    public async Task Char_TreatedAsUnsignedInteger()
    {
        // Char is treated like uint16 in NumSharp
        np.issubdtype(NPTypeCode.Char, "unsignedinteger").Should().BeTrue();
        np.issubdtype(NPTypeCode.Char, "integer").Should().BeTrue();
        np.issubdtype(NPTypeCode.Char, "number").Should().BeTrue();
    }

    [TestMethod]
    public async Task Char_MaximumSctype_ReturnsUInt64()
    {
        np.maximum_sctype(NPTypeCode.Char).Should().Be(NPTypeCode.UInt64);
    }

    #endregion
}
