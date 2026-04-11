using System.Threading.Tasks;
using TUnit.Core;
using TUnit.Assertions;
using TUnit.Assertions.Extensions;
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
public class NPTypeHierarchyBattleTest
{
    #region issubdtype - Type Hierarchy Tests (NumPy verified)

    // ==========================================================================
    // SIGNED INTEGERS: belong to generic, number, integer, signedinteger
    // ==========================================================================

    [Test]
    public async Task Int16_BelongsTo_Generic() =>
        await Assert.That(np.issubdtype(NPTypeCode.Int16, "generic")).IsTrue();

    [Test]
    public async Task Int16_BelongsTo_Number() =>
        await Assert.That(np.issubdtype(NPTypeCode.Int16, "number")).IsTrue();

    [Test]
    public async Task Int16_BelongsTo_Integer() =>
        await Assert.That(np.issubdtype(NPTypeCode.Int16, "integer")).IsTrue();

    [Test]
    public async Task Int16_BelongsTo_SignedInteger() =>
        await Assert.That(np.issubdtype(NPTypeCode.Int16, "signedinteger")).IsTrue();

    [Test]
    public async Task Int16_NotBelongsTo_UnsignedInteger() =>
        await Assert.That(np.issubdtype(NPTypeCode.Int16, "unsignedinteger")).IsFalse();

    [Test]
    public async Task Int16_NotBelongsTo_Inexact() =>
        await Assert.That(np.issubdtype(NPTypeCode.Int16, "inexact")).IsFalse();

    [Test]
    public async Task Int16_NotBelongsTo_Floating() =>
        await Assert.That(np.issubdtype(NPTypeCode.Int16, "floating")).IsFalse();

    [Test]
    public async Task Int32_BelongsTo_SignedInteger() =>
        await Assert.That(np.issubdtype(NPTypeCode.Int32, "signedinteger")).IsTrue();

    [Test]
    public async Task Int64_BelongsTo_SignedInteger() =>
        await Assert.That(np.issubdtype(NPTypeCode.Int64, "signedinteger")).IsTrue();

    // ==========================================================================
    // UNSIGNED INTEGERS: belong to generic, number, integer, unsignedinteger
    // ==========================================================================

    [Test]
    public async Task Byte_BelongsTo_Generic() =>
        await Assert.That(np.issubdtype(NPTypeCode.Byte, "generic")).IsTrue();

    [Test]
    public async Task Byte_BelongsTo_Number() =>
        await Assert.That(np.issubdtype(NPTypeCode.Byte, "number")).IsTrue();

    [Test]
    public async Task Byte_BelongsTo_Integer() =>
        await Assert.That(np.issubdtype(NPTypeCode.Byte, "integer")).IsTrue();

    [Test]
    public async Task Byte_BelongsTo_UnsignedInteger() =>
        await Assert.That(np.issubdtype(NPTypeCode.Byte, "unsignedinteger")).IsTrue();

    [Test]
    public async Task Byte_NotBelongsTo_SignedInteger() =>
        await Assert.That(np.issubdtype(NPTypeCode.Byte, "signedinteger")).IsFalse();

    [Test]
    public async Task UInt16_BelongsTo_UnsignedInteger() =>
        await Assert.That(np.issubdtype(NPTypeCode.UInt16, "unsignedinteger")).IsTrue();

    [Test]
    public async Task UInt32_BelongsTo_UnsignedInteger() =>
        await Assert.That(np.issubdtype(NPTypeCode.UInt32, "unsignedinteger")).IsTrue();

    [Test]
    public async Task UInt64_BelongsTo_UnsignedInteger() =>
        await Assert.That(np.issubdtype(NPTypeCode.UInt64, "unsignedinteger")).IsTrue();

    // ==========================================================================
    // FLOATING: belong to generic, number, inexact, floating
    // ==========================================================================

    [Test]
    public async Task Single_BelongsTo_Generic() =>
        await Assert.That(np.issubdtype(NPTypeCode.Single, "generic")).IsTrue();

    [Test]
    public async Task Single_BelongsTo_Number() =>
        await Assert.That(np.issubdtype(NPTypeCode.Single, "number")).IsTrue();

    [Test]
    public async Task Single_BelongsTo_Inexact() =>
        await Assert.That(np.issubdtype(NPTypeCode.Single, "inexact")).IsTrue();

    [Test]
    public async Task Single_BelongsTo_Floating() =>
        await Assert.That(np.issubdtype(NPTypeCode.Single, "floating")).IsTrue();

    [Test]
    public async Task Single_NotBelongsTo_Integer() =>
        await Assert.That(np.issubdtype(NPTypeCode.Single, "integer")).IsFalse();

    [Test]
    public async Task Single_NotBelongsTo_ComplexFloating() =>
        await Assert.That(np.issubdtype(NPTypeCode.Single, "complexfloating")).IsFalse();

    [Test]
    public async Task Double_BelongsTo_Floating() =>
        await Assert.That(np.issubdtype(NPTypeCode.Double, "floating")).IsTrue();

    [Test]
    public async Task Decimal_BelongsTo_Floating() =>
        await Assert.That(np.issubdtype(NPTypeCode.Decimal, "floating")).IsTrue();

    // ==========================================================================
    // COMPLEX: belong to generic, number, inexact, complexfloating
    // ==========================================================================

    [Test]
    public async Task Complex_BelongsTo_Generic() =>
        await Assert.That(np.issubdtype(NPTypeCode.Complex, "generic")).IsTrue();

    [Test]
    public async Task Complex_BelongsTo_Number() =>
        await Assert.That(np.issubdtype(NPTypeCode.Complex, "number")).IsTrue();

    [Test]
    public async Task Complex_BelongsTo_Inexact() =>
        await Assert.That(np.issubdtype(NPTypeCode.Complex, "inexact")).IsTrue();

    [Test]
    public async Task Complex_BelongsTo_ComplexFloating() =>
        await Assert.That(np.issubdtype(NPTypeCode.Complex, "complexfloating")).IsTrue();

    [Test]
    public async Task Complex_NotBelongsTo_Floating() =>
        await Assert.That(np.issubdtype(NPTypeCode.Complex, "floating")).IsFalse();

    [Test]
    public async Task Complex_NotBelongsTo_Integer() =>
        await Assert.That(np.issubdtype(NPTypeCode.Complex, "integer")).IsFalse();

    // ==========================================================================
    // BOOLEAN: belongs to generic ONLY (NumPy 2.x critical behavior!)
    // ==========================================================================

    [Test]
    public async Task Bool_BelongsTo_Generic() =>
        await Assert.That(np.issubdtype(NPTypeCode.Boolean, "generic")).IsTrue();

    [Test]
    public async Task Bool_BelongsTo_Boolean() =>
        await Assert.That(np.issubdtype(NPTypeCode.Boolean, "boolean")).IsTrue();

    [Test]
    [DisplayName("CRITICAL: Bool is NOT under Number in NumPy 2.x")]
    public async Task Bool_NotBelongsTo_Number() =>
        await Assert.That(np.issubdtype(NPTypeCode.Boolean, "number")).IsFalse();

    [Test]
    [DisplayName("CRITICAL: Bool is NOT under Integer in NumPy 2.x")]
    public async Task Bool_NotBelongsTo_Integer() =>
        await Assert.That(np.issubdtype(NPTypeCode.Boolean, "integer")).IsFalse();

    [Test]
    public async Task Bool_NotBelongsTo_SignedInteger() =>
        await Assert.That(np.issubdtype(NPTypeCode.Boolean, "signedinteger")).IsFalse();

    [Test]
    public async Task Bool_NotBelongsTo_UnsignedInteger() =>
        await Assert.That(np.issubdtype(NPTypeCode.Boolean, "unsignedinteger")).IsFalse();

    #endregion

    #region issubdtype - Concrete Type Comparison (NumPy verified)

    // NumPy: issubdtype(int32, int64) = False (different concrete types)
    // NumPy: issubdtype(int32, int32) = True (same type)

    [Test]
    public async Task ConcreteType_SameType_ReturnsTrue_Int32()
    {
        await Assert.That(np.issubdtype(NPTypeCode.Int32, NPTypeCode.Int32)).IsTrue();
    }

    [Test]
    public async Task ConcreteType_SameType_ReturnsTrue_Double()
    {
        await Assert.That(np.issubdtype(NPTypeCode.Double, NPTypeCode.Double)).IsTrue();
    }

    [Test]
    [DisplayName("NumPy: issubdtype(int32, int64) = False")]
    public async Task ConcreteType_DifferentType_SameKind_ReturnsFalse()
    {
        await Assert.That(np.issubdtype(NPTypeCode.Int32, NPTypeCode.Int64)).IsFalse();
    }

    [Test]
    [DisplayName("NumPy: issubdtype(int64, int32) = False")]
    public async Task ConcreteType_DifferentType_SameKind_ReturnsFalse_Reverse()
    {
        await Assert.That(np.issubdtype(NPTypeCode.Int64, NPTypeCode.Int32)).IsFalse();
    }

    [Test]
    [DisplayName("NumPy: issubdtype(float32, float64) = False")]
    public async Task ConcreteType_Float32_Float64_ReturnsFalse()
    {
        await Assert.That(np.issubdtype(NPTypeCode.Single, NPTypeCode.Double)).IsFalse();
    }

    [Test]
    [DisplayName("NumPy: issubdtype(uint8, uint64) = False")]
    public async Task ConcreteType_Uint8_Uint64_ReturnsFalse()
    {
        await Assert.That(np.issubdtype(NPTypeCode.Byte, NPTypeCode.UInt64)).IsFalse();
    }

    [Test]
    [DisplayName("NumPy: issubdtype(int32, float64) = False")]
    public async Task ConcreteType_DifferentKind_ReturnsFalse()
    {
        await Assert.That(np.issubdtype(NPTypeCode.Int32, NPTypeCode.Double)).IsFalse();
    }

    [Test]
    [DisplayName("NumPy: issubdtype(bool, int32) = False")]
    public async Task ConcreteType_Bool_Int32_ReturnsFalse()
    {
        await Assert.That(np.issubdtype(NPTypeCode.Boolean, NPTypeCode.Int32)).IsFalse();
    }

    #endregion

    #region isdtype - Category Checks (NumPy 2.0+ verified)

    [Test]
    [DisplayName("NumPy: isdtype(int32, 'integral') = True")]
    public async Task Isdtype_Int32_Integral()
    {
        await Assert.That(np.isdtype(NPTypeCode.Int32, "integral")).IsTrue();
    }

    [Test]
    [DisplayName("NumPy: isdtype(float64, 'real floating') = True")]
    public async Task Isdtype_Float64_RealFloating()
    {
        await Assert.That(np.isdtype(NPTypeCode.Double, "real floating")).IsTrue();
    }

    [Test]
    [DisplayName("NumPy: isdtype(complex128, 'complex floating') = True")]
    public async Task Isdtype_Complex_ComplexFloating()
    {
        await Assert.That(np.isdtype(NPTypeCode.Complex, "complex floating")).IsTrue();
    }

    [Test]
    [DisplayName("NumPy: isdtype(bool, 'bool') = True")]
    public async Task Isdtype_Bool_Bool()
    {
        await Assert.That(np.isdtype(NPTypeCode.Boolean, "bool")).IsTrue();
    }

    [Test]
    [DisplayName("NumPy: isdtype(int32, 'numeric') = True")]
    public async Task Isdtype_Int32_Numeric()
    {
        await Assert.That(np.isdtype(NPTypeCode.Int32, "numeric")).IsTrue();
    }

    [Test]
    [DisplayName("CRITICAL: NumPy: isdtype(bool, 'numeric') = False")]
    public async Task Isdtype_Bool_Numeric_IsFalse()
    {
        // Bool is excluded from 'numeric' in NumPy's isdtype
        await Assert.That(np.isdtype(NPTypeCode.Boolean, "numeric")).IsFalse();
    }

    [Test]
    public async Task Isdtype_AllIntegerTypes_AreIntegral()
    {
        await Assert.That(np.isdtype(NPTypeCode.Byte, "integral")).IsTrue();
        await Assert.That(np.isdtype(NPTypeCode.Int16, "integral")).IsTrue();
        await Assert.That(np.isdtype(NPTypeCode.UInt16, "integral")).IsTrue();
        await Assert.That(np.isdtype(NPTypeCode.Int32, "integral")).IsTrue();
        await Assert.That(np.isdtype(NPTypeCode.UInt32, "integral")).IsTrue();
        await Assert.That(np.isdtype(NPTypeCode.Int64, "integral")).IsTrue();
        await Assert.That(np.isdtype(NPTypeCode.UInt64, "integral")).IsTrue();
    }

    [Test]
    public async Task Isdtype_AllFloatTypes_AreRealFloating()
    {
        await Assert.That(np.isdtype(NPTypeCode.Single, "real floating")).IsTrue();
        await Assert.That(np.isdtype(NPTypeCode.Double, "real floating")).IsTrue();
        await Assert.That(np.isdtype(NPTypeCode.Decimal, "real floating")).IsTrue();
    }

    #endregion

    #region maximum_sctype (NumPy verified)

    [Test]
    [DisplayName("NumPy: maximum_sctype(int16) = int64")]
    public async Task MaximumSctype_Int16_ReturnsInt64()
    {
        await Assert.That(np.maximum_sctype(NPTypeCode.Int16)).IsEqualTo(NPTypeCode.Int64);
    }

    [Test]
    [DisplayName("NumPy: maximum_sctype(int32) = int64")]
    public async Task MaximumSctype_Int32_ReturnsInt64()
    {
        await Assert.That(np.maximum_sctype(NPTypeCode.Int32)).IsEqualTo(NPTypeCode.Int64);
    }

    [Test]
    [DisplayName("NumPy: maximum_sctype(int64) = int64")]
    public async Task MaximumSctype_Int64_ReturnsInt64()
    {
        await Assert.That(np.maximum_sctype(NPTypeCode.Int64)).IsEqualTo(NPTypeCode.Int64);
    }

    [Test]
    [DisplayName("NumPy: maximum_sctype(uint8) = uint64")]
    public async Task MaximumSctype_Byte_ReturnsUInt64()
    {
        await Assert.That(np.maximum_sctype(NPTypeCode.Byte)).IsEqualTo(NPTypeCode.UInt64);
    }

    [Test]
    [DisplayName("NumPy: maximum_sctype(uint32) = uint64")]
    public async Task MaximumSctype_UInt32_ReturnsUInt64()
    {
        await Assert.That(np.maximum_sctype(NPTypeCode.UInt32)).IsEqualTo(NPTypeCode.UInt64);
    }

    [Test]
    [DisplayName("NumPy: maximum_sctype(float32) = longdouble (NumSharp: Double)")]
    public async Task MaximumSctype_Single_ReturnsDouble()
    {
        // NumPy returns longdouble, NumSharp doesn't have longdouble so we return Double
        await Assert.That(np.maximum_sctype(NPTypeCode.Single)).IsEqualTo(NPTypeCode.Double);
    }

    [Test]
    [DisplayName("NumPy: maximum_sctype(float64) = longdouble (NumSharp: Double)")]
    public async Task MaximumSctype_Double_ReturnsDouble()
    {
        await Assert.That(np.maximum_sctype(NPTypeCode.Double)).IsEqualTo(NPTypeCode.Double);
    }

    [Test]
    [DisplayName("NumPy: maximum_sctype(bool) = bool")]
    public async Task MaximumSctype_Bool_ReturnsBool()
    {
        await Assert.That(np.maximum_sctype(NPTypeCode.Boolean)).IsEqualTo(NPTypeCode.Boolean);
    }

    [Test]
    public async Task MaximumSctype_Decimal_ReturnsDecimal()
    {
        // NumSharp-specific: Decimal stays Decimal
        await Assert.That(np.maximum_sctype(NPTypeCode.Decimal)).IsEqualTo(NPTypeCode.Decimal);
    }

    [Test]
    public async Task MaximumSctype_Complex_ReturnsComplex()
    {
        await Assert.That(np.maximum_sctype(NPTypeCode.Complex)).IsEqualTo(NPTypeCode.Complex);
    }

    #endregion

    #region NPTypeHierarchy.IsSameKind - Used by can_cast same_kind

    [Test]
    public async Task IsSameKind_SignedIntegers()
    {
        await Assert.That(NPTypeHierarchy.IsSameKind(NPTypeCode.Int16, NPTypeCode.Int32)).IsTrue();
        await Assert.That(NPTypeHierarchy.IsSameKind(NPTypeCode.Int32, NPTypeCode.Int64)).IsTrue();
    }

    [Test]
    public async Task IsSameKind_UnsignedIntegers()
    {
        await Assert.That(NPTypeHierarchy.IsSameKind(NPTypeCode.Byte, NPTypeCode.UInt16)).IsTrue();
        await Assert.That(NPTypeHierarchy.IsSameKind(NPTypeCode.UInt32, NPTypeCode.UInt64)).IsTrue();
    }

    [Test]
    public async Task IsSameKind_SignedAndUnsigned_AreInSameKind()
    {
        // In NumPy, same_kind casting allows int <-> uint (they're both integers)
        await Assert.That(NPTypeHierarchy.IsSameKind(NPTypeCode.Int32, NPTypeCode.UInt32)).IsTrue();
        await Assert.That(NPTypeHierarchy.IsSameKind(NPTypeCode.Int64, NPTypeCode.Byte)).IsTrue();
    }

    [Test]
    public async Task IsSameKind_Floats()
    {
        await Assert.That(NPTypeHierarchy.IsSameKind(NPTypeCode.Single, NPTypeCode.Double)).IsTrue();
        await Assert.That(NPTypeHierarchy.IsSameKind(NPTypeCode.Double, NPTypeCode.Decimal)).IsTrue();
    }

    [Test]
    public async Task IsSameKind_DifferentKinds_ReturnsFalse()
    {
        await Assert.That(NPTypeHierarchy.IsSameKind(NPTypeCode.Int32, NPTypeCode.Double)).IsFalse();
        await Assert.That(NPTypeHierarchy.IsSameKind(NPTypeCode.Single, NPTypeCode.Complex)).IsFalse();
        await Assert.That(NPTypeHierarchy.IsSameKind(NPTypeCode.Boolean, NPTypeCode.Int32)).IsFalse();
    }

    #endregion

    #region Category Alias Tests

    [Test]
    public async Task CategoryAlias_Signed_EqualsSignedInteger()
    {
        await Assert.That(np.issubdtype(NPTypeCode.Int32, "signed")).IsTrue();
        await Assert.That(np.issubdtype(NPTypeCode.Byte, "signed")).IsFalse();
    }

    [Test]
    public async Task CategoryAlias_Unsigned_EqualsUnsignedInteger()
    {
        await Assert.That(np.issubdtype(NPTypeCode.Byte, "unsigned")).IsTrue();
        await Assert.That(np.issubdtype(NPTypeCode.Int32, "unsigned")).IsFalse();
    }

    [Test]
    public async Task CategoryAlias_Float_EqualsFloating()
    {
        await Assert.That(np.issubdtype(NPTypeCode.Double, "float")).IsTrue();
        await Assert.That(np.issubdtype(NPTypeCode.Int32, "float")).IsFalse();
    }

    [Test]
    public async Task CategoryAlias_Complex_EqualsComplexFloating()
    {
        await Assert.That(np.issubdtype(NPTypeCode.Complex, "complex")).IsTrue();
        await Assert.That(np.issubdtype(NPTypeCode.Double, "complex")).IsFalse();
    }

    #endregion

    #region NumSharp-specific: Char type handling

    [Test]
    public async Task Char_TreatedAsUnsignedInteger()
    {
        // Char is treated like uint16 in NumSharp
        await Assert.That(np.issubdtype(NPTypeCode.Char, "unsignedinteger")).IsTrue();
        await Assert.That(np.issubdtype(NPTypeCode.Char, "integer")).IsTrue();
        await Assert.That(np.issubdtype(NPTypeCode.Char, "number")).IsTrue();
    }

    [Test]
    public async Task Char_MaximumSctype_ReturnsUInt64()
    {
        await Assert.That(np.maximum_sctype(NPTypeCode.Char)).IsEqualTo(NPTypeCode.UInt64);
    }

    #endregion
}
