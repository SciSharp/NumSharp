using System;

namespace NumSharp.Tests.Logic;

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
    public void Int16_BelongsTo_Generic() =>
        np.issubdtype(NPTypeCode.Int16, np.generic).Should().BeTrue();

    [TestMethod]
    public void Int16_BelongsTo_Number() =>
        np.issubdtype(NPTypeCode.Int16, np.number).Should().BeTrue();

    [TestMethod]
    public void Int16_BelongsTo_Integer() =>
        np.issubdtype(NPTypeCode.Int16, np.integer).Should().BeTrue();

    [TestMethod]
    public void Int16_BelongsTo_SignedInteger() =>
        np.issubdtype(NPTypeCode.Int16, np.signedinteger).Should().BeTrue();

    [TestMethod]
    public void Int16_NotBelongsTo_UnsignedInteger() =>
        np.issubdtype(NPTypeCode.Int16, np.unsignedinteger).Should().BeFalse();

    [TestMethod]
    public void Int16_NotBelongsTo_Inexact() =>
        np.issubdtype(NPTypeCode.Int16, np.inexact).Should().BeFalse();

    [TestMethod]
    public void Int16_NotBelongsTo_Floating() =>
        np.issubdtype(NPTypeCode.Int16, np.floating).Should().BeFalse();

    [TestMethod]
    public void Int32_BelongsTo_SignedInteger() =>
        np.issubdtype(NPTypeCode.Int32, np.signedinteger).Should().BeTrue();

    [TestMethod]
    public void Int64_BelongsTo_SignedInteger() =>
        np.issubdtype(NPTypeCode.Int64, np.signedinteger).Should().BeTrue();

    // ==========================================================================
    // UNSIGNED INTEGERS: belong to generic, number, integer, unsignedinteger
    // ==========================================================================

    [TestMethod]
    public void Byte_BelongsTo_Generic() =>
        np.issubdtype(NPTypeCode.Byte, np.generic).Should().BeTrue();

    [TestMethod]
    public void Byte_BelongsTo_Number() =>
        np.issubdtype(NPTypeCode.Byte, np.number).Should().BeTrue();

    [TestMethod]
    public void Byte_BelongsTo_Integer() =>
        np.issubdtype(NPTypeCode.Byte, np.integer).Should().BeTrue();

    [TestMethod]
    public void Byte_BelongsTo_UnsignedInteger() =>
        np.issubdtype(NPTypeCode.Byte, np.unsignedinteger).Should().BeTrue();

    [TestMethod]
    public void Byte_NotBelongsTo_SignedInteger() =>
        np.issubdtype(NPTypeCode.Byte, np.signedinteger).Should().BeFalse();

    [TestMethod]
    public void UInt16_BelongsTo_UnsignedInteger() =>
        np.issubdtype(NPTypeCode.UInt16, np.unsignedinteger).Should().BeTrue();

    [TestMethod]
    public void UInt32_BelongsTo_UnsignedInteger() =>
        np.issubdtype(NPTypeCode.UInt32, np.unsignedinteger).Should().BeTrue();

    [TestMethod]
    public void UInt64_BelongsTo_UnsignedInteger() =>
        np.issubdtype(NPTypeCode.UInt64, np.unsignedinteger).Should().BeTrue();

    // ==========================================================================
    // FLOATING: belong to generic, number, inexact, floating
    // ==========================================================================

    [TestMethod]
    public void Single_BelongsTo_Generic() =>
        np.issubdtype(NPTypeCode.Single, np.generic).Should().BeTrue();

    [TestMethod]
    public void Single_BelongsTo_Number() =>
        np.issubdtype(NPTypeCode.Single, np.number).Should().BeTrue();

    [TestMethod]
    public void Single_BelongsTo_Inexact() =>
        np.issubdtype(NPTypeCode.Single, np.inexact).Should().BeTrue();

    [TestMethod]
    public void Single_BelongsTo_Floating() =>
        np.issubdtype(NPTypeCode.Single, np.floating).Should().BeTrue();

    [TestMethod]
    public void Single_NotBelongsTo_Integer() =>
        np.issubdtype(NPTypeCode.Single, np.integer).Should().BeFalse();

    [TestMethod]
    public void Single_NotBelongsTo_ComplexFloating() =>
        np.issubdtype(NPTypeCode.Single, np.complexfloating).Should().BeFalse();

    [TestMethod]
    public void Double_BelongsTo_Floating() =>
        np.issubdtype(NPTypeCode.Double, np.floating).Should().BeTrue();

    [TestMethod]
    public void Decimal_BelongsTo_Floating() =>
        np.issubdtype(NPTypeCode.Decimal, np.floating).Should().BeTrue();

    // ==========================================================================
    // COMPLEX: belong to generic, number, inexact, complexfloating
    // ==========================================================================

    [TestMethod]
    public void Complex_BelongsTo_Generic() =>
        np.issubdtype(NPTypeCode.Complex, np.generic).Should().BeTrue();

    [TestMethod]
    public void Complex_BelongsTo_Number() =>
        np.issubdtype(NPTypeCode.Complex, np.number).Should().BeTrue();

    [TestMethod]
    public void Complex_BelongsTo_Inexact() =>
        np.issubdtype(NPTypeCode.Complex, np.inexact).Should().BeTrue();

    [TestMethod]
    public void Complex_BelongsTo_ComplexFloating() =>
        np.issubdtype(NPTypeCode.Complex, np.complexfloating).Should().BeTrue();

    [TestMethod]
    public void Complex_NotBelongsTo_Floating() =>
        np.issubdtype(NPTypeCode.Complex, np.floating).Should().BeFalse();

    [TestMethod]
    public void Complex_NotBelongsTo_Integer() =>
        np.issubdtype(NPTypeCode.Complex, np.integer).Should().BeFalse();

    // ==========================================================================
    // BOOLEAN: belongs to generic ONLY (NumPy 2.x critical behavior!)
    // ==========================================================================

    [TestMethod]
    public void Bool_BelongsTo_Generic() =>
        np.issubdtype(NPTypeCode.Boolean, np.generic).Should().BeTrue();

    [TestMethod]
    public void Bool_BelongsTo_Boolean() =>
        np.issubdtype(NPTypeCode.Boolean, "bool").Should().BeTrue();

    [TestMethod]
    public void Bool_NotBelongsTo_Number() =>
        np.issubdtype(NPTypeCode.Boolean, np.number).Should().BeFalse();

    [TestMethod]
    public void Bool_NotBelongsTo_Integer() =>
        np.issubdtype(NPTypeCode.Boolean, np.integer).Should().BeFalse();

    [TestMethod]
    public void Bool_NotBelongsTo_SignedInteger() =>
        np.issubdtype(NPTypeCode.Boolean, np.signedinteger).Should().BeFalse();

    [TestMethod]
    public void Bool_NotBelongsTo_UnsignedInteger() =>
        np.issubdtype(NPTypeCode.Boolean, np.unsignedinteger).Should().BeFalse();

    #endregion

    #region issubdtype - Concrete Type Comparison (NumPy verified)

    // NumPy: issubdtype(int32, int64) = False (different concrete types)
    // NumPy: issubdtype(int32, int32) = True (same type)

    [TestMethod]
    public void ConcreteType_SameType_ReturnsTrue_Int32()
    {
        np.issubdtype(NPTypeCode.Int32, NPTypeCode.Int32).Should().BeTrue();
    }

    [TestMethod]
    public void ConcreteType_SameType_ReturnsTrue_Double()
    {
        np.issubdtype(NPTypeCode.Double, NPTypeCode.Double).Should().BeTrue();
    }

    [TestMethod]
    public void ConcreteType_DifferentType_SameKind_ReturnsFalse()
    {
        np.issubdtype(NPTypeCode.Int32, NPTypeCode.Int64).Should().BeFalse();
    }

    [TestMethod]
    public void ConcreteType_DifferentType_SameKind_ReturnsFalse_Reverse()
    {
        np.issubdtype(NPTypeCode.Int64, NPTypeCode.Int32).Should().BeFalse();
    }

    [TestMethod]
    public void ConcreteType_Float32_Float64_ReturnsFalse()
    {
        np.issubdtype(NPTypeCode.Single, NPTypeCode.Double).Should().BeFalse();
    }

    [TestMethod]
    public void ConcreteType_Uint8_Uint64_ReturnsFalse()
    {
        np.issubdtype(NPTypeCode.Byte, NPTypeCode.UInt64).Should().BeFalse();
    }

    [TestMethod]
    public void ConcreteType_DifferentKind_ReturnsFalse()
    {
        np.issubdtype(NPTypeCode.Int32, NPTypeCode.Double).Should().BeFalse();
    }

    [TestMethod]
    public void ConcreteType_Bool_Int32_ReturnsFalse()
    {
        np.issubdtype(NPTypeCode.Boolean, NPTypeCode.Int32).Should().BeFalse();
    }

    #endregion

    #region isdtype - Category Checks (NumPy 2.0+ verified)

    [TestMethod]
    public void Isdtype_Int32_Integral()
    {
        np.isdtype(NPTypeCode.Int32, "integral").Should().BeTrue();
    }

    [TestMethod]
    public void Isdtype_Float64_RealFloating()
    {
        np.isdtype(NPTypeCode.Double, "real floating").Should().BeTrue();
    }

    [TestMethod]
    public void Isdtype_Complex_ComplexFloating()
    {
        np.isdtype(NPTypeCode.Complex, "complex floating").Should().BeTrue();
    }

    [TestMethod]
    public void Isdtype_Bool_Bool()
    {
        np.isdtype(NPTypeCode.Boolean, "bool").Should().BeTrue();
    }

    [TestMethod]
    public void Isdtype_Int32_Numeric()
    {
        np.isdtype(NPTypeCode.Int32, "numeric").Should().BeTrue();
    }

    [TestMethod]
    public void Isdtype_Bool_Numeric_IsFalse()
    {
        // Bool is excluded from 'numeric' in NumPy's isdtype
        np.isdtype(NPTypeCode.Boolean, "numeric").Should().BeFalse();
    }

    [TestMethod]
    public void Isdtype_AllIntegerTypes_AreIntegral()
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
    public void Isdtype_AllFloatTypes_AreRealFloating()
    {
        np.isdtype(NPTypeCode.Single, "real floating").Should().BeTrue();
        np.isdtype(NPTypeCode.Double, "real floating").Should().BeTrue();
        np.isdtype(NPTypeCode.Decimal, "real floating").Should().BeTrue();
    }

    #endregion

    #region maximum_sctype (NumPy verified)

    [TestMethod]
    public void MaximumSctype_Int16_ReturnsInt64()
    {
        np.maximum_sctype(NPTypeCode.Int16).Should().Be(NPTypeCode.Int64);
    }

    [TestMethod]
    public void MaximumSctype_Int32_ReturnsInt64()
    {
        np.maximum_sctype(NPTypeCode.Int32).Should().Be(NPTypeCode.Int64);
    }

    [TestMethod]
    public void MaximumSctype_Int64_ReturnsInt64()
    {
        np.maximum_sctype(NPTypeCode.Int64).Should().Be(NPTypeCode.Int64);
    }

    [TestMethod]
    public void MaximumSctype_Byte_ReturnsUInt64()
    {
        np.maximum_sctype(NPTypeCode.Byte).Should().Be(NPTypeCode.UInt64);
    }

    [TestMethod]
    public void MaximumSctype_UInt32_ReturnsUInt64()
    {
        np.maximum_sctype(NPTypeCode.UInt32).Should().Be(NPTypeCode.UInt64);
    }

    [TestMethod]
    public void MaximumSctype_Single_ReturnsDouble()
    {
        // NumPy returns longdouble, NumSharp doesn't have longdouble so we return Double
        np.maximum_sctype(NPTypeCode.Single).Should().Be(NPTypeCode.Double);
    }

    [TestMethod]
    public void MaximumSctype_Double_ReturnsDouble()
    {
        np.maximum_sctype(NPTypeCode.Double).Should().Be(NPTypeCode.Double);
    }

    [TestMethod]
    public void MaximumSctype_Bool_ReturnsBool()
    {
        np.maximum_sctype(NPTypeCode.Boolean).Should().Be(NPTypeCode.Boolean);
    }

    [TestMethod]
    public void MaximumSctype_Decimal_ReturnsDecimal()
    {
        // NumSharp-specific: Decimal stays Decimal
        np.maximum_sctype(NPTypeCode.Decimal).Should().Be(NPTypeCode.Decimal);
    }

    [TestMethod]
    public void MaximumSctype_Complex_ReturnsComplex()
    {
        np.maximum_sctype(NPTypeCode.Complex).Should().Be(NPTypeCode.Complex);
    }

    #endregion

    #region NPTypeHierarchy.IsSameKind - Used by can_cast same_kind

    [TestMethod]
    public void IsSameKind_SignedIntegers()
    {
        NPTypeHierarchy.IsSameKind(NPTypeCode.Int16, NPTypeCode.Int32).Should().BeTrue();
        NPTypeHierarchy.IsSameKind(NPTypeCode.Int32, NPTypeCode.Int64).Should().BeTrue();
    }

    [TestMethod]
    public void IsSameKind_UnsignedIntegers()
    {
        NPTypeHierarchy.IsSameKind(NPTypeCode.Byte, NPTypeCode.UInt16).Should().BeTrue();
        NPTypeHierarchy.IsSameKind(NPTypeCode.UInt32, NPTypeCode.UInt64).Should().BeTrue();
    }

    [TestMethod]
    public void IsSameKind_SignedAndUnsigned_AreInSameKind()
    {
        // In NumPy, same_kind casting allows int <-> uint (they're both integers)
        NPTypeHierarchy.IsSameKind(NPTypeCode.Int32, NPTypeCode.UInt32).Should().BeTrue();
        NPTypeHierarchy.IsSameKind(NPTypeCode.Int64, NPTypeCode.Byte).Should().BeTrue();
    }

    [TestMethod]
    public void IsSameKind_Floats()
    {
        NPTypeHierarchy.IsSameKind(NPTypeCode.Single, NPTypeCode.Double).Should().BeTrue();
        NPTypeHierarchy.IsSameKind(NPTypeCode.Double, NPTypeCode.Decimal).Should().BeTrue();
    }

    [TestMethod]
    public void IsSameKind_DifferentKinds_ReturnsFalse()
    {
        NPTypeHierarchy.IsSameKind(NPTypeCode.Int32, NPTypeCode.Double).Should().BeFalse();
        NPTypeHierarchy.IsSameKind(NPTypeCode.Single, NPTypeCode.Complex).Should().BeFalse();
        NPTypeHierarchy.IsSameKind(NPTypeCode.Boolean, NPTypeCode.Int32).Should().BeFalse();
    }

    #endregion

    #region Families are scalar classes; strings are dtype strings (NumPy 2.4.2)

    /// <summary>The signed family is the class <c>np.signedinteger</c> (the old "signed" string alias is gone: NumPy has none).</summary>
    [TestMethod]
    public void SignedFamily_IsTheSignedIntegerClass()
    {
        np.issubdtype(NPTypeCode.Int32, np.signedinteger).Should().BeTrue();
        np.issubdtype(NPTypeCode.Byte, np.signedinteger).Should().BeFalse();
        new Action(() => np.issubdtype(NPTypeCode.Int32, "signed")).Should().Throw<NotSupportedException>()
            .WithMessage("data type 'signed' not understood");
    }

    /// <summary>The unsigned family is the class <c>np.unsignedinteger</c>.</summary>
    [TestMethod]
    public void UnsignedFamily_IsTheUnsignedIntegerClass()
    {
        np.issubdtype(NPTypeCode.Byte, np.unsignedinteger).Should().BeTrue();
        np.issubdtype(NPTypeCode.Int32, np.unsignedinteger).Should().BeFalse();
    }

    /// <summary>"float" is NumPy's name for float64, not the floating family: float32 is not "float".</summary>
    [TestMethod]
    public void FloatString_IsFloat64()
    {
        np.issubdtype(NPTypeCode.Double, "float").Should().BeTrue();
        np.issubdtype(NPTypeCode.Int32, "float").Should().BeFalse();
        np.issubdtype(NPTypeCode.Single, "float").Should().BeFalse();
        np.issubdtype(NPTypeCode.Single, np.floating).Should().BeTrue();
    }

    /// <summary>"complex" is NumPy's name for complex128.</summary>
    [TestMethod]
    public void ComplexString_IsComplex128()
    {
        np.issubdtype(NPTypeCode.Complex, "complex").Should().BeTrue();
        np.issubdtype(NPTypeCode.Double, "complex").Should().BeFalse();
        np.issubdtype(NPTypeCode.Complex, np.complexfloating).Should().BeTrue();
    }

    #endregion

    #region NumSharp-specific: Char type handling

    [TestMethod]
    public void Char_TreatedAsUnsignedInteger()
    {
        // Char is treated like uint16 in NumSharp
        np.issubdtype(NPTypeCode.Char, np.unsignedinteger).Should().BeTrue();
        np.issubdtype(NPTypeCode.Char, np.integer).Should().BeTrue();
        np.issubdtype(NPTypeCode.Char, np.number).Should().BeTrue();
    }

    [TestMethod]
    public void Char_MaximumSctype_ReturnsUInt64()
    {
        np.maximum_sctype(NPTypeCode.Char).Should().Be(NPTypeCode.UInt64);
    }

    #endregion
}
