using System;

namespace NumSharp.Tests.APIs;

/// <summary>
/// Battle tests for np.issubdtype - comprehensive coverage of NumPy's scalar-type hierarchy. The abstract families are the
/// scalar CLASSES (<see cref="np.integer"/>, <see cref="np.floating"/>, ...), as in NumPy; a string second argument is a
/// dtype string (NumPy's <c>dtype(arg2).type</c>), never a family name. Expected values probed against NumPy 2.4.2.
/// </summary>
[TestClass]
public class NpIsSubdtypeBattleTests
{
    /// <summary>The storage codes swept by the generic-family test (every NumSharp dtype with a NumPy class or a NumSharp-only one).</summary>
    private static readonly NPTypeCode[] AllTypes = new[]
    {
        NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.Int16, NPTypeCode.UInt16,
        NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64,
        NPTypeCode.Char, NPTypeCode.Single, NPTypeCode.Double, NPTypeCode.Decimal
    };

    #region Generic Category

    /// <summary>Every dtype is under <c>np.generic</c>.</summary>
    [TestMethod]
    public void IsSubdtype_AllTypes_AreGeneric()
    {
        foreach (var type in AllTypes)
        {
            np.issubdtype(type, np.generic).Should().BeTrue();
        }
    }

    #endregion

    #region Number Category

    /// <summary>Signed and unsigned integers are numbers.</summary>
    [TestMethod]
    public void IsSubdtype_IntegersAreNumbers()
    {
        np.issubdtype(NPTypeCode.Int32, np.number).Should().BeTrue();
        np.issubdtype(NPTypeCode.Byte, np.number).Should().BeTrue();
    }

    /// <summary>Floats are numbers.</summary>
    [TestMethod]
    public void IsSubdtype_FloatsAreNumbers()
    {
        np.issubdtype(NPTypeCode.Double, np.number).Should().BeTrue();
    }

    /// <summary>NumPy 2.x: bool sits directly under generic, so it is not a number.</summary>
    [TestMethod]
    public void IsSubdtype_BoolNotNumber_NumPy2x()
    {
        np.issubdtype(NPTypeCode.Boolean, np.number).Should().BeFalse();
    }

    #endregion

    #region Integer Category

    /// <summary>The integer dtypes are under <c>np.integer</c>.</summary>
    [TestMethod]
    public void IsSubdtype_IntegersAreInteger()
    {
        np.issubdtype(NPTypeCode.Int32, np.integer).Should().BeTrue();
        np.issubdtype(NPTypeCode.Byte, np.integer).Should().BeTrue();
    }

    /// <summary>NumPy 2.x: bool is not an integer.</summary>
    [TestMethod]
    public void IsSubdtype_BoolNotInteger_NumPy2x()
    {
        np.issubdtype(NPTypeCode.Boolean, np.integer).Should().BeFalse();
    }

    /// <summary>A float is not an integer.</summary>
    [TestMethod]
    public void IsSubdtype_FloatNotInteger()
    {
        np.issubdtype(NPTypeCode.Double, np.integer).Should().BeFalse();
    }

    #endregion

    #region Signed/Unsigned Integer Categories

    /// <summary>int32 and int64 are signed integers.</summary>
    [TestMethod]
    public void IsSubdtype_SignedIntegers_AreSignedInteger()
    {
        np.issubdtype(NPTypeCode.Int32, np.signedinteger).Should().BeTrue();
        np.issubdtype(NPTypeCode.Int64, np.signedinteger).Should().BeTrue();
    }

    /// <summary>uint8 and uint32 are unsigned integers.</summary>
    [TestMethod]
    public void IsSubdtype_UnsignedIntegers_AreUnsignedInteger()
    {
        np.issubdtype(NPTypeCode.Byte, np.unsignedinteger).Should().BeTrue();
        np.issubdtype(NPTypeCode.UInt32, np.unsignedinteger).Should().BeTrue();
    }

    #endregion

    #region Floating Category

    /// <summary>float32 and float64 are floating.</summary>
    [TestMethod]
    public void IsSubdtype_Floats_AreFloating()
    {
        np.issubdtype(NPTypeCode.Single, np.floating).Should().BeTrue();
        np.issubdtype(NPTypeCode.Double, np.floating).Should().BeTrue();
    }

    /// <summary>An integer is not floating.</summary>
    [TestMethod]
    public void IsSubdtype_Integers_NotFloating()
    {
        np.issubdtype(NPTypeCode.Int32, np.floating).Should().BeFalse();
    }

    #endregion

    #region NDArray Overload

    /// <summary>
    /// NumSharp's array convenience takes a DTYPE string, as the second argument of every issubdtype overload does
    /// (NumPy itself refuses an array argument; ported code writes <c>arr.dtype</c>).
    /// </summary>
    [TestMethod]
    public void IsSubdtype_NDArray_String()
    {
        var arr = np.array(new int[] { 1, 2, 3 });
        np.issubdtype(arr, "i4").Should().BeTrue();
        np.issubdtype(arr, "f8").Should().BeFalse();
        np.issubdtype(arr.dtype, np.integer).Should().BeTrue();
        np.issubdtype(arr.dtype, np.floating).Should().BeFalse();
    }

    /// <summary>A null array is refused.</summary>
    [TestMethod]
    public void IsSubdtype_NDArray_Null_Throws()
    {
        new Action(() => np.issubdtype((NDArray)null!, "i4")).Should().Throw<ArgumentNullException>();
    }

    #endregion

    #region Type Overloads

    /// <summary>A C# element type binds the dtype overloads: <c>typeof(int)</c> is int32.</summary>
    [TestMethod]
    public void IsSubdtype_Type_ScalarClass()
    {
        np.issubdtype(typeof(int), np.integer).Should().BeTrue();
        np.issubdtype(typeof(double), np.floating).Should().BeTrue();
        np.issubdtype(typeof(int), "i4").Should().BeTrue();
        np.issubdtype(typeof(int), "i8").Should().BeFalse();
    }

    /// <summary>Two concrete classes are related only when equal (NumPy: <c>issubdtype(np.int32, np.int64)</c> is False).</summary>
    [TestMethod]
    public void IsSubdtype_Type_Type()
    {
        // Same concrete types: returns True
        np.issubdtype(typeof(int), typeof(int)).Should().BeTrue();
        np.issubdtype(typeof(long), typeof(long)).Should().BeTrue();

        // Different concrete types: returns False (even if same kind)
        // This matches NumPy: np.issubdtype(np.int32, np.int64) == False
        np.issubdtype(typeof(int), typeof(long)).Should().BeFalse();
        np.issubdtype(typeof(float), typeof(double)).Should().BeFalse();
    }

    #endregion

    #region String arguments are dtype strings (NumPy 2.x)

    /// <summary>
    /// A family NAME is not a dtype string: NumPy raises <c>TypeError: data type 'integer' not understood</c> (NumSharp's
    /// dtype parser raises its NotSupportedException with the same text).
    /// </summary>
    [TestMethod]
    public void IsSubdtype_FamilyNameString_IsNotADtype()
    {
        new Action(() => np.issubdtype(NPTypeCode.Int32, "integer")).Should().Throw<NotSupportedException>()
            .WithMessage("data type 'integer' not understood");
        new Action(() => np.issubdtype(NPTypeCode.Int32, "invalid_category")).Should().Throw<NotSupportedException>()
            .WithMessage("data type 'invalid_category' not understood");
    }

    /// <summary>
    /// "float" is float64 and "complex" complex128 (NumPy's dtype names), not the families: NumPy 2.4.2 answers
    /// <c>issubdtype(np.float32, 'float')</c> False, which NumSharp's old family reading answered True.
    /// </summary>
    [TestMethod]
    public void IsSubdtype_PythonTypeNames_AreDtypes()
    {
        np.issubdtype(np.float64, "float").Should().BeTrue();
        np.issubdtype(np.float32, "float").Should().BeFalse();
        np.issubdtype(np.complex128, "complex").Should().BeTrue();
        np.issubdtype(np.int64, "int").Should().BeTrue();   // "int" is NumPy's int_ (intp): int64 on a 64-bit host
        np.issubdtype(np.int32, "int").Should().BeFalse();
        np.issubdtype(np.float64, "f8").Should().BeTrue();
    }

    /// <summary>NumPy's <c>None</c> is float64 (<c>dtype(None)</c>): <c>np.issubdtype(None, np.floating)</c> is True.</summary>
    [TestMethod]
    public void IsSubdtype_Null_IsFloat64()
    {
        np.issubdtype((DType)null, np.floating).Should().BeTrue();
        np.issubdtype(np.float64, (DType)null).Should().BeTrue();
        np.issubdtype(np.int64, (string)null).Should().BeFalse();
    }

    #endregion
}
