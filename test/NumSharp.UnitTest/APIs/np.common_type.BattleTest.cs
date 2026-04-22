using System;

namespace NumSharp.UnitTest.APIs;

/// <summary>
/// Battle tests for np.common_type against NumPy 2.x.
///
/// NumPy rules verified:
///   - Boolean input: raises TypeError "can't get common type for non-numeric array"
///   - Any integer/char (without complex/decimal): returns float64 (Double)
///   - Pure float inputs: returns max float (Half &lt; Single &lt; Double)
///   - Any int mixed with any float: returns float64 (int forces at-least-float64)
///   - Any complex input: returns complex (NumSharp maps complex64/128 → Complex)
///   - Any decimal input (NumSharp extension): returns Decimal
///
/// NumPy reference commands (python_run):
///   np.common_type(np.array([1],dtype=np.int8)) → float64
///   np.common_type(np.array([1.0],dtype=np.float16)) → float16
///   np.common_type(np.array([1.0],dtype=np.float16), np.array([1],dtype=np.int8)) → float64
/// </summary>
[TestClass]
public class NpCommonTypeBattleTests
{
    #region Boolean input raises TypeError (NumPy parity)

    [TestMethod]
    public void CommonType_Bool_Throws()
    {
        // NumPy: np.common_type(np.array([True])) -> TypeError "can't get common type for non-numeric array"
        new Action(() => np.common_type_code(NPTypeCode.Boolean))
            .Should().Throw<TypeError>();
    }

    [TestMethod]
    public void CommonType_BoolArray_Throws()
    {
        var arr = np.array(new bool[] { true, false });
        new Action(() => np.common_type_code(arr)).Should().Throw<TypeError>();
    }

    [TestMethod]
    public void CommonType_BoolMixedWithInt32_Throws()
    {
        new Action(() => np.common_type_code(NPTypeCode.Boolean, NPTypeCode.Int32))
            .Should().Throw<TypeError>();
    }

    [TestMethod]
    public void CommonType_Int32MixedWithBool_Throws()
    {
        new Action(() => np.common_type_code(NPTypeCode.Int32, NPTypeCode.Boolean))
            .Should().Throw<TypeError>();
    }

    [TestMethod]
    public void CommonType_BoolMixedWithFloat_Throws()
    {
        new Action(() => np.common_type_code(NPTypeCode.Boolean, NPTypeCode.Double))
            .Should().Throw<TypeError>();
    }

    #endregion

    #region Single integer input → Double

    [TestMethod] public void CommonType_SByte_ReturnsDouble()  => np.common_type_code(NPTypeCode.SByte).Should().Be(NPTypeCode.Double);
    [TestMethod] public void CommonType_Byte_ReturnsDouble()   => np.common_type_code(NPTypeCode.Byte).Should().Be(NPTypeCode.Double);
    [TestMethod] public void CommonType_Int16_ReturnsDouble()  => np.common_type_code(NPTypeCode.Int16).Should().Be(NPTypeCode.Double);
    [TestMethod] public void CommonType_UInt16_ReturnsDouble() => np.common_type_code(NPTypeCode.UInt16).Should().Be(NPTypeCode.Double);
    [TestMethod] public void CommonType_Int32_ReturnsDouble()  => np.common_type_code(NPTypeCode.Int32).Should().Be(NPTypeCode.Double);
    [TestMethod] public void CommonType_UInt32_ReturnsDouble() => np.common_type_code(NPTypeCode.UInt32).Should().Be(NPTypeCode.Double);
    [TestMethod] public void CommonType_Int64_ReturnsDouble()  => np.common_type_code(NPTypeCode.Int64).Should().Be(NPTypeCode.Double);
    [TestMethod] public void CommonType_UInt64_ReturnsDouble() => np.common_type_code(NPTypeCode.UInt64).Should().Be(NPTypeCode.Double);
    [TestMethod] public void CommonType_Char_ReturnsDouble()   => np.common_type_code(NPTypeCode.Char).Should().Be(NPTypeCode.Double);

    #endregion

    #region Single float input → preserved

    [TestMethod] public void CommonType_Half_ReturnsHalf()     => np.common_type_code(NPTypeCode.Half).Should().Be(NPTypeCode.Half);
    [TestMethod] public void CommonType_Single_ReturnsSingle() => np.common_type_code(NPTypeCode.Single).Should().Be(NPTypeCode.Single);
    [TestMethod] public void CommonType_Double_ReturnsDouble() => np.common_type_code(NPTypeCode.Double).Should().Be(NPTypeCode.Double);

    #endregion

    #region Single complex/decimal

    [TestMethod] public void CommonType_Complex_ReturnsComplex() => np.common_type_code(NPTypeCode.Complex).Should().Be(NPTypeCode.Complex);
    [TestMethod] public void CommonType_Decimal_ReturnsDecimal() => np.common_type_code(NPTypeCode.Decimal).Should().Be(NPTypeCode.Decimal);

    #endregion

    #region Pure float combinations → max float

    [TestMethod] public void CommonType_Half_Half_ReturnsHalf()       => np.common_type_code(NPTypeCode.Half, NPTypeCode.Half).Should().Be(NPTypeCode.Half);
    [TestMethod] public void CommonType_Half_Single_ReturnsSingle()   => np.common_type_code(NPTypeCode.Half, NPTypeCode.Single).Should().Be(NPTypeCode.Single);
    [TestMethod] public void CommonType_Half_Double_ReturnsDouble()   => np.common_type_code(NPTypeCode.Half, NPTypeCode.Double).Should().Be(NPTypeCode.Double);
    [TestMethod] public void CommonType_Single_Half_ReturnsSingle()   => np.common_type_code(NPTypeCode.Single, NPTypeCode.Half).Should().Be(NPTypeCode.Single);
    [TestMethod] public void CommonType_Single_Single_ReturnsSingle() => np.common_type_code(NPTypeCode.Single, NPTypeCode.Single).Should().Be(NPTypeCode.Single);
    [TestMethod] public void CommonType_Single_Double_ReturnsDouble() => np.common_type_code(NPTypeCode.Single, NPTypeCode.Double).Should().Be(NPTypeCode.Double);
    [TestMethod] public void CommonType_Double_Single_ReturnsDouble() => np.common_type_code(NPTypeCode.Double, NPTypeCode.Single).Should().Be(NPTypeCode.Double);
    [TestMethod] public void CommonType_Double_Double_ReturnsDouble() => np.common_type_code(NPTypeCode.Double, NPTypeCode.Double).Should().Be(NPTypeCode.Double);

    [TestMethod]
    public void CommonType_Half_Single_Double_ReturnsDouble()
    {
        // NumPy: np.common_type(f16, f32, f64) -> float64
        np.common_type_code(NPTypeCode.Half, NPTypeCode.Single, NPTypeCode.Double)
            .Should().Be(NPTypeCode.Double);
    }

    #endregion

    #region Integer + Integer → Double (all combinations)

    [TestMethod] public void CommonType_SByte_SByte_ReturnsDouble() => np.common_type_code(NPTypeCode.SByte, NPTypeCode.SByte).Should().Be(NPTypeCode.Double);
    [TestMethod] public void CommonType_SByte_Byte_ReturnsDouble()  => np.common_type_code(NPTypeCode.SByte, NPTypeCode.Byte).Should().Be(NPTypeCode.Double);
    [TestMethod] public void CommonType_Byte_Int32_ReturnsDouble()  => np.common_type_code(NPTypeCode.Byte, NPTypeCode.Int32).Should().Be(NPTypeCode.Double);
    [TestMethod] public void CommonType_Int16_UInt16_ReturnsDouble()=> np.common_type_code(NPTypeCode.Int16, NPTypeCode.UInt16).Should().Be(NPTypeCode.Double);
    [TestMethod] public void CommonType_Int32_Int64_ReturnsDouble() => np.common_type_code(NPTypeCode.Int32, NPTypeCode.Int64).Should().Be(NPTypeCode.Double);
    [TestMethod] public void CommonType_Int64_UInt64_ReturnsDouble()=> np.common_type_code(NPTypeCode.Int64, NPTypeCode.UInt64).Should().Be(NPTypeCode.Double);

    [TestMethod]
    public void CommonType_ThreeInts_ReturnsDouble()
    {
        // NumPy: np.common_type(i8, i32, i64) -> float64
        np.common_type_code(NPTypeCode.SByte, NPTypeCode.Int32, NPTypeCode.Int64)
            .Should().Be(NPTypeCode.Double);
    }

    #endregion

    #region Integer + Float → Double (any int forces float64)

    [TestMethod] public void CommonType_SByte_Half_ReturnsDouble()   => np.common_type_code(NPTypeCode.SByte, NPTypeCode.Half).Should().Be(NPTypeCode.Double);
    [TestMethod] public void CommonType_SByte_Single_ReturnsDouble() => np.common_type_code(NPTypeCode.SByte, NPTypeCode.Single).Should().Be(NPTypeCode.Double);
    [TestMethod] public void CommonType_SByte_Double_ReturnsDouble() => np.common_type_code(NPTypeCode.SByte, NPTypeCode.Double).Should().Be(NPTypeCode.Double);
    [TestMethod] public void CommonType_Int16_Half_ReturnsDouble()   => np.common_type_code(NPTypeCode.Int16, NPTypeCode.Half).Should().Be(NPTypeCode.Double);
    [TestMethod] public void CommonType_Int32_Half_ReturnsDouble()   => np.common_type_code(NPTypeCode.Int32, NPTypeCode.Half).Should().Be(NPTypeCode.Double);
    [TestMethod] public void CommonType_Int64_Half_ReturnsDouble()   => np.common_type_code(NPTypeCode.Int64, NPTypeCode.Half).Should().Be(NPTypeCode.Double);
    [TestMethod] public void CommonType_Int32_Single_ReturnsDouble() => np.common_type_code(NPTypeCode.Int32, NPTypeCode.Single).Should().Be(NPTypeCode.Double);
    [TestMethod] public void CommonType_UInt64_Single_ReturnsDouble()=> np.common_type_code(NPTypeCode.UInt64, NPTypeCode.Single).Should().Be(NPTypeCode.Double);
    [TestMethod] public void CommonType_Half_Int32_ReturnsDouble()   => np.common_type_code(NPTypeCode.Half, NPTypeCode.Int32).Should().Be(NPTypeCode.Double);

    [TestMethod]
    public void CommonType_MixedIntsAndFloats_ReturnsDouble()
    {
        // NumPy: np.common_type(i8, f16, f32) -> float64 (any int wins)
        np.common_type_code(NPTypeCode.SByte, NPTypeCode.Half, NPTypeCode.Single)
            .Should().Be(NPTypeCode.Double);
    }

    #endregion

    #region Complex combinations → Complex (NumSharp has one complex type = complex128)

    [TestMethod] public void CommonType_Complex_Complex_ReturnsComplex() => np.common_type_code(NPTypeCode.Complex, NPTypeCode.Complex).Should().Be(NPTypeCode.Complex);
    [TestMethod] public void CommonType_Complex_Half_ReturnsComplex()    => np.common_type_code(NPTypeCode.Complex, NPTypeCode.Half).Should().Be(NPTypeCode.Complex);
    [TestMethod] public void CommonType_Complex_Single_ReturnsComplex()  => np.common_type_code(NPTypeCode.Complex, NPTypeCode.Single).Should().Be(NPTypeCode.Complex);
    [TestMethod] public void CommonType_Complex_Double_ReturnsComplex()  => np.common_type_code(NPTypeCode.Complex, NPTypeCode.Double).Should().Be(NPTypeCode.Complex);
    [TestMethod] public void CommonType_Complex_Int8_ReturnsComplex()    => np.common_type_code(NPTypeCode.Complex, NPTypeCode.SByte).Should().Be(NPTypeCode.Complex);
    [TestMethod] public void CommonType_Complex_Int32_ReturnsComplex()   => np.common_type_code(NPTypeCode.Complex, NPTypeCode.Int32).Should().Be(NPTypeCode.Complex);
    [TestMethod] public void CommonType_Complex_Int64_ReturnsComplex()   => np.common_type_code(NPTypeCode.Complex, NPTypeCode.Int64).Should().Be(NPTypeCode.Complex);
    [TestMethod] public void CommonType_Int32_Complex_ReturnsComplex()   => np.common_type_code(NPTypeCode.Int32, NPTypeCode.Complex).Should().Be(NPTypeCode.Complex);
    [TestMethod] public void CommonType_Float_Complex_ReturnsComplex()   => np.common_type_code(NPTypeCode.Double, NPTypeCode.Complex).Should().Be(NPTypeCode.Complex);

    #endregion

    #region Decimal combinations (NumSharp extension - dominates over int/float)

    [TestMethod] public void CommonType_Decimal_Half_ReturnsDecimal()    => np.common_type_code(NPTypeCode.Decimal, NPTypeCode.Half).Should().Be(NPTypeCode.Decimal);
    [TestMethod] public void CommonType_Decimal_Single_ReturnsDecimal()  => np.common_type_code(NPTypeCode.Decimal, NPTypeCode.Single).Should().Be(NPTypeCode.Decimal);
    [TestMethod] public void CommonType_Decimal_Double_ReturnsDecimal()  => np.common_type_code(NPTypeCode.Decimal, NPTypeCode.Double).Should().Be(NPTypeCode.Decimal);
    [TestMethod] public void CommonType_Decimal_Int32_ReturnsDecimal()   => np.common_type_code(NPTypeCode.Decimal, NPTypeCode.Int32).Should().Be(NPTypeCode.Decimal);

    [TestMethod]
    public void CommonType_Decimal_Complex_ReturnsComplex()
    {
        // Complex beats Decimal in NumSharp (Complex is more general).
        np.common_type_code(NPTypeCode.Complex, NPTypeCode.Decimal).Should().Be(NPTypeCode.Complex);
    }

    #endregion

    #region NDArray overloads

    [TestMethod]
    public void CommonType_SByteArray_ReturnsDouble()
    {
        var arr = np.array(new sbyte[] { 1, -2, 3 });
        np.common_type_code(arr).Should().Be(NPTypeCode.Double);
    }

    [TestMethod]
    public void CommonType_HalfArray_ReturnsHalf()
    {
        var arr = np.array(new Half[] { (Half)1, (Half)2 });
        np.common_type_code(arr).Should().Be(NPTypeCode.Half);
    }

    [TestMethod]
    public void CommonType_ComplexArray_ReturnsComplex()
    {
        var arr = np.array(new System.Numerics.Complex[] { new(1, 0), new(2, 3) });
        np.common_type_code(arr).Should().Be(NPTypeCode.Complex);
    }

    [TestMethod]
    public void CommonType_Float32Array_ReturnsSingle()
    {
        var arr = np.array(new float[] { 1.0f, 2.0f });
        np.common_type_code(arr).Should().Be(NPTypeCode.Single);
    }

    [TestMethod]
    public void CommonType_Float64Array_ReturnsDouble()
    {
        var arr = np.array(new double[] { 1.0, 2.0 });
        np.common_type_code(arr).Should().Be(NPTypeCode.Double);
    }

    [TestMethod]
    public void CommonType_Int32Array_ReturnsDouble()
    {
        var arr = np.array(new int[] { 1, 2, 3 });
        np.common_type_code(arr).Should().Be(NPTypeCode.Double);
    }

    [TestMethod]
    public void CommonType_ByteArray_ReturnsDouble()
    {
        var arr = np.array(new byte[] { 1, 2, 3 });
        np.common_type_code(arr).Should().Be(NPTypeCode.Double);
    }

    [TestMethod]
    public void CommonType_HalfArray_Int32Array_ReturnsDouble()
    {
        // Mixed half + int → NumPy promotes to float64.
        var h = np.array(new Half[] { (Half)1 });
        var i = np.array(new int[] { 1 });
        np.common_type_code(h, i).Should().Be(NPTypeCode.Double);
    }

    [TestMethod]
    public void CommonType_Float32AndFloat64_ReturnsDouble()
    {
        var a = np.array(new float[] { 1.0f });
        var b = np.array(new double[] { 1.0 });
        np.common_type_code(a, b).Should().Be(NPTypeCode.Double);
    }

    [TestMethod]
    public void CommonType_AllFloat32_ReturnsSingle()
    {
        var a = np.array(new float[] { 1.0f });
        var b = np.array(new float[] { 2.0f });
        np.common_type_code(a, b).Should().Be(NPTypeCode.Single);
    }

    #endregion

    #region Type overload (CLR Type return)

    [TestMethod]
    public void CommonType_Type_Int32_ReturnsDouble()
    {
        var arr = np.array(new int[] { 1, 2 });
        np.common_type(arr).Should().Be(typeof(double));
    }

    [TestMethod]
    public void CommonType_Type_Float32_ReturnsSingle()
    {
        var arr = np.array(new float[] { 1.0f, 2.0f });
        np.common_type(arr).Should().Be(typeof(float));
    }

    [TestMethod]
    public void CommonType_Type_Half_ReturnsHalf()
    {
        var arr = np.array(new Half[] { (Half)1 });
        np.common_type(arr).Should().Be(typeof(Half));
    }

    [TestMethod]
    public void CommonType_Type_Complex_ReturnsComplex()
    {
        var arr = np.array(new System.Numerics.Complex[] { new(1, 0) });
        np.common_type(arr).Should().Be(typeof(System.Numerics.Complex));
    }

    [TestMethod]
    public void CommonType_Type_Bool_Throws()
    {
        var arr = np.array(new bool[] { true });
        new Action(() => np.common_type(arr)).Should().Throw<TypeError>();
    }

    #endregion

    #region Argument validation

    [TestMethod]
    public void CommonType_EmptyArrays_Throws()
    {
        new Action(() => np.common_type_code(Array.Empty<NDArray>())).Should().Throw<Exception>();
    }

    [TestMethod]
    public void CommonType_NullArrays_Throws()
    {
        new Action(() => np.common_type_code((NDArray[])null!)).Should().Throw<Exception>();
    }

    [TestMethod]
    public void CommonTypeCode_EmptyTypes_Throws()
    {
        new Action(() => np.common_type_code(Array.Empty<NPTypeCode>())).Should().Throw<Exception>();
    }

    #endregion
}
