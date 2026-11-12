using System;
using System.Linq;

namespace NumSharp.Tests.APIs;

/// <summary>
/// Battle tests for np.finfo - comprehensive coverage of all floating point types.
/// </summary>
[TestClass]
public class NpFInfoBattleTests
{
    #region All Float Types Coverage

    [TestMethod]
    public void FInfo_Single_AllProperties()
    {
        var info = np.finfo(NPTypeCode.Single);
        info.bits.Should().Be(32);
        info.precision.Should().Be(6);
        info.maxexp.Should().Be(128);
        info.minexp.Should().Be(-126);   // NumPy 2.4.2 (2**-126 is smallest_normal); -125 before API-parity batch 7
        // max/min are stored as double, so compare with tolerance
        info.max.Should().BeGreaterThan(3.4e38);
        info.min.Should().BeLessThan(-3.4e38);
        info.tiny.Should().Be(info.smallest_normal);
        // smallest_subnormal for float is ~1.4e-45
        info.smallest_subnormal.Should().BeLessThan(1e-44);
        info.resolution.Should().Be(1e-6);
        // eps for float32 ~ 1.19e-07
        info.eps.Should().BeGreaterThan(1e-7);
        info.eps.Should().BeLessThan(2e-7);
    }

    [TestMethod]
    public void FInfo_Double_AllProperties()
    {
        var info = np.finfo(NPTypeCode.Double);
        info.bits.Should().Be(64);
        info.precision.Should().Be(15);
        info.maxexp.Should().Be(1024);
        info.minexp.Should().Be(-1022);  // NumPy 2.4.2 (2**-1022 is smallest_normal); -1021 before API-parity batch 7
        info.max.Should().Be(double.MaxValue);
        info.min.Should().Be(-double.MaxValue);
        info.smallest_subnormal.Should().Be(double.Epsilon);
        info.resolution.Should().Be(1e-15);
        // eps ~ 2.22e-16
        info.eps.Should().BeGreaterThan(2e-16);
        info.eps.Should().BeLessThan(3e-16);
    }

    [TestMethod]
    public void FInfo_Decimal_AllProperties()
    {
        var info = np.finfo(NPTypeCode.Decimal);
        info.bits.Should().Be(128);
        info.precision.Should().Be(28);
        info.resolution.Should().Be(1e-28);
    }

    #endregion

    #region Error Cases

    [TestMethod]
    public void FInfo_Int32_Throws()
    {
        new Action(() => np.finfo(NPTypeCode.Int32)).Should().Throw<Exception>();
    }

    [TestMethod]
    public void FInfo_Boolean_Throws()
    {
        new Action(() => np.finfo(NPTypeCode.Boolean)).Should().Throw<Exception>();
    }

    [TestMethod]
    public void FInfo_Byte_Throws()
    {
        new Action(() => np.finfo(NPTypeCode.Byte)).Should().Throw<Exception>();
    }

    [TestMethod]
    public void FInfo_Empty_Throws()
    {
        new Action(() => np.finfo(NPTypeCode.Empty)).Should().Throw<Exception>();
    }

    [TestMethod]
    public void FInfo_NullType_Throws()
    {
        new Action(() => np.finfo((Type)null!)).Should().Throw<Exception>();
    }

    [TestMethod]
    public void FInfo_HasNoArrayOverload_AsNumPyRefusesAnArray()
    {
        // NumPy 2.4.2: np.finfo(np.ones(2)) is "data type <class 'numpy.object_'> not inexact" (an array is not a dtype);
        // API-parity batch 7 removed the NDArray form, so a C# array argument does not compile: pass arr.dtype.
        typeof(np).GetMethods().Where(m => m.Name == "finfo")
            .Any(m => m.GetParameters().Any(p => p.ParameterType == typeof(NDArray))).Should().BeFalse();
    }

    [TestMethod]
    public void FInfo_EmptyDtypeString_Throws()
    {
        new Action(() => np.finfo("")).Should().Throw<Exception>();
    }

    [TestMethod]
    public void FInfo_InvalidDtypeString_Throws()
    {
        new Action(() => np.finfo("int32")).Should().Throw<Exception>();
    }

    #endregion

    #region C# type spellings (the generic finfo<T> went in API-parity batch 9, OD7)

    /// <summary>
    ///     The generic <c>finfo&lt;float&gt;()</c> went (OD7): <c>np.finfo(typeof(float))</c> is NumPy's
    ///     <c>np.finfo(np.float32)</c> (32 bits, precision 6).
    /// </summary>
    [TestMethod]
    public void FInfo_TypeofFloat()
    {
        // NumPy: np.finfo(np.float32).bits == 32, .precision == 6.
        var info = np.finfo(typeof(float));
        info.bits.Should().Be(32);
        info.precision.Should().Be(6);
    }

    /// <summary>
    ///     <c>np.finfo(typeof(double))</c> is NumPy's <c>np.finfo(np.float64)</c> (64 bits, precision 15).
    /// </summary>
    [TestMethod]
    public void FInfo_TypeofDouble()
    {
        var info = np.finfo(typeof(double));
        info.bits.Should().Be(64);
        info.precision.Should().Be(15);
    }

    /// <summary>
    ///     NumSharp's decimal dtype is inexact, so <c>np.finfo(typeof(decimal))</c> answers (128 bits).
    /// </summary>
    [TestMethod]
    public void FInfo_TypeofDecimal()
    {
        var info = np.finfo(typeof(decimal));
        info.bits.Should().Be(128);
    }

    /// <summary>
    ///     An integer type is NumPy's <c>ValueError: data type &lt;class 'numpy.int32'&gt; not inexact</c> (the generic
    ///     form threw some exception; the dtype form throws NumPy's).
    /// </summary>
    [TestMethod]
    public void FInfo_TypeofInt_IsNumPysValueError()
    {
        // NumPy: np.finfo(np.int32) -> ValueError: data type <class 'numpy.int32'> not inexact
        new Action(() => np.finfo(typeof(int))).Should().Throw<ValueError>()
            .WithMessage("data type <class 'numpy.int32'> not inexact");
    }

    #endregion

    #region NDArray Overload Tests

    [TestMethod]
    public void FInfo_ArrayDtype_Float32()
    {
        var arr = np.array(new float[] { 1.0f, 2.0f, 3.0f });
        var info = np.finfo(arr.dtype);
        info.bits.Should().Be(32);
    }

    [TestMethod]
    public void FInfo_ArrayDtype_Float64()
    {
        var arr = np.array(new double[] { 1.0, 2.0, 3.0 });
        var info = np.finfo(arr.dtype);
        info.bits.Should().Be(64);
    }

    [TestMethod]
    public void FInfo_ArrayDtype_Int_Throws()
    {
        // NumPy 2.4.2: np.finfo(np.int32) names the scalar class.
        var arr = np.array(new int[] { 1, 2, 3 });
        new Action(() => np.finfo(arr.dtype)).Should().Throw<ValueError>()
            .WithMessage("data type <class 'numpy.int32'> not inexact");
    }

    #endregion

    #region String dtype Overload Tests

    [TestMethod]
    public void FInfo_String_Float()
    {
        // NumPy parity: np.finfo("float") → 64 bits (float is an alias for float64)
        var info = np.finfo("float");
        info.bits.Should().Be(64);
    }

    [TestMethod]
    public void FInfo_String_Single()
    {
        var info = np.finfo("single");
        info.bits.Should().Be(32);
    }

    [TestMethod]
    public void FInfo_String_Double()
    {
        var info = np.finfo("double");
        info.bits.Should().Be(64);
    }

    [TestMethod]
    public void FInfo_String_Float32()
    {
        var info = np.finfo("float32");
        info.bits.Should().Be(32);
    }

    [TestMethod]
    public void FInfo_String_Float64()
    {
        var info = np.finfo("float64");
        info.bits.Should().Be(64);
    }

    // NumPy parity: np.finfo("float16") / np.finfo("half") should return bits=16.
    // NumSharp's np.finfo(NPTypeCode) constructor doesn't include Half in IsFloatType
    // (see np.finfo.cs:164). Tracked as a separate gap — np.dtype resolves "float16"
    // to Half correctly, but np.finfo throws "not inexact".

    #endregion

    #region Epsilon Verification

    [TestMethod]
    public void FInfo_Double_EpsIsNextFloat()
    {
        var info = np.finfo(NPTypeCode.Double);
        double expected = System.Math.BitIncrement(1.0) - 1.0;
        info.eps.Should().Be(expected);
    }

    [TestMethod]
    public void FInfo_Single_EpsIsNextFloat()
    {
        var info = np.finfo(NPTypeCode.Single);
        // Must use MathF for float operations
        double expected = MathF.BitIncrement(1.0f) - 1.0f;
        info.eps.Should().Be(expected);
    }

    #endregion

    #region ToString Tests

    [TestMethod]
    public void FInfo_ToString_ContainsExpectedInfo()
    {
        var info = np.finfo(NPTypeCode.Double);
        var str = info.ToString();
        str.Should().Contain("resolution=");
        str.Should().Contain("float64");
    }

    #endregion
}
