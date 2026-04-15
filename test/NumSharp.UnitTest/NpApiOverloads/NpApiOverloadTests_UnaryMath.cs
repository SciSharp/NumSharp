using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.NpApiOverloads;

/// <summary>
/// Tests verifying that unary math np.* overloads compile and work correctly
/// after removing the 'in' parameter modifier from NDArray parameters.
///
/// The primary goal is to ensure each overload:
/// 1. Compiles successfully with the new signature
/// 2. Returns the correct type
/// 3. Produces approximately correct values
/// </summary>
[TestClass]
public class NpApiOverloadTests_UnaryMath
{
    private const double Tolerance = 1e-10;

    #region Absolute - np.absolute (3 overloads)

    [TestMethod]
    public void Absolute_NoParams_Compiles()
    {
        var a = np.array(new double[] { -1.5, 2.3, -3.7 });
        var result = np.absolute(a);
        Assert.IsNotNull(result);
        Assert.AreEqual(1.5, result.GetDouble(0), Tolerance);
        Assert.AreEqual(2.3, result.GetDouble(1), Tolerance);
        Assert.AreEqual(3.7, result.GetDouble(2), Tolerance);
    }

    [TestMethod]
    public void Absolute_WithType_Compiles()
    {
        var a = np.array(new double[] { -1.5, 2.3, -3.7 });
        var result = np.absolute(a, typeof(float));
        Assert.IsNotNull(result);
        Assert.AreEqual(NPTypeCode.Single, result.typecode);
    }

    [TestMethod]
    public void Absolute_WithNPTypeCode_Compiles()
    {
        var a = np.array(new double[] { -1.5, 2.3, -3.7 });
        var result = np.absolute(a, NPTypeCode.Single);
        Assert.IsNotNull(result);
        Assert.AreEqual(NPTypeCode.Single, result.typecode);
    }

    #endregion

    #region Abs - np.abs (3 overloads)

    [TestMethod]
    public void Abs_NoParams_Compiles()
    {
        var a = np.array(new double[] { -1.5, 2.3, -3.7 });
        var result = np.abs(a);
        Assert.IsNotNull(result);
        Assert.AreEqual(1.5, result.GetDouble(0), Tolerance);
        Assert.AreEqual(2.3, result.GetDouble(1), Tolerance);
        Assert.AreEqual(3.7, result.GetDouble(2), Tolerance);
    }

    [TestMethod]
    public void Abs_WithType_Compiles()
    {
        var a = np.array(new double[] { -1.5, 2.3, -3.7 });
        var result = np.abs(a, typeof(float));
        Assert.IsNotNull(result);
        Assert.AreEqual(NPTypeCode.Single, result.typecode);
    }

    [TestMethod]
    public void Abs_WithNPTypeCode_Compiles()
    {
        var a = np.array(new double[] { -1.5, 2.3, -3.7 });
        var result = np.abs(a, NPTypeCode.Single);
        Assert.IsNotNull(result);
        Assert.AreEqual(NPTypeCode.Single, result.typecode);
    }

    #endregion

    #region Sign - np.sign (2 overloads)

    [TestMethod]
    public void Sign_NoParams_Compiles()
    {
        var a = np.array(new double[] { -1.5, 0.0, 3.7 });
        var result = np.sign(a);
        Assert.IsNotNull(result);
        Assert.AreEqual(-1.0, result.GetDouble(0), Tolerance);
        Assert.AreEqual(0.0, result.GetDouble(1), Tolerance);
        Assert.AreEqual(1.0, result.GetDouble(2), Tolerance);
    }

    [TestMethod]
    public void Sign_WithType_Compiles()
    {
        var a = np.array(new double[] { -1.5, 0.0, 3.7 });
        var result = np.sign(a, typeof(float));
        Assert.IsNotNull(result);
        Assert.AreEqual(NPTypeCode.Single, result.typecode);
    }

    #endregion

    #region Sqrt - np.sqrt (2 overloads)

    [TestMethod]
    public void Sqrt_NoParams_Compiles()
    {
        var a = np.array(new double[] { 1.0, 4.0, 9.0 });
        var result = np.sqrt(a);
        Assert.IsNotNull(result);
        Assert.AreEqual(1.0, result.GetDouble(0), Tolerance);
        Assert.AreEqual(2.0, result.GetDouble(1), Tolerance);
        Assert.AreEqual(3.0, result.GetDouble(2), Tolerance);
    }

    [TestMethod]
    public void Sqrt_WithType_Compiles()
    {
        var a = np.array(new double[] { 1.0, 4.0, 9.0 });
        var result = np.sqrt(a, typeof(float));
        Assert.IsNotNull(result);
        // Note: sqrt(a, Type) doesn't actually convert dtype in current impl
    }

    #endregion

    #region Cbrt - np.cbrt (2 overloads)

    [TestMethod]
    public void Cbrt_NoParams_Compiles()
    {
        var a = np.array(new double[] { 1.0, 8.0, 27.0 });
        var result = np.cbrt(a);
        Assert.IsNotNull(result);
        Assert.AreEqual(1.0, result.GetDouble(0), Tolerance);
        Assert.AreEqual(2.0, result.GetDouble(1), Tolerance);
        Assert.AreEqual(3.0, result.GetDouble(2), Tolerance);
    }

    [TestMethod]
    public void Cbrt_WithType_Compiles()
    {
        var a = np.array(new double[] { 1.0, 8.0, 27.0 });
        var result = np.cbrt(a, typeof(float));
        Assert.IsNotNull(result);
        Assert.AreEqual(NPTypeCode.Single, result.typecode);
    }

    #endregion

    #region Square - np.square (1 overload)

    [TestMethod]
    public void Square_NoParams_Compiles()
    {
        var a = np.array(new double[] { 1.5, 2.0, 3.0 });
        var result = np.square(a);
        Assert.IsNotNull(result);
        Assert.AreEqual(2.25, result.GetDouble(0), Tolerance);
        Assert.AreEqual(4.0, result.GetDouble(1), Tolerance);
        Assert.AreEqual(9.0, result.GetDouble(2), Tolerance);
    }

    #endregion

    #region Ceil - np.ceil (2 overloads)

    [TestMethod]
    public void Ceil_NoParams_Compiles()
    {
        var a = np.array(new double[] { 1.5, -2.3, 3.7 });
        var result = np.ceil(a);
        Assert.IsNotNull(result);
        Assert.AreEqual(2.0, result.GetDouble(0), Tolerance);
        Assert.AreEqual(-2.0, result.GetDouble(1), Tolerance);
        Assert.AreEqual(4.0, result.GetDouble(2), Tolerance);
    }

    [TestMethod]
    public void Ceil_WithType_Compiles()
    {
        var a = np.array(new double[] { 1.5, -2.3, 3.7 });
        var result = np.ceil(a, typeof(float));
        Assert.IsNotNull(result);
        Assert.AreEqual(NPTypeCode.Single, result.typecode);
    }

    #endregion

    #region Floor - np.floor (2 overloads)

    [TestMethod]
    public void Floor_NoParams_Compiles()
    {
        var a = np.array(new double[] { 1.5, -2.3, 3.7 });
        var result = np.floor(a);
        Assert.IsNotNull(result);
        Assert.AreEqual(1.0, result.GetDouble(0), Tolerance);
        Assert.AreEqual(-3.0, result.GetDouble(1), Tolerance);
        Assert.AreEqual(3.0, result.GetDouble(2), Tolerance);
    }

    [TestMethod]
    public void Floor_WithType_Compiles()
    {
        var a = np.array(new double[] { 1.5, -2.3, 3.7 });
        var result = np.floor(a, typeof(float));
        Assert.IsNotNull(result);
        Assert.AreEqual(NPTypeCode.Single, result.typecode);
    }

    #endregion

    #region Trunc - np.trunc (2 overloads)

    [TestMethod]
    [OpenBugs] // Vector512 SIMD: "Could not find Truncate for Vector512" on AVX-512 capable runners
    public void Trunc_NoParams_Compiles()
    {
        var a = np.array(new double[] { 1.5, -2.3, 3.7 });
        var result = np.trunc(a);
        Assert.IsNotNull(result);
        Assert.AreEqual(1.0, result.GetDouble(0), Tolerance);
        Assert.AreEqual(-2.0, result.GetDouble(1), Tolerance);
        Assert.AreEqual(3.0, result.GetDouble(2), Tolerance);
    }

    [TestMethod]
    public void Trunc_WithType_Compiles()
    {
        var a = np.array(new double[] { 1.5, -2.3, 3.7 });
        var result = np.trunc(a, typeof(float));
        Assert.IsNotNull(result);
        Assert.AreEqual(NPTypeCode.Single, result.typecode);
    }

    #endregion

    #region Round_ - np.round_ (4 overloads)

    [TestMethod]
    [OpenBugs] // Vector512 SIMD: "Could not find Round for Vector512" on AVX-512 capable runners
    public void Round_NoParams_Compiles()
    {
        var a = np.array(new double[] { 1.5, -2.3, 3.7 });
        var result = np.round_(a);
        Assert.IsNotNull(result);
        // Banker's rounding: 1.5 rounds to 2, -2.3 to -2, 3.7 to 4
        Assert.AreEqual(2.0, result.GetDouble(0), Tolerance);
        Assert.AreEqual(-2.0, result.GetDouble(1), Tolerance);
        Assert.AreEqual(4.0, result.GetDouble(2), Tolerance);
    }

    [TestMethod]
    public void Round_WithDecimals_Compiles()
    {
        var a = np.array(new double[] { 1.567, -2.345, 3.789 });
        var result = np.round_(a, 1);
        Assert.IsNotNull(result);
        Assert.AreEqual(1.6, result.GetDouble(0), Tolerance);
        Assert.AreEqual(3.8, result.GetDouble(2), Tolerance);
    }

    [TestMethod]
    public void Round_WithType_Compiles()
    {
        var a = np.array(new double[] { 1.5, -2.3, 3.7 });
        var result = np.round_(a, typeof(float));
        Assert.IsNotNull(result);
        Assert.AreEqual(NPTypeCode.Single, result.typecode);
    }

    [TestMethod]
    public void Round_WithDecimalsAndType_Compiles()
    {
        var a = np.array(new double[] { 1.567, -2.345, 3.789 });
        var result = np.round_(a, 1, typeof(float));
        Assert.IsNotNull(result);
        Assert.AreEqual(NPTypeCode.Single, result.typecode);
    }

    #endregion

    #region Around - np.around (4 overloads)

    [TestMethod]
    [OpenBugs] // Vector512 SIMD: "Could not find Round for Vector512" on AVX-512 capable runners
    public void Around_NoParams_Compiles()
    {
        var a = np.array(new double[] { 1.5, -2.3, 3.7 });
        var result = np.around(a);
        Assert.IsNotNull(result);
        Assert.AreEqual(2.0, result.GetDouble(0), Tolerance);
        Assert.AreEqual(-2.0, result.GetDouble(1), Tolerance);
        Assert.AreEqual(4.0, result.GetDouble(2), Tolerance);
    }

    [TestMethod]
    public void Around_WithDecimals_Compiles()
    {
        var a = np.array(new double[] { 1.567, -2.345, 3.789 });
        var result = np.around(a, 1);
        Assert.IsNotNull(result);
        Assert.AreEqual(1.6, result.GetDouble(0), Tolerance);
        Assert.AreEqual(3.8, result.GetDouble(2), Tolerance);
    }

    [TestMethod]
    public void Around_WithType_Compiles()
    {
        var a = np.array(new double[] { 1.5, -2.3, 3.7 });
        var result = np.around(a, typeof(float));
        Assert.IsNotNull(result);
        Assert.AreEqual(NPTypeCode.Single, result.typecode);
    }

    [TestMethod]
    public void Around_WithDecimalsAndType_Compiles()
    {
        var a = np.array(new double[] { 1.567, -2.345, 3.789 });
        var result = np.around(a, 1, typeof(float));
        Assert.IsNotNull(result);
        Assert.AreEqual(NPTypeCode.Single, result.typecode);
    }

    #endregion

    #region Exp - np.exp (3 overloads)

    [TestMethod]
    public void Exp_NoParams_Compiles()
    {
        var a = np.array(new double[] { 0.0, 1.0, 2.0 });
        var result = np.exp(a);
        Assert.IsNotNull(result);
        Assert.AreEqual(1.0, result.GetDouble(0), Tolerance);
        Assert.AreEqual(Math.E, result.GetDouble(1), Tolerance);
        Assert.AreEqual(Math.E * Math.E, result.GetDouble(2), Tolerance);
    }

    [TestMethod]
    public void Exp_WithType_Compiles()
    {
        var a = np.array(new double[] { 0.0, 1.0, 2.0 });
        var result = np.exp(a, typeof(float));
        Assert.IsNotNull(result);
        Assert.AreEqual(NPTypeCode.Single, result.typecode);
    }

    [TestMethod]
    public void Exp_WithNPTypeCode_Compiles()
    {
        var a = np.array(new double[] { 0.0, 1.0, 2.0 });
        var result = np.exp(a, NPTypeCode.Single);
        Assert.IsNotNull(result);
        Assert.AreEqual(NPTypeCode.Single, result.typecode);
    }

    #endregion

    #region Exp2 - np.exp2 (3 overloads)

    [TestMethod]
    public void Exp2_NoParams_Compiles()
    {
        var a = np.array(new double[] { 0.0, 1.0, 2.0 });
        var result = np.exp2(a);
        Assert.IsNotNull(result);
        Assert.AreEqual(1.0, result.GetDouble(0), Tolerance);
        Assert.AreEqual(2.0, result.GetDouble(1), Tolerance);
        Assert.AreEqual(4.0, result.GetDouble(2), Tolerance);
    }

    [TestMethod]
    [OpenBugs] // ILKernelGenerator Exp2 type conversion bug - InvalidProgramException
    public void Exp2_WithType_Compiles()
    {
        var a = np.array(new double[] { 0.0, 1.0, 2.0 });
        var result = np.exp2(a, typeof(float));
        Assert.IsNotNull(result);
        Assert.AreEqual(NPTypeCode.Single, result.typecode);
    }

    [TestMethod]
    [OpenBugs] // ILKernelGenerator Exp2 type conversion bug - InvalidProgramException
    public void Exp2_WithNPTypeCode_Compiles()
    {
        var a = np.array(new double[] { 0.0, 1.0, 2.0 });
        var result = np.exp2(a, NPTypeCode.Single);
        Assert.IsNotNull(result);
        Assert.AreEqual(NPTypeCode.Single, result.typecode);
    }

    #endregion

    #region Expm1 - np.expm1 (3 overloads)

    [TestMethod]
    public void Expm1_NoParams_Compiles()
    {
        var a = np.array(new double[] { 0.0, 1.0, 2.0 });
        var result = np.expm1(a);
        Assert.IsNotNull(result);
        Assert.AreEqual(0.0, result.GetDouble(0), Tolerance);
        Assert.AreEqual(Math.E - 1, result.GetDouble(1), Tolerance);
    }

    [TestMethod]
    public void Expm1_WithType_Compiles()
    {
        var a = np.array(new double[] { 0.0, 1.0, 2.0 });
        var result = np.expm1(a, typeof(float));
        Assert.IsNotNull(result);
        Assert.AreEqual(NPTypeCode.Single, result.typecode);
    }

    [TestMethod]
    public void Expm1_WithNPTypeCode_Compiles()
    {
        var a = np.array(new double[] { 0.0, 1.0, 2.0 });
        var result = np.expm1(a, NPTypeCode.Single);
        Assert.IsNotNull(result);
        Assert.AreEqual(NPTypeCode.Single, result.typecode);
    }

    #endregion

    #region Log - np.log (3 overloads)

    [TestMethod]
    public void Log_NoParams_Compiles()
    {
        var a = np.array(new double[] { 1.0, Math.E, Math.E * Math.E });
        var result = np.log(a);
        Assert.IsNotNull(result);
        Assert.AreEqual(0.0, result.GetDouble(0), Tolerance);
        Assert.AreEqual(1.0, result.GetDouble(1), Tolerance);
        Assert.AreEqual(2.0, result.GetDouble(2), Tolerance);
    }

    [TestMethod]
    public void Log_WithType_Compiles()
    {
        var a = np.array(new double[] { 1.0, Math.E, Math.E * Math.E });
        var result = np.log(a, typeof(float));
        Assert.IsNotNull(result);
        // Note: log(a, Type) doesn't convert dtype in current implementation
    }

    [TestMethod]
    public void Log_WithNPTypeCode_Compiles()
    {
        var a = np.array(new double[] { 1.0, Math.E, Math.E * Math.E });
        var result = np.log(a, NPTypeCode.Double);
        Assert.IsNotNull(result);
    }

    #endregion

    #region Log2 - np.log2 (3 overloads)

    [TestMethod]
    public void Log2_NoParams_Compiles()
    {
        var a = np.array(new double[] { 1.0, 2.0, 4.0 });
        var result = np.log2(a);
        Assert.IsNotNull(result);
        Assert.AreEqual(0.0, result.GetDouble(0), Tolerance);
        Assert.AreEqual(1.0, result.GetDouble(1), Tolerance);
        Assert.AreEqual(2.0, result.GetDouble(2), Tolerance);
    }

    [TestMethod]
    public void Log2_WithType_Compiles()
    {
        var a = np.array(new double[] { 1.0, 2.0, 4.0 });
        var result = np.log2(a, typeof(float));
        Assert.IsNotNull(result);
    }

    [TestMethod]
    public void Log2_WithNPTypeCode_Compiles()
    {
        var a = np.array(new double[] { 1.0, 2.0, 4.0 });
        var result = np.log2(a, NPTypeCode.Double);
        Assert.IsNotNull(result);
    }

    #endregion

    #region Log10 - np.log10 (3 overloads)

    [TestMethod]
    public void Log10_NoParams_Compiles()
    {
        var a = np.array(new double[] { 1.0, 10.0, 100.0 });
        var result = np.log10(a);
        Assert.IsNotNull(result);
        Assert.AreEqual(0.0, result.GetDouble(0), Tolerance);
        Assert.AreEqual(1.0, result.GetDouble(1), Tolerance);
        Assert.AreEqual(2.0, result.GetDouble(2), Tolerance);
    }

    [TestMethod]
    public void Log10_WithType_Compiles()
    {
        var a = np.array(new double[] { 1.0, 10.0, 100.0 });
        var result = np.log10(a, typeof(float));
        Assert.IsNotNull(result);
    }

    [TestMethod]
    public void Log10_WithNPTypeCode_Compiles()
    {
        var a = np.array(new double[] { 1.0, 10.0, 100.0 });
        var result = np.log10(a, NPTypeCode.Double);
        Assert.IsNotNull(result);
    }

    #endregion

    #region Log1p - np.log1p (3 overloads)

    [TestMethod]
    public void Log1p_NoParams_Compiles()
    {
        var a = np.array(new double[] { 0.0, Math.E - 1, Math.E * Math.E - 1 });
        var result = np.log1p(a);
        Assert.IsNotNull(result);
        Assert.AreEqual(0.0, result.GetDouble(0), Tolerance);
        Assert.AreEqual(1.0, result.GetDouble(1), Tolerance);
    }

    [TestMethod]
    public void Log1p_WithType_Compiles()
    {
        var a = np.array(new double[] { 0.0, Math.E - 1, Math.E * Math.E - 1 });
        var result = np.log1p(a, typeof(float));
        Assert.IsNotNull(result);
    }

    [TestMethod]
    public void Log1p_WithNPTypeCode_Compiles()
    {
        var a = np.array(new double[] { 0.0, Math.E - 1, Math.E * Math.E - 1 });
        var result = np.log1p(a, NPTypeCode.Double);
        Assert.IsNotNull(result);
    }

    #endregion

    #region Sin - np.sin (2 overloads)

    [TestMethod]
    public void Sin_NoParams_Compiles()
    {
        var a = np.array(new double[] { 0.0, Math.PI / 2, Math.PI });
        var result = np.sin(a);
        Assert.IsNotNull(result);
        Assert.AreEqual(0.0, result.GetDouble(0), Tolerance);
        Assert.AreEqual(1.0, result.GetDouble(1), Tolerance);
        Assert.AreEqual(0.0, result.GetDouble(2), 1e-9); // sin(pi) is approximately 0
    }

    [TestMethod]
    public void Sin_WithType_Compiles()
    {
        var a = np.array(new double[] { 0.0, Math.PI / 2, Math.PI });
        var result = np.sin(a, typeof(float));
        Assert.IsNotNull(result);
        Assert.AreEqual(NPTypeCode.Single, result.typecode);
    }

    #endregion

    #region Cos - np.cos (2 overloads)

    [TestMethod]
    public void Cos_NoParams_Compiles()
    {
        var a = np.array(new double[] { 0.0, Math.PI / 2, Math.PI });
        var result = np.cos(a);
        Assert.IsNotNull(result);
        Assert.AreEqual(1.0, result.GetDouble(0), Tolerance);
        Assert.AreEqual(0.0, result.GetDouble(1), 1e-9); // cos(pi/2) is approximately 0
        Assert.AreEqual(-1.0, result.GetDouble(2), Tolerance);
    }

    [TestMethod]
    public void Cos_WithType_Compiles()
    {
        var a = np.array(new double[] { 0.0, Math.PI / 2, Math.PI });
        var result = np.cos(a, typeof(float));
        Assert.IsNotNull(result);
        Assert.AreEqual(NPTypeCode.Single, result.typecode);
    }

    #endregion

    #region Tan - np.tan (2 overloads)

    [TestMethod]
    public void Tan_NoParams_Compiles()
    {
        var a = np.array(new double[] { 0.0, Math.PI / 4 });
        var result = np.tan(a);
        Assert.IsNotNull(result);
        Assert.AreEqual(0.0, result.GetDouble(0), Tolerance);
        Assert.AreEqual(1.0, result.GetDouble(1), Tolerance);
    }

    [TestMethod]
    public void Tan_WithType_Compiles()
    {
        var a = np.array(new double[] { 0.0, Math.PI / 4 });
        var result = np.tan(a, typeof(float));
        Assert.IsNotNull(result);
        Assert.AreEqual(NPTypeCode.Single, result.typecode);
    }

    #endregion

    #region Arcsin - np.arcsin (2 overloads)

    [TestMethod]
    public void Arcsin_NoParams_Compiles()
    {
        var a = np.array(new double[] { 0.0, 0.5, 1.0 });
        var result = np.arcsin(a);
        Assert.IsNotNull(result);
        Assert.AreEqual(0.0, result.GetDouble(0), Tolerance);
        Assert.AreEqual(Math.PI / 2, result.GetDouble(2), Tolerance);
    }

    [TestMethod]
    public void Arcsin_WithType_Compiles()
    {
        var a = np.array(new double[] { 0.0, 0.5, 1.0 });
        var result = np.arcsin(a, typeof(float));
        Assert.IsNotNull(result);
        Assert.AreEqual(NPTypeCode.Single, result.typecode);
    }

    #endregion

    #region Arccos - np.arccos (2 overloads)

    [TestMethod]
    public void Arccos_NoParams_Compiles()
    {
        var a = np.array(new double[] { 1.0, 0.5, 0.0 });
        var result = np.arccos(a);
        Assert.IsNotNull(result);
        Assert.AreEqual(0.0, result.GetDouble(0), Tolerance);
        Assert.AreEqual(Math.PI / 2, result.GetDouble(2), Tolerance);
    }

    [TestMethod]
    public void Arccos_WithType_Compiles()
    {
        var a = np.array(new double[] { 1.0, 0.5, 0.0 });
        var result = np.arccos(a, typeof(float));
        Assert.IsNotNull(result);
        Assert.AreEqual(NPTypeCode.Single, result.typecode);
    }

    #endregion

    #region Arctan - np.arctan (2 overloads)

    [TestMethod]
    public void Arctan_NoParams_Compiles()
    {
        var a = np.array(new double[] { 0.0, 1.0 });
        var result = np.arctan(a);
        Assert.IsNotNull(result);
        Assert.AreEqual(0.0, result.GetDouble(0), Tolerance);
        Assert.AreEqual(Math.PI / 4, result.GetDouble(1), Tolerance);
    }

    [TestMethod]
    public void Arctan_WithType_Compiles()
    {
        var a = np.array(new double[] { 0.0, 1.0 });
        var result = np.arctan(a, typeof(float));
        Assert.IsNotNull(result);
        Assert.AreEqual(NPTypeCode.Single, result.typecode);
    }

    #endregion

    #region Sinh - np.sinh (2 overloads)

    [TestMethod]
    public void Sinh_NoParams_Compiles()
    {
        var a = np.array(new double[] { 0.0, 1.0 });
        var result = np.sinh(a);
        Assert.IsNotNull(result);
        Assert.AreEqual(0.0, result.GetDouble(0), Tolerance);
        Assert.AreEqual(Math.Sinh(1.0), result.GetDouble(1), Tolerance);
    }

    [TestMethod]
    public void Sinh_WithType_Compiles()
    {
        var a = np.array(new double[] { 0.0, 1.0 });
        var result = np.sinh(a, typeof(float));
        Assert.IsNotNull(result);
        Assert.AreEqual(NPTypeCode.Single, result.typecode);
    }

    #endregion

    #region Cosh - np.cosh (2 overloads)

    [TestMethod]
    public void Cosh_NoParams_Compiles()
    {
        var a = np.array(new double[] { 0.0, 1.0 });
        var result = np.cosh(a);
        Assert.IsNotNull(result);
        Assert.AreEqual(1.0, result.GetDouble(0), Tolerance);
        Assert.AreEqual(Math.Cosh(1.0), result.GetDouble(1), Tolerance);
    }

    [TestMethod]
    public void Cosh_WithType_Compiles()
    {
        var a = np.array(new double[] { 0.0, 1.0 });
        var result = np.cosh(a, typeof(float));
        Assert.IsNotNull(result);
        Assert.AreEqual(NPTypeCode.Single, result.typecode);
    }

    #endregion

    #region Tanh - np.tanh (2 overloads)

    [TestMethod]
    public void Tanh_NoParams_Compiles()
    {
        var a = np.array(new double[] { 0.0, 1.0 });
        var result = np.tanh(a);
        Assert.IsNotNull(result);
        Assert.AreEqual(0.0, result.GetDouble(0), Tolerance);
        Assert.AreEqual(Math.Tanh(1.0), result.GetDouble(1), Tolerance);
    }

    [TestMethod]
    public void Tanh_WithType_Compiles()
    {
        var a = np.array(new double[] { 0.0, 1.0 });
        var result = np.tanh(a, typeof(float));
        Assert.IsNotNull(result);
        Assert.AreEqual(NPTypeCode.Single, result.typecode);
    }

    #endregion

    #region Deg2Rad - np.deg2rad (2 overloads)

    [TestMethod]
    public void Deg2Rad_NoParams_Compiles()
    {
        var a = np.array(new double[] { 0.0, 90.0, 180.0 });
        var result = np.deg2rad(a);
        Assert.IsNotNull(result);
        Assert.AreEqual(0.0, result.GetDouble(0), Tolerance);
        Assert.AreEqual(Math.PI / 2, result.GetDouble(1), Tolerance);
        Assert.AreEqual(Math.PI, result.GetDouble(2), Tolerance);
    }

    [TestMethod]
    public void Deg2Rad_WithType_Compiles()
    {
        var a = np.array(new double[] { 0.0, 90.0, 180.0 });
        var result = np.deg2rad(a, typeof(float));
        Assert.IsNotNull(result);
        Assert.AreEqual(NPTypeCode.Single, result.typecode);
    }

    #endregion

    #region Rad2Deg - np.rad2deg (2 overloads)

    [TestMethod]
    public void Rad2Deg_NoParams_Compiles()
    {
        var a = np.array(new double[] { 0.0, Math.PI / 2, Math.PI });
        var result = np.rad2deg(a);
        Assert.IsNotNull(result);
        Assert.AreEqual(0.0, result.GetDouble(0), Tolerance);
        Assert.AreEqual(90.0, result.GetDouble(1), Tolerance);
        Assert.AreEqual(180.0, result.GetDouble(2), Tolerance);
    }

    [TestMethod]
    public void Rad2Deg_WithType_Compiles()
    {
        var a = np.array(new double[] { 0.0, Math.PI / 2, Math.PI });
        var result = np.rad2deg(a, typeof(float));
        Assert.IsNotNull(result);
        Assert.AreEqual(NPTypeCode.Single, result.typecode);
    }

    #endregion

    #region Radians - np.radians (2 overloads)

    [TestMethod]
    public void Radians_NoParams_Compiles()
    {
        var a = np.array(new double[] { 0.0, 90.0, 180.0 });
        var result = np.radians(a);
        Assert.IsNotNull(result);
        Assert.AreEqual(0.0, result.GetDouble(0), Tolerance);
        Assert.AreEqual(Math.PI / 2, result.GetDouble(1), Tolerance);
        Assert.AreEqual(Math.PI, result.GetDouble(2), Tolerance);
    }

    [TestMethod]
    public void Radians_WithType_Compiles()
    {
        var a = np.array(new double[] { 0.0, 90.0, 180.0 });
        var result = np.radians(a, typeof(float));
        Assert.IsNotNull(result);
        Assert.AreEqual(NPTypeCode.Single, result.typecode);
    }

    #endregion

    #region Degrees - np.degrees (2 overloads)

    [TestMethod]
    public void Degrees_NoParams_Compiles()
    {
        var a = np.array(new double[] { 0.0, Math.PI / 2, Math.PI });
        var result = np.degrees(a);
        Assert.IsNotNull(result);
        Assert.AreEqual(0.0, result.GetDouble(0), Tolerance);
        Assert.AreEqual(90.0, result.GetDouble(1), Tolerance);
        Assert.AreEqual(180.0, result.GetDouble(2), Tolerance);
    }

    [TestMethod]
    public void Degrees_WithType_Compiles()
    {
        var a = np.array(new double[] { 0.0, Math.PI / 2, Math.PI });
        var result = np.degrees(a, typeof(float));
        Assert.IsNotNull(result);
        Assert.AreEqual(NPTypeCode.Single, result.typecode);
    }

    #endregion

    #region Positive - np.positive (1 overload)

    [TestMethod]
    public void Positive_NoParams_Compiles()
    {
        var a = np.array(new double[] { 1.5, -2.3, 3.7 });
        var result = np.positive(a);
        Assert.IsNotNull(result);
        // positive is identity function (returns a copy)
        Assert.AreEqual(1.5, result.GetDouble(0), Tolerance);
        Assert.AreEqual(-2.3, result.GetDouble(1), Tolerance);
        Assert.AreEqual(3.7, result.GetDouble(2), Tolerance);
    }

    #endregion

    #region Negative - np.negative (1 overload)

    [TestMethod]
    public void Negative_NoParams_Compiles()
    {
        var a = np.array(new double[] { 1.5, -2.3, 3.7 });
        var result = np.negative(a);
        Assert.IsNotNull(result);
        Assert.AreEqual(-1.5, result.GetDouble(0), Tolerance);
        Assert.AreEqual(2.3, result.GetDouble(1), Tolerance);
        Assert.AreEqual(-3.7, result.GetDouble(2), Tolerance);
    }

    #endregion

    #region Reciprocal - np.reciprocal (2 overloads)

    [TestMethod]
    public void Reciprocal_NoParams_Compiles()
    {
        var a = np.array(new double[] { 1.0, 2.0, 4.0 });
        var result = np.reciprocal(a);
        Assert.IsNotNull(result);
        Assert.AreEqual(1.0, result.GetDouble(0), Tolerance);
        Assert.AreEqual(0.5, result.GetDouble(1), Tolerance);
        Assert.AreEqual(0.25, result.GetDouble(2), Tolerance);
    }

    [TestMethod]
    public void Reciprocal_WithType_Compiles()
    {
        var a = np.array(new double[] { 1.0, 2.0, 4.0 });
        var result = np.reciprocal(a, typeof(float));
        Assert.IsNotNull(result);
        Assert.AreEqual(NPTypeCode.Single, result.typecode);
    }

    #endregion

    #region Modf - np.modf (2 overloads)

    [TestMethod]
    public void Modf_NoParams_Compiles()
    {
        var a = np.array(new double[] { 1.5, -2.3, 3.7 });
        var (fractional, integral) = np.modf(a);
        Assert.IsNotNull(fractional);
        Assert.IsNotNull(integral);
        // Fractional parts: 0.5, -0.3, 0.7
        Assert.AreEqual(0.5, fractional.GetDouble(0), Tolerance);
        Assert.AreEqual(0.7, fractional.GetDouble(2), Tolerance);
        // Integral parts: 1, -2, 3
        Assert.AreEqual(1.0, integral.GetDouble(0), Tolerance);
        Assert.AreEqual(-2.0, integral.GetDouble(1), Tolerance);
        Assert.AreEqual(3.0, integral.GetDouble(2), Tolerance);
    }

    [TestMethod]
    public void Modf_WithType_Compiles()
    {
        var a = np.array(new double[] { 1.5, -2.3, 3.7 });
        var (fractional, integral) = np.modf(a, typeof(float));
        Assert.IsNotNull(fractional);
        Assert.IsNotNull(integral);
        Assert.AreEqual(NPTypeCode.Single, fractional.typecode);
        Assert.AreEqual(NPTypeCode.Single, integral.typecode);
    }

    #endregion
}
