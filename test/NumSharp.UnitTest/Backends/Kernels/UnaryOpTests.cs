using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.UnitTest.Utilities;
using TUnit.Core;

namespace NumSharp.UnitTest.Backends.Kernels;

/// <summary>
/// Comprehensive tests for unary operations.
/// All expected values are verified against NumPy 2.x output.
/// </summary>
public class UnaryOpTests
{
    private const double Tolerance = 1e-10;
    private const double Pi = Math.PI;

    #region Trigonometric Functions

    [Test]
    public void Sin_Float64()
    {
        // NumPy: sin([0, π/6, π/4, π/2, π])
        var input = np.array(new double[] { 0, Pi / 6, Pi / 4, Pi / 2, Pi });
        var result = np.sin(input);

        Assert.IsTrue(Math.Abs(result.GetDouble(0) - 0.0) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(1) - 0.5) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(2) - 0.7071067811865476) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(3) - 1.0) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(4)) < Tolerance); // sin(π) ≈ 0
    }

    [Test]
    public void Cos_Float64()
    {
        // NumPy: cos([0, π/6, π/4, π/2, π])
        var input = np.array(new double[] { 0, Pi / 6, Pi / 4, Pi / 2, Pi });
        var result = np.cos(input);

        Assert.IsTrue(Math.Abs(result.GetDouble(0) - 1.0) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(1) - 0.8660254037844387) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(2) - 0.7071067811865476) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(3)) < Tolerance); // cos(π/2) ≈ 0
        Assert.IsTrue(Math.Abs(result.GetDouble(4) - (-1.0)) < Tolerance);
    }

    [Test]
    public void Tan_Float64()
    {
        // NumPy: tan([0, π/4, -π/4])
        var input = np.array(new double[] { 0, Pi / 4, -Pi / 4 });
        var result = np.tan(input);

        Assert.IsTrue(Math.Abs(result.GetDouble(0) - 0.0) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(1) - 1.0) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(2) - (-1.0)) < Tolerance);
    }

    #endregion

    #region Inverse Trigonometric Functions

    [Test]
    public void ASin_Float64()
    {
        // NumPy: arcsin([-1, -0.5, 0, 0.5, 1])
        var input = np.array(new double[] { -1, -0.5, 0, 0.5, 1 });
        var result = np.arcsin(input);

        Assert.IsTrue(Math.Abs(result.GetDouble(0) - (-Pi / 2)) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(1) - (-0.5235987755982989)) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(2) - 0.0) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(3) - 0.5235987755982989) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(4) - (Pi / 2)) < Tolerance);
    }

    [Test]
    public void ACos_Float64()
    {
        // NumPy: arccos([-1, -0.5, 0, 0.5, 1])
        var input = np.array(new double[] { -1, -0.5, 0, 0.5, 1 });
        var result = np.arccos(input);

        Assert.IsTrue(Math.Abs(result.GetDouble(0) - Pi) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(1) - 2.0943951023931957) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(2) - (Pi / 2)) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(3) - 1.0471975511965979) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(4) - 0.0) < Tolerance);
    }

    [Test]
    public void ATan_Float64()
    {
        // NumPy: arctan([-1, -0.5, 0, 0.5, 1])
        var input = np.array(new double[] { -1, -0.5, 0, 0.5, 1 });
        var result = np.arctan(input);

        Assert.IsTrue(Math.Abs(result.GetDouble(0) - (-Pi / 4)) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(1) - (-0.4636476090008061)) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(2) - 0.0) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(3) - 0.4636476090008061) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(4) - (Pi / 4)) < Tolerance);
    }

    #endregion

    #region Hyperbolic Functions

    [Test]
    public void Sinh_Float64()
    {
        // NumPy: sinh([-2, -1, 0, 1, 2])
        var input = np.array(new double[] { -2, -1, 0, 1, 2 });
        var result = np.sinh(input);

        Assert.IsTrue(Math.Abs(result.GetDouble(0) - (-3.6268604078470186)) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(1) - (-1.1752011936438014)) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(2) - 0.0) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(3) - 1.1752011936438014) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(4) - 3.6268604078470186) < Tolerance);
    }

    [Test]
    public void Cosh_Float64()
    {
        // NumPy: cosh([-2, -1, 0, 1, 2])
        var input = np.array(new double[] { -2, -1, 0, 1, 2 });
        var result = np.cosh(input);

        Assert.IsTrue(Math.Abs(result.GetDouble(0) - 3.7621956910836314) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(1) - 1.5430806348152437) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(2) - 1.0) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(3) - 1.5430806348152437) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(4) - 3.7621956910836314) < Tolerance);
    }

    [Test]
    public void Tanh_Float64()
    {
        // NumPy: tanh([-2, -1, 0, 1, 2])
        var input = np.array(new double[] { -2, -1, 0, 1, 2 });
        var result = np.tanh(input);

        Assert.IsTrue(Math.Abs(result.GetDouble(0) - (-0.9640275800758169)) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(1) - (-0.7615941559557649)) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(2) - 0.0) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(3) - 0.7615941559557649) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(4) - 0.9640275800758169) < Tolerance);
    }

    #endregion

    #region Logarithmic Functions

    [Test]
    public void Log_Float64()
    {
        // NumPy: log([0.001, 0.5, 1, e, 10, 100])
        var input = np.array(new double[] { 0.001, 0.5, 1, Math.E, 10, 100 });
        var result = np.log(input);

        Assert.IsTrue(Math.Abs(result.GetDouble(0) - (-6.907755278982137)) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(1) - (-0.6931471805599453)) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(2) - 0.0) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(3) - 1.0) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(4) - 2.302585092994046) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(5) - 4.605170185988092) < Tolerance);
    }

    [Test]
    public void Log2_Float64()
    {
        var input = np.array(new double[] { 0.5, 1, 2, 4, 8, 1024 });
        var result = np.log2(input);

        Assert.IsTrue(Math.Abs(result.GetDouble(0) - (-1.0)) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(1) - 0.0) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(2) - 1.0) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(3) - 2.0) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(4) - 3.0) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(5) - 10.0) < Tolerance);
    }

    [Test]
    public void Log10_Float64()
    {
        var input = np.array(new double[] { 0.001, 0.1, 1, 10, 100 });
        var result = np.log10(input);

        Assert.IsTrue(Math.Abs(result.GetDouble(0) - (-3.0)) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(1) - (-1.0)) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(2) - 0.0) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(3) - 1.0) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(4) - 2.0) < Tolerance);
    }

    [Test]
    public void Log1p_Float64()
    {
        // NumPy: log1p([0, 0.5, 1, e-1, 9, 99])
        var input = np.array(new double[] { 0, 0.5, 1, Math.E - 1, 9, 99 });
        var result = np.log1p(input);

        Assert.IsTrue(Math.Abs(result.GetDouble(0) - 0.0) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(1) - 0.4054651081081644) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(2) - 0.6931471805599453) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(3) - 1.0) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(4) - 2.302585092994046) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(5) - 4.605170185988092) < Tolerance);
    }

    #endregion

    #region Exponential Functions

    [Test]
    public void Exp_Float64()
    {
        // NumPy: exp([0, 0.5, 1, 2])
        var input = np.array(new double[] { 0, 0.5, 1, 2 });
        var result = np.exp(input);

        Assert.IsTrue(Math.Abs(result.GetDouble(0) - 1.0) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(1) - 1.6487212707001282) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(2) - Math.E) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(3) - 7.38905609893065) < Tolerance);
    }

    [Test]
    public void Exp2_Float64()
    {
        // NumPy: exp2([0, 1, 2, 3, 10])
        var input = np.array(new double[] { 0, 1, 2, 3, 10 });
        var result = np.exp2(input);

        Assert.IsTrue(Math.Abs(result.GetDouble(0) - 1.0) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(1) - 2.0) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(2) - 4.0) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(3) - 8.0) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(4) - 1024.0) < Tolerance);
    }

    [Test]
    public void Expm1_Float64()
    {
        // NumPy: expm1([0, 0.5, 1, 2])
        var input = np.array(new double[] { 0, 0.5, 1, 2 });
        var result = np.expm1(input);

        Assert.IsTrue(Math.Abs(result.GetDouble(0) - 0.0) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(1) - 0.6487212707001282) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(2) - (Math.E - 1)) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(3) - 6.38905609893065) < Tolerance);
    }

    #endregion

    #region General Operations

    [Test]
    public void Abs_Float64()
    {
        // NumPy: abs([-5.5, -1, 0, 1, 2.5, 100])
        var input = np.array(new double[] { -5.5, -1, 0, 1, 2.5, 100 });
        var result = np.abs(input);

        result.Should().BeOfValues(5.5, 1.0, 0.0, 1.0, 2.5, 100.0);
    }

    [Test]
    [OpenBugs] // IL kernel bug - returns incorrect values
    public void Abs_Int32()
    {
        var input = np.array(new[] { -5, -1, 0, 1, 5 });
        var result = np.abs(input);

        result.Should().BeOfValues(5, 1, 0, 1, 5).And.BeOfType(NPTypeCode.Int32);
    }

    [Test]
    [OpenBugs] // IL kernel bug - returns incorrect values
    public void Negative_Float64()
    {
        // NumPy: negative([-5.5, -1, 0, 1, 2.5])
        var input = np.array(new double[] { -5.5, -1, 0, 1, 2.5 });
        var result = np.negative(input);

        result.Should().BeOfValues(5.5, 1.0, 0.0, -1.0, -2.5);
    }

    [Test]
    [OpenBugs] // IL kernel bug - returns incorrect values
    public void Negative_Int32()
    {
        var input = np.array(new[] { -5, -1, 0, 1, 5 });
        var result = np.negative(input);

        result.Should().BeOfValues(5, 1, 0, -1, -5).And.BeOfType(NPTypeCode.Int32);
    }

    [Test]
    [OpenBugs] // NumSharp throws NotSupportedException for unsigned negative
    public void Negative_Byte_Overflow()
    {
        // NumPy: negative([1, 2, 3, 4, 5]) for uint8 = [255, 254, 253, 252, 251]
        var input = np.array(new byte[] { 1, 2, 3, 4, 5 });
        var result = np.negative(input);

        result.Should().BeOfValues(255, 254, 253, 252, 251).And.BeOfType(NPTypeCode.Byte);
    }

    [Test]
    public void Sqrt_Float64()
    {
        var input = np.array(new double[] { 0, 1, 4, 9, 100 });
        var result = np.sqrt(input);

        Assert.IsTrue(Math.Abs(result.GetDouble(0) - 0.0) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(1) - 1.0) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(2) - 2.0) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(3) - 3.0) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(4) - 10.0) < Tolerance);
    }

    [Test]
    public void Sign_Float64()
    {
        // NumPy: sign([-5.5, -1, 0, 1, 2.5])
        var input = np.array(new double[] { -5.5, -1, 0, 1, 2.5 });
        var result = np.sign(input);

        result.Should().BeOfValues(-1.0, -1.0, 0.0, 1.0, 1.0);
    }

    [Test]
    [OpenBugs] // IL kernel bug - returns incorrect values
    public void Sign_Int32()
    {
        var input = np.array(new[] { -5, -1, 0, 1, 5 });
        var result = np.sign(input);

        result.Should().BeOfValues(-1, -1, 0, 1, 1).And.BeOfType(NPTypeCode.Int32);
    }

    #endregion

    #region Rounding Operations - Banker's Rounding

    [Test]
    public void Round_BankersRounding()
    {
        // NumPy uses banker's rounding (round half to even)
        // NumPy: round([0.0, 0.49, 0.5, 0.51, 1.5, 2.5, -0.5, -1.5, -2.5])
        //      = [0.0, 0.0, 0.0, 1.0, 2.0, 2.0, -0.0, -2.0, -2.0]
        var input = np.array(new double[] { 0.0, 0.49, 0.5, 0.51, 1.5, 2.5, -0.5, -1.5, -2.5 });
        var result = np.round_(input);

        Assert.AreEqual(0.0, result.GetDouble(0));
        Assert.AreEqual(0.0, result.GetDouble(1));
        Assert.AreEqual(0.0, result.GetDouble(2));  // 0.5 rounds to 0 (even)
        Assert.AreEqual(1.0, result.GetDouble(3));
        Assert.AreEqual(2.0, result.GetDouble(4));  // 1.5 rounds to 2 (even)
        Assert.AreEqual(2.0, result.GetDouble(5));  // 2.5 rounds to 2 (even)
        Assert.AreEqual(0.0, result.GetDouble(6));  // -0.5 rounds to 0 (even)
        Assert.AreEqual(-2.0, result.GetDouble(7)); // -1.5 rounds to -2 (even)
        Assert.AreEqual(-2.0, result.GetDouble(8)); // -2.5 rounds to -2 (even)
    }

    [Test]
    public void Floor_Float64()
    {
        // NumPy: floor([0.0, 0.49, 0.5, 0.51, 1.5, 2.5, -0.5, -1.5, -2.5])
        var input = np.array(new double[] { 0.0, 0.49, 0.5, 0.51, 1.5, 2.5, -0.5, -1.5, -2.5 });
        var result = np.floor(input);

        result.Should().BeOfValues(0.0, 0.0, 0.0, 0.0, 1.0, 2.0, -1.0, -2.0, -3.0);
    }

    [Test]
    public void Ceil_Float64()
    {
        // NumPy: ceil([0.0, 0.49, 0.5, 0.51, 1.5, 2.5, -0.5, -1.5, -2.5])
        var input = np.array(new double[] { 0.0, 0.49, 0.5, 0.51, 1.5, 2.5, -0.5, -1.5, -2.5 });
        var result = np.ceil(input);

        result.Should().BeOfValues(0.0, 1.0, 1.0, 1.0, 2.0, 3.0, 0.0, -1.0, -2.0);
    }

    #endregion

    #region Edge Cases - Special Values

    [Test]
    public void Sin_EdgeCases()
    {
        // NumPy: sin([0, -0, inf, -inf, nan]) = [0, -0, nan, nan, nan]
        var input = np.array(new double[] { 0, -0.0, double.PositiveInfinity, double.NegativeInfinity, double.NaN });
        var result = np.sin(input);

        Assert.AreEqual(0.0, result.GetDouble(0));
        // -0.0 sin is -0.0 but comparing as 0.0
        Assert.IsTrue(double.IsNaN(result.GetDouble(2)));
        Assert.IsTrue(double.IsNaN(result.GetDouble(3)));
        Assert.IsTrue(double.IsNaN(result.GetDouble(4)));
    }

    [Test]
    public void Exp_EdgeCases()
    {
        // NumPy: exp([0, -0, inf, -inf, nan]) = [1, 1, inf, 0, nan]
        var input = np.array(new double[] { 0, -0.0, double.PositiveInfinity, double.NegativeInfinity, double.NaN });
        var result = np.exp(input);

        Assert.AreEqual(1.0, result.GetDouble(0));
        Assert.AreEqual(1.0, result.GetDouble(1));
        Assert.IsTrue(double.IsPositiveInfinity(result.GetDouble(2)));
        Assert.AreEqual(0.0, result.GetDouble(3));
        Assert.IsTrue(double.IsNaN(result.GetDouble(4)));
    }

    [Test]
    public void Log_EdgeCases()
    {
        // NumPy: log([0, -0, inf, -inf, nan]) = [-inf, -inf, inf, nan, nan]
        var input = np.array(new double[] { 0, -0.0, double.PositiveInfinity, double.NegativeInfinity, double.NaN });
        var result = np.log(input);

        Assert.IsTrue(double.IsNegativeInfinity(result.GetDouble(0)));
        Assert.IsTrue(double.IsNegativeInfinity(result.GetDouble(1)));
        Assert.IsTrue(double.IsPositiveInfinity(result.GetDouble(2)));
        Assert.IsTrue(double.IsNaN(result.GetDouble(3)));
        Assert.IsTrue(double.IsNaN(result.GetDouble(4)));
    }

    [Test]
    public void Sqrt_EdgeCases()
    {
        // NumPy: sqrt([0, -0, inf, -inf, nan]) = [0, -0, inf, nan, nan]
        var input = np.array(new double[] { 0, -0.0, double.PositiveInfinity, double.NegativeInfinity, double.NaN });
        var result = np.sqrt(input);

        Assert.AreEqual(0.0, result.GetDouble(0));
        // sqrt(-0) = -0
        Assert.IsTrue(double.IsPositiveInfinity(result.GetDouble(2)));
        Assert.IsTrue(double.IsNaN(result.GetDouble(3)));
        Assert.IsTrue(double.IsNaN(result.GetDouble(4)));
    }

    [Test]
    [OpenBugs] // .NET Math.Sign throws on NaN, NumPy returns NaN
    public void Sign_EdgeCases()
    {
        // NumPy: sign([0, -0, inf, -inf, nan]) = [0, 0, 1, -1, nan]
        var input = np.array(new double[] { 0, -0.0, double.PositiveInfinity, double.NegativeInfinity, double.NaN });
        var result = np.sign(input);

        Assert.AreEqual(0.0, result.GetDouble(0));
        Assert.AreEqual(0.0, result.GetDouble(1));
        Assert.AreEqual(1.0, result.GetDouble(2));
        Assert.AreEqual(-1.0, result.GetDouble(3));
        Assert.IsTrue(double.IsNaN(result.GetDouble(4)));
    }

    #endregion

    #region Float32 Tests

    [Test]
    public void Sin_Float32()
    {
        var input = np.array(new float[] { 0f, 0.5f, 1f, 1.5f, 2f });
        var result = np.sin(input);

        // NumPy float32 values
        Assert.IsTrue(Math.Abs(result.GetSingle(0) - 0f) < 1e-6f);
        Assert.IsTrue(Math.Abs(result.GetSingle(1) - 0.4794255495071411f) < 1e-6f);
        Assert.IsTrue(Math.Abs(result.GetSingle(2) - 0.8414710164070129f) < 1e-6f);
    }

    [Test]
    public void Cos_Float32()
    {
        var input = np.array(new float[] { 0f, 0.5f, 1f, 1.5f, 2f });
        var result = np.cos(input);

        Assert.IsTrue(Math.Abs(result.GetSingle(0) - 1f) < 1e-6f);
        Assert.IsTrue(Math.Abs(result.GetSingle(1) - 0.8775825500488281f) < 1e-6f);
        Assert.IsTrue(Math.Abs(result.GetSingle(2) - 0.5403022766113281f) < 1e-6f);
    }

    [Test]
    public void Exp_Float32()
    {
        var input = np.array(new float[] { 0f, 0.5f, 1f, 1.5f, 2f });
        var result = np.exp(input);

        Assert.IsTrue(Math.Abs(result.GetSingle(0) - 1f) < 1e-5f);
        Assert.IsTrue(Math.Abs(result.GetSingle(1) - 1.6487212181091309f) < 1e-5f);
        Assert.IsTrue(Math.Abs(result.GetSingle(2) - 2.7182819843292236f) < 1e-5f);
    }

    #endregion

    #region Sliced Array Tests

    [Test]
    public void Sin_SlicedArray()
    {
        // Test that unary ops work correctly on sliced/strided arrays
        var original = np.array(new double[] { 0, Pi / 6, Pi / 4, Pi / 3, Pi / 2, Pi });
        var sliced = original["::2"]; // [0, π/4, π/2]
        var result = np.sin(sliced);

        Assert.AreEqual(3, result.size);
        Assert.IsTrue(Math.Abs(result.GetDouble(0) - 0.0) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(1) - 0.7071067811865476) < Tolerance);
        Assert.IsTrue(Math.Abs(result.GetDouble(2) - 1.0) < Tolerance);
    }

    [Test]
    public void Abs_2DArray()
    {
        var input = np.array(new[,] { { -1.0, 2.0 }, { -3.0, 4.0 } });
        var result = np.abs(input);

        result.Should().BeShaped(2, 2);
        result.Should().BeOfValues(1.0, 2.0, 3.0, 4.0);
    }

    #endregion
}
