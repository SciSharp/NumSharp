using System;
using System.Numerics;
using AwesomeAssertions;
using NumSharp;

namespace NumSharp.UnitTest;

/// <summary>
/// Byte-exact parity tests for NumPy 2.4.2's array printing across <c>str</c> (<see cref="NDArray.ToString()"/> /
/// <c>np.array_str</c>), <c>repr</c> (<c>ToString(true)</c> / <c>np.array_repr</c>), <c>np.array2string</c>,
/// <c>np.format_float_positional</c> / <c>np.format_float_scientific</c> and the print options. Every
/// expected string was produced by running the equivalent NumPy 2.4.2 code.
/// </summary>
[TestClass]
public class np_ArrayPrint_ParityTests
{
    private static void Str(NDArray a, string expected) => np.array_str(a).Should().Be(expected);
    private static void Repr(NDArray a, string expected) => np.array_repr(a).Should().Be(expected);

    #region repr: dtype & shape suffixes

    [TestMethod] public void Repr_Int64_NoDtype() => Repr(np.arange(5), "array([0, 1, 2, 3, 4])");
    [TestMethod] public void Repr_Int8_DtypeShown() => Repr(np.array(new sbyte[] { 1, 2, 3 }), "array([1, 2, 3], dtype=int8)");
    [TestMethod] public void Repr_UInt8_DtypeShown() => Repr(np.array(new byte[] { 1, 2, 3 }), "array([1, 2, 3], dtype=uint8)");
    [TestMethod] public void Repr_Int16_DtypeShown() => Repr(np.array(new short[] { 1, 2, 3 }), "array([1, 2, 3], dtype=int16)");
    [TestMethod] public void Repr_Float64_NoDtype() => Repr(np.array(new double[] { 1.0, 2.0, 3.0 }), "array([1., 2., 3.])");
    [TestMethod] public void Repr_Float32_DtypeShown() => Repr(np.array(new float[] { 1.5f, 2.5f, 3.5f }), "array([1.5, 2.5, 3.5], dtype=float32)");
    [TestMethod] public void Repr_Bool_NoDtype() => Repr(np.array(new bool[] { true, false, true }), "array([ True, False,  True])");
    [TestMethod] public void Repr_2D() => Repr(np.arange(6).reshape(2, 3), "array([[0, 1, 2],\n       [3, 4, 5]])");
    [TestMethod] public void Repr_3D() => Repr(np.arange(8).reshape(2, 2, 2), "array([[[0, 1],\n        [2, 3]],\n\n       [[4, 5],\n        [6, 7]]])");
    [TestMethod] public void Repr_NegativeAlign() => Repr(np.arange(-5, 1), "array([-5, -4, -3, -2, -1,  0])");

    #endregion

    #region float formatting (maxprec, decimal-point alignment)

    [TestMethod] public void Float_Whole() => Str(np.array(new double[] { 1.0, 2.0, 3.0 }), "[1. 2. 3.]");
    [TestMethod] public void Float_Frac() => Str(np.array(new double[] { 0.5, 1.5, 2.5 }), "[0.5 1.5 2.5]");
    [TestMethod] public void Float_Align() => Str(np.array(new double[] { 1.0, 22.5, 333.25 }), "[  1.    22.5  333.25]");
    [TestMethod] public void Float_AlignRepr() => Repr(np.array(new double[] { 1.0, 22.5, 333.25 }), "array([  1.  ,  22.5 , 333.25])");
    [TestMethod] public void Float_NegAlign() => Str(np.array(new double[] { -1.5, 2.25, -3.125 }), "[-1.5    2.25  -3.125]");
    [TestMethod] public void Float_Precision8() => Str(np.array(new double[] { 1.123456789, 2.0 }), "[1.12345679 2.        ]");
    [TestMethod] public void Float_IntFrac() => Str(np.array(new double[] { 1.0, 10.0, 100.0, 0.5 }), "[  1.   10.  100.    0.5]");
    [TestMethod] public void Float_Pi() => Str(np.array(new double[] { Math.PI }), "[3.14159265]");

    // The adversarial-tie / carry cases proven divergent for shortest-string rounding.
    [TestMethod] public void Float_Tie_RoundsTrueValue() => Str(np.array(new double[] { 179.065948445 }), "[179.06594845]");
    [TestMethod] public void Float_Tie_Negative() => Str(np.array(new double[] { -626.721831885 }), "[-626.72183189]");
    [TestMethod] public void Float_Carry_Changes_IntWidth() => Str(np.array(new double[] { 9.999999996 }), "[10.]");

    #endregion

    #region exponential format

    [TestMethod] public void Exp_Small() => Str(np.array(new double[] { 1e-16, 1, 2, 3 }), "[1.e-16 1.e+00 2.e+00 3.e+00]");
    [TestMethod] public void Exp_Big() => Str(np.array(new double[] { 1e16, 1, 2, 3 }), "[1.e+16 1.e+00 2.e+00 3.e+00]");
    [TestMethod] public void Exp_MixedMag() => Str(np.array(new double[] { 0.001, 1000.0 }), "[1.e-03 1.e+03]");
    [TestMethod] public void Exp_RatioBoundary_1000_Positional() => Str(np.array(new double[] { 1.0, 1000.0 }), "[   1. 1000.]");
    [TestMethod] public void Exp_RatioBoundary_1001_Exp() => Str(np.array(new double[] { 1.0, 1001.0 }), "[1.000e+00 1.001e+03]");

    #endregion

    #region nan / inf

    [TestMethod] public void NanInf_NanMix() => Str(np.array(new double[] { 1.0, double.NaN, 3.0 }), "[ 1. nan  3.]");
    [TestMethod] public void NanInf_Inf() => Str(np.array(new double[] { double.PositiveInfinity, double.NegativeInfinity, 1.0 }), "[ inf -inf   1.]");
    [TestMethod] public void NanInf_Mixed() => Str(np.array(new double[] { double.NaN, double.PositiveInfinity, 1.0, 2.0 }), "[nan inf  1.  2.]");

    #endregion

    #region complex

    [TestMethod] public void Complex_Ints() => Str(np.array(new Complex[] { new(1, 2), new(3, -4) }), "[1.+2.j 3.-4.j]");
    [TestMethod] public void Complex_IntsRepr() => Repr(np.array(new Complex[] { new(1, 2), new(3, -4) }), "array([1.+2.j, 3.-4.j])");
    [TestMethod] public void Complex_Real() => Str(np.array(new Complex[] { new(1, 0), new(2, 0) }), "[1.+0.j 2.+0.j]");
    [TestMethod] public void Complex_Frac() => Str(np.array(new Complex[] { new(1.5, 2.5), new(0.25, -1.75) }), "[1.5 +2.5j  0.25-1.75j]");
    [TestMethod] public void Complex_Mixed() => Str(np.array(new Complex[] { new(1, 2), new(3.5, -4.25), new(0, 0) }), "[1. +2.j   3.5-4.25j 0. +0.j  ]");

    #endregion

    #region float32 / float16

    [TestMethod] public void Float32_Basic() => Str(np.array(new float[] { 1.0f, 2.0f, 3.0f }), "[1. 2. 3.]");
    [TestMethod] public void Float32_Pi() => Str(np.array(new float[] { (float)Math.PI }), "[3.1415927]");
    [TestMethod] public void Float16_Basic() => Str(np.array(new Half[] { (Half)1.0, (Half)2.5, (Half)3.14 }), "[1.   2.5  3.14]");
    [TestMethod] public void Float16_Repr() => Repr(np.array(new Half[] { (Half)1.0, (Half)2.5, (Half)3.14 }), "array([1.  , 2.5 , 3.14], dtype=float16)");

    #endregion

    #region summarization (threshold) & line wrapping

    [TestMethod] public void Summarize_1D_Str() => Str(np.arange(2000), "[   0    1    2 ... 1997 1998 1999]");
    [TestMethod] public void Summarize_1D_Repr() => Repr(np.arange(2000), "array([   0,    1,    2, ..., 1997, 1998, 1999], shape=(2000,))");

    [TestMethod]
    public void Summarize_2D_Str() => Str(np.arange(1600).reshape(40, 40),
        "[[   0    1    2 ...   37   38   39]\n [  40   41   42 ...   77   78   79]\n [  80   81   82 ...  117  118  119]\n ...\n [1480 1481 1482 ... 1517 1518 1519]\n [1520 1521 1522 ... 1557 1558 1559]\n [1560 1561 1562 ... 1597 1598 1599]]");

    [TestMethod]
    public void Wrap_1D_Str() => Str(np.arange(30),
        "[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23\n 24 25 26 27 28 29]");

    [TestMethod]
    public void Wrap_1D_Repr() => Repr(np.arange(30),
        "array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29])");

    [TestMethod]
    public void Wrap_CustomLineWidth() => np.array2string(np.arange(20), max_line_width: 20)
        .Should().Be("[ 0  1  2  3  4  5\n  6  7  8  9 10 11\n 12 13 14 15 16 17\n 18 19]");

    #endregion

    #region 0-d scalars (str uses scalar repr; repr uses the array formatter)

    [TestMethod] public void Scalar_Int_Str() => Str(np.array(42L), "42");
    [TestMethod] public void Scalar_Int_Repr() => Repr(np.array(42L), "array(42)");
    [TestMethod] public void Scalar_FloatWhole_Str() => Str(np.array(5.0), "5.0");
    [TestMethod] public void Scalar_FloatWhole_Repr() => Repr(np.array(5.0), "array(5.)");
    [TestMethod] public void Scalar_FloatFrac_Str() => Str(np.array(3.14), "3.14");
    [TestMethod] public void Scalar_FloatThird_Str() => Str(np.array(1.0 / 3.0), "0.3333333333333333");
    [TestMethod] public void Scalar_FloatThird_Repr() => Repr(np.array(1.0 / 3.0), "array(0.33333333)");
    [TestMethod] public void Scalar_Big_Str() => Str(np.array(1e20), "1e+20");
    [TestMethod] public void Scalar_Bool_Str() => Str(np.array(true), "True");
    [TestMethod] public void Scalar_Complex_Str() => Str(np.array(new Complex[] { new(1, 2) }).reshape(new int[0]), "(1+2j)");
    [TestMethod] public void Scalar_Complex_Repr() => Repr(np.array(new Complex[] { new(1, 2) }).reshape(new int[0]), "array(1.+2.j)");

    #endregion

    #region empty arrays

    [TestMethod] public void Empty_1D_Str() => Str(np.array(new long[0]), "[]");
    [TestMethod] public void Empty_1D_Repr() => Repr(np.array(new long[0]), "array([], dtype=int64)");
    [TestMethod] public void Empty_2D_Repr() => Repr(np.zeros(new Shape(0, 3)), "array([], shape=(0, 3), dtype=float64)");
    [TestMethod] public void Empty_2x0_Repr() => Repr(np.zeros(new Shape(2, 0)), "array([], shape=(2, 0), dtype=float64)");

    #endregion

    #region np.array2string options

    [TestMethod] public void A2S_SuppressSmall() => np.array2string(np.array(new double[] { 1e-10, 1.0, 2.0 }), suppress_small: true).Should().Be("[0. 1. 2.]");
    [TestMethod] public void A2S_SignPlus() => np.array2string(np.array(new double[] { 1.0, -2.0, 3.0 }), sign: '+').Should().Be("[+1. -2. +3.]");
    [TestMethod] public void A2S_SignSpace() => np.array2string(np.array(new double[] { 1.0, -2.0, 3.0 }), sign: ' ').Should().Be("[ 1. -2.  3.]");
    [TestMethod] public void A2S_Precision2() => np.array2string(np.array(new double[] { 1.0 / 3.0, 2.0 / 3.0 }), precision: 2).Should().Be("[0.33 0.67]");
    [TestMethod] public void A2S_Threshold() => np.array2string(np.arange(10), threshold: 5).Should().Be("[0 1 2 ... 7 8 9]");
    [TestMethod] public void A2S_EdgeItems() => np.array2string(np.arange(20), threshold: 5, edgeitems: 2).Should().Be("[ 0  1 ... 18 19]");
    [TestMethod] public void A2S_Separator() => np.array2string(np.arange(3), separator: ", ").Should().Be("[0, 1, 2]");
    [TestMethod] public void A2S_FloatmodeMaxprecEqual() => np.array2string(np.array(new double[] { 1.0, 22.5, 333.25 }), floatmode: "maxprec_equal").Should().Be("[  1.00  22.50 333.25]");
    [TestMethod] public void A2S_FloatmodeFixed() => np.array2string(np.array(new double[] { 1.0, 2.5 }), floatmode: "fixed", precision: 3).Should().Be("[1.000 2.500]");

    #endregion

    #region format_float_positional / scientific

    [TestMethod] public void FFP_Pi_Unique() => np.format_float_positional(Math.PI).Should().Be("3.141592653589793");
    [TestMethod] public void FFP_Third_Prec8() => np.format_float_positional(1.0 / 3.0, precision: 8, unique: true, trim: '.', fractional: true).Should().Be("0.33333333");
    [TestMethod] public void FFP_TrimModes()
    {
        np.format_float_positional(2.0, precision: 8, unique: true, trim: 'k').Should().Be("2.");
        np.format_float_positional(2.0, precision: 8, unique: true, trim: '0').Should().Be("2.0");
        np.format_float_positional(2.0, precision: 8, unique: true, trim: '-').Should().Be("2");
    }
    [TestMethod] public void FFS_Big() => np.format_float_scientific(123456.789, precision: 8, unique: true, trim: '.').Should().Be("1.23456789e+05");
    [TestMethod] public void FFS_Small() => np.format_float_scientific(1e-16, precision: 8, unique: true, trim: '.').Should().Be("1.e-16");

    #endregion

    #region print options

    [TestMethod]
    public void GetPrintOptions_Defaults()
    {
        var o = np.get_printoptions();
        o["precision"].Should().Be(8);
        o["threshold"].Should().Be(1000);
        o["edgeitems"].Should().Be(3);
        o["linewidth"].Should().Be(75);
        o["suppress"].Should().Be(false);
        o["nanstr"].Should().Be("nan");
        o["infstr"].Should().Be("inf");
        o["floatmode"].Should().Be("maxprec");
    }

    [TestMethod]
    public void PrintOptions_Context_RestoresPrecision()
    {
        var before = np.get_printoptions()["precision"];
        using (np.printoptions(precision: 2))
        {
            np.array_str(np.array(new double[] { 1.0 / 3.0 })).Should().Be("[0.33]");
        }
        np.get_printoptions()["precision"].Should().Be(before);
    }

    [TestMethod]
    public void SetPrintOptions_NanInfStr()
    {
        try
        {
            np.set_printoptions(nanstr: "NaN", infstr: "Inf");
            np.array_str(np.array(new double[] { double.NaN, double.PositiveInfinity, 1.0 })).Should().Be("[NaN Inf  1.]");
        }
        finally
        {
            np.set_printoptions(nanstr: "nan", infstr: "inf");
        }
    }

    #endregion

    #region edge cases: subnormals, extremes, signed zero, empties, 0-d specials, complex specials

    private static NDArray Scalar0d(NDArray a) => a.reshape(new int[0]);

    [TestMethod] public void Edge_Subnormal_Str() => Str(np.array(new double[] { 5e-324 }), "[5.e-324]");
    [TestMethod] public void Edge_Subnormal_Repr() => Repr(np.array(new double[] { 5e-324 }), "array([5.e-324])");
    [TestMethod] public void Edge_ThreeDigitExp() => Str(np.array(new double[] { 1e-100, 1e100 }), "[1.e-100 1.e+100]");
    [TestMethod] public void Edge_Float16_Subnormal() => Str(np.array(new Half[] { (Half)6e-8 }), "[6.e-08]");

    [TestMethod] public void Edge_AllNan() => Str(np.array(new double[] { double.NaN, double.NaN }), "[nan nan]");
    [TestMethod] public void Edge_AllInf() => Str(np.array(new double[] { double.PositiveInfinity, double.PositiveInfinity }), "[inf inf]");
    [TestMethod] public void Edge_AllNegInf() => Str(np.array(new double[] { double.NegativeInfinity, double.NegativeInfinity }), "[-inf -inf]");
    [TestMethod] public void Edge_NanInExpField() => Str(np.array(new double[] { double.NaN, 1e-16, 1e16 }), "[   nan 1.e-16 1.e+16]");

    [TestMethod] public void Edge_NegativeZero() => Str(np.array(new double[] { -0.0, 0.0 }), "[-0.  0.]");

    [TestMethod]
    public void Edge_UInt64_Big() => Repr(np.array(new ulong[] { ulong.MaxValue, 0, 9223372036854775808UL }),
        "array([18446744073709551615,                    0,  9223372036854775808],\n      dtype=uint64)");

    [TestMethod] public void Edge_0d_Float32_Str() => Str(Scalar0d(np.array(new float[] { 3.14f })), "3.14");
    [TestMethod] public void Edge_0d_Float32_Repr() => Repr(Scalar0d(np.array(new float[] { 3.14f })), "array(3.14, dtype=float32)");
    [TestMethod] public void Edge_0d_Float16_Str() => Str(Scalar0d(np.array(new Half[] { (Half)3.14 })), "3.14");
    [TestMethod] public void Edge_0d_Float32Big_Str() => Str(Scalar0d(np.array(new float[] { 1e20f })), "1e+20");
    [TestMethod] public void Edge_0d_Int64Min_Str() => Str(Scalar0d(np.array(new long[] { long.MinValue })), "-9223372036854775808");
    [TestMethod] public void Edge_0d_UInt64Max_Repr() => Repr(Scalar0d(np.array(new ulong[] { ulong.MaxValue })), "array(18446744073709551615, dtype=uint64)");
    [TestMethod] public void Edge_0d_NegativeZero_Str() => Str(np.array(-0.0), "-0.0");
    [TestMethod] public void Edge_0d_Tiny_Str() => Str(np.array(1e-20), "1e-20");
    [TestMethod] public void Edge_0d_Complex_NanInf_Str() => Str(Scalar0d(np.array(new Complex[] { new(double.NaN, double.PositiveInfinity) })), "(nan+infj)");
    [TestMethod] public void Edge_0d_Complex_NanInf_Repr() => Repr(Scalar0d(np.array(new Complex[] { new(double.NaN, double.PositiveInfinity) })), "array(nan+infj)");

    [TestMethod] public void Edge_Empty_0x3_Repr() => Repr(np.zeros(new Shape(0, 3)), "array([], shape=(0, 3), dtype=float64)");
    [TestMethod] public void Edge_Empty_3x0_Repr() => Repr(np.zeros(new Shape(3, 0)), "array([], shape=(3, 0), dtype=float64)");
    [TestMethod] public void Edge_Empty_2x0x4_Repr() => Repr(np.array(new int[0]).reshape(2, 0, 4), "array([], shape=(2, 0, 4), dtype=int32)");
    [TestMethod] public void Edge_Empty_0x0_Repr() => Repr(np.zeros(new Shape(0, 0)), "array([], shape=(0, 0), dtype=float64)");
    [TestMethod] public void Edge_Empty_Complex_Repr() => Repr(np.array(new Complex[0]), "array([], dtype=complex128)");
    [TestMethod] public void Edge_Empty_Bool_Repr() => Repr(np.array(new bool[0]), "array([], dtype=bool)");

    [TestMethod] public void Edge_Complex_NanInf() => Str(np.array(new Complex[] { new(double.NaN, 1), new(1, double.NegativeInfinity) }), "[nan +1.j  1.-infj]");
    [TestMethod] public void Edge_Complex_SignedZero() => Str(np.array(new Complex[] { new(0, 0), new(-0.0, -0.0) }), "[ 0.+0.j -0.-0.j]");
    [TestMethod] public void Edge_Complex_Exp() => Str(np.array(new Complex[] { new(1e-16, 1e16) }), "[1.e-16+1.e+16j]");

    // summarization: hidden middle values must NOT widen columns (format computed over edge items only)
    [TestMethod]
    public void Edge_Summary_HiddenValuesIgnored() =>
        np.array2string(np.array(new long[] { 1, 1, 999999, 999999, 999999, 2, 2 }), threshold: 5, edgeitems: 2)
        .Should().Be("[1 1 ... 2 2]");

    [TestMethod]
    public void Edge_FloatmodeEqual_2D() =>
        np.array2string(np.array(new double[] { 1.0, 22.5, 333.25, 0.5, 1.125, 9.0 }).reshape(2, 3), floatmode: "maxprec_equal", precision: 3)
        .Should().Be("[[  1.000  22.500 333.250]\n [  0.500   1.125   9.000]]");

    [TestMethod] public void Edge_Precision0() => np.array2string(np.array(new double[] { 1234.5, 6789.1 }), precision: 0).Should().Be("[1234. 6789.]");

    [TestMethod]
    public void Edge_TinyLineWidth() =>
        np.array2string(np.arange(15), max_line_width: 8).Should().Be("[ 0  1\n  2  3\n  4  5\n  6  7\n  8  9\n 10 11\n 12 13\n 14]");

    #endregion
}
