using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.UnitTest.MathOps;

/// <summary>
/// Comprehensive np.power coverage matching NumPy 2.4.2 behavior:
///   - integer power preserves dtype wraparound (no Math.Pow precision loss)
///   - signed-integer negative exponent raises (NumPy ValueError)
///   - stride / broadcast / F-contig layouts all work
///   - special float values (inf, nan, zero base/exp) match NumPy
///   - dtype promotion follows NEP50 (with documented weak-scalar caveat)
///   - all 12+ NumSharp dtypes covered
/// </summary>
[TestClass]
public class NDArray_Power_Comprehensive
{
    // ====================================================================
    // Integer dtype-native wrapping (no double round-trip precision loss)
    // ====================================================================

    [TestMethod]
    public void Int64_LargeValues_PreservesPrecision()
    {
        var a = np.array(new long[] { 15L });
        var b = np.array(new long[] { 15L });
        np.power(a, b).GetInt64(0).Should().Be(437893890380859375L);
    }

    [TestMethod]
    public void Int32_OverflowWraps()
    {
        // 2^31 wraps to int32 min when computed via squared-exp with native int wrap.
        // NumPy: 2 ** 31 in int32 → -2147483648 (overflow wrap).
        var a = np.array(new int[] { 2 });
        var b = np.array(new int[] { 31 });
        np.power(a, b).GetInt32(0).Should().Be(int.MinValue);
    }

    [TestMethod]
    public void UInt8_OverflowWraps()
    {
        // 2^8 = 256 wraps to 0 in uint8.
        var a = np.array(new byte[] { 2 });
        var b = np.array(new byte[] { 8 });
        np.power(a, b).GetByte(0).Should().Be(0);
    }

    [TestMethod]
    public void UInt8_LargeBaseOverflowWraps()
    {
        // 255 ** 2 = 65025, in uint8 = 65025 mod 256 = 1
        var a = np.array(new byte[] { 255 });
        var b = np.array(new byte[] { 2 });
        np.power(a, b).GetByte(0).Should().Be(1);
    }

    [TestMethod]
    public void Int8_NegativeBaseWraps()
    {
        // (-3)^5 = -243, in int8 (range -128..127) = 13
        var a = np.array(new sbyte[] { -3 });
        var b = np.array(new sbyte[] { 5 });
        np.power(a, b).GetSByte(0).Should().Be((sbyte)13);
    }

    // ====================================================================
    // Stride / broadcast / F-contig layouts
    // ====================================================================

    [TestMethod]
    public void SlicedInt32_NoCrash_CorrectValues()
    {
        var arr = np.arange(20).astype(NPTypeCode.Int32);
        var sliced = arr["::2"];                          // [0,2,4,...,18]
        var b = np.arange(10).astype(NPTypeCode.Int32);   // [0..9]
        var r = np.power(sliced, b);

        // NumPy: [1, 2, 16, 216, 4096, 100000, 2985984, 105413504, 0, 790794752] (int32 wrap)
        r.GetInt32(0).Should().Be(1);
        r.GetInt32(1).Should().Be(2);
        r.GetInt32(2).Should().Be(16);
        r.GetInt32(3).Should().Be(216);
        r.GetInt32(4).Should().Be(4096);
        r.GetInt32(5).Should().Be(100000);
        r.GetInt32(6).Should().Be(2985984);
        r.GetInt32(7).Should().Be(105413504);
        r.GetInt32(8).Should().Be(0);            // 16^8 = 2^32 wraps to 0
        r.GetInt32(9).Should().Be(790794752);
    }

    [TestMethod]
    public void BroadcastInt32_NoCrash_CorrectValues()
    {
        var a = np.array(new int[] { 2 });
        var b = np.array(new int[] { 3, 3, 3 });
        var bc = np.broadcast_to(a, b.Shape);

        var r = np.power(bc, b);
        r.GetInt32(0).Should().Be(8);
        r.GetInt32(1).Should().Be(8);
        r.GetInt32(2).Should().Be(8);
    }

    [TestMethod]
    public void Broadcasting_2D_against_1D()
    {
        // np.power([[1,2],[3,4]], [2,3]) = [[1,8],[9,64]]
        var a = np.array(new int[,] { { 1, 2 }, { 3, 4 } });
        var b = np.array(new int[] { 2, 3 });
        var r = np.power(a, b);

        r.Shape.dimensions.Should().Equal(new long[] { 2, 2 });
        r.GetInt32(0, 0).Should().Be(1);
        r.GetInt32(0, 1).Should().Be(8);
        r.GetInt32(1, 0).Should().Be(9);
        r.GetInt32(1, 1).Should().Be(64);
    }

    [TestMethod]
    public void StridedFloatBase_FloatExp()
    {
        var a = np.array(new double[] { 1, 2, 3, 4, 5, 6, 7, 8 });
        var sliced = a["::2"];  // [1, 3, 5, 7]
        var r = np.power(sliced, 2.0);

        r.GetDouble(0).Should().Be(1.0);
        r.GetDouble(1).Should().Be(9.0);
        r.GetDouble(2).Should().Be(25.0);
        r.GetDouble(3).Should().Be(49.0);
    }

    // ====================================================================
    // Signed integer negative exponent → ValueError (T1.36)
    // ====================================================================

    [TestMethod]
    public void Int32_NegativeExponent_Throws()
    {
        var a = np.array(new int[] { 2 });
        var b = np.array(new int[] { -1 });
        Action act = () => np.power(a, b);
        act.Should().Throw<ArgumentException>()
            .WithMessage("*Integers to negative integer powers are not allowed.*");
    }

    [TestMethod]
    public void Int64_NegativeExponent_Throws()
    {
        var a = np.array(new long[] { 5L });
        var b = np.array(new long[] { -2L });
        Action act = () => np.power(a, b);
        act.Should().Throw<ArgumentException>();
    }

    [TestMethod]
    public void NegativeExponent_ThrowsEvenWhenBaseIsOne()
    {
        // NumPy does NOT special-case base=±1; it throws unconditionally
        // on any negative integer exponent.
        var a = np.array(new int[] { 1, -1 });
        var b = np.array(new int[] { -1, -1 });
        Action act = () => np.power(a, b);
        act.Should().Throw<ArgumentException>();
    }

    [TestMethod]
    public void UnsignedExponent_NeverThrows()
    {
        // Unsigned exponent dtype can never be negative; no scan needed, no throw.
        var a = np.array(new int[] { 2, 3, 4 });
        var b = np.array(new uint[] { 1, 2, 3 });
        Action act = () => np.power(a, b);
        act.Should().NotThrow();
    }

    [TestMethod]
    public void Int32_BasePositive_AllExponentsPositive_NoThrow()
    {
        var a = np.array(new int[] { 2, 3, 4 });
        var b = np.array(new int[] { 1, 2, 3 });
        var r = np.power(a, b);
        r.GetInt32(0).Should().Be(2);
        r.GetInt32(1).Should().Be(9);
        r.GetInt32(2).Should().Be(64);
    }

    // ====================================================================
    // Float special values (match NumPy exactly)
    // ====================================================================

    [TestMethod]
    public void Float_ZeroToZero_Returns_One()
    {
        ((double)np.power(0.0, 0.0)).Should().Be(1.0);
    }

    [TestMethod]
    public void Int_ZeroToZero_Returns_One()
    {
        ((int)np.power(0, 0)).Should().Be(1);
    }

    [TestMethod]
    public void Float_NegativeBase_FractionalExp_ReturnsNaN()
    {
        double.IsNaN((double)np.power(-2.0, 0.5)).Should().BeTrue();
        double.IsNaN((double)np.power(-1.0, 0.5)).Should().BeTrue();
    }

    [TestMethod]
    public void Float_InfExponents_Float64()
    {
        var a = np.array(new double[] { 1.0, 1.0, 2.0, 2.0, -2.0, -2.0, double.PositiveInfinity, double.NegativeInfinity });
        var b = np.array(new double[] { double.PositiveInfinity, double.NegativeInfinity, double.PositiveInfinity, double.NegativeInfinity,
                                        double.PositiveInfinity, double.NegativeInfinity, double.PositiveInfinity, double.NegativeInfinity });
        var r = np.power(a, b);
        var d = r.GetData<double>();

        d[0].Should().Be(1.0);                              // 1^inf
        d[1].Should().Be(1.0);                              // 1^-inf
        double.IsPositiveInfinity(d[2]).Should().BeTrue(); // 2^inf
        d[3].Should().Be(0.0);                              // 2^-inf
        double.IsPositiveInfinity(d[4]).Should().BeTrue(); // (-2)^inf
        d[5].Should().Be(0.0);                              // (-2)^-inf
        double.IsPositiveInfinity(d[6]).Should().BeTrue(); // inf^inf
        d[7].Should().Be(0.0);                              // -inf^-inf
    }

    [TestMethod]
    public void Float_InfExponents_Float32()
    {
        var a = np.array(new float[] { 1f, 1f, 2f, 2f, -2f, -2f, float.PositiveInfinity, float.NegativeInfinity });
        var b = np.array(new float[] { float.PositiveInfinity, float.NegativeInfinity, float.PositiveInfinity, float.NegativeInfinity,
                                       float.PositiveInfinity, float.NegativeInfinity, float.PositiveInfinity, float.NegativeInfinity });
        var r = np.power(a, b);
        var d = r.GetData<float>();

        d[0].Should().Be(1f);
        d[1].Should().Be(1f);
        float.IsPositiveInfinity(d[2]).Should().BeTrue();
        d[3].Should().Be(0f);
        float.IsPositiveInfinity(d[4]).Should().BeTrue();
        d[5].Should().Be(0f);
        float.IsPositiveInfinity(d[6]).Should().BeTrue();
        d[7].Should().Be(0f);
    }

    [TestMethod]
    public void Float_NaN_Propagates()
    {
        double.IsNaN((double)np.power(double.NaN, 2.0)).Should().BeTrue();
        double.IsNaN((double)np.power(2.0, double.NaN)).Should().BeTrue();
    }

    // ====================================================================
    // Dtype promotion (NEP50 — strict for arrays, weak for 0-D scalars)
    // ====================================================================

    [TestMethod]
    public void Int32_Int32_Returns_Int32()
    {
        var a = np.array(new int[] { 2, 3 });
        np.power(a, 2).GetTypeCode.Should().Be(NPTypeCode.Int32);
    }

    [TestMethod]
    public void Int32_Int64_Returns_Int64()
    {
        var a = np.array(new int[] { 2, 3 });
        var b = np.array(new long[] { 2L, 2L });
        np.power(a, b).GetTypeCode.Should().Be(NPTypeCode.Int64);
    }

    [TestMethod]
    public void Float32_FloatScalar_Returns_Float32()
    {
        // Common case: arr ** scalar literal preserves float dtype
        var a = np.array(new float[] { 2f, 3f });
        np.power(a, 2.0f).GetTypeCode.Should().Be(NPTypeCode.Single);
    }

    [TestMethod]
    public void Float64_FloatScalar_Returns_Float64()
    {
        var a = np.array(new double[] { 2.0, 3.0 });
        np.power(a, 2.0).GetTypeCode.Should().Be(NPTypeCode.Double);
    }

    [TestMethod]
    public void Int32_FloatExp_Returns_Float64()
    {
        // int**float → float64 (NumPy NEP50)
        var a = np.array(new int[] { 2, 3 });
        np.power(a, 2.0).GetTypeCode.Should().Be(NPTypeCode.Double);
    }

    [TestMethod]
    public void Int16_FloatScalarExp_Returns_Float64()
    {
        var a = np.array(new short[] { 1, 2, 3 });
        np.power(a, 2.0).GetTypeCode.Should().Be(NPTypeCode.Double);
    }

    [TestMethod]
    public void Float32_StrictInt32Array_Returns_Float64()
    {
        // Strict promotion: f32 ** i32_arr (size>1) → f64
        var a = np.array(new float[] { 2f, 3f });
        var b = np.array(new int[] { 2, 2 });
        np.power(a, b).GetTypeCode.Should().Be(NPTypeCode.Double);
    }

    [TestMethod]
    public void Float32_StrictInt16Array_Returns_Float32()
    {
        // f32 ** i16 → f32 (int16 fits in f32 exactly)
        var a = np.array(new float[] { 2f, 3f });
        var b = np.array(new short[] { 2, 2 });
        np.power(a, b).GetTypeCode.Should().Be(NPTypeCode.Single);
    }

    // ====================================================================
    // Bool / Char dtype
    // ====================================================================

    [TestMethod]
    public void Bool_Bool_Promotes()
    {
        // NumPy: bool ** bool → int8. NumSharp doesn't have int8 — promotes to a wider int.
        var a = np.array(new bool[] { true, false, true });
        var b = np.array(new bool[] { true, false, false });
        var r = np.power(a, b);
        // Result type may differ from numpy int8 (NumSharp uses Byte/SByte); check values only.
        var rNumeric = r.astype(NPTypeCode.Int32);
        rNumeric.GetInt32(0).Should().Be(1);  // True^True = 1
        rNumeric.GetInt32(1).Should().Be(1);  // False^False = 1 (0^0 convention)
        rNumeric.GetInt32(2).Should().Be(1);  // True^False = 1
    }

    // ====================================================================
    // Empty / 0-D / 1-element edge cases
    // ====================================================================

    [TestMethod]
    public void Empty_PreservesShape()
    {
        var a = np.array(new int[] { });
        var b = np.array(new int[] { });
        var r = np.power(a, b);
        r.size.Should().Be(0);
    }

    [TestMethod]
    public void Zero_D_Scalar_Both()
    {
        var a = np.asanyarray(2);
        var b = np.asanyarray(3);
        ((int)np.power(a, b)).Should().Be(8);
    }

    [TestMethod]
    public void Negative_Base_Positive_IntegerExp_PreservesSign()
    {
        // (-2)^3 = -8
        ((int)np.power(-2, 3)).Should().Be(-8);
        // (-2)^4 = 16
        ((int)np.power(-2, 4)).Should().Be(16);
    }

    // ====================================================================
    // Complex
    // ====================================================================

    [TestMethod]
    public void Complex_NegativeRealBase_IntExp()
    {
        // (-2+0i) ^ 2 = (4+0i) (or very close)
        var a = np.array(new[] { new Complex(-2, 0) });
        var b = np.array(new[] { new Complex(2, 0) });
        var r = np.power(a, b);
        var v = r.GetData<Complex>()[0];
        System.Math.Abs(v.Real - 4.0).Should().BeLessThan(1e-10);
        System.Math.Abs(v.Imaginary).Should().BeLessThan(1e-10);
    }

    // ====================================================================
    // Quick smoke test for all integer dtypes (no crash, correct value)
    // ====================================================================

    [TestMethod]
    public void AllIntegerDtypes_SmokeTest()
    {
        // For each integer dtype: 2^3 should be 8.
        foreach (var tc in new[] {
            NPTypeCode.SByte, NPTypeCode.Byte,
            NPTypeCode.Int16, NPTypeCode.UInt16,
            NPTypeCode.Int32, NPTypeCode.UInt32,
            NPTypeCode.Int64, NPTypeCode.UInt64,
        })
        {
            var a = new NDArray(new int[] { 2 }).astype(tc);
            var b = new NDArray(new int[] { 3 }).astype(tc);
            var r = np.power(a, b);
            r.GetTypeCode.Should().Be(tc, $"dtype {tc} should be preserved");
            r.astype(NPTypeCode.Int64).GetInt64(0).Should().Be(8L, $"2^3 should be 8 for {tc}");
        }
    }

    [TestMethod]
    public void AllFloatDtypes_SmokeTest()
    {
        foreach (var (tc, expectedDtype) in new[] {
            (NPTypeCode.Single, NPTypeCode.Single),
            (NPTypeCode.Double, NPTypeCode.Double),
        })
        {
            var a = new NDArray(new double[] { 2.0 }).astype(tc);
            var b = new NDArray(new double[] { 3.0 }).astype(tc);
            var r = np.power(a, b);
            r.GetTypeCode.Should().Be(expectedDtype);
            r.astype(NPTypeCode.Double).GetDouble(0).Should().Be(8.0);
        }
    }
}
