using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;

namespace NumSharp.UnitTest.NewDtypes
{
    /// <summary>
    /// Round 13 — Arithmetic + operator sweep for Half / Complex / SByte,
    /// battletested against NumPy 2.4.2 (109-case matrix, 96.3% parity, the
    /// remaining 3.7% being documented BCL-level divergences for complex
    /// power-at-infinity and integer-by-zero with seterr-dependent semantics).
    ///
    /// Bugs closed:
    ///   B3  / B38 — Complex 1/0 returned (NaN, NaN). NumPy returns (inf, NaN)
    ///               via component-wise IEEE division.
    ///   B33       — Half/float/double floor_divide(inf, x) returned inf.
    ///               NumPy returns NaN per its npy_floor_divide rule (non-finite
    ///               a/b produces NaN).
    ///   B35       — Integer power (SByte/Byte/Int16-64) routed through
    ///               Math.Pow(double, double) which loses precision past 2^52.
    ///               Now uses native integer exponentiation with modular wrap.
    ///   B36       — np.reciprocal on integer dtypes promoted to float64. NumPy
    ///               preserves integer dtype with C-truncated 1/x (so 1/2 = 0).
    ///   B37       — np.floor / np.ceil / np.trunc on integer dtypes promoted
    ///               to float64. NumPy: no-op preserving input dtype.
    /// </summary>
    [TestClass]
    public class NewDtypesCoverageSweep_Arithmetic_Tests
    {
        private const double HalfTol = 1e-3;
        private static Complex C(double r, double i) => new Complex(r, i);

        #region B3 / B38 — Complex 1/0 via component-wise IEEE

        [TestMethod]
        public void B3_ComplexDivideByZero_Scalar()
        {
            // np.complex128(1+0j) / np.complex128(0+0j) == inf + nan*1j
            var a = np.array(new Complex[] { C(1, 0) });
            var b = np.array(new Complex[] { C(0, 0) });
            var r = (a / b).GetAtIndex<Complex>(0);
            double.IsPositiveInfinity(r.Real).Should().BeTrue();
            double.IsNaN(r.Imaginary).Should().BeTrue();
        }

        [TestMethod]
        public void B3_ComplexDivideByZero_NonzeroImag()
        {
            // (1+1j) / (0+0j) → component-wise: (1/0, 1/0) = (inf, inf)
            var a = np.array(new Complex[] { C(1, 1) });
            var b = np.array(new Complex[] { C(0, 0) });
            var r = (a / b).GetAtIndex<Complex>(0);
            double.IsPositiveInfinity(r.Real).Should().BeTrue();
            double.IsPositiveInfinity(r.Imaginary).Should().BeTrue();
        }

        [TestMethod]
        public void B3_ComplexZeroByZero_ReturnsNaN()
        {
            // (0+0j) / (0+0j) → (0/0, 0/0) = (NaN, NaN)
            var a = np.array(new Complex[] { C(0, 0) });
            var b = np.array(new Complex[] { C(0, 0) });
            var r = (a / b).GetAtIndex<Complex>(0);
            double.IsNaN(r.Real).Should().BeTrue();
            double.IsNaN(r.Imaginary).Should().BeTrue();
        }

        [TestMethod]
        public void B3_ComplexFiniteDivision_RegressionCheck()
        {
            // Ensure normal division path still works (BCL op_Division for b != 0)
            var a = np.array(new Complex[] { C(2, 3) });
            var b = np.array(new Complex[] { C(1, 0) });
            var r = (a / b).GetAtIndex<Complex>(0);
            r.Should().Be(C(2, 3));
        }

        #endregion

        #region B33 — Half/float/double floor_divide(inf, x) → NaN

        [TestMethod]
        public void B33_Half_FloorDivide_InfOverFinite_ReturnsNaN()
        {
            // NumPy: np.array([inf], f16) // np.array([1], f16) → [nan]
            var a = np.array(new Half[] { Half.PositiveInfinity });
            var b = np.array(new Half[] { (Half)1 });
            var r = np.floor_divide(a, b).GetAtIndex<Half>(0);
            Half.IsNaN(r).Should().BeTrue();
        }

        [TestMethod]
        public void B33_Half_FloorDivide_FiniteOverZero_ReturnsNaN()
        {
            // 1 / 0 = inf, floor(inf) should become nan per NumPy.
            var a = np.array(new Half[] { (Half)1 });
            var b = np.array(new Half[] { (Half)0 });
            var r = np.floor_divide(a, b).GetAtIndex<Half>(0);
            Half.IsNaN(r).Should().BeTrue();
        }

        [TestMethod]
        public void B33_Half_FloorDivide_FiniteOverFinite_NormalPath()
        {
            // Non-inf path: floor_divide should still work normally.
            var a = np.array(new Half[] { (Half)7, (Half)(-7) });
            var b = np.array(new Half[] { (Half)2, (Half)2 });
            var r = np.floor_divide(a, b);
            ((double)r.GetAtIndex<Half>(0)).Should().Be(3.0);
            ((double)r.GetAtIndex<Half>(1)).Should().Be(-4.0);  // floor(-3.5) = -4
        }

        [TestMethod]
        public void B33_Double_FloorDivide_InfReturnsNaN()
        {
            var a = np.array(new double[] { double.PositiveInfinity });
            var b = np.array(new double[] { 1.0 });
            var r = np.floor_divide(a, b).GetAtIndex<double>(0);
            double.IsNaN(r).Should().BeTrue();
        }

        #endregion

        #region B35 — Integer power with modular wrap

        [TestMethod]
        public void B35_SByte_Power_Overflow_WrapsModulo256()
        {
            // NumPy: np.array([50], i8) ** np.array([7], i8) = -128
            // (50^7 = 78_125_000_000 mod 256 = 128 wraps in int8 to -128)
            var a = np.array(new sbyte[] { 50 });
            var b = np.array(new sbyte[] { 7 });
            var r = np.power(a, b).GetAtIndex<sbyte>(0);
            r.Should().Be((sbyte)(-128));
        }

        [TestMethod]
        public void B35_SByte_Power_SmallExponent()
        {
            var a = np.array(new sbyte[] { 2, -3, 5 });
            var b = np.array(new sbyte[] { 3, 2, 0 });
            var r = np.power(a, b);
            r.GetAtIndex<sbyte>(0).Should().Be((sbyte)8);
            r.GetAtIndex<sbyte>(1).Should().Be((sbyte)9);
            r.GetAtIndex<sbyte>(2).Should().Be((sbyte)1);
        }

        [TestMethod]
        public void B35_SByte_Power_NegativeExponent_BaseGt1_ReturnsZero()
        {
            // NumPy: np.array([2], i8) ** np.array([-1], i8) = 0 (integer reciprocal)
            var a = np.array(new sbyte[] { 2, 100 });
            var b = np.array(new sbyte[] { -1, -3 });
            var r = np.power(a, b);
            r.GetAtIndex<sbyte>(0).Should().Be((sbyte)0);
            r.GetAtIndex<sbyte>(1).Should().Be((sbyte)0);
        }

        [TestMethod]
        public void B35_SByte_Power_NegativeExponent_BaseIs1_OrMinus1()
        {
            // 1^(-anything) = 1; (-1)^(-n) alternates ±1 per parity of n
            var a = np.array(new sbyte[] { 1, -1, -1 });
            var b = np.array(new sbyte[] { -5, -2, -3 });
            var r = np.power(a, b);
            r.GetAtIndex<sbyte>(0).Should().Be((sbyte)1);
            r.GetAtIndex<sbyte>(1).Should().Be((sbyte)1);   // (-1)^(-2) = (-1)^2 = 1
            r.GetAtIndex<sbyte>(2).Should().Be((sbyte)(-1)); // (-1)^(-3) = (-1)^3 = -1
        }

        [TestMethod]
        public void B35_Int32_Power_Wraps()
        {
            // 2^31 = 2147483648 wraps int32 to -2147483648
            var a = np.array(new int[] { 2 });
            var b = np.array(new int[] { 31 });
            var r = np.power(a, b).GetAtIndex<int>(0);
            r.Should().Be(int.MinValue);
        }

        #endregion

        #region B36 — SByte reciprocal preserves integer dtype

        [TestMethod]
        public void B36_SByte_Reciprocal_PreservesIntegerDtype()
        {
            // NumPy: np.reciprocal(np.array([1,-2,100,0], i8)) → array([1,0,0,0], i8)
            var a = np.array(new sbyte[] { 1, -2, 100, 0, 10, -50 });
            var r = np.reciprocal(a);
            r.typecode.Should().Be(NPTypeCode.SByte);
            r.GetAtIndex<sbyte>(0).Should().Be((sbyte)1);
            r.GetAtIndex<sbyte>(1).Should().Be((sbyte)0);
            r.GetAtIndex<sbyte>(2).Should().Be((sbyte)0);
            r.GetAtIndex<sbyte>(3).Should().Be((sbyte)0);  // 1/0 under seterr=ignore = 0
            r.GetAtIndex<sbyte>(4).Should().Be((sbyte)0);
            r.GetAtIndex<sbyte>(5).Should().Be((sbyte)0);
        }

        [TestMethod]
        public void B36_Int32_Reciprocal_PreservesIntegerDtype()
        {
            var a = np.array(new int[] { 1, -1, 2, -2, 0 });
            var r = np.reciprocal(a);
            r.typecode.Should().Be(NPTypeCode.Int32);
            r.GetAtIndex<int>(0).Should().Be(1);
            r.GetAtIndex<int>(1).Should().Be(-1);
            r.GetAtIndex<int>(2).Should().Be(0);
            r.GetAtIndex<int>(3).Should().Be(0);
            r.GetAtIndex<int>(4).Should().Be(0);
        }

        [TestMethod]
        public void B36_Half_Reciprocal_StillReturnsFloat()
        {
            // Regression: float inputs should still compute true 1/x, not integer division.
            var a = np.array(new Half[] { (Half)2, (Half)0.5 });
            var r = np.reciprocal(a);
            r.typecode.Should().Be(NPTypeCode.Half);
            ((double)r.GetAtIndex<Half>(0)).Should().BeApproximately(0.5, HalfTol);
            ((double)r.GetAtIndex<Half>(1)).Should().BeApproximately(2.0, HalfTol);
        }

        #endregion

        #region B37 — floor/ceil/trunc preserve integer dtypes

        [TestMethod]
        public void B37_SByte_Floor_NoOp_PreservesDtype()
        {
            var a = np.array(new sbyte[] { 1, -2, 100, -100 });
            var r = np.floor(a);
            r.typecode.Should().Be(NPTypeCode.SByte);
            for (int i = 0; i < 4; i++)
                r.GetAtIndex<sbyte>(i).Should().Be(a.GetAtIndex<sbyte>(i));
        }

        [TestMethod]
        public void B37_SByte_Ceil_NoOp_PreservesDtype()
        {
            var a = np.array(new sbyte[] { 0, 127, -128, 42 });
            var r = np.ceil(a);
            r.typecode.Should().Be(NPTypeCode.SByte);
            for (int i = 0; i < 4; i++)
                r.GetAtIndex<sbyte>(i).Should().Be(a.GetAtIndex<sbyte>(i));
        }

        [TestMethod]
        public void B37_SByte_Trunc_NoOp_PreservesDtype()
        {
            var a = np.array(new sbyte[] { -50, 50, 0, 1 });
            var r = np.trunc(a);
            r.typecode.Should().Be(NPTypeCode.SByte);
            for (int i = 0; i < 4; i++)
                r.GetAtIndex<sbyte>(i).Should().Be(a.GetAtIndex<sbyte>(i));
        }

        [TestMethod]
        public void B37_Int32_Floor_NoOp_PreservesDtype()
        {
            var a = np.array(new int[] { 1, 1000000, -1000000 });
            var r = np.floor(a);
            r.typecode.Should().Be(NPTypeCode.Int32);
            r.GetAtIndex<int>(0).Should().Be(1);
            r.GetAtIndex<int>(1).Should().Be(1000000);
            r.GetAtIndex<int>(2).Should().Be(-1000000);
        }

        [TestMethod]
        public void B37_Half_Floor_StillWorksForFloat()
        {
            // Regression: float inputs should still floor normally.
            var a = np.array(new Half[] { (Half)1.7, (Half)(-1.7) });
            var r = np.floor(a);
            r.typecode.Should().Be(NPTypeCode.Half);
            ((double)r.GetAtIndex<Half>(0)).Should().Be(1.0);
            ((double)r.GetAtIndex<Half>(1)).Should().Be(-2.0);
        }

        #endregion

        #region Round 13 — arithmetic smoke tests for Half / Complex / SByte

        [TestMethod]
        public void Arith_Half_AddSubMulDiv_ArrayArray()
        {
            var a = np.array(new Half[] { (Half)1, (Half)2 });
            var b = np.array(new Half[] { (Half)0.5, (Half)(-1) });
            ((double)(a + b).GetAtIndex<Half>(0)).Should().BeApproximately(1.5, HalfTol);
            ((double)(a - b).GetAtIndex<Half>(1)).Should().BeApproximately(3.0, HalfTol);
            ((double)(a * b).GetAtIndex<Half>(0)).Should().BeApproximately(0.5, HalfTol);
            ((double)(a / b).GetAtIndex<Half>(1)).Should().BeApproximately(-2.0, HalfTol);
        }

        [TestMethod]
        public void Arith_Complex_AddSubMulDiv_ArrayArray()
        {
            var a = np.array(new Complex[] { C(1, 2), C(3, -4) });
            var b = np.array(new Complex[] { C(2, 1), C(1, 1) });
            (a + b).GetAtIndex<Complex>(0).Should().Be(C(3, 3));
            (a - b).GetAtIndex<Complex>(1).Should().Be(C(2, -5));
            (a * b).GetAtIndex<Complex>(0).Should().Be(C(0, 5));
            // (3-4j)/(1+1j) = ((3*1 + -4*1) + (-4*1 - 3*1)i) / (1+1) = (-1 - 7j) / 2
            (a / b).GetAtIndex<Complex>(1).Should().Be(C(-0.5, -3.5));
        }

        [TestMethod]
        public void Arith_SByte_AddSubMul_Wraps()
        {
            var a = np.array(new sbyte[] { 100, 1, -50 });
            var b = np.array(new sbyte[] { 50, -1, -50 });
            (a + b).GetAtIndex<sbyte>(0).Should().Be((sbyte)(-106)); // 150 wraps to -106
            (a - b).GetAtIndex<sbyte>(1).Should().Be((sbyte)2);
            (a * b).GetAtIndex<sbyte>(2).Should().Be((sbyte)(-60)); // (-50)*(-50)=2500 mod 256 = 196 -> signed -60
        }

        [TestMethod]
        public void Arith_SByte_Overflow_127Plus1_WrapsToMinus128()
        {
            var a = np.array(new sbyte[] { 127 });
            var b = np.array(new sbyte[] { 1 });
            (a + b).GetAtIndex<sbyte>(0).Should().Be((sbyte)(-128));
        }

        [TestMethod]
        public void Unary_Negate_Half()
        {
            var a = np.array(new Half[] { (Half)1, (Half)(-2), (Half)0 });
            var r = -a;
            ((double)r.GetAtIndex<Half>(0)).Should().Be(-1.0);
            ((double)r.GetAtIndex<Half>(1)).Should().Be(2.0);
            ((double)r.GetAtIndex<Half>(2)).Should().Be(-0.0);  // signed zero
        }

        [TestMethod]
        public void Unary_Negate_Complex()
        {
            var a = np.array(new Complex[] { C(1, 2), C(-3, 4) });
            var r = -a;
            r.GetAtIndex<Complex>(0).Should().Be(C(-1, -2));
            r.GetAtIndex<Complex>(1).Should().Be(C(3, -4));
        }

        [TestMethod]
        public void Unary_Negate_SByte_Wraps()
        {
            // -(-128) wraps back to -128 in int8 (since 128 doesn't fit)
            var a = np.array(new sbyte[] { 1, -1, -128 });
            var r = -a;
            r.GetAtIndex<sbyte>(0).Should().Be((sbyte)(-1));
            r.GetAtIndex<sbyte>(1).Should().Be((sbyte)1);
            r.GetAtIndex<sbyte>(2).Should().Be((sbyte)(-128)); // wrap
        }

        [TestMethod]
        public void Abs_Complex_ReturnsFloat64Magnitude()
        {
            // NumPy: abs(3+4j) = 5.0 (float64)
            var a = np.array(new Complex[] { C(3, 4), C(-5, 12) });
            var r = np.abs(a);
            r.typecode.Should().Be(NPTypeCode.Double);
            r.GetAtIndex<double>(0).Should().BeApproximately(5.0, 1e-12);
            r.GetAtIndex<double>(1).Should().BeApproximately(13.0, 1e-12);
        }

        [TestMethod]
        public void Square_Complex()
        {
            // (1+2j)^2 = 1+4j-4 = -3+4j
            var a = np.array(new Complex[] { C(1, 2) });
            var r = np.square(a);
            r.GetAtIndex<Complex>(0).Should().Be(C(-3, 4));
        }

        [TestMethod]
        public void Sign_Half_IEEEZeroAndNaN()
        {
            var a = np.array(new Half[] { (Half)1, (Half)(-2), (Half)0, Half.NaN });
            var r = np.sign(a);
            ((double)r.GetAtIndex<Half>(0)).Should().Be(1.0);
            ((double)r.GetAtIndex<Half>(1)).Should().Be(-1.0);
            ((double)r.GetAtIndex<Half>(2)).Should().Be(0.0);
            Half.IsNaN(r.GetAtIndex<Half>(3)).Should().BeTrue();
        }

        [TestMethod]
        public void Broadcasting_Half_MatrixPlusVector()
        {
            var mat = np.array(new Half[,] { { (Half)1, (Half)2, (Half)3 }, { (Half)4, (Half)5, (Half)6 } });
            var vec = np.array(new Half[] { (Half)10, (Half)20, (Half)30 });
            var r = mat + vec;
            r.shape.Should().Equal(new long[] { 2, 3 });
            ((double)r.GetAtIndex<Half>(0)).Should().Be(11.0);
            ((double)r.GetAtIndex<Half>(5)).Should().Be(36.0);
        }

        [TestMethod]
        public void Broadcasting_Complex_MatrixPlusVector()
        {
            var mat = np.array(new Complex[,] { { C(1, 0), C(2, 0) }, { C(3, 0), C(4, 0) } });
            var vec = np.array(new Complex[] { C(1, 1), C(1, -1) });
            var r = mat + vec;
            r.GetAtIndex<Complex>(0).Should().Be(C(2, 1));
            r.GetAtIndex<Complex>(3).Should().Be(C(5, -1));
        }

        #endregion
    }
}
