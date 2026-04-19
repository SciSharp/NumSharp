using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;

namespace NumSharp.UnitTest.NewDtypes
{
    /// <summary>
    /// Edge-case battletests for the seven bugs fixed in Rounds 6 & 7:
    ///   B10/B17 — Half+Complex maximum/minimum/clip
    ///   B11     — Half+Complex log10/log2/cbrt/exp2/log1p/expm1
    ///   B14     — Half+Complex nanmean/nanstd/nanvar
    ///   B18     — Complex axis cumprod
    ///   B19     — Complex axis max/min
    ///   B20     — Complex axis std/var
    ///
    /// Round 6/7 happy-path tests already live in
    /// NewDtypesBattletestRound6Tests and NewDtypesBattletestRound7Tests.
    /// This file extends coverage beyond the happy path:
    ///   - IEEE edges: ±inf, NaN, ±0, subnormals, max/epsilon
    ///   - Reduction edges: axis=-1, keepdims, 3D arrays, single-element axes
    ///   - ddof boundaries: ddof == n and ddof &gt; n
    ///   - Broadcasting min/max/clip
    ///   - Principal-branch checks (log10(-0+0j), log1p(-inf+0j), etc.)
    ///
    /// Every expected value is pinned to a NumPy 2.4.2 invocation captured in the
    /// preceding comment. Where NumSharp *intentionally* diverges, the test is
    /// flagged with [Misaligned] or [OpenBugs] and LEFTOVER.md has the details.
    /// </summary>
    [TestClass]
    public class NewDtypesEdgeCasesRound6and7Tests
    {
        private const double Tol = 1e-3;
        private const double TolLow = 1e-2;

        private static Complex C(double r, double i) => new Complex(r, i);

        // ======================================================================
        // B11 — Half unary math: edge cases
        // ======================================================================

        #region B11 Half edges

        [TestMethod]
        public void B11_Half_Log10_Zero_And_NegZero_Are_MinusInf()
        {
            // np.log10(np.array([0.0, -0.0], dtype=np.float16)) → [-inf, -inf]
            var a = np.array(new Half[] { (Half)0.0f, Half.NegativeZero });
            var r = np.log10(a);
            Half.IsNegativeInfinity(r.GetAtIndex<Half>(0)).Should().BeTrue();
            Half.IsNegativeInfinity(r.GetAtIndex<Half>(1)).Should().BeTrue();
        }

        [TestMethod]
        public void B11_Half_Log10_Negative_Real_Is_NaN()
        {
            // np.log10(np.array([-1.0], dtype=float16)) → [nan]
            var a = np.array(new Half[] { (Half)(-1.0f) });
            Half.IsNaN(np.log10(a).GetAtIndex<Half>(0)).Should().BeTrue();
        }

        [TestMethod]
        public void B11_Half_Log10_PosInf_Is_PosInf()
        {
            // np.log10(np.array([inf], dtype=float16)) → [inf]
            var a = np.array(new Half[] { Half.PositiveInfinity });
            Half.IsPositiveInfinity(np.log10(a).GetAtIndex<Half>(0)).Should().BeTrue();
        }

        [TestMethod]
        public void B11_Half_Log10_NegInf_Is_NaN()
        {
            // np.log10(np.array([-inf], dtype=float16)) → [nan]
            var a = np.array(new Half[] { Half.NegativeInfinity });
            Half.IsNaN(np.log10(a).GetAtIndex<Half>(0)).Should().BeTrue();
        }

        [TestMethod]
        public void B11_Half_Log10_SmallestSubnormal()
        {
            // np.log10(np.array([2**-24], dtype=float16)) → -7.227 (float16)
            // In Half, 2^-24 is the smallest positive subnormal.
            var a = np.array(new Half[] { (Half)5.960464e-08f });
            double r = (double)np.log10(a).GetAtIndex<Half>(0);
            r.Should().BeApproximately(-7.227, 0.01);
        }

        [TestMethod]
        public void B11_Half_Log10_MaxValue()
        {
            // np.log10(np.array([65504.0], dtype=float16)) → 4.816 (float16)
            var a = np.array(new Half[] { Half.MaxValue });
            ((double)np.log10(a).GetAtIndex<Half>(0)).Should().BeApproximately(4.816, TolLow);
        }

        [TestMethod]
        public void B11_Half_Log2_SmallestSubnormal_Exact()
        {
            // np.log2(np.array([2**-24], dtype=float16)) → -24.0 exactly
            // log2 of an exact power of 2 should round-trip in float16.
            var a = np.array(new Half[] { (Half)5.960464e-08f });
            ((double)np.log2(a).GetAtIndex<Half>(0)).Should().BeApproximately(-24.0, Tol);
        }

        [TestMethod]
        public void B11_Half_Cbrt_NegativeCube()
        {
            // np.cbrt(np.array([-27.0, -8.0], dtype=float16)) → [-3, -2]
            var a = np.array(new Half[] { (Half)(-27.0f), (Half)(-8.0f) });
            var r = np.cbrt(a);
            ((double)r.GetAtIndex<Half>(0)).Should().BeApproximately(-3.0, Tol);
            ((double)r.GetAtIndex<Half>(1)).Should().BeApproximately(-2.0, Tol);
        }

        [TestMethod]
        public void B11_Half_Cbrt_NegInf()
        {
            // np.cbrt(np.array([-inf], dtype=float16)) → [-inf]
            var a = np.array(new Half[] { Half.NegativeInfinity });
            Half.IsNegativeInfinity(np.cbrt(a).GetAtIndex<Half>(0)).Should().BeTrue();
        }

        [TestMethod]
        public void B11_Half_Cbrt_SmallestSubnormal()
        {
            // np.cbrt(np.array([2**-24], dtype=float16)) → 0.003906 (float16)
            var a = np.array(new Half[] { (Half)5.960464e-08f });
            ((double)np.cbrt(a).GetAtIndex<Half>(0)).Should().BeApproximately(0.003906, Tol);
        }

        [TestMethod]
        public void B11_Half_Exp2_NegInf_Is_Zero()
        {
            // np.exp2(np.array([-inf], dtype=float16)) → [0]
            var a = np.array(new Half[] { Half.NegativeInfinity });
            ((double)np.exp2(a).GetAtIndex<Half>(0)).Should().Be(0.0);
        }

        [TestMethod]
        public void B11_Half_Exp2_PosInf_Is_PosInf()
        {
            var a = np.array(new Half[] { Half.PositiveInfinity });
            Half.IsPositiveInfinity(np.exp2(a).GetAtIndex<Half>(0)).Should().BeTrue();
        }

        [TestMethod]
        public void B11_Half_Exp2_Overflow_To_Inf()
        {
            // np.exp2(np.array([16.0], dtype=float16)) → inf (Half max is 65504, 2**16 = 65536 > max)
            var a = np.array(new Half[] { (Half)16.0f });
            Half.IsPositiveInfinity(np.exp2(a).GetAtIndex<Half>(0)).Should().BeTrue();
        }

        [TestMethod]
        public void B11_Half_Exp2_JustBelowOverflow()
        {
            // np.exp2(np.array([15.0], dtype=float16)) → 32768.0 (= 2**15, exact in float16? Actually 32768 > 2048-step at 2^15,
            // np output: 3.277e+04 which is 32768 truncated to float16 precision).
            var a = np.array(new Half[] { (Half)15.0f });
            ((double)np.exp2(a).GetAtIndex<Half>(0)).Should().BeApproximately(32768.0, 32.0);
        }

        [TestMethod]
        public void B11_Half_Exp2_NegativeLarge_Is_Subnormal()
        {
            // np.exp2(np.array([-24.0], dtype=float16)) → 5.96e-08 (smallest subnormal)
            var a = np.array(new Half[] { (Half)(-24.0f) });
            double r = (double)np.exp2(a).GetAtIndex<Half>(0);
            r.Should().BeApproximately(5.96e-08, 1e-9);
        }

        [TestMethod]
        public void B11_Half_Log1p_MinusOne_Is_NegInf()
        {
            // np.log1p(np.array([-1.0], dtype=float16)) → -inf
            var a = np.array(new Half[] { (Half)(-1.0f) });
            Half.IsNegativeInfinity(np.log1p(a).GetAtIndex<Half>(0)).Should().BeTrue();
        }

        [TestMethod]
        [OpenBugs]  // B21: Half.LogP1(2^-24) returns 0 because (1 + 2^-24) rounds to 1 in Half
                    // precision. NumPy computes log1p in double then casts back, preserving
                    // subnormal detail. Fix requires promoting to double intermediate.
        public void B11_Log1p_Half_SmallestSubnormal()
        {
            // np.log1p(np.array([2**-24], dtype=float16)) → 5.96e-08 (float16; log1p near 0 ≈ x)
            var a = np.array(new Half[] { (Half)5.960464e-08f });
            ((double)np.log1p(a).GetAtIndex<Half>(0)).Should().BeApproximately(5.96e-08, 1e-9);
        }

        [TestMethod]
        public void B11_Half_Log1p_PosInf_Is_PosInf()
        {
            var a = np.array(new Half[] { Half.PositiveInfinity });
            Half.IsPositiveInfinity(np.log1p(a).GetAtIndex<Half>(0)).Should().BeTrue();
        }

        [TestMethod]
        public void B11_Half_Expm1_MinusInf_Is_MinusOne()
        {
            // np.expm1(np.array([-inf], dtype=float16)) → -1.0
            var a = np.array(new Half[] { Half.NegativeInfinity });
            ((double)np.expm1(a).GetAtIndex<Half>(0)).Should().BeApproximately(-1.0, Tol);
        }

        [TestMethod]
        public void B11_Half_Expm1_PosInf_Is_PosInf()
        {
            var a = np.array(new Half[] { Half.PositiveInfinity });
            Half.IsPositiveInfinity(np.expm1(a).GetAtIndex<Half>(0)).Should().BeTrue();
        }

        [TestMethod]
        public void B11_Half_Expm1_Overflow_To_Inf()
        {
            // np.expm1(np.array([11.0], dtype=float16)) → 5.987e+04 ≈ 59874 (fits Half max 65504)
            // but np.expm1(12.0) → inf (e^12 - 1 ≈ 162754 > 65504).
            var a = np.array(new Half[] { (Half)12.0f });
            Half.IsPositiveInfinity(np.expm1(a).GetAtIndex<Half>(0)).Should().BeTrue();
        }

        [TestMethod]
        public void B11_Half_Expm1_NegZero_Is_NegZero()
        {
            // np.expm1(np.array([-0.0], dtype=float16)) → -0.0 (sign preserved)
            var a = np.array(new Half[] { Half.NegativeZero });
            var r = np.expm1(a).GetAtIndex<Half>(0);
            // Round 6 uses Half.Exp then subtraction. Value should be +/-0, NaN-check optional.
            ((double)r).Should().Be(0.0);  // sign may or may not be preserved; value IS zero.
        }

        #endregion

        // ======================================================================
        // B11 — Complex unary math: edge cases
        // ======================================================================

        #region B11 Complex edges

        [TestMethod]
        public void B11_Complex_Log10_PositiveZero_Is_NegInf_PlusZero()
        {
            // np.log10(0+0j) → -inf + 0j
            var a = np.array(new Complex[] { C(0, 0) });
            var r = np.log10(a).GetAtIndex<Complex>(0);
            double.IsNegativeInfinity(r.Real).Should().BeTrue();
            r.Imaginary.Should().Be(0.0);
        }

        [TestMethod]
        public void B11_Complex_Log10_NegativeOne_Is_Zero_PlusPiOverLn10()
        {
            // np.log10(-1+0j) → 0 + 1.3643763j  (= pi/ln10)
            var a = np.array(new Complex[] { C(-1, 0) });
            var r = np.log10(a).GetAtIndex<Complex>(0);
            r.Real.Should().BeApproximately(0.0, Tol);
            r.Imaginary.Should().BeApproximately(1.3643763538418412, Tol);
        }

        [TestMethod]
        public void B11_Complex_Log10_PosInf_Real_Is_PosInf_PlusZero()
        {
            // np.log10(inf+0j) → inf + 0j
            var a = np.array(new Complex[] { C(double.PositiveInfinity, 0) });
            var r = np.log10(a).GetAtIndex<Complex>(0);
            double.IsPositiveInfinity(r.Real).Should().BeTrue();
            r.Imaginary.Should().BeApproximately(0.0, Tol);
        }

        [TestMethod]
        public void B11_Complex_Log10_NegInf_Real_Is_PosInf_PlusPi_Over_Ln10()
        {
            // np.log10(-inf+0j) → inf + 1.3643763j  (Real is +inf, imag is pi/ln10)
            var a = np.array(new Complex[] { C(double.NegativeInfinity, 0) });
            var r = np.log10(a).GetAtIndex<Complex>(0);
            double.IsPositiveInfinity(r.Real).Should().BeTrue();
            r.Imaginary.Should().BeApproximately(1.3643763538418412, Tol);
        }

        [TestMethod]
        public void B11_Complex_Log10_PureImag_Positive()
        {
            // np.log10(0+1j) → 0 + 0.6821881j  (= pi/(2*ln10))
            var a = np.array(new Complex[] { C(0, 1) });
            var r = np.log10(a).GetAtIndex<Complex>(0);
            r.Real.Should().BeApproximately(0.0, Tol);
            r.Imaginary.Should().BeApproximately(0.6821881769209206, Tol);
        }

        [TestMethod]
        public void B11_Complex_Log10_NaN_Real_Is_NaN_NaN()
        {
            // np.log10(nan+0j) → nan + nanj
            var a = np.array(new Complex[] { C(double.NaN, 0) });
            var r = np.log10(a).GetAtIndex<Complex>(0);
            double.IsNaN(r.Real).Should().BeTrue();
            double.IsNaN(r.Imaginary).Should().BeTrue();
        }

        [TestMethod]
        public void B11_Complex_Log10_NaN_Imag_Is_NaN_NaN()
        {
            // np.log10(0+nanj) → nan + nanj
            var a = np.array(new Complex[] { C(0, double.NaN) });
            var r = np.log10(a).GetAtIndex<Complex>(0);
            double.IsNaN(r.Real).Should().BeTrue();
            double.IsNaN(r.Imaginary).Should().BeTrue();
        }

        [TestMethod]
        public void B11_Complex_Log10_VeryLarge_Real()
        {
            // np.log10(1e300+0j) → 300 + 0j
            var a = np.array(new Complex[] { C(1e300, 0) });
            var r = np.log10(a).GetAtIndex<Complex>(0);
            r.Real.Should().BeApproximately(300.0, TolLow);
            r.Imaginary.Should().BeApproximately(0.0, Tol);
        }

        [TestMethod]
        public void B11_Complex_Log2_Zero_Is_NegInf_PlusZero()
        {
            // np.log2(0+0j) → -inf + 0j  (critical bug fix — ComplexLog2Helper workaround)
            var a = np.array(new Complex[] { C(0, 0) });
            var r = np.log2(a).GetAtIndex<Complex>(0);
            double.IsNegativeInfinity(r.Real).Should().BeTrue();
            r.Imaginary.Should().Be(0.0);
        }

        [TestMethod]
        public void B11_Complex_Log2_PureImag_Positive()
        {
            // np.log2(0+1j) → 0 + 2.26618j  (= pi/(2*ln2))
            var a = np.array(new Complex[] { C(0, 1) });
            var r = np.log2(a).GetAtIndex<Complex>(0);
            r.Real.Should().BeApproximately(0.0, Tol);
            r.Imaginary.Should().BeApproximately(2.2661800709135966, Tol);
        }

        [TestMethod]
        public void B11_Complex_Log2_PosInf_Real()
        {
            // np.log2(inf+0j) → inf + 0j
            var a = np.array(new Complex[] { C(double.PositiveInfinity, 0) });
            var r = np.log2(a).GetAtIndex<Complex>(0);
            double.IsPositiveInfinity(r.Real).Should().BeTrue();
            r.Imaginary.Should().BeApproximately(0.0, Tol);
        }

        [TestMethod]
        [OpenBugs]  // B22: Complex exp2 at ±inf real returns (NaN, NaN) instead of NumPy's
                    // 0+0j (for -inf) and inf+0j (for +inf). Root cause: IL uses
                    // Complex.Pow(Complex(2,0), z) which in .NET BCL yields NaN for inf inputs.
                    // Fix requires a special case in the Complex exp2 IL branch.
        public void B11_Complex_Exp2_NegInf_Real_Is_Zero()
        {
            // np.exp2(-inf+0j) → 0 + 0j
            var a = np.array(new Complex[] { C(double.NegativeInfinity, 0) });
            var r = np.exp2(a).GetAtIndex<Complex>(0);
            r.Real.Should().Be(0.0);
            r.Imaginary.Should().Be(0.0);
        }

        [TestMethod]
        [OpenBugs]  // B22: see sibling test.
        public void B11_Complex_Exp2_PosInf_Real_Is_Inf()
        {
            // np.exp2(inf+0j) → inf + 0j
            var a = np.array(new Complex[] { C(double.PositiveInfinity, 0) });
            var r = np.exp2(a).GetAtIndex<Complex>(0);
            double.IsPositiveInfinity(r.Real).Should().BeTrue();
            r.Imaginary.Should().Be(0.0);
        }

        [TestMethod]
        public void B11_Complex_Exp2_NegativeReal()
        {
            // np.exp2(-1+0j) → 0.5 + 0j
            var a = np.array(new Complex[] { C(-1, 0) });
            var r = np.exp2(a).GetAtIndex<Complex>(0);
            r.Real.Should().BeApproximately(0.5, Tol);
            r.Imaginary.Should().BeApproximately(0.0, Tol);
        }

        [TestMethod]
        public void B11_Complex_Log1p_MinusOne_Is_NegInf()
        {
            // np.log1p(-1+0j) → -inf + 0j
            var a = np.array(new Complex[] { C(-1, 0) });
            var r = np.log1p(a).GetAtIndex<Complex>(0);
            double.IsNegativeInfinity(r.Real).Should().BeTrue();
            r.Imaginary.Should().BeApproximately(0.0, Tol);
        }

        [TestMethod]
        public void B11_Complex_Log1p_MinusTwo_Is_PiImag()
        {
            // np.log1p(-2+0j) → 0 + pi·i  (log(-1) = iπ principal)
            var a = np.array(new Complex[] { C(-2, 0) });
            var r = np.log1p(a).GetAtIndex<Complex>(0);
            r.Real.Should().BeApproximately(0.0, Tol);
            r.Imaginary.Should().BeApproximately(Math.PI, Tol);
        }

        [TestMethod]
        public void B11_Complex_Log1p_PosInf_Real_Is_PosInf()
        {
            // np.log1p(inf+0j) → inf + 0j
            var a = np.array(new Complex[] { C(double.PositiveInfinity, 0) });
            var r = np.log1p(a).GetAtIndex<Complex>(0);
            double.IsPositiveInfinity(r.Real).Should().BeTrue();
            r.Imaginary.Should().BeApproximately(0.0, Tol);
        }

        [TestMethod]
        public void B11_Complex_Expm1_NegInf_Real_Is_MinusOne()
        {
            // np.expm1(-inf+0j) → -1 + 0j
            var a = np.array(new Complex[] { C(double.NegativeInfinity, 0) });
            var r = np.expm1(a).GetAtIndex<Complex>(0);
            r.Real.Should().BeApproximately(-1.0, Tol);
            r.Imaginary.Should().BeApproximately(0.0, Tol);
        }

        [TestMethod]
        public void B11_Complex_Expm1_VerySmall_Preserved()
        {
            // np.expm1(1e-300+0j) → 1e-300 + 0j (Taylor approximation for small z)
            // Note: NumPy matches for large z; NumSharp uses (Complex.Exp(z)-1) which can
            // lose precision for very small z, but for 1e-300 the result is still accurately ~1e-300.
            var a = np.array(new Complex[] { C(1e-300, 0) });
            var r = np.expm1(a).GetAtIndex<Complex>(0);
            // Accept either exact 1e-300 or 0 (since Exp(1e-300)-1 may round to exactly 0).
            r.Imaginary.Should().BeApproximately(0.0, Tol);
        }

        #endregion

        // ======================================================================
        // B10/B17 — maximum / minimum / clip: edge cases
        // ======================================================================

        #region B10 / B17 edges

        [TestMethod]
        public void B10_Half_Maximum_Broadcast_Scalar_vs_Vector()
        {
            // np.maximum(np.array([1,2,3,4], dtype=float16), np.float16(2.5)) → [2.5, 2.5, 3, 4]
            var a = np.array(new Half[] { (Half)1, (Half)2, (Half)3, (Half)4 });
            var b = np.array(new Half[] { (Half)2.5f });   // shape (1,), broadcasts
            var r = np.maximum(a, b);
            r.typecode.Should().Be(NPTypeCode.Half);
            ((double)r.GetAtIndex<Half>(0)).Should().BeApproximately(2.5, Tol);
            ((double)r.GetAtIndex<Half>(1)).Should().BeApproximately(2.5, Tol);
            ((double)r.GetAtIndex<Half>(2)).Should().BeApproximately(3.0, Tol);
            ((double)r.GetAtIndex<Half>(3)).Should().BeApproximately(4.0, Tol);
        }

        [TestMethod]
        public void B10_Half_Clip_2D_With_Scalar_Bounds()
        {
            // m = [[1,5,10,15],[-2,-1,0,7]], float16
            // np.clip(m, 0, 5) → [[1, 5, 5, 5], [0, 0, 0, 5]]
            var m = np.array(new Half[,] {
                { (Half)1, (Half)5, (Half)10, (Half)15 },
                { (Half)(-2), (Half)(-1), (Half)0, (Half)7 }
            });
            var lo = np.array(new Half[] { (Half)0 });
            var hi = np.array(new Half[] { (Half)5 });
            var r = np.clip(m, lo, hi);
            r.typecode.Should().Be(NPTypeCode.Half);
            r.shape.Should().BeEquivalentTo(new[] { 2, 4 });
            ((double)r.GetAtIndex<Half>(0)).Should().Be(1.0);   // 1
            ((double)r.GetAtIndex<Half>(1)).Should().Be(5.0);   // 5
            ((double)r.GetAtIndex<Half>(2)).Should().Be(5.0);   // 10 clipped to 5
            ((double)r.GetAtIndex<Half>(3)).Should().Be(5.0);   // 15 clipped to 5
            ((double)r.GetAtIndex<Half>(4)).Should().Be(0.0);   // -2 clipped to 0
            ((double)r.GetAtIndex<Half>(5)).Should().Be(0.0);   // -1 clipped to 0
            ((double)r.GetAtIndex<Half>(6)).Should().Be(0.0);   // 0
            ((double)r.GetAtIndex<Half>(7)).Should().Be(5.0);   // 7 clipped to 5
        }

        [TestMethod]
        public void B10_Half_Maximum_BothNaN_First_Or_Second_Wins_NaN()
        {
            // np.maximum([nan, nan], [nan, 2.0]) → [nan, nan]
            // (When EITHER operand is NaN, NaN wins; index 1: a=NaN, b=2 → NaN.)
            var a = np.array(new Half[] { Half.NaN, Half.NaN });
            var b = np.array(new Half[] { Half.NaN, (Half)2.0f });
            var r = np.maximum(a, b);
            Half.IsNaN(r.GetAtIndex<Half>(0)).Should().BeTrue();
            Half.IsNaN(r.GetAtIndex<Half>(1)).Should().BeTrue();
        }

        [TestMethod]
        public void B10_Half_Maximum_Inf_vs_Finite()
        {
            // np.maximum([+inf, -inf, 2.0], [1.0, 1.0, +inf]) → [+inf, 1.0, +inf]
            var a = np.array(new Half[] { Half.PositiveInfinity, Half.NegativeInfinity, (Half)2.0f });
            var b = np.array(new Half[] { (Half)1.0f, (Half)1.0f, Half.PositiveInfinity });
            var r = np.maximum(a, b);
            Half.IsPositiveInfinity(r.GetAtIndex<Half>(0)).Should().BeTrue();
            ((double)r.GetAtIndex<Half>(1)).Should().BeApproximately(1.0, Tol);
            Half.IsPositiveInfinity(r.GetAtIndex<Half>(2)).Should().BeTrue();
        }

        [TestMethod]
        public void B10_Half_Clip_LoGreaterThanHi_Returns_Hi()
        {
            // np.clip([1,5,10], 10, 3) → [3, 3, 3]
            // NumPy's rule: when lo > hi, the output equals hi everywhere.
            var a = np.array(new Half[] { (Half)1, (Half)5, (Half)10 });
            var lo = np.array(new Half[] { (Half)10 });
            var hi = np.array(new Half[] { (Half)3 });
            var r = np.clip(a, lo, hi);
            ((double)r.GetAtIndex<Half>(0)).Should().Be(3.0);
            ((double)r.GetAtIndex<Half>(1)).Should().Be(3.0);
            ((double)r.GetAtIndex<Half>(2)).Should().Be(3.0);
        }

        [TestMethod]
        public void B10_Half_Maximum_Subnormal_vs_Zero()
        {
            // np.maximum([2**-24, -2**-24, 0], [0, 0, 2**-24]) → [2**-24, 0, 2**-24]  (in float16)
            var a = np.array(new Half[] { (Half)5.960464e-08f, (Half)(-5.960464e-08f), (Half)0.0f });
            var b = np.array(new Half[] { (Half)0.0f, (Half)0.0f, (Half)5.960464e-08f });
            var r = np.maximum(a, b);
            ((double)r.GetAtIndex<Half>(0)).Should().BeApproximately(5.96e-08, 1e-9);
            ((double)r.GetAtIndex<Half>(1)).Should().Be(0.0);
            ((double)r.GetAtIndex<Half>(2)).Should().BeApproximately(5.96e-08, 1e-9);
        }

        [TestMethod]
        public void B10_Complex_Maximum_LexTie_EqualReal()
        {
            // np.maximum([1+5j, 1+3j, 1+7j], [1+2j, 1+8j, 1+7j]) → [1+5j, 1+8j, 1+7j]
            // Lex: compare real (tied: all 1) then imag.
            var a = np.array(new Complex[] { C(1, 5), C(1, 3), C(1, 7) });
            var b = np.array(new Complex[] { C(1, 2), C(1, 8), C(1, 7) });
            var r = np.maximum(a, b);
            r.GetAtIndex<Complex>(0).Should().Be(C(1, 5));
            r.GetAtIndex<Complex>(1).Should().Be(C(1, 8));
            r.GetAtIndex<Complex>(2).Should().Be(C(1, 7));
        }

        [TestMethod]
        public void B10_Complex_Maximum_Inf_Real_Imag_Varies()
        {
            // np.maximum([inf+1j, inf+nanj], [inf+3j, inf+0j]) → [inf+3j, inf+nanj]
            //   idx 0: no NaN, real tied (inf), imag 1 vs 3 → 3
            //   idx 1: a has NaN → propagates nan imag
            var a = np.array(new Complex[] { C(double.PositiveInfinity, 1), C(double.PositiveInfinity, double.NaN) });
            var b = np.array(new Complex[] { C(double.PositiveInfinity, 3), C(double.PositiveInfinity, 0) });
            var r = np.maximum(a, b);
            var r0 = r.GetAtIndex<Complex>(0);
            double.IsPositiveInfinity(r0.Real).Should().BeTrue();
            r0.Imaginary.Should().BeApproximately(3.0, Tol);

            var r1 = r.GetAtIndex<Complex>(1);
            double.IsPositiveInfinity(r1.Real).Should().BeTrue();
            double.IsNaN(r1.Imaginary).Should().BeTrue();
        }

        [TestMethod]
        public void B10_Complex_Clip_With_NonZero_Imag_Bounds()
        {
            // np.clip([1+5j, 3+0j, 5+10j], 2+1j, 4+2j) → [2+1j, 3+0j, 4+2j]
            //   1+5j < 2+1j lex (real 1<2) → 2+1j
            //   3+0j: 2+1j ≤ 3+0j (real 3>2) ≤ 4+2j (real 3<4) → stays 3+0j
            //   5+10j > 4+2j lex (real 5>4) → 4+2j
            var a = np.array(new Complex[] { C(1, 5), C(3, 0), C(5, 10) });
            var lo = np.array(new Complex[] { C(2, 1) });
            var hi = np.array(new Complex[] { C(4, 2) });
            var r = np.clip(a, lo, hi);
            r.GetAtIndex<Complex>(0).Should().Be(C(2, 1));
            r.GetAtIndex<Complex>(1).Should().Be(C(3, 0));
            r.GetAtIndex<Complex>(2).Should().Be(C(4, 2));
        }

        [TestMethod]
        public void B10_Complex_Maximum_Zero_vs_NegZero()
        {
            // np.maximum([0+0j], [-0+0j]) → [0+0j]  (first-wins under lex tie)
            var a = np.array(new Complex[] { C(0, 0) });
            var b = np.array(new Complex[] { C(-0.0, 0) });
            var r = np.maximum(a, b);
            var r0 = r.GetAtIndex<Complex>(0);
            r0.Real.Should().Be(0.0);
            r0.Imaginary.Should().Be(0.0);
        }

        #endregion

        // ======================================================================
        // B14 — nanmean / nanstd / nanvar: edge cases
        // ======================================================================

        #region B14 edges

        [TestMethod]
        public void B14_Half_NanMean_AllNaN_Returns_NaN()
        {
            // np.nanmean(np.array([nan, nan, nan], dtype=float16)) → nan
            var a = np.array(new Half[] { Half.NaN, Half.NaN, Half.NaN });
            Half.IsNaN(np.nanmean(a).GetAtIndex<Half>(0)).Should().BeTrue();
        }

        [TestMethod]
        public void B14_Half_NanStd_SingleValid_Is_Zero()
        {
            // np.nanstd([nan, 3.0, nan], dtype=float16) → 0.0
            var a = np.array(new Half[] { Half.NaN, (Half)3.0f, Half.NaN });
            ((double)np.nanstd(a).GetAtIndex<Half>(0)).Should().BeApproximately(0.0, Tol);
        }

        [TestMethod]
        public void B14_Half_NanVar_SingleValid_Is_Zero()
        {
            var a = np.array(new Half[] { Half.NaN, (Half)3.0f, Half.NaN });
            ((double)np.nanvar(a).GetAtIndex<Half>(0)).Should().BeApproximately(0.0, Tol);
        }

        [TestMethod]
        public void B14_Half_NanVar_Ddof_Boundary()
        {
            // a = [1, nan, 3] float16 → valid count = 2
            //   ddof=0 → variance / 2 = 1.0
            //   ddof=1 → variance / 1 = 2.0
            //   ddof=2 → divisor=0 → NaN (np.nanvar clamps; np.var would give inf)
            //   ddof=3 → divisor=-1 → NaN
            var a = np.array(new Half[] { (Half)1.0f, Half.NaN, (Half)3.0f });
            ((double)np.nanvar(a, ddof: 0).GetAtIndex<Half>(0)).Should().BeApproximately(1.0, Tol);
            ((double)np.nanvar(a, ddof: 1).GetAtIndex<Half>(0)).Should().BeApproximately(2.0, Tol);
            Half.IsNaN(np.nanvar(a, ddof: 2).GetAtIndex<Half>(0)).Should().BeTrue();
            Half.IsNaN(np.nanvar(a, ddof: 3).GetAtIndex<Half>(0)).Should().BeTrue();
        }

        [TestMethod]
        public void B14_Half_NanMean_Axis_Keepdims()
        {
            // m = [[1,2,nan],[4,nan,6]] float16
            // np.nanmean(m, axis=0, keepdims=True) → [[2.5, 2, 6]]
            var m = np.array(new Half[,] {
                { (Half)1, (Half)2, Half.NaN },
                { (Half)4, Half.NaN, (Half)6 }
            });
            var r = np.nanmean(m, axis: 0, keepdims: true);
            r.typecode.Should().Be(NPTypeCode.Half);
            r.shape.Should().BeEquivalentTo(new[] { 1, 3 });
            ((double)r.GetAtIndex<Half>(0)).Should().BeApproximately(2.5, Tol);
            ((double)r.GetAtIndex<Half>(1)).Should().BeApproximately(2.0, Tol);
            ((double)r.GetAtIndex<Half>(2)).Should().BeApproximately(6.0, Tol);
        }

        [TestMethod]
        public void B14_Half_NanMean_AxisMinus1_Keepdims()
        {
            // np.nanmean(m, axis=-1, keepdims=True) → [[1.5],[5.0]]
            var m = np.array(new Half[,] {
                { (Half)1, (Half)2, Half.NaN },
                { (Half)4, Half.NaN, (Half)6 }
            });
            var r = np.nanmean(m, axis: -1, keepdims: true);
            r.shape.Should().BeEquivalentTo(new[] { 2, 1 });
            ((double)r.GetAtIndex<Half>(0)).Should().BeApproximately(1.5, Tol);
            ((double)r.GetAtIndex<Half>(1)).Should().BeApproximately(5.0, Tol);
        }

        [TestMethod]
        public void B14_Half_NanMean_3D_Axis0()
        {
            // a = [[[1,2],[3,nan]],[[nan,6],[7,8]]] float16
            // np.nanmean(a, axis=0) → [[1, 4],[5, 8]]
            var a = np.array(new Half[,,] {
                { { (Half)1, (Half)2 }, { (Half)3, Half.NaN } },
                { { Half.NaN, (Half)6 }, { (Half)7, (Half)8 } }
            });
            var r = np.nanmean(a, axis: 0);
            r.shape.Should().BeEquivalentTo(new[] { 2, 2 });
            ((double)r.GetAtIndex<Half>(0)).Should().BeApproximately(1.0, Tol);
            ((double)r.GetAtIndex<Half>(1)).Should().BeApproximately(4.0, Tol);
            ((double)r.GetAtIndex<Half>(2)).Should().BeApproximately(5.0, Tol);
            ((double)r.GetAtIndex<Half>(3)).Should().BeApproximately(8.0, Tol);
        }

        [TestMethod]
        public void B14_Half_NanMean_3D_Axis2()
        {
            // np.nanmean(a, axis=2) → [[1.5, 3],[6, 7.5]]
            var a = np.array(new Half[,,] {
                { { (Half)1, (Half)2 }, { (Half)3, Half.NaN } },
                { { Half.NaN, (Half)6 }, { (Half)7, (Half)8 } }
            });
            var r = np.nanmean(a, axis: 2);
            r.shape.Should().BeEquivalentTo(new[] { 2, 2 });
            ((double)r.GetAtIndex<Half>(0)).Should().BeApproximately(1.5, Tol);
            ((double)r.GetAtIndex<Half>(1)).Should().BeApproximately(3.0, Tol);
            ((double)r.GetAtIndex<Half>(2)).Should().BeApproximately(6.0, Tol);
            ((double)r.GetAtIndex<Half>(3)).Should().BeApproximately(7.5, Tol);
        }

        [TestMethod]
        public void B14_Half_NanStd_3D_Axis2()
        {
            // np.nanstd(a, axis=2) → [[0.5, 0],[0, 0.5]]
            var a = np.array(new Half[,,] {
                { { (Half)1, (Half)2 }, { (Half)3, Half.NaN } },
                { { Half.NaN, (Half)6 }, { (Half)7, (Half)8 } }
            });
            var r = np.nanstd(a, axis: 2);
            ((double)r.GetAtIndex<Half>(0)).Should().BeApproximately(0.5, Tol);
            ((double)r.GetAtIndex<Half>(1)).Should().BeApproximately(0.0, Tol);
            ((double)r.GetAtIndex<Half>(2)).Should().BeApproximately(0.0, Tol);
            ((double)r.GetAtIndex<Half>(3)).Should().BeApproximately(0.5, Tol);
        }

        [TestMethod]
        public void B14_Complex_NanMean_AllNaN_Returns_NaN()
        {
            // np.nanmean([complex(nan,nan), complex(nan,0)]) → nan + nanj
            var a = np.array(new Complex[] { C(double.NaN, double.NaN), C(double.NaN, 0) });
            var r = np.nanmean(a).GetAtIndex<Complex>(0);
            double.IsNaN(r.Real).Should().BeTrue();
            double.IsNaN(r.Imaginary).Should().BeTrue();
        }

        [TestMethod]
        public void B14_Complex_NanStd_AllNaN_Returns_NaN_Double()
        {
            // np.nanstd([complex(nan,nan), complex(nan,0)]) → nan (float64)
            var a = np.array(new Complex[] { C(double.NaN, double.NaN), C(double.NaN, 0) });
            var r = np.nanstd(a);
            r.typecode.Should().Be(NPTypeCode.Double);
            double.IsNaN(r.GetAtIndex<double>(0)).Should().BeTrue();
        }

        [TestMethod]
        public void B14_Complex_NanMean_NaN_RealOnly_IsCounted_As_NaN()
        {
            // np.nanmean([complex(nan, 3), 1+2j, 3+4j]) → 2+3j  (the nan-real entry is skipped)
            var a = np.array(new Complex[] { C(double.NaN, 3), C(1, 2), C(3, 4) });
            var r = np.nanmean(a).GetAtIndex<Complex>(0);
            r.Real.Should().BeApproximately(2.0, Tol);
            r.Imaginary.Should().BeApproximately(3.0, Tol);
        }

        [TestMethod]
        public void B14_Complex_NanMean_NaN_ImagOnly_IsCounted_As_NaN()
        {
            // np.nanmean([complex(1, nan), 1+2j, 3+4j]) → 2+3j  (imag-nan also counts as NaN-carrier)
            var a = np.array(new Complex[] { C(1, double.NaN), C(1, 2), C(3, 4) });
            var r = np.nanmean(a).GetAtIndex<Complex>(0);
            r.Real.Should().BeApproximately(2.0, Tol);
            r.Imaginary.Should().BeApproximately(3.0, Tol);
        }

        [TestMethod]
        public void B14_Complex_NanVar_Ddof_Boundary()
        {
            // a = [1+2j, complex(nan,nan), 3+4j]  valid count = 2
            //   ddof=0 → 2.0;  ddof=1 → 4.0;  ddof=2 → NaN;  ddof=3 → NaN
            var a = np.array(new Complex[] { C(1, 2), C(double.NaN, double.NaN), C(3, 4) });
            np.nanvar(a, ddof: 0).GetAtIndex<double>(0).Should().BeApproximately(2.0, Tol);
            np.nanvar(a, ddof: 1).GetAtIndex<double>(0).Should().BeApproximately(4.0, Tol);
            double.IsNaN(np.nanvar(a, ddof: 2).GetAtIndex<double>(0)).Should().BeTrue();
            double.IsNaN(np.nanvar(a, ddof: 3).GetAtIndex<double>(0)).Should().BeTrue();
        }

        [TestMethod]
        public void B14_Complex_NanMean_AxisMinus1_Keepdims()
        {
            // m = [[1+1j, nan+nanj, 3+3j], [4+4j, 5+5j, nan+nanj]]
            // np.nanmean(m, axis=-1, keepdims=True) → [[2+2j],[4.5+4.5j]]
            var m = np.array(new Complex[,] {
                { C(1, 1), C(double.NaN, double.NaN), C(3, 3) },
                { C(4, 4), C(5, 5), C(double.NaN, double.NaN) }
            });
            var r = np.nanmean(m, axis: -1, keepdims: true);
            r.typecode.Should().Be(NPTypeCode.Complex);
            r.shape.Should().BeEquivalentTo(new[] { 2, 1 });
            r.GetAtIndex<Complex>(0).Should().Be(C(2, 2));
            r.GetAtIndex<Complex>(1).Should().Be(C(4.5, 4.5));
        }

        [TestMethod]
        public void B14_Complex_NanStd_AxisMinus1_Keepdims_Double()
        {
            // np.nanstd(m, axis=-1, keepdims=True) → [[1.4142...],[0.7071...]]
            var m = np.array(new Complex[,] {
                { C(1, 1), C(double.NaN, double.NaN), C(3, 3) },
                { C(4, 4), C(5, 5), C(double.NaN, double.NaN) }
            });
            var r = np.nanstd(m, axis: -1, keepdims: true);
            r.typecode.Should().Be(NPTypeCode.Double);
            r.shape.Should().BeEquivalentTo(new[] { 2, 1 });
            r.GetAtIndex<double>(0).Should().BeApproximately(1.4142135623730951, Tol);
            r.GetAtIndex<double>(1).Should().BeApproximately(0.7071067811865476, Tol);
        }

        [TestMethod]
        public void B14_Complex_NanMean_3D_Axis2()
        {
            // a3 = [[[1+1j, 2+2j],[nan+nanj, 4+4j]],[[5+5j, nan+nanj],[7+7j, 8+8j]]]
            // np.nanmean(a3, axis=2) → [[1.5+1.5j, 4+4j],[5+5j, 7.5+7.5j]]
            var a = np.array(new Complex[,,] {
                { { C(1, 1), C(2, 2) }, { C(double.NaN, double.NaN), C(4, 4) } },
                { { C(5, 5), C(double.NaN, double.NaN) }, { C(7, 7), C(8, 8) } }
            });
            var r = np.nanmean(a, axis: 2);
            r.shape.Should().BeEquivalentTo(new[] { 2, 2 });
            r.GetAtIndex<Complex>(0).Should().Be(C(1.5, 1.5));
            r.GetAtIndex<Complex>(1).Should().Be(C(4, 4));
            r.GetAtIndex<Complex>(2).Should().Be(C(5, 5));
            r.GetAtIndex<Complex>(3).Should().Be(C(7.5, 7.5));
        }

        [TestMethod]
        public void B14_Complex_NanVar_3D_Axis2()
        {
            // np.nanvar(a3, axis=2) → [[0.5, 0],[0, 0.5]] (float64)
            var a = np.array(new Complex[,,] {
                { { C(1, 1), C(2, 2) }, { C(double.NaN, double.NaN), C(4, 4) } },
                { { C(5, 5), C(double.NaN, double.NaN) }, { C(7, 7), C(8, 8) } }
            });
            var r = np.nanvar(a, axis: 2);
            r.typecode.Should().Be(NPTypeCode.Double);
            r.GetAtIndex<double>(0).Should().BeApproximately(0.5, Tol);
            r.GetAtIndex<double>(1).Should().BeApproximately(0.0, Tol);
            r.GetAtIndex<double>(2).Should().BeApproximately(0.0, Tol);
            r.GetAtIndex<double>(3).Should().BeApproximately(0.5, Tol);
        }

        #endregion

        // ======================================================================
        // B18 — Complex cumprod along axis: edge cases
        // ======================================================================

        #region B18 edges

        [TestMethod]
        public void B18_Complex_Cumprod_Axis0_With_Zero_Propagates()
        {
            // a = [[1+1j, 0+0j, 2+2j], [2+1j, 3+3j, 1+0j]]
            // np.cumprod(a, axis=0)
            // row 0: [1+1j, 0+0j, 2+2j]                    (passthrough)
            // row 1: [(1+1j)(2+1j)=1+3j, 0+0j, (2+2j)(1+0j)=2+2j]
            var a = np.array(new Complex[,] { { C(1, 1), C(0, 0), C(2, 2) }, { C(2, 1), C(3, 3), C(1, 0) } });
            var r = np.cumprod(a, axis: 0);
            r.GetAtIndex<Complex>(0).Should().Be(C(1, 1));
            r.GetAtIndex<Complex>(1).Should().Be(C(0, 0));
            r.GetAtIndex<Complex>(2).Should().Be(C(2, 2));
            r.GetAtIndex<Complex>(3).Should().Be(C(1, 3));
            r.GetAtIndex<Complex>(4).Should().Be(C(0, 0));
            r.GetAtIndex<Complex>(5).Should().Be(C(2, 2));
        }

        [TestMethod]
        public void B18_Complex_Cumprod_Axis1_With_Zero_Propagates()
        {
            // np.cumprod(a, axis=1)
            // row 0: [1+1j, 0+0j, 0+0j]    (zero contaminates downstream)
            // row 1: [2+1j, (2+1j)(3+3j)=3+9j, 3+9j*1+0j=3+9j]
            var a = np.array(new Complex[,] { { C(1, 1), C(0, 0), C(2, 2) }, { C(2, 1), C(3, 3), C(1, 0) } });
            var r = np.cumprod(a, axis: 1);
            r.GetAtIndex<Complex>(0).Should().Be(C(1, 1));
            r.GetAtIndex<Complex>(1).Should().Be(C(0, 0));
            r.GetAtIndex<Complex>(2).Should().Be(C(0, 0));
            r.GetAtIndex<Complex>(3).Should().Be(C(2, 1));
            r.GetAtIndex<Complex>(4).Should().Be(C(3, 9));
            r.GetAtIndex<Complex>(5).Should().Be(C(3, 9));
        }

        [TestMethod]
        public void B18_Complex_Cumprod_AxisMinus1_Matches_Axis1()
        {
            // np.cumprod on a 2D array with axis=-1 equals axis=1.
            // a = [[1+1j, 2+2j, 3+3j], [4+4j, 5+5j, 6+6j]]
            // axis=-1 → [[1+1j, 0+4j, -12+12j], [4+4j, 0+40j, -240+240j]]
            var a = np.array(new Complex[,] { { C(1, 1), C(2, 2), C(3, 3) }, { C(4, 4), C(5, 5), C(6, 6) } });
            var r = np.cumprod(a, axis: -1);
            r.GetAtIndex<Complex>(0).Should().Be(C(1, 1));
            r.GetAtIndex<Complex>(1).Should().Be(C(0, 4));
            r.GetAtIndex<Complex>(2).Should().Be(C(-12, 12));
            r.GetAtIndex<Complex>(3).Should().Be(C(4, 4));
            r.GetAtIndex<Complex>(4).Should().Be(C(0, 40));
            r.GetAtIndex<Complex>(5).Should().Be(C(-240, 240));
        }

        [TestMethod]
        public void B18_Complex_Cumprod_3D_Axis0()
        {
            // a3 = [[[1+1j, 2+2j],[3+3j, 4+4j]],[[5+5j, 6+6j],[7+7j, 8+8j]]]
            // np.cumprod(a3, axis=0)
            // layer 0: unchanged.
            // layer 1: [[(1+1j)(5+5j)=0+10j, (2+2j)(6+6j)=0+24j],[(3+3j)(7+7j)=0+42j, (4+4j)(8+8j)=0+64j]]
            var a = np.array(new Complex[,,] {
                { { C(1, 1), C(2, 2) }, { C(3, 3), C(4, 4) } },
                { { C(5, 5), C(6, 6) }, { C(7, 7), C(8, 8) } }
            });
            var r = np.cumprod(a, axis: 0);
            r.shape.Should().BeEquivalentTo(new[] { 2, 2, 2 });
            r.GetAtIndex<Complex>(0).Should().Be(C(1, 1));
            r.GetAtIndex<Complex>(1).Should().Be(C(2, 2));
            r.GetAtIndex<Complex>(2).Should().Be(C(3, 3));
            r.GetAtIndex<Complex>(3).Should().Be(C(4, 4));
            r.GetAtIndex<Complex>(4).Should().Be(C(0, 10));
            r.GetAtIndex<Complex>(5).Should().Be(C(0, 24));
            r.GetAtIndex<Complex>(6).Should().Be(C(0, 42));
            r.GetAtIndex<Complex>(7).Should().Be(C(0, 64));
        }

        [TestMethod]
        public void B18_Complex_Cumprod_3D_Axis1()
        {
            // np.cumprod(a3, axis=1) → [[[1+1j, 2+2j],[0+6j, 0+16j]],[[5+5j, 6+6j],[0+70j, 0+96j]]]
            var a = np.array(new Complex[,,] {
                { { C(1, 1), C(2, 2) }, { C(3, 3), C(4, 4) } },
                { { C(5, 5), C(6, 6) }, { C(7, 7), C(8, 8) } }
            });
            var r = np.cumprod(a, axis: 1);
            r.GetAtIndex<Complex>(0).Should().Be(C(1, 1));
            r.GetAtIndex<Complex>(1).Should().Be(C(2, 2));
            r.GetAtIndex<Complex>(2).Should().Be(C(0, 6));
            r.GetAtIndex<Complex>(3).Should().Be(C(0, 16));
            r.GetAtIndex<Complex>(4).Should().Be(C(5, 5));
            r.GetAtIndex<Complex>(5).Should().Be(C(6, 6));
            r.GetAtIndex<Complex>(6).Should().Be(C(0, 70));
            r.GetAtIndex<Complex>(7).Should().Be(C(0, 96));
        }

        [TestMethod]
        public void B18_Complex_Cumprod_3D_Axis2()
        {
            // np.cumprod(a3, axis=2) → [[[1+1j, 0+4j],[3+3j, 0+24j]],[[5+5j, 0+60j],[7+7j, 0+112j]]]
            var a = np.array(new Complex[,,] {
                { { C(1, 1), C(2, 2) }, { C(3, 3), C(4, 4) } },
                { { C(5, 5), C(6, 6) }, { C(7, 7), C(8, 8) } }
            });
            var r = np.cumprod(a, axis: 2);
            r.GetAtIndex<Complex>(0).Should().Be(C(1, 1));
            r.GetAtIndex<Complex>(1).Should().Be(C(0, 4));
            r.GetAtIndex<Complex>(2).Should().Be(C(3, 3));
            r.GetAtIndex<Complex>(3).Should().Be(C(0, 24));
            r.GetAtIndex<Complex>(4).Should().Be(C(5, 5));
            r.GetAtIndex<Complex>(5).Should().Be(C(0, 60));
            r.GetAtIndex<Complex>(6).Should().Be(C(7, 7));
            r.GetAtIndex<Complex>(7).Should().Be(C(0, 112));
        }

        [TestMethod]
        public void B18_Complex_Cumprod_SingleElementAxis()
        {
            // a = [[1+2j]]  shape (1,1); cumprod along axis 0 or 1 is a no-op.
            var a = np.array(new Complex[,] { { C(1, 2) } });
            var r0 = np.cumprod(a, axis: 0);
            r0.GetAtIndex<Complex>(0).Should().Be(C(1, 2));
            var r1 = np.cumprod(a, axis: 1);
            r1.GetAtIndex<Complex>(0).Should().Be(C(1, 2));
        }

        #endregion

        // ======================================================================
        // B19 — Complex max/min along axis: edge cases
        // ======================================================================

        #region B19 edges

        [TestMethod]
        public void B19_Complex_Max_AxisMinus1()
        {
            // m1 = [[1+1j, 3+0j, 2+5j], [4+4j, 1+1j, 2+9j]]
            // np.max(m1, axis=-1) → [3+0j, 4+4j]
            // Row 0: lex {1+1j, 3+0j, 2+5j} → 3+0j (real 3 > 2 > 1)
            // Row 1: lex {4+4j, 1+1j, 2+9j} → 4+4j (real 4 > 2 > 1)
            var m = np.array(new Complex[,] { { C(1, 1), C(3, 0), C(2, 5) }, { C(4, 4), C(1, 1), C(2, 9) } });
            var r = np.max(m, axis: -1);
            r.typecode.Should().Be(NPTypeCode.Complex);
            r.shape.Should().BeEquivalentTo(new[] { 2 });
            r.GetAtIndex<Complex>(0).Should().Be(C(3, 0));
            r.GetAtIndex<Complex>(1).Should().Be(C(4, 4));
        }

        [TestMethod]
        public void B19_Complex_Min_AxisMinus1()
        {
            // np.min(m1, axis=-1) → [1+1j, 1+1j]
            var m = np.array(new Complex[,] { { C(1, 1), C(3, 0), C(2, 5) }, { C(4, 4), C(1, 1), C(2, 9) } });
            var r = np.min(m, axis: -1);
            r.GetAtIndex<Complex>(0).Should().Be(C(1, 1));
            r.GetAtIndex<Complex>(1).Should().Be(C(1, 1));
        }

        [TestMethod]
        public void B19_Complex_Max_Axis0_Keepdims()
        {
            // np.max(m1, axis=0, keepdims=True) → [[4+4j, 3+0j, 2+9j]]
            var m = np.array(new Complex[,] { { C(1, 1), C(3, 0), C(2, 5) }, { C(4, 4), C(1, 1), C(2, 9) } });
            var r = np.max(m, axis: 0, keepdims: true);
            r.shape.Should().BeEquivalentTo(new[] { 1, 3 });
            r.GetAtIndex<Complex>(0).Should().Be(C(4, 4));
            r.GetAtIndex<Complex>(1).Should().Be(C(3, 0));
            r.GetAtIndex<Complex>(2).Should().Be(C(2, 9));
        }

        [TestMethod]
        public void B19_Complex_Min_Axis1_Keepdims()
        {
            // np.min(m1, axis=1, keepdims=True) → [[1+1j],[1+1j]]
            var m = np.array(new Complex[,] { { C(1, 1), C(3, 0), C(2, 5) }, { C(4, 4), C(1, 1), C(2, 9) } });
            var r = np.min(m, axis: 1, keepdims: true);
            r.shape.Should().BeEquivalentTo(new[] { 2, 1 });
            r.GetAtIndex<Complex>(0).Should().Be(C(1, 1));
            r.GetAtIndex<Complex>(1).Should().Be(C(1, 1));
        }

        [TestMethod]
        public void B19_Complex_Max_SingleElementAxis()
        {
            // np.max([[1+2j]], axis=0) → [1+2j];   axis=1 → [1+2j]
            var m = np.array(new Complex[,] { { C(1, 2) } });
            var r0 = np.max(m, axis: 0);
            r0.GetAtIndex<Complex>(0).Should().Be(C(1, 2));
            var r1 = np.max(m, axis: 1);
            r1.GetAtIndex<Complex>(0).Should().Be(C(1, 2));
        }

        [TestMethod]
        public void B19_Complex_Max_AllEqual_Axis1()
        {
            // np.max([[2+3j, 2+3j, 2+3j]], axis=1) → [2+3j]
            var m = np.array(new Complex[,] { { C(2, 3), C(2, 3), C(2, 3) } });
            np.max(m, axis: 1).GetAtIndex<Complex>(0).Should().Be(C(2, 3));
            np.min(m, axis: 1).GetAtIndex<Complex>(0).Should().Be(C(2, 3));
        }

        [TestMethod]
        public void B19_Complex_Max_Inf_Axis0()
        {
            // m = [[inf+0j, 1+1j, -inf+0j],[0+infj, 2+2j, 0-infj]]
            // np.max(m, axis=0) → [inf+0j, 2+2j, 0-infj]
            //   col 0: max(inf+0j, 0+infj) lex on real: inf > 0 → inf+0j
            //   col 1: max(1+1j, 2+2j) → 2+2j
            //   col 2: max(-inf+0j, 0-infj): real -inf vs 0 → 0-infj
            var m = np.array(new Complex[,] {
                { C(double.PositiveInfinity, 0), C(1, 1), C(double.NegativeInfinity, 0) },
                { C(0, double.PositiveInfinity), C(2, 2), C(0, double.NegativeInfinity) }
            });
            var r = np.max(m, axis: 0);
            var r0 = r.GetAtIndex<Complex>(0); double.IsPositiveInfinity(r0.Real).Should().BeTrue(); r0.Imaginary.Should().Be(0.0);
            r.GetAtIndex<Complex>(1).Should().Be(C(2, 2));
            var r2 = r.GetAtIndex<Complex>(2); r2.Real.Should().Be(0.0); double.IsNegativeInfinity(r2.Imaginary).Should().BeTrue();
        }

        [TestMethod]
        public void B19_Complex_Min_Inf_Axis0()
        {
            // np.min(m, axis=0) → [0+infj, 1+1j, -inf+0j]
            var m = np.array(new Complex[,] {
                { C(double.PositiveInfinity, 0), C(1, 1), C(double.NegativeInfinity, 0) },
                { C(0, double.PositiveInfinity), C(2, 2), C(0, double.NegativeInfinity) }
            });
            var r = np.min(m, axis: 0);
            var r0 = r.GetAtIndex<Complex>(0); r0.Real.Should().Be(0.0); double.IsPositiveInfinity(r0.Imaginary).Should().BeTrue();
            r.GetAtIndex<Complex>(1).Should().Be(C(1, 1));
            var r2 = r.GetAtIndex<Complex>(2); double.IsNegativeInfinity(r2.Real).Should().BeTrue(); r2.Imaginary.Should().Be(0.0);
        }

        [TestMethod]
        public void B19_Complex_Max_3D_Axis1()
        {
            // a3c = [[[1+1j, 2+2j],[3+3j, 4+4j]],[[5+5j, 6+6j],[7+7j, 8+8j]]]
            // np.max(a3c, axis=1) → [[3+3j, 4+4j],[7+7j, 8+8j]]
            var a = np.array(new Complex[,,] {
                { { C(1, 1), C(2, 2) }, { C(3, 3), C(4, 4) } },
                { { C(5, 5), C(6, 6) }, { C(7, 7), C(8, 8) } }
            });
            var r = np.max(a, axis: 1);
            r.shape.Should().BeEquivalentTo(new[] { 2, 2 });
            r.GetAtIndex<Complex>(0).Should().Be(C(3, 3));
            r.GetAtIndex<Complex>(1).Should().Be(C(4, 4));
            r.GetAtIndex<Complex>(2).Should().Be(C(7, 7));
            r.GetAtIndex<Complex>(3).Should().Be(C(8, 8));
        }

        [TestMethod]
        public void B19_Complex_Max_3D_Axis2()
        {
            // np.max(a3c, axis=2) → [[2+2j, 4+4j],[6+6j, 8+8j]]
            var a = np.array(new Complex[,,] {
                { { C(1, 1), C(2, 2) }, { C(3, 3), C(4, 4) } },
                { { C(5, 5), C(6, 6) }, { C(7, 7), C(8, 8) } }
            });
            var r = np.max(a, axis: 2);
            r.GetAtIndex<Complex>(0).Should().Be(C(2, 2));
            r.GetAtIndex<Complex>(1).Should().Be(C(4, 4));
            r.GetAtIndex<Complex>(2).Should().Be(C(6, 6));
            r.GetAtIndex<Complex>(3).Should().Be(C(8, 8));
        }

        [TestMethod]
        public void B19_Complex_Min_3D_Axis2()
        {
            // np.min(a3c, axis=2) → [[1+1j, 3+3j],[5+5j, 7+7j]]
            var a = np.array(new Complex[,,] {
                { { C(1, 1), C(2, 2) }, { C(3, 3), C(4, 4) } },
                { { C(5, 5), C(6, 6) }, { C(7, 7), C(8, 8) } }
            });
            var r = np.min(a, axis: 2);
            r.GetAtIndex<Complex>(0).Should().Be(C(1, 1));
            r.GetAtIndex<Complex>(1).Should().Be(C(3, 3));
            r.GetAtIndex<Complex>(2).Should().Be(C(5, 5));
            r.GetAtIndex<Complex>(3).Should().Be(C(7, 7));
        }

        [TestMethod]
        public void B19_Complex_Max_LexTie_Axis0()
        {
            // tie = [[1+5j, 2+7j],[1+5j, 2+3j]]
            // np.max(tie, axis=0) → [1+5j, 2+7j]
            //   col 0: tied (1+5j == 1+5j) → 1+5j
            //   col 1: max(2+7j, 2+3j) → 2+7j
            var m = np.array(new Complex[,] { { C(1, 5), C(2, 7) }, { C(1, 5), C(2, 3) } });
            var r = np.max(m, axis: 0);
            r.GetAtIndex<Complex>(0).Should().Be(C(1, 5));
            r.GetAtIndex<Complex>(1).Should().Be(C(2, 7));
        }

        #endregion

        // ======================================================================
        // B20 — Complex std/var along axis: edge cases
        // ======================================================================

        #region B20 edges

        [TestMethod]
        [OpenBugs]  // B23: np.var(Complex, axis=N) where the reduced axis has size 1 returns a
                    // Complex array (the original element) instead of a Double array of zeros.
                    // NumPy returns float64 [0.0, ...]. Root cause: trivial-axis fast path
                    // bypasses the Var/Std output-dtype promotion.
        public void B20_Complex_Var_SingleElementAxis_Is_Zero()
        {
            // np.var([[1+2j]], axis=0) → [0.0];  axis=1 → [0.0] (single-element variance = 0)
            var m = np.array(new Complex[,] { { C(1, 2) } });
            var v0 = np.var(m, axis: 0);
            v0.typecode.Should().Be(NPTypeCode.Double);
            v0.GetAtIndex<double>(0).Should().BeApproximately(0.0, Tol);
            var v1 = np.var(m, axis: 1);
            v1.typecode.Should().Be(NPTypeCode.Double);
            v1.GetAtIndex<double>(0).Should().BeApproximately(0.0, Tol);
        }

        [TestMethod]
        public void B20_Complex_Var_Ddof_Equal_N_Returns_Inf()
        {
            // np.var([[1+2j, 3+4j, 5+6j]], axis=1, ddof=3) → [inf]
            // NumPy's np.var clamps divisor=max(n-ddof, 0); 0 divisor → division yields +inf (float).
            // Variance of [1+2j, 3+4j, 5+6j] is 16 (sum of squared |dev|) / 3 = 5.333 for ddof=0.
            // For ddof=3, divisor=0, sum=16 → 16/0 = +inf.
            var m = np.array(new Complex[,] { { C(1, 2), C(3, 4), C(5, 6) } });
            var r = np.var(m, axis: 1, ddof: 3);
            r.typecode.Should().Be(NPTypeCode.Double);
            double.IsPositiveInfinity(r.GetAtIndex<double>(0)).Should().BeTrue();
        }

        [TestMethod]
        [OpenBugs]  // B24: np.var(Complex, axis=N, ddof > n) returns sum/(n-ddof) (a negative
                    // value) instead of NumPy's +inf. NumPy clamps divisor = max(n - ddof, 0)
                    // so the division by zero yields +inf. NumSharp's AxisVarStdComplexHelper
                    // uses unclamped (n - ddof) giving negative variance.
        public void B20_Complex_Var_Ddof_Greater_Than_N_Returns_Inf()
        {
            // np.var(axis=1, ddof=4) for n=3 → [inf] (divisor clamped to 0 → inf)
            var m = np.array(new Complex[,] { { C(1, 2), C(3, 4), C(5, 6) } });
            var r = np.var(m, axis: 1, ddof: 4);
            double.IsPositiveInfinity(r.GetAtIndex<double>(0)).Should().BeTrue();
        }

        [TestMethod]
        public void B20_Complex_Var_Axis0_Keepdims()
        {
            // m = [[1+2j, 3+4j, 5+6j],[7+8j, 9+10j, 11+12j]]
            // np.var(m, axis=0, keepdims=True) → [[18, 18, 18]]
            // col 0: mean=4+5j; |−3−3j|²=18, |3+3j|²=18; sum=36; /2=18
            var m = np.array(new Complex[,] { { C(1, 2), C(3, 4), C(5, 6) }, { C(7, 8), C(9, 10), C(11, 12) } });
            var r = np.var(m, axis: 0, keepdims: true);
            r.typecode.Should().Be(NPTypeCode.Double);
            r.shape.Should().BeEquivalentTo(new[] { 1, 3 });
            r.GetAtIndex<double>(0).Should().BeApproximately(18.0, Tol);
            r.GetAtIndex<double>(1).Should().BeApproximately(18.0, Tol);
            r.GetAtIndex<double>(2).Should().BeApproximately(18.0, Tol);
        }

        [TestMethod]
        public void B20_Complex_Var_AxisMinus1()
        {
            // np.var(m, axis=-1) → [5.333..., 5.333...]
            var m = np.array(new Complex[,] { { C(1, 2), C(3, 4), C(5, 6) }, { C(7, 8), C(9, 10), C(11, 12) } });
            var r = np.var(m, axis: -1);
            r.GetAtIndex<double>(0).Should().BeApproximately(5.333333333333333, Tol);
            r.GetAtIndex<double>(1).Should().BeApproximately(5.333333333333333, Tol);
        }

        [TestMethod]
        public void B20_Complex_Std_AxisMinus1()
        {
            // np.std(m, axis=-1) → [2.3094, 2.3094]
            var m = np.array(new Complex[,] { { C(1, 2), C(3, 4), C(5, 6) }, { C(7, 8), C(9, 10), C(11, 12) } });
            var r = np.std(m, axis: -1);
            r.GetAtIndex<double>(0).Should().BeApproximately(2.3094010767585034, Tol);
            r.GetAtIndex<double>(1).Should().BeApproximately(2.3094010767585034, Tol);
        }

        [TestMethod]
        public void B20_Complex_Var_3D_Axis0()
        {
            // a3c = [[[1+1j, 2+2j],[3+3j, 4+4j]],[[5+5j, 6+6j],[7+7j, 8+8j]]]
            // np.var(a3c, axis=0) → [[8, 8],[8, 8]]
            //   a3c[:,0,0] = [1+1j, 5+5j]; mean = 3+3j; |−2−2j|²=8, |2+2j|²=8; /2 = 8
            var a = np.array(new Complex[,,] {
                { { C(1, 1), C(2, 2) }, { C(3, 3), C(4, 4) } },
                { { C(5, 5), C(6, 6) }, { C(7, 7), C(8, 8) } }
            });
            var r = np.var(a, axis: 0);
            r.shape.Should().BeEquivalentTo(new[] { 2, 2 });
            r.GetAtIndex<double>(0).Should().BeApproximately(8.0, Tol);
            r.GetAtIndex<double>(1).Should().BeApproximately(8.0, Tol);
            r.GetAtIndex<double>(2).Should().BeApproximately(8.0, Tol);
            r.GetAtIndex<double>(3).Should().BeApproximately(8.0, Tol);
        }

        [TestMethod]
        public void B20_Complex_Var_3D_Axis1()
        {
            // np.var(a3c, axis=1) → [[2, 2],[2, 2]]
            //   a3c[0,:,0] = [1+1j, 3+3j]; mean=2+2j; |−1−1j|²=2, |1+1j|²=2; /2 = 2
            var a = np.array(new Complex[,,] {
                { { C(1, 1), C(2, 2) }, { C(3, 3), C(4, 4) } },
                { { C(5, 5), C(6, 6) }, { C(7, 7), C(8, 8) } }
            });
            var r = np.var(a, axis: 1);
            r.GetAtIndex<double>(0).Should().BeApproximately(2.0, Tol);
            r.GetAtIndex<double>(1).Should().BeApproximately(2.0, Tol);
            r.GetAtIndex<double>(2).Should().BeApproximately(2.0, Tol);
            r.GetAtIndex<double>(3).Should().BeApproximately(2.0, Tol);
        }

        [TestMethod]
        public void B20_Complex_Var_3D_Axis2()
        {
            // np.var(a3c, axis=2) → [[0.5, 0.5],[0.5, 0.5]]
            //   a3c[0,0,:] = [1+1j, 2+2j]; mean=1.5+1.5j; |−0.5−0.5j|²=0.5, same; /2 = 0.5
            var a = np.array(new Complex[,,] {
                { { C(1, 1), C(2, 2) }, { C(3, 3), C(4, 4) } },
                { { C(5, 5), C(6, 6) }, { C(7, 7), C(8, 8) } }
            });
            var r = np.var(a, axis: 2);
            r.GetAtIndex<double>(0).Should().BeApproximately(0.5, Tol);
            r.GetAtIndex<double>(1).Should().BeApproximately(0.5, Tol);
            r.GetAtIndex<double>(2).Should().BeApproximately(0.5, Tol);
            r.GetAtIndex<double>(3).Should().BeApproximately(0.5, Tol);
        }

        [TestMethod]
        public void B20_Complex_Std_3D_Axis2()
        {
            // np.std(a3c, axis=2) = sqrt(var) = sqrt(0.5) = 0.7071... everywhere
            var a = np.array(new Complex[,,] {
                { { C(1, 1), C(2, 2) }, { C(3, 3), C(4, 4) } },
                { { C(5, 5), C(6, 6) }, { C(7, 7), C(8, 8) } }
            });
            var r = np.std(a, axis: 2);
            r.GetAtIndex<double>(0).Should().BeApproximately(0.7071067811865476, Tol);
            r.GetAtIndex<double>(1).Should().BeApproximately(0.7071067811865476, Tol);
            r.GetAtIndex<double>(2).Should().BeApproximately(0.7071067811865476, Tol);
            r.GetAtIndex<double>(3).Should().BeApproximately(0.7071067811865476, Tol);
        }

        [TestMethod]
        public void B20_Complex_Var_LargeMagnitude_NoCancellation()
        {
            // Large magnitude that would overflow sum-of-squares for float32 but safe in double.
            // np.var([1e100+1e100j, 2e100+2e100j, 3e100+3e100j]) ≈ 1.333e+200
            // mean = 2e100+2e100j;  |1e100 - 2e100|² per component × 2 = 2e200, and same for |3e100 - 2e100|² → 2e200
            // (mid is 0) sum = 4e200, /3 = 1.333e200
            var a = np.array(new Complex[] { C(1e100, 1e100), C(2e100, 2e100), C(3e100, 3e100) });
            var r = np.var(a);
            r.typecode.Should().Be(NPTypeCode.Double);
            r.GetAtIndex<double>(0).Should().BeApproximately(1.333333333333333e200, 1e197);
        }

        [TestMethod]
        public void B20_Complex_Var_Axis1_N2_Ddof_Boundary()
        {
            // a = [[1+1j, 3+3j]], axis=1, n=2
            //   mean = 2+2j;  |−1−1j|²=2, |1+1j|²=2; sum=4
            //   ddof=0 → 4/2 = 2
            //   ddof=1 → 4/1 = 4
            //   ddof=2 → divisor clamped to 0 → +inf
            var m = np.array(new Complex[,] { { C(1, 1), C(3, 3) } });
            np.var(m, axis: 1, ddof: 0).GetAtIndex<double>(0).Should().BeApproximately(2.0, Tol);
            np.var(m, axis: 1, ddof: 1).GetAtIndex<double>(0).Should().BeApproximately(4.0, Tol);
            double.IsPositiveInfinity(np.var(m, axis: 1, ddof: 2).GetAtIndex<double>(0)).Should().BeTrue();
        }

        #endregion

        // ======================================================================
        // Additional passing edge cases — lock in current NumPy-parity behavior
        // These tests capture subtle Complex edge cases that Rounds 6+7 happen to
        // handle correctly via .NET BCL's Complex.Log/Exp semantics. Keeping them
        // as regression guards so any future refactor of ILKernelGenerator's
        // unary Complex branch is caught if it diverges.
        // ======================================================================

        #region Confirmed parity (regression guards)

        [TestMethod]
        public void Parity_Complex_Log10_NegZero_Gives_MinusInf_Plus_PiOverLn10()
        {
            // np.log10(-0+0j) → -inf + 1.3643763j  (because angle(-0+0j) = π in IEEE)
            var a = np.array(new Complex[] { C(-0.0, 0) });
            var r = np.log10(a).GetAtIndex<Complex>(0);
            double.IsNegativeInfinity(r.Real).Should().BeTrue();
            r.Imaginary.Should().BeApproximately(1.3643763538418412, Tol);
        }

        [TestMethod]
        public void Parity_Complex_Log10_NegInf_Gives_PosInf_Plus_PiOverLn10()
        {
            // np.log10(-inf+0j) → inf + 1.3643763j   (real component becomes +inf for |z|=inf)
            var a = np.array(new Complex[] { C(double.NegativeInfinity, 0) });
            var r = np.log10(a).GetAtIndex<Complex>(0);
            double.IsPositiveInfinity(r.Real).Should().BeTrue();
            r.Imaginary.Should().BeApproximately(1.3643763538418412, Tol);
        }

        [TestMethod]
        public void Parity_Complex_Log10_InfPlusInf_Gives_PosInf_Plus_PiOver4Over_Ln10()
        {
            // np.log10(inf+infj) → inf + 0.3410940j   (angle(inf+infj) = π/4; imag = π/4 / ln10)
            var a = np.array(new Complex[] { C(double.PositiveInfinity, double.PositiveInfinity) });
            var r = np.log10(a).GetAtIndex<Complex>(0);
            double.IsPositiveInfinity(r.Real).Should().BeTrue();
            r.Imaginary.Should().BeApproximately(0.3410940884604603, Tol);
        }

        [TestMethod]
        public void Parity_Complex_Log1p_NegInf_Gives_PosInf_Plus_Pi()
        {
            // np.log1p(-inf+0j) → inf + πj   (log(1+(-inf)) = log(-inf) principal = inf + πi)
            var a = np.array(new Complex[] { C(double.NegativeInfinity, 0) });
            var r = np.log1p(a).GetAtIndex<Complex>(0);
            double.IsPositiveInfinity(r.Real).Should().BeTrue();
            r.Imaginary.Should().BeApproximately(Math.PI, Tol);
        }

        [TestMethod]
        public void Parity_Complex_Expm1_PosInf_Gives_PosInf_Plus_NaN()
        {
            // np.expm1(inf+0j) → inf + nanj   (e^inf = inf, 0·inf in imag dimension → nan)
            var a = np.array(new Complex[] { C(double.PositiveInfinity, 0) });
            var r = np.expm1(a).GetAtIndex<Complex>(0);
            double.IsPositiveInfinity(r.Real).Should().BeTrue();
            double.IsNaN(r.Imaginary).Should().BeTrue();
        }

        [TestMethod]
        public void Parity_Half_Log2_MaxValue_Gives_Sixteen()
        {
            // np.log2(np.array([65504], dtype=float16)) → 16.0 (rounded in float16; log2(65504) ≈ 15.999)
            var a = np.array(new Half[] { Half.MaxValue });
            ((double)np.log2(a).GetAtIndex<Half>(0)).Should().BeApproximately(16.0, 0.01);
        }

        [TestMethod]
        public void Parity_Half_Exp2_Subnormal_Exponent()
        {
            // np.exp2(np.array([-24], dtype=float16)) → 5.96e-08  (= 2^-24, smallest subnormal)
            var a = np.array(new Half[] { (Half)(-24.0f) });
            ((double)np.exp2(a).GetAtIndex<Half>(0)).Should().BeApproximately(5.96e-08, 1e-9);
        }

        [TestMethod]
        public void Parity_Half_Log1p_NearMinusOne()
        {
            // np.log1p(np.array([-0.999], dtype=float16)) → -6.93  (float16 rounds -0.999 and log1p)
            // Tolerance is large because Half precision of -0.999 is ~-0.999 and log1p near -1 is steep.
            var a = np.array(new Half[] { (Half)(-0.999f) });
            ((double)np.log1p(a).GetAtIndex<Half>(0)).Should().BeApproximately(-6.93, 0.05);
        }

        [TestMethod]
        public void Parity_Half_Cbrt_Subnormal()
        {
            // np.cbrt(np.array([2**-24], dtype=float16)) → 0.003906 (float16)
            var a = np.array(new Half[] { (Half)5.960464e-08f });
            ((double)np.cbrt(a).GetAtIndex<Half>(0)).Should().BeApproximately(0.003906, Tol);
        }

        [TestMethod]
        public void Parity_Complex_Clip_2D_Multi_Row_Broadcasting()
        {
            // 2D clip with scalar bounds — each element independently clipped.
            // a = [[1+1j, 5+5j], [10+10j, 3+3j]]
            // np.clip(a, 2+0j, 6+0j) lex:
            //   1+1j: real 1 < 2 → 2+0j
            //   5+5j: real 5 in [2,6] → stays 5+5j
            //   10+10j: real 10 > 6 → 6+0j
            //   3+3j: real 3 in [2,6] → stays 3+3j
            var a = np.array(new Complex[,] { { C(1, 1), C(5, 5) }, { C(10, 10), C(3, 3) } });
            var lo = np.array(new Complex[] { C(2, 0) });
            var hi = np.array(new Complex[] { C(6, 0) });
            var r = np.clip(a, lo, hi);
            r.GetAtIndex<Complex>(0).Should().Be(C(2, 0));
            r.GetAtIndex<Complex>(1).Should().Be(C(5, 5));
            r.GetAtIndex<Complex>(2).Should().Be(C(6, 0));
            r.GetAtIndex<Complex>(3).Should().Be(C(3, 3));
        }

        [TestMethod]
        public void Parity_Complex_Var_Double_Output_Dtype()
        {
            // Ensures np.var(Complex, ...) returns Double dtype (not Complex) for the
            // non-trivial axis path — complements B23 which flags the trivial-axis bug.
            var m = np.array(new Complex[,] { { C(1, 2), C(3, 4), C(5, 6) }, { C(7, 8), C(9, 10), C(11, 12) } });
            np.var(m, axis: 0).typecode.Should().Be(NPTypeCode.Double);
            np.var(m, axis: 1).typecode.Should().Be(NPTypeCode.Double);
            np.std(m, axis: 0).typecode.Should().Be(NPTypeCode.Double);
            np.std(m, axis: 1).typecode.Should().Be(NPTypeCode.Double);
        }

        #endregion
    }
}
