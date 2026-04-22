using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;

namespace NumSharp.UnitTest.NewDtypes
{
    /// <summary>
    /// Battletests for Round 6 fixes (B10, B11, B14).
    /// All expected values verified against NumPy 2.4.2.
    ///
    /// Covers:
    ///   - B11: Half + Complex unary math (log10, log2, cbrt, exp2, log1p, expm1)
    ///   - B10/B17: Half + Complex maximum/minimum/clip (with NaN propagation and lex ordering)
    ///   - B14: Half + Complex nanmean/nanstd/nanvar (NaN-skipping)
    /// </summary>
    [TestClass]
    public class NewDtypesBattletestRound6Tests
    {
        private const double Tol = 1e-3;

        #region B11 — Half unary math

        [TestMethod]
        public void B11_Half_Log10()
        {
            // np.log10(np.array([0.5, 1.0, 2.0, 4.0, 10.0], dtype=np.float16))
            // → [-0.301, 0., 0.301, 0.602, 1.], dtype=float16
            var a = np.array(new Half[] { (Half)0.5, (Half)1.0, (Half)2.0, (Half)4.0, (Half)10.0 });
            var r = np.log10(a);
            r.typecode.Should().Be(NPTypeCode.Half);
            ((double)r.GetAtIndex<Half>(0)).Should().BeApproximately(-0.301, Tol);
            ((double)r.GetAtIndex<Half>(1)).Should().BeApproximately(0.0, Tol);
            ((double)r.GetAtIndex<Half>(2)).Should().BeApproximately(0.301, Tol);
            ((double)r.GetAtIndex<Half>(3)).Should().BeApproximately(0.602, Tol);
            ((double)r.GetAtIndex<Half>(4)).Should().BeApproximately(1.0, Tol);
        }

        [TestMethod]
        public void B11_Half_Log2()
        {
            // np.log2(np.array([0.5, 1.0, 2.0, 4.0, 10.0], dtype=np.float16))
            // → [-1., 0., 1., 2., 3.322], dtype=float16
            var a = np.array(new Half[] { (Half)0.5, (Half)1.0, (Half)2.0, (Half)4.0, (Half)10.0 });
            var r = np.log2(a);
            r.typecode.Should().Be(NPTypeCode.Half);
            ((double)r.GetAtIndex<Half>(0)).Should().BeApproximately(-1.0, Tol);
            ((double)r.GetAtIndex<Half>(1)).Should().BeApproximately(0.0, Tol);
            ((double)r.GetAtIndex<Half>(2)).Should().BeApproximately(1.0, Tol);
            ((double)r.GetAtIndex<Half>(3)).Should().BeApproximately(2.0, Tol);
            ((double)r.GetAtIndex<Half>(4)).Should().BeApproximately(3.322, Tol);
        }

        [TestMethod]
        public void B11_Half_Cbrt()
        {
            // np.cbrt(np.array([0.5, 1.0, 2.0, 4.0, 10.0], dtype=np.float16))
            // → [0.7935, 1., 1.26, 1.587, 2.154], dtype=float16
            var a = np.array(new Half[] { (Half)0.5, (Half)1.0, (Half)2.0, (Half)4.0, (Half)10.0 });
            var r = np.cbrt(a);
            r.typecode.Should().Be(NPTypeCode.Half);
            ((double)r.GetAtIndex<Half>(0)).Should().BeApproximately(0.7935, Tol);
            ((double)r.GetAtIndex<Half>(1)).Should().BeApproximately(1.0, Tol);
            ((double)r.GetAtIndex<Half>(2)).Should().BeApproximately(1.26, Tol);
            ((double)r.GetAtIndex<Half>(3)).Should().BeApproximately(1.587, Tol);
            ((double)r.GetAtIndex<Half>(4)).Should().BeApproximately(2.154, Tol);
        }

        [TestMethod]
        public void B11_Half_Exp2()
        {
            // np.exp2(np.array([0.5, 1.0, 2.0, 4.0, 10.0], dtype=np.float16))
            // → [1.414, 2., 4., 16., 1024.], dtype=float16
            var a = np.array(new Half[] { (Half)0.5, (Half)1.0, (Half)2.0, (Half)4.0, (Half)10.0 });
            var r = np.exp2(a);
            r.typecode.Should().Be(NPTypeCode.Half);
            ((double)r.GetAtIndex<Half>(0)).Should().BeApproximately(1.414, Tol);
            ((double)r.GetAtIndex<Half>(1)).Should().BeApproximately(2.0, Tol);
            ((double)r.GetAtIndex<Half>(2)).Should().BeApproximately(4.0, Tol);
            ((double)r.GetAtIndex<Half>(3)).Should().BeApproximately(16.0, Tol);
            ((double)r.GetAtIndex<Half>(4)).Should().BeApproximately(1024.0, 1.0);
        }

        [TestMethod]
        public void B11_Half_Log1p()
        {
            // np.log1p(np.array([0.5, 1.0, 2.0, 4.0, 10.0], dtype=np.float16))
            // → [0.4055, 0.6934, 1.099, 1.609, 2.398], dtype=float16
            var a = np.array(new Half[] { (Half)0.5, (Half)1.0, (Half)2.0, (Half)4.0, (Half)10.0 });
            var r = np.log1p(a);
            r.typecode.Should().Be(NPTypeCode.Half);
            ((double)r.GetAtIndex<Half>(0)).Should().BeApproximately(0.4055, Tol);
            ((double)r.GetAtIndex<Half>(1)).Should().BeApproximately(0.6934, Tol);
            ((double)r.GetAtIndex<Half>(2)).Should().BeApproximately(1.099, Tol);
            ((double)r.GetAtIndex<Half>(3)).Should().BeApproximately(1.609, Tol);
            ((double)r.GetAtIndex<Half>(4)).Should().BeApproximately(2.398, Tol);
        }

        [TestMethod]
        public void B11_Half_Expm1()
        {
            // np.expm1(np.array([0.5, 1.0, 2.0, 4.0], dtype=np.float16))
            // → [0.649, 1.719, 6.39, 53.6], dtype=float16
            var a = np.array(new Half[] { (Half)0.5, (Half)1.0, (Half)2.0, (Half)4.0 });
            var r = np.expm1(a);
            r.typecode.Should().Be(NPTypeCode.Half);
            ((double)r.GetAtIndex<Half>(0)).Should().BeApproximately(0.649, Tol);
            ((double)r.GetAtIndex<Half>(1)).Should().BeApproximately(1.719, Tol);
            ((double)r.GetAtIndex<Half>(2)).Should().BeApproximately(6.39, 0.01);
            ((double)r.GetAtIndex<Half>(3)).Should().BeApproximately(53.6, 0.1);
        }

        [TestMethod]
        public void B11_Half_Log_NaN_Propagates()
        {
            // np.log10/log2 on NaN → NaN in Half.
            var a = np.array(new Half[] { Half.NaN });
            Half.IsNaN(np.log10(a).GetAtIndex<Half>(0)).Should().BeTrue();
            Half.IsNaN(np.log2(a).GetAtIndex<Half>(0)).Should().BeTrue();
            Half.IsNaN(np.cbrt(a).GetAtIndex<Half>(0)).Should().BeTrue();
            Half.IsNaN(np.exp2(a).GetAtIndex<Half>(0)).Should().BeTrue();
            Half.IsNaN(np.log1p(a).GetAtIndex<Half>(0)).Should().BeTrue();
            Half.IsNaN(np.expm1(a).GetAtIndex<Half>(0)).Should().BeTrue();
        }

        #endregion

        #region B11 — Complex unary math

        [TestMethod]
        public void B11_Complex_Log10()
        {
            // np.log10(np.array([1+2j, 3+4j, -1+0j, 2+0j]))
            // → [0.349+0.481j, 0.699+0.403j, 0+1.364j, 0.301+0j], dtype=complex128
            var a = np.array(new Complex[] { new Complex(1, 2), new Complex(3, 4), new Complex(-1, 0), new Complex(2, 0) });
            var r = np.log10(a);
            r.typecode.Should().Be(NPTypeCode.Complex);
            var r0 = r.GetAtIndex<Complex>(0); r0.Real.Should().BeApproximately(0.34948500, Tol); r0.Imaginary.Should().BeApproximately(0.48082859, Tol);
            var r1 = r.GetAtIndex<Complex>(1); r1.Real.Should().BeApproximately(0.69897000, Tol); r1.Imaginary.Should().BeApproximately(0.40271920, Tol);
            var r2 = r.GetAtIndex<Complex>(2); r2.Real.Should().BeApproximately(0.0, Tol); r2.Imaginary.Should().BeApproximately(1.36437634, Tol);
            var r3 = r.GetAtIndex<Complex>(3); r3.Real.Should().BeApproximately(0.30103000, Tol); r3.Imaginary.Should().BeApproximately(0.0, Tol);
        }

        [TestMethod]
        public void B11_Complex_Log2()
        {
            // np.log2([1+2j, 3+4j, -1+0j, 0+0j])
            // → [1.161+1.597j, 2.322+1.338j, 0+4.532j, -inf+0j]
            var a = np.array(new Complex[] { new Complex(1, 2), new Complex(3, 4), new Complex(-1, 0), new Complex(0, 0) });
            var r = np.log2(a);
            r.typecode.Should().Be(NPTypeCode.Complex);
            var r0 = r.GetAtIndex<Complex>(0); r0.Real.Should().BeApproximately(1.16096405, Tol); r0.Imaginary.Should().BeApproximately(1.59727796, Tol);
            var r1 = r.GetAtIndex<Complex>(1); r1.Real.Should().BeApproximately(2.32192809, Tol); r1.Imaginary.Should().BeApproximately(1.33780421, Tol);
            var r2 = r.GetAtIndex<Complex>(2); r2.Real.Should().BeApproximately(0.0, Tol); r2.Imaginary.Should().BeApproximately(4.53236014, Tol);
            // Edge case: log2(0+0j) = -inf + 0j — critical, earlier bug produced -inf+NaNj via Complex.Log(z, 2).
            var r3 = r.GetAtIndex<Complex>(3);
            double.IsNegativeInfinity(r3.Real).Should().BeTrue();
            r3.Imaginary.Should().Be(0.0);
        }

        [TestMethod]
        public void B11_Complex_Exp2()
        {
            // np.exp2([1+2j, 3+4j, -1+0j, 0+0j, 2+0j])
            // → [0.367+1.966j, -7.461+2.885j, 0.5+0j, 1+0j, 4+0j]
            var a = np.array(new Complex[] { new Complex(1, 2), new Complex(3, 4), new Complex(-1, 0), new Complex(0, 0), new Complex(2, 0) });
            var r = np.exp2(a);
            r.typecode.Should().Be(NPTypeCode.Complex);
            var r0 = r.GetAtIndex<Complex>(0); r0.Real.Should().BeApproximately(0.36691395, Tol); r0.Imaginary.Should().BeApproximately(1.96605548, Tol);
            var r1 = r.GetAtIndex<Complex>(1); r1.Real.Should().BeApproximately(-7.46149661, Tol); r1.Imaginary.Should().BeApproximately(2.88549273, Tol);
            var r2 = r.GetAtIndex<Complex>(2); r2.Real.Should().BeApproximately(0.5, Tol); r2.Imaginary.Should().BeApproximately(0.0, Tol);
            var r3 = r.GetAtIndex<Complex>(3); r3.Real.Should().BeApproximately(1.0, Tol); r3.Imaginary.Should().BeApproximately(0.0, Tol);
            var r4 = r.GetAtIndex<Complex>(4); r4.Real.Should().BeApproximately(4.0, Tol); r4.Imaginary.Should().BeApproximately(0.0, Tol);
        }

        [TestMethod]
        public void B11_Complex_Log1p()
        {
            // np.log1p([1+2j, 3+4j, 2+0j, 0+0j])
            // → [1.040+0.785j, 1.733+0.785j, 1.099+0j, 0+0j]
            var a = np.array(new Complex[] { new Complex(1, 2), new Complex(3, 4), new Complex(2, 0), new Complex(0, 0) });
            var r = np.log1p(a);
            r.typecode.Should().Be(NPTypeCode.Complex);
            var r0 = r.GetAtIndex<Complex>(0); r0.Real.Should().BeApproximately(1.03972077, Tol); r0.Imaginary.Should().BeApproximately(0.78539816, Tol);
            var r1 = r.GetAtIndex<Complex>(1); r1.Real.Should().BeApproximately(1.73286795, Tol); r1.Imaginary.Should().BeApproximately(0.78539816, Tol);
            var r2 = r.GetAtIndex<Complex>(2); r2.Real.Should().BeApproximately(1.09861229, Tol); r2.Imaginary.Should().BeApproximately(0.0, Tol);
            var r3 = r.GetAtIndex<Complex>(3); r3.Real.Should().BeApproximately(0.0, Tol); r3.Imaginary.Should().BeApproximately(0.0, Tol);
        }

        [TestMethod]
        public void B11_Complex_Expm1()
        {
            // np.expm1([1+2j, 3+4j, -1+0j, 0+0j, 2+0j])
            // → [-2.131+2.472j, -14.129-15.201j, -0.632+0j, 0+0j, 6.389+0j]
            var a = np.array(new Complex[] { new Complex(1, 2), new Complex(3, 4), new Complex(-1, 0), new Complex(0, 0), new Complex(2, 0) });
            var r = np.expm1(a);
            r.typecode.Should().Be(NPTypeCode.Complex);
            var r0 = r.GetAtIndex<Complex>(0); r0.Real.Should().BeApproximately(-2.13120438, Tol); r0.Imaginary.Should().BeApproximately(2.47172667, Tol);
            var r1 = r.GetAtIndex<Complex>(1); r1.Real.Should().BeApproximately(-14.12878308, Tol); r1.Imaginary.Should().BeApproximately(-15.20078446, Tol);
            var r2 = r.GetAtIndex<Complex>(2); r2.Real.Should().BeApproximately(-0.63212056, Tol); r2.Imaginary.Should().BeApproximately(0.0, Tol);
            var r3 = r.GetAtIndex<Complex>(3); r3.Real.Should().BeApproximately(0.0, Tol); r3.Imaginary.Should().BeApproximately(0.0, Tol);
            var r4 = r.GetAtIndex<Complex>(4); r4.Real.Should().BeApproximately(6.38905610, Tol); r4.Imaginary.Should().BeApproximately(0.0, Tol);
        }

        [TestMethod]
        public void B11_Complex_Cbrt_NotSupported()
        {
            // NumPy does NOT support np.cbrt(complex) — it raises TypeError.
            // NumSharp should match by throwing NotSupportedException.
            var a = np.array(new Complex[] { new Complex(1, 2) });
            Action act = () => np.cbrt(a);
            act.Should().Throw<NotSupportedException>();
        }

        #endregion

        #region B10 — Half maximum/minimum (binary) + clip

        [TestMethod]
        public void B10_Half_Maximum_NaN_Propagates()
        {
            // np.maximum(np.array([1, nan, 3], dtype=float16), np.array([2, 2, nan], dtype=float16))
            // → [2, nan, nan]  (NaN wins)
            var a = np.array(new Half[] { (Half)1, Half.NaN, (Half)3 });
            var b = np.array(new Half[] { (Half)2, (Half)2, Half.NaN });
            var r = np.maximum(a, b);
            r.typecode.Should().Be(NPTypeCode.Half);
            r.GetAtIndex<Half>(0).Should().Be((Half)2);
            Half.IsNaN(r.GetAtIndex<Half>(1)).Should().BeTrue();
            Half.IsNaN(r.GetAtIndex<Half>(2)).Should().BeTrue();
        }

        [TestMethod]
        public void B10_Half_Minimum_NaN_Propagates()
        {
            // np.minimum: [1, nan, nan]
            var a = np.array(new Half[] { (Half)1, Half.NaN, (Half)3 });
            var b = np.array(new Half[] { (Half)2, (Half)2, Half.NaN });
            var r = np.minimum(a, b);
            r.typecode.Should().Be(NPTypeCode.Half);
            r.GetAtIndex<Half>(0).Should().Be((Half)1);
            Half.IsNaN(r.GetAtIndex<Half>(1)).Should().BeTrue();
            Half.IsNaN(r.GetAtIndex<Half>(2)).Should().BeTrue();
        }

        [TestMethod]
        public void B10_Half_Clip()
        {
            // np.clip(np.array([1, 5, 10, 15], dtype=float16), 3, 10) → [3, 5, 10, 10]
            var a = np.array(new Half[] { (Half)1, (Half)5, (Half)10, (Half)15 });
            var lo = np.array(new Half[] { (Half)3 });
            var hi = np.array(new Half[] { (Half)10 });
            var r = np.clip(a, lo, hi);
            r.typecode.Should().Be(NPTypeCode.Half);
            r.GetAtIndex<Half>(0).Should().Be((Half)3);
            r.GetAtIndex<Half>(1).Should().Be((Half)5);
            r.GetAtIndex<Half>(2).Should().Be((Half)10);
            r.GetAtIndex<Half>(3).Should().Be((Half)10);
        }

        #endregion

        #region B10 — Complex maximum/minimum (binary) + clip

        [TestMethod]
        public void B10_Complex_Maximum_LexOrder()
        {
            // NumPy lex: compare real, then imag.
            // np.maximum([1+2j, 1+5j, 1+0j, 2+1j], [1+3j, 1+4j, 2+0j, 1+0j])
            // → [1+3j, 1+5j, 2+0j, 2+1j]
            var a = np.array(new Complex[] { new Complex(1, 2), new Complex(1, 5), new Complex(1, 0), new Complex(2, 1) });
            var b = np.array(new Complex[] { new Complex(1, 3), new Complex(1, 4), new Complex(2, 0), new Complex(1, 0) });
            var r = np.maximum(a, b);
            r.typecode.Should().Be(NPTypeCode.Complex);
            r.GetAtIndex<Complex>(0).Should().Be(new Complex(1, 3));
            r.GetAtIndex<Complex>(1).Should().Be(new Complex(1, 5));
            r.GetAtIndex<Complex>(2).Should().Be(new Complex(2, 0));
            r.GetAtIndex<Complex>(3).Should().Be(new Complex(2, 1));
        }

        [TestMethod]
        public void B10_Complex_Minimum_LexOrder()
        {
            // np.minimum([1+2j, 1+5j, 1+0j, 2+1j], [1+3j, 1+4j, 2+0j, 1+0j])
            // → [1+2j, 1+4j, 1+0j, 1+0j]
            var a = np.array(new Complex[] { new Complex(1, 2), new Complex(1, 5), new Complex(1, 0), new Complex(2, 1) });
            var b = np.array(new Complex[] { new Complex(1, 3), new Complex(1, 4), new Complex(2, 0), new Complex(1, 0) });
            var r = np.minimum(a, b);
            r.typecode.Should().Be(NPTypeCode.Complex);
            r.GetAtIndex<Complex>(0).Should().Be(new Complex(1, 2));
            r.GetAtIndex<Complex>(1).Should().Be(new Complex(1, 4));
            r.GetAtIndex<Complex>(2).Should().Be(new Complex(1, 0));
            r.GetAtIndex<Complex>(3).Should().Be(new Complex(1, 0));
        }

        [TestMethod]
        public void B10_Complex_Maximum_NaN_FirstWins()
        {
            // np.maximum([1+1j, nan+0j, 3+4j], [2+0j, 3+5j, nan+0j])
            // NumPy rule: if either has NaN (real or imag), that element is returned.
            //   pos 0: no NaN, lex → 2+0j
            //   pos 1: a has NaN → a = nan+0j
            //   pos 2: b has NaN → b = nan+0j
            var a = np.array(new Complex[] { new Complex(1, 1), new Complex(double.NaN, 0), new Complex(3, 4) });
            var b = np.array(new Complex[] { new Complex(2, 0), new Complex(3, 5), new Complex(double.NaN, 0) });
            var r = np.maximum(a, b);
            r.typecode.Should().Be(NPTypeCode.Complex);
            r.GetAtIndex<Complex>(0).Should().Be(new Complex(2, 0));
            var r1 = r.GetAtIndex<Complex>(1); double.IsNaN(r1.Real).Should().BeTrue(); r1.Imaginary.Should().Be(0.0);
            var r2 = r.GetAtIndex<Complex>(2); double.IsNaN(r2.Real).Should().BeTrue(); r2.Imaginary.Should().Be(0.0);
        }

        [TestMethod]
        public void B10_Complex_Maximum_BothNaN_FirstWins()
        {
            // Both NaN → first operand wins.
            // np.maximum([complex(nan, 1), complex(nan, 2)], [complex(nan, 3), complex(nan, 4)])
            // → [nan+1j, nan+2j] (first's imag preserved)
            var a = np.array(new Complex[] { new Complex(double.NaN, 1), new Complex(double.NaN, 2) });
            var b = np.array(new Complex[] { new Complex(double.NaN, 3), new Complex(double.NaN, 4) });
            var r = np.maximum(a, b);
            r.typecode.Should().Be(NPTypeCode.Complex);
            var r0 = r.GetAtIndex<Complex>(0); double.IsNaN(r0.Real).Should().BeTrue(); r0.Imaginary.Should().Be(1.0);
            var r1 = r.GetAtIndex<Complex>(1); double.IsNaN(r1.Real).Should().BeTrue(); r1.Imaginary.Should().Be(2.0);
        }

        [TestMethod]
        public void B10_Complex_Maximum_ImagOnlyNaN()
        {
            // Imag-only NaN counts as NaN-carrier too.
            // np.maximum([complex(1, nan), 2+0j], [1+1j, 3+0j])
            // → [1+nanj, 3+0j]
            var a = np.array(new Complex[] { new Complex(1, double.NaN), new Complex(2, 0) });
            var b = np.array(new Complex[] { new Complex(1, 1), new Complex(3, 0) });
            var r = np.maximum(a, b);
            r.typecode.Should().Be(NPTypeCode.Complex);
            var r0 = r.GetAtIndex<Complex>(0); r0.Real.Should().Be(1.0); double.IsNaN(r0.Imaginary).Should().BeTrue();
            r.GetAtIndex<Complex>(1).Should().Be(new Complex(3, 0));
        }

        [TestMethod]
        public void B10_Complex_Clip()
        {
            // np.clip([1+1j, 5+5j, 10+10j], 2+0j, 8+0j) → [2+0j, 5+5j, 8+0j] (lex ordering)
            var a = np.array(new Complex[] { new Complex(1, 1), new Complex(5, 5), new Complex(10, 10) });
            var lo = np.array(new Complex[] { new Complex(2, 0) });
            var hi = np.array(new Complex[] { new Complex(8, 0) });
            var r = np.clip(a, lo, hi);
            r.typecode.Should().Be(NPTypeCode.Complex);
            r.GetAtIndex<Complex>(0).Should().Be(new Complex(2, 0));
            r.GetAtIndex<Complex>(1).Should().Be(new Complex(5, 5));
            r.GetAtIndex<Complex>(2).Should().Be(new Complex(8, 0));
        }

        #endregion

        #region B14 — Half NaN-aware mean/std/var

        [TestMethod]
        public void B14_Half_NanMean_SkipsNaN()
        {
            // np.nanmean([1, 2, nan, 4], dtype=float16) → 2.334 (float16)
            var a = np.array(new Half[] { (Half)1, (Half)2, Half.NaN, (Half)4 });
            var r = np.nanmean(a);
            r.typecode.Should().Be(NPTypeCode.Half);
            ((double)r.GetAtIndex<Half>(0)).Should().BeApproximately(2.334, 0.002);
        }

        [TestMethod]
        public void B14_Half_NanStd_SkipsNaN()
        {
            // np.nanstd → 1.247 (float16)
            var a = np.array(new Half[] { (Half)1, (Half)2, Half.NaN, (Half)4 });
            var r = np.nanstd(a);
            r.typecode.Should().Be(NPTypeCode.Half);
            ((double)r.GetAtIndex<Half>(0)).Should().BeApproximately(1.247, 0.002);
        }

        [TestMethod]
        public void B14_Half_NanVar_SkipsNaN()
        {
            // np.nanvar → 1.556 (float16)
            var a = np.array(new Half[] { (Half)1, (Half)2, Half.NaN, (Half)4 });
            var r = np.nanvar(a);
            r.typecode.Should().Be(NPTypeCode.Half);
            ((double)r.GetAtIndex<Half>(0)).Should().BeApproximately(1.556, 0.002);
        }

        [TestMethod]
        public void B14_Half_AllNaN_ReturnsNaN()
        {
            var a = np.array(new Half[] { Half.NaN, Half.NaN });
            Half.IsNaN(np.nanmean(a).GetAtIndex<Half>(0)).Should().BeTrue();
            Half.IsNaN(np.nanstd(a).GetAtIndex<Half>(0)).Should().BeTrue();
            Half.IsNaN(np.nanvar(a).GetAtIndex<Half>(0)).Should().BeTrue();
        }

        [TestMethod]
        public void B14_Half_NanMean_Axis()
        {
            // np.nanmean([[1, nan, 3], [4, 5, nan], [7, 8, 9]], dtype=float16, axis=0)
            // → [4, 6.5, 6]
            var m = np.array(new Half[,] {
                { (Half)1, Half.NaN, (Half)3 },
                { (Half)4, (Half)5, Half.NaN },
                { (Half)7, (Half)8, (Half)9 }
            });
            var r = np.nanmean(m, axis: 0);
            r.typecode.Should().Be(NPTypeCode.Half);
            ((double)r.GetAtIndex<Half>(0)).Should().BeApproximately(4.0, Tol);
            ((double)r.GetAtIndex<Half>(1)).Should().BeApproximately(6.5, Tol);
            ((double)r.GetAtIndex<Half>(2)).Should().BeApproximately(6.0, Tol);
        }

        [TestMethod]
        public void B14_Half_NanStd_Axis()
        {
            // np.nanstd(...) axis=0 → [2.45, 1.5, 3.]
            var m = np.array(new Half[,] {
                { (Half)1, Half.NaN, (Half)3 },
                { (Half)4, (Half)5, Half.NaN },
                { (Half)7, (Half)8, (Half)9 }
            });
            var r = np.nanstd(m, axis: 0);
            r.typecode.Should().Be(NPTypeCode.Half);
            ((double)r.GetAtIndex<Half>(0)).Should().BeApproximately(2.45, 0.01);
            ((double)r.GetAtIndex<Half>(1)).Should().BeApproximately(1.5, 0.01);
            ((double)r.GetAtIndex<Half>(2)).Should().BeApproximately(3.0, 0.01);
        }

        [TestMethod]
        public void B14_Half_NanVar_Axis()
        {
            // np.nanvar(...) axis=0 → [6, 2.25, 9]
            var m = np.array(new Half[,] {
                { (Half)1, Half.NaN, (Half)3 },
                { (Half)4, (Half)5, Half.NaN },
                { (Half)7, (Half)8, (Half)9 }
            });
            var r = np.nanvar(m, axis: 0);
            r.typecode.Should().Be(NPTypeCode.Half);
            ((double)r.GetAtIndex<Half>(0)).Should().BeApproximately(6.0, 0.05);
            ((double)r.GetAtIndex<Half>(1)).Should().BeApproximately(2.25, 0.01);
            ((double)r.GetAtIndex<Half>(2)).Should().BeApproximately(9.0, 0.05);
        }

        #endregion

        #region B14 — Complex NaN-aware mean/std/var

        [TestMethod]
        public void B14_Complex_NanMean_SkipsNaN_ReturnsComplex()
        {
            // np.nanmean([1+2j, complex(nan, 0), 3+4j]) → (2+3j), dtype=complex128
            var a = np.array(new Complex[] { new Complex(1, 2), new Complex(double.NaN, 0), new Complex(3, 4) });
            var r = np.nanmean(a);
            r.typecode.Should().Be(NPTypeCode.Complex);
            r.GetAtIndex<Complex>(0).Should().Be(new Complex(2, 3));
        }

        [TestMethod]
        public void B14_Complex_NanStd_SkipsNaN_ReturnsDouble()
        {
            // np.nanstd([1+2j, complex(nan, 0), 3+4j]) → 1.4142135623730951 (float64)
            var a = np.array(new Complex[] { new Complex(1, 2), new Complex(double.NaN, 0), new Complex(3, 4) });
            var r = np.nanstd(a);
            r.typecode.Should().Be(NPTypeCode.Double);
            r.GetAtIndex<double>(0).Should().BeApproximately(1.41421356, Tol);
        }

        [TestMethod]
        public void B14_Complex_NanVar_SkipsNaN_ReturnsDouble()
        {
            // np.nanvar → 2.0 (float64). Variance of [1+2j, 3+4j] = mean(|z - mean|²).
            //   mean = 2+3j. |1+2j - (2+3j)|² = |-1-1j|² = 2. |3+4j - (2+3j)|² = |1+1j|² = 2.
            //   var = (2+2)/2 = 2.
            var a = np.array(new Complex[] { new Complex(1, 2), new Complex(double.NaN, 0), new Complex(3, 4) });
            var r = np.nanvar(a);
            r.typecode.Should().Be(NPTypeCode.Double);
            r.GetAtIndex<double>(0).Should().BeApproximately(2.0, Tol);
        }

        [TestMethod]
        public void B14_Complex_NanMean_Axis()
        {
            // np.nanmean over axis=0, Complex:
            // column 0: 1+1j, 4+4j, 7+7j (none NaN) → mean = 4+4j
            // column 1: NaN+0j, 5+5j, 8+8j → skip NaN → mean = 6.5+6.5j
            // column 2: 3+3j, NaN+0j, 9+9j → skip NaN → mean = 6+6j
            var m = np.array(new Complex[,] {
                { new Complex(1, 1), new Complex(double.NaN, 0), new Complex(3, 3) },
                { new Complex(4, 4), new Complex(5, 5), new Complex(double.NaN, 0) },
                { new Complex(7, 7), new Complex(8, 8), new Complex(9, 9) }
            });
            var r = np.nanmean(m, axis: 0);
            r.typecode.Should().Be(NPTypeCode.Complex);
            r.GetAtIndex<Complex>(0).Should().Be(new Complex(4, 4));
            r.GetAtIndex<Complex>(1).Should().Be(new Complex(6.5, 6.5));
            r.GetAtIndex<Complex>(2).Should().Be(new Complex(6, 6));
        }

        [TestMethod]
        public void B14_Complex_NanStd_Axis()
        {
            // np.nanstd axis=0 → [3.464, 2.121, 4.243] (float64)
            var m = np.array(new Complex[,] {
                { new Complex(1, 1), new Complex(double.NaN, 0), new Complex(3, 3) },
                { new Complex(4, 4), new Complex(5, 5), new Complex(double.NaN, 0) },
                { new Complex(7, 7), new Complex(8, 8), new Complex(9, 9) }
            });
            var r = np.nanstd(m, axis: 0);
            r.typecode.Should().Be(NPTypeCode.Double);
            r.GetAtIndex<double>(0).Should().BeApproximately(3.46410162, Tol);
            r.GetAtIndex<double>(1).Should().BeApproximately(2.12132034, Tol);
            r.GetAtIndex<double>(2).Should().BeApproximately(4.24264069, Tol);
        }

        [TestMethod]
        public void B14_Complex_NanVar_Axis()
        {
            // np.nanvar axis=0 → [12, 4.5, 18] (float64)
            var m = np.array(new Complex[,] {
                { new Complex(1, 1), new Complex(double.NaN, 0), new Complex(3, 3) },
                { new Complex(4, 4), new Complex(5, 5), new Complex(double.NaN, 0) },
                { new Complex(7, 7), new Complex(8, 8), new Complex(9, 9) }
            });
            var r = np.nanvar(m, axis: 0);
            r.typecode.Should().Be(NPTypeCode.Double);
            r.GetAtIndex<double>(0).Should().BeApproximately(12.0, Tol);
            r.GetAtIndex<double>(1).Should().BeApproximately(4.5, Tol);
            r.GetAtIndex<double>(2).Should().BeApproximately(18.0, Tol);
        }

        #endregion
    }
}
