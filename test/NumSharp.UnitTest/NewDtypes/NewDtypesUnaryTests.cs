using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;

namespace NumSharp.UnitTest.NewDtypes
{
    /// <summary>
    /// Unary operation tests for SByte (int8), Half (float16), Complex (complex128)
    /// All expected values verified against NumPy 2.x
    /// </summary>
    [TestClass]
    public class NewDtypesUnaryTests
    {
        #region SByte (int8) Unary

        [TestMethod]
        public void SByte_Abs()
        {
            // NumPy: np.abs(np.array([-128, -1, 0, 1, 127], dtype=np.int8))
            // Result: [-128, 1, 0, 1, 127] - note: abs(-128) overflows back to -128
            var a = np.array(new sbyte[] { -128, -1, 0, 1, 127 });
            var result = np.abs(a);

            result.typecode.Should().Be(NPTypeCode.SByte);
            result.GetAtIndex<sbyte>(0).Should().Be((sbyte)-128); // overflow!
            result.GetAtIndex<sbyte>(1).Should().Be((sbyte)1);
            result.GetAtIndex<sbyte>(2).Should().Be((sbyte)0);
            result.GetAtIndex<sbyte>(3).Should().Be((sbyte)1);
            result.GetAtIndex<sbyte>(4).Should().Be((sbyte)127);
        }

        [TestMethod]
        public void SByte_Sign()
        {
            // NumPy: np.sign(np.array([-128, -1, 0, 1, 127], dtype=np.int8))
            // Result: [-1, -1, 0, 1, 1] (dtype: int8)
            var a = np.array(new sbyte[] { -128, -1, 0, 1, 127 });
            var result = np.sign(a);

            result.typecode.Should().Be(NPTypeCode.SByte);
            result.GetAtIndex<sbyte>(0).Should().Be((sbyte)-1);
            result.GetAtIndex<sbyte>(1).Should().Be((sbyte)-1);
            result.GetAtIndex<sbyte>(2).Should().Be((sbyte)0);
            result.GetAtIndex<sbyte>(3).Should().Be((sbyte)1);
            result.GetAtIndex<sbyte>(4).Should().Be((sbyte)1);
        }

        [TestMethod]
        public void SByte_Square()
        {
            // NumPy: np.square(np.array([1, 2, 3, 4], dtype=np.int8))
            // Result: [1, 4, 9, 16] (dtype: int8)
            var a = np.array(new sbyte[] { 1, 2, 3, 4 });
            var result = np.square(a);

            result.typecode.Should().Be(NPTypeCode.SByte);
            result.GetAtIndex<sbyte>(0).Should().Be((sbyte)1);
            result.GetAtIndex<sbyte>(1).Should().Be((sbyte)4);
            result.GetAtIndex<sbyte>(2).Should().Be((sbyte)9);
            result.GetAtIndex<sbyte>(3).Should().Be((sbyte)16);
        }

        #endregion

        #region Half (float16) Unary

        [TestMethod]
        public void Half_Abs()
        {
            // NumPy: np.abs(np.array([0.0, 1.5, -2.5, nan, inf], dtype=np.float16))
            // Result: [0.0, 1.5, 2.5, nan, inf] (dtype: float16)
            var h = np.array(new Half[] { (Half)0.0, (Half)1.5, (Half)(-2.5), Half.NaN, Half.PositiveInfinity });
            var result = np.abs(h);

            result.typecode.Should().Be(NPTypeCode.Half);
            result.GetAtIndex<Half>(0).Should().Be((Half)0.0);
            result.GetAtIndex<Half>(1).Should().Be((Half)1.5);
            result.GetAtIndex<Half>(2).Should().Be((Half)2.5);
            Half.IsNaN(result.GetAtIndex<Half>(3)).Should().BeTrue();
            Half.IsPositiveInfinity(result.GetAtIndex<Half>(4)).Should().BeTrue();
        }

        [TestMethod]
        public void Half_Sign()
        {
            // NumPy: np.sign(np.array([0.0, 1.5, -2.5, nan, inf], dtype=np.float16))
            // Result: [0.0, 1.0, -1.0, nan, 1.0] (dtype: float16)
            var h = np.array(new Half[] { (Half)0.0, (Half)1.5, (Half)(-2.5), Half.NaN, Half.PositiveInfinity });
            var result = np.sign(h);

            result.typecode.Should().Be(NPTypeCode.Half);
            result.GetAtIndex<Half>(0).Should().Be((Half)0.0);
            result.GetAtIndex<Half>(1).Should().Be((Half)1.0);
            result.GetAtIndex<Half>(2).Should().Be((Half)(-1.0));
            Half.IsNaN(result.GetAtIndex<Half>(3)).Should().BeTrue();
            result.GetAtIndex<Half>(4).Should().Be((Half)1.0);
        }

        [TestMethod]
        public void Half_Sqrt()
        {
            // NumPy: np.sqrt(np.array([0, 1, 4, 9], dtype=np.float16))
            // Result: [0.0, 1.0, 2.0, 3.0] (dtype: float16)
            var h = np.array(new Half[] { (Half)0, (Half)1, (Half)4, (Half)9 });
            var result = np.sqrt(h);

            result.typecode.Should().Be(NPTypeCode.Half);
            result.GetAtIndex<Half>(0).Should().Be((Half)0.0);
            result.GetAtIndex<Half>(1).Should().Be((Half)1.0);
            result.GetAtIndex<Half>(2).Should().Be((Half)2.0);
            result.GetAtIndex<Half>(3).Should().Be((Half)3.0);
        }

        [TestMethod]
        public void Half_Floor()
        {
            // NumPy: np.floor(np.array([1.2, 2.7, -1.5, -2.8], dtype=np.float16))
            // Result: [1.0, 2.0, -2.0, -3.0] (dtype: float16)
            var h = np.array(new Half[] { (Half)1.2, (Half)2.7, (Half)(-1.5), (Half)(-2.8) });
            var result = np.floor(h);

            result.typecode.Should().Be(NPTypeCode.Half);
            result.GetAtIndex<Half>(0).Should().Be((Half)1.0);
            result.GetAtIndex<Half>(1).Should().Be((Half)2.0);
            result.GetAtIndex<Half>(2).Should().Be((Half)(-2.0));
            result.GetAtIndex<Half>(3).Should().Be((Half)(-3.0));
        }

        [TestMethod]
        public void Half_Ceil()
        {
            // NumPy: np.ceil(np.array([1.2, 2.7, -1.5, -2.8], dtype=np.float16))
            // Result: [2.0, 3.0, -1.0, -2.0] (dtype: float16)
            var h = np.array(new Half[] { (Half)1.2, (Half)2.7, (Half)(-1.5), (Half)(-2.8) });
            var result = np.ceil(h);

            result.typecode.Should().Be(NPTypeCode.Half);
            result.GetAtIndex<Half>(0).Should().Be((Half)2.0);
            result.GetAtIndex<Half>(1).Should().Be((Half)3.0);
            result.GetAtIndex<Half>(2).Should().Be((Half)(-1.0));
            result.GetAtIndex<Half>(3).Should().Be((Half)(-2.0));
        }

        [TestMethod]
        public void Half_Exp()
        {
            // NumPy: np.exp(np.array([0.0, 1.0, 2.0], dtype=np.float16))
            // Result: [1.0, 2.719, 7.39] (dtype: float16)
            var h = np.array(new Half[] { (Half)0.0, (Half)1.0, (Half)2.0 });
            var result = np.exp(h);

            result.typecode.Should().Be(NPTypeCode.Half);
            result.GetAtIndex<Half>(0).Should().Be((Half)1.0);
            ((double)result.GetAtIndex<Half>(1)).Should().BeApproximately(2.718, 0.01);
            ((double)result.GetAtIndex<Half>(2)).Should().BeApproximately(7.39, 0.1);
        }

        [TestMethod]
        public void Half_Sin()
        {
            // NumPy: np.sin(np.array([0.0, pi/6, pi/4, pi/2], dtype=np.float16))
            // Result: [0.0, 0.5, 0.707, 1.0] (dtype: float16)
            var h = np.array(new Half[] { (Half)0.0, (Half)(Math.PI / 6), (Half)(Math.PI / 4), (Half)(Math.PI / 2) });
            var result = np.sin(h);

            result.typecode.Should().Be(NPTypeCode.Half);
            ((double)result.GetAtIndex<Half>(0)).Should().BeApproximately(0.0, 0.01);
            ((double)result.GetAtIndex<Half>(1)).Should().BeApproximately(0.5, 0.01);
            ((double)result.GetAtIndex<Half>(2)).Should().BeApproximately(0.707, 0.01);
            ((double)result.GetAtIndex<Half>(3)).Should().BeApproximately(1.0, 0.01);
        }

        #endregion

        #region Complex (complex128) Unary

        [TestMethod]
        public void Complex_Abs_ReturnsFloat64()
        {
            // NumPy: np.abs(np.array([1+2j, 3+4j, 0+0j, -1-1j]))
            // Result: [2.236, 5.0, 0.0, 1.414] (dtype: float64)
            var z = np.array(new Complex[] { new(1, 2), new(3, 4), new(0, 0), new(-1, -1) });
            var result = np.abs(z);

            result.typecode.Should().Be(NPTypeCode.Double);
            result.GetAtIndex<double>(0).Should().BeApproximately(2.23606797749979, 0.0001);
            result.GetAtIndex<double>(1).Should().BeApproximately(5.0, 0.0001);
            result.GetAtIndex<double>(2).Should().BeApproximately(0.0, 0.0001);
            result.GetAtIndex<double>(3).Should().BeApproximately(1.4142135623730951, 0.0001);
        }

        [TestMethod]
        public void Complex_Sign_ReturnsUnitVector()
        {
            // NumPy: np.sign(np.array([1+2j, 3+4j, 0+0j, -1-1j]))
            // Result: unit vectors [0.447+0.894j, 0.6+0.8j, 0+0j, -0.707-0.707j]
            var z = np.array(new Complex[] { new(1, 2), new(3, 4), new(0, 0), new(-1, -1) });
            var result = np.sign(z);

            result.typecode.Should().Be(NPTypeCode.Complex);

            var r0 = result.GetAtIndex<Complex>(0);
            r0.Real.Should().BeApproximately(0.4472136, 0.0001);
            r0.Imaginary.Should().BeApproximately(0.8944272, 0.0001);

            var r1 = result.GetAtIndex<Complex>(1);
            r1.Real.Should().BeApproximately(0.6, 0.0001);
            r1.Imaginary.Should().BeApproximately(0.8, 0.0001);

            result.GetAtIndex<Complex>(2).Should().Be(Complex.Zero);

            var r3 = result.GetAtIndex<Complex>(3);
            r3.Real.Should().BeApproximately(-0.7071068, 0.0001);
            r3.Imaginary.Should().BeApproximately(-0.7071068, 0.0001);
        }

        [TestMethod]
        public void Complex_Sqrt()
        {
            // NumPy: np.sqrt(np.array([1+0j, 0+1j, 1+1j]))
            // Result: [1+0j, 0.707+0.707j, 1.099+0.455j]
            var z = np.array(new Complex[] { new(1, 0), new(0, 1), new(1, 1) });
            var result = np.sqrt(z);

            result.typecode.Should().Be(NPTypeCode.Complex);

            result.GetAtIndex<Complex>(0).Should().Be(new Complex(1, 0));

            var r1 = result.GetAtIndex<Complex>(1);
            r1.Real.Should().BeApproximately(0.7071068, 0.0001);
            r1.Imaginary.Should().BeApproximately(0.7071068, 0.0001);

            var r2 = result.GetAtIndex<Complex>(2);
            r2.Real.Should().BeApproximately(1.0986841, 0.0001);
            r2.Imaginary.Should().BeApproximately(0.4550899, 0.0001);
        }

        [TestMethod]
        public void Complex_Exp()
        {
            // NumPy: np.exp(np.array([0+0j, 1+0j, 0+1j]))
            // Result: [1+0j, 2.718+0j, 0.540+0.841j]
            var z = np.array(new Complex[] { new(0, 0), new(1, 0), new(0, 1) });
            var result = np.exp(z);

            result.typecode.Should().Be(NPTypeCode.Complex);
            result.GetAtIndex<Complex>(0).Should().Be(new Complex(1, 0));

            var r1 = result.GetAtIndex<Complex>(1);
            r1.Real.Should().BeApproximately(Math.E, 0.0001);
            r1.Imaginary.Should().BeApproximately(0, 0.0001);

            var r2 = result.GetAtIndex<Complex>(2);
            r2.Real.Should().BeApproximately(0.5403023, 0.0001);
            r2.Imaginary.Should().BeApproximately(0.8414710, 0.0001);
        }

        [TestMethod]
        public void Complex_Log()
        {
            // NumPy: np.log(np.array([1+0j, 0+1j, 1+1j]))
            // Result: [0+0j, 0+1.571j, 0.347+0.785j]
            var z = np.array(new Complex[] { new(1, 0), new(0, 1), new(1, 1) });
            var result = np.log(z);

            result.typecode.Should().Be(NPTypeCode.Complex);
            result.GetAtIndex<Complex>(0).Should().Be(new Complex(0, 0));

            var r1 = result.GetAtIndex<Complex>(1);
            r1.Real.Should().BeApproximately(0, 0.0001);
            r1.Imaginary.Should().BeApproximately(Math.PI / 2, 0.0001);

            var r2 = result.GetAtIndex<Complex>(2);
            r2.Real.Should().BeApproximately(0.3465736, 0.0001);
            r2.Imaginary.Should().BeApproximately(Math.PI / 4, 0.0001);
        }

        // ---- Complex hyperbolic / inverse-trig (csinh/ccosh/ctanh/casin/cacos/catan) ----
        // All expected values verified against NumPy 2.4.2. These six previously threw
        // NotSupportedException for Complex; they now match NumPy including C99 Annex G
        // special values, branch-cut signs, and signed zeros.

        private static void AssertComplex(Complex actual, double re, double im, double tol = 1e-12)
        {
            actual.Real.Should().BeApproximately(re, tol);
            actual.Imaginary.Should().BeApproximately(im, tol);
        }

        [TestMethod]
        public void Complex_Sinh()
        {
            // NumPy: np.sinh([1+2j, 3-4j, -5+0j])
            var z = np.array(new Complex[] { new(1, 2), new(3, -4), new(-5, 0) });
            var r = np.sinh(z);
            r.typecode.Should().Be(NPTypeCode.Complex);
            AssertComplex(r.GetAtIndex<Complex>(0), -0.4890562590412937, 1.4031192506220405);
            AssertComplex(r.GetAtIndex<Complex>(1), -6.5481200409110025, 7.61923172032141);
            AssertComplex(r.GetAtIndex<Complex>(2), -74.20321057778875, 0.0);
        }

        [TestMethod]
        public void Complex_Cosh()
        {
            // NumPy: np.cosh([1+2j, 3-4j, 0+0j])
            var z = np.array(new Complex[] { new(1, 2), new(3, -4), new(0, 0) });
            var r = np.cosh(z);
            r.typecode.Should().Be(NPTypeCode.Complex);
            AssertComplex(r.GetAtIndex<Complex>(0), -0.64214812471552, 1.0686074213827783);
            AssertComplex(r.GetAtIndex<Complex>(1), -6.580663040551157, 7.581552742746545);
            AssertComplex(r.GetAtIndex<Complex>(2), 1.0, 0.0);
        }

        [TestMethod]
        public void Complex_Tanh()
        {
            // NumPy: np.tanh([1+2j, 3-4j])
            var z = np.array(new Complex[] { new(1, 2), new(3, -4) });
            var r = np.tanh(z);
            r.typecode.Should().Be(NPTypeCode.Complex);
            AssertComplex(r.GetAtIndex<Complex>(0), 1.16673625724092, -0.24345820118572525);
            AssertComplex(r.GetAtIndex<Complex>(1), 1.000709536067233, -0.004908258067496059);
        }

        [TestMethod]
        public void Complex_Arcsin()
        {
            // NumPy: np.arcsin([1+2j, 0.5+0.5j])
            var z = np.array(new Complex[] { new(1, 2), new(0.5, 0.5) });
            var r = np.arcsin(z);
            r.typecode.Should().Be(NPTypeCode.Complex);
            AssertComplex(r.GetAtIndex<Complex>(0), 0.42707858639247614, 1.5285709194809982);
            AssertComplex(r.GetAtIndex<Complex>(1), 0.45227844715119064, 0.5306375309525179);
        }

        [TestMethod]
        public void Complex_Arccos()
        {
            // NumPy: np.arccos([1+2j, 0.5+0.5j])
            var z = np.array(new Complex[] { new(1, 2), new(0.5, 0.5) });
            var r = np.arccos(z);
            r.typecode.Should().Be(NPTypeCode.Complex);
            AssertComplex(r.GetAtIndex<Complex>(0), 1.1437177404024204, -1.5285709194809982);
            AssertComplex(r.GetAtIndex<Complex>(1), 1.118517879643706, -0.5306375309525179);
        }

        [TestMethod]
        public void Complex_Arctan()
        {
            // NumPy: np.arctan([1+2j, 3-4j])
            var z = np.array(new Complex[] { new(1, 2), new(3, -4) });
            var r = np.arctan(z);
            r.typecode.Should().Be(NPTypeCode.Complex);
            AssertComplex(r.GetAtIndex<Complex>(0), 1.3389725222944935, 0.40235947810852507);
            AssertComplex(r.GetAtIndex<Complex>(1), 1.4483069952314644, -0.15899719167999918);
        }

        [TestMethod]
        public void Complex_InverseTrig_BranchCuts()
        {
            // On the real-axis cut |x|>1 the sign of the imaginary output follows the sign of the
            // (possibly signed-zero) input imaginary part. NumPy 2.4.2:
            //   arcsin(2+0j) = (pi/2,  1.3169578969248166)   arcsin(2-0j) = (pi/2, -1.3169578969248166)
            //   arccos(2+0j) = (0,    -1.3169578969248166)
            //   arctan(0+2j) = (pi/2,  0.5493061443340549)   [imaginary-axis cut |y|>1]
            var asinPos = np.arcsin(np.array(new Complex[] { new(2, 0.0) })).GetAtIndex<Complex>(0);
            AssertComplex(asinPos, Math.PI / 2, 1.3169578969248166);
            var asinNeg = np.arcsin(np.array(new Complex[] { new(2, -0.0) })).GetAtIndex<Complex>(0);
            AssertComplex(asinNeg, Math.PI / 2, -1.3169578969248166);

            var acos = np.arccos(np.array(new Complex[] { new(2, 0.0) })).GetAtIndex<Complex>(0);
            AssertComplex(acos, 0.0, -1.3169578969248166);

            var atan = np.arctan(np.array(new Complex[] { new(0.0, 2) })).GetAtIndex<Complex>(0);
            AssertComplex(atan, Math.PI / 2, 0.5493061443340549);
        }

        [TestMethod]
        public void Complex_HyperbolicInverse_SpecialValues()
        {
            // C99 Annex G special values, verified against NumPy 2.4.2.
            var sinh = np.sinh(np.array(new Complex[] {
                new(double.PositiveInfinity, 0.0),   // (inf, 0)
                new(double.NegativeInfinity, double.PositiveInfinity), // (-inf, nan)
            }));
            AssertComplex(sinh.GetAtIndex<Complex>(0), double.PositiveInfinity, 0.0);
            sinh.GetAtIndex<Complex>(1).Real.Should().Be(double.NegativeInfinity);
            double.IsNaN(sinh.GetAtIndex<Complex>(1).Imaginary).Should().BeTrue();

            // cosh(inf+i) = inf+i*inf ; tanh(inf+i*inf) = 1+i0
            var cosh = np.cosh(np.array(new Complex[] { new(double.NegativeInfinity, 1.0) })).GetAtIndex<Complex>(0);
            cosh.Real.Should().Be(double.PositiveInfinity);
            cosh.Imaginary.Should().Be(double.PositiveInfinity);

            var tanh = np.tanh(np.array(new Complex[] { new(double.PositiveInfinity, double.PositiveInfinity) })).GetAtIndex<Complex>(0);
            AssertComplex(tanh, 1.0, 0.0);

            // arcsin(inf+1j) = pi/2 + i*inf ; arctan(inf+0j) = pi/2 + 0j ; arccos(0+nan*j) = pi/2 + nan*j
            var asin = np.arcsin(np.array(new Complex[] { new(double.PositiveInfinity, 1.0) })).GetAtIndex<Complex>(0);
            asin.Real.Should().BeApproximately(Math.PI / 2, 1e-12);
            asin.Imaginary.Should().Be(double.PositiveInfinity);

            var atan = np.arctan(np.array(new Complex[] { new(double.PositiveInfinity, 0.0) })).GetAtIndex<Complex>(0);
            AssertComplex(atan, Math.PI / 2, 0.0);

            var acos = np.arccos(np.array(new Complex[] { new(0.0, double.NaN) })).GetAtIndex<Complex>(0);
            acos.Real.Should().BeApproximately(Math.PI / 2, 1e-12);
            double.IsNaN(acos.Imaginary).Should().BeTrue();
        }

        [TestMethod]
        public void Complex_Sinh_PreservesSignedZero()
        {
            // NumPy: np.sinh(-0+inf*1j) = (-0, nan) — the real part keeps the sign of the input zero.
            var r = np.sinh(np.array(new Complex[] { new(-0.0, double.PositiveInfinity) })).GetAtIndex<Complex>(0);
            r.Real.Should().Be(0.0);
            double.IsNegative(r.Real).Should().BeTrue("NumPy sinh(-0+inf j).real is -0.0");
            double.IsNaN(r.Imaginary).Should().BeTrue();
        }

        // ---- Full-port behaviors (csinh/ccosh ladder, Kahan ctanh, npy_clog, npy_catanh, FMA z*z,
        // Goldberg expm1). All values verified against NumPy 2.4.2. These exercise the large-magnitude,
        // near-unit-circle, and tiny/subnormal regimes the BCL transcendentals previously got wrong. ----

        [TestMethod]
        public void Complex_SinCos_HugeImaginary()
        {
            // sin/cos of a large *finite* imaginary part. Complex.Sin/Cos return NaN here (cosh(huge)*0);
            // NumPy's csin/ccos route through csinh/ccosh, which the port now matches: the imaginary
            // magnitude overflows to +-inf but the other component stays exact.
            var s = np.sin(np.array(new Complex[] { new(0.0, 1e300) })).GetAtIndex<Complex>(0);
            s.Real.Should().Be(0.0);              // NOT NaN
            s.Imaginary.Should().Be(double.PositiveInfinity);

            var c = np.cos(np.array(new Complex[] { new(0.0, 1e300) })).GetAtIndex<Complex>(0);
            c.Real.Should().Be(double.PositiveInfinity);
            c.Imaginary.Should().Be(0.0);
            double.IsNegative(c.Imaginary).Should().BeTrue("NumPy cos(0+1e300 j).imag = -0.0");
        }

        [TestMethod]
        public void Complex_SinhCosh_LargeRealOverflow()
        {
            // Large finite real part: still finite where cosh(x) is representable (NumPy sinh(700+1j)).
            var s = np.sinh(np.array(new Complex[] { new(700.0, 1.0) })).GetAtIndex<Complex>(0);
            AssertComplex(s, 2.7399595892935213e+303, 4.2672342296080034e+303, 1e+289);

            // Past the overflow edge (|x| >= ~710.5) both NumPy and the port go to +-inf, component-wise.
            var c = np.cosh(np.array(new Complex[] { new(710.5, 2.0) })).GetAtIndex<Complex>(0);
            c.Real.Should().Be(double.NegativeInfinity);    // cosh*cos(2), cos(2)<0
            c.Imaginary.Should().Be(double.PositiveInfinity);

            var s2 = np.sinh(np.array(new Complex[] { new(1000.0, 0.5) })).GetAtIndex<Complex>(0);
            s2.Real.Should().Be(double.PositiveInfinity);
            s2.Imaginary.Should().Be(double.PositiveInfinity);
        }

        [TestMethod]
        public void Complex_Tan_RealAxis_KahanAccuracy()
        {
            // tan(1.5+0j) == tan(1.5). The BCL Complex.Tan drifts ~33 ULP (~5.9e-14) here; the Kahan
            // ctanh port matches NumPy's 14.101419947171719 to the bit. tol smaller than the old error.
            var t = np.tan(np.array(new Complex[] { new(1.5, 0.0) })).GetAtIndex<Complex>(0);
            AssertComplex(t, 14.101419947171719, 0.0, 1e-14);

            // tanh(0+1.5j) = (0, tan(1.5)) — the imaginary-axis image of the same value.
            var th = np.tanh(np.array(new Complex[] { new(0.0, 1.5) })).GetAtIndex<Complex>(0);
            AssertComplex(th, 0.0, 14.101419947171719, 1e-14);
        }

        [TestMethod]
        public void Complex_Log_NearUnitCircle()
        {
            // log(z) with |z| ~ 1: the real part is log|z|, which cancels to 0 under the naive
            // log(hypot(re,im)). NumPy's clog uses a log1p path; the port reproduces 5e-21 (not 0).
            var l = np.log(np.array(new Complex[] { new(1.0, 1e-10) })).GetAtIndex<Complex>(0);
            l.Real.Should().BeApproximately(5.0000000000000005e-21, 1e-36);
            l.Real.Should().NotBe(0.0, "clog's log1p path keeps the tiny real part");
            l.Imaginary.Should().BeApproximately(1e-10, 1e-25);

            // A point exactly on the unit circle: log(0.6+0.8j).real ~ 0 (|z| == 1).
            var u = np.log(np.array(new Complex[] { new(0.6, 0.8) })).GetAtIndex<Complex>(0);
            u.Real.Should().BeApproximately(2.7755575615628914e-17, 1e-30);
            u.Imaginary.Should().BeApproximately(0.9272952180016123, 1e-15);
        }

        [TestMethod]
        public void Complex_Log10_Log2_Log1p()
        {
            // log10/log2 = clog * (1/ln10 | 1/ln2); they inherit clog's near-1 accuracy.
            var l10 = np.log10(np.array(new Complex[] { new(1.0, 1e-10) })).GetAtIndex<Complex>(0);
            AssertComplex(l10, 2.1714724095162594e-21, 4.342944819032518e-11, 1e-25);

            var l2 = np.log2(np.array(new Complex[] { new(1.0, 1e-10) })).GetAtIndex<Complex>(0);
            AssertComplex(l2, 7.213475204444818e-21, 1.4426950408889633e-10, 1e-24);

            // Complex log1p is the *naive* clog(1+z) (NumPy does NOT refine it): log1p(1e-10 j).real == 0.
            var l1p = np.log1p(np.array(new Complex[] { new(0.0, 1e-10) })).GetAtIndex<Complex>(0);
            l1p.Real.Should().Be(0.0, "NumPy's complex log1p does not apply clog's near-1 refinement");
            l1p.Imaginary.Should().BeApproximately(1e-10, 1e-25);
        }

        [TestMethod]
        public void Complex_Expm1_TinyInput_Goldberg()
        {
            // expm1(x) for tiny x: exp(x)-1 cancels ~10 digits and underflows; the Goldberg correction
            // (e^x-1)*x/log(e^x) recovers it. NumPy expm1(1e-10) = 1.00000000005e-10 (not 1.0000000827e-10).
            var t = np.expm1(np.array(new Complex[] { new(1e-10, 0.0) })).GetAtIndex<Complex>(0);
            t.Real.Should().BeApproximately(1.00000000005e-10, 1e-25);

            // No underflow to 0: expm1(1e-300) == 1e-300.
            var u = np.expm1(np.array(new Complex[] { new(1e-300, 0.0) })).GetAtIndex<Complex>(0);
            u.Real.Should().Be(1e-300);

            // Interior value unaffected: expm1(2+3j).
            var m = np.expm1(np.array(new Complex[] { new(2.0, 3.0) })).GetAtIndex<Complex>(0);
            AssertComplex(m, -8.315110094901103, 1.0427436562359045, 1e-13);
        }

        [TestMethod]
        public void Complex_Exp2()
        {
            // exp2(z) = exp(z*ln2). Interior + non-finite (the C99-correct Exp keeps the signed zero).
            var e = np.exp2(np.array(new Complex[] { new(3.0, 1.0) })).GetAtIndex<Complex>(0);
            AssertComplex(e, 6.153911210911775, 5.111690210509077, 1e-13);

            var inf = np.exp2(np.array(new Complex[] { new(double.NegativeInfinity, double.NegativeInfinity) })).GetAtIndex<Complex>(0);
            inf.Real.Should().Be(0.0);
            inf.Imaginary.Should().Be(0.0);
            double.IsNegative(inf.Imaginary).Should().BeTrue("NumPy exp2(-inf-inf j).imag = -0.0");
        }

        [TestMethod]
        public void Complex_Square_FmaContraction()
        {
            // np.square(z) == z*z with FMA-contracted multiply. Complex.op_Multiply (no FMA) returns 0
            // for the real part here (exact 1e-20 - 1e-20); NumPy's fma(re,re,-(im*im)) keeps -2.275e-37.
            var s = np.square(np.array(new Complex[] { new(1e-10, 1e-10) })).GetAtIndex<Complex>(0);
            s.Real.Should().BeApproximately(-2.275215372846689e-37, 1e-52);
            s.Real.Should().NotBe(0.0, "FMA exposes the rounding of im*im");
            s.Imaginary.Should().BeApproximately(2.0000000000000002e-20, 1e-35);

            // Interior unaffected, and the overflow gives -inf (not NaN from inf-inf).
            AssertComplex(np.square(np.array(new Complex[] { new(2.0, 3.0) })).GetAtIndex<Complex>(0), -5.0, 12.0, 1e-13);
            var big = np.square(np.array(new Complex[] { new(1e300, 1e300) })).GetAtIndex<Complex>(0);
            big.Real.Should().Be(double.NegativeInfinity);
            big.Imaginary.Should().Be(double.PositiveInfinity);
        }

        [TestMethod]
        public void Complex_Reciprocal()
        {
            // nc_recip (Smith's algorithm): correct signed zeros + overflow-safe.
            AssertComplex(np.reciprocal(np.array(new Complex[] { new(3.0, 4.0) })).GetAtIndex<Complex>(0), 0.12, -0.16, 1e-15);

            var one = np.reciprocal(np.array(new Complex[] { new(1.0, 0.0) })).GetAtIndex<Complex>(0);
            one.Real.Should().Be(1.0);
            one.Imaginary.Should().Be(0.0);
            double.IsNegative(one.Imaginary).Should().BeTrue("NumPy reciprocal(1+0j).imag = -0.0");

            // reciprocal(inf+inf j) = nan+nan j (overflow-safe path doesn't flush to 0).
            var inf = np.reciprocal(np.array(new Complex[] { new(double.PositiveInfinity, double.PositiveInfinity) })).GetAtIndex<Complex>(0);
            double.IsNaN(inf.Real).Should().BeTrue();
            double.IsNaN(inf.Imaginary).Should().BeTrue();
        }

        [TestMethod]
        public void Complex_Arctan_TinyAndLarge()
        {
            // catanh port: tiny imaginary part is preserved (Complex.Atan cancels/underflows it).
            var t = np.arctan(np.array(new Complex[] { new(0.0, 1e-10) })).GetAtIndex<Complex>(0);
            t.Real.Should().Be(0.0);
            t.Imaginary.Should().BeApproximately(1e-10, 1e-25);

            AssertComplex(np.arctan(np.array(new Complex[] { new(1.0, 1e-10) })).GetAtIndex<Complex>(0),
                          0.7853981633974483, 5e-11, 1e-25);

            // Large real part: real_part_reciprocal keeps Im = 1/x (not flushed) ~ 0; Re -> pi/2.
            var big = np.arctan(np.array(new Complex[] { new(1e300, 1.0) })).GetAtIndex<Complex>(0);
            big.Real.Should().BeApproximately(Math.PI / 2, 1e-15);
            big.Imaginary.Should().BeApproximately(0.0, 1e-290);
        }

        [TestMethod]
        public void Complex_Exp_NegInfinity_SignedZeroImaginary()
        {
            // exp(-inf - inf j) = 0 - 0j: the system libm keeps sign(y) on the imaginary zero, which
            // npy_cexp's flat (0,0) drops. NumPy 2.4.2 -> imag is -0.0.
            var r = np.exp(np.array(new Complex[] { new(double.NegativeInfinity, double.NegativeInfinity) })).GetAtIndex<Complex>(0);
            r.Real.Should().Be(0.0);
            r.Imaginary.Should().Be(0.0);
            double.IsNegative(r.Imaginary).Should().BeTrue("NumPy exp(-inf-inf j).imag = -0.0");
        }

        [TestMethod]
        public void Complex_FloatUfunc_NarrowingDtype_RaisesCastError_NotSegfault()
        {
            // REGRESSION: np.<floatufunc>(complex, dtype=<real float>) used to SEGFAULT — the float
            // resolver honored the narrower dtype= and allocated a real output buffer half the width
            // of the complex the kernel writes (buffer overflow). NumPy 2.4.2 raises a UFuncTypeError:
            //   "Cannot cast ufunc '<name>' input from dtype('complex128') to dtype('float64') with
            //    casting rule 'same_kind'".
            var z = np.array(new Complex[] { new(1, 2), new(3, -1) });
            foreach (var op in new (string, Func<NPTypeCode?, NDArray>)[]
            {
                ("exp",    d => np.exp(z, dtype: d)),
                ("log",    d => np.log(z, dtype: d)),
                ("sqrt",   d => np.sqrt(z, dtype: d)),
                ("sin",    d => np.sin(z, dtype: d)),
                ("tanh",   d => np.tanh(z, dtype: d)),
                ("arctan", d => np.arctan(z, dtype: d)),
            })
            {
                // complex input + real-float dtype= -> verbatim NumPy cast error (no crash).
                Action narrow = () => op.Item2(NPTypeCode.Double);
                narrow.Should().Throw<ArgumentException>($"{op.Item1}(complex, dtype=float64) must reject the cast")
                    .WithMessage($"Cannot cast ufunc '{op.Item1}' input from dtype('complex128') to dtype('float64') with casting rule 'same_kind'");

                // complex input + integer dtype= -> no loop exists at all (different NumPy error).
                Action noloop = () => op.Item2(NPTypeCode.Int64);
                noloop.Should().Throw<Exception>($"{op.Item1}(complex, dtype=int64) selects no loop");

                // complex input + complex dtype= is a legal no-op loop and must return complex.
                op.Item2(NPTypeCode.Complex).typecode.Should().Be(NPTypeCode.Complex, $"{op.Item1}(complex, dtype=complex128) is valid");
            }
        }

        #endregion

        #region Exp2 Single-output (regression: malformed IL emitter)

        // np.exp2's float32-output IL kernel left the evaluation stack unbalanced (a spurious extra
        // Ldc_R8 2.0 that also clobbered the exponent local with the constant 2.0), so EVERY exp2 call
        // resolving to a Single output threw InvalidProgramException — and, had it not crashed, would
        // have returned Pow(2,2)=4 for every element. The Single output is reached by int16/uint16/
        // char/float32 inputs (ResolveUnaryFloatReturnType) or any input with dtype=float32. These
        // assert both no-crash AND correct 2^x values, verified against NumPy 2.4.2: exp2([1,2,3])=[2,4,8].

        private static void AssertSingleExp2(NDArray r, params float[] expected)
        {
            r.typecode.Should().Be(NPTypeCode.Single);
            r.size.Should().Be(expected.Length);
            for (int i = 0; i < expected.Length; i++)
                r.GetSingle(i).Should().BeApproximately(expected[i], 1e-6f, $"exp2 element {i}");
        }

        [TestMethod]
        public void Exp2_SingleOutput_Int16_Uint16_Char()
        {
            // int16/uint16/char all promote to a float32 (Single) exp2 output in NumPy. exp2([1,2,3])=[2,4,8].
            AssertSingleExp2(np.exp2(np.array(new short[] { 1, 2, 3 })), 2f, 4f, 8f);
            AssertSingleExp2(np.exp2(np.array(new ushort[] { 1, 2, 3 })), 2f, 4f, 8f);
            AssertSingleExp2(np.exp2(np.array(new char[] { (char)1, (char)2, (char)3 })), 2f, 4f, 8f);
        }

        [TestMethod]
        public void Exp2_SingleOutput_Float32_PreservesDtypeAndValues()
        {
            // 2^[0,.5,1,-1,3] = [1, sqrt(2), 2, 0.5, 8]; sqrt(2) in float32 = 1.4142135.
            AssertSingleExp2(np.exp2(np.array(new float[] { 0f, 0.5f, 1f, -1f, 3f })),
                1f, 1.4142135f, 2f, 0.5f, 8f);
        }

        [TestMethod]
        public void Exp2_SingleOutput_DtypeOverride_FromIntAndDouble()
        {
            // dtype=float32 forces the Single emitter from a wider input. NumPy: exp2([1,2,3], dtype=f32)=[2,4,8].
            AssertSingleExp2(np.exp2(np.array(new int[] { 1, 2, 3 }), NPTypeCode.Single), 2f, 4f, 8f);
            AssertSingleExp2(np.exp2(np.array(new double[] { 1, 2, 3 }), typeof(float)), 2f, 4f, 8f);
        }

        [TestMethod]
        public void Exp2_SingleOutput_StridedView()
        {
            // The kernel must also be correct on a non-contiguous float32 input (every-other element).
            var a = np.array(new float[] { 1, 99, 2, 99, 3, 99 });
            AssertSingleExp2(np.exp2(a["::2"]), 2f, 4f, 8f);   // exp2 of [1, 2, 3]
        }

        [TestMethod]
        public void Exp2_DoubleAndHalfOutput_StillCorrect()
        {
            // The unaffected branches: int32->float64 and int8->float16 must remain correct.
            var dbl = np.exp2(np.array(new int[] { 1, 2, 3 }));
            dbl.typecode.Should().Be(NPTypeCode.Double);
            dbl.GetDouble(0).Should().Be(2.0);
            dbl.GetDouble(1).Should().Be(4.0);
            dbl.GetDouble(2).Should().Be(8.0);

            var half = np.exp2(np.array(new sbyte[] { 1, 2, 3 }));
            half.typecode.Should().Be(NPTypeCode.Half);
            ((double)half.GetAtIndex<Half>(0)).Should().Be(2.0);
            ((double)half.GetAtIndex<Half>(2)).Should().Be(8.0);
        }

        #endregion
    }
}
