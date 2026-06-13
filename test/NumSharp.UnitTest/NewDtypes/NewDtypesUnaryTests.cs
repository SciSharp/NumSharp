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

        #endregion
    }
}
