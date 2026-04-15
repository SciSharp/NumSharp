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
        [OpenBugs] // Sqrt not supported for Half yet
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
        [OpenBugs] // Floor not supported for Half yet
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
        [OpenBugs] // Ceil not supported for Half yet
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
        [OpenBugs] // Exp not supported for Half yet
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
        [OpenBugs] // Sin not supported for Half yet
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
        [OpenBugs] // Sqrt not supported for Complex yet
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
        [OpenBugs] // Exp not supported for Complex yet
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
        [OpenBugs] // Log not supported for Complex yet
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

        #endregion
    }
}
