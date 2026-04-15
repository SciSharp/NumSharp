using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;

namespace NumSharp.UnitTest.NewDtypes
{
    /// <summary>
    /// Arithmetic operation tests for SByte (int8), Half (float16), Complex (complex128)
    /// All expected values verified against NumPy 2.x
    /// </summary>
    [TestClass]
    public class NewDtypesArithmeticTests
    {
        #region SByte (int8) Arithmetic

        [TestMethod]
        public void SByte_Add()
        {
            // NumPy: np.array([-128, -1, 0, 1, 127], dtype=np.int8) + np.array([1, 2, 3, 4, 5], dtype=np.int8)
            // Result: [-127, 1, 3, 5, -124] (overflow at 127+5)
            var a = np.array(new sbyte[] { -128, -1, 0, 1, 127 });
            var b = np.array(new sbyte[] { 1, 2, 3, 4, 5 });
            var result = a + b;

            result.typecode.Should().Be(NPTypeCode.SByte);
            result.GetAtIndex<sbyte>(0).Should().Be((sbyte)-127);
            result.GetAtIndex<sbyte>(1).Should().Be((sbyte)1);
            result.GetAtIndex<sbyte>(2).Should().Be((sbyte)3);
            result.GetAtIndex<sbyte>(3).Should().Be((sbyte)5);
            result.GetAtIndex<sbyte>(4).Should().Be((sbyte)-124); // overflow
        }

        [TestMethod]
        public void SByte_Multiply_Scalar()
        {
            // NumPy: np.array([-128, -1, 0, 1, 127], dtype=np.int8) * 2
            // Result: [0, -2, 0, 2, -2] (overflow)
            var a = np.array(new sbyte[] { -128, -1, 0, 1, 127 });
            var result = a * 2;

            result.typecode.Should().Be(NPTypeCode.SByte);
            result.GetAtIndex<sbyte>(0).Should().Be((sbyte)0);    // -256 overflows to 0
            result.GetAtIndex<sbyte>(1).Should().Be((sbyte)-2);
            result.GetAtIndex<sbyte>(2).Should().Be((sbyte)0);
            result.GetAtIndex<sbyte>(3).Should().Be((sbyte)2);
            result.GetAtIndex<sbyte>(4).Should().Be((sbyte)-2);   // 254 overflows to -2
        }

        [TestMethod]
        public void SByte_Negate()
        {
            // NumPy: -np.array([-128, -1, 0, 1, 127], dtype=np.int8)
            // Result: [-128, 1, 0, -1, -127]
            var a = np.array(new sbyte[] { -128, -1, 0, 1, 127 });
            var result = -a;

            result.typecode.Should().Be(NPTypeCode.SByte);
            result.GetAtIndex<sbyte>(0).Should().Be((sbyte)-128); // -(-128) overflows back to -128
            result.GetAtIndex<sbyte>(1).Should().Be((sbyte)1);
            result.GetAtIndex<sbyte>(2).Should().Be((sbyte)0);
            result.GetAtIndex<sbyte>(3).Should().Be((sbyte)-1);
            result.GetAtIndex<sbyte>(4).Should().Be((sbyte)-127);
        }

        #endregion

        #region Half (float16) Arithmetic

        [TestMethod]
        public void Half_Add()
        {
            // NumPy: np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float16) + same
            // Result: [2, 4, 6, 8, 10]
            var h = np.array(new Half[] { (Half)1.0, (Half)2.0, (Half)3.0, (Half)4.0, (Half)5.0 });
            var result = h + h;

            result.typecode.Should().Be(NPTypeCode.Half);
            result.GetAtIndex<Half>(0).Should().Be((Half)2.0);
            result.GetAtIndex<Half>(1).Should().Be((Half)4.0);
            result.GetAtIndex<Half>(2).Should().Be((Half)6.0);
            result.GetAtIndex<Half>(3).Should().Be((Half)8.0);
            result.GetAtIndex<Half>(4).Should().Be((Half)10.0);
        }

        [TestMethod]
        public void Half_Multiply_Scalar()
        {
            // NumPy: np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float16) * 2
            // Result: [2, 4, 6, 8, 10]
            var h = np.array(new Half[] { (Half)1.0, (Half)2.0, (Half)3.0, (Half)4.0, (Half)5.0 });
            var result = h * 2;

            result.typecode.Should().Be(NPTypeCode.Half);
            result.GetAtIndex<Half>(0).Should().Be((Half)2.0);
            result.GetAtIndex<Half>(4).Should().Be((Half)10.0);
        }

        [TestMethod]
        public void Half_Divide_Scalar()
        {
            // NumPy: np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float16) / 2
            // Result: [0.5, 1.0, 1.5, 2.0, 2.5]
            var h = np.array(new Half[] { (Half)1.0, (Half)2.0, (Half)3.0, (Half)4.0, (Half)5.0 });
            var result = h / 2;

            result.typecode.Should().Be(NPTypeCode.Half);
            result.GetAtIndex<Half>(0).Should().Be((Half)0.5);
            result.GetAtIndex<Half>(1).Should().Be((Half)1.0);
            result.GetAtIndex<Half>(2).Should().Be((Half)1.5);
        }

        #endregion

        #region Complex (complex128) Arithmetic

        [TestMethod]
        public void Complex_Add()
        {
            // NumPy: z + z2 where z=[1+2j, 3+4j, 0+0j, -1-1j], z2=[1+0j, 0+1j, 1+1j, 2+2j]
            // Result: [2+2j, 3+5j, 1+1j, 1+1j]
            var z = np.array(new Complex[] { new(1, 2), new(3, 4), new(0, 0), new(-1, -1) });
            var z2 = np.array(new Complex[] { new(1, 0), new(0, 1), new(1, 1), new(2, 2) });
            var result = z + z2;

            result.typecode.Should().Be(NPTypeCode.Complex);
            result.GetAtIndex<Complex>(0).Should().Be(new Complex(2, 2));
            result.GetAtIndex<Complex>(1).Should().Be(new Complex(3, 5));
            result.GetAtIndex<Complex>(2).Should().Be(new Complex(1, 1));
            result.GetAtIndex<Complex>(3).Should().Be(new Complex(1, 1));
        }

        [TestMethod]
        [OpenBugs] // Complex multiply not supported in IL kernel yet
        public void Complex_Multiply()
        {
            // NumPy: z * z2 where z=[1+2j, 3+4j, 0+0j, -1-1j], z2=[1+0j, 0+1j, 1+1j, 2+2j]
            // Result: [1+2j, -4+3j, 0+0j, 0-4j]
            var z = np.array(new Complex[] { new(1, 2), new(3, 4), new(0, 0), new(-1, -1) });
            var z2 = np.array(new Complex[] { new(1, 0), new(0, 1), new(1, 1), new(2, 2) });
            var result = z * z2;

            result.typecode.Should().Be(NPTypeCode.Complex);
            result.GetAtIndex<Complex>(0).Should().Be(new Complex(1, 2));
            result.GetAtIndex<Complex>(1).Should().Be(new Complex(-4, 3));
            result.GetAtIndex<Complex>(2).Should().Be(new Complex(0, 0));
            result.GetAtIndex<Complex>(3).Should().Be(new Complex(0, -4));
        }

        [TestMethod]
        [OpenBugs] // Complex multiply not supported in IL kernel yet
        public void Complex_Multiply_Scalar()
        {
            // NumPy: np.array([1+2j, 3+4j, 0+0j, -1-1j]) * 2
            // Result: [2+4j, 6+8j, 0+0j, -2-2j]
            var z = np.array(new Complex[] { new(1, 2), new(3, 4), new(0, 0), new(-1, -1) });
            var result = z * 2;

            result.typecode.Should().Be(NPTypeCode.Complex);
            result.GetAtIndex<Complex>(0).Should().Be(new Complex(2, 4));
            result.GetAtIndex<Complex>(1).Should().Be(new Complex(6, 8));
            result.GetAtIndex<Complex>(2).Should().Be(new Complex(0, 0));
            result.GetAtIndex<Complex>(3).Should().Be(new Complex(-2, -2));
        }

        [TestMethod]
        public void Complex_Divide_Scalar()
        {
            // NumPy: np.array([1+2j, 3+4j, 0+0j, -1-1j]) / 2
            // Result: [0.5+1j, 1.5+2j, 0+0j, -0.5-0.5j]
            var z = np.array(new Complex[] { new(1, 2), new(3, 4), new(0, 0), new(-1, -1) });
            var result = z / 2;

            result.typecode.Should().Be(NPTypeCode.Complex);
            result.GetAtIndex<Complex>(0).Should().Be(new Complex(0.5, 1));
            result.GetAtIndex<Complex>(1).Should().Be(new Complex(1.5, 2));
            result.GetAtIndex<Complex>(2).Should().Be(new Complex(0, 0));
            result.GetAtIndex<Complex>(3).Should().Be(new Complex(-0.5, -0.5));
        }

        [TestMethod]
        [OpenBugs] // Complex negate not fully supported in IL kernel yet
        public void Complex_Negate()
        {
            // NumPy: -np.array([1+2j, 3+4j, 0+0j, -1-1j])
            // Result: [-1-2j, -3-4j, -0-0j, 1+1j]
            var z = np.array(new Complex[] { new(1, 2), new(3, 4), new(0, 0), new(-1, -1) });
            var result = -z;

            result.typecode.Should().Be(NPTypeCode.Complex);
            result.GetAtIndex<Complex>(0).Should().Be(new Complex(-1, -2));
            result.GetAtIndex<Complex>(1).Should().Be(new Complex(-3, -4));
            result.GetAtIndex<Complex>(3).Should().Be(new Complex(1, 1));
        }

        #endregion
    }
}
