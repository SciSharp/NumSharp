using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;

namespace NumSharp.UnitTest.NewDtypes
{
    /// <summary>
    /// Type promotion tests for SByte (int8), Half (float16), Complex (complex128)
    /// All expected values verified against NumPy 2.x
    /// </summary>
    [TestClass]
    public class NewDtypesTypePromotionTests
    {
        #region SByte + Other Types

        [TestMethod]
        public void SByte_Plus_Half_PromotesToHalf()
        {
            // NumPy: int8 + float16 = float16
            var a = np.array(new sbyte[] { 1, 2, 3 });
            var b = np.array(new Half[] { (Half)0.5, (Half)1.5, (Half)2.5 });
            var result = a + b;

            result.typecode.Should().Be(NPTypeCode.Half);
            result.GetAtIndex<Half>(0).Should().Be((Half)1.5);
            result.GetAtIndex<Half>(1).Should().Be((Half)3.5);
            result.GetAtIndex<Half>(2).Should().Be((Half)5.5);
        }

        [TestMethod]
        public void SByte_Plus_Complex_PromotesToComplex()
        {
            // NumPy: int8 + complex128 = complex128
            var a = np.array(new sbyte[] { 1, 2, 3 });
            var c = np.array(new Complex[] { new(1, 0), new(2, 0), new(3, 0) });
            var result = a + c;

            result.typecode.Should().Be(NPTypeCode.Complex);
            result.GetAtIndex<Complex>(0).Should().Be(new Complex(2, 0));
            result.GetAtIndex<Complex>(1).Should().Be(new Complex(4, 0));
            result.GetAtIndex<Complex>(2).Should().Be(new Complex(6, 0));
        }

        [TestMethod]
        public void SByte_Plus_IntScalar_StaysSByte()
        {
            // NumPy: int8 + int scalar = int8
            var a = np.array(new sbyte[] { 1, 2, 3 });
            var result = a + 1;

            result.typecode.Should().Be(NPTypeCode.SByte);
            result.GetAtIndex<sbyte>(0).Should().Be((sbyte)2);
            result.GetAtIndex<sbyte>(1).Should().Be((sbyte)3);
            result.GetAtIndex<sbyte>(2).Should().Be((sbyte)4);
        }

        [TestMethod]
        public void SByte_Plus_FloatScalar_PromotesToFloat64()
        {
            // NumPy: int8 + float scalar = float64
            var a = np.array(new sbyte[] { 1, 2, 3 });
            var result = a + 1.0;

            result.typecode.Should().Be(NPTypeCode.Double);
            result.GetAtIndex<double>(0).Should().Be(2.0);
            result.GetAtIndex<double>(1).Should().Be(3.0);
            result.GetAtIndex<double>(2).Should().Be(4.0);
        }

        #endregion

        #region Half + Other Types

        [TestMethod]
        [OpenBugs] // Half + Complex type promotion not fully supported yet
        public void Half_Plus_Complex_PromotesToComplex()
        {
            // NumPy: float16 + complex128 = complex128
            var h = np.array(new Half[] { (Half)1.0, (Half)2.0, (Half)3.0 });
            var c = np.array(new Complex[] { new(1, 1), new(2, 2), new(3, 3) });
            var result = h + c;

            result.typecode.Should().Be(NPTypeCode.Complex);
            result.GetAtIndex<Complex>(0).Should().Be(new Complex(2, 1));
            result.GetAtIndex<Complex>(1).Should().Be(new Complex(4, 2));
            result.GetAtIndex<Complex>(2).Should().Be(new Complex(6, 3));
        }

        [TestMethod]
        public void Half_Plus_IntScalar_StaysHalf()
        {
            // NumPy: float16 + int scalar = float16
            var h = np.array(new Half[] { (Half)1.0, (Half)2.0, (Half)3.0 });
            var result = h + 1;

            result.typecode.Should().Be(NPTypeCode.Half);
            result.GetAtIndex<Half>(0).Should().Be((Half)2.0);
            result.GetAtIndex<Half>(1).Should().Be((Half)3.0);
            result.GetAtIndex<Half>(2).Should().Be((Half)4.0);
        }

        #endregion

        #region Complex + Other Types

        [TestMethod]
        public void Complex_Plus_IntScalar_StaysComplex()
        {
            // NumPy: complex128 + int scalar = complex128
            var c = np.array(new Complex[] { new(1, 1), new(2, 2), new(3, 3) });
            var result = c + 1;

            result.typecode.Should().Be(NPTypeCode.Complex);
            result.GetAtIndex<Complex>(0).Should().Be(new Complex(2, 1));
            result.GetAtIndex<Complex>(1).Should().Be(new Complex(3, 2));
            result.GetAtIndex<Complex>(2).Should().Be(new Complex(4, 3));
        }

        #endregion
    }
}
