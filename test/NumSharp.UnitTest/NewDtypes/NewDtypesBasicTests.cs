using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;

namespace NumSharp.UnitTest.NewDtypes
{
    /// <summary>
    /// Basic tests for new dtype support: SByte (int8), Half (float16), Complex (complex128)
    /// </summary>
    [TestClass]
    public class NewDtypesBasicTests
    {
        [TestMethod]
        public void SByte_CreateArray()
        {
            // Create sbyte array
            var data = new sbyte[] { -128, -1, 0, 1, 127 };
            var arr = np.array(data);

            arr.dtype.Should().Be(typeof(sbyte));
            arr.typecode.Should().Be(NPTypeCode.SByte);
            arr.size.Should().Be(5);
            arr.GetAtIndex<sbyte>(0).Should().Be((sbyte)-128);
            arr.GetAtIndex<sbyte>(4).Should().Be((sbyte)127);
        }

        [TestMethod]
        public void SByte_Zeros()
        {
            var arr = np.zeros(new Shape(3, 3), NPTypeCode.SByte);

            arr.dtype.Should().Be(typeof(sbyte));
            arr.typecode.Should().Be(NPTypeCode.SByte);
            arr.size.Should().Be(9);
        }

        [TestMethod]
        public void Half_CreateArray()
        {
            // Create half array
            var data = new Half[] { (Half)0.0, (Half)1.0, (Half)(-1.0), Half.MaxValue, Half.MinValue };
            var arr = np.array(data);

            arr.dtype.Should().Be(typeof(Half));
            arr.typecode.Should().Be(NPTypeCode.Half);
            arr.size.Should().Be(5);
        }

        [TestMethod]
        public void Half_Zeros()
        {
            var arr = np.zeros(new Shape(3, 3), NPTypeCode.Half);

            arr.dtype.Should().Be(typeof(Half));
            arr.typecode.Should().Be(NPTypeCode.Half);
            arr.size.Should().Be(9);
        }

        [TestMethod]
        public void Complex_CreateArray()
        {
            // Create complex array
            var data = new Complex[] { new Complex(1, 2), new Complex(3, 4), Complex.Zero, Complex.One };
            var arr = np.array(data);

            arr.dtype.Should().Be(typeof(Complex));
            arr.typecode.Should().Be(NPTypeCode.Complex);
            arr.size.Should().Be(4);
        }

        [TestMethod]
        public void Complex_Zeros()
        {
            var arr = np.zeros(new Shape(3, 3), NPTypeCode.Complex);

            arr.dtype.Should().Be(typeof(Complex));
            arr.typecode.Should().Be(NPTypeCode.Complex);
            arr.size.Should().Be(9);
        }

        [TestMethod]
        public void NPTypeCode_SByte_Properties()
        {
            NPTypeCode.SByte.SizeOf().Should().Be(1);
            NPTypeCode.SByte.IsInteger().Should().BeTrue();
            NPTypeCode.SByte.IsSigned().Should().BeTrue();
            NPTypeCode.SByte.AsNumpyDtypeName().Should().Be("int8");
        }

        [TestMethod]
        public void NPTypeCode_Half_Properties()
        {
            NPTypeCode.Half.SizeOf().Should().Be(2);
            NPTypeCode.Half.IsFloatingPoint().Should().BeTrue();
            NPTypeCode.Half.IsRealNumber().Should().BeTrue();
            NPTypeCode.Half.AsNumpyDtypeName().Should().Be("float16");
        }

        [TestMethod]
        public void NPTypeCode_Complex_Properties()
        {
            NPTypeCode.Complex.SizeOf().Should().Be(16);
            NPTypeCode.Complex.IsRealNumber().Should().BeTrue();
            NPTypeCode.Complex.AsNumpyDtypeName().Should().Be("complex128");
        }

        [TestMethod]
        public void DType_Parsing_Int8()
        {
            var int8Dtype = np.dtype("int8");
            int8Dtype.typecode.Should().Be(NPTypeCode.SByte);
        }

        [TestMethod]
        public void DType_Parsing_Float16()
        {
            var float16Dtype = np.dtype("float16");
            float16Dtype.typecode.Should().Be(NPTypeCode.Half);
        }

        [TestMethod]
        public void DType_Parsing_Complex128()
        {
            var complex128Dtype = np.dtype("complex128");
            complex128Dtype.typecode.Should().Be(NPTypeCode.Complex);
        }

        #region np.ones with new dtypes

        [TestMethod]
        public void SByte_Ones()
        {
            // NumPy: np.ones(3, dtype=np.int8) -> [1, 1, 1]
            var arr = np.ones(3, typeof(sbyte));

            arr.dtype.Should().Be(typeof(sbyte));
            arr.typecode.Should().Be(NPTypeCode.SByte);
            arr.GetAtIndex<sbyte>(0).Should().Be((sbyte)1);
            arr.GetAtIndex<sbyte>(1).Should().Be((sbyte)1);
            arr.GetAtIndex<sbyte>(2).Should().Be((sbyte)1);
        }

        [TestMethod]
        public void Half_Ones()
        {
            // NumPy: np.ones(3, dtype=np.float16) -> [1., 1., 1.]
            var arr = np.ones(3, typeof(Half));

            arr.dtype.Should().Be(typeof(Half));
            arr.typecode.Should().Be(NPTypeCode.Half);
            arr.GetAtIndex<Half>(0).Should().Be((Half)1.0);
            arr.GetAtIndex<Half>(1).Should().Be((Half)1.0);
            arr.GetAtIndex<Half>(2).Should().Be((Half)1.0);
        }

        [TestMethod]
        public void Complex_Ones()
        {
            // NumPy: np.ones(3, dtype=np.complex128) -> [1.+0.j, 1.+0.j, 1.+0.j]
            var arr = np.ones(3, typeof(Complex));

            arr.dtype.Should().Be(typeof(Complex));
            arr.typecode.Should().Be(NPTypeCode.Complex);
            arr.GetAtIndex<Complex>(0).Should().Be(new Complex(1, 0));
            arr.GetAtIndex<Complex>(1).Should().Be(new Complex(1, 0));
            arr.GetAtIndex<Complex>(2).Should().Be(new Complex(1, 0));
        }

        #endregion

        #region np.full with new dtypes

        [TestMethod]
        public void SByte_Full()
        {
            // NumPy: np.full(3, -5, dtype=np.int8) -> [-5, -5, -5]
            var arr = np.full(3, (sbyte)-5);

            arr.dtype.Should().Be(typeof(sbyte));
            arr.typecode.Should().Be(NPTypeCode.SByte);
            arr.GetAtIndex<sbyte>(0).Should().Be((sbyte)-5);
            arr.GetAtIndex<sbyte>(1).Should().Be((sbyte)-5);
            arr.GetAtIndex<sbyte>(2).Should().Be((sbyte)-5);
        }

        [TestMethod]
        public void Half_Full()
        {
            // NumPy: np.full(3, 3.14, dtype=np.float16) -> [3.14, 3.14, 3.14]
            var arr = np.full(3, (Half)3.14);

            arr.dtype.Should().Be(typeof(Half));
            arr.typecode.Should().Be(NPTypeCode.Half);
            // Half has limited precision
            ((double)arr.GetAtIndex<Half>(0)).Should().BeApproximately(3.14, 0.01);
            ((double)arr.GetAtIndex<Half>(1)).Should().BeApproximately(3.14, 0.01);
            ((double)arr.GetAtIndex<Half>(2)).Should().BeApproximately(3.14, 0.01);
        }

        [TestMethod]
        public void Complex_Full()
        {
            // NumPy: np.full(3, 2+3j) -> [2.+3.j, 2.+3.j, 2.+3.j]
            var arr = np.full(3, new Complex(2, 3));

            arr.dtype.Should().Be(typeof(Complex));
            arr.typecode.Should().Be(NPTypeCode.Complex);
            arr.GetAtIndex<Complex>(0).Should().Be(new Complex(2, 3));
            arr.GetAtIndex<Complex>(1).Should().Be(new Complex(2, 3));
            arr.GetAtIndex<Complex>(2).Should().Be(new Complex(2, 3));
        }

        #endregion
    }
}
