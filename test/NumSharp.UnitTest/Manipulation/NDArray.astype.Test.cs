using System;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;

namespace NumSharp.UnitTest.Manipulation
{
    [TestClass]
    public class astypeTests
    {
        [TestMethod]
        public void Upcasting()
        {
            var nd = np.ones(np.int32, 3, 3);
            var int64_copied = nd.astype(np.int64, true);
            var int64 = nd.astype(np.int64, false);

            //test copying
            Assert.IsTrue(int64_copied.Array != nd.Array);
            Assert.IsTrue(int64.Array == nd.Array);
            int64_copied.GetTypeCode.Should().Be(NPTypeCode.Int64);
            int64.GetTypeCode.Should().Be(NPTypeCode.Int64);
        }

        [TestMethod]
        public void UpcastingByteToLong()
        {
            var nd = np.ones(np.uint8, 3, 3);
            var int64_copied = nd.astype(np.int64, true);
            var int64 = nd.astype(np.int64, false);

            //test copying
            Assert.IsTrue(int64_copied.Array != nd.Array);
            Assert.IsTrue(int64.Array == nd.Array);
            int64_copied.GetTypeCode.Should().Be(NPTypeCode.Int64);
            int64.GetTypeCode.Should().Be(NPTypeCode.Int64);
        }

        [TestMethod]
        public void UpcastingCharsToLong()
        {
            var nd = np.ones(np.@char, 3, 3);
            var int64_copied = nd.astype(np.int64, true);
            var int64 = nd.astype(np.int64, false);

            //test copying
            Assert.IsTrue(int64_copied.Array != nd.Array);
            Assert.IsTrue(int64.Array == nd.Array);
            int64_copied.GetTypeCode.Should().Be(NPTypeCode.Int64);
            int64.GetTypeCode.Should().Be(NPTypeCode.Int64);
        }

        [TestMethod]
        public void DowncastingIntToShort()
        {
            var nd = np.ones(np.int32, 3, 3);
            var int16_copied = nd.astype(np.int16, true);
            var int16 = nd.astype(np.int16, false);

            //test copying
            Assert.IsTrue(int16_copied.Array != nd.Array);
            Assert.IsTrue(int16.Array == nd.Array);
            int16_copied.GetTypeCode.Should().Be(NPTypeCode.Int16);
            int16.GetTypeCode.Should().Be(NPTypeCode.Int16);
        }

        [TestMethod]
        public void DowncastingDoubleToInt()
        {
            var nd = np.ones(np.float64, 3, 3);
            for (int i = 0; i < nd.size; i++)
            {
                nd.SetAtIndex(1.3d, i);
            }

            var int32_copied = nd.astype(np.int32, true);
            var int32 = nd.astype(np.int32, false);

            //test copying
            Assert.IsTrue(int32_copied.Array != nd.Array);
            Assert.IsTrue(int32.Array == nd.Array);
            int32_copied.GetTypeCode.Should().Be(NPTypeCode.Int32);
            int32.GetTypeCode.Should().Be(NPTypeCode.Int32);

            for (int i = 0; i < nd.size; i++)
            {
                nd.GetAtIndex<int>(i).Should().Be(1);
            }
        }

        [TestMethod]
        public void DowncastingIntToUShort()
        {
            var nd = np.ones(np.int32, 3, 3);
            nd[2, 2].Data<int>()[0].Should().Be(1);
            var int16_copied = nd.astype(np.uint16, true);
            var int16 = nd.astype(np.uint16, false);

            //test copying
            int16_copied.Array.Should().Equal(nd.Array);
            int16.Array.Should().Equal(nd.Array);
            int16_copied.GetTypeCode.Should().Be(NPTypeCode.UInt16);
            int16.GetTypeCode.Should().Be(NPTypeCode.UInt16);
        }

        [TestMethod]
        public void CastEmptyNDArray()
        {
            var nd = new NDArray(NPTypeCode.Int32);
            var int16_copied = nd.astype(np.uint16, true);
            int16_copied.Should().BeEquivalentTo(nd);
            int16_copied.Shape.IsEmpty.Should().BeTrue();
        }

        [TestMethod, Ignore("String dtype is not supported")]
        public void CastingStringToByte()
        {
            throw new NotSupportedException();
            //var nd = np.ones(np.chars, 3, 3);
            //nd[2, 2].Data<string>()[0].Should().Be("1");
            //var output_copied = nd.astype(np.uint8, true);
            //var output = nd.astype(np.uint8, false);

            ////test copying
            //output_copied.Array.Should().Equal(nd.Array);
            //output.Array.Should().Equal(nd.Array);
            //output_copied.Array.GetType().GetElementType().Should().Be<byte>();
            //output.Array.GetType().GetElementType().Should().Be<byte>();
        }

        [TestMethod, Ignore("String dtype is not supported")]
        public void CastingByteToString()
        {
            throw new NotSupportedException();
            //var nd = np.ones(np.uint8, 3, 3);
            //nd[2, 2].Data<byte>()[0].Should().Be(1);
            //var output_copied = nd.astype(np.chars, true);
            //var output = nd.astype(np.chars, false);

            //output_copied[2, 2].Data<string>()[0].Should().Be("1");

            ////test copying
            //output_copied.Array.Should().Equal(nd.Array);
            //output.Array.Should().Equal(nd.Array);
            //output_copied.Array.GetType().GetElementType().Should().Be<string>();
            //output.Array.GetType().GetElementType().Should().Be<string>();
        }


        [TestMethod, Ignore("Complex dtype is not supported yet")] //TODO!
        public void CastingIntToComplex()
        {
            //var nd = np.ones(np.int32, 3, 3);
            //nd[2, 2].Data<int>()[0].Should().Be(1);
            //var output_copied = nd.astype(np.complex128, true);
            //var output = nd.astype(np.complex128, false);
            //
            ////test copying
            //output_copied.Array.Should().Equal(nd.Array);
            //output.Array.Should().Equal(nd.Array);
            //output_copied.Array.GetType().GetElementType().Should().Be<Complex>();
            //output.Array.GetType().GetElementType().Should().Be<Complex>();
            //
            //output_copied[2, 2].GetComplex(0).Should().Be(new Complex(1, 0));
        }
    }
}
