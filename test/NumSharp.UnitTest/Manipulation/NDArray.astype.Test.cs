using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Numerics;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;
using FluentAssertions;
using NumSharp;

namespace NumSharp.UnitTest
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
            Assert.IsTrue(int64_copied.Array.GetType().GetElementType() == typeof(Int64));
            Assert.IsTrue(int64.Array.GetType().GetElementType() == typeof(Int64));
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
            Assert.IsTrue(int64_copied.Array.GetType().GetElementType() == typeof(Int64));
            Assert.IsTrue(int64.Array.GetType().GetElementType() == typeof(Int64));
        }

        [TestMethod]
        public void UpcastingCharsToLong()
        {
            var nd = np.ones(np.chars, 3, 3);
            var int64_copied = nd.astype(np.int64, true);
            var int64 = nd.astype(np.int64, false);

            //test copying
            Assert.IsTrue(int64_copied.Array != nd.Array);
            Assert.IsTrue(int64.Array == nd.Array);
            Assert.IsTrue(int64_copied.Array.GetType().GetElementType() == typeof(Int64));
            Assert.IsTrue(int64.Array.GetType().GetElementType() == typeof(Int64));
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
            Assert.IsTrue(int16_copied.Array.GetType().GetElementType() == typeof(Int16));
            Assert.IsTrue(int16.Array.GetType().GetElementType() == typeof(Int16));
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
            int16_copied.Array.GetType().GetElementType().Should().Be<UInt16>();
            int16.Array.GetType().GetElementType().Should().Be<UInt16>();
        }

        [TestMethod]
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

        [TestMethod]
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


        [TestMethod]
        public void CastingIntToComplex()
        {
            var nd = np.ones(np.int32, 3, 3);
            nd[2, 2].Data<int>()[0].Should().Be(1);
            var output_copied = nd.astype(np.complex128, true);
            var output = nd.astype(np.complex128, false);

            //test copying
            output_copied.Array.Should().Equal(nd.Array);
            output.Array.Should().Equal(nd.Array);
            output_copied.Array.GetType().GetElementType().Should().Be<Complex>();
            output.Array.GetType().GetElementType().Should().Be<Complex>();

            output_copied[2, 2].Data<Complex>(0).Should().Be(new Complex(1, 0));
        }
    }
}
