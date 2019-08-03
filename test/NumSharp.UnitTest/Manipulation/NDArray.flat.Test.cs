using System;
using System.Linq;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;

namespace NumSharp.UnitTest.Manipulation
{
    [TestClass]
    public class NDArray_flat_Test
    {
        [TestMethod]
        public void flat_3_3()
        {
            var nd = np.full(5, (3, 3), NPTypeCode.Int32);
            var flat = nd.flat;
            Console.WriteLine((string)flat);
            flat.size.Should().Be(9);
            flat.ndim.Should().Be(1);
            flat.shape[0].Should().Be(9);
            flat.Cast<int>().Should().AllBeEquivalentTo(5);
        }

        [TestMethod]
        public void flat_3_3_sliced()
        {
            var nd = np.full(5, (3, 3), NPTypeCode.Int32);
            var sliced = nd["0,:"];
            var flat = sliced.flat;
            Console.WriteLine((string)flat);
            flat.size.Should().Be(3);
            flat.ndim.Should().Be(1);
            flat.shape[0].Should().Be(3);
            flat.Cast<int>().Should().AllBeEquivalentTo(5);
        }

        [TestMethod]
        public void flat_scalar_sliced()
        {
            var nd = NDArray.Scalar(1);
            var sliced = nd[":"];
            var flat = sliced.flat;
            Console.WriteLine((string)flat);
            flat.size.Should().Be(1);
            flat.ndim.Should().Be(1);
            flat.shape[0].Should().Be(1);
            flat.Cast<int>().Should().AllBeEquivalentTo(1);
        }

        [TestMethod]
        public void flat_1_3_1_3()
        {
            var nd = np.full(5, (1, 3, 1, 3), NPTypeCode.Int32);
            var flat = nd.flat;
            Console.WriteLine((string)flat);
            flat.size.Should().Be(9);
            flat.ndim.Should().Be(1);
            flat.shape[0].Should().Be(9);
            flat.Cast<int>().Should().AllBeEquivalentTo(5);
        }

        [TestMethod]
        public void flat_2_3_1_3_sliced()
        {
            var nd = np.full(5, (2, 3, 1, 3), NPTypeCode.Int32);
            var sliced = nd["0,:"];
            var flat = sliced.flat;
            Console.WriteLine((string)flat);
            flat.size.Should().Be(9);
            flat.ndim.Should().Be(1);
            flat.shape[0].Should().Be(9);
            flat.Cast<int>().Should().AllBeEquivalentTo(5);
        }

        [TestMethod]
        public void flat_3()
        {
            var nd = np.full(5, Shape.Vector(3), NPTypeCode.Int32);
            var flat = nd.flat;
            Console.WriteLine((string)flat);
            flat.size.Should().Be(3);
            flat.ndim.Should().Be(1);
            flat.shape[0].Should().Be(3);
            flat.Cast<int>().Should().AllBeEquivalentTo(5);
        }

        [TestMethod]
        public void flat_scalar()
        {
            var nd = NDArray.Scalar(1);
            var flat = nd.flat;
            Console.WriteLine((string)flat);
            flat.size.Should().Be(1);
            flat.ndim.Should().Be(1);
            flat.shape[0].Should().Be(1);
            flat.Cast<int>().Should().AllBeEquivalentTo(1);
        }
    }
}
