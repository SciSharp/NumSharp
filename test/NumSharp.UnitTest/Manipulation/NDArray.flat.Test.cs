using System;
using System.Linq;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;
using NumSharp.UnitTest.Utilities;

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
            flat.ndim.Should().Be(0);
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
            flat.ndim.Should().Be(0);
            flat.Shape.Should().BeScalar();
            flat.Cast<int>().Should().AllBeEquivalentTo(1);
        }

        [TestMethod]
        public void flat_broadcasted_Case1()
        {
            var a = np.arange(4 * 1 * 1 * 1).reshape(4, 1, 1, 1)["3, :"];
            var b = np.arange(4 * 1 * 10 * 1).reshape(4, 1, 10, 1)["3, :"];

            (a, b) = np.broadcast_arrays(a, b);

            a.Should().BeBroadcasted().And.BeShaped(1, 10, 1);
            b.Should().BeBroadcasted().And.BeShaped(1, 10, 1);
            a.flat.Should().BeShaped(10).And.AllValuesBe(3);
            b.flat.Should().BeShaped(10).And.BeOfValues(30, 31, 32, 33, 34, 35, 36, 37, 38, 39);
        }

        [TestMethod]
        public void flat_broadcasted_Case2()
        {
            var a = np.arange(2 * 1 * 3).reshape((2, 1, 3)); //0, 1
            var b = np.arange(2 * 3 * 3).reshape((2, 3, 3)); //0, 1

            a = a["-1"];
            b = b["-1"];

            (a, b) = np.broadcast_arrays(a, b);

            a.Should().BeOfValues(3, 4, 5, 3, 4, 5, 3, 4, 5).And.BeShaped(3, 3);
            b.Should().BeOfValues(9, 10, 11, 12, 13, 14, 15, 16, 17).And.BeShaped(3, 3);
            var aflat = a.flat;
            var bflat = b.flat;
            aflat.Should().BeOfValues(3, 4, 5, 3, 4, 5, 3, 4, 5).And.BeShaped(3 * 3);
            bflat.Should().BeOfValues(9, 10, 11, 12, 13, 14, 15, 16, 17).And.BeShaped(3 * 3);
        }

        [TestMethod]
        public void flat_broadcasted_Case3()
        {
            var a = np.arange(3 * 1 * 3 * 3).reshape((3, 1, 3, 3)); //0, 1

            var b = a["-1, :, 1, :"];
            b.GetValue(0, 0).Should().Be(21);
            //or b = a[Slice.Index(-1), Slice.All, Slice.Index(1), Slice.All];

            b.Should().BeShaped(1, 3).And.BeOfValues(21, 22, 23);
        }
    }
}
