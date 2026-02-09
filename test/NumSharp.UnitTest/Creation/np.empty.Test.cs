using System;
using System.Linq;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;

namespace NumSharp.UnitTest.Creation
{
    public class np_empty_Test
    {

        [Test]
        public void Empty_Like()
        {
            var a = np.zeros((10, 10));
            var alike = np.empty_like(a);
            alike.Shape.size.Should().Be(100);
            alike.shape.Should().ContainInOrder(10, 10);
            alike.Array.GetIndex(0).GetType().Should().Be<double>();
        }

        [Test]
        public void SimpleInt1D()
        {
            var np1 = np.empty(new Shape(5));
            np1.size.Should().Be(5);
            np1[2] = 2;
            np1.GetAtIndex<double>(2).Should().Be(2d);
            np1.Array.GetIndex(0).GetType().Should().Be<double>();
        }


        [Test]
        public void SimpleDouble3D()
        {
            var np1 = np.empty(new Shape(5, 5, 5));
            np1.size.Should().Be(5 * 5 * 5);
            np1.dtype.Should().Be<double>();
        }

        [Test]
        [Arguments(typeof(double))]
        [Arguments(typeof(float))]
        [Arguments(typeof(byte))]
        [Arguments(typeof(int))]
        [Arguments(typeof(long))]
        [Arguments(typeof(char))]
        [Arguments(typeof(short))]
        [Arguments(typeof(uint))]
        [Arguments(typeof(ulong))]
        [Arguments(typeof(ushort))]
        [Arguments(typeof(decimal))]
        //TODO! [Arguments(typeof(Complex))]
        [Arguments(typeof(bool))]
        public void Empty_AllTypes(Type dtype)
        {
            var np1 = np.empty(new Shape(3, 3, 3), dtype);
            np1.dtype.Should().Be(dtype);
            np1.Array.GetIndex(0).GetType().Should().Be(dtype);
        }
    }
}
