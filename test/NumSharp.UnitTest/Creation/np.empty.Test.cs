using System;
using System.Linq;
using System.Numerics;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;

namespace NumSharp.UnitTest.Creation
{
    [TestClass]
    public class np_empty_Test
    {

        [TestMethod]
        public void Empty_Like()
        {
            var a = np.zeros((10, 10));
            var alike = np.empty_like(a);
            alike.Shape.size.Should().Be(100);
            alike.shape.Should().ContainInOrder(10, 10);
            alike.Array.GetIndex(0).GetType().Should().Be<double>();
        }

        [TestMethod]
        public void SimpleInt1D()
        {
            var np1 = np.empty(new Shape(5));
            np1.size.Should().Be(5);
            np1[2] = 2;
            np1.GetAtIndex<double>(2).Should().Be(2d);
            np1.Array.GetIndex(0).GetType().Should().Be<double>();
        }


        [TestMethod]
        public void SimpleDouble3D()
        {
            var np1 = np.empty(new Shape(5, 5, 5));
            np1.size.Should().Be(5 * 5 * 5);
            np1.dtype.Should().Be<double>();
        }

        [DataTestMethod]
        [DataRow(typeof(double))]
        [DataRow(typeof(float))]
        [DataRow(typeof(byte))]
        [DataRow(typeof(int))]
        [DataRow(typeof(long))]
        [DataRow(typeof(char))]
        [DataRow(typeof(short))]
        [DataRow(typeof(uint))]
        [DataRow(typeof(ulong))]
        [DataRow(typeof(ushort))]
        [DataRow(typeof(decimal))]
        //TODO! [DataRow(typeof(Complex))]
        [DataRow(typeof(bool))]
        public void Empty_AllTypes(Type dtype)
        {
            var np1 = np.empty(new Shape(3, 3, 3), dtype);
            np1.dtype.Should().Be(dtype);
            np1.Array.GetIndex(0).GetType().Should().Be(dtype);
        }
    }
}
