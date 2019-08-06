using System;
using System.Linq;
using System.Numerics;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;

namespace NumSharp.UnitTest.Creation
{
    [TestClass]
    public class np_full_Test
    {
        [TestMethod]
        public void Full_Like()
        {
            var a = np.zeros((10, 10));
            var alike = np.full_like(a, 5d);
            alike.Shape.size.Should().Be(100);
            alike.shape.Should().ContainInOrder(10, 10);
            alike.Array.GetIndex(0).Should().Be(5).And.BeOfType<double>();
        }

        [TestMethod]
        public void SimpleInt1D()
        {
            var np1 = np.full(1d, new Shape(5));

            Assert.IsTrue(np1.Data<double>().Where(x => x == 1).ToArray().Length == 5);
        }

        [TestMethod]
        public void SimpleInt2D()
        {
            var np1 = np.full(1d, new Shape(5, 5));

            Assert.IsTrue(np1.Data<double>().Where(x => x == 1).ToArray().Length == 25);
        }

        [TestMethod]
        public void SimpleDouble3D()
        {
            var np1 = np.full(1d, new Shape(5, 5, 5));

            Assert.IsTrue(np1.Data<double>().Where(x => x == 1).ToArray().Length == 125);
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
        public void Full_AllTypes(Type dtype)
        {
            var np1 = np.full((ValueType) Activator.CreateInstance(dtype), new Shape(3, 3, 3), dtype);
            np1.dtype.Should().Be(dtype);
            np1.Array.GetIndex(0).GetType().Should().Be(dtype);
        }
    }
}
