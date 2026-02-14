using System;
using System.Linq;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;

namespace NumSharp.UnitTest.Creation
{
    public class np_full_Test
    {
        [Test]
        public void Full_Like()
        {
            var a = np.zeros((10, 10));
            var alike = np.full_like(a, 5d);
            alike.Shape.size.Should().Be(100);
            alike.shape.Should().ContainInOrder(10, 10);
            alike.Array.GetIndex(0).Should().Be(5).And.BeOfType<double>();
        }

        [Test]
        public void SimpleInt1D()
        {
            var np1 = np.full(1d, new Shape(5));

            Assert.IsTrue(np1.Data<double>().Where(x => x == 1).ToArray().Length == 5);
        }

        [Test]
        public void SimpleInt2D()
        {
            var np1 = np.full(1d, new Shape(5, 5));

            Assert.IsTrue(np1.Data<double>().Where(x => x == 1).ToArray().Length == 25);
        }

        [Test]
        public void SimpleDouble3D()
        {
            var np1 = np.full(1d, new Shape(5, 5, 5));

            Assert.IsTrue(np1.Data<double>().Where(x => x == 1).ToArray().Length == 125);
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
        public void Full_AllTypes(Type dtype)
        {
            var np1 = np.full((ValueType) Activator.CreateInstance(dtype), new Shape(3, 3, 3), dtype);
            np1.dtype.Should().Be(dtype);
            np1.Array.GetIndex(0).GetType().Should().Be(dtype);
        }
    }
}
