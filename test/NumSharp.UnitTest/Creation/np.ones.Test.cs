using System;
using System.Linq;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Creation
{
    public class np_ones_Test
    {
        [Test]
        public void SimpleInt1D()
        {
            var np1 = np.ones(new Shape(5));

            Assert.IsTrue(np1.Data<double>().Where(x => x == 1).ToArray().Length == 5);
        }

        [Test]
        public void SimpleInt2D()
        {
            var np1 = np.ones(new Shape(5, 5));

            Assert.IsTrue(np1.Data<double>().Where(x => x == 1).ToArray().Length == 25);
        }

        [Test]
        public void SimpleDouble3D()
        {
            var np1 = np.ones(new Shape(5, 5, 5));

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
        public void One_AllTypes(Type dtype)
        {
            var np1 = np.ones(new Shape(3, 3, 3), dtype);
            Assert.IsTrue(np1.dtype == dtype);
            Assert.IsTrue(np1.Array.GetIndex(0).GetType() == dtype);
        }
    }
}
