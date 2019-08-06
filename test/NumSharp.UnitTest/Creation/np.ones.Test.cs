using System;
using System.Linq;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Creation
{
    [TestClass]
    public class np_ones_Test
    {
        [TestMethod]
        public void SimpleInt1D()
        {
            var np1 = np.ones(new Shape(5));

            Assert.IsTrue(np1.Data<double>().Where(x => x == 1).ToArray().Length == 5);
        }

        [TestMethod]
        public void SimpleInt2D()
        {
            var np1 = np.ones(new Shape(5, 5));

            Assert.IsTrue(np1.Data<double>().Where(x => x == 1).ToArray().Length == 25);
        }

        [TestMethod]
        public void SimpleDouble3D()
        {
            var np1 = np.ones(new Shape(5, 5, 5));

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
        public void One_AllTypes(Type dtype)
        {
            var np1 = np.ones(new Shape(3, 3, 3), dtype);
            Assert.IsTrue(np1.dtype == dtype);
            Assert.IsTrue(np1.Array.GetIndex(0).GetType() == dtype);
        }
    }
}
