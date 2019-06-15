using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Logic
{
    [TestClass]
    public class np_isscalar_tests
    {
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
        public void AllPrimitiveTypes(Type type)
        {
            var value = Convert.ChangeType((byte)0, type);
            Assert.IsTrue(np.isscalar(value));
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
        public void AllPrimitiveArrayTypes(Type type)
        {
            var value = Convert.ChangeType((byte)0, type);
            var arr = Array.CreateInstance(type, 1);
            arr.SetValue(value, 0);
            Assert.IsFalse(np.isscalar(arr));
        }

        [TestMethod]
        public void Complex()
        {
            var value = new Complex(15, 15);
            Assert.IsTrue(np.isscalar(value));
        }

        [TestMethod]
        public void Null()
        {
            Assert.IsFalse(np.isscalar(null));
        }

        [DataTestMethod]
        [DataRow("")]
        [DataRow("Hi")]
        public void String(string value)
        {
            Assert.IsTrue(np.isscalar(value));
        }

        [TestMethod]
        public void NDArray()
        {
            var value = np.zeros(3, 3);
            Assert.IsFalse(np.isscalar(value));
            NDArray nd = 1d;
            Assert.IsTrue(np.isscalar(nd));
        }
    }
}
