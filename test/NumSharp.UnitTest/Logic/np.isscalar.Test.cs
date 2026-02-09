using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Logic
{
    public class np_isscalar_tests
    {
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
        public void AllPrimitiveTypes(Type type)
        {
            var value = Convert.ChangeType((byte)0, type);
            Assert.IsTrue(np.isscalar(value));
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
        public void AllPrimitiveArrayTypes(Type type)
        {
            var value = Convert.ChangeType((byte)0, type);
            var arr = Array.CreateInstance(type, 1);
            arr.SetValue(value, 0);
            Assert.IsFalse(np.isscalar(arr));
        }

        [Test]
        public void Complex()
        {
            var value = new Complex(15, 15);
            Assert.IsTrue(np.isscalar(value));
        }

        [Test]
        public void Null()
        {
            Assert.IsFalse(np.isscalar(null));
        }

        [Test]
        [Arguments("")]
        [Arguments("Hi")]
        public void String(string value)
        {
            Assert.IsTrue(np.isscalar(value));
        }

        [Test]
        public void NDArray()
        {
            var value = np.zeros(3, 3);
            Assert.IsFalse(np.isscalar(value));
            NDArray nd = 1d;
            Assert.IsTrue(np.isscalar(nd));
        }
    }
}
