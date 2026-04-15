using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Numerics;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using NumSharp;
using NumSharp.Generic;

namespace NumSharp.UnitTest.Generic
{
    [TestClass]
    public class NDArrayGenericTest
    {
        [TestMethod]
        public void Generic1DBool_NDArray()
        {
            var np1 = new NDArray<bool>(new[] {true, true, false, false}, new Shape(4));
            var np2 = new NDArray<bool>(new Shape(2));
            var np3 = new NDArray<bool>();
            Assert.IsTrue(Enumerable.SequenceEqual(new[] {true, true, false, false}, np1.Data<bool>()));
            Assert.AreEqual(4, np1.size);
            Assert.AreEqual(1, np1.ndim);
            Assert.IsTrue(Enumerable.SequenceEqual(new[] {false, false}, np2.Data<bool>()));
            Assert.AreEqual(2, np2.size);
            Assert.AreEqual(1, np2.ndim);
        }

        [TestMethod]
        public void Generic2DBool_NDArrayOR()
        {
            var np1 = new NDArray<bool>(new Shape(2, 3));
            np1.ReplaceData(new bool[] {true, true, false, false, true, false});
            Assert.IsTrue(Enumerable.SequenceEqual(new[] {true, true, false, false, true, false}, np1.Data<bool>()));
            Assert.AreEqual(6, np1.size);
            Assert.AreEqual(2, np1.ndim);
        }
    }
}
