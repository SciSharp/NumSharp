using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using NumSharp.Core;

namespace NumSharp.UnitTest.Extensions
{
    [TestClass]
    public class NdArrayZerosTest
    {
        [TestMethod]
        public void Zeros1Dim()
        {
            var np = new NumPyGeneric<int>();
            var n = np.zeros(3);
            Assert.IsTrue(Enumerable.SequenceEqual(n.Data, new int[] { 0, 0, 0 }));
        }

        [TestMethod]
        public void Zeros2Dim()
        {
            var np = new NumPyGeneric<int>();
            var n = np.zeros(3, 2);
            Assert.IsTrue(Enumerable.SequenceEqual(n.Data, new int[] { 0, 0, 0, 0, 0, 0 }));
        }
    }
}
