using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using NumSharp.Core;

namespace NumSharp.UnitTest.Extensions
{
    [TestClass]
    public class NdArrayZerosTest : TestBase
    {
        [TestMethod]
        public void Zeros1Dim()
        {
            var n = np.zeros(3);
            Assert.IsTrue(Enumerable.SequenceEqual(n.float64, new double[] { 0, 0, 0 }));
        }

        [TestMethod]
        public void Zeros2Dim()
        {
            var n = np.zeros(3, 2);
            Assert.IsTrue(Enumerable.SequenceEqual(n.float64, new double[] { 0, 0, 0, 0, 0, 0 }));
        }

        [TestMethod]
        public void Zeros1DimWithDtype()
        {
            var n = np.zeros(new Shape(3), np.int32);
            Assert.IsTrue(Enumerable.SequenceEqual(n.int32, new int[] { 0, 0, 0 }));
        }
    }
}
