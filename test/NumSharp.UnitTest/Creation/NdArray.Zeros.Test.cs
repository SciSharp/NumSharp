using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using NumSharp;

namespace NumSharp.UnitTest.Creation
{
    [TestClass]
    public class NdArrayZerosTest
    {
        [TestMethod]
        public void Zeros1Dim()
        {
            var n = np.zeros(3);
            Assert.IsTrue(Enumerable.SequenceEqual(n.Storage.GetData<double>(), new double[] { 0, 0, 0 }));
        }

        [TestMethod]
        public void Zeros2Dim()
        {
            var n = np.zeros(3, 2);
            Assert.IsTrue(Enumerable.SequenceEqual(n.Storage.GetData<double>(), new double[] { 0, 0, 0, 0, 0, 0 }));
        }

        [TestMethod]
        public void Zeros1DimWithDtype()
        {
            var n = np.zeros(new Shape(3), np.int32);
            Assert.IsTrue(Enumerable.SequenceEqual(n.Storage.GetData<int>(), new int[] { 0, 0, 0 }));
        }
    }
}
