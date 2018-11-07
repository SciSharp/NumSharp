using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;

namespace NumSharp.UnitTest.Extensions
{
    [TestClass]
    public class NdArrayMeanTest
    {
        [TestMethod]
        public void MeanAxis0()
        {
            var np = new NDArray<double>().ARange(5,1).ReShape(2,2);

            Assert.IsTrue(Enumerable.SequenceEqual(np.Mean(0).Data, new double[] { 2, 3 }));
        }
        [TestMethod]
        public void MeanAxis1()
        {
            var np = new NDArray<double>().ARange(5,1).ReShape(2,2);
            
            Assert.IsTrue(Enumerable.SequenceEqual(np.Mean(1).Data, new double[] { 1.5, 3.5 }));
        }
        [TestMethod]
        public void MeanAxisMinus1()
        {
            var np = new NDArray<double>().ARange(5,1).ReShape(2,2);

            Assert.IsTrue(Enumerable.SequenceEqual(np.Mean().Data, new double[] { 2.5 }));
            
        }
    }
}
