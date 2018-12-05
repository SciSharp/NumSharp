using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Core.Extensions;
using System.Linq;
using NumSharp.Core;

namespace NumSharp.UnitTest.Extensions
{
    [TestClass]
    public class NdArrayMeanTest
    {
        [TestMethod]
        public void MeanAxis0()
        {
            var np = new NumPyGeneric<double>().array(new double[]{1,2,3,4}).reshape(2,2);

            Assert.IsTrue(Enumerable.SequenceEqual(np.Mean(0).Data, new double[] { 2, 3 }));
        }
        [TestMethod]
        public void MeanAxis1()
        {
            var np = new NumPyGeneric<double>().array(new double[]{1,2,3,4}).reshape(2,2);
            
            Assert.IsTrue(Enumerable.SequenceEqual(np.Mean(1).Data, new double[] { 1.5, 3.5 }));
        }
        [TestMethod]
        public void MeanAxisMinus1()
        {
            var np = new NumPyGeneric<double>().array(new double[]{1,2,3,4}).reshape(2,2);

            Assert.IsTrue(Enumerable.SequenceEqual(np.Mean().Data, new double[] { 2.5 }));
            
        }
    }
}
