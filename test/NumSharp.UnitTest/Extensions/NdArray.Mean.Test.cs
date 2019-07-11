using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;
using NumSharp;

namespace NumSharp.UnitTest.Extensions
{
    [TestClass]
    public class NdArrayMeanTest
    {
        [TestMethod]
        public void MeanAxis0()
        {
            var np1 = np.array(new double[] {1, 2, 3, 4}).reshape(2, 2);
            var mean = np.mean(np1, 0);
            Assert.IsTrue(Enumerable.SequenceEqual(mean.Data<double>(), new double[] {2, 3}));
        }

        [TestMethod]
        public void MeanAxis1()
        {
            var np1 = np.array(new double[] {1, 2, 3, 4}).reshape(2, 2);
            var mean = np.mean(np1, 1);
            Assert.IsTrue(Enumerable.SequenceEqual(mean.Data<double>(), new double[] {1.5, 3.5}));
        }

        [TestMethod]
        public void MeanAxisMinus1()
        {
            var np1 = np.array(new double[] {1, 2, 3, 4}).reshape(2, 2);
            var mean = np.mean(np1);
            Assert.IsTrue(Enumerable.SequenceEqual(mean.Data<double>(), new double[] {2.5}));
        }
    }
}
