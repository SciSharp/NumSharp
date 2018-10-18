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
        public void Mean()
        {
            var series1 = new NDArray<double>();
            series1.Data = new double[] { 1, 2 };

            var series2 = new NDArray<double>();
            series2.Data = new double[] { 3, 4 };

            var np = new NDArray<NDArray<double>>();
            np.Data.Add(series1);
            np.Data.Add(series2);

            double[] expectedResultDefault = new double[] { 2.5 };
            double[] expectedResult0 = new double[] { 2, 3 };
            double[] expectedResult1 = new double[] { 1.5, 3.5 };

            Assert.IsTrue(Enumerable.SequenceEqual(np.Mean().Data, expectedResultDefault));
            Assert.IsTrue(Enumerable.SequenceEqual(np.Mean(0).Data, expectedResult0));
            Assert.IsTrue(Enumerable.SequenceEqual(np.Mean(1).Data, expectedResult1));
        }
    }
}
