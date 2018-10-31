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
            var series1 = new NDArray_Legacy<double>();
            series1.Data = new double[] { 1, 2 };

            var series2 = new NDArray_Legacy<double>();
            series2.Data = new double[] { 3, 4 };

            var np = new NDArray_Legacy<NDArray_Legacy<double>>();
            np.Data.Add(series1);
            np.Data.Add(series2);

            //Assert.IsTrue(Enumerable.SequenceEqual(np.Mean().Data, new double[] { 2.5 }));
            Assert.IsTrue(Enumerable.SequenceEqual(np.Mean(0).Data, new double[] { 2, 3 }));
            Assert.IsTrue(Enumerable.SequenceEqual(np.Mean(1).Data, new double[] { 1.5, 3.5 }));
        }
    }
}
