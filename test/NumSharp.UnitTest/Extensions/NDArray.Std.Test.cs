using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;

namespace NumSharp.UnitTest.Extensions
{
    [TestClass]
    public class NDArrayStdTest
    {
        [TestMethod]
        public void StdTest()
        {
            var series1 = new NDArray_Legacy<double>();
            series1.Data = new double[] { 1, 2 };

            var series2 = new NDArray_Legacy<double>();
            series2.Data = new double[] { 3, 4 };

            var np = new NDArray_Legacy<NDArray_Legacy<double>>();
            np.Data.Add(series1);
            np.Data.Add(series2);

            Assert.IsTrue(Enumerable.SequenceEqual(np.Std().Data, new double[] { 1.1180339887498949 }));
            // Assert.IsTrue(Enumerable.SequenceEqual(np.Std(0).Data, new double[] { 1, 1 }));
            // Assert.IsTrue(Enumerable.SequenceEqual(np.Std(1).Data, new double[] { 0.5, 3.5 }));
        }
    }
}
