using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;

namespace NumSharp.UnitTest.Extensions
{
    /// <summary>
    /// Tests following https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.hstack.html
    /// </summary>
    [TestClass]
    public class NdArrayHStackTest
    {
        [TestMethod]
        public void HStackTwo1DArrays()
        {
            var series1 = new NDArray<double>();
            series1.Data = new double[]{1, 2, 3};
            
            var series2 = new NDArray<double>();
            series2.Data = new double[]{2,3,4};

            var series3 = series1.HStack(series2);

            var expectedValue = new double[]{1,2,3,2,3,4};

            Assert.IsTrue(Enumerable.SequenceEqual(series3.Data,expectedValue));
        }
    }
}
