using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Extensions
{
    /// <summary>
    /// Test mean with standard example from 
    /// https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.mean.html
    /// </summary>
    [TestClass]
    public class NdArrayMeanTest
    {
        [TestMethod]
        public void Mean1D()
        {
            var series = new NDArray<double>();
            series.Zeros(100);

            series.Data = Enumerable.Range(1,100)
                                    .Select(x => (double)x)
                                    .ToArray();

            var mean = series.Mean();

            Assert.IsTrue( mean.Data[0] == 50.5 );
        } 
    }
}