using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Extensions
{
    /// <summary>
    /// Test concolve with standard example from 
    /// https://www.numpy.org/devdocs/reference/generated/numpy.convolve.html
    /// </summary>
    [TestClass]
    public class NdArrayConvolveTest
    {
        [TestMethod]
        public void ConvoleFull()
        {
            var series1 = new NDArray<double>();
            series1.Data = new double[]{1, 2, 3};
            
            var series2 = new NDArray<double>();
            series2.Data = new double[]{0, 1, 0.5};

            var series3 = series1.Convolve(series2);
            
            double[] expectedResult = new double[]{0, 1, 2.5, 4, 1.5};

            Assert.IsTrue(Enumerable.SequenceEqual(series3.Data.ToArray(), expectedResult));
        }
        [TestMethod]
        public void ConvoleValid()
        {
            // To be done 
        }
        [TestMethod]
        public void ConvoleSame()
        {
            // To be done 
        }
        
    }
}