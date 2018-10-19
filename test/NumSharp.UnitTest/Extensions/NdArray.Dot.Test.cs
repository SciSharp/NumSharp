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
    public class NdArrayDotTest
    {
        [TestMethod]
        public void DotTwo1DDouble()
        {
            var series1 = new NDArray<double>();
            series1.Data = new double[]{1, 2, 3};
            
            var series2 = new NDArray<double>();
            series2.Data = new double[]{0, 1, 0.5};

            var innerProduct = series1.Dot(series2);
            Assert.IsTrue(innerProduct == 3.5);
        }
        [TestMethod]
        public void ConvoleValid()
        {
            var series1 = new NDArray<int>();
            series1.Data = new int[]{1, 2, 3};
            
            var series2 = new NDArray<int>();
            series2.Data = new int[]{2, 3, 4};

            var innerProduct = series1.Dot(series2);
            
            Assert.IsTrue(innerProduct == 20);
        }
        [TestMethod]
        public void ConvoleSame()
        {
            var series1 = new NDArray<double>();
            series1.Data = new double[]{1, 2, 3};
            
            var series2 = new NDArray<double>();
            series2.Data = new double[]{0, 1, 0.5};

            var series3 = series1.Convolve(series2, "same");

            double[] expectedResult = new double[]{1, 2.5, 4};

            Assert.IsTrue(Enumerable.SequenceEqual(series3.Data.ToArray(),expectedResult));
            
        }
        
    }
}