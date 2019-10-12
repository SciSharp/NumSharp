using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest
{
    /// <summary>
    /// Test concolve with standard example from 
    /// https://www.numpy.org/devdocs/reference/generated/numpy.convolve.html
    /// </summary>
    [TestClass]
    public class NdArrayConvolveTest
    {
        [Ignore("TODO: fix this test")]
        [TestMethod]
        public void ConvoleFull()
        {
            var series1 = np.array(new double[] {1, 2, 3});
            var series2 = np.array(new double[] {0, 1, 0.5});

            var series3 = series1.convolve(series2);

            double[] expectedResult = new double[] {0, 1, 2.5, 4, 1.5};

            Assert.IsTrue(Enumerable.SequenceEqual(series3.Data<double>(), expectedResult));
        }

        [Ignore("TODO: fix this test")]
        [TestMethod]
        public void ConvoleValid()
        {
            var series1 = np.array(new double[] {1, 2, 3});
            var series2 = np.array(new double[] {0, 1, 0.5});

            var series3 = series1.convolve(series2, "valid");

            double[] expectedResult = new double[] {2.5};

            Assert.IsTrue(Enumerable.SequenceEqual(series3.Data<double>(), expectedResult));
        }

        [Ignore("TODO: fix this test")]
        [TestMethod]
        public void ConvoleSame()
        {
            var series1 = np.array(new double[] {1, 2, 3});
            var series2 = np.array(new double[] {0, 1, 0.5});

            var series3 = series1.convolve(series2, "same");

            double[] expectedResult = new double[] {1, 2.5, 4};

            Assert.IsTrue(Enumerable.SequenceEqual(series3.Data<double>(), expectedResult));
        }
    }
}
