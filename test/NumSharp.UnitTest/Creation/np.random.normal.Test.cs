using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Core.Extensions;
using System.Linq;
using NumSharp.Core;

namespace NumSharp.UnitTest
{
    [TestClass]
    public class NumPyRandomNormalTest
    {
        [TestMethod]
        public void NormalDistributionTest()
        {
            // https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.normal.html
            double mu = 0; // mean
            double sigma = 0.1; // standard deviation
            var s = np.random.normal(mu, sigma, 10, 100);

            var mean = np.mean(s);
            Assert.IsTrue(Math.Abs(mu - mean.Data<float>()[0]) < 0.01);
            Assert.IsTrue(s.shape[0] == 10);
            Assert.IsTrue(s.shape[1] == 100);

            // var std = np.std(s, ddof = 1);
            // Assert.IsTrue(Math.Abs(sigma - std)) < 0.01;
        }
    }
}
