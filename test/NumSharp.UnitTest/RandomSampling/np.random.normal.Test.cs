using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;
using NumSharp;

namespace NumSharp.UnitTest.RandomSampling
{
    [TestClass]
    public class NpRandomNormalTest
    {
        [TestMethod]
        public void NormalDistributionTest()
        {
            // https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.normal.html
            double mu = 0; // mean
            double sigma = 0.1; // standard deviation
            var s = np.random.normal(mu, sigma, 10, 100);

            var mean = np.mean(s);
            Assert.IsTrue(Math.Abs(mu - mean.Data<double>()[0]) < 0.01);

            var std = s.std();
            // Assert.IsTrue(Math.Abs(sigma - std.Data<double>()[0] )  < 0.01);

            Assert.IsTrue(s.shape[0] == 10);
            Assert.IsTrue(s.shape[1] == 100);
        }
    }
}
