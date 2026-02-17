using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;
using NumSharp;

namespace NumSharp.UnitTest.RandomSampling
{
    public class NpRandomNormalTest
    {
        [Test]
        public void NormalDistributionTest()
        {
            // https://numpy.org/doc/stable/reference/generated/numpy.random.normal.html
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
