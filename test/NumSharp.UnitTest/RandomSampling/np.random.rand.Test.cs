using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Numerics;
using System.Collections.Generic;
using System.Text;
using System.Linq;

namespace NumSharp.UnitTest.RandomSampling
{
    [TestClass]
    public class NpRandomRandTests : TestClass
    {
        [TestMethod]
        public void Rand1D()
        {
            var rand = np.random.rand(5);
            Assert.IsTrue(rand.Data<double>().All(v => v >= 0 && v < 1));
            Assert.IsTrue(rand.ndim == 1);
            Assert.IsTrue(rand.size == 5);
        }

        [TestMethod]
        public void Rand1DByShape()
        {
            var rand = np.random.rand(new Shape(5));
            Assert.IsTrue(rand.Data<double>().All(v => v >= 0 && v < 1));
            Assert.IsTrue(rand.ndim == 1);
            Assert.IsTrue(rand.size == 5);
        }

        [TestMethod]
        public void Rand2D()
        {
            var rand = np.random.rand(5, 5);
            Assert.IsTrue(rand.Data<double>().All(v => v >= 0 && v < 1));
            Assert.IsTrue(rand.ndim == 2);
            Assert.IsTrue(rand.size == 25);
        }

        [TestMethod]
        public void Rand2DByShape()
        {
            var rand = np.random.rand(new Shape(5, 5));
            Assert.IsTrue(rand.Data<double>().All(v => v >= 0 && v < 1));
            Assert.IsTrue(rand.ndim == 2);
            Assert.IsTrue(rand.size == 25);
        }
    }
}
