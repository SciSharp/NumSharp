using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Numerics;
using System.Collections.Generic;
using System.Text;
using System.Linq;

namespace NumSharp.UnitTest.RandomSampling
{
    public class NpRandomBetaTests : TestClass
    {
        [Test]
        public void Rand1D()
        {
            var rand = np.random.beta(1, 2, 5);
            Assert.IsTrue(rand.ndim == 1);
            Assert.IsTrue(rand.size == 5);
        }

        [Test]
        public void Rand2D()
        {
            var rand = np.random.beta(1, 2, 5, 5);
            Assert.IsTrue(rand.ndim == 2);
            Assert.IsTrue(rand.size == 25);
        }

        [Test]
        public void Rand2DByShape()
        {
            var rand = np.random.beta(1, 2, new Shape(5, 5));
            Assert.IsTrue(rand.ndim == 2);
            Assert.IsTrue(rand.size == 25);
        }
    }
}
