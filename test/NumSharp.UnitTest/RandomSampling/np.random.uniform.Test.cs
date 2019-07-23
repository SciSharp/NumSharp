using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Numerics;
using System.Collections.Generic;
using System.Text;
using System.Linq;

namespace NumSharp.UnitTest.RandomSampling
{
    [TestClass]
    public class NpRandomUniformTests : TestClass
    {
        [TestMethod]
        public void UniformByNDArrays()
        {
            var low = np.array(1d, 2d, 3d, 4d, 5d);
            var lowdata = low.Data<double>();
            var high = low + 1d;
            var uniformed = np.random.uniform(low, high);
            var data = uniformed.Data<double>();

            //assertions
            for (int i = 1; i <= 5; i++)
            {
                Assert.IsTrue(lowdata[i - 1] <= data[i - 1] && lowdata[i - 1] + 1 > data[i - 1]);
            }

            Assert.IsTrue(uniformed.ndim == 1);
            Assert.IsTrue(uniformed.size == 5);
        }

        [TestMethod]
        public void UniformByIntegers1D()
        {
            var low = 1d;
            var high = 2d;
            var uniformed = np.random.uniform(low, high, 1);
            var data = uniformed.Data<double>();

            Assert.IsTrue(uniformed.ndim == 1);
            Assert.IsTrue(uniformed.size == 1);

            Assert.IsTrue(data.Count == 1);
            Assert.IsTrue(data[0] >= 1 && data[0] < 2);
        }

        [TestMethod]
        public void UniformByIntegers2D()
        {
            var low = 1d;
            var high = 2d;
            var uniformed = np.random.uniform(low, high, 3, 3);
            var data = uniformed.Data<double>();
            Assert.IsTrue(uniformed.ndim == 2);
            Assert.IsTrue(uniformed.size == 9);
            Assert.IsTrue(data.Count == 9);
            Assert.IsTrue(data.All(v => v >= 1) && data.All(v => v < 2));
        }
    }
}
