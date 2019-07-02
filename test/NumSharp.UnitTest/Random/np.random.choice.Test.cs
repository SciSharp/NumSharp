using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace NumSharp.UnitTest.RandomSampling
{
    [TestClass]
    public class NpRandomChoiceTests : TestClass
    {
        [TestMethod]
        public void UniformSample1()
        {
            // Generate a uniform random sample from np.arange(5) of size 3:
            // This is equivalent to np.random.randint(0,5,3)
            int low = 0;
            int high = 5;
            int nrSamples = 3;

            NDArray arr = np.random.choice(high, nrSamples);
            Assert.AreEqual(arr.len, nrSamples, "Unexpected number of elements");

            // Verify that all elements in output are within the range
            for (int i = 0; i < arr.len; i++)
            {
                Assert.IsTrue(arr[i] >= low, "Element was less than expected");
                Assert.IsTrue(arr[i] < high, "Element was greater than expected");
            }
        }

        [TestMethod]
        public void NonUniformSample1()
        {
            // Generate a non-uniform random sample from np.arange(5) of size 3:
            int low = 0;
            int high = 5;
            int nrSamples = 3;
            double[] probabilities = new double[] {0.1, 0, 0.3, 0.6, 0};

            NDArray arr = np.random.choice(5, nrSamples, probabilities);
            Assert.AreEqual(arr.len, nrSamples, "Unexpected number of elements");

            // Verify that all elements in output are within the range
            for (int i = 0; i < arr.len; i++)
            {
                Assert.IsTrue(arr[i] >= low, "Element was less than expected");
                Assert.IsTrue(arr[i] < high, "Element was greater than expected");
                Assert.IsTrue(arr[i] != 1, "Sampled zero-probability element");
                Assert.IsTrue(arr[i] != 4, "Sampled zero-probability element");
            }
        }
    }
}
