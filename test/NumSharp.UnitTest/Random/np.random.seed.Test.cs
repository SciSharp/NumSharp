using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace NumSharp.UnitTest.RandomSampling
{
    /// <summary>
    /// The random seed tests are only supposed to test the consistent output from the random state when the
    /// same seed is applied. No testing of the actual output from the random state is expected here. Just test
    /// the consistent output after repeatedly setting the same seed value.
    /// </summary>
    [TestClass]
    public class NpRandomSeedTests : TestClass
    {
        [TestMethod]
        public void SeedTest()
        {
            NumPyRandom rando = np.random.RandomState(1000);
            Assert.AreEqual(1000, rando.Seed, "The seed value given in the ctor does not match the seed value attribute.");
        }

        [TestMethod]
        public void UniformOneSample()
        {
            NumPyRandom rando = np.random.RandomState(1000);
            // Generate a uniform random sample from np.arange(5) of size 1:
            // This is equivalent to np.random.randint(0,5,1)
            int low = 0;
            int high = 5;

            // Start with the known, which is what we expect to see every time
            NDArray actual = rando.choice(high);
            for (int i = 0; i < 10; i++) {
                rando.seed(1000);
                NDArray test = rando.choice(high); // Not specifying size means 1 single value is wanted
                Assert.AreEqual(actual, test, "Inconsistent random result with same seed. Expected the value to be equal every time.");
            }

        }

        [TestMethod]
        public void UniformMultipleSample()
        {
            NumPyRandom rando = np.random.RandomState(1000);
            // Generate a uniform random sample from np.arange(5) of size 3:
            // This is equivalent to np.random.randint(0,5,3)
            int low = 0;
            int high = 5;
            int nrSamples = 3;

            NDArray actual = rando.choice(high, (Shape)nrSamples);

            for (int i = 0; i < 10; i++) {
                rando.seed(1000);
                NDArray test = rando.choice(high, (Shape)nrSamples);
                for (int j = 0; j < actual.size; j++) {
                    Assert.AreEqual(actual.GetAtIndex<int>(j), test.GetAtIndex<int>(j), "Inconsistent choice sampling with the same seed. Expected the results to always be the same.");
                }
            }
        }

        [TestMethod]
        public void NonUniformSample()
        {
            NumPyRandom rando = np.random.RandomState(1000);
            // Generate a non-uniform random sample from np.arange(5) of size 3:
            int low = 0;
            int high = 5;
            int nrSamples = 3;
            double[] probabilities = new double[] { 0.1, 0, 0.3, 0.6, 0 };

            NDArray actual = rando.choice(5, (Shape)nrSamples, probabilities: probabilities);

            for (int i = 0; i < 10; i++) {
                rando.seed(1000);
                NDArray test = rando.choice(5, (Shape)nrSamples, probabilities: probabilities);
                for (int j = 0; j < actual.size; j++) {
                    Assert.AreEqual(actual.GetAtIndex<int>(j), test.GetAtIndex<int>(j), "Inconsistent choice sampling with the same seed. Expected the results to always be the same.");
                }
            }
        }


        [TestMethod]
        public void IntegerArraySample()
        {
            NumPyRandom rando = np.random.RandomState(1000);
            int nrSamples = 5;

            NDArray int_arr = new int[] { 42, 96, 3, 101 };
            double[] probabilities = new double[] { 0.5, 0.1, 0.0, 0.3 };

            NDArray actual = rando.choice(int_arr, (Shape)nrSamples, probabilities: probabilities);

            for (int i = 0; i < 10; i++)
            {
                rando.seed(1000);
                NDArray test = rando.choice(int_arr, (Shape)nrSamples, probabilities: probabilities);
                for (int j = 0; j < actual.size; j++)
                {
                    Assert.AreEqual(actual.GetAtIndex<int>(j), test.GetAtIndex<int>(j), "Inconsistent choice sampling with the same seed. Expected the results to always be the same.");
                }
            }
        }
    }
}
