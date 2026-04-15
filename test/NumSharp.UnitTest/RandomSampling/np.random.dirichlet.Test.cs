using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.RandomSampling
{
    /// <summary>
    /// Tests for np.random.dirichlet (Dirichlet distribution)
    /// </summary>
    public class NpRandomDirichletTests : TestClass
    {
        [TestMethod]
        public void Dirichlet_SingleSample_ReturnsCorrectShape()
        {
            var alpha = new double[] { 1, 2, 3 };
            var result = np.random.dirichlet(alpha);
            Assert.AreEqual(1, result.ndim);
            Assert.AreEqual(3, result.size);
            Assert.AreEqual(3, result.shape[0]);
        }

        [TestMethod]
        public void Dirichlet_MultipleSamples_ReturnsCorrectShape()
        {
            var alpha = new double[] { 1, 2, 3 };
            var result = np.random.dirichlet(alpha, 5);
            Assert.AreEqual(2, result.ndim);
            Assert.AreEqual(5, result.shape[0]);
            Assert.AreEqual(3, result.shape[1]);
            Assert.AreEqual(15, result.size);
        }

        [TestMethod]
        public void Dirichlet_2DSize_ReturnsCorrectShape()
        {
            var alpha = new double[] { 1, 2, 3 };
            var result = np.random.dirichlet(alpha, new Shape(2, 3));
            Assert.AreEqual(3, result.ndim);
            Assert.AreEqual(2, result.shape[0]);
            Assert.AreEqual(3, result.shape[1]);
            Assert.AreEqual(3, result.shape[2]);
            Assert.AreEqual(18, result.size);
        }

        [TestMethod]
        public void Dirichlet_ReturnsFloat64()
        {
            var alpha = new double[] { 1, 2, 3 };
            var result = np.random.dirichlet(alpha, 5);
            Assert.AreEqual(NPTypeCode.Double, result.typecode);
        }

        [TestMethod]
        public void Dirichlet_EachRowSumsToOne()
        {
            var rng = np.random.RandomState(42);
            var alpha = new double[] { 1, 2, 3 };
            var samples = rng.dirichlet(alpha, 100);

            for (int i = 0; i < 100; i++)
            {
                double rowSum = 0;
                for (int j = 0; j < 3; j++)
                {
                    rowSum += (double)samples[i, j];
                }
                Assert.IsTrue(Math.Abs(rowSum - 1.0) < 1e-10, $"Row {i} sum should be 1.0, got {rowSum}");
            }
        }

        [TestMethod]
        public void Dirichlet_AllValuesInZeroOne()
        {
            var rng = np.random.RandomState(42);
            var alpha = new double[] { 1, 2, 3 };
            var samples = rng.dirichlet(alpha, 1000);

            foreach (var val in samples.AsIterator<double>())
            {
                Assert.IsTrue(val >= 0.0 && val <= 1.0, $"Value should be in [0,1], got {val}");
            }
        }

        [TestMethod]
        public void Dirichlet_HasCorrectMean()
        {
            // Mean of component i = alpha[i] / sum(alpha)
            // For alpha = [1, 2, 3], sum = 6, means = [1/6, 2/6, 3/6] = [0.167, 0.333, 0.5]
            var rng = np.random.RandomState(42);
            var alpha = new double[] { 1, 2, 3 };
            var samples = rng.dirichlet(alpha, 100000);

            double[] means = new double[3];
            for (int i = 0; i < 100000; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    means[j] += (double)samples[i, j];
                }
            }
            for (int j = 0; j < 3; j++)
            {
                means[j] /= 100000;
            }

            double alphaSum = 6.0;
            Assert.IsTrue(Math.Abs(means[0] - 1.0 / alphaSum) < 0.01, $"Mean[0] should be ~0.167, got {means[0]}");
            Assert.IsTrue(Math.Abs(means[1] - 2.0 / alphaSum) < 0.01, $"Mean[1] should be ~0.333, got {means[1]}");
            Assert.IsTrue(Math.Abs(means[2] - 3.0 / alphaSum) < 0.01, $"Mean[2] should be ~0.5, got {means[2]}");
        }

        [TestMethod]
        public void Dirichlet_UniformAlpha_HasEqualMeans()
        {
            // For alpha = [1, 1, 1], all means should be 1/3
            var rng = np.random.RandomState(42);
            var alpha = new double[] { 1, 1, 1 };
            var samples = rng.dirichlet(alpha, 10000);

            double[] means = new double[3];
            for (int i = 0; i < 10000; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    means[j] += (double)samples[i, j];
                }
            }
            for (int j = 0; j < 3; j++)
            {
                means[j] /= 10000;
            }

            for (int j = 0; j < 3; j++)
            {
                Assert.IsTrue(Math.Abs(means[j] - 1.0 / 3.0) < 0.02, $"Mean[{j}] should be ~0.333, got {means[j]}");
            }
        }

        [TestMethod]
        public void Dirichlet_SingleCategory_ReturnsAllOnes()
        {
            // k=1: only one category, so each sample is [1.0]
            var alpha = new double[] { 5.0 };
            var samples = np.random.dirichlet(alpha, 5);

            Assert.AreEqual(2, samples.ndim);
            Assert.AreEqual(5, samples.shape[0]);
            Assert.AreEqual(1, samples.shape[1]);

            for (int i = 0; i < 5; i++)
            {
                Assert.AreEqual(1.0, (double)samples[i, 0], 1e-10, $"Single category sample should be 1.0");
            }
        }

        [TestMethod]
        public void Dirichlet_SameSeed_ProducesSameResults()
        {
            var alpha = new double[] { 1, 2, 3 };

            var rng1 = np.random.RandomState(42);
            var samples1 = rng1.dirichlet(alpha, 10);

            var rng2 = np.random.RandomState(42);
            var samples2 = rng2.dirichlet(alpha, 10);

            for (int i = 0; i < samples1.size; i++)
            {
                Assert.AreEqual((double)samples1.flat[i], (double)samples2.flat[i], $"Values at index {i} should match with same seed");
            }
        }

        // ========== Validation Tests ==========

        [TestMethod]
        public void Dirichlet_EmptyAlpha_ThrowsArgumentException()
        {
            Assert.ThrowsException<ArgumentException>(() => np.random.dirichlet(new double[0], 5));
        }

        [TestMethod]
        public void Dirichlet_NullAlpha_ThrowsArgumentException()
        {
            Assert.ThrowsException<ArgumentException>(() => np.random.dirichlet((double[])null, 5));
        }

        [TestMethod]
        public void Dirichlet_NegativeAlpha_ThrowsArgumentException()
        {
            Assert.ThrowsException<ArgumentException>(() => np.random.dirichlet(new double[] { 1, -1, 2 }, 5));
        }

        [TestMethod]
        public void Dirichlet_ZeroAlpha_ThrowsArgumentException()
        {
            Assert.ThrowsException<ArgumentException>(() => np.random.dirichlet(new double[] { 0, 1, 2 }, 5));
        }

        [TestMethod]
        public void Dirichlet_NaNAlpha_ThrowsArgumentException()
        {
            Assert.ThrowsException<ArgumentException>(() => np.random.dirichlet(new double[] { 1, double.NaN, 2 }, 5));
        }

        // ========== Tests migrated from NumPy ==========

        /// <summary>
        /// Migrated from NumPy test_randomstate.py
        /// Basic smoke test.
        /// </summary>
        [TestMethod]
        public void Dirichlet_NumPy_SmokeTest()
        {
            var alpha = new double[] { 1.0, 2.0, 3.0 };
            var vals = np.random.dirichlet(alpha, 10);
            Assert.AreEqual(10, vals.shape[0]);
            Assert.AreEqual(3, vals.shape[1]);
        }

        /// <summary>
        /// Migrated from NumPy - verify samples sum to 1.
        /// </summary>
        [TestMethod]
        public void Dirichlet_NumPy_SamplesSumToOne()
        {
            var rng = np.random.RandomState(12345);
            var alpha = new double[] { 0.5, 0.5, 0.5, 0.5 };
            var samples = rng.dirichlet(alpha, 100);

            for (int i = 0; i < 100; i++)
            {
                double sum = 0;
                for (int j = 0; j < 4; j++)
                {
                    sum += (double)samples[i, j];
                }
                Assert.IsTrue(Math.Abs(sum - 1.0) < 1e-10, $"Sample {i} should sum to 1.0");
            }
        }

        /// <summary>
        /// Test with NDArray input for alpha.
        /// </summary>
        [TestMethod]
        public void Dirichlet_NDArrayAlpha_Works()
        {
            var alpha = np.array(new double[] { 1, 2, 3 });
            var result = np.random.dirichlet(alpha, size: new Shape(5));
            Assert.AreEqual(2, result.ndim);
            Assert.AreEqual(5, result.shape[0]);
            Assert.AreEqual(3, result.shape[1]);
        }
    }
}
