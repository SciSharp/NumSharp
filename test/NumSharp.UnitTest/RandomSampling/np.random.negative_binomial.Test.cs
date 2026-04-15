using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.RandomSampling
{
    /// <summary>
    /// Tests for np.random.negative_binomial (negative binomial distribution)
    /// </summary>
    
    public class NpRandomNegativeBinomialTests : TestClass
    {
        [TestMethod]
        public void NegativeBinomial_1D_ReturnsCorrectShape()
        {
            var rand = np.random.negative_binomial(10, 0.5, 5L);
            Assert.AreEqual(1, rand.ndim);
            Assert.AreEqual(5, rand.size);
        }

        [TestMethod]
        public void NegativeBinomial_2D_ReturnsCorrectShape()
        {
            var rand = np.random.negative_binomial(10, 0.5, new Shape(5, 5));
            Assert.AreEqual(2, rand.ndim);
            Assert.AreEqual(25, rand.size);
        }

        [TestMethod]
        public void NegativeBinomial_2DByShape_ReturnsCorrectShape()
        {
            var rand = np.random.negative_binomial(10, 0.5, new Shape(5, 5));
            Assert.AreEqual(2, rand.ndim);
            Assert.AreEqual(25, rand.size);
        }

        [TestMethod]
        public void NegativeBinomial_ReturnsInt64()
        {
            var result = np.random.negative_binomial(10, 0.5, 5L);
            Assert.AreEqual(NPTypeCode.Int64, result.typecode);
        }

        [TestMethod]
        public void NegativeBinomial_AllValuesNonNegative()
        {
            // Negative binomial produces non-negative integers (number of failures)
            var rng = np.random.RandomState(42);
            var samples = rng.negative_binomial(10, 0.5, 10000L);

            foreach (var val in samples.AsIterator<long>())
            {
                Assert.IsTrue(val >= 0L, $"Negative binomial values should be >= 0, got {val}");
            }
        }

        [TestMethod]
        public void NegativeBinomial_HasCorrectMean()
        {
            // mean = n * (1-p) / p
            // For n=10, p=0.5: mean = 10 * 0.5 / 0.5 = 10
            var rng = np.random.RandomState(42);
            var samples = rng.negative_binomial(10, 0.5, 100000L);

            double mean = 0;
            foreach (var val in samples.AsIterator<long>())
                mean += val;
            mean /= samples.size;

            double expectedMean = 10.0 * (1.0 - 0.5) / 0.5;  // = 10
            Assert.IsTrue(Math.Abs(mean - expectedMean) < 0.5, $"Mean should be near {expectedMean}, got {mean}");
        }

        [TestMethod]
        public void NegativeBinomial_HasCorrectVariance()
        {
            // variance = n * (1-p) / p^2
            // For n=10, p=0.5: var = 10 * 0.5 / 0.25 = 20
            var rng = np.random.RandomState(42);
            var samples = rng.negative_binomial(10, 0.5, 100000L);

            double mean = 0;
            foreach (var val in samples.AsIterator<long>())
                mean += val;
            mean /= samples.size;

            double variance = 0;
            foreach (var val in samples.AsIterator<long>())
                variance += (val - mean) * (val - mean);
            variance /= samples.size;

            double expectedVar = 10.0 * (1.0 - 0.5) / (0.5 * 0.5);  // = 20
            Assert.IsTrue(Math.Abs(variance - expectedVar) < 2.0, $"Variance should be near {expectedVar}, got {variance}");
        }

        [TestMethod]
        public void NegativeBinomial_PEqualsOne_ReturnsAllZeros()
        {
            // p=1 means immediate success, so 0 failures
            var samples = np.random.negative_binomial(10, 1.0, 10L);

            foreach (var val in samples.AsIterator<long>())
            {
                Assert.AreEqual(0L, val, "p=1 should produce all 0s (0 failures)");
            }
        }

        [TestMethod]
        public void NegativeBinomial_HighP_FewFailures()
        {
            // High p means few failures expected
            var rng = np.random.RandomState(42);
            var samples = rng.negative_binomial(10, 0.9, 1000L);

            double mean = 0;
            foreach (var val in samples.AsIterator<long>())
                mean += val;
            mean /= samples.size;

            // Expected mean = 10 * 0.1 / 0.9 ≈ 1.11
            Assert.IsTrue(mean < 5, $"High p should give low mean, got {mean}");
        }

        [TestMethod]
        public void NegativeBinomial_LowP_ManyFailures()
        {
            // Low p means many failures expected
            var rng = np.random.RandomState(42);
            var samples = rng.negative_binomial(10, 0.1, 1000L);

            double mean = 0;
            foreach (var val in samples.AsIterator<long>())
                mean += val;
            mean /= samples.size;

            // Expected mean = 10 * 0.9 / 0.1 = 90
            Assert.IsTrue(mean > 50, $"Low p should give high mean, got {mean}");
        }

        [TestMethod]
        public void NegativeBinomial_Scalar_ReturnsScalar()
        {
            var rng = np.random.RandomState(42);
            var result = rng.negative_binomial(10, 0.5);
            // NumPy returns a scalar (0-dimensional) when no size is given
            Assert.AreEqual(0, result.ndim);
            Assert.AreEqual(1, result.size);
        }

        [TestMethod]
        public void NegativeBinomial_SameSeed_ProducesSameResults()
        {
            var rng1 = np.random.RandomState(42);
            var samples1 = rng1.negative_binomial(10, 0.5, 10L);

            var rng2 = np.random.RandomState(42);
            var samples2 = rng2.negative_binomial(10, 0.5, 10L);

            for (int i = 0; i < 10; i++)
            {
                Assert.AreEqual((long)samples1[i], (long)samples2[i], $"Values at index {i} should match with same seed");
            }
        }

        // ========== Validation Tests ==========

        [TestMethod]
        public void NegativeBinomial_NZero_ThrowsArgumentException()
        {
            Assert.ThrowsException<ArgumentException>(() => np.random.negative_binomial(0, 0.5, 5L));
        }

        [TestMethod]
        public void NegativeBinomial_NNegative_ThrowsArgumentException()
        {
            Assert.ThrowsException<ArgumentException>(() => np.random.negative_binomial(-1, 0.5, 5L));
        }

        [TestMethod]
        public void NegativeBinomial_PZero_ThrowsArgumentException()
        {
            Assert.ThrowsException<ArgumentException>(() => np.random.negative_binomial(10, 0, 5L));
        }

        [TestMethod]
        public void NegativeBinomial_PNegative_ThrowsArgumentException()
        {
            Assert.ThrowsException<ArgumentException>(() => np.random.negative_binomial(10, -0.1, 5L));
        }

        [TestMethod]
        public void NegativeBinomial_PGreaterThanOne_ThrowsArgumentException()
        {
            Assert.ThrowsException<ArgumentException>(() => np.random.negative_binomial(10, 1.5, 5L));
        }

        [TestMethod]
        public void NegativeBinomial_PNaN_ThrowsArgumentException()
        {
            Assert.ThrowsException<ArgumentException>(() => np.random.negative_binomial(10, double.NaN, 5L));
        }

        // ========== Tests migrated from NumPy ==========

        /// <summary>
        /// Migrated from NumPy test_randomstate.py
        /// Test that negative binomial accepts floating point arguments.
        /// </summary>
        [TestMethod]
        public void NegativeBinomial_NumPy_AcceptsFloatN()
        {
            // From NumPy: random_state.negative_binomial(0.5, 0.5)
            // n can be non-integer (generalized negative binomial)
            var result = np.random.negative_binomial(0.5, 0.5, 10L);
            Assert.AreEqual(10, result.size);
        }

        /// <summary>
        /// Migrated from NumPy test_smoke.py
        /// Basic smoke test.
        /// </summary>
        [TestMethod]
        public void NegativeBinomial_NumPy_SmokeTest()
        {
            var vals = np.random.negative_binomial(10, 0.3, 10L);
            Assert.AreEqual(10, vals.size);
        }
    }
}
