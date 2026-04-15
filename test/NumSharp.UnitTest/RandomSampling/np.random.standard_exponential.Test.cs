using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.RandomSampling
{
    /// <summary>
    /// Tests for np.random.standard_exponential (standard exponential distribution)
    /// </summary>
    
    public class NpRandomStandardExponentialTests : TestClass
    {
        [TestMethod]
        public void StandardExponential_1D_ReturnsCorrectShape()
        {
            var rand = np.random.standard_exponential(5L);
            Assert.AreEqual(1, rand.ndim);
            Assert.AreEqual(5, rand.size);
        }

        [TestMethod]
        public void StandardExponential_2D_ReturnsCorrectShape()
        {
            var rand = np.random.standard_exponential(new Shape(5, 5));
            Assert.AreEqual(2, rand.ndim);
            Assert.AreEqual(25, rand.size);
        }

        [TestMethod]
        public void StandardExponential_2DByShape_ReturnsCorrectShape()
        {
            var rand = np.random.standard_exponential(new Shape(2, 3));
            Assert.AreEqual(2, rand.ndim);
            Assert.AreEqual(6, rand.size);
        }

        [TestMethod]
        public void StandardExponential_ReturnsFloat64()
        {
            var result = np.random.standard_exponential(5L);
            Assert.AreEqual(NPTypeCode.Double, result.typecode);
        }

        [TestMethod]
        public void StandardExponential_AllValuesPositive()
        {
            // Exponential distribution produces strictly positive values
            var rng = np.random.RandomState(42);
            var samples = rng.standard_exponential(10000L);

            foreach (var val in samples.AsIterator<double>())
            {
                Assert.IsTrue(val > 0.0, $"Standard exponential values should be > 0, got {val}");
            }
        }

        [TestMethod]
        public void StandardExponential_HasCorrectMean()
        {
            // mean = 1 for standard exponential
            var rng = np.random.RandomState(42);
            var samples = rng.standard_exponential(100000L);

            double mean = 0;
            foreach (var val in samples.AsIterator<double>())
                mean += val;
            mean /= samples.size;

            Assert.IsTrue(Math.Abs(mean - 1.0) < 0.05, $"Mean should be near 1.0, got {mean}");
        }

        [TestMethod]
        public void StandardExponential_HasCorrectVariance()
        {
            // variance = 1 for standard exponential
            var rng = np.random.RandomState(42);
            var samples = rng.standard_exponential(100000L);

            double mean = 0;
            foreach (var val in samples.AsIterator<double>())
                mean += val;
            mean /= samples.size;

            double variance = 0;
            foreach (var val in samples.AsIterator<double>())
                variance += (val - mean) * (val - mean);
            variance /= samples.size;

            Assert.IsTrue(Math.Abs(variance - 1.0) < 0.1, $"Variance should be near 1.0, got {variance}");
        }

        [TestMethod]
        public void StandardExponential_Scalar_ReturnsScalar()
        {
            var rng = np.random.RandomState(42);
            var result = rng.standard_exponential();
            // NumPy returns a scalar (0-dimensional) when no size is given
            Assert.AreEqual(0, result.ndim);
            Assert.AreEqual(1, result.size);
        }

        [TestMethod]
        public void StandardExponential_SameSeed_ProducesSameResults()
        {
            var rng1 = np.random.RandomState(42);
            var samples1 = rng1.standard_exponential(10L);

            var rng2 = np.random.RandomState(42);
            var samples2 = rng2.standard_exponential(10L);

            for (int i = 0; i < 10; i++)
            {
                Assert.AreEqual((double)samples1[i], (double)samples2[i], $"Values at index {i} should match with same seed");
            }
        }

        [TestMethod]
        public void StandardExponential_EquivalentToExponentialScale1()
        {
            // standard_exponential() should be equivalent to exponential(scale=1)
            var rng1 = np.random.RandomState(42);
            var se = rng1.standard_exponential(10L);

            var rng2 = np.random.RandomState(42);
            var e = rng2.exponential(1.0, 10L);

            for (int i = 0; i < 10; i++)
            {
                Assert.AreEqual((double)se[i], (double)e[i], 1e-10, $"standard_exponential should equal exponential(1) at index {i}");
            }
        }

        [TestMethod]
        public void StandardExponential_Size1_ReturnsShape1Array()
        {
            var result = np.random.standard_exponential(1L);
            Assert.AreEqual(1, result.ndim);
            Assert.AreEqual(1, result.size);
        }

        [TestMethod]
        public void StandardExponential_LargeSample_NoInfinities()
        {
            var rng = np.random.RandomState(42);
            var samples = rng.standard_exponential(100000L);

            foreach (var val in samples.AsIterator<double>())
            {
                Assert.IsFalse(double.IsInfinity(val), "Should not produce infinity");
                Assert.IsFalse(double.IsNaN(val), "Should not produce NaN");
            }
        }

        // ========== Tests migrated from NumPy ==========

        /// <summary>
        /// Migrated from NumPy test_randomstate.py
        /// Basic smoke test.
        /// </summary>
        [TestMethod]
        public void StandardExponential_NumPy_SmokeTest()
        {
            var vals = np.random.standard_exponential(10L);
            Assert.AreEqual(10, vals.size);
        }

        /// <summary>
        /// Migrated from NumPy - verify output is always positive.
        /// </summary>
        [TestMethod]
        public void StandardExponential_NumPy_OutputAlwaysPositive()
        {
            var rng = np.random.RandomState(12345);
            var samples = rng.standard_exponential(1000L);

            var minVal = double.MaxValue;
            foreach (var val in samples.AsIterator<double>())
            {
                if (val < minVal) minVal = val;
            }

            Assert.IsTrue(minVal > 0, $"Minimum value should be > 0, got {minVal}");
        }
    }
}
