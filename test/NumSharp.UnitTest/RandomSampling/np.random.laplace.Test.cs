using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.RandomSampling
{
    /// <summary>
    /// Tests for np.random.laplace (Laplace/double exponential distribution)
    /// </summary>
    public class NpRandomLaplaceTests : TestClass
    {
        [Test]
        public void Laplace_1D_ReturnsCorrectShape()
        {
            var rand = np.random.laplace(0, 1, 5);
            Assert.AreEqual(1, rand.ndim);
            Assert.AreEqual(5, rand.size);
        }

        [Test]
        public void Laplace_2D_ReturnsCorrectShape()
        {
            var rand = np.random.laplace(0, 1, 5, 5);
            Assert.AreEqual(2, rand.ndim);
            Assert.AreEqual(25, rand.size);
        }

        [Test]
        public void Laplace_2DByShape_ReturnsCorrectShape()
        {
            var rand = np.random.laplace(0, 1, new Shape(5, 5));
            Assert.AreEqual(2, rand.ndim);
            Assert.AreEqual(25, rand.size);
        }

        [Test]
        public void Laplace_DefaultParameters_HasCorrectStatistics()
        {
            // Laplace(0, 1) has mean=0 and std=sqrt(2)≈1.414, variance=2
            np.random.seed(42);
            var samples = np.random.laplace(0, 1, 100000);

            var mean = (double)np.mean(samples);
            var std = (double)np.std(samples);
            var variance = (double)np.var(samples);

            // Allow some tolerance for statistical sampling
            Assert.IsTrue(Math.Abs(mean) < 0.05, $"Mean should be near 0, got {mean}");
            Assert.IsTrue(Math.Abs(std - Math.Sqrt(2)) < 0.05, $"Std should be near sqrt(2)≈1.414, got {std}");
            Assert.IsTrue(Math.Abs(variance - 2.0) < 0.1, $"Variance should be near 2, got {variance}");
        }

        [Test]
        public void Laplace_WithLocScale_TransformsCorrectly()
        {
            // Laplace(μ, λ) has mean=μ and variance=2λ²
            np.random.seed(42);
            double loc = 5.0;
            double scale = 2.0;
            var samples = np.random.laplace(loc, scale, 100000);

            var mean = (double)np.mean(samples);
            var variance = (double)np.var(samples);
            double expectedVariance = 2 * scale * scale; // 2 * 4 = 8

            Assert.IsTrue(Math.Abs(mean - loc) < 0.1, $"Mean should be near {loc}, got {mean}");
            Assert.IsTrue(Math.Abs(variance - expectedVariance) < 0.5, $"Variance should be near {expectedVariance}, got {variance}");
        }

        [Test]
        public void Laplace_NegativeScale_ThrowsArgumentException()
        {
            Assert.ThrowsException<ArgumentException>(() => np.random.laplace(0, -1, 5));
        }

        [Test]
        public void Laplace_ScaleZero_ReturnsConstantAtLoc()
        {
            double loc = 5.0;
            var samples = np.random.laplace(loc, 0.0, 5);

            foreach (var val in samples.AsIterator<double>())
            {
                Assert.AreEqual(loc, val, $"All values should be {loc} when scale=0");
            }
        }

        [Test]
        public void Laplace_Scalar_ReturnsScalar()
        {
            np.random.seed(42);
            var result = np.random.laplace();
            // NumPy returns a scalar (0-dimensional) when no size is given
            Assert.AreEqual(0, result.ndim);
            Assert.AreEqual(1, result.size);
        }

        [Test]
        public void Laplace_ReturnsFloat64()
        {
            var result = np.random.laplace(0, 1, 5);
            Assert.AreEqual(NPTypeCode.Double, result.typecode);
        }

        [Test]
        public void Laplace_DifferentSeeds_ProduceDifferentResults()
        {
            np.random.seed(42);
            var samples1 = np.random.laplace(0, 1, 10);

            np.random.seed(123);
            var samples2 = np.random.laplace(0, 1, 10);

            bool anyDifferent = false;
            for (int i = 0; i < 10; i++)
            {
                if ((double)samples1[i] != (double)samples2[i])
                {
                    anyDifferent = true;
                    break;
                }
            }
            Assert.IsTrue(anyDifferent, "Different seeds should produce different results");
        }

        [Test]
        public void Laplace_SameSeed_ProducesSameResults()
        {
            np.random.seed(42);
            var samples1 = np.random.laplace(0, 1, 10);

            np.random.seed(42);
            var samples2 = np.random.laplace(0, 1, 10);

            for (int i = 0; i < 10; i++)
            {
                Assert.AreEqual((double)samples1[i], (double)samples2[i], $"Values at index {i} should match with same seed");
            }
        }

        // ========== Tests migrated from NumPy test_random.py ==========

        /// <summary>
        /// Migrated from NumPy test_random.py test_laplace_0
        /// Tests that scale=0 returns loc, and negative scale raises error.
        /// </summary>
        [Test]
        public void Laplace_NumPy_ScaleZeroReturnsLoc()
        {
            // From NumPy: assert_equal(np.random.laplace(scale=0), 0)
            var result = np.random.laplace(scale: 0);
            Assert.AreEqual(0.0, result.GetDouble(0), "laplace(scale=0) should return 0 (the default loc)");
        }

        /// <summary>
        /// Migrated from NumPy test_random.py test_laplace_0
        /// Negative zero scale should raise ValueError in NumPy.
        /// Note: In C#, -0.0 == 0.0, so we test explicit negative values.
        /// </summary>
        [Test]
        public void Laplace_NumPy_NegativeScaleRaises()
        {
            // From NumPy: assert_raises(ValueError, np.random.laplace, scale=-0.)
            // Note: -0.0 in C# is equal to 0.0, but we test with explicit negative
            Assert.ThrowsException<ArgumentException>(() => np.random.laplace(0, -1));
            Assert.ThrowsException<ArgumentException>(() => np.random.laplace(0, -0.001));
        }

        /// <summary>
        /// Migrated from NumPy test_smoke.py test_laplace
        /// Basic smoke test that laplace produces correct output size.
        /// </summary>
        [Test]
        public void Laplace_NumPy_SmokeTest()
        {
            // From NumPy: vals = rg.laplace(2.0, 2.0, 10); assert_(len(vals) == 10)
            var vals = np.random.laplace(2.0, 2.0, 10);
            Assert.AreEqual(10, vals.size);
        }

        /// <summary>
        /// Test that Laplace can produce both positive and negative values.
        /// </summary>
        [Test]
        public void Laplace_ProducesBothPositiveAndNegative()
        {
            np.random.seed(42);
            var samples = np.random.laplace(0, 1, 10000);

            bool hasPositive = false;
            bool hasNegative = false;
            foreach (var val in samples.AsIterator<double>())
            {
                if (val > 0) hasPositive = true;
                if (val < 0) hasNegative = true;
                if (hasPositive && hasNegative) break;
            }

            Assert.IsTrue(hasPositive, "Laplace(0,1) should produce positive values");
            Assert.IsTrue(hasNegative, "Laplace(0,1) should produce negative values");
        }
    }
}
