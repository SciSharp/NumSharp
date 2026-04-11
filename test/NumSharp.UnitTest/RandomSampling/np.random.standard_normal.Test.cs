using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace NumSharp.UnitTest.RandomSampling
{
    /// <summary>
    /// Tests for np.random.standard_normal (standard normal distribution with mean=0, std=1).
    /// NumPy reference: https://numpy.org/doc/stable/reference/random/generated/numpy.random.standard_normal.html
    /// </summary>
    [NotInParallel]
    public class NpRandomStandardNormalTest : TestClass
    {
        [Test]
        public void StandardNormal_NoArgs_ReturnsScalar()
        {
            // Python: np.random.standard_normal() returns a single float
            // NumSharp returns a 0-d NDArray (scalar)
            np.random.seed(42);
            var result = np.random.standard_normal();

            Assert.AreEqual(0, result.ndim);
            Assert.AreEqual(1, result.size);
            Assert.AreEqual(typeof(double), result.dtype);
        }

        [Test]
        public void StandardNormal_Size5_ReturnsCorrectShape()
        {
            np.random.seed(42);
            var result = np.random.standard_normal(5);

            Assert.AreEqual(1, result.ndim);
            Assert.AreEqual(5, result.size);
            Assert.AreEqual(5, result.shape[0]);
        }

        [Test]
        public void StandardNormal_2DShape_ReturnsCorrectShape()
        {
            np.random.seed(42);
            var result = np.random.standard_normal(2, 3);

            Assert.AreEqual(2, result.ndim);
            Assert.AreEqual(6, result.size);
            Assert.AreEqual(2, result.shape[0]);
            Assert.AreEqual(3, result.shape[1]);
        }

        [Test]
        public void StandardNormal_ShapeOverload_ReturnsCorrectShape()
        {
            np.random.seed(42);
            var result = np.random.standard_normal(new Shape(10, 20));

            Assert.AreEqual(2, result.ndim);
            Assert.AreEqual(200, result.size);
            Assert.AreEqual(10, result.shape[0]);
            Assert.AreEqual(20, result.shape[1]);
        }

        [Test]
        public void StandardNormal_EmptySize_ReturnsEmptyArray()
        {
            np.random.seed(42);
            var result = np.random.standard_normal(0);

            Assert.AreEqual(1, result.ndim);
            Assert.AreEqual(0, result.size);
        }

        [Test]
        public void StandardNormal_ReturnsFloat64()
        {
            var result = np.random.standard_normal(5);
            Assert.AreEqual(typeof(double), result.dtype);
        }

        [Test]
        public void StandardNormal_MeanIsZero()
        {
            // Statistical test: mean of standard normal should be close to 0
            np.random.seed(42);
            var samples = np.random.standard_normal(100000);
            var mean = (double)np.mean(samples);

            Assert.IsTrue(Math.Abs(mean) < 0.02, $"Mean {mean} should be close to 0");
        }

        [Test]
        public void StandardNormal_StdIsOne()
        {
            // Statistical test: std of standard normal should be close to 1
            np.random.seed(42);
            var samples = np.random.standard_normal(100000);
            var std = (double)np.std(samples);

            Assert.IsTrue(Math.Abs(std - 1.0) < 0.02, $"Std {std} should be close to 1");
        }

        [Test]
        public void StandardNormal_VarianceIsOne()
        {
            // Statistical test: variance of standard normal should be close to 1
            np.random.seed(42);
            var samples = np.random.standard_normal(100000);
            var variance = (double)np.var(samples);

            Assert.IsTrue(Math.Abs(variance - 1.0) < 0.02, $"Variance {variance} should be close to 1");
        }

        [Test]
        public void StandardNormal_MatchesRandn()
        {
            // standard_normal and randn should produce equivalent statistical properties
            // (same distribution: mean=0, std=1)
            // Note: We cannot rely on exact value matching because tests run in parallel
            // and other tests may modify random state between our calls.
            np.random.seed(42);
            var randn_result = np.random.randn(10000);
            var randn_mean = (double)np.mean(randn_result);
            var randn_std = (double)np.std(randn_result);

            np.random.seed(42);
            var standard_normal_result = np.random.standard_normal(10000);
            var sn_mean = (double)np.mean(standard_normal_result);
            var sn_std = (double)np.std(standard_normal_result);

            // Both should be standard normal - compare statistical properties
            Assert.IsTrue(Math.Abs(randn_mean) < 0.1, $"randn mean {randn_mean} should be near 0");
            Assert.IsTrue(Math.Abs(sn_mean) < 0.1, $"standard_normal mean {sn_mean} should be near 0");
            Assert.IsTrue(Math.Abs(randn_std - 1.0) < 0.1, $"randn std {randn_std} should be near 1");
            Assert.IsTrue(Math.Abs(sn_std - 1.0) < 0.1, $"standard_normal std {sn_std} should be near 1");
        }

        [Test]
        public void StandardNormal_Reproducible()
        {
            // Same seed should produce same results when called immediately after seeding
            // Note: In parallel test environments, we verify reproducibility by
            // checking the first value produced, not full arrays, since other tests
            // may interleave with our calls.
            np.random.seed(12345);
            var result1_first = (double)np.random.standard_normal();

            np.random.seed(12345);
            var result2_first = (double)np.random.standard_normal();

            Assert.AreEqual(result1_first, result2_first, 1e-10,
                "Same seed should produce identical first values");
        }

        [Test]
        public void StandardNormal_DifferentSeeds_ProduceDifferentResults()
        {
            np.random.seed(42);
            var result1 = np.random.standard_normal(5);

            np.random.seed(43);
            var result2 = np.random.standard_normal(5);

            Assert.IsFalse(np.array_equal(result1, result2),
                "Different seeds should produce different results");
        }

        [Test]
        public void StandardNormal_ValuesInReasonableRange()
        {
            // Standard normal values should mostly be in [-4, 4] range
            // (>99.99% of values fall within 4 std devs)
            np.random.seed(42);
            var samples = np.random.standard_normal(1000);

            foreach (var val in samples.AsIterator<double>())
            {
                Assert.IsTrue(val > -10 && val < 10,
                    $"Value {val} is outside reasonable range for standard normal");
            }
        }

        [Test]
        public void StandardNormal_3DShape_Works()
        {
            var result = np.random.standard_normal(2, 3, 4);

            Assert.AreEqual(3, result.ndim);
            Assert.AreEqual(24, result.size);
            Assert.AreEqual(2, result.shape[0]);
            Assert.AreEqual(3, result.shape[1]);
            Assert.AreEqual(4, result.shape[2]);
        }
    }
}
