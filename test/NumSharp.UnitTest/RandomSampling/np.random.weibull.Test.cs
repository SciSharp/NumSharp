using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace NumSharp.UnitTest.RandomSampling
{
    
    [TestClass]
    public class NpRandomWeibullTest : TestClass
    {
        [TestMethod]
        public void Weibull_ScalarReturn()
        {
            var rng = np.random.RandomState(42);
            var result = rng.weibull(1.5);

            Assert.AreEqual(0, result.ndim);
            Assert.AreEqual(1, result.size);
            Assert.IsTrue((double)result > 0);
        }

        [TestMethod]
        public void Weibull_1DArray()
        {
            var result = np.random.weibull(2, new Shape(5));

            Assert.AreEqual(1, result.ndim);
            Assert.AreEqual(5, result.shape[0]);
            Assert.AreEqual(typeof(double), result.dtype);
        }

        [TestMethod]
        public void Weibull_2DArray()
        {
            var result = np.random.weibull(2, new Shape(2, 3));

            Assert.AreEqual(2, result.ndim);
            Assert.AreEqual(2, result.shape[0]);
            Assert.AreEqual(3, result.shape[1]);
        }

        [TestMethod]
        public void Weibull_ShapeOverload()
        {
            var result = np.random.weibull(1.5, new Shape(10, 5));

            Assert.AreEqual(10, result.shape[0]);
            Assert.AreEqual(5, result.shape[1]);
        }

        [TestMethod]
        public void Weibull_NegativeA_ThrowsArgumentException()
        {
            Assert.ThrowsException<ArgumentException>(() => np.random.weibull(-1));
        }

        [TestMethod]
        public void Weibull_ZeroA_ReturnsZeros()
        {
            var result = np.random.weibull(0, 5);

            foreach (var val in result.AsIterator<double>())
                Assert.AreEqual(0.0, val);
        }

        [TestMethod]
        public void Weibull_A1_IsExponential()
        {
            // When a=1, Weibull reduces to exponential distribution with mean=1
            var rng = np.random.RandomState(42);
            var samples = rng.weibull(1, 100000);
            var mean = (double)np.mean(samples);

            // Exponential(1) has mean=1
            Assert.IsTrue(Math.Abs(mean - 1.0) < 0.05, $"Mean {mean} should be close to 1.0");
        }

        [TestMethod]
        public void Weibull_A2_StatisticalProperties()
        {
            // For Weibull(a=2), mean = Gamma(1 + 1/a) = Gamma(1.5) ≈ 0.886
            var rng = np.random.RandomState(42);
            var samples = rng.weibull(2, 100000);
            var mean = (double)np.mean(samples);

            Assert.IsTrue(Math.Abs(mean - 0.886) < 0.02, $"Mean {mean} should be close to 0.886");
        }

        [TestMethod]
        public void Weibull_AllValuesNonNegative()
        {
            var samples = np.random.weibull(0.5, 1000);

            foreach (var val in samples.AsIterator<double>())
                Assert.IsTrue(val >= 0, $"Value {val} should be non-negative");
        }

        [TestMethod]
        public void Weibull_SmallA_ProducesLargeValues()
        {
            // Small a (shape < 1) produces heavy-tailed distribution
            var rng = np.random.RandomState(42);
            var samples = rng.weibull(0.5, 1000);
            var max = (double)np.amax(samples);

            // Should have some large values due to heavy tail
            Assert.IsTrue(max > 1.0, $"Max {max} should be > 1 for heavy-tailed distribution");
        }

        [TestMethod]
        public void Weibull_LargeA_ProducesConcentratedValues()
        {
            // Large a produces values concentrated near 1
            var rng = np.random.RandomState(42);
            var samples = rng.weibull(10, 1000);
            var mean = (double)np.mean(samples);
            var std = (double)np.std(samples);

            // Mean should be close to 1, std should be small
            Assert.IsTrue(Math.Abs(mean - 0.95) < 0.1, $"Mean {mean} should be near 0.95");
            Assert.IsTrue(std < 0.2, $"Std {std} should be small for large a");
        }

        [TestMethod]
        public void Weibull_ReturnsFloat64()
        {
            var result = np.random.weibull(2, size: new Shape(5));

            Assert.AreEqual(typeof(double), result.dtype);
        }
    }
}
