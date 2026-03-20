using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.RandomSampling
{
    /// <summary>
    /// Tests for np.random.rayleigh (Rayleigh distribution)
    /// </summary>
    public class NpRandomRayleighTests : TestClass
    {
        [Test]
        public void Rayleigh_1D_ReturnsCorrectShape()
        {
            var rand = np.random.rayleigh(1, 5);
            Assert.AreEqual(1, rand.ndim);
            Assert.AreEqual(5, rand.size);
        }

        [Test]
        public void Rayleigh_2D_ReturnsCorrectShape()
        {
            var rand = np.random.rayleigh(1, 5, 5);
            Assert.AreEqual(2, rand.ndim);
            Assert.AreEqual(25, rand.size);
        }

        [Test]
        public void Rayleigh_2DByShape_ReturnsCorrectShape()
        {
            var rand = np.random.rayleigh(1, new Shape(5, 5));
            Assert.AreEqual(2, rand.ndim);
            Assert.AreEqual(25, rand.size);
        }

        [Test]
        public void Rayleigh_DefaultParameters_HasCorrectStatistics()
        {
            // Rayleigh(scale=1) has mean = sqrt(pi/2) ≈ 1.253
            // and std ≈ 0.655
            np.random.seed(42);
            var samples = np.random.rayleigh(1, 100000);

            var mean = (double)np.mean(samples);
            var std = (double)np.std(samples);
            double expectedMean = Math.Sqrt(Math.PI / 2.0); // ≈ 1.2533

            // Allow some tolerance for statistical sampling
            Assert.IsTrue(Math.Abs(mean - expectedMean) < 0.05, $"Mean should be near {expectedMean}, got {mean}");
            Assert.IsTrue(Math.Abs(std - 0.655) < 0.05, $"Std should be near 0.655, got {std}");
        }

        [Test]
        public void Rayleigh_WithScale_TransformsCorrectly()
        {
            // Rayleigh(scale) has mean = scale * sqrt(pi/2)
            np.random.seed(42);
            double scale = 2.0;
            var samples = np.random.rayleigh(scale, 100000);

            var mean = (double)np.mean(samples);
            double expectedMean = scale * Math.Sqrt(Math.PI / 2.0);

            Assert.IsTrue(Math.Abs(mean - expectedMean) < 0.1, $"Mean should be near {expectedMean}, got {mean}");
        }

        [Test]
        public void Rayleigh_NegativeScale_ThrowsArgumentException()
        {
            Assert.ThrowsException<ArgumentException>(() => np.random.rayleigh(-1, 5));
        }

        [Test]
        public void Rayleigh_ScaleZero_ReturnsAllZeros()
        {
            var samples = np.random.rayleigh(0.0, 5);

            foreach (var val in samples.AsIterator<double>())
            {
                Assert.AreEqual(0.0, val, "All values should be 0 when scale=0");
            }
        }

        [Test]
        public void Rayleigh_Scalar_ReturnsShape1Array()
        {
            np.random.seed(42);
            var result = np.random.rayleigh();
            Assert.AreEqual(1, result.ndim);
            Assert.AreEqual(1, result.size);
        }

        [Test]
        public void Rayleigh_ReturnsFloat64()
        {
            var result = np.random.rayleigh(1, 5);
            Assert.AreEqual(NPTypeCode.Double, result.typecode);
        }

        [Test]
        public void Rayleigh_AllValuesPositive()
        {
            // Rayleigh distribution only produces non-negative values
            np.random.seed(42);
            var samples = np.random.rayleigh(1, 10000);

            foreach (var val in samples.AsIterator<double>())
            {
                Assert.IsTrue(val >= 0.0, $"Rayleigh values should be non-negative, got {val}");
            }
        }

        [Test]
        public void Rayleigh_SameSeed_ProducesSameResults()
        {
            np.random.seed(42);
            var samples1 = np.random.rayleigh(1, 10);

            np.random.seed(42);
            var samples2 = np.random.rayleigh(1, 10);

            for (int i = 0; i < 10; i++)
            {
                Assert.AreEqual((double)samples1[i], (double)samples2[i], $"Values at index {i} should match with same seed");
            }
        }

        [Test]
        public void Rayleigh_DifferentSeeds_ProduceDifferentResults()
        {
            np.random.seed(42);
            var samples1 = np.random.rayleigh(1, 10);

            np.random.seed(123);
            var samples2 = np.random.rayleigh(1, 10);

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

        // ========== Tests migrated from NumPy test_random.py ==========

        /// <summary>
        /// Migrated from NumPy test_random.py test_rayleigh_0
        /// Tests that scale=0 returns 0.
        /// </summary>
        [Test]
        public void Rayleigh_NumPy_ScaleZeroReturnsZero()
        {
            // From NumPy: assert_equal(np.random.rayleigh(scale=0), 0)
            var result = np.random.rayleigh(scale: 0);
            Assert.AreEqual(0.0, (double)result[0], "rayleigh(scale=0) should return 0");
        }

        /// <summary>
        /// Migrated from NumPy test_random.py test_rayleigh_0
        /// Negative scale should raise ValueError.
        /// </summary>
        [Test]
        public void Rayleigh_NumPy_NegativeScaleRaises()
        {
            // From NumPy: assert_raises(ValueError, np.random.rayleigh, scale=-0.)
            Assert.ThrowsException<ArgumentException>(() => np.random.rayleigh(-1));
            Assert.ThrowsException<ArgumentException>(() => np.random.rayleigh(-0.001));
        }

        /// <summary>
        /// Migrated from NumPy test_smoke.py test_rayleigh
        /// Basic smoke test that rayleigh produces correct output size.
        /// </summary>
        [Test]
        public void Rayleigh_NumPy_SmokeTest()
        {
            // From NumPy: vals = rg.rayleigh(0.2, 10); assert_(len(vals) == 10)
            var vals = np.random.rayleigh(0.2, 10);
            Assert.AreEqual(10, vals.size);
        }
    }
}
