using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.RandomSampling
{
    /// <summary>
    /// Tests for np.random.gumbel (Gumbel/extreme value type I distribution)
    /// </summary>
    public class NpRandomGumbelTests : TestClass
    {
        // Euler-Mascheroni constant
        private const double EulerGamma = 0.5772156649015329;

        [Test]
        public void Gumbel_1D_ReturnsCorrectShape()
        {
            var rand = np.random.gumbel(0, 1, 5);
            Assert.AreEqual(1, rand.ndim);
            Assert.AreEqual(5, rand.size);
        }

        [Test]
        public void Gumbel_2D_ReturnsCorrectShape()
        {
            var rand = np.random.gumbel(0, 1, new Shape(5, 5));
            Assert.AreEqual(2, rand.ndim);
            Assert.AreEqual(25, rand.size);
        }

        [Test]
        public void Gumbel_2DByShape_ReturnsCorrectShape()
        {
            var rand = np.random.gumbel(0, 1, new Shape(5, 5));
            Assert.AreEqual(2, rand.ndim);
            Assert.AreEqual(25, rand.size);
        }

        [Test]
        public void Gumbel_DefaultParameters_HasCorrectStatistics()
        {
            // Gumbel(0, 1) has mean = γ ≈ 0.5772 (Euler-Mascheroni constant)
            // and std = π / sqrt(6) ≈ 1.283
            np.random.seed(42);
            var samples = np.random.gumbel(0, 1, 100000);

            var mean = (double)np.mean(samples);
            var std = (double)np.std(samples);
            double expectedStd = Math.PI / Math.Sqrt(6); // ≈ 1.2825

            // Allow some tolerance for statistical sampling
            Assert.IsTrue(Math.Abs(mean - EulerGamma) < 0.05, $"Mean should be near {EulerGamma}, got {mean}");
            Assert.IsTrue(Math.Abs(std - expectedStd) < 0.05, $"Std should be near {expectedStd}, got {std}");
        }

        [Test]
        public void Gumbel_WithLocScale_TransformsCorrectly()
        {
            // Gumbel(loc, scale) has mean = loc + scale * γ
            np.random.seed(42);
            double loc = 2.0;
            double scale = 3.0;
            var samples = np.random.gumbel(loc, scale, 100000);

            var mean = (double)np.mean(samples);
            double expectedMean = loc + scale * EulerGamma;

            Assert.IsTrue(Math.Abs(mean - expectedMean) < 0.15, $"Mean should be near {expectedMean}, got {mean}");
        }

        [Test]
        public void Gumbel_NegativeScale_ThrowsArgumentException()
        {
            Assert.ThrowsException<ArgumentException>(() => np.random.gumbel(0, -1, 5));
        }

        [Test]
        public void Gumbel_ScaleZero_ReturnsConstantAtLoc()
        {
            double loc = 5.0;
            var samples = np.random.gumbel(loc, 0.0, 5);

            foreach (var val in samples.AsIterator<double>())
            {
                Assert.AreEqual(loc, val, $"All values should be {loc} when scale=0");
            }
        }

        [Test]
        public void Gumbel_Scalar_ReturnsScalar()
        {
            np.random.seed(42);
            var result = np.random.gumbel();
            // NumPy returns a scalar (0-dimensional) when no size is given
            Assert.AreEqual(0, result.ndim);
            Assert.AreEqual(1, result.size);
        }

        [Test]
        public void Gumbel_ReturnsFloat64()
        {
            var result = np.random.gumbel(0, 1, 5);
            Assert.AreEqual(NPTypeCode.Double, result.typecode);
        }

        [Test]
        public void Gumbel_SameSeed_ProducesSameResults()
        {
            np.random.seed(42);
            var samples1 = np.random.gumbel(0, 1, 10);

            np.random.seed(42);
            var samples2 = np.random.gumbel(0, 1, 10);

            for (int i = 0; i < 10; i++)
            {
                Assert.AreEqual((double)samples1[i], (double)samples2[i], $"Values at index {i} should match with same seed");
            }
        }

        [Test]
        public void Gumbel_DifferentSeeds_ProduceDifferentResults()
        {
            np.random.seed(42);
            var samples1 = np.random.gumbel(0, 1, 10);

            np.random.seed(123);
            var samples2 = np.random.gumbel(0, 1, 10);

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
        public void Gumbel_CanProduceNegativeValues()
        {
            // Gumbel distribution can produce negative values (unlike Rayleigh)
            np.random.seed(42);
            var samples = np.random.gumbel(0, 1, 10000);

            bool hasNegative = false;
            foreach (var val in samples.AsIterator<double>())
            {
                if (val < 0)
                {
                    hasNegative = true;
                    break;
                }
            }
            Assert.IsTrue(hasNegative, "Gumbel(0,1) should produce some negative values");
        }

        // ========== Tests migrated from NumPy test_random.py ==========

        /// <summary>
        /// Migrated from NumPy test_random.py test_gumbel_0
        /// Tests that scale=0 returns loc (default 0).
        /// </summary>
        [Test]
        public void Gumbel_NumPy_ScaleZeroReturnsLoc()
        {
            // From NumPy: assert_equal(np.random.gumbel(scale=0), 0)
            var result = np.random.gumbel(scale: 0);
            Assert.AreEqual(0.0, result.GetDouble(0), "gumbel(scale=0) should return 0 (the default loc)");
        }

        /// <summary>
        /// Migrated from NumPy test_random.py test_gumbel_0
        /// Negative scale should raise ValueError.
        /// </summary>
        [Test]
        public void Gumbel_NumPy_NegativeScaleRaises()
        {
            // From NumPy: assert_raises(ValueError, np.random.gumbel, scale=-0.)
            Assert.ThrowsException<ArgumentException>(() => np.random.gumbel(0, -1));
            Assert.ThrowsException<ArgumentException>(() => np.random.gumbel(0, -0.001));
        }

        /// <summary>
        /// Migrated from NumPy test_smoke.py test_gumbel
        /// Basic smoke test that gumbel produces correct output size.
        /// </summary>
        [Test]
        public void Gumbel_NumPy_SmokeTest()
        {
            // From NumPy: vals = rg.gumbel(2.0, 2.0, 10); assert_(len(vals) == 10)
            var vals = np.random.gumbel(2.0, 2.0, 10);
            Assert.AreEqual(10, vals.size);
        }
    }
}
