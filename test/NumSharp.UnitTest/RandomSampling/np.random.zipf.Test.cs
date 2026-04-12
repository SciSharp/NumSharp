using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.RandomSampling
{
    /// <summary>
    /// Tests for np.random.zipf (Zipf/zeta distribution)
    /// </summary>
    
    public class NpRandomZipfTests : TestClass
    {
        [Test]
        public void Zipf_1D_ReturnsCorrectShape()
        {
            var rand = np.random.zipf(2, 5);
            Assert.AreEqual(1, rand.ndim);
            Assert.AreEqual(5, rand.size);
        }

        [Test]
        public void Zipf_2D_ReturnsCorrectShape()
        {
            var rand = np.random.zipf(2, new Shape(5, 5));
            Assert.AreEqual(2, rand.ndim);
            Assert.AreEqual(25, rand.size);
        }

        [Test]
        public void Zipf_2DByShape_ReturnsCorrectShape()
        {
            var rand = np.random.zipf(2, new Shape(5, 5));
            Assert.AreEqual(2, rand.ndim);
            Assert.AreEqual(25, rand.size);
        }

        [Test]
        public void Zipf_ReturnsInt64()
        {
            var result = np.random.zipf(2, 5);
            Assert.AreEqual(NPTypeCode.Int64, result.typecode);
        }

        [Test]
        public void Zipf_AllValuesPositive()
        {
            // Zipf distribution produces positive integers >= 1
            var rng = np.random.RandomState(42);
            var samples = rng.zipf(2, 10000);

            foreach (var val in samples.AsIterator<long>())
            {
                Assert.IsTrue(val >= 1L, $"Zipf values should be >= 1, got {val}");
            }
        }

        [Test]
        public void Zipf_MinValueIsOne()
        {
            // Minimum value should be 1
            var rng = np.random.RandomState(42);
            var samples = rng.zipf(2, 10000);
            long min = long.MaxValue;
            foreach (var val in samples.AsIterator<long>())
            {
                if (val < min) min = val;
            }
            Assert.AreEqual(1L, min, "Minimum Zipf value should be 1");
        }

        [Test]
        public void Zipf_LargeA_ReturnsAllOnes()
        {
            // For very large a (>= 1025), NumPy returns 1
            var rng = np.random.RandomState(42);
            var samples = rng.zipf(2000, 100);

            foreach (var val in samples.AsIterator<long>())
            {
                Assert.AreEqual(1L, val, "For large a, all values should be 1");
            }
        }

        [Test]
        public void Zipf_Scalar_ReturnsScalar()
        {
            var rng = np.random.RandomState(42);
            var result = rng.zipf(2);
            // NumPy returns a scalar (0-dimensional) when no size is given
            Assert.AreEqual(0, result.ndim);
            Assert.AreEqual(1, result.size);
        }

        [Test]
        public void Zipf_SameSeed_ProducesSameResults()
        {
            var rng1 = np.random.RandomState(42);
            var samples1 = rng1.zipf(2, 10);

            var rng2 = np.random.RandomState(42);
            var samples2 = rng2.zipf(2, 10);

            for (int i = 0; i < 10; i++)
            {
                Assert.AreEqual((long)samples1[i], (long)samples2[i], $"Values at index {i} should match with same seed");
            }
        }

        [Test]
        public void Zipf_DifferentSeeds_ProduceDifferentResults()
        {
            var rng1 = np.random.RandomState(42);
            var samples1 = rng1.zipf(2, 10);

            var rng2 = np.random.RandomState(123);
            var samples2 = rng2.zipf(2, 10);

            bool anyDifferent = false;
            for (int i = 0; i < 10; i++)
            {
                if ((long)samples1[i] != (long)samples2[i])
                {
                    anyDifferent = true;
                    break;
                }
            }
            Assert.IsTrue(anyDifferent, "Different seeds should produce different results");
        }

        // ========== Validation Tests ==========

        [Test]
        public void Zipf_AEqualsOne_ThrowsArgumentException()
        {
            // a must be > 1
            Assert.ThrowsException<ArgumentException>(() => np.random.zipf(1.0, 5));
        }

        [Test]
        public void Zipf_ALessThanOne_ThrowsArgumentException()
        {
            Assert.ThrowsException<ArgumentException>(() => np.random.zipf(0.5, 5));
            Assert.ThrowsException<ArgumentException>(() => np.random.zipf(0, 5));
            Assert.ThrowsException<ArgumentException>(() => np.random.zipf(-1, 5));
        }

        [Test]
        public void Zipf_ANaN_ThrowsArgumentException()
        {
            Assert.ThrowsException<ArgumentException>(() => np.random.zipf(double.NaN, 5));
        }

        // ========== Tests migrated from NumPy ==========

        /// <summary>
        /// Migrated from NumPy test_smoke.py test_zipf
        /// Basic smoke test.
        /// </summary>
        [Test]
        public void Zipf_NumPy_SmokeTest()
        {
            // From NumPy: vals = rg.zipf(10, 10); assert_(len(vals) == 10)
            var vals = np.random.zipf(10, 10);
            Assert.AreEqual(10, vals.size);
        }

        /// <summary>
        /// Migrated from NumPy test_generator_mt19937_regressions.py test_zipf_large_parameter
        /// Large a should return all ones and not hang.
        /// </summary>
        [Test]
        public void Zipf_NumPy_LargeParameter()
        {
            // From NumPy: sample = mt19937.zipf(10000, size=n); assert_array_equal(sample, np.ones(n, dtype=np.int64))
            int n = 8;
            var sample = np.random.zipf(10000, n);
            foreach (var val in sample.AsIterator<long>())
            {
                Assert.AreEqual(1L, val, "zipf(10000) should return all 1s");
            }
        }

        /// <summary>
        /// Migrated from NumPy test_random.py TestBroadcast test_zipf
        /// Tests validation of bad a values.
        /// </summary>
        [Test]
        public void Zipf_NumPy_BadAThrows()
        {
            // From NumPy: assert_raises(ValueError, zipf, bad_a * 3) where bad_a = [0]
            Assert.ThrowsException<ArgumentException>(() => np.random.zipf(0, 3));
        }

        /// <summary>
        /// Test different a values produce different distributions.
        /// Small a = heavy tail (larger values more likely)
        /// Large a = light tail (mostly 1s)
        /// </summary>
        [Test]
        public void Zipf_DifferentA_DifferentDistributions()
        {
            var rng1 = np.random.RandomState(42);
            var smallA = rng1.zipf(1.5, 1000);  // Heavy tail

            var largeA = rng1.zipf(5, 1000);    // Light tail

            // Small a should have larger mean (more big values)
            double smallMean = 0;
            foreach (var v in smallA.AsIterator<long>()) smallMean += v;
            smallMean /= smallA.size;

            double largeMean = 0;
            foreach (var v in largeA.AsIterator<long>()) largeMean += v;
            largeMean /= largeA.size;

            // Large a distribution should be more concentrated near 1
            Assert.IsTrue(largeMean < smallMean, $"Large a should have smaller mean. small={smallMean}, large={largeMean}");
        }
    }
}
