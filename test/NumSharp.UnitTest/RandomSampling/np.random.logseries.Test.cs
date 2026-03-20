using System;
using System.Linq;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.UnitTest.Utilities;
using TUnit.Core;

namespace NumSharp.UnitTest.RandomSampling
{
    /// <summary>
    /// Tests for np.random.logseries (logarithmic series distribution).
    /// Based on NumPy 2.4.2 behavior.
    /// </summary>
    public class np_random_logseries_Tests
    {
        #region Basic Functionality

        [Test]
        public void Logseries_Scalar_ReturnsNDArray()
        {
            np.random.seed(42);
            var result = np.random.logseries(0.6);

            result.Should().NotBeNull();
            result.ndim.Should().Be(0);  // Scalar
            ((long)result).Should().BeGreaterThanOrEqualTo(1);
        }

        [Test]
        public void Logseries_1D_ReturnsCorrectShape()
        {
            np.random.seed(42);
            var result = np.random.logseries(0.6, 10);

            result.Should().BeShaped(10);
            result.dtype.Should().Be(typeof(long));
        }

        [Test]
        public void Logseries_2D_ReturnsCorrectShape()
        {
            np.random.seed(42);
            var result = np.random.logseries(0.6, new int[] { 2, 3 });

            result.Should().BeShaped(2, 3);
            result.dtype.Should().Be(typeof(long));
        }

        [Test]
        public void Logseries_SizeNull_ReturnsScalar()
        {
            np.random.seed(42);
            var result = np.random.logseries(0.6, size: null);

            result.ndim.Should().Be(0);
        }

        [Test]
        public void Logseries_3DShape()
        {
            np.random.seed(42);
            var result = np.random.logseries(0.5, new Shape(2, 3, 4));

            result.Should().BeShaped(2, 3, 4);
            result.size.Should().Be(24);
        }

        #endregion

        #region Value Verification

        [Test]
        public void Logseries_AllValuesAtLeast1()
        {
            np.random.seed(42);
            var result = np.random.logseries(0.7, 1000);
            var data = result.ToArray<long>();

            foreach (var v in data)
            {
                v.Should().BeGreaterThanOrEqualTo(1);
            }
        }

        [Test]
        public void Logseries_SmallP_MostlyOnes()
        {
            // With small p, values should be mostly 1
            np.random.seed(42);
            var result = np.random.logseries(0.1, 100);
            var data = result.ToArray<long>();

            int countOnes = data.Count(v => v == 1);
            // With p=0.1, should have ~90% 1s
            countOnes.Should().BeGreaterThan(80);
        }

        [Test]
        public void Logseries_HighP_LargerValues()
        {
            // With high p close to 1, values should have more variation
            np.random.seed(42);
            var result = np.random.logseries(0.99, 100);
            var data = result.ToArray<long>();

            long maxValue = data.Max();
            // With p=0.99, should have some larger values
            maxValue.Should().BeGreaterThan(10);
        }

        [Test]
        public void Logseries_PZero_AllOnes()
        {
            // When p=0, all values should be 1
            np.random.seed(42);
            var result = np.random.logseries(0.0, 10);
            var data = result.ToArray<long>();

            foreach (var v in data)
            {
                v.Should().Be(1);
            }
        }

        [Test]
        public void Logseries_MeanApproximatesTheoretical()
        {
            // Theoretical mean = -p / ((1-p) * ln(1-p))
            double p = 0.6;
            double theoreticalMean = -p / ((1 - p) * Math.Log(1 - p));

            np.random.seed(42);
            var result = np.random.logseries(p, 100000);
            var data = result.ToArray<long>();

            double sampleMean = data.Average();
            // Should be within 5% of theoretical mean
            Math.Abs(sampleMean - theoreticalMean).Should().BeLessThan(0.1);
        }

        #endregion

        #region Validation

        [Test]
        public void Logseries_NegativeP_Throws()
        {
            Assert.ThrowsException<ArgumentException>(() => np.random.logseries(-0.1, 10));
        }

        [Test]
        public void Logseries_PEqualsOne_Throws()
        {
            Assert.ThrowsException<ArgumentException>(() => np.random.logseries(1.0, 10));
        }

        [Test]
        public void Logseries_PGreaterThanOne_Throws()
        {
            Assert.ThrowsException<ArgumentException>(() => np.random.logseries(1.5, 10));
        }

        [Test]
        public void Logseries_PIsNaN_Throws()
        {
            Assert.ThrowsException<ArgumentException>(() => np.random.logseries(double.NaN, 10));
        }

        #endregion

        #region Reproducibility

        [Test]
        public void Logseries_SameSeed_SameResults()
        {
            np.random.seed(12345);
            var result1 = np.random.logseries(0.6, 10);

            np.random.seed(12345);
            var result2 = np.random.logseries(0.6, 10);

            var data1 = result1.ToArray<long>();
            var data2 = result2.ToArray<long>();

            for (int i = 0; i < data1.Length; i++)
            {
                data1[i].Should().Be(data2[i]);
            }
        }

        [Test]
        public void Logseries_DifferentSeeds_DifferentResults()
        {
            np.random.seed(111);
            var result1 = np.random.logseries(0.6, 100);

            np.random.seed(222);
            var result2 = np.random.logseries(0.6, 100);

            var data1 = result1.ToArray<long>();
            var data2 = result2.ToArray<long>();

            // At least some values should differ
            bool anyDifferent = false;
            for (int i = 0; i < data1.Length; i++)
            {
                if (data1[i] != data2[i])
                {
                    anyDifferent = true;
                    break;
                }
            }
            anyDifferent.Should().BeTrue();
        }

        #endregion

        #region Edge Cases

        [Test]
        public void Logseries_VerySmallP()
        {
            // Edge case: very small p should still produce valid output
            np.random.seed(42);
            var result = np.random.logseries(1e-10, 10);
            var data = result.ToArray<long>();

            foreach (var v in data)
            {
                v.Should().Be(1);  // Very small p produces all 1s
            }
        }

        [Test]
        public void Logseries_PCloseToOne()
        {
            // Edge case: p very close to 1 (but less than 1)
            np.random.seed(42);
            var result = np.random.logseries(0.9999, 10);
            var data = result.ToArray<long>();

            foreach (var v in data)
            {
                v.Should().BeGreaterThanOrEqualTo(1);
            }
            // With p=0.9999, expect some fairly large values
            data.Max().Should().BeGreaterThan(1);
        }

        [Test]
        public void Logseries_Size1_ReturnsSingleElementArray()
        {
            np.random.seed(42);
            var result = np.random.logseries(0.5, 1);

            result.size.Should().Be(1);
            result.shape.Should().BeEquivalentTo(new[] { 1 });
        }

        [Test]
        public void Logseries_VariousP_AllValuesPositive()
        {
            np.random.seed(42);
            foreach (double p in new[] { 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99 })
            {
                var result = np.random.logseries(p, 100);
                var data = result.ToArray<long>();

                foreach (var v in data)
                {
                    v.Should().BeGreaterThanOrEqualTo(1);
                }
            }
        }

        #endregion
    }
}
