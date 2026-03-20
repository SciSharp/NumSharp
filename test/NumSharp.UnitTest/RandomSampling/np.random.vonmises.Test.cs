using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.RandomSampling
{
    /// <summary>
    /// Tests for np.random.vonmises (von Mises / circular normal distribution).
    /// Based on NumPy 2.4.2 behavior.
    /// </summary>
    public class np_random_vonmises_Tests : TestClass
    {
        #region Basic Functionality

        [Test]
        public void VonMises_Scalar_ReturnsNDArray()
        {
            np.random.seed(42);
            var result = np.random.vonmises(0, 1);

            result.Should().NotBeNull();
            result.ndim.Should().Be(0);  // Scalar
        }

        [Test]
        public void VonMises_1D_ReturnsCorrectShape()
        {
            np.random.seed(42);
            var result = np.random.vonmises(0, 1, 5);

            result.Should().BeShaped(5);
            result.dtype.Should().Be(typeof(double));
        }

        [Test]
        public void VonMises_2D_ReturnsCorrectShape()
        {
            np.random.seed(42);
            var result = np.random.vonmises(0, 1, new int[] { 2, 3 });

            result.Should().BeShaped(2, 3);
            result.dtype.Should().Be(typeof(double));
        }

        [Test]
        public void VonMises_SizeNull_ReturnsScalar()
        {
            np.random.seed(42);
            var result = np.random.vonmises(0, 1, size: null);

            result.ndim.Should().Be(0);
        }

        #endregion

        #region Range Verification

        [Test]
        public void VonMises_ResultsInRange_MinusPiToPi()
        {
            // All results should be in [-pi, pi]
            np.random.seed(42);
            var result = np.random.vonmises(0, 1, 10000);
            var data = result.ToArray<double>();

            foreach (var v in data)
            {
                v.Should().BeGreaterThanOrEqualTo(-Math.PI);
                v.Should().BeLessThanOrEqualTo(Math.PI);
            }
        }

        [Test]
        public void VonMises_Kappa0_UniformOnCircle()
        {
            // kappa=0 gives uniform distribution on [-pi, pi]
            np.random.seed(42);
            var result = np.random.vonmises(0, 0, 10000);
            var data = result.ToArray<double>();

            // All in range
            foreach (var v in data)
            {
                v.Should().BeGreaterThanOrEqualTo(-Math.PI);
                v.Should().BeLessThanOrEqualTo(Math.PI);
            }

            // Mean should be close to 0 for uniform on symmetric interval
            double mean = 0;
            foreach (var v in data) mean += v;
            mean /= data.Length;
            Math.Abs(mean).Should().BeLessThan(0.1);
        }

        [Test]
        public void VonMises_HighKappa_ConcentratedAroundMu()
        {
            // High kappa = concentrated around mu
            np.random.seed(42);
            var result = np.random.vonmises(0, 100, 10000);
            var data = result.ToArray<double>();

            double mean = 0;
            foreach (var v in data) mean += v;
            mean /= data.Length;

            // Mean should be close to mu=0
            Math.Abs(mean).Should().BeLessThan(0.05);

            // Standard deviation should be small
            double variance = 0;
            foreach (var v in data) variance += (v - mean) * (v - mean);
            variance /= data.Length;
            double std = Math.Sqrt(variance);
            std.Should().BeLessThan(0.15);
        }

        [Test]
        public void VonMises_HighKappa_ConcentratedAroundMuPiHalf()
        {
            // High kappa around mu=pi/2
            np.random.seed(42);
            var result = np.random.vonmises(Math.PI / 2, 100, 10000);
            var data = result.ToArray<double>();

            double mean = 0;
            foreach (var v in data) mean += v;
            mean /= data.Length;

            // Mean should be close to pi/2 (~1.571)
            Math.Abs(mean - Math.PI / 2).Should().BeLessThan(0.05);
        }

        #endregion

        #region Kappa Parameter Ranges

        [Test]
        public void VonMises_VerySmallKappa_NearUniform()
        {
            // kappa < 1e-8 uses uniform distribution
            np.random.seed(42);
            var result = np.random.vonmises(0, 1e-9, 10000);
            var data = result.ToArray<double>();

            // All in range
            foreach (var v in data)
            {
                v.Should().BeGreaterThanOrEqualTo(-Math.PI);
                v.Should().BeLessThanOrEqualTo(Math.PI);
            }
        }

        [Test]
        public void VonMises_VeryHighKappa_WrappedNormalFallback()
        {
            // kappa > 1e6 uses wrapped normal approximation
            np.random.seed(42);
            var result = np.random.vonmises(0, 1e7, 1000);
            var data = result.ToArray<double>();

            // All in range
            foreach (var v in data)
            {
                v.Should().BeGreaterThanOrEqualTo(-Math.PI);
                v.Should().BeLessThanOrEqualTo(Math.PI);
            }

            // Should be very concentrated around 0
            double mean = 0;
            foreach (var v in data) mean += v;
            mean /= data.Length;
            Math.Abs(mean).Should().BeLessThan(0.01);
        }

        [Test]
        public void VonMises_VariousKappaValues_AllInRange()
        {
            np.random.seed(42);
            foreach (double kappa in new[] { 0.0, 0.1, 1.0, 10.0, 100.0, 1000.0 })
            {
                var result = np.random.vonmises(0, kappa, 1000);
                var data = result.ToArray<double>();

                foreach (var v in data)
                {
                    v.Should().BeGreaterThanOrEqualTo(-Math.PI);
                    v.Should().BeLessThanOrEqualTo(Math.PI);
                }
            }
        }

        #endregion

        #region Validation

        [Test]
        public void VonMises_NegativeKappa_ThrowsArgumentException()
        {
            Assert.ThrowsException<ArgumentException>(() => np.random.vonmises(0, -1));
        }

        [Test]
        public void VonMises_SmallNegativeKappa_ThrowsArgumentException()
        {
            Assert.ThrowsException<ArgumentException>(() => np.random.vonmises(0, -0.001));
        }

        #endregion

        #region Mu Parameter

        [Test]
        public void VonMises_DifferentMu_MeanFollowsMu()
        {
            np.random.seed(42);
            foreach (double mu in new[] { 0.0, Math.PI / 4, Math.PI / 2, -Math.PI / 2 })
            {
                var result = np.random.vonmises(mu, 10, 5000);
                var data = result.ToArray<double>();

                double mean = 0;
                foreach (var v in data) mean += v;
                mean /= data.Length;

                // Mean should be close to mu
                Math.Abs(mean - mu).Should().BeLessThan(0.1);
            }
        }

        [Test]
        public void VonMises_MuAtPi_WrapsCorrectly()
        {
            // mu = pi should wrap around properly
            np.random.seed(42);
            var result = np.random.vonmises(Math.PI, 10, 5000);
            var data = result.ToArray<double>();

            // All in range
            foreach (var v in data)
            {
                v.Should().BeGreaterThanOrEqualTo(-Math.PI);
                v.Should().BeLessThanOrEqualTo(Math.PI);
            }
        }

        #endregion

        #region Edge Cases

        [Test]
        public void VonMises_EmptyShape_ReturnsEmptyArray()
        {
            np.random.seed(42);
            var result = np.random.vonmises(0, 1, new int[] { 0 });

            result.size.Should().Be(0);
        }

        [Test]
        public void VonMises_Size1_ReturnsSingleElementArray()
        {
            np.random.seed(42);
            var result = np.random.vonmises(0, 1, 1);

            result.size.Should().Be(1);
        }

        [Test]
        public void VonMises_Reproducibility_SameSeedSameResults()
        {
            np.random.seed(42);
            var result1 = np.random.vonmises(0, 1, 10);

            np.random.seed(42);
            var result2 = np.random.vonmises(0, 1, 10);

            var data1 = result1.ToArray<double>();
            var data2 = result2.ToArray<double>();

            for (int i = 0; i < data1.Length; i++)
            {
                data1[i].Should().Be(data2[i]);
            }
        }

        #endregion
    }
}
