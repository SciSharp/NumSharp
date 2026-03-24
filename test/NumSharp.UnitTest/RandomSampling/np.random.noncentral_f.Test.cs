using System;
using System.Linq;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.RandomSampling
{
    public class NoncentralFTests
    {
        [Test]
        public void NoncentralF_ReturnsScalar_WhenNoSize()
        {
            np.random.seed(42);
            var result = np.random.noncentral_f(5, 10, 2);

            // NumPy returns a scalar (0-dimensional) when no size is given
            result.ndim.Should().Be(0);
            result.size.Should().Be(1);
            result.dtype.Should().Be(typeof(double));
        }

        [Test]
        public void NoncentralF_Returns1DArray()
        {
            np.random.seed(42);
            var result = np.random.noncentral_f(5, 10, 2, 5);

            result.shape.Should().ContainInOrder(5);
            result.size.Should().Be(5);
        }

        [Test]
        public void NoncentralF_Returns2DArray()
        {
            np.random.seed(42);
            var result = np.random.noncentral_f(5, 10, 2, new[] { 2, 3 });

            result.shape.Should().ContainInOrder(2, 3);
            result.size.Should().Be(6);
        }

        [Test]
        public void NoncentralF_AllValuesPositive()
        {
            np.random.seed(42);
            var samples = np.random.noncentral_f(5, 10, 2, 10000);

            var min = (double)np.amin(samples);
            min.Should().BeGreaterThan(0.0);
        }

        [Test]
        public void NoncentralF_NoncZero_IsCentralF()
        {
            // When nonc=0, should behave like central F distribution
            np.random.seed(42);
            var samples = np.random.noncentral_f(5, 10, 0, 10000);

            // All should be positive
            var min = (double)np.amin(samples);
            min.Should().BeGreaterThan(0.0);

            // Mean of F(dfnum, dfden) = dfden / (dfden - 2) for dfden > 2
            // For dfnum=5, dfden=10: mean = 10/8 = 1.25
            var mean = (double)np.mean(samples);
            Math.Abs(mean - 1.25).Should().BeLessThan(0.1);
        }

        [Test]
        public void NoncentralF_LargeNonc_IncreasesValues()
        {
            np.random.seed(42);
            var samplesSmall = np.random.noncentral_f(5, 10, 1, 10000);
            np.random.seed(42);
            var samplesLarge = np.random.noncentral_f(5, 10, 10, 10000);

            var meanSmall = (double)np.mean(samplesSmall);
            var meanLarge = (double)np.mean(samplesLarge);

            // Larger nonc should give larger mean
            meanLarge.Should().BeGreaterThan(meanSmall);
        }

        [Test]
        public void NoncentralF_DifferentDf_Works()
        {
            np.random.seed(42);
            var samples = np.random.noncentral_f(3, 20, 3, 1000);

            samples.size.Should().Be(1000);
            var min = (double)np.amin(samples);
            min.Should().BeGreaterThan(0.0);
        }

        [Test]
        public void NoncentralF_DfnumZero_ThrowsArgumentException()
        {
            Action act = () => np.random.noncentral_f(0, 10, 2);
            act.Should().Throw<ArgumentException>();
        }

        [Test]
        public void NoncentralF_DfnumNegative_ThrowsArgumentException()
        {
            Action act = () => np.random.noncentral_f(-1, 10, 2);
            act.Should().Throw<ArgumentException>();
        }

        [Test]
        public void NoncentralF_DfdenZero_ThrowsArgumentException()
        {
            Action act = () => np.random.noncentral_f(5, 0, 2);
            act.Should().Throw<ArgumentException>();
        }

        [Test]
        public void NoncentralF_DfdenNegative_ThrowsArgumentException()
        {
            Action act = () => np.random.noncentral_f(5, -1, 2);
            act.Should().Throw<ArgumentException>();
        }

        [Test]
        public void NoncentralF_NoncNegative_ThrowsArgumentException()
        {
            Action act = () => np.random.noncentral_f(5, 10, -1);
            act.Should().Throw<ArgumentException>();
        }

        [Test]
        public void NoncentralF_ShapeOverload_Works()
        {
            np.random.seed(42);
            var result = np.random.noncentral_f(5, 10, 2, new Shape(3, 4));

            result.shape.Should().ContainInOrder(3, 4);
        }

        [Test]
        public void NoncentralF_Reproducibility_WithSeed()
        {
            np.random.seed(42);
            var first = np.random.noncentral_f(5, 10, 2, 5).ToArray<double>();

            np.random.seed(42);
            var second = np.random.noncentral_f(5, 10, 2, 5).ToArray<double>();

            first.Should().BeEquivalentTo(second);
        }

        [Test]
        public void NoncentralF_SmallDf_Works()
        {
            // df < 1 uses different algorithm branch in noncentral chisquare
            np.random.seed(42);
            var samples = np.random.noncentral_f(0.5, 10, 2, 1000);

            samples.size.Should().Be(1000);
            var min = (double)np.amin(samples);
            min.Should().BeGreaterThanOrEqualTo(0.0);
        }
    }
}
