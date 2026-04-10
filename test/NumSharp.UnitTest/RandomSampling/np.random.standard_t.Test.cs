using System;
using System.Linq;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.RandomSampling
{
    public class StandardTTests
    {
        [Test]
        public void StandardT_ReturnsScalar_WhenNoSize()
        {
            np.random.seed(42);
            var result = np.random.standard_t(10);

            // NumPy returns a scalar (0-dimensional) when no size is given
            result.ndim.Should().Be(0);
            result.size.Should().Be(1);
            result.dtype.Should().Be(typeof(double));
        }

        [Test]
        public void StandardT_Returns1DArray()
        {
            np.random.seed(42);
            var result = np.random.standard_t(10, 5L);

            result.shape.Should().ContainInOrder(5);
            result.size.Should().Be(5);
        }

        [Test]
        public void StandardT_Returns2DArray()
        {
            np.random.seed(42);
            var result = np.random.standard_t(10, new[] { 2, 3 });

            result.shape.Should().ContainInOrder(2, 3);
            result.size.Should().Be(6);
        }

        [Test]
        public void StandardT_HasMeanNearZero()
        {
            // For t distribution with df > 1: E[X] = 0
            np.random.seed(42);
            var samples = np.random.standard_t(10, 100000L);

            var mean = (double)np.mean(samples);
            // Mean should be close to 0
            Math.Abs(mean).Should().BeLessThan(0.05);
        }

        [Test]
        public void StandardT_HasExpectedStd_df10()
        {
            // For t distribution: Var[X] = df/(df-2) for df > 2
            // df=10: Var = 10/8 = 1.25, Std = sqrt(1.25) ≈ 1.118
            np.random.seed(42);
            var samples = np.random.standard_t(10, 100000L);

            var std = (double)np.std(samples);
            var expectedStd = Math.Sqrt(10.0 / 8.0);
            Math.Abs(std - expectedStd).Should().BeLessThan(0.05);
        }

        [Test]
        public void StandardT_HasExpectedStd_df5()
        {
            // df=5: Var = 5/3, Std = sqrt(5/3) ≈ 1.291
            np.random.seed(42);
            var samples = np.random.standard_t(5, 100000L);

            var std = (double)np.std(samples);
            var expectedStd = Math.Sqrt(5.0 / 3.0);
            Math.Abs(std - expectedStd).Should().BeLessThan(0.1);
        }

        [Test]
        public void StandardT_IsSymmetric()
        {
            // t distribution is symmetric around 0
            np.random.seed(42);
            var samples = np.random.standard_t(10, 100000L);

            int negCount = 0, posCount = 0;
            foreach (var val in samples.AsIterator<double>())
            {
                if (val < 0) negCount++;
                else if (val > 0) posCount++;
            }

            // Ratio should be close to 1
            var ratio = (double)negCount / posCount;
            Math.Abs(ratio - 1.0).Should().BeLessThan(0.05);
        }

        [Test]
        public void StandardT_LargeDf_ApproachesNormal()
        {
            // As df → ∞, t distribution approaches standard normal
            np.random.seed(42);
            var samples = np.random.standard_t(1000, 100000L);

            var mean = (double)np.mean(samples);
            var std = (double)np.std(samples);

            // Should be very close to N(0,1)
            Math.Abs(mean).Should().BeLessThan(0.02);
            Math.Abs(std - 1.0).Should().BeLessThan(0.02);
        }

        [Test]
        public void StandardT_SmallDf_HasHeavierTails()
        {
            // df=3 should have heavier tails than df=100
            np.random.seed(42);
            var samples3 = np.random.standard_t(3, 100000L);
            np.random.seed(42);
            var samples100 = np.random.standard_t(100, 100000L);

            var max3 = (double)np.amax(np.abs(samples3));
            var max100 = (double)np.amax(np.abs(samples100));

            // df=3 should have more extreme values
            max3.Should().BeGreaterThan(max100);
        }

        [Test]
        public void StandardT_DfZero_ThrowsArgumentException()
        {
            Action act = () => np.random.standard_t(0);
            act.Should().Throw<ArgumentException>();
        }

        [Test]
        public void StandardT_DfNegative_ThrowsArgumentException()
        {
            Action act = () => np.random.standard_t(-1);
            act.Should().Throw<ArgumentException>();
        }

        [Test]
        public void StandardT_ShapeOverload_Works()
        {
            np.random.seed(42);
            var result = np.random.standard_t(10, new Shape(3, 4));

            result.shape.Should().ContainInOrder(3, 4);
        }

        [Test]
        public void StandardT_Reproducibility_WithSeed()
        {
            np.random.seed(42);
            var first = np.random.standard_t(10, 5L).ToArray<double>();

            np.random.seed(42);
            var second = np.random.standard_t(10, 5L).ToArray<double>();

            first.Should().BeEquivalentTo(second);
        }

        [Test]
        public void StandardT_FractionalDf_Works()
        {
            // df can be fractional
            np.random.seed(42);
            var samples = np.random.standard_t(2.5, 1000L);

            samples.size.Should().Be(1000);
            // Mean should still be ~0
            var mean = (double)np.mean(samples);
            Math.Abs(mean).Should().BeLessThan(0.2);
        }
    }
}
