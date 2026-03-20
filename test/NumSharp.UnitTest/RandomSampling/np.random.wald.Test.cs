using System;
using System.Linq;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.RandomSampling
{
    public class WaldTests
    {
        [Test]
        public void Wald_ReturnsScalarArray_WhenNoSize()
        {
            np.random.seed(42);
            var result = np.random.wald(1, 1);

            result.shape.Should().ContainInOrder(1);
            result.dtype.Should().Be(typeof(double));
        }

        [Test]
        public void Wald_Returns1DArray()
        {
            np.random.seed(42);
            var result = np.random.wald(1, 1, 5);

            result.shape.Should().ContainInOrder(5);
            result.size.Should().Be(5);
        }

        [Test]
        public void Wald_Returns2DArray()
        {
            np.random.seed(42);
            var result = np.random.wald(1, 1, new[] { 2, 3 });

            result.shape.Should().ContainInOrder(2, 3);
            result.size.Should().Be(6);
        }

        [Test]
        public void Wald_AllValuesPositive()
        {
            np.random.seed(42);
            var samples = np.random.wald(1, 1, 10000);

            var min = (double)np.amin(samples);
            min.Should().BeGreaterThan(0.0);
        }

        [Test]
        public void Wald_HasExpectedMean()
        {
            // For Wald: E[X] = mean
            np.random.seed(42);
            var samples = np.random.wald(1, 1, 100000);

            var mean = (double)np.mean(samples);
            // Mean should be close to 1
            Math.Abs(mean - 1.0).Should().BeLessThan(0.05);
        }

        [Test]
        public void Wald_HasExpectedVariance()
        {
            // For Wald: Var[X] = mean^3 / scale
            // With mean=1, scale=1: Var = 1
            np.random.seed(42);
            var samples = np.random.wald(1, 1, 100000);

            var variance = (double)np.var(samples);
            // Variance should be close to 1
            Math.Abs(variance - 1.0).Should().BeLessThan(0.1);
        }

        [Test]
        public void Wald_DifferentMean_HasExpectedMean()
        {
            // With mean=3, scale=2
            np.random.seed(42);
            var samples = np.random.wald(3, 2, 100000);

            var mean = (double)np.mean(samples);
            // Mean should be close to 3
            Math.Abs(mean - 3.0).Should().BeLessThan(0.1);
        }

        [Test]
        public void Wald_DifferentParams_HasExpectedVariance()
        {
            // For Wald: Var[X] = mean^3 / scale
            // With mean=3, scale=2: Var = 27/2 = 13.5
            np.random.seed(42);
            var samples = np.random.wald(3, 2, 100000);

            var variance = (double)np.var(samples);
            // Variance should be close to 13.5
            Math.Abs(variance - 13.5).Should().BeLessThan(1.0);
        }

        [Test]
        public void Wald_SmallParameters_AllPositive()
        {
            np.random.seed(42);
            var samples = np.random.wald(0.1, 0.1, 10000);

            foreach (var val in samples.AsIterator<double>())
            {
                val.Should().BeGreaterThan(0.0);
            }
        }

        [Test]
        public void Wald_MeanZero_ThrowsArgumentException()
        {
            Action act = () => np.random.wald(0, 1);
            act.Should().Throw<ArgumentException>();
        }

        [Test]
        public void Wald_MeanNegative_ThrowsArgumentException()
        {
            Action act = () => np.random.wald(-1, 1);
            act.Should().Throw<ArgumentException>();
        }

        [Test]
        public void Wald_ScaleZero_ThrowsArgumentException()
        {
            Action act = () => np.random.wald(1, 0);
            act.Should().Throw<ArgumentException>();
        }

        [Test]
        public void Wald_ScaleNegative_ThrowsArgumentException()
        {
            Action act = () => np.random.wald(1, -1);
            act.Should().Throw<ArgumentException>();
        }

        [Test]
        public void Wald_ShapeOverload_Works()
        {
            np.random.seed(42);
            var result = np.random.wald(1, 1, new Shape(3, 4));

            result.shape.Should().ContainInOrder(3, 4);
        }

        [Test]
        public void Wald_Reproducibility_WithSeed()
        {
            np.random.seed(42);
            var first = np.random.wald(1, 1, 5).ToArray<double>();

            np.random.seed(42);
            var second = np.random.wald(1, 1, 5).ToArray<double>();

            first.Should().BeEquivalentTo(second);
        }
    }
}
