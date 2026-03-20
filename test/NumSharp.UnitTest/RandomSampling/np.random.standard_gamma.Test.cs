using System;
using System.Linq;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.RandomSampling
{
    public class StandardGammaTests
    {
        [Test]
        public void StandardGamma_ReturnsScalarArray_WhenNoSize()
        {
            np.random.seed(42);
            var result = np.random.standard_gamma(2);

            result.shape.Should().ContainInOrder(1);
            result.dtype.Should().Be(typeof(double));
        }

        [Test]
        public void StandardGamma_Returns1DArray()
        {
            np.random.seed(42);
            var result = np.random.standard_gamma(2, 5);

            result.shape.Should().ContainInOrder(5);
            result.size.Should().Be(5);
        }

        [Test]
        public void StandardGamma_Returns2DArray()
        {
            np.random.seed(42);
            var result = np.random.standard_gamma(2, new[] { 2, 3 });

            result.shape.Should().ContainInOrder(2, 3);
            result.size.Should().Be(6);
        }

        [Test]
        public void StandardGamma_AllValuesPositive()
        {
            np.random.seed(42);
            var samples = np.random.standard_gamma(2, 10000);

            var min = (double)np.amin(samples);
            min.Should().BeGreaterThan(0.0);
        }

        [Test]
        public void StandardGamma_HasExpectedMean_Shape2()
        {
            // For standard gamma: mean = shape
            np.random.seed(42);
            var samples = np.random.standard_gamma(2, 100000);

            var mean = (double)np.mean(samples);
            Math.Abs(mean - 2.0).Should().BeLessThan(0.05);
        }

        [Test]
        public void StandardGamma_HasExpectedStd_Shape2()
        {
            // For standard gamma: variance = shape, std = sqrt(shape)
            np.random.seed(42);
            var samples = np.random.standard_gamma(2, 100000);

            var std = (double)np.std(samples);
            var expectedStd = Math.Sqrt(2.0);
            Math.Abs(std - expectedStd).Should().BeLessThan(0.05);
        }

        [Test]
        public void StandardGamma_HasExpectedMean_Shape5()
        {
            np.random.seed(42);
            var samples = np.random.standard_gamma(5, 100000);

            var mean = (double)np.mean(samples);
            Math.Abs(mean - 5.0).Should().BeLessThan(0.1);
        }

        [Test]
        public void StandardGamma_HasExpectedStd_Shape5()
        {
            np.random.seed(42);
            var samples = np.random.standard_gamma(5, 100000);

            var std = (double)np.std(samples);
            var expectedStd = Math.Sqrt(5.0);
            Math.Abs(std - expectedStd).Should().BeLessThan(0.1);
        }

        [Test]
        public void StandardGamma_ShapeLessThanOne_Works()
        {
            // Shape < 1 uses different algorithm branch
            np.random.seed(42);
            var samples = np.random.standard_gamma(0.5, 10000);

            // All should be positive
            foreach (var val in samples.AsIterator<double>())
            {
                val.Should().BeGreaterThan(0.0);
            }

            // Mean should be close to 0.5
            var mean = (double)np.mean(samples);
            Math.Abs(mean - 0.5).Should().BeLessThan(0.05);
        }

        [Test]
        public void StandardGamma_ShapeZero_ReturnsZeros()
        {
            // Special case: shape=0 returns all zeros (NumPy behavior)
            np.random.seed(42);
            var samples = np.random.standard_gamma(0, 5);

            foreach (var val in samples.AsIterator<double>())
            {
                val.Should().Be(0.0);
            }
        }

        [Test]
        public void StandardGamma_ShapeNegative_ThrowsArgumentException()
        {
            Action act = () => np.random.standard_gamma(-1);
            act.Should().Throw<ArgumentException>();
        }

        [Test]
        public void StandardGamma_ShapeOverload_Works()
        {
            np.random.seed(42);
            var result = np.random.standard_gamma(2, new Shape(3, 4));

            result.shape.Should().ContainInOrder(3, 4);
        }

        [Test]
        public void StandardGamma_Reproducibility_WithSeed()
        {
            np.random.seed(42);
            var first = np.random.standard_gamma(2, 5).ToArray<double>();

            np.random.seed(42);
            var second = np.random.standard_gamma(2, 5).ToArray<double>();

            first.Should().BeEquivalentTo(second);
        }

        [Test]
        public void StandardGamma_FractionalShape_Works()
        {
            np.random.seed(42);
            var samples = np.random.standard_gamma(2.5, 1000);

            samples.size.Should().Be(1000);
            // Mean should be close to 2.5
            var mean = (double)np.mean(samples);
            Math.Abs(mean - 2.5).Should().BeLessThan(0.2);
        }

        [Test]
        public void StandardGamma_VerySmallShape_AllPositive()
        {
            np.random.seed(42);
            var samples = np.random.standard_gamma(0.1, 1000);

            foreach (var val in samples.AsIterator<double>())
            {
                val.Should().BeGreaterThanOrEqualTo(0.0);
            }
        }
    }
}
