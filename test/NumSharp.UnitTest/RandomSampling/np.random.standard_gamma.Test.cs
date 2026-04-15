using System;
using System.Linq;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.RandomSampling
{
    
    public class StandardGammaTests
    {
        [TestMethod]
        public void StandardGamma_ReturnsScalar_WhenNoSize()
        {
            var rng = np.random.RandomState(42);
            var result = rng.standard_gamma(2);

            // NumPy returns a scalar (0-dimensional) when no size is given
            result.ndim.Should().Be(0);
            result.size.Should().Be(1);
            result.dtype.Should().Be(typeof(double));
        }

        [TestMethod]
        public void StandardGamma_Returns1DArray()
        {
            var rng = np.random.RandomState(42);
            var result = rng.standard_gamma(2, 5L);

            result.shape.Should().ContainInOrder(5);
            result.size.Should().Be(5);
        }

        [TestMethod]
        public void StandardGamma_Returns2DArray()
        {
            var rng = np.random.RandomState(42);
            var result = rng.standard_gamma(2, new[] { 2, 3 });

            result.shape.Should().ContainInOrder(2, 3);
            result.size.Should().Be(6);
        }

        [TestMethod]
        public void StandardGamma_AllValuesPositive()
        {
            var rng = np.random.RandomState(42);
            var samples = rng.standard_gamma(2, 10000L);

            var min = (double)np.amin(samples);
            min.Should().BeGreaterThan(0.0);
        }

        [TestMethod]
        public void StandardGamma_HasExpectedMean_Shape2()
        {
            // For standard gamma: mean = shape
            var rng = np.random.RandomState(42);
            var samples = rng.standard_gamma(2, 100000L);

            var mean = (double)np.mean(samples);
            Math.Abs(mean - 2.0).Should().BeLessThan(0.05);
        }

        [TestMethod]
        public void StandardGamma_HasExpectedStd_Shape2()
        {
            // For standard gamma: variance = shape, std = sqrt(shape)
            var rng = np.random.RandomState(42);
            var samples = rng.standard_gamma(2, 100000L);

            var std = (double)np.std(samples);
            var expectedStd = Math.Sqrt(2.0);
            Math.Abs(std - expectedStd).Should().BeLessThan(0.05);
        }

        [TestMethod]
        public void StandardGamma_HasExpectedMean_Shape5()
        {
            var rng = np.random.RandomState(42);
            var samples = rng.standard_gamma(5, 100000L);

            var mean = (double)np.mean(samples);
            Math.Abs(mean - 5.0).Should().BeLessThan(0.1);
        }

        [TestMethod]
        public void StandardGamma_HasExpectedStd_Shape5()
        {
            var rng = np.random.RandomState(42);
            var samples = rng.standard_gamma(5, 100000L);

            var std = (double)np.std(samples);
            var expectedStd = Math.Sqrt(5.0);
            Math.Abs(std - expectedStd).Should().BeLessThan(0.1);
        }

        [TestMethod]
        public void StandardGamma_ShapeLessThanOne_Works()
        {
            // Shape < 1 uses different algorithm branch
            var rng = np.random.RandomState(42);
            var samples = rng.standard_gamma(0.5, 10000L);

            // All should be positive
            foreach (var val in samples.AsIterator<double>())
            {
                val.Should().BeGreaterThan(0.0);
            }

            // Mean should be close to 0.5
            var mean = (double)np.mean(samples);
            Math.Abs(mean - 0.5).Should().BeLessThan(0.05);
        }

        [TestMethod]
        public void StandardGamma_ShapeZero_ReturnsZeros()
        {
            // Special case: shape=0 returns all zeros (NumPy behavior)
            var rng = np.random.RandomState(42);
            var samples = rng.standard_gamma(0, 5L);

            foreach (var val in samples.AsIterator<double>())
            {
                val.Should().Be(0.0);
            }
        }

        [TestMethod]
        public void StandardGamma_ShapeNegative_ThrowsArgumentException()
        {
            Action act = () => np.random.standard_gamma(-1);
            act.Should().Throw<ArgumentException>();
        }

        [TestMethod]
        public void StandardGamma_ShapeOverload_Works()
        {
            var rng = np.random.RandomState(42);
            var result = rng.standard_gamma(2, new Shape(3, 4));

            result.shape.Should().ContainInOrder(3, 4);
        }

        [TestMethod]
        public void StandardGamma_Reproducibility_WithSeed()
        {
            var rng1 = np.random.RandomState(42);
            var first = rng1.standard_gamma(2, 5L).ToArray<double>();

            var rng2 = np.random.RandomState(42);
            var second = rng2.standard_gamma(2, 5L).ToArray<double>();

            first.Should().BeEquivalentTo(second);
        }

        [TestMethod]
        public void StandardGamma_FractionalShape_Works()
        {
            var rng = np.random.RandomState(42);
            var samples = rng.standard_gamma(2.5, 1000L);

            samples.size.Should().Be(1000);
            // Mean should be close to 2.5
            var mean = (double)np.mean(samples);
            Math.Abs(mean - 2.5).Should().BeLessThan(0.2);
        }

        [TestMethod]
        public void StandardGamma_VerySmallShape_AllPositive()
        {
            var rng = np.random.RandomState(42);
            var samples = rng.standard_gamma(0.1, 1000L);

            foreach (var val in samples.AsIterator<double>())
            {
                val.Should().BeGreaterThanOrEqualTo(0.0);
            }
        }
    }
}
