using System;
using System.Linq;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.RandomSampling
{
    
    [TestClass]
    public class WaldTests
    {
        [TestMethod]
        public void Wald_ReturnsScalar_WhenNoSize()
        {
            var rng = np.random.RandomState(42);
            var result = rng.wald(1, 1);

            // NumPy returns a scalar (0-dimensional) when no size is given
            result.ndim.Should().Be(0);
            result.size.Should().Be(1);
            result.dtype.Should().Be(typeof(double));
        }

        [TestMethod]
        public void Wald_Returns1DArray()
        {
            var rng = np.random.RandomState(42);
            var result = rng.wald(1, 1, 5);

            result.shape.Should().ContainInOrder(5);
            result.size.Should().Be(5);
        }

        [TestMethod]
        public void Wald_Returns2DArray()
        {
            var rng = np.random.RandomState(42);
            var result = rng.wald(1, 1, new[] { 2, 3 });

            result.shape.Should().ContainInOrder(2, 3);
            result.size.Should().Be(6);
        }

        [TestMethod]
        public void Wald_AllValuesPositive()
        {
            var rng = np.random.RandomState(42);
            var samples = rng.wald(1, 1, 10000);

            var min = (double)np.amin(samples);
            min.Should().BeGreaterThan(0.0);
        }

        [TestMethod]
        public void Wald_HasExpectedMean()
        {
            // For Wald: E[X] = mean
            var rng = np.random.RandomState(42);
            var samples = rng.wald(1, 1, 100000);

            var mean = (double)np.mean(samples);
            // Mean should be close to 1
            Math.Abs(mean - 1.0).Should().BeLessThan(0.05);
        }

        [TestMethod]
        public void Wald_HasExpectedVariance()
        {
            // For Wald: Var[X] = mean^3 / scale
            // With mean=1, scale=1: Var = 1
            var rng = np.random.RandomState(42);
            var samples = rng.wald(1, 1, 100000);

            var variance = (double)np.var(samples);
            // Variance should be close to 1
            Math.Abs(variance - 1.0).Should().BeLessThan(0.1);
        }

        [TestMethod]
        public void Wald_DifferentMean_HasExpectedMean()
        {
            // With mean=3, scale=2
            var rng = np.random.RandomState(42);
            var samples = rng.wald(3, 2, 100000);

            var mean = (double)np.mean(samples);
            // Mean should be close to 3
            Math.Abs(mean - 3.0).Should().BeLessThan(0.1);
        }

        [TestMethod]
        public void Wald_DifferentParams_HasExpectedVariance()
        {
            // For Wald: Var[X] = mean^3 / scale
            // With mean=3, scale=2: Var = 27/2 = 13.5
            var rng = np.random.RandomState(42);
            var samples = rng.wald(3, 2, 100000);

            var variance = (double)np.var(samples);
            // Variance should be close to 13.5
            Math.Abs(variance - 13.5).Should().BeLessThan(1.0);
        }

        [TestMethod]
        public void Wald_SmallParameters_AllPositive()
        {
            var rng = np.random.RandomState(42);
            var samples = rng.wald(0.1, 0.1, 10000);

            foreach (var val in samples.AsIterator<double>())
            {
                val.Should().BeGreaterThan(0.0);
            }
        }

        [TestMethod]
        public void Wald_MeanZero_ThrowsArgumentException()
        {
            Action act = () => np.random.wald(0, 1);
            act.Should().Throw<ArgumentException>();
        }

        [TestMethod]
        public void Wald_MeanNegative_ThrowsArgumentException()
        {
            Action act = () => np.random.wald(-1, 1);
            act.Should().Throw<ArgumentException>();
        }

        [TestMethod]
        public void Wald_ScaleZero_ThrowsArgumentException()
        {
            Action act = () => np.random.wald(1, 0);
            act.Should().Throw<ArgumentException>();
        }

        [TestMethod]
        public void Wald_ScaleNegative_ThrowsArgumentException()
        {
            Action act = () => np.random.wald(1, -1);
            act.Should().Throw<ArgumentException>();
        }

        [TestMethod]
        public void Wald_ShapeOverload_Works()
        {
            var rng = np.random.RandomState(42);
            var result = rng.wald(1, 1, new Shape(3, 4));

            result.shape.Should().ContainInOrder(3, 4);
        }

        [TestMethod]
        public void Wald_Reproducibility_WithSeed()
        {
            var rng1 = np.random.RandomState(42);
            var first = rng1.wald(1, 1, 5).ToArray<double>();

            var rng2 = np.random.RandomState(42);
            var second = rng2.wald(1, 1, 5).ToArray<double>();

            first.Should().BeEquivalentTo(second);
        }
    }
}
