using System;
using System.Linq;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.RandomSampling
{
    
    public class TriangularTests
    {
        [Test]
        public void Triangular_ReturnsScalar_WhenNoSize()
        {
            var rng = np.random.RandomState(42);
            var result = rng.triangular(0, 0.5, 1);

            // NumPy returns a scalar (0-dimensional) when no size is given
            result.ndim.Should().Be(0);
            result.size.Should().Be(1);
            result.dtype.Should().Be(typeof(double));
        }

        [Test]
        public void Triangular_Returns1DArray()
        {
            var rng = np.random.RandomState(42);
            var result = rng.triangular(0, 0.5, 1, 5);

            result.shape.Should().ContainInOrder(5);
            result.size.Should().Be(5);
        }

        [Test]
        public void Triangular_Returns2DArray()
        {
            var rng = np.random.RandomState(42);
            var result = rng.triangular(0, 0.5, 1, new[] { 2, 3 });

            result.shape.Should().ContainInOrder(2, 3);
            result.size.Should().Be(6);
        }

        [Test]
        public void Triangular_AllValuesWithinBounds()
        {
            var rng = np.random.RandomState(42);
            var samples = rng.triangular(0, 0.5, 1, 10000);

            var min = (double)np.amin(samples);
            var max = (double)np.amax(samples);

            min.Should().BeGreaterThanOrEqualTo(0.0);
            max.Should().BeLessThanOrEqualTo(1.0);
        }

        [Test]
        public void Triangular_SymmetricMode_HasExpectedMean()
        {
            // For triangular(left, mode, right), mean = (left + mode + right) / 3
            var rng = np.random.RandomState(42);
            var samples = rng.triangular(0, 0.5, 1, 100000);

            var mean = (double)np.mean(samples);
            var expectedMean = (0.0 + 0.5 + 1.0) / 3.0; // 0.5

            Math.Abs(mean - expectedMean).Should().BeLessThan(0.01);
        }

        [Test]
        public void Triangular_LeftSkewed_HasExpectedMean()
        {
            // mode=0.1, mean should be (0 + 0.1 + 1) / 3 ≈ 0.3667
            var rng = np.random.RandomState(42);
            var samples = rng.triangular(0, 0.1, 1, 100000);

            var mean = (double)np.mean(samples);
            var expectedMean = (0.0 + 0.1 + 1.0) / 3.0;

            Math.Abs(mean - expectedMean).Should().BeLessThan(0.01);
        }

        [Test]
        public void Triangular_RightSkewed_HasExpectedMean()
        {
            // mode=0.9, mean should be (0 + 0.9 + 1) / 3 ≈ 0.6333
            var rng = np.random.RandomState(42);
            var samples = rng.triangular(0, 0.9, 1, 100000);

            var mean = (double)np.mean(samples);
            var expectedMean = (0.0 + 0.9 + 1.0) / 3.0;

            Math.Abs(mean - expectedMean).Should().BeLessThan(0.01);
        }

        [Test]
        public void Triangular_ModeAtLeft_StillValid()
        {
            var rng = np.random.RandomState(42);
            var samples = rng.triangular(0, 0, 1, 5);

            samples.size.Should().Be(5);
            // All values should be in [0, 1]
            foreach (var val in samples.AsIterator<double>())
            {
                val.Should().BeGreaterThanOrEqualTo(0.0);
                val.Should().BeLessThanOrEqualTo(1.0);
            }
        }

        [Test]
        public void Triangular_ModeAtRight_StillValid()
        {
            var rng = np.random.RandomState(42);
            var samples = rng.triangular(0, 1, 1, 5);

            samples.size.Should().Be(5);
            // All values should be in [0, 1]
            foreach (var val in samples.AsIterator<double>())
            {
                val.Should().BeGreaterThanOrEqualTo(0.0);
                val.Should().BeLessThanOrEqualTo(1.0);
            }
        }

        [Test]
        public void Triangular_NegativeRange_Works()
        {
            var rng = np.random.RandomState(42);
            var samples = rng.triangular(-10, -5, 0, 5);

            // All values should be in [-10, 0]
            foreach (var val in samples.AsIterator<double>())
            {
                val.Should().BeGreaterThanOrEqualTo(-10.0);
                val.Should().BeLessThanOrEqualTo(0.0);
            }
        }

        [Test]
        public void Triangular_LeftGreaterThanMode_ThrowsArgumentException()
        {
            Action act = () => np.random.triangular(1, 0.5, 2); // left=1, mode=0.5, right=2
            act.Should().Throw<ArgumentException>();
        }

        [Test]
        public void Triangular_ModeGreaterThanRight_ThrowsArgumentException()
        {
            Action act = () => np.random.triangular(0, 1.5, 1); // mode > right
            act.Should().Throw<ArgumentException>();
        }

        [Test]
        public void Triangular_LeftEqualsRight_ThrowsArgumentException()
        {
            Action act = () => np.random.triangular(5, 5, 5); // degenerate case not allowed
            act.Should().Throw<ArgumentException>();
        }

        [Test]
        public void Triangular_ShapeOverload_Works()
        {
            var rng = np.random.RandomState(42);
            var result = rng.triangular(0, 0.5, 1, new Shape(3, 4));

            result.shape.Should().ContainInOrder(3, 4);
        }

        [Test]
        public void Triangular_Reproducibility_WithSeed()
        {
            var rng1 = np.random.RandomState(42);
            var first = rng1.triangular(0, 0.5, 1, 5).ToArray<double>();

            var rng2 = np.random.RandomState(42);
            var second = rng2.triangular(0, 0.5, 1, 5).ToArray<double>();

            first.Should().BeEquivalentTo(second);
        }
    }
}
