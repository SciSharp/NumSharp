using System;
using System.Linq;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.RandomSampling
{
    public class MultinomialTests
    {
        private static readonly double[] DicePvals = { 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6 };

        [TestMethod]
        public void Multinomial_ReturnsSingleSample_WhenNoSize()
        {
            var rng = np.random.RandomState(42);
            var result = rng.multinomial(20, DicePvals);

            result.shape.Should().ContainInOrder(6);
            result.dtype.Should().Be(typeof(int));
        }

        [TestMethod]
        public void Multinomial_SumEqualsN()
        {
            var rng = np.random.RandomState(42);
            var result = rng.multinomial(20, DicePvals);

            var sum = (int)np.sum(result);
            sum.Should().Be(20);
        }

        [TestMethod]
        public void Multinomial_Returns2DArray_WithSize()
        {
            var rng = np.random.RandomState(42);
            var result = rng.multinomial(20, DicePvals, 5);

            result.shape.Should().ContainInOrder(5, 6);
        }

        [TestMethod]
        public void Multinomial_Returns3DArray_With2DSize()
        {
            var rng = np.random.RandomState(42);
            var result = rng.multinomial(20, DicePvals, new[] { 2, 3 });

            result.shape.Should().ContainInOrder(2, 3, 6);
        }

        [TestMethod]
        public void Multinomial_AllRowsSumToN()
        {
            var rng = np.random.RandomState(42);
            var result = rng.multinomial(100, DicePvals, 10);

            for (int i = 0; i < 10; i++)
            {
                int rowSum = 0;
                for (int j = 0; j < 6; j++)
                {
                    rowSum += (int)result[i, j];
                }
                rowSum.Should().Be(100);
            }
        }

        [TestMethod]
        public void Multinomial_AllValuesNonNegative()
        {
            var rng = np.random.RandomState(42);
            var result = rng.multinomial(100, DicePvals, 100);

            foreach (var val in result.AsIterator<int>())
            {
                val.Should().BeGreaterThanOrEqualTo(0);
            }
        }

        [TestMethod]
        public void Multinomial_BiasedPvals_ProducesMoreInHighProbCategory()
        {
            var rng = np.random.RandomState(42);
            var biasedPvals = new double[] { 0.1, 0.1, 0.1, 0.1, 0.1, 0.5 };
            var result = rng.multinomial(1000, biasedPvals, 100);

            // Sum counts for last category (should be highest)
            int lastCategoryTotal = 0;
            int firstCategoryTotal = 0;
            for (int i = 0; i < 100; i++)
            {
                lastCategoryTotal += (int)result[i, 5];
                firstCategoryTotal += (int)result[i, 0];
            }

            // Last category should have more than first
            lastCategoryTotal.Should().BeGreaterThan(firstCategoryTotal);
        }

        [TestMethod]
        public void Multinomial_NZero_ReturnsAllZeros()
        {
            var rng = np.random.RandomState(42);
            var result = rng.multinomial(0, DicePvals);

            foreach (var val in result.AsIterator<int>())
            {
                val.Should().Be(0);
            }
        }

        [TestMethod]
        public void Multinomial_NNegative_ThrowsArgumentException()
        {
            Action act = () => np.random.multinomial(-1, DicePvals);
            act.Should().Throw<ArgumentException>();
        }

        [TestMethod]
        public void Multinomial_PvalsNegative_ThrowsArgumentException()
        {
            Action act = () => np.random.multinomial(20, new[] { 0.5, -0.1, 0.6 });
            act.Should().Throw<ArgumentException>();
        }

        [TestMethod]
        public void Multinomial_PvalsSumGreaterThanOne_ThrowsArgumentException()
        {
            Action act = () => np.random.multinomial(20, new[] { 0.5, 0.6, 0.1 });
            act.Should().Throw<ArgumentException>();
        }

        [TestMethod]
        public void Multinomial_ShapeOverload_Works()
        {
            var rng = np.random.RandomState(42);
            var result = rng.multinomial(20, DicePvals, new Shape(3, 4));

            result.shape.Should().ContainInOrder(3, 4, 6);
        }

        [TestMethod]
        public void Multinomial_Reproducibility_WithSeed()
        {
            var rng1 = np.random.RandomState(42);
            var first = rng1.multinomial(20, DicePvals, 5).ToArray<int>();

            var rng2 = np.random.RandomState(42);
            var second = rng2.multinomial(20, DicePvals, 5).ToArray<int>();

            first.Should().BeEquivalentTo(second);
        }

        [TestMethod]
        public void Multinomial_TwoCategories_IsBinomialLike()
        {
            var rng = np.random.RandomState(42);
            var pvals = new double[] { 0.3, 0.7 };
            var result = rng.multinomial(100, pvals, 1000);

            // First category should have ~30% on average
            double avgFirst = 0;
            for (int i = 0; i < 1000; i++)
            {
                avgFirst += (int)result[i, 0];
            }
            avgFirst /= 1000;

            // Should be close to 30
            Math.Abs(avgFirst - 30).Should().BeLessThan(3);
        }

        [TestMethod]
        public void Multinomial_LargeN_Works()
        {
            var rng = np.random.RandomState(42);
            var result = rng.multinomial(10000, DicePvals);

            var sum = (int)np.sum(result);
            sum.Should().Be(10000);
        }
    }
}
