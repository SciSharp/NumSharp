using System;
using System.Linq;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.RandomSampling
{
    public class MultinomialTests
    {
        private static readonly double[] DicePvals = { 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6 };

        [Test]
        public void Multinomial_ReturnsSingleSample_WhenNoSize()
        {
            np.random.seed(42);
            var result = np.random.multinomial(20, DicePvals);

            result.shape.Should().ContainInOrder(6);
            result.dtype.Should().Be(typeof(int));
        }

        [Test]
        public void Multinomial_SumEqualsN()
        {
            np.random.seed(42);
            var result = np.random.multinomial(20, DicePvals);

            var sum = (int)np.sum(result);
            sum.Should().Be(20);
        }

        [Test]
        public void Multinomial_Returns2DArray_WithSize()
        {
            np.random.seed(42);
            var result = np.random.multinomial(20, DicePvals, 5);

            result.shape.Should().ContainInOrder(5, 6);
        }

        [Test]
        public void Multinomial_Returns3DArray_With2DSize()
        {
            np.random.seed(42);
            var result = np.random.multinomial(20, DicePvals, new[] { 2, 3 });

            result.shape.Should().ContainInOrder(2, 3, 6);
        }

        [Test]
        public void Multinomial_AllRowsSumToN()
        {
            np.random.seed(42);
            var result = np.random.multinomial(100, DicePvals, 10);

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

        [Test]
        public void Multinomial_AllValuesNonNegative()
        {
            np.random.seed(42);
            var result = np.random.multinomial(100, DicePvals, 100);

            foreach (var val in result.AsIterator<int>())
            {
                val.Should().BeGreaterThanOrEqualTo(0);
            }
        }

        [Test]
        public void Multinomial_BiasedPvals_ProducesMoreInHighProbCategory()
        {
            np.random.seed(42);
            var biasedPvals = new double[] { 0.1, 0.1, 0.1, 0.1, 0.1, 0.5 };
            var result = np.random.multinomial(1000, biasedPvals, 100);

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

        [Test]
        public void Multinomial_NZero_ReturnsAllZeros()
        {
            np.random.seed(42);
            var result = np.random.multinomial(0, DicePvals);

            foreach (var val in result.AsIterator<int>())
            {
                val.Should().Be(0);
            }
        }

        [Test]
        public void Multinomial_NNegative_ThrowsArgumentException()
        {
            Action act = () => np.random.multinomial(-1, DicePvals);
            act.Should().Throw<ArgumentException>();
        }

        [Test]
        public void Multinomial_PvalsNegative_ThrowsArgumentException()
        {
            Action act = () => np.random.multinomial(20, new[] { 0.5, -0.1, 0.6 });
            act.Should().Throw<ArgumentException>();
        }

        [Test]
        public void Multinomial_PvalsSumGreaterThanOne_ThrowsArgumentException()
        {
            Action act = () => np.random.multinomial(20, new[] { 0.5, 0.6, 0.1 });
            act.Should().Throw<ArgumentException>();
        }

        [Test]
        public void Multinomial_ShapeOverload_Works()
        {
            np.random.seed(42);
            var result = np.random.multinomial(20, DicePvals, new Shape(3, 4));

            result.shape.Should().ContainInOrder(3, 4, 6);
        }

        [Test]
        public void Multinomial_Reproducibility_WithSeed()
        {
            np.random.seed(42);
            var first = np.random.multinomial(20, DicePvals, 5).ToArray<int>();

            np.random.seed(42);
            var second = np.random.multinomial(20, DicePvals, 5).ToArray<int>();

            first.Should().BeEquivalentTo(second);
        }

        [Test]
        public void Multinomial_TwoCategories_IsBinomialLike()
        {
            np.random.seed(42);
            var pvals = new double[] { 0.3, 0.7 };
            var result = np.random.multinomial(100, pvals, 1000);

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

        [Test]
        public void Multinomial_LargeN_Works()
        {
            np.random.seed(42);
            var result = np.random.multinomial(10000, DicePvals);

            var sum = (int)np.sum(result);
            sum.Should().Be(10000);
        }
    }
}
