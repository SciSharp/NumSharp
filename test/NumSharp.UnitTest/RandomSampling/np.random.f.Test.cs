using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace NumSharp.UnitTest.RandomSampling
{
    public class NpRandomFTest : TestClass
    {
        [Test]
        public void F_ScalarReturn()
        {
            var rng = np.random.RandomState(42);
            var result = rng.f(5, 10);

            Assert.AreEqual(0, result.ndim);
            Assert.AreEqual(1, result.size);
            Assert.IsTrue((double)result > 0);
        }

        [Test]
        public void F_1DArray()
        {
            var result = np.random.f(5, 10, 5);

            Assert.AreEqual(1, result.ndim);
            Assert.AreEqual(5, result.shape[0]);
            Assert.AreEqual(typeof(double), result.dtype);
        }

        [Test]
        public void F_2DArray()
        {
            var result = np.random.f(5, 10, new Shape(2, 3));

            Assert.AreEqual(2, result.ndim);
            Assert.AreEqual(2, result.shape[0]);
            Assert.AreEqual(3, result.shape[1]);
        }

        [Test]
        public void F_ShapeOverload()
        {
            var result = np.random.f(5, 10, new Shape(10, 5));

            Assert.AreEqual(10, result.shape[0]);
            Assert.AreEqual(5, result.shape[1]);
        }

        [Test]
        public void F_DfnumZero_ThrowsArgumentException()
        {
            Assert.ThrowsException<ArgumentException>(() => np.random.f(0, 10));
        }

        [Test]
        public void F_DfdenZero_ThrowsArgumentException()
        {
            Assert.ThrowsException<ArgumentException>(() => np.random.f(5, 0));
        }

        [Test]
        public void F_DfnumNegative_ThrowsArgumentException()
        {
            Assert.ThrowsException<ArgumentException>(() => np.random.f(-1, 10));
        }

        [Test]
        public void F_DfdenNegative_ThrowsArgumentException()
        {
            Assert.ThrowsException<ArgumentException>(() => np.random.f(5, -1));
        }

        [Test]
        public void F_MeanMatchesTheory()
        {
            // For F(dfnum, dfden), mean = dfden / (dfden - 2) when dfden > 2
            var rng = np.random.RandomState(42);
            var samples = rng.f(5, 10, 100000);
            double expectedMean = 10.0 / (10.0 - 2.0);  // 1.25
            double actualMean = (double)np.mean(samples);

            Assert.IsTrue(Math.Abs(actualMean - expectedMean) < 0.05,
                $"Mean {actualMean} should be close to {expectedMean}");
        }

        [Test]
        public void F_AllValuesPositive()
        {
            var rng = np.random.RandomState(42);
            var samples = rng.f(5, 10, 10000);

            foreach (var val in samples.AsIterator<double>())
                Assert.IsTrue(val > 0, $"Value {val} should be positive");
        }

        [Test]
        public void F_DifferentDf_MeanMatchesTheory()
        {
            // Test with different df values
            var rng = np.random.RandomState(42);
            var samples = rng.f(10, 20, 100000);
            double expectedMean = 20.0 / (20.0 - 2.0);  // 1.111
            double actualMean = (double)np.mean(samples);

            Assert.IsTrue(Math.Abs(actualMean - expectedMean) < 0.05,
                $"Mean {actualMean} should be close to {expectedMean}");
        }

        [Test]
        public void F_ReturnsFloat64()
        {
            var result = np.random.f(5, 10, size: new Shape(5));

            Assert.AreEqual(typeof(double), result.dtype);
        }

        [Test]
        public void F_Reproducible()
        {
            var rng1 = np.random.RandomState(123);
            var a = rng1.f(5, 10, 5);

            var rng2 = np.random.RandomState(123);
            var b = rng2.f(5, 10, 5);

            var aData = a.Data<double>();
            var bData = b.Data<double>();
            for (int i = 0; i < 5; i++)
                Assert.AreEqual(aData[i], bData[i]);
        }
    }
}
