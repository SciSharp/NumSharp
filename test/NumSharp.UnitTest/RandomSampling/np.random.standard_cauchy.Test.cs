using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace NumSharp.UnitTest.RandomSampling
{
    
    public class NpRandomStandardCauchyTest : TestClass
    {
        [TestMethod]
        public void StandardCauchy_ScalarReturn()
        {
            var rng = np.random.RandomState(42);
            var result = rng.standard_cauchy();

            Assert.AreEqual(0, result.ndim);
            Assert.AreEqual(1, result.size);
        }

        [TestMethod]
        public void StandardCauchy_1DArray()
        {
            var result = np.random.standard_cauchy(5L);

            Assert.AreEqual(1, result.ndim);
            Assert.AreEqual(5, result.shape[0]);
            Assert.AreEqual(typeof(double), result.dtype);
        }

        [TestMethod]
        public void StandardCauchy_2DArray()
        {
            var result = np.random.standard_cauchy(new Shape(2, 3));

            Assert.AreEqual(2, result.ndim);
            Assert.AreEqual(2, result.shape[0]);
            Assert.AreEqual(3, result.shape[1]);
        }

        [TestMethod]
        public void StandardCauchy_ShapeOverload()
        {
            var result = np.random.standard_cauchy(new Shape(10, 5));

            Assert.AreEqual(10, result.shape[0]);
            Assert.AreEqual(5, result.shape[1]);
        }

        [TestMethod]
        public void StandardCauchy_MedianNearZero()
        {
            // Cauchy has no mean/variance, but median = 0
            var rng = np.random.RandomState(42);
            var samples = rng.standard_cauchy(100000L);

            // Calculate median manually
            var sorted = samples.Data<double>().OrderBy(x => x).ToArray();
            double median = sorted[sorted.Length / 2];

            Assert.IsTrue(Math.Abs(median) < 0.05, $"Median {median} should be close to 0");
        }

        [TestMethod]
        public void StandardCauchy_InterquartileRange()
        {
            // For standard Cauchy, Q1 = -1, Q3 = 1, so IQR = 2
            var rng = np.random.RandomState(42);
            var samples = rng.standard_cauchy(100000L);

            var sorted = samples.Data<double>().OrderBy(x => x).ToArray();
            double q1 = sorted[sorted.Length / 4];
            double q3 = sorted[3 * sorted.Length / 4];
            double iqr = q3 - q1;

            // IQR should be approximately 2
            Assert.IsTrue(Math.Abs(iqr - 2.0) < 0.1, $"IQR {iqr} should be close to 2");
            // Q1 should be approximately -1
            Assert.IsTrue(Math.Abs(q1 + 1.0) < 0.05, $"Q1 {q1} should be close to -1");
            // Q3 should be approximately 1
            Assert.IsTrue(Math.Abs(q3 - 1.0) < 0.05, $"Q3 {q3} should be close to 1");
        }

        [TestMethod]
        public void StandardCauchy_HasHeavyTails()
        {
            // Cauchy has heavy tails - should have some extreme values
            var rng = np.random.RandomState(42);
            var samples = rng.standard_cauchy(10000L);

            var max = (double)np.amax(samples);
            var min = (double)np.amin(samples);

            // Should have values far from 0 due to heavy tails
            Assert.IsTrue(max > 10 || min < -10,
                $"Should have extreme values (max={max}, min={min})");
        }

        [TestMethod]
        public void StandardCauchy_ReturnsFloat64()
        {
            var result = np.random.standard_cauchy(size: new Shape(5));

            Assert.AreEqual(typeof(double), result.dtype);
        }

        [TestMethod]
        public void StandardCauchy_EmptySize()
        {
            var result = np.random.standard_cauchy(0L);

            Assert.AreEqual(1, result.ndim);
            Assert.AreEqual(0, result.size);
        }

        [TestMethod]
        public void StandardCauchy_Reproducible()
        {
            var rng1 = np.random.RandomState(123);
            var a = rng1.standard_cauchy(5L);

            var rng2 = np.random.RandomState(123);
            var b = rng2.standard_cauchy(5L);

            var aData = a.Data<double>();
            var bData = b.Data<double>();
            for (int i = 0; i < 5; i++)
                Assert.AreEqual(aData[i], bData[i]);
        }
    }
}
