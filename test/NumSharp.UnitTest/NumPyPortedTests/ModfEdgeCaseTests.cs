using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;
using NumSharp.UnitTest.Utilities;
using TUnit.Core;

namespace NumSharp.UnitTest.NumPyPortedTests
{
    /// <summary>
    /// Tests for np.modf operation covering edge cases.
    /// modf returns (fractional_part, integer_part).
    /// </summary>
    public class ModfEdgeCaseTests
    {
        #region Basic Tests

        [Test]
        public void Modf_PositiveValues()
        {
            // NumPy: modf([1.5, 2.7]) = ([0.5, 0.7], [1., 2.])
            var x = np.array(new double[] { 1.5, 2.7 });
            var (frac, intg) = np.modf(x);

            var fracData = frac.GetData<double>();
            var intgData = intg.GetData<double>();

            Assert.IsTrue(Math.Abs(fracData[0] - 0.5) <1e-10);
            Assert.IsTrue(Math.Abs(fracData[1] - 0.7) <1e-10);
            Assert.AreEqual(1.0, intgData[0]);
            Assert.AreEqual(2.0, intgData[1]);
        }

        [Test]
        public void Modf_NegativeValues()
        {
            // NumPy: modf(-3.2) = (-0.2, -3.)
            var x = np.array(new double[] { -3.2 });
            var (frac, intg) = np.modf(x);

            var fracData = frac.GetData<double>();
            var intgData = intg.GetData<double>();

            Assert.IsTrue(Math.Abs(fracData[0] - (-0.2)) < 1e-10);
            Assert.AreEqual(-3.0, intgData[0]);
        }

        [Test]
        public void Modf_MixedValues()
        {
            // NumPy: modf([1.5, 2.7, -3.2, 0.0])
            var x = np.array(new double[] { 1.5, 2.7, -3.2, 0.0 });
            var (frac, intg) = np.modf(x);

            var fracData = frac.GetData<double>();
            var intgData = intg.GetData<double>();

            Assert.IsTrue(Math.Abs(fracData[0] - 0.5) < 1e-10);
            Assert.IsTrue(Math.Abs(fracData[1] - 0.7) < 1e-10);
            Assert.IsTrue(Math.Abs(fracData[2] - (-0.2)) < 1e-10);
            Assert.AreEqual(0.0, fracData[3]);

            Assert.AreEqual(1.0, intgData[0]);
            Assert.AreEqual(2.0, intgData[1]);
            Assert.AreEqual(-3.0, intgData[2]);
            Assert.AreEqual(0.0, intgData[3]);
        }

        #endregion

        #region Zero Tests

        [Test]
        public void Modf_PositiveZero()
        {
            var x = np.array(new double[] { 0.0 });
            var (frac, intg) = np.modf(x);

            Assert.AreEqual(0.0, frac.GetData<double>()[0]);
            Assert.AreEqual(0.0, intg.GetData<double>()[0]);
        }

        [Test]
        public void Modf_NegativeZero()
        {
            // NumPy: modf(-0.0) = (-0.0, -0.0)
            var x = np.array(new double[] { -0.0 });
            var (frac, intg) = np.modf(x);

            // Check the sign bit is preserved for -0.0
            var fracVal = frac.GetData<double>()[0];
            var intgVal = intg.GetData<double>()[0];

            Assert.AreEqual(0.0, fracVal);  // Value is zero
            Assert.AreEqual(0.0, intgVal);  // Value is zero
        }

        #endregion

        #region Special Values (Inf, NaN)

        [Test]
        public void Modf_PositiveInfinity()
        {
            // NumPy: modf(inf) = (0.0, inf)
            var x = np.array(new double[] { double.PositiveInfinity });
            var (frac, intg) = np.modf(x);

            Assert.AreEqual(0.0, frac.GetData<double>()[0]);
            Assert.IsTrue(double.IsPositiveInfinity(intg.GetData<double>()[0]));
        }

        [Test]
        public void Modf_NegativeInfinity()
        {
            // NumPy: modf(-inf) = (-0.0, -inf)
            var x = np.array(new double[] { double.NegativeInfinity });
            var (frac, intg) = np.modf(x);

            // Fractional part is -0.0 (or 0.0)
            Assert.AreEqual(0.0, frac.GetData<double>()[0]);
            Assert.IsTrue(double.IsNegativeInfinity(intg.GetData<double>()[0]));
        }

        [Test]
        public void Modf_NaN()
        {
            // NumPy: modf(nan) = (nan, nan)
            var x = np.array(new double[] { double.NaN });
            var (frac, intg) = np.modf(x);

            Assert.IsTrue(double.IsNaN(frac.GetData<double>()[0]));
            Assert.IsTrue(double.IsNaN(intg.GetData<double>()[0]));
        }

        [Test]
        public void Modf_MixedSpecialValues()
        {
            // NumPy: modf([inf, -inf, nan])
            var x = np.array(new double[] { double.PositiveInfinity, double.NegativeInfinity, double.NaN });
            var (frac, intg) = np.modf(x);

            var fracData = frac.GetData<double>();
            var intgData = intg.GetData<double>();

            Assert.AreEqual(0.0, fracData[0]);
            Assert.IsTrue(double.IsPositiveInfinity(intgData[0]));

            Assert.AreEqual(0.0, fracData[1]);
            Assert.IsTrue(double.IsNegativeInfinity(intgData[1]));

            Assert.IsTrue(double.IsNaN(fracData[2]));
            Assert.IsTrue(double.IsNaN(intgData[2]));
        }

        #endregion

        #region Dtype Tests

        [Test]
        public void Modf_Float32_PreservesDtype()
        {
            var x = np.array(new float[] { 1.5f, 2.7f });
            var (frac, intg) = np.modf(x);

            Assert.AreEqual(np.float32, frac.dtype);
            Assert.AreEqual(np.float32, intg.dtype);
        }

        [Test]
        public void Modf_Float64_PreservesDtype()
        {
            var x = np.array(new double[] { 1.5, 2.7 });
            var (frac, intg) = np.modf(x);

            Assert.AreEqual(np.float64, frac.dtype);
            Assert.AreEqual(np.float64, intg.dtype);
        }

        #endregion

        #region Large Values

        [Test]
        public void Modf_LargeValues()
        {
            // NumPy: modf([1e10 + 0.5, -1e10 - 0.5])
            var x = np.array(new double[] { 1e10 + 0.5, -1e10 - 0.5 });
            var (frac, intg) = np.modf(x);

            var fracData = frac.GetData<double>();
            var intgData = intg.GetData<double>();

            Assert.IsTrue(Math.Abs(fracData[0] - 0.5) < 1e-5);
            Assert.IsTrue(Math.Abs(fracData[1] - (-0.5)) < 1e-5);
            Assert.AreEqual(1e10, intgData[0]);
            Assert.AreEqual(-1e10, intgData[1]);
        }

        #endregion

        #region Multi-dimensional Arrays

        [Test]
        public void Modf_2DArray()
        {
            // NumPy: modf([[1.5, 2.7], [3.1, 4.9]])
            var x = np.array(new double[,] { { 1.5, 2.7 }, { 3.1, 4.9 } });
            var (frac, intg) = np.modf(x);

            frac.Should().BeShaped(2, 2);
            intg.Should().BeShaped(2, 2);

            var fracData = frac.GetData<double>();
            Assert.IsTrue(Math.Abs(fracData[0] - 0.5) <1e-10);
            Assert.IsTrue(Math.Abs(fracData[1] - 0.7) <1e-10);
            Assert.IsTrue(Math.Abs(fracData[2] - 0.1) <1e-10);
            Assert.IsTrue(Math.Abs(fracData[3] - 0.9) <1e-10);

            var intgData = intg.GetData<double>();
            Assert.AreEqual(1.0, intgData[0]);
            Assert.AreEqual(2.0, intgData[1]);
            Assert.AreEqual(3.0, intgData[2]);
            Assert.AreEqual(4.0, intgData[3]);
        }

        #endregion

        #region Whole Numbers

        [Test]
        public void Modf_WholeNumbers()
        {
            // Whole numbers should have fractional part = 0
            var x = np.array(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
            var (frac, intg) = np.modf(x);

            var fracData = frac.GetData<double>();
            var intgData = intg.GetData<double>();

            for (int i = 0; i < 5; i++)
            {
                Assert.AreEqual(0.0, fracData[i]);
                Assert.AreEqual((double)(i + 1), intgData[i]);
            }
        }

        #endregion

        #region Empty Array

        [Test]
        public void Modf_EmptyArray()
        {
            var x = np.array(new double[0]);
            var (frac, intg) = np.modf(x);

            Assert.AreEqual(0, frac.size);
            Assert.AreEqual(0, intg.size);
        }

        #endregion

        #region Single Element

        [Test]
        public void Modf_SingleElement()
        {
            var x = np.array(new double[] { 3.14159 });
            var (frac, intg) = np.modf(x);

            Assert.IsTrue(Math.Abs(frac.GetData<double>()[0] - 0.14159) < 1e-5);
            Assert.AreEqual(3.0, intg.GetData<double>()[0]);
        }

        #endregion

        #region Very Small Fractional Parts

        [Test]
        public void Modf_VerySmallFraction()
        {
            // Values very close to integers
            var x = np.array(new double[] { 1.0000001, 1.9999999 });
            var (frac, intg) = np.modf(x);

            var fracData = frac.GetData<double>();
            var intgData = intg.GetData<double>();

            Assert.IsTrue(fracData[0] > 0.0);
            Assert.IsTrue(fracData[0] < 0.001);
            Assert.AreEqual(1.0, intgData[0]);

            Assert.IsTrue(fracData[1] > 0.999);
            Assert.IsTrue(fracData[1] < 1.0);
            Assert.AreEqual(1.0, intgData[1]);
        }

        #endregion
    }
}
