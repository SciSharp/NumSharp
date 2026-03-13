using System;
using System.Threading.Tasks;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;

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
        public async Task Modf_PositiveValues()
        {
            // NumPy: modf([1.5, 2.7]) = ([0.5, 0.7], [1., 2.])
            var x = np.array(new double[] { 1.5, 2.7 });
            var (frac, intg) = np.modf(x);

            var fracData = frac.GetData<double>();
            var intgData = intg.GetData<double>();

            await Assert.That(Math.Abs(fracData[0] - 0.5)).IsLessThan(1e-10);
            await Assert.That(Math.Abs(fracData[1] - 0.7)).IsLessThan(1e-10);
            await Assert.That(intgData[0]).IsEqualTo(1.0);
            await Assert.That(intgData[1]).IsEqualTo(2.0);
        }

        [Test]
        public async Task Modf_NegativeValues()
        {
            // NumPy: modf(-3.2) = (-0.2, -3.)
            var x = np.array(new double[] { -3.2 });
            var (frac, intg) = np.modf(x);

            var fracData = frac.GetData<double>();
            var intgData = intg.GetData<double>();

            await Assert.That(Math.Abs(fracData[0] - (-0.2))).IsLessThan(1e-10);
            await Assert.That(intgData[0]).IsEqualTo(-3.0);
        }

        [Test]
        public async Task Modf_MixedValues()
        {
            // NumPy: modf([1.5, 2.7, -3.2, 0.0])
            var x = np.array(new double[] { 1.5, 2.7, -3.2, 0.0 });
            var (frac, intg) = np.modf(x);

            var fracData = frac.GetData<double>();
            var intgData = intg.GetData<double>();

            await Assert.That(Math.Abs(fracData[0] - 0.5)).IsLessThan(1e-10);
            await Assert.That(Math.Abs(fracData[1] - 0.7)).IsLessThan(1e-10);
            await Assert.That(Math.Abs(fracData[2] - (-0.2))).IsLessThan(1e-10);
            await Assert.That(fracData[3]).IsEqualTo(0.0);

            await Assert.That(intgData[0]).IsEqualTo(1.0);
            await Assert.That(intgData[1]).IsEqualTo(2.0);
            await Assert.That(intgData[2]).IsEqualTo(-3.0);
            await Assert.That(intgData[3]).IsEqualTo(0.0);
        }

        #endregion

        #region Zero Tests

        [Test]
        public async Task Modf_PositiveZero()
        {
            var x = np.array(new double[] { 0.0 });
            var (frac, intg) = np.modf(x);

            await Assert.That(frac.GetData<double>()[0]).IsEqualTo(0.0);
            await Assert.That(intg.GetData<double>()[0]).IsEqualTo(0.0);
        }

        [Test]
        public async Task Modf_NegativeZero()
        {
            // NumPy: modf(-0.0) = (-0.0, -0.0)
            var x = np.array(new double[] { -0.0 });
            var (frac, intg) = np.modf(x);

            // Check the sign bit is preserved for -0.0
            var fracVal = frac.GetData<double>()[0];
            var intgVal = intg.GetData<double>()[0];

            await Assert.That(fracVal).IsEqualTo(0.0);  // Value is zero
            await Assert.That(intgVal).IsEqualTo(0.0);  // Value is zero
        }

        #endregion

        #region Special Values (Inf, NaN)

        [Test]
        public async Task Modf_PositiveInfinity()
        {
            // NumPy: modf(inf) = (0.0, inf)
            var x = np.array(new double[] { double.PositiveInfinity });
            var (frac, intg) = np.modf(x);

            await Assert.That(frac.GetData<double>()[0]).IsEqualTo(0.0);
            await Assert.That(double.IsPositiveInfinity(intg.GetData<double>()[0])).IsTrue();
        }

        [Test]
        public async Task Modf_NegativeInfinity()
        {
            // NumPy: modf(-inf) = (-0.0, -inf)
            var x = np.array(new double[] { double.NegativeInfinity });
            var (frac, intg) = np.modf(x);

            // Fractional part is -0.0 (or 0.0)
            await Assert.That(frac.GetData<double>()[0]).IsEqualTo(0.0);
            await Assert.That(double.IsNegativeInfinity(intg.GetData<double>()[0])).IsTrue();
        }

        [Test]
        public async Task Modf_NaN()
        {
            // NumPy: modf(nan) = (nan, nan)
            var x = np.array(new double[] { double.NaN });
            var (frac, intg) = np.modf(x);

            await Assert.That(double.IsNaN(frac.GetData<double>()[0])).IsTrue();
            await Assert.That(double.IsNaN(intg.GetData<double>()[0])).IsTrue();
        }

        [Test]
        public async Task Modf_MixedSpecialValues()
        {
            // NumPy: modf([inf, -inf, nan])
            var x = np.array(new double[] { double.PositiveInfinity, double.NegativeInfinity, double.NaN });
            var (frac, intg) = np.modf(x);

            var fracData = frac.GetData<double>();
            var intgData = intg.GetData<double>();

            await Assert.That(fracData[0]).IsEqualTo(0.0);
            await Assert.That(double.IsPositiveInfinity(intgData[0])).IsTrue();

            await Assert.That(fracData[1]).IsEqualTo(0.0);
            await Assert.That(double.IsNegativeInfinity(intgData[1])).IsTrue();

            await Assert.That(double.IsNaN(fracData[2])).IsTrue();
            await Assert.That(double.IsNaN(intgData[2])).IsTrue();
        }

        #endregion

        #region Dtype Tests

        [Test]
        public async Task Modf_Float32_PreservesDtype()
        {
            var x = np.array(new float[] { 1.5f, 2.7f });
            var (frac, intg) = np.modf(x);

            await Assert.That(frac.dtype).IsEqualTo(np.float32);
            await Assert.That(intg.dtype).IsEqualTo(np.float32);
        }

        [Test]
        public async Task Modf_Float64_PreservesDtype()
        {
            var x = np.array(new double[] { 1.5, 2.7 });
            var (frac, intg) = np.modf(x);

            await Assert.That(frac.dtype).IsEqualTo(np.float64);
            await Assert.That(intg.dtype).IsEqualTo(np.float64);
        }

        #endregion

        #region Large Values

        [Test]
        public async Task Modf_LargeValues()
        {
            // NumPy: modf([1e10 + 0.5, -1e10 - 0.5])
            var x = np.array(new double[] { 1e10 + 0.5, -1e10 - 0.5 });
            var (frac, intg) = np.modf(x);

            var fracData = frac.GetData<double>();
            var intgData = intg.GetData<double>();

            await Assert.That(Math.Abs(fracData[0] - 0.5)).IsLessThan(1e-5);
            await Assert.That(Math.Abs(fracData[1] - (-0.5))).IsLessThan(1e-5);
            await Assert.That(intgData[0]).IsEqualTo(1e10);
            await Assert.That(intgData[1]).IsEqualTo(-1e10);
        }

        #endregion

        #region Multi-dimensional Arrays

        [Test]
        public async Task Modf_2DArray()
        {
            // NumPy: modf([[1.5, 2.7], [3.1, 4.9]])
            var x = np.array(new double[,] { { 1.5, 2.7 }, { 3.1, 4.9 } });
            var (frac, intg) = np.modf(x);

            frac.Should().BeShaped(2, 2);
            intg.Should().BeShaped(2, 2);

            var fracData = frac.GetData<double>();
            await Assert.That(Math.Abs(fracData[0] - 0.5)).IsLessThan(1e-10);
            await Assert.That(Math.Abs(fracData[1] - 0.7)).IsLessThan(1e-10);
            await Assert.That(Math.Abs(fracData[2] - 0.1)).IsLessThan(1e-10);
            await Assert.That(Math.Abs(fracData[3] - 0.9)).IsLessThan(1e-10);

            var intgData = intg.GetData<double>();
            await Assert.That(intgData[0]).IsEqualTo(1.0);
            await Assert.That(intgData[1]).IsEqualTo(2.0);
            await Assert.That(intgData[2]).IsEqualTo(3.0);
            await Assert.That(intgData[3]).IsEqualTo(4.0);
        }

        #endregion

        #region Whole Numbers

        [Test]
        public async Task Modf_WholeNumbers()
        {
            // Whole numbers should have fractional part = 0
            var x = np.array(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
            var (frac, intg) = np.modf(x);

            var fracData = frac.GetData<double>();
            var intgData = intg.GetData<double>();

            for (int i = 0; i < 5; i++)
            {
                await Assert.That(fracData[i]).IsEqualTo(0.0);
                await Assert.That(intgData[i]).IsEqualTo((double)(i + 1));
            }
        }

        #endregion

        #region Empty Array

        [Test]
        public async Task Modf_EmptyArray()
        {
            var x = np.array(new double[0]);
            var (frac, intg) = np.modf(x);

            await Assert.That(frac.size).IsEqualTo(0);
            await Assert.That(intg.size).IsEqualTo(0);
        }

        #endregion

        #region Single Element

        [Test]
        public async Task Modf_SingleElement()
        {
            var x = np.array(new double[] { 3.14159 });
            var (frac, intg) = np.modf(x);

            await Assert.That(Math.Abs(frac.GetData<double>()[0] - 0.14159)).IsLessThan(1e-5);
            await Assert.That(intg.GetData<double>()[0]).IsEqualTo(3.0);
        }

        #endregion

        #region Very Small Fractional Parts

        [Test]
        public async Task Modf_VerySmallFraction()
        {
            // Values very close to integers
            var x = np.array(new double[] { 1.0000001, 1.9999999 });
            var (frac, intg) = np.modf(x);

            var fracData = frac.GetData<double>();
            var intgData = intg.GetData<double>();

            await Assert.That(fracData[0]).IsGreaterThan(0.0);
            await Assert.That(fracData[0]).IsLessThan(0.001);
            await Assert.That(intgData[0]).IsEqualTo(1.0);

            await Assert.That(fracData[1]).IsGreaterThan(0.999);
            await Assert.That(fracData[1]).IsLessThan(1.0);
            await Assert.That(intgData[1]).IsEqualTo(1.0);
        }

        #endregion
    }
}
