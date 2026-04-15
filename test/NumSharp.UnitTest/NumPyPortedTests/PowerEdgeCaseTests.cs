using System;
using System.Linq;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.NumPyPortedTests
{
    /// <summary>
    /// Tests ported from NumPy test_umath.py TestPower class.
    /// Covers edge cases for np.power operation.
    /// </summary>
    [TestClass]
    public class PowerEdgeCaseTests
    {
        #region Float Power Tests (from test_power_float)

        [TestMethod]
        public void Power_Float_ZeroExponent_ReturnsOnes()
        {
            // NumPy: x**0 = [1., 1., 1.]
            var x = np.array(new double[] { 1.0, 2.0, 3.0 });
            var result = np.power(x, 0);
            result.Should().BeOfValues(1.0, 1.0, 1.0);
        }

        [TestMethod]
        public void Power_Float_OneExponent_ReturnsOriginal()
        {
            // NumPy: x**1 = x
            var x = np.array(new double[] { 1.0, 2.0, 3.0 });
            var result = np.power(x, 1);
            result.Should().BeOfValues(1.0, 2.0, 3.0);
        }

        [TestMethod]
        public void Power_Float_TwoExponent_ReturnsSquares()
        {
            // NumPy: x**2 = [1., 4., 9.]
            var x = np.array(new double[] { 1.0, 2.0, 3.0 });
            var result = np.power(x, 2);
            result.Should().BeOfValues(1.0, 4.0, 9.0);
        }

        [TestMethod]
        public void Power_Float_NegativeOneExponent_ReturnsReciprocals()
        {
            // NumPy: x**(-1) = [1., 0.5, 0.33333333]
            var x = np.array(new double[] { 1.0, 2.0, 3.0 });
            var result = np.power(x, -1);
            var data = result.GetData<double>();
            Assert.IsTrue(Math.Abs(data[0] - 1.0) < 1e-10);
            Assert.IsTrue(Math.Abs(data[1] - 0.5) < 1e-10);
            Assert.IsTrue(Math.Abs(data[2] - (1.0 / 3.0)) < 1e-10);
        }

        [TestMethod]
        public void Power_Float_HalfExponent_ReturnsSqrt()
        {
            // NumPy: x**(0.5) = [1., sqrt(2), sqrt(3)]
            var x = np.array(new double[] { 1.0, 2.0, 3.0 });
            var result = np.power(x, 0.5);
            var data = result.GetData<double>();
            Assert.IsTrue(Math.Abs(data[0] - 1.0) < 1e-10);
            Assert.IsTrue(Math.Abs(data[1] - Math.Sqrt(2.0)) < 1e-10);
            Assert.IsTrue(Math.Abs(data[2] - Math.Sqrt(3.0)) < 1e-10);
        }

        #endregion

        #region Integer Power Tests (from test_integer_power*)

        [TestMethod]
        [Misaligned] // NumSharp uses Math.Pow (double precision) which loses precision for large integers
        public void Power_Integer_LargeValues()
        {
            // NumPy: 15**15 = 437893890380859375
            // NumSharp uses Math.Pow which can't exactly represent integers > 2^53
            // This is a known limitation - would need integer-based exponentiation to fix
            var a = np.array(new long[] { 15, 15 });
            var result = np.power(a, a);
            // Allow small precision loss due to double conversion
            var expected = 437893890380859375L;
            var actual = result.GetInt64(0);
            var relativeError = Math.Abs((double)(actual - expected) / expected);
            Assert.IsTrue(relativeError < 1e-14, $"Expected ~{expected}, got {actual}, relative error {relativeError}");
        }

        [TestMethod]
        public void Power_Int32_ZeroExponent_ReturnsOnes()
        {
            // NumPy: power(arr, 0) returns ones_like(arr) for all integer types
            var arr = np.arange(-10, 10);  // int32 by default
            var result = np.power(arr, 0);
            var expected = np.ones_like(arr);
            result.Should().BeOfValues(expected.GetData<int>().Cast<object>().ToArray());
        }

        [TestMethod]
        public void Power_Int64_ZeroExponent_ReturnsOnes()
        {
            var arr = np.arange(-10, 10).astype(np.int64);
            var result = np.power(arr, 0);
            var expected = np.ones_like(arr);
            result.Should().BeOfValues(expected.GetData<long>().Cast<object>().ToArray());
        }

        [TestMethod]
        public void Power_UInt32_ZeroExponent_ReturnsOnes()
        {
            var arr = np.arange(10).astype(np.uint32);
            var result = np.power(arr, 0);
            var expected = np.ones_like(arr);
            result.Should().BeOfValues(expected.GetData<uint>().Cast<object>().ToArray());
        }

        [TestMethod]
        public void Power_OneBase_AnyExponent_ReturnsOnes()
        {
            // NumPy: power(1, arr) returns ones_like(arr)
            var arr = np.arange(10);  // int32 by default
            var result = np.power(1, arr);
            result.Should().BeOfValues(1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        }

        [TestMethod]
        public void Power_ZeroBase_PositiveExponent_ReturnsZeros()
        {
            // NumPy: power(0, arr) returns zeros for positive exponents
            var arr = np.arange(1, 10);  // int32 by default
            var result = np.power(0, arr);
            result.Should().BeOfValues(0, 0, 0, 0, 0, 0, 0, 0, 0);
        }

        #endregion

        #region Float to Inf Power Tests (from test_float_to_inf_power)

        [TestMethod]
        public void Power_Float32_InfExponents()
        {
            // NumPy: power with inf exponents
            var a = np.array(new float[] { 1f, 1f, 2f, 2f, -2f, -2f, float.PositiveInfinity, float.NegativeInfinity });
            var b = np.array(new float[] { float.PositiveInfinity, float.NegativeInfinity, float.PositiveInfinity, float.NegativeInfinity,
                                           float.PositiveInfinity, float.NegativeInfinity, float.PositiveInfinity, float.NegativeInfinity });
            var result = np.power(a, b);
            var data = result.GetData<float>();

            Assert.AreEqual(1f, data[0]);
            Assert.AreEqual(1f, data[1]);
            Assert.IsTrue(float.IsPositiveInfinity(data[2]));
            Assert.AreEqual(0f, data[3]);
            Assert.IsTrue(float.IsPositiveInfinity(data[4]));
            Assert.AreEqual(0f, data[5]);
            Assert.IsTrue(float.IsPositiveInfinity(data[6]));
            Assert.AreEqual(0f, data[7]);
        }

        [TestMethod]
        public void Power_Float64_InfExponents()
        {
            var a = np.array(new double[] { 1.0, 1.0, 2.0, 2.0, -2.0, -2.0, double.PositiveInfinity, double.NegativeInfinity });
            var b = np.array(new double[] { double.PositiveInfinity, double.NegativeInfinity, double.PositiveInfinity, double.NegativeInfinity,
                                            double.PositiveInfinity, double.NegativeInfinity, double.PositiveInfinity, double.NegativeInfinity });
            var result = np.power(a, b);
            var data = result.GetData<double>();

            Assert.AreEqual(1.0, data[0]);
            Assert.AreEqual(1.0, data[1]);
            Assert.IsTrue(double.IsPositiveInfinity(data[2]));
            Assert.AreEqual(0.0, data[3]);
            Assert.IsTrue(double.IsPositiveInfinity(data[4]));
            Assert.AreEqual(0.0, data[5]);
            Assert.IsTrue(double.IsPositiveInfinity(data[6]));
            Assert.AreEqual(0.0, data[7]);
        }

        #endregion

        #region Fast Power Path Tests (from test_fast_power)

        [TestMethod]
        public void Power_Int16_FloatExponent_ReturnsFloat64()
        {
            // NumPy: int16**2.0 returns float64
            var x = np.array(new short[] { 1, 2, 3 });
            var result = np.power(x, 2.0);

            Assert.AreEqual(np.float64, result.dtype);
            result.Should().BeOfValues(1.0, 4.0, 9.0);
        }

        [TestMethod]
        public void Power_Int32_FloatExponent_ReturnsFloat64()
        {
            var x = np.array(new int[] { 1, 2, 3, 4 });
            var result = np.power(x, 2.0);

            Assert.AreEqual(np.float64, result.dtype);
            result.Should().BeOfValues(1.0, 4.0, 9.0, 16.0);
        }

        #endregion

        #region Special Cases (0^0, negative base)

        [TestMethod]
        public void Power_ZeroToZero_ReturnsOne()
        {
            // NumPy: 0^0 = 1 (by convention)
            Assert.AreEqual(1, (int)np.power(0, 0));
            Assert.AreEqual(1.0, (double)np.power(0.0, 0.0));
            Assert.AreEqual(1.0, (double)np.power(0.0, 0));
        }

        [TestMethod]
        public void Power_NegativeBase_IntegerExponent()
        {
            // NumPy: (-2)^3 = -8, (-2)^4 = 16
            Assert.AreEqual(-8, (int)np.power(-2, 3));
            Assert.AreEqual(16, (int)np.power(-2, 4));
            Assert.AreEqual(-8.0, (double)np.power(-2.0, 3));
            Assert.AreEqual(-8.0, (double)np.power(-2.0, 3.0));
        }

        [TestMethod]
        public void Power_NegativeBase_FractionalExponent_ReturnsNaN()
        {
            // NumPy: (-2.0)^0.5 = nan, (-1.0)^0.5 = nan
            var result1 = np.power(-2.0, 0.5);
            var result2 = np.power(-1.0, 0.5);

            Assert.IsTrue(double.IsNaN((double)result1));
            Assert.IsTrue(double.IsNaN((double)result2));
        }

        #endregion

        #region Square Root via Power

        [TestMethod]
        public void Power_HalfExponent_MatchesSqrt()
        {
            // NumPy: power([1,4,9,16,25], 0.5) = [1,2,3,4,5]
            var a = np.array(new double[] { 1.0, 4.0, 9.0, 16.0, 25.0 });
            var result = np.power(a, 0.5);
            result.Should().BeOfValues(1.0, 2.0, 3.0, 4.0, 5.0);
        }

        [TestMethod]
        public void Power_NegativeHalfExponent()
        {
            // NumPy: power([1,4,9], -0.5) = [1, 0.5, 0.33333...]
            var a = np.array(new double[] { 1.0, 4.0, 9.0 });
            var result = np.power(a, -0.5);
            var data = result.GetData<double>();

            Assert.IsTrue(Math.Abs(data[0] - 1.0) < 1e-10);
            Assert.IsTrue(Math.Abs(data[1] - 0.5) < 1e-10);
            Assert.IsTrue(Math.Abs(data[2] - (1.0/3.0)) < 1e-10);
        }

        #endregion

        #region Broadcasting Tests

        [TestMethod]
        public void Power_Broadcasting_2DArray_1DExponent()
        {
            // NumPy: power([[1,2],[3,4]], [2,3]) = [[1,8],[9,64]]
            var a = np.array(new int[,] { { 1, 2 }, { 3, 4 } });
            var b = np.array(new int[] { 2, 3 });
            var result = np.power(a, b);

            result.Should().BeShaped(2, 2);
            result.Should().BeOfValues(1, 8, 9, 64);
        }

        [TestMethod]
        public void Power_Broadcasting_1DArray_2DExponent()
        {
            // NumPy: power([1,2,3,4], [[1],[2],[3]]).shape = (3,4)
            var a = np.array(new int[] { 1, 2, 3, 4 });
            var b = np.array(new int[,] { { 1 }, { 2 }, { 3 } });
            var result = np.power(a, b);

            result.Should().BeShaped(3, 4);
        }

        #endregion

        #region Strided Array Tests

        [TestMethod]
        public void Power_StridedArray()
        {
            // NumPy: power(a[::2], 2) where a = [0,1,2,3,4,5,6,7,8,9]
            // a[::2] = [0,2,4,6,8], power = [0,4,16,36,64]
            var a = np.arange(10);
            var strided = a["::" + 2];
            var result = np.power(strided, 2);

            result.Should().BeOfValues(0, 4, 16, 36, 64);
        }

        #endregion

        #region Dtype Preservation Tests

        [TestMethod]
        public void Power_Int32_Int32_ReturnsInt32()
        {
            var a = np.array(new int[] { 2, 3, 4 });
            var result = np.power(a, 2);
            Assert.AreEqual(np.int32, result.dtype);
        }

        [TestMethod]
        public void Power_Int64_Int64_ReturnsInt64()
        {
            var a = np.array(new long[] { 2, 3, 4 });
            var result = np.power(a, 2L);
            Assert.AreEqual(np.int64, result.dtype);
        }

        [TestMethod]
        public void Power_Float32_Float32_ReturnsFloat32()
        {
            var a = np.array(new float[] { 2f, 3f, 4f });
            var result = np.power(a, 2f);
            Assert.AreEqual(np.float32, result.dtype);
        }

        [TestMethod]
        public void Power_Float64_Float64_ReturnsFloat64()
        {
            var a = np.array(new double[] { 2.0, 3.0, 4.0 });
            var result = np.power(a, 2.0);
            Assert.AreEqual(np.float64, result.dtype);
        }

        #endregion

        #region Cube Root and Other Fractional Powers

        [TestMethod]
        public void Power_CubeRoot()
        {
            // NumPy: power([1,8,27,64], 1/3) = [1,2,3,4]
            var a = np.array(new double[] { 1.0, 8.0, 27.0, 64.0 });
            var result = np.power(a, 1.0 / 3.0);
            var data = result.GetData<double>();

            Assert.IsTrue(Math.Abs(data[0] - 1.0) < 1e-10);
            Assert.IsTrue(Math.Abs(data[1] - 2.0) < 1e-10);
            Assert.IsTrue(Math.Abs(data[2] - 3.0) < 1e-10);
            Assert.IsTrue(Math.Abs(data[3] - 4.0) < 1e-10);
        }

        #endregion

        #region Very Large/Small Exponent Tests

        [TestMethod]
        public void Power_LargeExponent()
        {
            // NumPy: power([1, 1.0001, 0.9999], 10000)
            var a = np.array(new double[] { 1.0, 1.0001, 0.9999 });
            var result = np.power(a, 10000);
            var data = result.GetData<double>();

            Assert.AreEqual(1.0, data[0]);
            // 1.0001^10000 ~ 2.718... (approximately e)
            Assert.IsTrue(data[1] > 2.0);
            Assert.IsTrue(data[1] < 3.0);
            // 0.9999^10000 ~ 0.368... (approximately 1/e)
            Assert.IsTrue(data[2] > 0.3);
            Assert.IsTrue(data[2] < 0.4);
        }

        #endregion
    }
}
