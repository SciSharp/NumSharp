using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using AwesomeAssertions;
using NumSharp.Backends;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Maths
{
    /// <summary>
    /// Tests for cumulative product (cumprod) functionality.
    /// These test the ILKernelGenerator.Scan.cs CumProd implementation.
    /// </summary>
    [TestClass]
    public class NDArrayCumprodTest : TestClass
    {
        /// <summary>
        /// Basic cumprod test with double array.
        /// NumPy: np.cumprod([1, 2, 3, 4, 5]) = [1, 2, 6, 24, 120]
        /// </summary>
        [TestMethod]
        public void CumprodBasicDouble()
        {
            NDArray arr = new double[] { 1, 2, 3, 4, 5 };
            NDArray expected = new double[] { 1, 2, 6, 24, 120 };

            var result = np.cumprod(arr);
            result.Should().Be(expected);
        }

        /// <summary>
        /// Cumprod with zeros - product becomes 0 and stays 0.
        /// NumPy: np.cumprod([1, 2, 0, 4, 5]) = [1, 2, 0, 0, 0]
        /// </summary>
        [TestMethod]
        public void CumprodWithZero()
        {
            NDArray arr = new double[] { 1, 2, 0, 4, 5 };
            NDArray expected = new double[] { 1, 2, 0, 0, 0 };

            var result = np.cumprod(arr);
            result.Should().Be(expected);
        }

        /// <summary>
        /// Cumprod with negative numbers.
        /// NumPy: np.cumprod([1, -2, 3, -4]) = [1, -2, -6, 24]
        /// </summary>
        [TestMethod]
        public void CumprodWithNegatives()
        {
            NDArray arr = new double[] { 1, -2, 3, -4 };
            NDArray expected = new double[] { 1, -2, -6, 24 };

            var result = np.cumprod(arr);
            result.Should().Be(expected);
        }

        /// <summary>
        /// Cumprod 2D without axis (flattens first).
        /// NumPy: np.cumprod([[1, 2, 3], [4, 5, 6]]) = [1, 2, 6, 24, 120, 720]
        /// For int32 input, NumPy 2.x returns int64.
        /// </summary>
        [TestMethod]
        public void Cumprod2dFlattened()
        {
            NDArray arr = new int[,] { { 1, 2, 3 }, { 4, 5, 6 } };
            // NumPy-aligned: cumprod of int32 returns int64
            NDArray expected = new long[] { 1, 2, 6, 24, 120, 720 };

            var result = np.cumprod(arr);
            result.Should().Be(expected);
        }

        /// <summary>
        /// Cumprod along axis 0 (rows).
        /// NumPy: np.cumprod([[1, 2, 3], [4, 5, 6]], axis=0) = [[1, 2, 3], [4, 10, 18]]
        /// For int32 input, NumPy 2.x returns int64.
        /// </summary>
        [TestMethod]
        public void Cumprod2dAxis0()
        {
            NDArray arr = new int[,] { { 1, 2, 3 }, { 4, 5, 6 } };
            // NumPy-aligned: cumprod of int32 returns int64
            NDArray expected = new long[,] { { 1, 2, 3 }, { 4, 10, 18 } };

            var result = np.cumprod(arr, axis: 0);
            result.Should().Be(expected);
        }

        /// <summary>
        /// Cumprod along axis 1 (columns).
        /// NumPy: np.cumprod([[1, 2, 3], [4, 5, 6]], axis=1) = [[1, 2, 6], [4, 20, 120]]
        /// For int32 input, NumPy 2.x returns int64.
        /// </summary>
        [TestMethod]
        public void Cumprod2dAxis1()
        {
            NDArray arr = new int[,] { { 1, 2, 3 }, { 4, 5, 6 } };
            // NumPy-aligned: cumprod of int32 returns int64
            NDArray expected = new long[,] { { 1, 2, 6 }, { 4, 20, 120 } };

            var result = np.cumprod(arr, axis: 1);
            result.Should().Be(expected);
        }

        /// <summary>
        /// Cumprod with single element.
        /// NumPy: np.cumprod([42]) = [42]
        /// </summary>
        [TestMethod]
        public void CumprodSingleElement()
        {
            NDArray arr = new double[] { 42 };
            NDArray expected = new double[] { 42 };

            var result = np.cumprod(arr);
            result.Should().Be(expected);
        }

        /// <summary>
        /// Cumprod with all ones.
        /// NumPy: np.cumprod([1, 1, 1, 1]) = [1, 1, 1, 1]
        /// </summary>
        [TestMethod]
        public void CumprodAllOnes()
        {
            NDArray arr = new double[] { 1, 1, 1, 1 };
            NDArray expected = new double[] { 1, 1, 1, 1 };

            var result = np.cumprod(arr);
            result.Should().Be(expected);
        }

        /// <summary>
        /// Cumprod with float32 type.
        /// </summary>
        [TestMethod]
        public void CumprodFloat32()
        {
            NDArray arr = np.array(new float[] { 1f, 2f, 3f, 4f });
            NDArray expected = np.array(new float[] { 1f, 2f, 6f, 24f });

            var result = np.cumprod(arr);
            result.typecode.Should().Be(NPTypeCode.Single);
            result.Should().Be(expected);
        }
    }
}
