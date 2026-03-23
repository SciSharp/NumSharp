using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;
using NumSharp.UnitTest.Utilities;
using TUnit.Core;

namespace NumSharp.UnitTest.NumPyPortedTests
{
    /// <summary>
    /// Tests for np.cumsum covering edge cases.
    /// Ported from NumPy test patterns.
    /// </summary>
    public class CumSumEdgeCaseTests
    {
        #region Basic Tests

        [Test]
        public void CumSum_1DArray_Int32()
        {
            // NumPy: cumsum([1,2,3,4,5]) = [1, 3, 6, 10, 15]
            var a = np.array(new int[] { 1, 2, 3, 4, 5 });
            var result = np.cumsum(a);
            // NumPy 2.x: cumsum of int32 returns int64
            result.Should().BeOfValues(1L, 3L, 6L, 10L, 15L);
        }

        [Test]
        public void CumSum_1DArray_Float64()
        {
            var a = np.array(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
            var result = np.cumsum(a);
            result.Should().BeOfValues(1.0, 3.0, 6.0, 10.0, 15.0);
        }

        #endregion

        #region 2D Array Tests

        [Test]
        public void CumSum_2DArray_Flattened()
        {
            // NumPy: cumsum([[1,2,3],[4,5,6]]) = [1, 3, 6, 10, 15, 21]
            var a = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var result = np.cumsum(a);
            result.Should().BeShaped(6);
            result.Should().BeOfValues(1L, 3L, 6L, 10L, 15L, 21L);
        }

        [Test]
        public void CumSum_2DArray_Axis0()
        {
            // NumPy: cumsum([[1,2,3],[4,5,6]], axis=0) = [[1,2,3],[5,7,9]]
            var a = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var result = np.cumsum(a, axis: 0);
            result.Should().BeShaped(2, 3);
            result.Should().BeOfValues(1L, 2L, 3L, 5L, 7L, 9L);
        }

        [Test]
        public void CumSum_2DArray_Axis1()
        {
            // NumPy: cumsum([[1,2,3],[4,5,6]], axis=1) = [[1,3,6],[4,9,15]]
            var a = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var result = np.cumsum(a, axis: 1);
            result.Should().BeShaped(2, 3);
            result.Should().BeOfValues(1L, 3L, 6L, 4L, 9L, 15L);
        }

        #endregion

        #region Dtype Handling

        [Test]
        public void CumSum_Int32_ReturnsInt64()
        {
            // NumPy 2.x: cumsum(int32) returns int64
            var arr = np.array(new int[] { 1, 2, 3 });
            var result = np.cumsum(arr);
            Assert.AreEqual(np.int64, result.dtype);
        }

        [Test]
        public void CumSum_Int64_ReturnsInt64()
        {
            var arr = np.array(new long[] { 1, 2, 3 });
            var result = np.cumsum(arr);
            Assert.AreEqual(np.int64, result.dtype);
        }

        [Test]
        public void CumSum_Float32_ReturnsFloat32()
        {
            var arr = np.array(new float[] { 1f, 2f, 3f });
            var result = np.cumsum(arr);
            Assert.AreEqual(np.float32, result.dtype);
        }

        [Test]
        public void CumSum_Float64_ReturnsFloat64()
        {
            var arr = np.array(new double[] { 1.0, 2.0, 3.0 });
            var result = np.cumsum(arr);
            Assert.AreEqual(np.float64, result.dtype);
        }

        [Test]
        public void CumSum_WithExplicitDtype()
        {
            var arr = np.array(new int[] { 1, 2, 3, 4, 5, 6 });
            var result = np.cumsum(arr, typeCode: NPTypeCode.Single);
            Assert.AreEqual(np.float32, result.dtype);
            result.Should().BeOfValues(1f, 3f, 6f, 10f, 15f, 21f);
        }

        #endregion

        #region Empty Array

        [Test]
        public void CumSum_EmptyArray_ReturnsEmptyArray()
        {
            // NumPy: cumsum([]) = array([], dtype=float64)
            var a = np.array(new double[0]);
            var result = np.cumsum(a);
            Assert.AreEqual(0, result.size);
        }

        #endregion

        #region Single Element

        [Test]
        public void CumSum_SingleElement()
        {
            // NumPy: cumsum([5]) = array([5])
            var a = np.array(new int[] { 5 });
            var result = np.cumsum(a);
            result.Should().BeOfValues(5L);
        }

        #endregion

        #region 3D Array

        [Test]
        public void CumSum_3DArray_Axis0()
        {
            // NumPy: cumsum(arange(24).reshape(2,3,4), axis=0).shape = (2, 3, 4)
            var a = np.arange(24).reshape(2, 3, 4);
            var result = np.cumsum(a, axis: 0);
            result.Should().BeShaped(2, 3, 4);
        }

        [Test]
        public void CumSum_3DArray_Axis1()
        {
            var a = np.arange(24).reshape(2, 3, 4);
            var result = np.cumsum(a, axis: 1);
            result.Should().BeShaped(2, 3, 4);
        }

        [Test]
        public void CumSum_3DArray_Axis2()
        {
            var a = np.arange(24).reshape(2, 3, 4);
            var result = np.cumsum(a, axis: 2);
            result.Should().BeShaped(2, 3, 4);
        }

        #endregion

        #region Negative Axis

        [Test]
        public void CumSum_NegativeAxis_Minus1()
        {
            // NumPy: cumsum(a, axis=-1) is same as axis=last dimension
            var a = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var result = np.cumsum(a, axis: -1);
            result.Should().BeShaped(2, 3);
            result.Should().BeOfValues(1L, 3L, 6L, 4L, 9L, 15L);
        }

        [Test]
        public void CumSum_NegativeAxis_Minus2()
        {
            // NumPy: cumsum(a, axis=-2) for 2D is same as axis=0
            var a = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var result = np.cumsum(a, axis: -2);
            result.Should().BeShaped(2, 3);
            result.Should().BeOfValues(1L, 2L, 3L, 5L, 7L, 9L);
        }

        #endregion

        #region Strided Array

        [Test]
        public void CumSum_StridedArray()
        {
            // NumPy: cumsum(arange(10)[::2]) = cumsum([0,2,4,6,8]) = [0,2,6,12,20]
            var a = np.arange(10);
            var strided = a["::" + 2];
            var result = np.cumsum(strided);
            result.Should().BeOfValues(0L, 2L, 6L, 12L, 20L);
        }

        #endregion

        #region Transposed Array

        [Test]
        public void CumSum_TransposedArray_Axis1()
        {
            // NumPy: cumsum([[1,2,3],[4,5,6]].T, axis=1)
            // [[1,2,3],[4,5,6]].T = [[1,4],[2,5],[3,6]]
            // cumsum axis=1: [[1,5],[2,7],[3,9]]
            var a = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var result = np.cumsum(a.T, axis: 1);
            result.Should().BeShaped(3, 2);
            result.Should().BeOfValues(1L, 5L, 2L, 7L, 3L, 9L);
        }

        #endregion

        #region All Zeros

        [Test]
        public void CumSum_AllZeros()
        {
            var a = np.zeros<int>(5);
            var result = np.cumsum(a);
            result.Should().BeOfValues(0L, 0L, 0L, 0L, 0L);
        }

        #endregion

        #region All Ones

        [Test]
        public void CumSum_AllOnes()
        {
            var a = np.ones<int>(5);
            var result = np.cumsum(a);
            result.Should().BeOfValues(1L, 2L, 3L, 4L, 5L);
        }

        #endregion

        #region Large Values (Overflow potential)

        [Test]
        public void CumSum_LargeValues_Int64()
        {
            // Test that large values don't overflow with int64
            var a = np.array(new long[] { 1_000_000_000L, 2_000_000_000L, 3_000_000_000L });
            var result = np.cumsum(a);
            result.Should().BeOfValues(1_000_000_000L, 3_000_000_000L, 6_000_000_000L);
        }

        #endregion

        #region Mixed Positive/Negative Values

        [Test]
        public void CumSum_MixedValues()
        {
            var a = np.array(new int[] { 1, -2, 3, -4, 5 });
            var result = np.cumsum(a);
            // 1, 1-2=-1, -1+3=2, 2-4=-2, -2+5=3
            result.Should().BeOfValues(1L, -1L, 2L, -2L, 3L);
        }

        #endregion

        #region Float Special Values

        [Test]
        public void CumSum_WithNaN()
        {
            var a = np.array(new double[] { 1.0, double.NaN, 3.0 });
            var result = np.cumsum(a);
            var data = result.GetData<double>();
            Assert.AreEqual(1.0, data[0]);
            Assert.IsTrue(double.IsNaN(data[1]));
            Assert.IsTrue(double.IsNaN(data[2]));  // NaN propagates
        }

        [Test]
        public void CumSum_WithInf()
        {
            var a = np.array(new double[] { 1.0, double.PositiveInfinity, 3.0 });
            var result = np.cumsum(a);
            var data = result.GetData<double>();
            Assert.AreEqual(1.0, data[0]);
            Assert.IsTrue(double.IsPositiveInfinity(data[1]));
            Assert.IsTrue(double.IsPositiveInfinity(data[2]));  // inf + 3 = inf
        }

        #endregion

        #region Numerical Precision

        [Test]
        public void CumSum_FloatPrecision()
        {
            // Test with values that might lose precision
            var a = np.array(new double[] { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 });
            var result = np.cumsum(a);
            var data = result.GetData<double>();

            // Last value should be close to 1.0
            Assert.IsTrue(Math.Abs(data[9] - 1.0) < 1e-10);
        }

        #endregion
    }
}
