using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.NumPyPortedTests
{
    /// <summary>
    /// Tests ported from NumPy test_numeric.py TestClip class.
    /// Covers edge cases for np.clip operation.
    /// </summary>
    [TestClass]
    public class ClipEdgeCaseTests
    {
        #region Basic Clip Tests

        [TestMethod]
        public void Clip_BasicIntArray()
        {
            // NumPy: clip([-1,5,2,3,10,-4,-9], 2, 7) = [2, 5, 2, 3, 7, 2, 2]
            var arr = np.array(new int[] { -1, 5, 2, 3, 10, -4, -9 });
            var result = np.clip(arr, 2, 7);
            result.Should().BeOfValues(2, 5, 2, 3, 7, 2, 2);
        }

        [TestMethod]
        public void Clip_2DArray()
        {
            // NumPy: clip(arange(12).reshape(3,4), 3, 8)
            var a = np.arange(12).reshape(3, 4);
            var result = np.clip(a, 3, 8);
            result.Should().BeShaped(3, 4);
            result.Should().BeOfValues(3, 3, 3, 3, 4, 5, 6, 7, 8, 8, 8, 8);
        }

        #endregion

        #region Clip with None Tests

        [TestMethod]
        public void Clip_NoneMin_OnlyMaxApplied()
        {
            // NumPy: clip(arange(10), None, 5) = [0,1,2,3,4,5,5,5,5,5]
            var a = np.arange(10);
            var result = np.clip(a, null, 5);
            result.Should().BeOfValues(0, 1, 2, 3, 4, 5, 5, 5, 5, 5);
        }

        [TestMethod]
        public void Clip_NoneMax_OnlyMinApplied()
        {
            // NumPy: clip(arange(10), 3, None) = [3,3,3,3,4,5,6,7,8,9]
            var a = np.arange(10);
            var result = np.clip(a, 3, null);
            result.Should().BeOfValues(3, 3, 3, 3, 4, 5, 6, 7, 8, 9);
        }

        [TestMethod]
        public void Clip_BothNone_ReturnsOriginal()
        {
            // NumPy: clip(arange(10), None, None) = arange(10)
            var a = np.arange(10);
            var result = np.clip(a, null, null);
            result.Should().BeOfValues(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
        }

        #endregion

        #region Min > Max Edge Case (from test_clip_value_min_max_flip)

        [TestMethod]
        public void Clip_MinGreaterThanMax_UsesMaxValue()
        {
            // NumPy: clip(arange(10), 5, 3) = all 3s
            // Per NumPy: result = minimum(maximum(a, amin), amax)
            var a = np.arange(10).astype(np.int64);
            var result = np.clip(a, 5L, 3L);
            result.Should().BeOfValues(3L, 3L, 3L, 3L, 3L, 3L, 3L, 3L, 3L, 3L);
        }

        #endregion

        #region NaN Handling (from test_clip_nan)

        [TestMethod]
        public void Clip_NaNMin_ReturnsAllNaN()
        {
            // NumPy: clip(arange(7.), min=np.nan) = [nan, nan, ...]
            var d = np.arange(7.0);  // float64 from double overload
            var result = np.clip(d, double.NaN, null);
            var data = result.GetData<double>();

            foreach (var val in data)
            {
                Assert.IsTrue(double.IsNaN(val));
            }
        }

        [TestMethod]
        public void Clip_NaNMax_ReturnsAllNaN()
        {
            // NumPy: clip(arange(7.), max=np.nan) = [nan, nan, ...]
            var d = np.arange(7.0);  // float64 from double overload
            var result = np.clip(d, null, double.NaN);
            var data = result.GetData<double>();

            foreach (var val in data)
            {
                Assert.IsTrue(double.IsNaN(val));
            }
        }

        #endregion

        #region Dtype Preservation Tests

        [TestMethod]
        public void Clip_Int32_PreservesDtype()
        {
            // Explicit int32 array for testing dtype preservation
            var arr = np.array(new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
            var result = np.clip(arr, 2, 7);
            Assert.AreEqual(np.int32, result.dtype);
        }

        [TestMethod]
        public void Clip_Int64_PreservesDtype()
        {
            // np.arange returns int64 by default (NumPy 2.x)
            var arr = np.arange(10);
            var result = np.clip(arr, 2L, 7L);
            Assert.AreEqual(np.int64, result.dtype);
        }

        [TestMethod]
        public void Clip_Float32_PreservesDtype()
        {
            var arr = np.arange(10f);  // float32 from float overload
            var result = np.clip(arr, 2f, 7f);
            Assert.AreEqual(np.float32, result.dtype);
        }

        [TestMethod]
        public void Clip_Float64_PreservesDtype()
        {
            var arr = np.arange(10.0);  // float64 from double overload
            var result = np.clip(arr, 2.0, 7.0);
            Assert.AreEqual(np.float64, result.dtype);
        }

        #endregion

        #region Strided Array Tests (from test_clip_non_contig)

        [TestMethod]
        public void Clip_StridedArray()
        {
            // NumPy: clip(a[::2], 3, 15) where a = arange(20)
            var a = np.arange(20);
            var strided = a["::" + 2];
            var result = np.clip(strided, 3, 15);

            result.Should().BeOfValues(3, 3, 4, 6, 8, 10, 12, 14, 15, 15);
        }

        #endregion

        #region Transposed Array Tests (from test_clip_with_out_transposed)

        [TestMethod]
        public void Clip_TransposedArray()
        {
            // NumPy: clip(arange(16).reshape(4,4).T, 4, 10)
            var a = np.arange(16).reshape(4, 4);
            var result = np.clip(a.T, 4, 10);

            result.Should().BeShaped(4, 4);
            result.Should().BeOfValues(4, 4, 8, 10, 4, 5, 9, 10, 4, 6, 10, 10, 4, 7, 10, 10);
        }

        #endregion

        #region Broadcasting Tests

        [TestMethod]
        public void Clip_BroadcastMin()
        {
            var a = np.arange(12).reshape(3, 4);
            var min = np.repeat(3, 12).reshape(3, 4);
            var result = np.clip(a, min, 8);

            result.Should().BeShaped(3, 4);
            result.Should().BeOfValues(3, 3, 3, 3, 4, 5, 6, 7, 8, 8, 8, 8);
        }

        [TestMethod]
        public void Clip_BroadcastMax()
        {
            var a = np.arange(12).reshape(3, 4);
            var max = np.repeat(8, 12).reshape(3, 4);
            var result = np.clip(a, 3, max);

            result.Should().BeShaped(3, 4);
            result.Should().BeOfValues(3, 3, 3, 3, 4, 5, 6, 7, 8, 8, 8, 8);
        }

        #endregion

        #region Special Float Values

        [TestMethod]
        public void Clip_WithInfinity()
        {
            var a = np.array(new double[] { -100.0, -1.0, 0.0, 1.0, 100.0 });

            var result1 = np.clip(a, double.NegativeInfinity, 50.0);
            result1.Should().BeOfValues(-100.0, -1.0, 0.0, 1.0, 50.0);

            var result2 = np.clip(a, -50.0, double.PositiveInfinity);
            result2.Should().BeOfValues(-50.0, -1.0, 0.0, 1.0, 100.0);
        }

        [TestMethod]
        public void Clip_InfiniteValuesInArray()
        {
            var a = np.array(new double[] { double.NegativeInfinity, -1.0, 0.0, 1.0, double.PositiveInfinity });
            var result = np.clip(a, -10.0, 10.0);

            result.Should().BeOfValues(-10.0, -1.0, 0.0, 1.0, 10.0);
        }

        #endregion

        #region Empty Array

        [TestMethod]
        public void Clip_EmptyArray_ReturnsEmptyArray()
        {
            var a = np.array(new double[0]);
            var result = np.clip(a, 0.0, 1.0);

            Assert.AreEqual(0, result.size);
        }

        #endregion

        #region Single Element

        [TestMethod]
        public void Clip_SingleElement_BelowMin()
        {
            var a = np.array(new double[] { -5.0 });
            var result = np.clip(a, 0.0, 10.0);
            result.Should().BeOfValues(0.0);
        }

        [TestMethod]
        public void Clip_SingleElement_AboveMax()
        {
            var a = np.array(new double[] { 15.0 });
            var result = np.clip(a, 0.0, 10.0);
            result.Should().BeOfValues(10.0);
        }

        [TestMethod]
        public void Clip_SingleElement_InRange()
        {
            var a = np.array(new double[] { 5.0 });
            var result = np.clip(a, 0.0, 10.0);
            result.Should().BeOfValues(5.0);
        }

        #endregion

        #region Scalar Min/Max

        [TestMethod]
        public void Clip_ScalarMinMax()
        {
            var a = np.arange(10);
            var result = np.clip(a, (NDArray)3, (NDArray)7);
            result.Should().BeOfValues(3, 3, 3, 3, 4, 5, 6, 7, 7, 7);
        }

        #endregion

        #region All Same Value

        [TestMethod]
        public void Clip_AllSameValue_BelowMin()
        {
            // NumPy-aligned: np.full(shape, fill_value)
            var a = np.full(new Shape(10), -5);
            var result = np.clip(a, 0, 10);
            result.GetData<long>().Should().AllBeEquivalentTo(0L);
        }

        [TestMethod]
        public void Clip_AllSameValue_AboveMax()
        {
            var a = np.full(new Shape(10), 15);
            var result = np.clip(a, 0, 10);
            result.GetData<long>().Should().AllBeEquivalentTo(10L);
        }

        [TestMethod]
        public void Clip_AllSameValue_InRange()
        {
            var a = np.full(new Shape(10), 5);
            var result = np.clip(a, 0, 10);
            result.GetData<long>().Should().AllBeEquivalentTo(5L);
        }

        #endregion
    }
}
