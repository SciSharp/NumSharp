using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.NumPyPortedTests
{
    /// <summary>
    /// Tests for np.clip with array-valued min/max bounds.
    /// Validates the IL kernel migration of Default.ClipNDArray.cs.
    /// </summary>
    public class ClipNDArrayTests
    {
        #region Basic Array Bounds Tests

        [TestMethod]
        public void ClipNDArray_BasicArrayBounds_MatchesNumPy()
        {
            // NumPy: np.clip([1,2,3,4,5,6,7,8,9], [2,2,2,3,3,3,4,4,4], [5,5,5,6,6,6,7,7,7])
            // Expected: [2, 2, 3, 4, 5, 6, 7, 7, 7]
            var a = np.array(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 });
            var min_arr = np.array(new int[] { 2, 2, 2, 3, 3, 3, 4, 4, 4 });
            var max_arr = np.array(new int[] { 5, 5, 5, 6, 6, 6, 7, 7, 7 });

            var result = np.clip(a, min_arr, max_arr);

            result.Should().BeOfValues(2, 2, 3, 4, 5, 6, 7, 7, 7);
        }

        [TestMethod]
        public void ClipNDArray_MinArrayOnly_MatchesNumPy()
        {
            // NumPy: np.clip([1,2,3,4,5], [2,2,2,2,2], None) = [2,2,3,4,5]
            var a = np.array(new int[] { 1, 2, 3, 4, 5 });
            var min_arr = np.array(new int[] { 2, 2, 2, 2, 2 });

            var result = np.clip(a, min_arr, null);

            result.Should().BeOfValues(2, 2, 3, 4, 5);
        }

        [TestMethod]
        public void ClipNDArray_MaxArrayOnly_MatchesNumPy()
        {
            // NumPy: np.clip([1,2,3,4,5], None, [3,3,3,3,3]) = [1,2,3,3,3]
            var a = np.array(new int[] { 1, 2, 3, 4, 5 });
            var max_arr = np.array(new int[] { 3, 3, 3, 3, 3 });

            var result = np.clip(a, null, max_arr);

            result.Should().BeOfValues(1, 2, 3, 3, 3);
        }

        #endregion

        #region Broadcasting Tests

        [TestMethod]
        public void ClipNDArray_BroadcastMinAlongAxis0_MatchesNumPy()
        {
            // NumPy:
            // a = np.arange(12).reshape(3, 4) = [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
            // min_arr = [2, 3, 4, 5] (broadcasts along axis 0)
            // np.clip(a, min_arr, None) = [[2,3,4,5],[4,5,6,7],[8,9,10,11]]
            var a = np.arange(12).reshape(3, 4);
            var min_arr = np.array(new int[] { 2, 3, 4, 5 });

            var result = np.clip(a, min_arr, null);

            result.Should().BeShaped(3, 4);
            result.Should().BeOfValues(2, 3, 4, 5, 4, 5, 6, 7, 8, 9, 10, 11);
        }

        [TestMethod]
        public void ClipNDArray_BroadcastMaxAlongAxis0_MatchesNumPy()
        {
            // NumPy:
            // a = np.arange(12).reshape(3, 4)
            // max_arr = [7, 8, 9, 10]
            // np.clip(a, None, max_arr) = [[0,1,2,3],[4,5,6,7],[7,8,9,10]]
            var a = np.arange(12).reshape(3, 4);
            var max_arr = np.array(new int[] { 7, 8, 9, 10 });

            var result = np.clip(a, null, max_arr);

            result.Should().BeShaped(3, 4);
            result.Should().BeOfValues(0, 1, 2, 3, 4, 5, 6, 7, 7, 8, 9, 10);
        }

        [TestMethod]
        public void ClipNDArray_ScalarMinArrayMax_MatchesNumPy()
        {
            // Mixed: scalar min, array max
            var a = np.arange(12).reshape(3, 4);
            var max_arr = np.repeat(8, 12).reshape(3, 4);

            var result = np.clip(a, 3, max_arr);

            result.Should().BeShaped(3, 4);
            result.Should().BeOfValues(3, 3, 3, 3, 4, 5, 6, 7, 8, 8, 8, 8);
        }

        [TestMethod]
        public void ClipNDArray_ArrayMinNullMax_MatchesNumPy()
        {
            // From np.clip.Test.cs Case2
            var a = np.arange(12).reshape(3, 4);
            var minmax = np.repeat(8, 12).reshape(3, 4);

            var result = np.clip(a, minmax, null);

            result.Should().BeShaped(3, 4);
            result.Should().BeOfValues(8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 10, 11);
        }

        [TestMethod]
        public void ClipNDArray_NullMinArrayMax_MatchesNumPy()
        {
            // From np.clip.Test.cs Case3
            var a = np.arange(12).reshape(3, 4);
            var max = np.repeat(8, 12).reshape(3, 4);

            var result = np.clip(a, null, max);

            result.Should().BeShaped(3, 4);
            result.Should().BeOfValues(0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8);
        }

        #endregion

        #region Edge Cases

        [TestMethod]
        public void ClipNDArray_MinGreaterThanMax_UsesMaxValue()
        {
            // NumPy behavior: when min[i] > max[i], result is max[i]
            // np.clip([1,2,3,4,5], [6,6,6,6,6], [3,3,3,3,3]) = [3,3,3,3,3]
            var a = np.array(new int[] { 1, 2, 3, 4, 5 });
            var min_arr = np.array(new int[] { 6, 6, 6, 6, 6 });
            var max_arr = np.array(new int[] { 3, 3, 3, 3, 3 });

            var result = np.clip(a, min_arr, max_arr);

            result.Should().BeOfValues(3, 3, 3, 3, 3);
        }

        [TestMethod]
        [Misaligned]
        public void ClipNDArray_NaNInBoundsArray_PropagatesNaN()
        {
            // NumPy: NaN in bounds propagates to result
            // NumSharp: IComparable.CompareTo doesn't handle NaN propagation
            // This is a known behavioral difference - NaN comparison returns false,
            // so the value is not clipped and remains unchanged.
            var a = np.array(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
            var min_arr = np.array(new double[] { double.NaN, 2.0, 3.0, 3.0, 4.0 });

            var result = np.clip(a, min_arr, null);
            var data = result.GetData<double>();

            // In NumPy: data[0] would be NaN
            // In NumSharp: data[0] stays as 1.0 because NaN comparison returns false
            Assert.IsTrue(double.IsNaN(data[0]), "Expected NaN at index 0");
            Assert.AreEqual(2.0, data[1]);
            Assert.AreEqual(3.0, data[2]);
            Assert.AreEqual(4.0, data[3]);
            Assert.AreEqual(5.0, data[4]);
        }

        [TestMethod]
        public void ClipNDArray_EmptyArray_ReturnsEmpty()
        {
            var a = np.array(new double[0]);
            var min_arr = np.array(new double[0]);
            var max_arr = np.array(new double[0]);

            var result = np.clip(a, min_arr, max_arr);

            Assert.AreEqual(0, result.size);
        }

        [TestMethod]
        public void ClipNDArray_BothNone_ReturnsCopy()
        {
            var a = np.arange(10);
            var result = np.clip(a, null, null);

            result.Should().BeOfValues(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
        }

        #endregion

        #region Dtype Tests

        [TestMethod]
        public void ClipNDArray_Float64Array_PreservesDtype()
        {
            var a = np.arange(10.0);
            var min_arr = np.array(new double[] { 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 });
            var max_arr = np.array(new double[] { 7, 7, 7, 7, 7, 7, 7, 7, 7, 7 });

            var result = np.clip(a, min_arr, max_arr);

            Assert.AreEqual(np.float64, result.dtype);
            result.Should().BeOfValues(2.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.0, 7.0);
        }

        [TestMethod]
        public void ClipNDArray_Int32Array_PreservesDtype()
        {
            // Explicit int32 array for testing dtype preservation
            var a = np.array(new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
            var min_arr = np.array(new int[] { 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 });
            var max_arr = np.array(new int[] { 7, 7, 7, 7, 7, 7, 7, 7, 7, 7 });

            var result = np.clip(a, min_arr, max_arr);

            Assert.AreEqual(np.int32, result.dtype);
            result.Should().BeOfValues(2, 2, 2, 3, 4, 5, 6, 7, 7, 7);
        }

        #endregion

        #region Contiguous vs Non-Contiguous Tests

        [TestMethod]
        public void ClipNDArray_TransposedArray_MatchesNumPy()
        {
            // Non-contiguous input (transposed)
            // NumPy: np.clip(arange(12).reshape(3,4).T, 2, 8)
            var a = np.arange(12).reshape(3, 4).T;
            var min_arr = np.array(new int[] { 2, 2, 2 });  // broadcasts to (4,3)
            var max_arr = np.array(new int[] { 8, 8, 8 });

            var result = np.clip(a, min_arr, max_arr);

            result.Should().BeShaped(4, 3);
            // Expected: [[2,4,8],[2,5,8],[2,6,8],[3,7,8]]
            result.Should().BeOfValues(2, 4, 8, 2, 5, 8, 2, 6, 8, 3, 7, 8);
        }

        [TestMethod]
        public void ClipNDArray_SlicedArray_MatchesNumPy()
        {
            // Sliced input (every other element)
            var a = np.arange(20);
            var sliced = a["::" + 2];  // [0,2,4,6,8,10,12,14,16,18]
            var min_arr = np.array(new int[] { 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 });
            var max_arr = np.array(new int[] { 15, 15, 15, 15, 15, 15, 15, 15, 15, 15 });

            var result = np.clip(sliced, min_arr, max_arr);

            result.Should().BeOfValues(3, 3, 4, 6, 8, 10, 12, 14, 15, 15);
        }

        #endregion
    }
}
