using System;
using System.Threading.Tasks;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;

namespace NumSharp.UnitTest.NumPyPortedTests
{
    /// <summary>
    /// Tests ported from NumPy for np.argmax and np.argmin.
    /// Covers edge cases including NaN handling, axis combinations, and keepdims.
    /// </summary>
    public class ArgMaxArgMinEdgeCaseTests
    {
        #region Basic ArgMax Tests

        [Test]
        public async Task ArgMax_1DArray()
        {
            // NumPy: argmax([1,3,2,4,0]) = 3
            var a = np.array(new int[] { 1, 3, 2, 4, 0 });
            var result = np.argmax(a);
            await Assert.That((int)result).IsEqualTo(3);
        }

        [Test]
        public async Task ArgMin_1DArray()
        {
            // NumPy: argmin([1,3,2,4,0]) = 4
            var a = np.array(new int[] { 1, 3, 2, 4, 0 });
            var result = np.argmin(a);
            await Assert.That((int)result).IsEqualTo(4);
        }

        #endregion

        #region 2D Array Tests

        [Test]
        public async Task ArgMax_2DArray_Flattened()
        {
            // NumPy: argmax([[1,5,3],[4,2,6]]) = 5 (flattened index)
            var a = np.array(new int[,] { { 1, 5, 3 }, { 4, 2, 6 } });
            var result = np.argmax(a);
            await Assert.That((int)result).IsEqualTo(5);
        }

        [Test]
        public async Task ArgMax_2DArray_Axis0()
        {
            // NumPy: argmax([[1,5,3],[4,2,6]], axis=0) = [1, 0, 1]
            var a = np.array(new int[,] { { 1, 5, 3 }, { 4, 2, 6 } });
            var result = np.argmax(a, axis: 0);
            result.Should().BeShaped(3);
            result.Should().BeOfValues(1, 0, 1);
        }

        [Test]
        public async Task ArgMax_2DArray_Axis1()
        {
            // NumPy: argmax([[1,5,3],[4,2,6]], axis=1) = [1, 2]
            var a = np.array(new int[,] { { 1, 5, 3 }, { 4, 2, 6 } });
            var result = np.argmax(a, axis: 1);
            result.Should().BeShaped(2);
            result.Should().BeOfValues(1, 2);
        }

        [Test]
        public async Task ArgMin_2DArray_Axis0()
        {
            // NumPy: argmin([[1,5,3],[4,2,6]], axis=0) = [0, 1, 0]
            var a = np.array(new int[,] { { 1, 5, 3 }, { 4, 2, 6 } });
            var result = np.argmin(a, axis: 0);
            result.Should().BeShaped(3);
            result.Should().BeOfValues(0, 1, 0);
        }

        [Test]
        public async Task ArgMin_2DArray_Axis1()
        {
            // NumPy: argmin([[1,5,3],[4,2,6]], axis=1) = [0, 1]
            var a = np.array(new int[,] { { 1, 5, 3 }, { 4, 2, 6 } });
            var result = np.argmin(a, axis: 1);
            result.Should().BeShaped(2);
            result.Should().BeOfValues(0, 1);
        }

        #endregion

        #region keepdims Tests

        [Test]
        public async Task ArgMax_Keepdims_True()
        {
            // NumPy: argmax([[1,5,3],[4,2,6]], axis=1, keepdims=True) = [[1],[2]]
            var a = np.array(new int[,] { { 1, 5, 3 }, { 4, 2, 6 } });
            var result = np.argmax(a, axis: 1, keepdims: true);
            result.Should().BeShaped(2, 1);
            result.Should().BeOfValues(1, 2);
        }

        [Test]
        public async Task ArgMax_Keepdims_False()
        {
            var a = np.array(new int[,] { { 1, 5, 3 }, { 4, 2, 6 } });
            var result = np.argmax(a, axis: 1, keepdims: false);
            result.Should().BeShaped(2);
        }

        [Test]
        public async Task ArgMin_Keepdims_True()
        {
            var a = np.array(new int[,] { { 1, 5, 3 }, { 4, 2, 6 } });
            var result = np.argmin(a, axis: 1, keepdims: true);
            result.Should().BeShaped(2, 1);
        }

        #endregion

        #region Negative Axis Tests

        [Test]
        public async Task ArgMax_NegativeAxis_Minus1()
        {
            var a = np.array(new int[,] { { 1, 5, 3 }, { 4, 2, 6 } });
            var result = np.argmax(a, axis: -1);
            result.Should().BeShaped(2);
            result.Should().BeOfValues(1, 2);
        }

        [Test]
        public async Task ArgMax_NegativeAxis_Minus2()
        {
            var a = np.array(new int[,] { { 1, 5, 3 }, { 4, 2, 6 } });
            var result = np.argmax(a, axis: -2);
            result.Should().BeShaped(3);
            result.Should().BeOfValues(1, 0, 1);
        }

        #endregion

        #region NaN Handling Tests

        [Test]
        public async Task ArgMax_FirstNaN_ReturnsNaNIndex()
        {
            // NumPy: argmax([1, nan, 3, 2]) = 1 (first NaN wins)
            var a = np.array(new double[] { 1.0, double.NaN, 3.0, 2.0 });
            var result = np.argmax(a);
            await Assert.That((int)result).IsEqualTo(1);
        }

        [Test]
        public async Task ArgMin_FirstNaN_ReturnsNaNIndex()
        {
            // NumPy: argmin([1, nan, 3, 2]) = 1 (first NaN wins)
            var a = np.array(new double[] { 1.0, double.NaN, 3.0, 2.0 });
            var result = np.argmin(a);
            await Assert.That((int)result).IsEqualTo(1);
        }

        [Test]
        public async Task ArgMax_NaNLater_ReturnsNaNIndex()
        {
            // NumPy: argmax([1, 3, nan, 2]) = 2 (NaN index)
            var a = np.array(new double[] { 1.0, 3.0, double.NaN, 2.0 });
            var result = np.argmax(a);
            await Assert.That((int)result).IsEqualTo(2);
        }

        [Test]
        public async Task ArgMax_AllNaN_ReturnsFirst()
        {
            // NumPy: argmax([nan, nan]) = 0
            var a = np.array(new double[] { double.NaN, double.NaN });
            var result = np.argmax(a);
            await Assert.That((int)result).IsEqualTo(0);
        }

        #endregion

        #region 3D Array Shape Tests

        [Test]
        public async Task ArgMax_3DArray_Axis0()
        {
            var a = np.arange(24).reshape(2, 3, 4);
            var result = np.argmax(a, axis: 0);
            result.Should().BeShaped(3, 4);
        }

        [Test]
        public async Task ArgMax_3DArray_Axis1()
        {
            var a = np.arange(24).reshape(2, 3, 4);
            var result = np.argmax(a, axis: 1);
            result.Should().BeShaped(2, 4);
        }

        [Test]
        public async Task ArgMax_3DArray_Axis2()
        {
            var a = np.arange(24).reshape(2, 3, 4);
            var result = np.argmax(a, axis: 2);
            result.Should().BeShaped(2, 3);
        }

        #endregion

        #region Tie-Breaking Tests (First Wins)

        [Test]
        public async Task ArgMax_TieBreaking_FirstWins()
        {
            // NumPy: argmax([3,1,3,2,3]) = 0 (first max wins)
            var a = np.array(new int[] { 3, 1, 3, 2, 3 });
            var result = np.argmax(a);
            await Assert.That((int)result).IsEqualTo(0);
        }

        [Test]
        public async Task ArgMin_TieBreaking_FirstWins()
        {
            // NumPy: argmin([1,3,1,2,1]) = 0 (first min wins)
            var a = np.array(new int[] { 1, 3, 1, 2, 1 });
            var result = np.argmin(a);
            await Assert.That((int)result).IsEqualTo(0);
        }

        #endregion

        #region Result Dtype Tests

        [Test]
        public async Task ArgMax_ResultDtype_IsInt()
        {
            var a = np.arange(1000);
            var result = np.argmax(a);
            await Assert.That(result.dtype == np.int32 || result.dtype == np.int64).IsTrue();
        }

        #endregion

        #region Single Element

        [Test]
        public async Task ArgMax_SingleElement()
        {
            var a = np.array(new int[] { 42 });
            var result = np.argmax(a);
            await Assert.That((int)result).IsEqualTo(0);
        }

        [Test]
        public async Task ArgMin_SingleElement()
        {
            var a = np.array(new int[] { 42 });
            var result = np.argmin(a);
            await Assert.That((int)result).IsEqualTo(0);
        }

        #endregion

        #region Negative Values

        [Test]
        public async Task ArgMax_WithNegativeValues()
        {
            var a = np.array(new int[] { -5, -2, -8, -1, -3 });
            var result = np.argmax(a);
            await Assert.That((int)result).IsEqualTo(3);  // -1 is the max
        }

        [Test]
        public async Task ArgMin_WithNegativeValues()
        {
            var a = np.array(new int[] { -5, -2, -8, -1, -3 });
            var result = np.argmin(a);
            await Assert.That((int)result).IsEqualTo(2);  // -8 is the min
        }

        #endregion

        #region Float Values

        [Test]
        public async Task ArgMax_FloatArray()
        {
            var a = np.array(new double[] { 1.1, 3.3, 2.2, 4.4, 0.0 });
            var result = np.argmax(a);
            await Assert.That((int)result).IsEqualTo(3);
        }

        [Test]
        public async Task ArgMin_FloatArray()
        {
            var a = np.array(new double[] { 1.1, 3.3, 2.2, 4.4, 0.0 });
            var result = np.argmin(a);
            await Assert.That((int)result).IsEqualTo(4);
        }

        #endregion

        #region Infinity Handling

        [Test]
        public async Task ArgMax_WithPositiveInf()
        {
            var a = np.array(new double[] { 1.0, double.PositiveInfinity, 3.0, 2.0 });
            var result = np.argmax(a);
            await Assert.That((int)result).IsEqualTo(1);
        }

        [Test]
        public async Task ArgMin_WithNegativeInf()
        {
            var a = np.array(new double[] { 1.0, double.NegativeInfinity, 3.0, 2.0 });
            var result = np.argmin(a);
            await Assert.That((int)result).IsEqualTo(1);
        }

        #endregion

        #region All Same Values

        [Test]
        public async Task ArgMax_AllSameValues_ReturnsFirst()
        {
            var a = np.full(5, 7);
            var result = np.argmax(a);
            await Assert.That((int)result).IsEqualTo(0);
        }

        [Test]
        public async Task ArgMin_AllSameValues_ReturnsFirst()
        {
            var a = np.full(5, 7);
            var result = np.argmin(a);
            await Assert.That((int)result).IsEqualTo(0);
        }

        #endregion

        #region Boolean Array

        [Test]
        public async Task ArgMax_BooleanArray()
        {
            var a = np.array(new bool[] { false, true, false, true, false });
            var result = np.argmax(a);
            await Assert.That((int)result).IsEqualTo(1);
        }

        [Test]
        public async Task ArgMin_BooleanArray()
        {
            var a = np.array(new bool[] { true, false, true, false, true });
            var result = np.argmin(a);
            await Assert.That((int)result).IsEqualTo(1);
        }

        #endregion

        #region Strided Array

        [Test]
        public async Task ArgMax_StridedArray()
        {
            var a = np.array(new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
            var strided = a["::" + 2];  // [0, 2, 4, 6, 8]
            var result = np.argmax(strided);
            await Assert.That((int)result).IsEqualTo(4);
        }

        #endregion

        #region Large Array

        [Test]
        public async Task ArgMax_LargeArray()
        {
            var a = np.arange(10000);
            var result = np.argmax(a);
            await Assert.That((int)result).IsEqualTo(9999);
        }

        [Test]
        public async Task ArgMin_LargeArray()
        {
            var a = np.arange(10000);
            var result = np.argmin(a);
            await Assert.That((int)result).IsEqualTo(0);
        }

        #endregion

        #region Multiple Dtypes

        [Test]
        public async Task ArgMax_Int32()
        {
            var a = np.array(new int[] { 1, 3, 2 });
            var result = np.argmax(a);
            await Assert.That((int)result).IsEqualTo(1);
        }

        [Test]
        public async Task ArgMax_Int64()
        {
            var a = np.array(new long[] { 1, 3, 2 });
            var result = np.argmax(a);
            await Assert.That((int)result).IsEqualTo(1);
        }

        [Test]
        public async Task ArgMax_Float32()
        {
            var a = np.array(new float[] { 1f, 3f, 2f });
            var result = np.argmax(a);
            await Assert.That((int)result).IsEqualTo(1);
        }

        [Test]
        public async Task ArgMax_Byte()
        {
            var a = np.array(new byte[] { 1, 3, 2 });
            var result = np.argmax(a);
            await Assert.That((int)result).IsEqualTo(1);
        }

        #endregion
    }
}
