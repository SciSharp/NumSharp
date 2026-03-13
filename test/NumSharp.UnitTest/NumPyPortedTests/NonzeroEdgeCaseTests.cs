using System;
using System.Threading.Tasks;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;

namespace NumSharp.UnitTest.NumPyPortedTests
{
    /// <summary>
    /// Tests ported from NumPy test_numeric.py TestNonzero class.
    /// Covers edge cases for np.nonzero operation.
    /// </summary>
    public class NonzeroEdgeCaseTests
    {
        #region Trivial Cases (from test_nonzero_trivial)

        [Test]
        public async Task Nonzero_EmptyArray_ReturnsEmptyTuple()
        {
            // NumPy: nonzero([]) = (array([], dtype=int64),)
            var result = np.nonzero(np.array(new double[0]));

            await Assert.That(result.Length).IsEqualTo(1);
            await Assert.That(result[0].size).IsEqualTo(0);
        }

        [Test]
        public async Task Nonzero_SingleZero_ReturnsEmpty()
        {
            // NumPy: nonzero([0]) = (array([], dtype=int64),)
            var result = np.nonzero(np.array(new int[] { 0 }));

            await Assert.That(result.Length).IsEqualTo(1);
            await Assert.That(result[0].size).IsEqualTo(0);
        }

        [Test]
        public async Task Nonzero_SingleOne_ReturnsIndexZero()
        {
            // NumPy: nonzero([1]) = (array([0]),)
            var result = np.nonzero(np.array(new int[] { 1 }));

            await Assert.That(result.Length).IsEqualTo(1);
            result[0].Should().BeOfValues(0);
        }

        #endregion

        #region 1D Array Tests (from test_nonzero_onedim)

        [Test]
        public async Task Nonzero_1DArray_MixedValues()
        {
            // NumPy: nonzero([1,0,2,-1,0,0,8]) = (array([0, 2, 3, 6]),)
            var x = np.array(new int[] { 1, 0, 2, -1, 0, 0, 8 });
            var result = np.nonzero(x);

            await Assert.That(result.Length).IsEqualTo(1);
            result[0].Should().BeOfValues(0, 2, 3, 6);
        }

        [Test]
        public async Task Nonzero_1DArray_AllZeros()
        {
            var x = np.zeros(5, np.int32);
            var result = np.nonzero(x);

            await Assert.That(result.Length).IsEqualTo(1);
            await Assert.That(result[0].size).IsEqualTo(0);
        }

        [Test]
        public async Task Nonzero_1DArray_AllNonzero()
        {
            var x = np.array(new int[] { 1, 2, 3, 4, 5 });
            var result = np.nonzero(x);

            await Assert.That(result.Length).IsEqualTo(1);
            result[0].Should().BeOfValues(0, 1, 2, 3, 4);
        }

        #endregion

        #region 2D Array Tests (from test_nonzero_twodim)

        [Test]
        public async Task Nonzero_2DArray_SparseValues()
        {
            // NumPy: nonzero([[0,1,0],[2,0,3]]) = (array([0, 1, 1]), array([1, 0, 2]))
            var x = np.array(new int[,] { { 0, 1, 0 }, { 2, 0, 3 } });
            var result = np.nonzero(x);

            await Assert.That(result.Length).IsEqualTo(2);
            result[0].Should().BeOfValues(0, 1, 1);
            result[1].Should().BeOfValues(1, 0, 2);
        }

        [Test]
        public async Task Nonzero_EyeMatrix()
        {
            // NumPy: nonzero(eye(3)) = (array([0, 1, 2]), array([0, 1, 2]))
            var x = np.eye(3);
            var result = np.nonzero(x);

            await Assert.That(result.Length).IsEqualTo(2);
            result[0].Should().BeOfValues(0, 1, 2);
            result[1].Should().BeOfValues(0, 1, 2);
        }

        [Test]
        public async Task Nonzero_2DArray_Arange()
        {
            // NumPy test Case2: arange(9).reshape(3,3) - 0 is only nonzero at position 0
            var x = np.arange(9).reshape(3, 3);
            var result = np.nonzero(x);

            await Assert.That(result.Length).IsEqualTo(2);
            // Indices for non-zero elements (1-8)
            result[0].Should().BeOfValues(0, 0, 1, 1, 1, 2, 2, 2);
            result[1].Should().BeOfValues(1, 2, 0, 1, 2, 0, 1, 2);
        }

        #endregion

        #region 3D Array Tests

        [Test]
        public async Task Nonzero_3DArray()
        {
            // NumPy: nonzero([[[0,1],[0,0]],[[1,0],[0,1]]]) =
            // (array([0, 1, 1]), array([0, 0, 1]), array([1, 0, 1]))
            var x = np.array(new int[,,] { { { 0, 1 }, { 0, 0 } }, { { 1, 0 }, { 0, 1 } } });
            var result = np.nonzero(x);

            await Assert.That(result.Length).IsEqualTo(3);
            result[0].Should().BeOfValues(0, 1, 1);
            result[1].Should().BeOfValues(0, 0, 1);
            result[2].Should().BeOfValues(1, 0, 1);
        }

        #endregion

        #region Boolean Array Tests

        [Test]
        public async Task Nonzero_BooleanArray()
        {
            // NumPy: nonzero([True, False, True, False, True]) = (array([0, 2, 4]),)
            var x = np.array(new bool[] { true, false, true, false, true });
            var result = np.nonzero(x);

            await Assert.That(result.Length).IsEqualTo(1);
            result[0].Should().BeOfValues(0, 2, 4);
        }

        [Test]
        public async Task Nonzero_AllTrue()
        {
            var x = np.ones(5, np.@bool);
            var result = np.nonzero(x);

            await Assert.That(result.Length).IsEqualTo(1);
            result[0].Should().BeOfValues(0, 1, 2, 3, 4);
        }

        [Test]
        public async Task Nonzero_AllFalse()
        {
            var x = np.zeros(5, np.@bool);
            var result = np.nonzero(x);

            await Assert.That(result.Length).IsEqualTo(1);
            await Assert.That(result[0].size).IsEqualTo(0);
        }

        #endregion

        #region Float Array Tests (from test_nonzero_float_dtypes)

        [Test]
        public async Task Nonzero_Float32Array()
        {
            var x = np.array(new float[] { 0f, 1.5f, 0f, -2.3f, 0f });
            var result = np.nonzero(x);

            await Assert.That(result.Length).IsEqualTo(1);
            result[0].Should().BeOfValues(1, 3);
        }

        [Test]
        public async Task Nonzero_Float64Array()
        {
            var x = np.array(new double[] { 0.0, 1.5, 0.0, -2.3, 0.0 });
            var result = np.nonzero(x);

            await Assert.That(result.Length).IsEqualTo(1);
            result[0].Should().BeOfValues(1, 3);
        }

        [Test]
        public async Task Nonzero_NegativeZero_TreatedAsZero()
        {
            // NumPy: nonzero([0.0, -0.0, 1.0]) = (array([2]),)
            var x = np.array(new double[] { 0.0, -0.0, 1.0 });
            var result = np.nonzero(x);

            await Assert.That(result.Length).IsEqualTo(1);
            result[0].Should().BeOfValues(2);
        }

        #endregion

        #region Integer Dtype Tests (from test_nonzero_integer_dtypes)

        [Test]
        public async Task Nonzero_Int16Array()
        {
            var x = np.array(new short[] { 0, 1, 0, -1, 2 });
            var result = np.nonzero(x);

            result[0].Should().BeOfValues(1, 3, 4);
        }

        [Test]
        public async Task Nonzero_Int32Array()
        {
            var x = np.array(new int[] { 0, 1, 0, -1, 2 });
            var result = np.nonzero(x);

            result[0].Should().BeOfValues(1, 3, 4);
        }

        [Test]
        public async Task Nonzero_Int64Array()
        {
            var x = np.array(new long[] { 0, 1, 0, -1, 2 });
            var result = np.nonzero(x);

            result[0].Should().BeOfValues(1, 3, 4);
        }

        [Test]
        public async Task Nonzero_UInt8Array()
        {
            var x = np.array(new byte[] { 0, 1, 0, 5, 2 });
            var result = np.nonzero(x);

            result[0].Should().BeOfValues(1, 3, 4);
        }

        [Test]
        public async Task Nonzero_UInt16Array()
        {
            var x = np.array(new ushort[] { 0, 1, 0, 5, 2 });
            var result = np.nonzero(x);

            result[0].Should().BeOfValues(1, 3, 4);
        }

        [Test]
        public async Task Nonzero_UInt32Array()
        {
            var x = np.array(new uint[] { 0, 1, 0, 5, 2 });
            var result = np.nonzero(x);

            result[0].Should().BeOfValues(1, 3, 4);
        }

        [Test]
        public async Task Nonzero_UInt64Array()
        {
            var x = np.array(new ulong[] { 0, 1, 0, 5, 2 });
            var result = np.nonzero(x);

            result[0].Should().BeOfValues(1, 3, 4);
        }

        #endregion

        #region Transposed Array Tests

        [Test]
        public async Task Nonzero_TransposedArray()
        {
            // NumPy: nonzero([[0,1,0],[2,0,3]].T) = (array([0, 1, 2]), array([1, 0, 1]))
            var x = np.array(new int[,] { { 0, 1, 0 }, { 2, 0, 3 } }).T;
            var result = np.nonzero(x);

            await Assert.That(result.Length).IsEqualTo(2);
            result[0].Should().BeOfValues(0, 1, 2);
            result[1].Should().BeOfValues(1, 0, 1);
        }

        #endregion

        #region Using Result for Indexing

        [Test]
        public async Task Nonzero_UseResultForIndexing()
        {
            // NumPy: x[nonzero(x)] returns the non-zero values
            var x = np.array(new int[,] { { 3, 0, 0 }, { 0, 4, 0 }, { 5, 6, 0 } });
            var indices = np.nonzero(x);
            var values = x[indices];

            values.Should().BeOfValues(3, 4, 5, 6);
        }

        #endregion

        #region Sparse Pattern Tests (from test_sparse)

        [Test]
        public async Task Nonzero_SparsePattern_SingleTruePerBlock()
        {
            // Test sparse boolean pattern
            var c = np.zeros(200, np.@bool);
            for (int i = 0; i < 200; i += 20)
            {
                c[i.ToString()] = true;
            }

            var result = np.nonzero(c);
            result[0].Should().BeOfValues(0, 20, 40, 60, 80, 100, 120, 140, 160, 180);
        }

        #endregion

        #region Large Array Tests

        [Test]
        public async Task Nonzero_LargeArray()
        {
            // Create array with known non-zero pattern
            var x = np.zeros(1000, np.int32);
            x["100"] = 1;
            x["500"] = 2;
            x["999"] = 3;

            var result = np.nonzero(x);
            result[0].Should().BeOfValues(100, 500, 999);
        }

        #endregion

        #region Edge Cases with Slices

        [Test]
        public async Task Nonzero_SlicedArray()
        {
            var x = np.arange(10);
            var sliced = x["2:8"];  // [2, 3, 4, 5, 6, 7]
            var result = np.nonzero(sliced);

            // All values are non-zero
            result[0].Should().BeOfValues(0, 1, 2, 3, 4, 5);
        }

        [Test]
        public async Task Nonzero_StridedSlice()
        {
            var x = np.array(new int[] { 0, 1, 0, 2, 0, 3, 0, 4 });
            var strided = x["::" + 2];  // [0, 0, 0, 0]
            var result = np.nonzero(strided);

            await Assert.That(result[0].size).IsEqualTo(0);
        }

        #endregion

        #region NaN Handling

        [Test]
        public async Task Nonzero_NaN_TreatedAsNonzero()
        {
            // NaN is non-zero (it's not equal to zero)
            var x = np.array(new double[] { 0.0, double.NaN, 0.0, 1.0 });
            var result = np.nonzero(x);

            result[0].Should().BeOfValues(1, 3);
        }

        [Test]
        public async Task Nonzero_Inf_TreatedAsNonzero()
        {
            var x = np.array(new double[] { 0.0, double.PositiveInfinity, 0.0, double.NegativeInfinity });
            var result = np.nonzero(x);

            result[0].Should().BeOfValues(1, 3);
        }

        #endregion
    }
}
