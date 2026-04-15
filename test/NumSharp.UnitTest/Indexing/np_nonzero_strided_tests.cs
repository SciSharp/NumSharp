using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Indexing
{
    /// <summary>
    /// Tests for np.nonzero with strided (non-contiguous) arrays.
    /// Based on NumPy 2.4.2 behavior.
    /// </summary>
    [TestClass]
    public class np_nonzero_strided_tests : TestClass
    {
        [TestMethod]
        public void Transposed_2D()
        {
            // NumPy:
            // a = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
            // np.nonzero(a.T) -> (array([0, 1, 2]), array([0, 1, 2]))
            var a = np.array(new[,] { { 1, 0, 0 }, { 0, 2, 0 }, { 0, 0, 3 } });
            var aT = a.T;

            Assert.IsFalse(aT.Shape.IsContiguous, "Transposed array should be non-contiguous");

            var result = np.nonzero(aT);

            Assert.AreEqual(2, result.Length);
            result[0].Should().BeOfValues(0, 1, 2);  // row indices
            result[1].Should().BeOfValues(0, 1, 2);  // col indices
        }

        [TestMethod]
        public void Strided_1D_EveryOther()
        {
            // NumPy:
            // arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            // strided = arr[::2]  -> [1, 3, 5, 7, 9]
            // np.nonzero(strided) -> (array([0, 1, 2, 3, 4]),)
            var arr = np.arange(1, 11);  // [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            var strided = arr["::2"];    // [1, 3, 5, 7, 9]

            Assert.IsFalse(strided.Shape.IsContiguous, "Strided array should be non-contiguous");
            strided.Should().BeOfValues(1, 3, 5, 7, 9);

            var result = np.nonzero(strided);

            Assert.AreEqual(1, result.Length);
            result[0].Should().BeOfValues(0, 1, 2, 3, 4);  // all elements are non-zero
        }

        [TestMethod]
        public void Strided_1D_WithZeros()
        {
            // NumPy:
            // arr = np.array([0, 1, 0, 2, 0, 3, 0, 4, 0, 5])
            // strided = arr[::2]  -> [0, 0, 0, 0, 0]
            // np.nonzero(strided) -> (array([], dtype=int64),)
            var arr = np.array(new[] { 0, 1, 0, 2, 0, 3, 0, 4, 0, 5 });
            var strided = arr["::2"];  // [0, 0, 0, 0, 0]

            Assert.IsFalse(strided.Shape.IsContiguous);

            var result = np.nonzero(strided);

            Assert.AreEqual(1, result.Length);
            Assert.AreEqual(0, result[0].size);  // all zeros
        }

        [TestMethod]
        public void Strided_1D_Reversed()
        {
            // NumPy:
            // arr = np.array([0, 1, 0, 2, 0, 3])
            // reversed_arr = arr[::-1]  -> [3, 0, 2, 0, 1, 0]
            // np.nonzero(reversed_arr) -> (array([0, 2, 4]),)
            var arr = np.array(new[] { 0, 1, 0, 2, 0, 3 });
            var reversed = arr["::-1"];

            Assert.IsFalse(reversed.Shape.IsContiguous, "Reversed array should be non-contiguous");
            reversed.Should().BeOfValues(3, 0, 2, 0, 1, 0);

            var result = np.nonzero(reversed);

            Assert.AreEqual(1, result.Length);
            result[0].Should().BeOfValues(0, 2, 4);  // indices 0, 2, 4 have values 3, 2, 1
        }

        [TestMethod]
        public void Sliced_2D_NonContiguous()
        {
            // NumPy:
            // arr2d = np.array([[1, 0, 3], [0, 5, 0], [7, 0, 9]])
            // sliced = arr2d[::2, ::2]  -> [[1, 3], [7, 9]]
            // np.nonzero(sliced) -> (array([0, 0, 1, 1]), array([0, 1, 0, 1]))
            var arr2d = np.array(new[,] { { 1, 0, 3 }, { 0, 5, 0 }, { 7, 0, 9 } });
            var sliced = arr2d["::2, ::2"];

            Assert.IsFalse(sliced.Shape.IsContiguous, "2D sliced array should be non-contiguous");

            var result = np.nonzero(sliced);

            Assert.AreEqual(2, result.Length);
            result[0].Should().BeOfValues(0, 0, 1, 1);  // rows
            result[1].Should().BeOfValues(0, 1, 0, 1);  // cols
        }

        [TestMethod]
        public void Transposed_3D()
        {
            // Test transpose of 3D array
            // NumPy: a = np.zeros((2, 3, 4)); a[0,1,2] = 1; a[1,0,3] = 2
            // np.nonzero(a.T) gives indices in transposed order
            var arr3d = np.zeros(new Shape(2, 3, 4), NPTypeCode.Int32);
            arr3d.SetInt32(1, 0, 1, 2);
            arr3d.SetInt32(2, 1, 0, 3);

            var arrT = np.transpose(arr3d);  // Shape (4, 3, 2)

            Assert.IsFalse(arrT.Shape.IsContiguous, "Transposed 3D array should be non-contiguous");

            var result = np.nonzero(arrT);

            Assert.AreEqual(3, result.Length);
            // Original positions: (0,1,2) and (1,0,3)
            // Transposed positions: (2,1,0) and (3,0,1)
            result[0].Should().BeOfValues(2, 3);  // dim 0 (was dim 2)
            result[1].Should().BeOfValues(1, 0);  // dim 1 (same)
            result[2].Should().BeOfValues(0, 1);  // dim 2 (was dim 0)
        }

        [TestMethod]
        public void Boolean_Strided()
        {
            // NumPy:
            // arr = np.array([False, True, False, True, False, True])
            // strided = arr[::2]  -> [False, False, False]
            // np.nonzero(strided) -> (array([], dtype=int64),)
            var arr = np.array(new[] { false, true, false, true, false, true });
            var strided = arr["::2"];

            Assert.IsFalse(strided.Shape.IsContiguous);

            var result = np.nonzero(strided);

            Assert.AreEqual(1, result.Length);
            Assert.AreEqual(0, result[0].size);  // all false at even indices
        }

        [TestMethod]
        public void Boolean_Strided_Odd()
        {
            // NumPy:
            // arr = np.array([False, True, False, True, False, True])
            // strided = arr[1::2]  -> [True, True, True]
            // np.nonzero(strided) -> (array([0, 1, 2]),)
            var arr = np.array(new[] { false, true, false, true, false, true });
            var strided = arr["1::2"];

            Assert.IsFalse(strided.Shape.IsContiguous);
            strided.Should().BeOfValues(true, true, true);

            var result = np.nonzero(strided);

            Assert.AreEqual(1, result.Length);
            result[0].Should().BeOfValues(0, 1, 2);  // all true
        }

        [TestMethod]
        public void Float_Strided_WithNaN()
        {
            // NaN should be treated as non-zero even in strided arrays
            var arr = np.array(new[] { 0.0, double.NaN, 0.0, 1.0, 0.0, double.NaN });
            var strided = arr["1::2"];  // [NaN, 1.0, NaN]

            Assert.IsFalse(strided.Shape.IsContiguous);

            var result = np.nonzero(strided);

            Assert.AreEqual(1, result.Length);
            result[0].Should().BeOfValues(0, 1, 2);  // all non-zero (NaN is non-zero)
        }

        [TestMethod]
        public void Column_Slice()
        {
            // Test extracting a column (non-contiguous in row-major)
            // NumPy:
            // a = np.array([[1, 0], [0, 2], [3, 0]])
            // col0 = a[:, 0]  -> [1, 0, 3]
            // np.nonzero(col0) -> (array([0, 2]),)
            var a = np.array(new[,] { { 1, 0 }, { 0, 2 }, { 3, 0 } });
            var col0 = a[":, 0"];

            Assert.IsFalse(col0.Shape.IsContiguous, "Column slice should be non-contiguous");
            col0.Should().BeOfValues(1, 0, 3);

            var result = np.nonzero(col0);

            Assert.AreEqual(1, result.Length);
            result[0].Should().BeOfValues(0, 2);  // indices 0 and 2 are non-zero
        }

        [TestMethod]
        public void Row_Slice_Contiguous()
        {
            // Test extracting a row (should be contiguous in row-major)
            // This tests that the contiguous fast path works after slicing
            var a = np.array(new[,] { { 1, 0, 3 }, { 0, 2, 0 }, { 4, 0, 5 } });
            var row0 = a["0, :"];

            // Row slice in row-major should be contiguous
            Assert.IsTrue(row0.Shape.IsContiguous, "Row slice should be contiguous in C-order");
            row0.Should().BeOfValues(1, 0, 3);

            var result = np.nonzero(row0);

            Assert.AreEqual(1, result.Length);
            result[0].Should().BeOfValues(0, 2);  // indices 0 and 2 are non-zero
        }
    }
}
