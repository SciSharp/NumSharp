using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Backends.Kernels;

/// <summary>
/// Edge case tests for axis reduction operations.
/// Covers scenarios from Task #118: Battle-test axis reduction SIMD kernels.
/// </summary>
[TestClass]
public class AxisReductionEdgeCaseTests
{
    #region Basic axis parameter tests

    [TestMethod]
    public void Sum_Axis0_2D_MatchesNumPy()
    {
        // NumPy: np.sum([[1,2,3],[4,5,6]], axis=0) = [5, 7, 9]
        var arr = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        var result = np.sum(arr, axis: 0);

        result.Should().BeShaped(3);
        Assert.AreEqual(5L, result.GetInt64(0));
        Assert.AreEqual(7L, result.GetInt64(1));
        Assert.AreEqual(9L, result.GetInt64(2));
    }

    [TestMethod]
    public void Sum_Axis1_2D_MatchesNumPy()
    {
        // NumPy: np.sum([[1,2,3],[4,5,6]], axis=1) = [6, 15]
        var arr = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        var result = np.sum(arr, axis: 1);

        result.Should().BeShaped(2);
        Assert.AreEqual(6L, result.GetInt64(0));
        Assert.AreEqual(15L, result.GetInt64(1));
    }

    [TestMethod]
    public void Sum_AxisNeg1_2D_MatchesNumPy()
    {
        // NumPy: np.sum([[1,2,3],[4,5,6]], axis=-1) = [6, 15]
        var arr = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        var result = np.sum(arr, axis: -1);

        result.Should().BeShaped(2);
        Assert.AreEqual(6L, result.GetInt64(0));
        Assert.AreEqual(15L, result.GetInt64(1));
    }

    #endregion

    #region keepdims=True tests

    [TestMethod]
    public void Sum_Axis0_Keepdims_2D()
    {
        // NumPy: np.sum([[1,2,3],[4,5,6]], axis=0, keepdims=True) = [[5, 7, 9]]
        var arr = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        var result = np.sum(arr, axis: 0, keepdims: true);

        result.Should().BeShaped(1, 3);
        Assert.AreEqual(5L, result.GetInt64(0, 0));
        Assert.AreEqual(7L, result.GetInt64(0, 1));
        Assert.AreEqual(9L, result.GetInt64(0, 2));
    }

    [TestMethod]
    public void Sum_Axis1_Keepdims_2D()
    {
        // NumPy: np.sum([[1,2,3],[4,5,6]], axis=1, keepdims=True) = [[6], [15]]
        var arr = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        var result = np.sum(arr, axis: 1, keepdims: true);

        result.Should().BeShaped(2, 1);
        Assert.AreEqual(6L, result.GetInt64(0, 0));
        Assert.AreEqual(15L, result.GetInt64(1, 0));
    }

    [TestMethod]
    public void Max_Axis0_Keepdims_3D()
    {
        // NumPy: np.max(np.arange(24).reshape(2,3,4), axis=0, keepdims=True).shape = (1, 3, 4)
        var arr = np.arange(24).reshape(2, 3, 4);
        var result = np.amax(arr, axis: 0, keepdims: true);

        result.Should().BeShaped(1, 3, 4);
    }

    [TestMethod]
    public void Min_Axis1_Keepdims_3D()
    {
        // NumPy: np.min(np.arange(24).reshape(2,3,4), axis=1, keepdims=True).shape = (2, 1, 4)
        var arr = np.arange(24).reshape(2, 3, 4);
        var result = np.amin(arr, axis: 1, keepdims: true);

        result.Should().BeShaped(2, 1, 4);
    }

    [TestMethod]
    public void Prod_Axis2_Keepdims_3D()
    {
        // NumPy: np.prod(np.arange(1,25).reshape(2,3,4), axis=2, keepdims=True).shape = (2, 3, 1)
        var arr = np.arange(1, 25).reshape(2, 3, 4);
        var result = np.prod(arr, axis: 2, keepdims: true);

        result.Should().BeShaped(2, 3, 1);
    }

    #endregion

    #region Negative axis tests

    [TestMethod]
    public void Sum_AxisNeg2_3D()
    {
        // NumPy: np.sum(np.arange(24).reshape(2,3,4), axis=-2).shape = (2, 4)
        var arr = np.arange(24).reshape(2, 3, 4);
        var result = np.sum(arr, axis: -2);

        result.Should().BeShaped(2, 4);
    }

    [TestMethod]
    public void Max_AxisNeg1_3D()
    {
        // NumPy: np.max(np.arange(24).reshape(2,3,4), axis=-1).shape = (2, 3)
        var arr = np.arange(24).reshape(2, 3, 4);
        var result = np.amax(arr, axis: -1);

        result.Should().BeShaped(2, 3);
    }

    [TestMethod]
    public void Min_AxisNeg3_3D()
    {
        // NumPy: np.min(np.arange(24).reshape(2,3,4), axis=-3).shape = (3, 4)
        // axis=-3 is axis=0 for 3D array
        var arr = np.arange(24).reshape(2, 3, 4);
        var result = np.amin(arr, axis: -3);

        result.Should().BeShaped(3, 4);
    }

    #endregion

    #region Empty array along axis tests

    [TestMethod]
    public void Sum_EmptyAlongAxis0()
    {
        // NumPy: np.sum(np.zeros((0, 3)), axis=0) = [0., 0., 0.]
        // NumSharp BUG: Returns scalar 0.0 instead of array of shape (3,)
        var arr = np.zeros(new int[] { 0, 3 });
        var result = np.sum(arr, axis: 0);

        result.Should().BeShaped(3);
        Assert.AreEqual(0.0, result.GetDouble(0));
        Assert.AreEqual(0.0, result.GetDouble(1));
        Assert.AreEqual(0.0, result.GetDouble(2));
    }

    [TestMethod]
    public void Sum_EmptyAlongAxis1()
    {
        // NumPy: np.sum(np.zeros((2, 0)), axis=1) = [0., 0.]
        // NumSharp BUG: Returns scalar 0.0 instead of array of shape (2,)
        var arr = np.zeros(new int[] { 2, 0 });
        var result = np.sum(arr, axis: 1);

        result.Should().BeShaped(2);
        Assert.AreEqual(0.0, result.GetDouble(0));
        Assert.AreEqual(0.0, result.GetDouble(1));
    }

    [TestMethod]
    public void Prod_EmptyAlongAxis()
    {
        // NumPy: np.prod(np.ones((0, 3)), axis=0) = [1., 1., 1.]
        // NumSharp BUG: Returns scalar 1.0 instead of array of shape (3,)
        var arr = np.ones(new int[] { 0, 3 });
        var result = np.prod(arr, axis: 0);

        result.Should().BeShaped(3);
        Assert.AreEqual(1.0, result.GetDouble(0));
        Assert.AreEqual(1.0, result.GetDouble(1));
        Assert.AreEqual(1.0, result.GetDouble(2));
    }

    #endregion

    #region NaN handling in reductions

    [TestMethod]
    public void Sum_WithNaN_Axis0_PropagatesNaN()
    {
        // NumPy: np.sum([[1., np.nan], [3., 4.]], axis=0) = [4., nan]
        var arr = np.array(new double[,] { { 1.0, double.NaN }, { 3.0, 4.0 } });
        var result = np.sum(arr, axis: 0);

        result.Should().BeShaped(2);
        Assert.AreEqual(4.0, result.GetDouble(0));
        Assert.IsTrue(double.IsNaN(result.GetDouble(1)));
    }

    [TestMethod]
    public void Sum_WithNaN_Axis1_PropagatesNaN()
    {
        // NumPy: np.sum([[1., 2.], [np.nan, 4.]], axis=1) = [3., nan]
        var arr = np.array(new double[,] { { 1.0, 2.0 }, { double.NaN, 4.0 } });
        var result = np.sum(arr, axis: 1);

        result.Should().BeShaped(2);
        Assert.AreEqual(3.0, result.GetDouble(0));
        Assert.IsTrue(double.IsNaN(result.GetDouble(1)));
    }

    [TestMethod]
    public void Max_WithNaN_PropagatesNaN()
    {
        // NumPy: np.max([[1., np.nan], [3., 4.]], axis=0) = [3., nan]
        var arr = np.array(new double[,] { { 1.0, double.NaN }, { 3.0, 4.0 } });
        var result = np.amax(arr, axis: 0);

        result.Should().BeShaped(2);
        Assert.AreEqual(3.0, result.GetDouble(0));
        Assert.IsTrue(double.IsNaN(result.GetDouble(1)));
    }

    [TestMethod]
    public void Min_WithNaN_PropagatesNaN()
    {
        // NumPy: np.min([[1., 2.], [np.nan, 4.]], axis=1) = [1., nan]
        var arr = np.array(new double[,] { { 1.0, 2.0 }, { double.NaN, 4.0 } });
        var result = np.amin(arr, axis: 1);

        result.Should().BeShaped(2);
        Assert.AreEqual(1.0, result.GetDouble(0));
        Assert.IsTrue(double.IsNaN(result.GetDouble(1)));
    }

    #endregion

    #region Broadcast array with axis reduction

    [TestMethod]
    public void Sum_BroadcastArray_Axis0()
    {
        // Create a broadcast array and reduce along axis
        var arr = np.array(new int[] { 1, 2, 3 });
        var broadcast = np.broadcast_to(arr, new Shape(4, 3));

        // Sum along axis 0: each column sums 4 copies of its value
        var result = np.sum(broadcast, axis: 0);

        result.Should().BeShaped(3);
        Assert.AreEqual(4L, result.GetInt64(0));  // 1+1+1+1 = 4
        Assert.AreEqual(8L, result.GetInt64(1));  // 2+2+2+2 = 8
        Assert.AreEqual(12L, result.GetInt64(2)); // 3+3+3+3 = 12
    }

    [TestMethod]
    public void Sum_BroadcastArray_Axis1()
    {
        // Create a broadcast array and reduce along axis
        var arr = np.array(new int[] { 1, 2, 3 });
        var broadcast = np.broadcast_to(arr, new Shape(2, 3));

        // Sum along axis 1: each row sums 1+2+3 = 6
        var result = np.sum(broadcast, axis: 1);

        result.Should().BeShaped(2);
        Assert.AreEqual(6L, result.GetInt64(0));
        Assert.AreEqual(6L, result.GetInt64(1));
    }

    #endregion

    #region prod with axis parameter

    [TestMethod]
    public void Prod_Axis0_2D()
    {
        // NumPy: np.prod([[1,2,3],[4,5,6]], axis=0) = [4, 10, 18]
        var arr = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        var result = np.prod(arr, axis: 0);

        result.Should().BeShaped(3);
        Assert.AreEqual(4L, result.GetInt64(0));
        Assert.AreEqual(10L, result.GetInt64(1));
        Assert.AreEqual(18L, result.GetInt64(2));
    }

    [TestMethod]
    public void Prod_Axis1_2D()
    {
        // NumPy: np.prod([[1,2,3],[4,5,6]], axis=1) = [6, 120]
        var arr = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        var result = np.prod(arr, axis: 1);

        result.Should().BeShaped(2);
        Assert.AreEqual(6L, result.GetInt64(0));
        Assert.AreEqual(120L, result.GetInt64(1));
    }

    #endregion

    #region min/max with axis parameter

    [TestMethod]
    public void Max_Axis0_2D()
    {
        // NumPy: np.max([[1,5,3],[4,2,6]], axis=0) = [4, 5, 6]
        var arr = np.array(new int[,] { { 1, 5, 3 }, { 4, 2, 6 } });
        var result = np.amax(arr, axis: 0);

        result.Should().BeShaped(3);
        Assert.AreEqual(4, result.GetInt32(0));
        Assert.AreEqual(5, result.GetInt32(1));
        Assert.AreEqual(6, result.GetInt32(2));
    }

    [TestMethod]
    public void Max_Axis1_2D()
    {
        // NumPy: np.max([[1,5,3],[4,2,6]], axis=1) = [5, 6]
        var arr = np.array(new int[,] { { 1, 5, 3 }, { 4, 2, 6 } });
        var result = np.amax(arr, axis: 1);

        result.Should().BeShaped(2);
        Assert.AreEqual(5, result.GetInt32(0));
        Assert.AreEqual(6, result.GetInt32(1));
    }

    [TestMethod]
    public void Min_Axis0_2D()
    {
        // NumPy: np.min([[1,5,3],[4,2,6]], axis=0) = [1, 2, 3]
        var arr = np.array(new int[,] { { 1, 5, 3 }, { 4, 2, 6 } });
        var result = np.amin(arr, axis: 0);

        result.Should().BeShaped(3);
        Assert.AreEqual(1, result.GetInt32(0));
        Assert.AreEqual(2, result.GetInt32(1));
        Assert.AreEqual(3, result.GetInt32(2));
    }

    [TestMethod]
    public void Min_Axis1_2D()
    {
        // NumPy: np.min([[1,5,3],[4,2,6]], axis=1) = [1, 2]
        var arr = np.array(new int[,] { { 1, 5, 3 }, { 4, 2, 6 } });
        var result = np.amin(arr, axis: 1);

        result.Should().BeShaped(2);
        Assert.AreEqual(1, result.GetInt32(0));
        Assert.AreEqual(2, result.GetInt32(1));
    }

    #endregion

    #region mean with axis parameter

    [TestMethod]
    public void Mean_Axis0_2D()
    {
        // NumPy: np.mean([[1,2,3],[4,5,6]], axis=0) = [2.5, 3.5, 4.5]
        var arr = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        var result = np.mean(arr, axis: 0);

        result.Should().BeShaped(3);
        Assert.AreEqual(2.5, result.GetDouble(0), 1e-10);
        Assert.AreEqual(3.5, result.GetDouble(1), 1e-10);
        Assert.AreEqual(4.5, result.GetDouble(2), 1e-10);
    }

    [TestMethod]
    public void Mean_Axis1_2D()
    {
        // NumPy: np.mean([[1,2,3],[4,5,6]], axis=1) = [2., 5.]
        var arr = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        var result = np.mean(arr, axis: 1);

        result.Should().BeShaped(2);
        Assert.AreEqual(2.0, result.GetDouble(0), 1e-10);
        Assert.AreEqual(5.0, result.GetDouble(1), 1e-10);
    }

    [TestMethod]
    public void Mean_Axis0_Keepdims()
    {
        // NumPy: np.mean([[1,2,3],[4,5,6]], axis=0, keepdims=True) = [[2.5, 3.5, 4.5]]
        var arr = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        var result = np.mean(arr, axis: 0, keepdims: true);

        result.Should().BeShaped(1, 3);
        Assert.AreEqual(2.5, result.GetDouble(0, 0), 1e-10);
        Assert.AreEqual(3.5, result.GetDouble(0, 1), 1e-10);
        Assert.AreEqual(4.5, result.GetDouble(0, 2), 1e-10);
    }

    #endregion

    #region argmax/argmin with axis parameter

    [TestMethod]
    public void Argmax_Axis0_2D()
    {
        // NumPy: np.argmax([[1,5,3],[4,2,6]], axis=0) = [1, 0, 1]
        var arr = np.array(new int[,] { { 1, 5, 3 }, { 4, 2, 6 } });
        var result = np.argmax(arr, axis: 0);

        result.Should().BeShaped(3);
        Assert.AreEqual(1L, result.GetInt64(0));
        Assert.AreEqual(0L, result.GetInt64(1));
        Assert.AreEqual(1L, result.GetInt64(2));
    }

    [TestMethod]
    public void Argmax_Axis1_2D()
    {
        // NumPy: np.argmax([[1,5,3],[4,2,6]], axis=1) = [1, 2]
        var arr = np.array(new int[,] { { 1, 5, 3 }, { 4, 2, 6 } });
        var result = np.argmax(arr, axis: 1);

        result.Should().BeShaped(2);
        Assert.AreEqual(1L, result.GetInt64(0));
        Assert.AreEqual(2L, result.GetInt64(1));
    }

    [TestMethod]
    public void Argmin_Axis0_2D()
    {
        // NumPy: np.argmin([[1,5,3],[4,2,6]], axis=0) = [0, 1, 0]
        var arr = np.array(new int[,] { { 1, 5, 3 }, { 4, 2, 6 } });
        var result = np.argmin(arr, axis: 0);

        result.Should().BeShaped(3);
        Assert.AreEqual(0L, result.GetInt64(0));
        Assert.AreEqual(1L, result.GetInt64(1));
        Assert.AreEqual(0L, result.GetInt64(2));
    }

    [TestMethod]
    public void Argmin_Axis1_2D()
    {
        // NumPy: np.argmin([[1,5,3],[4,2,6]], axis=1) = [0, 1]
        var arr = np.array(new int[,] { { 1, 5, 3 }, { 4, 2, 6 } });
        var result = np.argmin(arr, axis: 1);

        result.Should().BeShaped(2);
        Assert.AreEqual(0L, result.GetInt64(0));
        Assert.AreEqual(1L, result.GetInt64(1));
    }

    [TestMethod]
    public void Argmax_Axis0_Keepdims()
    {
        // NumPy: np.argmax([[1,5,3],[4,2,6]], axis=0, keepdims=True) = [[1, 0, 1]]
        var arr = np.array(new int[,] { { 1, 5, 3 }, { 4, 2, 6 } });
        var result = np.argmax(arr, axis: 0, keepdims: true);

        result.Should().BeShaped(1, 3);
        Assert.AreEqual(1L, result.GetInt64(0, 0));
        Assert.AreEqual(0L, result.GetInt64(0, 1));
        Assert.AreEqual(1L, result.GetInt64(0, 2));
    }

    [TestMethod]
    public void Argmax_WithNaN_Axis0()
    {
        // NumPy: np.argmax([[1., np.nan], [3., 4.]], axis=0) = [1, 0]
        // NaN at position [0,1] returns index 0 (first NaN wins)
        var arr = np.array(new double[,] { { 1.0, double.NaN }, { 3.0, 4.0 } });
        var result = np.argmax(arr, axis: 0);

        result.Should().BeShaped(2);
        Assert.AreEqual(1L, result.GetInt64(0)); // 3 > 1
        Assert.AreEqual(0L, result.GetInt64(1)); // NaN wins (first occurrence)
    }

    #endregion

    #region Single element axis tests

    [TestMethod]
    public void Sum_SingleRowMatrix_Axis0()
    {
        // NumPy: np.sum([[1, 2, 3]], axis=0) = [1, 2, 3]
        // NumSharp BUG: squeeze_fast shortcut causes "index < Count, Memory corruption expected"
        // The code takes a shortcut when shape[axis]==1 but returns the wrong result
        var arr = np.array(new int[,] { { 1, 2, 3 } });
        var result = np.sum(arr, axis: 0);

        result.Should().BeShaped(3);
        Assert.AreEqual(1L, result.GetInt64(0));
        Assert.AreEqual(2L, result.GetInt64(1));
        Assert.AreEqual(3L, result.GetInt64(2));
    }

    [TestMethod]
    public void Sum_SingleColumnMatrix_Axis1()
    {
        // NumPy: np.sum([[1], [2], [3]], axis=1) = [1, 2, 3]
        // NumSharp BUG: squeeze_fast shortcut causes "index < Count, Memory corruption expected"
        var arr = np.array(new int[,] { { 1 }, { 2 }, { 3 } });
        var result = np.sum(arr, axis: 1);

        result.Should().BeShaped(3);
        Assert.AreEqual(1L, result.GetInt64(0));
        Assert.AreEqual(2L, result.GetInt64(1));
        Assert.AreEqual(3L, result.GetInt64(2));
    }

    #endregion

    #region Large array tests (ensure SIMD path is used)

    [TestMethod]
    public void Sum_LargeArray_Axis0_CorrectResults()
    {
        // Create a large 2D array to trigger SIMD path
        int rows = 1000;
        int cols = 64;  // Multiple of Vector256 count
        var data = new double[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                data[i, j] = 1.0;  // All ones for easy verification

        var arr = np.array(data);
        var result = np.sum(arr, axis: 0);

        result.Should().BeShaped(cols);
        for (int j = 0; j < cols; j++)
        {
            Assert.AreEqual((double)rows, result.GetDouble(j), 1e-10);
        }
    }

    [TestMethod]
    public void Sum_LargeArray_Axis1_CorrectResults()
    {
        // Create a large 2D array to trigger SIMD path
        int rows = 64;
        int cols = 1000;  // Large inner dimension
        var data = new float[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                data[i, j] = 1.0f;

        var arr = np.array(data);
        var result = np.sum(arr, axis: 1);

        result.Should().BeShaped(rows);
        for (int i = 0; i < rows; i++)
        {
            Assert.AreEqual((float)cols, result.GetSingle(i), 1e-3f);
        }
    }

    [TestMethod]
    public void Max_LargeArray_Axis0()
    {
        int rows = 256;
        int cols = 64;
        var data = new int[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                data[i, j] = i; // Each row has value i

        var arr = np.array(data);
        var result = np.amax(arr, axis: 0);

        result.Should().BeShaped(cols);
        for (int j = 0; j < cols; j++)
        {
            Assert.AreEqual(rows - 1, result.GetInt32(j));
        }
    }

    [TestMethod]
    public void Min_LargeArray_Axis1()
    {
        int rows = 64;
        int cols = 256;
        var data = new int[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                data[i, j] = j; // Each column has value j

        var arr = np.array(data);
        var result = np.amin(arr, axis: 1);

        result.Should().BeShaped(rows);
        for (int i = 0; i < rows; i++)
        {
            Assert.AreEqual(0, result.GetInt32(i)); // Min is always 0 (first column)
        }
    }

    #endregion

    #region Sliced array tests

    [TestMethod]
    public void Sum_SlicedArray_Axis0()
    {
        // Create array and slice it, then reduce
        var arr = np.arange(24).reshape(4, 6);
        var sliced = arr["1:3", "1:5"]; // Get middle portion (2x4)

        var result = np.sum(sliced, axis: 0);

        result.Should().BeShaped(4);
        // Values at [1,1:5] = [7,8,9,10]
        // Values at [2,1:5] = [13,14,15,16]
        // Sum = [20, 22, 24, 26]
        Assert.AreEqual(20L, result.GetInt64(0));
        Assert.AreEqual(22L, result.GetInt64(1));
        Assert.AreEqual(24L, result.GetInt64(2));
        Assert.AreEqual(26L, result.GetInt64(3));
    }

    [TestMethod]
    public void Sum_SlicedArray_Axis1()
    {
        var arr = np.arange(24).reshape(4, 6);
        var sliced = arr["1:3", "1:5"]; // Get middle portion (2x4)

        var result = np.sum(sliced, axis: 1);

        result.Should().BeShaped(2);
        // Row 0 (original row 1): 7+8+9+10 = 34
        // Row 1 (original row 2): 13+14+15+16 = 58
        Assert.AreEqual(34L, result.GetInt64(0));
        Assert.AreEqual(58L, result.GetInt64(1));
    }

    #endregion
}
