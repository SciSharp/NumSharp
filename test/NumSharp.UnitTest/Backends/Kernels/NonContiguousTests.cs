using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Backends.Kernels;

/// <summary>
/// Comprehensive tests for operations on non-contiguous arrays.
/// Covers: transposed arrays, broadcast views, reversed slices, and multi-dimensional slices.
/// All expected values are verified against NumPy 2.4.2 output.
///
/// This complements SlicedArrayOpTests.cs which focuses on step-based slices.
/// </summary>
public class NonContiguousTests
{
    #region Transposed Array Operations

    [TestMethod]
    public void Transpose_2D_Sqrt()
    {
        // NumPy: np.sqrt(arr.T) where arr = [[1, 4], [9, 16], [25, 36]]
        // arr.T = [[1, 9, 25], [4, 16, 36]]
        // sqrt = [[1, 3, 5], [2, 4, 6]]
        var arr = np.array(new double[,] { { 1, 4 }, { 9, 16 }, { 25, 36 } });
        var transposed = arr.T;

        Assert.AreEqual(2, transposed.shape[0]);
        Assert.AreEqual(3, transposed.shape[1]);

        var result = np.sqrt(transposed);

        Assert.AreEqual(1.0, result.GetDouble(0, 0), 1e-10);
        Assert.AreEqual(3.0, result.GetDouble(0, 1), 1e-10);
        Assert.AreEqual(5.0, result.GetDouble(0, 2), 1e-10);
        Assert.AreEqual(2.0, result.GetDouble(1, 0), 1e-10);
        Assert.AreEqual(4.0, result.GetDouble(1, 1), 1e-10);
        Assert.AreEqual(6.0, result.GetDouble(1, 2), 1e-10);
    }

    [TestMethod]
    public void Transpose_2D_Add()
    {
        // NumPy: arr.T + arr.T where arr = [[1, 2], [3, 4]]
        // arr.T = [[1, 3], [2, 4]]
        // Result = [[2, 6], [4, 8]]
        var arr = np.array(new double[,] { { 1, 2 }, { 3, 4 } });
        var transposed = arr.T;

        var result = transposed + transposed;

        Assert.AreEqual(2.0, result.GetDouble(0, 0), 1e-10);
        Assert.AreEqual(6.0, result.GetDouble(0, 1), 1e-10);
        Assert.AreEqual(4.0, result.GetDouble(1, 0), 1e-10);
        Assert.AreEqual(8.0, result.GetDouble(1, 1), 1e-10);
    }

    [TestMethod]
    public void Transpose_2D_Sum()
    {
        // NumPy: np.sum(arr.T) where arr = [[1, 2, 3], [4, 5, 6]]
        // arr.T = [[1, 4], [2, 5], [3, 6]]
        // sum = 21
        var arr = np.array(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        var transposed = arr.T;

        var result = np.sum(transposed);

        Assert.AreEqual(21.0, result.GetDouble(0), 1e-10);
    }

    [TestMethod]
    public void Transpose_2D_Multiply()
    {
        // arr * arr.T (both 2x2)
        // arr = [[1, 2], [3, 4]], arr.T = [[1, 3], [2, 4]]
        // Element-wise: [[1, 6], [6, 16]]
        var arr = np.array(new double[,] { { 1, 2 }, { 3, 4 } });

        var result = arr * arr.T;

        Assert.AreEqual(1.0, result.GetDouble(0, 0), 1e-10);
        Assert.AreEqual(6.0, result.GetDouble(0, 1), 1e-10);
        Assert.AreEqual(6.0, result.GetDouble(1, 0), 1e-10);
        Assert.AreEqual(16.0, result.GetDouble(1, 1), 1e-10);
    }

    [TestMethod]
    public void Transpose_3D_Sum()
    {
        // NumPy: np.sum(arr.T) where arr has shape (2, 3, 4)
        var arr = np.arange(24).reshape(2, 3, 4);
        var transposed = np.transpose(arr);  // shape becomes (4, 3, 2)

        Assert.AreEqual(4, transposed.shape[0]);
        Assert.AreEqual(3, transposed.shape[1]);
        Assert.AreEqual(2, transposed.shape[2]);

        var result = np.sum(transposed);
        // Sum of 0..23 = 276
        Assert.AreEqual(276.0, (double)result, 1e-10);
    }

    #endregion

    #region Broadcast View Operations

    [TestMethod]
    public void BroadcastTo_Scalar_Sum()
    {
        // NumPy: np.sum(np.broadcast_to(5, (3, 3)))
        // Broadcasts 5 to [[5,5,5], [5,5,5], [5,5,5]]
        // sum = 45
        var scalar = np.array(5);
        var broadcasted = np.broadcast_to(scalar, new Shape(3, 3));

        Assert.AreEqual(3, broadcasted.shape[0]);
        Assert.AreEqual(3, broadcasted.shape[1]);

        var result = np.sum(broadcasted);

        Assert.AreEqual(45.0, (double)result, 1e-10);
    }

    [TestMethod]
    public void BroadcastTo_1D_Sqrt()
    {
        // NumPy: np.sqrt(np.broadcast_to([1, 4, 9], (3, 3)))
        // Broadcasts [1,4,9] to [[1,4,9], [1,4,9], [1,4,9]]
        // sqrt = [[1,2,3], [1,2,3], [1,2,3]]
        var arr = np.array(new double[] { 1, 4, 9 });
        var broadcasted = np.broadcast_to(arr, new Shape(3, 3));

        var result = np.sqrt(broadcasted);

        for (int i = 0; i < 3; i++)
        {
            Assert.AreEqual(1.0, result.GetDouble(i, 0), 1e-10);
            Assert.AreEqual(2.0, result.GetDouble(i, 1), 1e-10);
            Assert.AreEqual(3.0, result.GetDouble(i, 2), 1e-10);
        }
    }

    [TestMethod]
    public void BroadcastTo_Column_Add()
    {
        // NumPy: np.broadcast_to([[1], [2], [3]], (3, 3)) + np.broadcast_to([10, 20, 30], (3, 3))
        // [[1,1,1], [2,2,2], [3,3,3]] + [[10,20,30], [10,20,30], [10,20,30]]
        // = [[11,21,31], [12,22,32], [13,23,33]]
        var col = np.array(new double[,] { { 1 }, { 2 }, { 3 } });
        var row = np.array(new double[] { 10, 20, 30 });
        var bCol = np.broadcast_to(col, new Shape(3, 3));
        var bRow = np.broadcast_to(row, new Shape(3, 3));

        var result = bCol + bRow;

        Assert.AreEqual(11.0, result.GetDouble(0, 0), 1e-10);
        Assert.AreEqual(21.0, result.GetDouble(0, 1), 1e-10);
        Assert.AreEqual(31.0, result.GetDouble(0, 2), 1e-10);
        Assert.AreEqual(12.0, result.GetDouble(1, 0), 1e-10);
        Assert.AreEqual(33.0, result.GetDouble(2, 2), 1e-10);
    }

    [TestMethod]
    public void BroadcastTo_Mean()
    {
        // NumPy: np.mean(np.broadcast_to([1, 2, 3], (4, 3)))
        // mean of 12 elements all being [1,2,3] repeated 4 times = (1+2+3)*4/12 = 2.0
        var arr = np.array(new double[] { 1, 2, 3 });
        var broadcasted = np.broadcast_to(arr, new Shape(4, 3));

        var result = np.mean(broadcasted);

        Assert.AreEqual(2.0, result.GetDouble(0), 1e-10);
    }

    #endregion

    #region Reversed Array Operations

    [TestMethod]
    public void Reversed_Sqrt()
    {
        // NumPy: np.sqrt(arr[::-1]) where arr = [1, 4, 9, 16, 25]
        // arr[::-1] = [25, 16, 9, 4, 1]
        // sqrt = [5, 4, 3, 2, 1]
        var arr = np.array(new double[] { 1, 4, 9, 16, 25 });
        var reversed = arr["::-1"];

        var result = np.sqrt(reversed);

        Assert.AreEqual(5.0, result.GetDouble(0), 1e-10);
        Assert.AreEqual(4.0, result.GetDouble(1), 1e-10);
        Assert.AreEqual(3.0, result.GetDouble(2), 1e-10);
        Assert.AreEqual(2.0, result.GetDouble(3), 1e-10);
        Assert.AreEqual(1.0, result.GetDouble(4), 1e-10);
    }

    [TestMethod]
    public void Reversed_Add_WithOriginal()
    {
        // NumPy: arr + arr[::-1] where arr = [1, 2, 3, 4, 5]
        // Result = [6, 6, 6, 6, 6]
        var arr = np.array(new double[] { 1, 2, 3, 4, 5 });

        var result = arr + arr["::-1"];

        for (int i = 0; i < 5; i++)
        {
            Assert.AreEqual(6.0, result.GetDouble(i), 1e-10);
        }
    }

    [TestMethod]
    public void Reversed_2D_Rows()
    {
        // NumPy: np.sum(arr[::-1, :]) where arr = [[1,2,3], [4,5,6]]
        // arr[::-1, :] = [[4,5,6], [1,2,3]]
        // sum = 21
        var arr = np.array(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        var reversed = arr["::-1, :"];

        Assert.AreEqual(4.0, reversed.GetDouble(0, 0), 1e-10);
        Assert.AreEqual(1.0, reversed.GetDouble(1, 0), 1e-10);

        var result = np.sum(reversed);
        Assert.AreEqual(21.0, result.GetDouble(0), 1e-10);
    }

    [TestMethod]
    public void Reversed_2D_Cols()
    {
        // NumPy: arr[:, ::-1] where arr = [[1,2,3], [4,5,6]]
        // Result = [[3,2,1], [6,5,4]]
        var arr = np.array(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        var reversed = arr[":, ::-1"];

        Assert.AreEqual(3.0, reversed.GetDouble(0, 0), 1e-10);
        Assert.AreEqual(2.0, reversed.GetDouble(0, 1), 1e-10);
        Assert.AreEqual(1.0, reversed.GetDouble(0, 2), 1e-10);
        Assert.AreEqual(6.0, reversed.GetDouble(1, 0), 1e-10);
    }

    #endregion

    #region Multi-Dimensional Slice Operations

    [TestMethod]
    public void Slice3D_Sum()
    {
        // NumPy: np.sum(arr[::, ::2, ::]) where arr has shape (2, 3, 4)
        var arr = np.arange(24).reshape(2, 3, 4);
        var sliced = arr[":, ::2, :"];  // shape (2, 2, 4)

        Assert.AreEqual(2, sliced.shape[0]);
        Assert.AreEqual(2, sliced.shape[1]);
        Assert.AreEqual(4, sliced.shape[2]);

        // Elements: [0..3, 8..11] from first (2,3,4) slice + [12..15, 20..23] from second
        // = 0+1+2+3 + 8+9+10+11 + 12+13+14+15 + 20+21+22+23 = 6 + 38 + 54 + 86 = 184
        var result = np.sum(sliced);
        Assert.AreEqual(184.0, (double)result, 1e-10);
    }

    [TestMethod]
    public void Slice3D_Sqrt()
    {
        // Create array with perfect squares for testing
        var arr = np.array(new double[] { 1, 4, 9, 16, 25, 36, 49, 64 }).reshape(2, 2, 2);
        var sliced = arr[":, :, 0"];  // Take first element of last dimension

        // sliced = [[1, 9], [25, 49]]
        var result = np.sqrt(sliced);

        Assert.AreEqual(1.0, result.GetDouble(0, 0), 1e-10);
        Assert.AreEqual(3.0, result.GetDouble(0, 1), 1e-10);
        Assert.AreEqual(5.0, result.GetDouble(1, 0), 1e-10);
        Assert.AreEqual(7.0, result.GetDouble(1, 1), 1e-10);
    }

    #endregion

    #region ATan2 on Non-Contiguous Arrays (Task #73 fix verification)

    [TestMethod]
    public void ATan2_SlicedArrays()
    {
        // NumPy: np.arctan2(y[::2], x[::2])
        // y = [0, 999, 1, 999, 0], x = [1, 999, 0, 999, -1]
        // y[::2] = [0, 1, 0], x[::2] = [1, 0, -1]
        // arctan2 = [0, pi/2, pi]
        var y = np.array(new double[] { 0, 999, 1, 999, 0 });
        var x = np.array(new double[] { 1, 999, 0, 999, -1 });

        var result = np.arctan2(y["::2"], x["::2"]);

        Assert.AreEqual(0.0, result.GetDouble(0), 1e-10);
        Assert.AreEqual(Math.PI / 2, result.GetDouble(1), 1e-10);
        Assert.AreEqual(Math.PI, result.GetDouble(2), 1e-10);
    }

    [TestMethod]
    public void ATan2_TransposedArrays()
    {
        // NumPy: np.arctan2(y.T, x.T)
        var y = np.array(new double[,] { { 0, 1 }, { 1, 0 } });
        var x = np.array(new double[,] { { 1, 0 }, { 0, -1 } });

        var result = np.arctan2(y.T, x.T);

        // y.T = [[0, 1], [1, 0]], x.T = [[1, 0], [0, -1]]
        // arctan2(0,1)=0, arctan2(1,0)=pi/2, arctan2(1,0)=pi/2, arctan2(0,-1)=pi
        Assert.AreEqual(0.0, result.GetDouble(0, 0), 1e-10);
        Assert.AreEqual(Math.PI / 2, result.GetDouble(0, 1), 1e-10);
        Assert.AreEqual(Math.PI / 2, result.GetDouble(1, 0), 1e-10);
        Assert.AreEqual(Math.PI, result.GetDouble(1, 1), 1e-10);
    }

    [TestMethod]
    public void ATan2_ReversedArrays()
    {
        // NumPy: np.arctan2(y[::-1], x[::-1])
        var y = np.array(new double[] { 0, 1, 0 });
        var x = np.array(new double[] { -1, 0, 1 });

        // y[::-1] = [0, 1, 0], x[::-1] = [1, 0, -1]
        // arctan2 = [0, pi/2, pi]
        var result = np.arctan2(y["::-1"], x["::-1"]);

        Assert.AreEqual(0.0, result.GetDouble(0), 1e-10);
        Assert.AreEqual(Math.PI / 2, result.GetDouble(1), 1e-10);
        Assert.AreEqual(Math.PI, result.GetDouble(2), 1e-10);
    }

    #endregion

    #region Negate Boolean on Non-Contiguous Arrays (Task #74 fix verification)

    [TestMethod]
    [OpenBugs]  // Remove when Task #74 is fixed
    public void NegateBoolean_SlicedArray()
    {
        // NumPy: np.negative(arr[::2]) where arr = [True, False, True, False, True]
        // arr[::2] = [True, True, True]
        // negative(bool) in NumPy returns int: [-1, -1, -1]
        var arr = np.array(new[] { true, false, true, false, true });
        var sliced = arr["::2"];

        var result = np.negative(sliced);

        // NumPy negative on bool converts to int and negates
        Assert.AreEqual(-1, result.GetInt32(0));
        Assert.AreEqual(-1, result.GetInt32(1));
        Assert.AreEqual(-1, result.GetInt32(2));
    }

    [TestMethod]
    [OpenBugs]  // Remove when Task #74 is fixed
    public void NegateBoolean_ReversedArray()
    {
        // NumPy: np.negative(arr[::-1]) where arr = [True, False, False, True]
        // arr[::-1] = [True, False, False, True]
        var arr = np.array(new[] { true, false, false, true });
        var reversed = arr["::-1"];

        var result = np.negative(reversed);

        Assert.AreEqual(-1, result.GetInt32(0));
        Assert.AreEqual(0, result.GetInt32(1));
        Assert.AreEqual(0, result.GetInt32(2));
        Assert.AreEqual(-1, result.GetInt32(3));
    }

    #endregion

    #region Combined Non-Contiguous Patterns

    [TestMethod]
    public void TransposedSlice_Sum()
    {
        // NumPy: np.sum(arr.T[::2, :])
        var arr = np.array(new double[,] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } });
        // arr.T shape = (4, 2), arr.T[::2, :] = [[1,5], [3,7]] shape (2, 2)
        var transposed = arr.T;
        var sliced = transposed["::2, :"];

        Assert.AreEqual(2, sliced.shape[0]);
        Assert.AreEqual(2, sliced.shape[1]);

        var result = np.sum(sliced);
        Assert.AreEqual(16.0, result.GetDouble(0), 1e-10);  // 1+5+3+7=16
    }

    [TestMethod]
    public void SlicedTransposed_Multiply()
    {
        // Slice first, then transpose
        var arr = np.array(new double[,] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 }, { 9, 10, 11, 12 } });
        // arr[::2, :] = [[1,2,3,4], [9,10,11,12]] shape (2, 4)
        // Transpose = shape (4, 2)
        var sliced = arr["::2, :"];
        var transposed = sliced.T;

        Assert.AreEqual(4, transposed.shape[0]);
        Assert.AreEqual(2, transposed.shape[1]);

        // transposed = [[1,9], [2,10], [3,11], [4,12]]
        var result = transposed * transposed;

        Assert.AreEqual(1.0, result.GetDouble(0, 0), 1e-10);
        Assert.AreEqual(81.0, result.GetDouble(0, 1), 1e-10);
        Assert.AreEqual(4.0, result.GetDouble(1, 0), 1e-10);
        Assert.AreEqual(100.0, result.GetDouble(1, 1), 1e-10);
    }

    [TestMethod]
    public void ReversedTransposed_Add()
    {
        var arr = np.array(new double[,] { { 1, 2 }, { 3, 4 } });
        // arr.T = [[1, 3], [2, 4]]
        // arr.T[::-1, :] = [[2, 4], [1, 3]]
        var transposed = arr.T;
        var reversed = transposed["::-1, :"];

        var result = transposed + reversed;
        // [[1,3], [2,4]] + [[2,4], [1,3]] = [[3,7], [3,7]]

        Assert.AreEqual(3.0, result.GetDouble(0, 0), 1e-10);
        Assert.AreEqual(7.0, result.GetDouble(0, 1), 1e-10);
        Assert.AreEqual(3.0, result.GetDouble(1, 0), 1e-10);
        Assert.AreEqual(7.0, result.GetDouble(1, 1), 1e-10);
    }

    #endregion

    #region All/Any on Non-Contiguous Arrays

    [TestMethod]
    public void All_SlicedBoolArray()
    {
        // NumPy: np.all(arr[::2]) where arr = [True, False, True, False, True]
        // arr[::2] = [True, True, True]
        // Result: True
        var arr = np.array(new[] { true, false, true, false, true });
        var sliced = arr["::2"];

        var result = np.all(sliced);

        Assert.IsTrue(result);
    }

    [TestMethod]
    public void All_SlicedBoolArray_False()
    {
        // NumPy: np.all(arr[1::2]) where arr = [True, False, True, False, True]
        // arr[1::2] = [False, False]
        // Result: False
        var arr = np.array(new[] { true, false, true, false, true });
        var sliced = arr["1::2"];

        var result = np.all(sliced);

        Assert.IsFalse(result);
    }

    [TestMethod]
    public void Any_SlicedBoolArray()
    {
        // NumPy: np.any(arr[::2]) where arr = [False, True, False, True, False]
        // arr[::2] = [False, False, False]
        // Result: False
        var arr = np.array(new[] { false, true, false, true, false });
        var sliced = arr["::2"];

        var result = np.any(sliced);

        Assert.IsFalse(result);
    }

    [TestMethod]
    public void Any_TransposedBoolArray()
    {
        // NumPy: np.any(arr.T) where arr = [[False, False], [True, False]]
        // arr.T = [[False, True], [False, False]]
        // Result: True
        var arr = np.array(new[,] { { false, false }, { true, false } });
        var transposed = arr.T;

        var result = np.any(transposed);

        Assert.IsTrue(result);
    }

    #endregion

    #region ArgMax/ArgMin on Non-Contiguous Arrays

    [TestMethod]
    public void ArgMax_SlicedArray()
    {
        // NumPy: np.argmax(arr[::2]) where arr = [1, 999, 5, 999, 3, 999, 7, 999]
        // arr[::2] = [1, 5, 3, 7]
        // argmax = 3 (index of 7)
        var arr = np.array(new double[] { 1, 999, 5, 999, 3, 999, 7, 999 });
        var sliced = arr["::2"];

        var result = np.argmax(sliced);

        Assert.AreEqual(3, result);
    }

    [TestMethod]
    public void ArgMin_ReversedArray()
    {
        // NumPy: np.argmin(arr[::-1]) where arr = [5, 2, 8, 1, 9]
        // arr[::-1] = [9, 1, 8, 2, 5]
        // argmin = 1 (index of 1)
        var arr = np.array(new double[] { 5, 2, 8, 1, 9 });
        var reversed = arr["::-1"];

        var result = np.argmin(reversed);

        Assert.AreEqual(1, result);
    }

    [TestMethod]
    public void ArgMax_TransposedArray()
    {
        // NumPy: np.argmax(arr.T) where arr = [[1, 2], [9, 4]]
        // arr.T = [[1, 9], [2, 4]]
        // flattened = [1, 9, 2, 4]
        // argmax = 1 (index of 9)
        var arr = np.array(new double[,] { { 1, 2 }, { 9, 4 } });
        var transposed = arr.T;

        var result = np.argmax(transposed);

        Assert.AreEqual(1, result);
    }

    #endregion

    #region Power on Non-Contiguous Arrays

    [TestMethod]
    public void Power_SlicedArrays()
    {
        // NumPy: np.power(base[::2], exp[::2])
        // base = [2, 999, 3, 999, 4], exp = [1, 999, 2, 999, 3]
        // base[::2] = [2, 3, 4], exp[::2] = [1, 2, 3]
        // Result: [2, 9, 64]
        var bases = np.array(new double[] { 2, 999, 3, 999, 4 });
        var exponents = np.array(new double[] { 1, 999, 2, 999, 3 });

        var result = np.power(bases["::2"], exponents["::2"]);

        Assert.AreEqual(2.0, result.GetDouble(0), 1e-10);
        Assert.AreEqual(9.0, result.GetDouble(1), 1e-10);
        Assert.AreEqual(64.0, result.GetDouble(2), 1e-10);
    }

    [TestMethod]
    public void Power_TransposedArrays()
    {
        // NumPy: np.power(base.T, exp.T)
        var bases = np.array(new double[,] { { 2, 3 }, { 4, 5 } });
        var exponents = np.array(new double[,] { { 1, 2 }, { 2, 1 } });

        var result = np.power(bases.T, exponents.T);

        // bases.T = [[2, 4], [3, 5]], exp.T = [[1, 2], [2, 1]]
        // Result: [[2^1, 4^2], [3^2, 5^1]] = [[2, 16], [9, 5]]
        Assert.AreEqual(2.0, result.GetDouble(0, 0), 1e-10);
        Assert.AreEqual(16.0, result.GetDouble(0, 1), 1e-10);
        Assert.AreEqual(9.0, result.GetDouble(1, 0), 1e-10);
        Assert.AreEqual(5.0, result.GetDouble(1, 1), 1e-10);
    }

    #endregion

    #region FloorDivide on Non-Contiguous Arrays

    [TestMethod]
    public void FloorDivide_SlicedArrays()
    {
        // NumPy: np.floor_divide(a[::2], b[::2])
        var a = np.array(new double[] { 10, 999, 23, 999, 35 });
        var b = np.array(new double[] { 3, 999, 5, 999, 6 });

        var result = np.floor_divide(a["::2"], b["::2"]);

        // 10//3=3, 23//5=4, 35//6=5
        Assert.AreEqual(3.0, result.GetDouble(0), 1e-10);
        Assert.AreEqual(4.0, result.GetDouble(1), 1e-10);
        Assert.AreEqual(5.0, result.GetDouble(2), 1e-10);
    }

    #endregion
}
