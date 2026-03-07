using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.UnitTest.Utilities;
using TUnit.Core;

namespace NumSharp.UnitTest.Backends.Kernels;

/// <summary>
/// Comprehensive tests for NonZero, ArgMax/ArgMin, and Boolean Masking operations.
/// All expected values are verified against NumPy 2.4.2 output.
///
/// Test cases are ported from NumPy's test suite:
/// - src/numpy/numpy/_core/tests/test_numeric.py (nonzero tests)
/// - src/numpy/numpy/_core/tests/test_multiarray.py (argmax/argmin, masking)
/// - src/numpy/numpy/_core/tests/test_indexing.py (boolean indexing)
/// </summary>
public class SimdOptimizationTests
{
    #region NonZero Tests (from NumPy test_numeric.py)

    [Test]
    public void NonZero_1D_Basic()
    {
        // NumPy: np.nonzero([0, 1, 0, 3, 0, 5]) = [[1, 3, 5]]
        var a = np.array(new[] { 0, 1, 0, 3, 0, 5 });
        var result = np.nonzero(a);

        Assert.AreEqual(1, result.Length);
        result[0].Should().BeOfValues(1, 3, 5);
    }

    [Test]
    public void NonZero_2D_Basic()
    {
        // NumPy: np.nonzero([[0, 1, 0], [3, 0, 5]]) = [[0, 1, 1], [1, 0, 2]]
        var a = np.array(new[,] { { 0, 1, 0 }, { 3, 0, 5 } });
        var result = np.nonzero(a);

        Assert.AreEqual(2, result.Length);
        result[0].Should().BeOfValues(0, 1, 1);  // row indices
        result[1].Should().BeOfValues(1, 0, 2);  // col indices
    }

    [Test]
    public void NonZero_AllZeros()
    {
        // NumPy: np.nonzero(zeros(5)) = [[]]
        var a = np.zeros<int>(5);
        var result = np.nonzero(a);

        Assert.AreEqual(1, result.Length);
        Assert.AreEqual(0, result[0].size);
    }

    [Test]
    public void NonZero_AllNonzero()
    {
        // NumPy: np.nonzero([1, 2, 3, 4, 5]) = [[0, 1, 2, 3, 4]]
        var a = np.array(new[] { 1, 2, 3, 4, 5 });
        var result = np.nonzero(a);

        Assert.AreEqual(1, result.Length);
        result[0].Should().BeOfValues(0, 1, 2, 3, 4);
    }

    [Test]
    public void NonZero_Boolean()
    {
        // NumPy: np.nonzero([True, False, True, False, True]) = [[0, 2, 4]]
        var a = np.array(new[] { true, false, true, false, true });
        var result = np.nonzero(a);

        Assert.AreEqual(1, result.Length);
        result[0].Should().BeOfValues(0, 2, 4);
    }

    [Test]
    public void NonZero_Float()
    {
        // NumPy: np.nonzero([0.0, 1.5, 0.0, -2.5, 0.0]) = [[1, 3]]
        var a = np.array(new[] { 0.0, 1.5, 0.0, -2.5, 0.0 });
        var result = np.nonzero(a);

        Assert.AreEqual(1, result.Length);
        result[0].Should().BeOfValues(1, 3);
    }

    [Test]
    public void NonZero_Large_SparseValues()
    {
        // NumPy: Large array with sparse nonzero values (tests SIMD path)
        var a = np.zeros<int>(1000);
        a[100] = 1;
        a[500] = 2;
        a[999] = 3;
        var result = np.nonzero(a);

        Assert.AreEqual(1, result.Length);
        result[0].Should().BeOfValues(100, 500, 999);
    }

    [Test]
    [OpenBugs]  // NumSharp throws "size > 0" for empty arrays
    public void NonZero_Empty()
    {
        // NumPy: np.nonzero([]) = [[]]
        var a = np.array(Array.Empty<int>());
        var result = np.nonzero(a);

        Assert.AreEqual(1, result.Length);
        Assert.AreEqual(0, result[0].size);
    }

    [Test]
    public void NonZero_3D()
    {
        // NumPy: 3D array with sparse nonzero values
        // result = [[0, 1], [1, 2], [2, 3]]
        var a = np.zeros(new Shape(2, 3, 4), typeof(int));
        a.SetInt32(1, 0, 1, 2);
        a.SetInt32(2, 1, 2, 3);
        var result = np.nonzero(a);

        Assert.AreEqual(3, result.Length);
        result[0].Should().BeOfValues(0, 1);  // dim 0 indices
        result[1].Should().BeOfValues(1, 2);  // dim 1 indices
        result[2].Should().BeOfValues(2, 3);  // dim 2 indices
    }

    [Test]
    public void NonZero_NaN_IsNonzero()
    {
        // NumPy: NaN is considered non-zero
        // np.nonzero([0.0, nan, 1.0, nan]) = [[1, 2, 3]]
        var a = np.array(new[] { 0.0, double.NaN, 1.0, double.NaN });
        var result = np.nonzero(a);

        Assert.AreEqual(1, result.Length);
        result[0].Should().BeOfValues(1, 2, 3);
    }

    [Test]
    public void NonZero_EyeMatrix()
    {
        // NumPy: np.nonzero(eye(3)) = [[0, 1, 2], [0, 1, 2]]
        var a = np.eye(3);
        var result = np.nonzero(a);

        Assert.AreEqual(2, result.Length);
        result[0].Should().BeOfValues(0, 1, 2);
        result[1].Should().BeOfValues(0, 1, 2);
    }

    [Test]
    [OpenBugs]  // sbyte (int8) not supported by NumSharp
    public void NonZero_Int8()
    {
        // NumPy: dtype=int8, result = [[0, 2, 3, 6]]
        var a = np.array(new sbyte[] { 1, 0, 2, -1, 0, 0, 8 });
        var result = np.nonzero(a);

        Assert.AreEqual(1, result.Length);
        result[0].Should().BeOfValues(0, 2, 3, 6);
    }

    [Test]
    public void NonZero_UInt16()
    {
        // NumPy: dtype=uint16, result = [[1, 3]]
        var a = np.array(new ushort[] { 0, 255, 0, 128, 0 });
        var result = np.nonzero(a);

        Assert.AreEqual(1, result.Length);
        result[0].Should().BeOfValues(1, 3);
    }

    [Test]
    public void NonZero_SparsePattern()
    {
        // From NumPy test_sparse: sparse boolean pattern
        var c = np.zeros<bool>(200);
        for (int i = 0; i < 200; i += 20)
            c[i] = true;
        var result = np.nonzero(c);

        result[0].Should().BeOfValues(0, 20, 40, 60, 80, 100, 120, 140, 160, 180);
    }

    [Test]
    public void NonZero_FromNumPyTest_Onedim()
    {
        // NumPy test_nonzero_onedim: x = [1, 0, 2, -1, 0, 0, 8]
        var x = np.array(new[] { 1, 0, 2, -1, 0, 0, 8 });
        var result = np.nonzero(x);

        result[0].Should().BeOfValues(0, 2, 3, 6);
    }

    [Test]
    public void NonZero_FromNumPyTest_Twodim()
    {
        // NumPy test_nonzero_twodim: x = [[0, 1, 0], [2, 0, 3]]
        var x = np.array(new[,] { { 0, 1, 0 }, { 2, 0, 3 } });
        var result = np.nonzero(x);

        result[0].Should().BeOfValues(0, 1, 1);
        result[1].Should().BeOfValues(1, 0, 2);
    }

    #endregion

    #region ArgMax/ArgMin Tests (from NumPy test_multiarray.py, test_regression.py)

    [Test]
    public void ArgMax_1D_Basic()
    {
        // NumPy: np.argmax([3, 1, 4, 1, 5, 9, 2, 6]) = 5
        var a = np.array(new[] { 3, 1, 4, 1, 5, 9, 2, 6 });

        Assert.AreEqual(5, np.argmax(a));
    }

    [Test]
    public void ArgMin_1D_Basic()
    {
        // NumPy: np.argmin([3, 1, 4, 1, 5, 9, 2, 6]) = 1
        var a = np.array(new[] { 3, 1, 4, 1, 5, 9, 2, 6 });

        Assert.AreEqual(1, np.argmin(a));
    }

    [Test]
    public void ArgMax_Ties_ReturnsFirstOccurrence()
    {
        // NumPy: np.argmax([5, 1, 5, 1, 5]) = 0 (first occurrence)
        var a = np.array(new[] { 5, 1, 5, 1, 5 });

        Assert.AreEqual(0, np.argmax(a));
    }

    [Test]
    public void ArgMin_Ties_ReturnsFirstOccurrence()
    {
        // NumPy: np.argmin([5, 1, 5, 1, 5]) = 1 (first occurrence of min)
        var a = np.array(new[] { 5, 1, 5, 1, 5 });

        Assert.AreEqual(1, np.argmin(a));
    }

    [Test]
    public void ArgMax_SingleElement()
    {
        // NumPy: np.argmax([42]) = 0
        var a = np.array(new[] { 42 });

        Assert.AreEqual(0, np.argmax(a));
    }

    [Test]
    public void ArgMin_SingleElement()
    {
        // NumPy: np.argmin([42]) = 0
        var a = np.array(new[] { 42 });

        Assert.AreEqual(0, np.argmin(a));
    }

    [Test]
    public void ArgMax_NegativeValues()
    {
        // NumPy: np.argmax([-5, -1, -3, -2, -4]) = 1
        var a = np.array(new[] { -5, -1, -3, -2, -4 });

        Assert.AreEqual(1, np.argmax(a));
    }

    [Test]
    public void ArgMin_NegativeValues()
    {
        // NumPy: np.argmin([-5, -1, -3, -2, -4]) = 0
        var a = np.array(new[] { -5, -1, -3, -2, -4 });

        Assert.AreEqual(0, np.argmin(a));
    }

    [Test]
    public void ArgMax_Infinity()
    {
        // NumPy: np.argmax([1.0, inf, -inf, 0.0]) = 1
        var a = np.array(new[] { 1.0, double.PositiveInfinity, double.NegativeInfinity, 0.0 });

        Assert.AreEqual(1, np.argmax(a));
    }

    [Test]
    public void ArgMin_Infinity()
    {
        // NumPy: np.argmin([1.0, inf, -inf, 0.0]) = 2
        var a = np.array(new[] { 1.0, double.PositiveInfinity, double.NegativeInfinity, 0.0 });

        Assert.AreEqual(2, np.argmin(a));
    }

    [Test]
    public void ArgMax_NaN_FirstNaNWins()
    {
        // NumPy: np.argmax([1.0, nan, 3.0, nan]) = 1 (NaN propagates, first NaN index)
        var a = np.array(new[] { 1.0, double.NaN, 3.0, double.NaN });

        Assert.AreEqual(1, np.argmax(a));
    }

    [Test]
    public void ArgMin_NaN_FirstNaNWins()
    {
        // NumPy: np.argmin([1.0, nan, 3.0, nan]) = 1 (NaN propagates, first NaN index)
        var a = np.array(new[] { 1.0, double.NaN, 3.0, double.NaN });

        Assert.AreEqual(1, np.argmin(a));
    }

    [Test]
    public void ArgMax_2D_Flattened()
    {
        // NumPy: np.argmax([[1, 2, 3], [4, 5, 6]]) = 5 (flat index)
        var a = np.array(new[,] { { 1, 2, 3 }, { 4, 5, 6 } });

        Assert.AreEqual(5, np.argmax(a));
    }

    [Test]
    public void ArgMin_2D_Flattened()
    {
        // NumPy: np.argmin([[1, 2, 3], [4, 5, 6]]) = 0 (flat index)
        var a = np.array(new[,] { { 1, 2, 3 }, { 4, 5, 6 } });

        Assert.AreEqual(0, np.argmin(a));
    }

    [Test]
    public void ArgMax_2D_Axis0()
    {
        // NumPy: np.argmax([[1, 5, 3], [4, 2, 6]], axis=0) = [1, 0, 1]
        var a = np.array(new[,] { { 1, 5, 3 }, { 4, 2, 6 } });

        var result = np.argmax(a, 0);

        result.Should().BeOfValues(1, 0, 1);
    }

    [Test]
    public void ArgMin_2D_Axis0()
    {
        // NumPy: np.argmin([[1, 5, 3], [4, 2, 6]], axis=0) = [0, 1, 0]
        var a = np.array(new[,] { { 1, 5, 3 }, { 4, 2, 6 } });

        var result = np.argmin(a, 0);

        result.Should().BeOfValues(0, 1, 0);
    }

    [Test]
    public void ArgMax_2D_Axis1()
    {
        // NumPy: np.argmax([[1, 5, 3], [4, 2, 6]], axis=1) = [1, 2]
        var a = np.array(new[,] { { 1, 5, 3 }, { 4, 2, 6 } });

        var result = np.argmax(a, 1);

        result.Should().BeOfValues(1, 2);
    }

    [Test]
    public void ArgMin_2D_Axis1()
    {
        // NumPy: np.argmin([[1, 5, 3], [4, 2, 6]], axis=1) = [0, 1]
        var a = np.array(new[,] { { 1, 5, 3 }, { 4, 2, 6 } });

        var result = np.argmin(a, 1);

        result.Should().BeOfValues(0, 1);
    }

    [Test]
    public void ArgMax_Large_SIMDPath()
    {
        // NumPy: Large array (tests SIMD path), max at end
        var a = np.arange(10000);

        Assert.AreEqual(9999, np.argmax(a));
    }

    [Test]
    public void ArgMin_Large_SIMDPath()
    {
        // NumPy: Large array (tests SIMD path), min at start
        var a = np.arange(10000);

        Assert.AreEqual(0, np.argmin(a));
    }

    [Test]
    public void ArgMax_Large_MaxInMiddle()
    {
        // NumPy: Large array with max in middle
        var a = np.zeros<double>(10000);
        a[5000] = 1.0;

        Assert.AreEqual(5000, np.argmax(a));
    }

    [Test]
    public void ArgMax_UInt8()
    {
        // NumPy: np.argmax([255, 0, 128, 64], dtype=uint8) = 0
        var a = np.array(new byte[] { 255, 0, 128, 64 });

        Assert.AreEqual(0, np.argmax(a));
    }

    [Test]
    public void ArgMin_UInt8()
    {
        // NumPy: np.argmin([255, 0, 128, 64], dtype=uint8) = 1
        var a = np.array(new byte[] { 255, 0, 128, 64 });

        Assert.AreEqual(1, np.argmin(a));
    }

    [Test]
    [OpenBugs]  // ArgMax not supported for Boolean type
    public void ArgMax_Boolean()
    {
        // NumPy: np.argmax([False, True, False, True]) = 1
        var a = np.array(new[] { false, true, false, true });

        Assert.AreEqual(1, np.argmax(a));
    }

    [Test]
    [OpenBugs]  // ArgMin not supported for Boolean type
    public void ArgMin_Boolean()
    {
        // NumPy: np.argmin([False, True, False, True]) = 0
        var a = np.array(new[] { false, true, false, true });

        Assert.AreEqual(0, np.argmin(a));
    }

    [Test]
    public void ArgMax_Int16()
    {
        // NumPy: np.argmax([100, -200, 300, -400, 500], dtype=int16) = 4
        var a = np.array(new short[] { 100, -200, 300, -400, 500 });

        Assert.AreEqual(4, np.argmax(a));
    }

    [Test]
    public void ArgMin_Int16()
    {
        // NumPy: np.argmin([100, -200, 300, -400, 500], dtype=int16) = 3
        var a = np.array(new short[] { 100, -200, 300, -400, 500 });

        Assert.AreEqual(3, np.argmin(a));
    }

    [Test]
    public void ArgMax_Int64()
    {
        // NumPy: np.argmax([1000000000000, -2000000000000, 3000000000000], dtype=int64) = 2
        var a = np.array(new long[] { 1000000000000, -2000000000000, 3000000000000 });

        Assert.AreEqual(2, np.argmax(a));
    }

    [Test]
    public void ArgMin_Int64()
    {
        // NumPy: np.argmin([1000000000000, -2000000000000, 3000000000000], dtype=int64) = 1
        var a = np.array(new long[] { 1000000000000, -2000000000000, 3000000000000 });

        Assert.AreEqual(1, np.argmin(a));
    }

    [Test]
    public void ArgMax_Float32()
    {
        // NumPy: np.argmax([1.5, 2.5, 0.5, 3.5], dtype=float32) = 3
        var a = np.array(new float[] { 1.5f, 2.5f, 0.5f, 3.5f });

        Assert.AreEqual(3, np.argmax(a));
    }

    [Test]
    public void ArgMin_Float32()
    {
        // NumPy: np.argmin([1.5, 2.5, 0.5, 3.5], dtype=float32) = 2
        var a = np.array(new float[] { 1.5f, 2.5f, 0.5f, 3.5f });

        Assert.AreEqual(2, np.argmin(a));
    }

    [Test]
    public void ArgMax_2D_NegativeAxis()
    {
        // NumPy: np.argmax([[1, 5, 3], [4, 2, 6]], axis=-1) = [1, 2]
        var a = np.array(new[,] { { 1, 5, 3 }, { 4, 2, 6 } });

        var result = np.argmax(a, -1);

        result.Should().BeOfValues(1, 2);
    }

    [Test]
    public void ArgMin_2D_NegativeAxis()
    {
        // NumPy: np.argmin([[1, 5, 3], [4, 2, 6]], axis=-1) = [0, 1]
        var a = np.array(new[,] { { 1, 5, 3 }, { 4, 2, 6 } });

        var result = np.argmin(a, -1);

        result.Should().BeOfValues(0, 1);
    }

    [Test]
    public void ArgMax_AllSameValues()
    {
        // NumPy: np.argmax([7, 7, 7, 7, 7]) = 0 (returns first)
        var a = np.array(new[] { 7, 7, 7, 7, 7 });

        Assert.AreEqual(0, np.argmax(a));
    }

    [Test]
    public void ArgMin_AllSameValues()
    {
        // NumPy: np.argmin([7, 7, 7, 7, 7]) = 0 (returns first)
        var a = np.array(new[] { 7, 7, 7, 7, 7 });

        Assert.AreEqual(0, np.argmin(a));
    }

    [Test]
    public void ArgMax_DecreasingOrder()
    {
        // NumPy: np.argmax([9, 8, 7, 6, 5, 4, 3, 2, 1, 0]) = 0
        var a = np.array(new[] { 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 });

        Assert.AreEqual(0, np.argmax(a));
    }

    [Test]
    public void ArgMin_DecreasingOrder()
    {
        // NumPy: np.argmin([9, 8, 7, 6, 5, 4, 3, 2, 1, 0]) = 9
        var a = np.array(new[] { 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 });

        Assert.AreEqual(9, np.argmin(a));
    }

    #endregion

    #region Boolean Masking Tests (from NumPy test_indexing.py, test_multiarray.py)

    [Test]
    [OpenBugs]  // Boolean masking with explicit mask array fails - returns all elements
    public void BooleanMask_1D_Basic()
    {
        // NumPy: a[[T,F,T,F,T,F]] = [1, 3, 5]
        var a = np.array(new[] { 1, 2, 3, 4, 5, 6 });
        var mask = np.array(new[] { true, false, true, false, true, false });

        var result = a[mask];

        result.Should().BeOfValues(1, 3, 5);
    }

    [Test]
    public void BooleanMask_Condition()
    {
        // NumPy: a[a > 3] = [4, 5, 6]
        var a = np.array(new[] { 1, 2, 3, 4, 5, 6 });

        var result = a[a > 3];

        result.Should().BeOfValues(4, 5, 6);
    }

    [Test]
    [OpenBugs]  // Boolean masking with explicit mask array returns wrong values
    public void BooleanMask_AllTrue()
    {
        // NumPy: a[[T, T, T]] = [1, 2, 3]
        var a = np.array(new[] { 1, 2, 3 });
        var mask = np.array(new[] { true, true, true });

        var result = a[mask];

        result.Should().BeOfValues(1, 2, 3);
    }

    [Test]
    [OpenBugs]  // Boolean masking ignores mask, returns all elements
    public void BooleanMask_AllFalse()
    {
        // NumPy: a[[F, F, F]] = []
        var a = np.array(new[] { 1, 2, 3 });
        var mask = np.array(new[] { false, false, false });

        var result = a[mask];

        Assert.AreEqual(0, result.size);
    }

    [Test]
    [OpenBugs]  // Boolean masking ignores mask, returns all elements
    public void BooleanMask_EmptyResult_Shape()
    {
        // NumPy: Empty result has shape (0,) and preserves dtype
        var a = np.array(new[] { 1, 2, 3 });
        var mask = np.array(new[] { false, false, false });

        var result = a[mask];

        Assert.AreEqual(1, result.ndim);
        Assert.AreEqual(0, result.shape[0]);
    }

    [Test]
    [OpenBugs]  // Boolean row selection may fail
    public void BooleanMask_2D_RowSelection()
    {
        // NumPy: arr2d[[T, F, T]] selects rows 0 and 2 -> [[1,2,3], [7,8,9]]
        var a = np.array(new[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } });
        var mask = np.array(new[] { true, false, true });

        var result = a[mask];

        Assert.AreEqual(2, result.shape[0]);
        Assert.AreEqual(3, result.shape[1]);
        Assert.AreEqual(1, result.GetInt32(0, 0));
        Assert.AreEqual(7, result.GetInt32(1, 0));
    }

    [Test]
    [OpenBugs]  // 2D boolean masking doesn't flatten correctly
    public void BooleanMask_2D_Flattens()
    {
        // NumPy: 2D mask flattens result: [[T,F],[F,T]] on [[1,2],[3,4]] -> [1, 4]
        var a = np.array(new[,] { { 1, 2 }, { 3, 4 } });
        var mask = np.array(new[,] { { true, false }, { false, true } });

        var result = a[mask];

        Assert.AreEqual(1, result.ndim);
        result.Should().BeOfValues(1, 4);
    }

    [Test]
    public void BooleanMask_Float()
    {
        // NumPy: a[a > 2.0] = [2.5, 3.5, 4.5]
        var a = np.array(new[] { 1.5, 2.5, 3.5, 4.5 });
        var mask = a > 2.0;

        var result = a[mask];

        result.Should().BeOfValues(2.5, 3.5, 4.5);
    }

    [Test]
    public void BooleanMask_Large_SIMDPath()
    {
        // NumPy: Large array (tests SIMD path)
        var a = np.arange(10000);
        var mask = (a % 1000) == 0;

        var result = a[mask];

        result.Should().BeOfValues(0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000);
    }

    [Test]
    [OpenBugs]  // Boolean masking with explicit mask array returns all elements
    public void BooleanMask_Int16_PreservesDtype()
    {
        // NumPy: Result preserves dtype
        var a = np.array(new short[] { 1, 2, 3 });
        var mask = np.array(new[] { true, false, true });

        var result = a[mask];

        Assert.AreEqual(NPTypeCode.Int16, result.typecode);
        result.Should().BeOfValues((short)1, (short)3);
    }

    [Test]
    public void BooleanMask_ComplexCondition()
    {
        // NumPy: a[(a > 3) & (a < 8)] = [4, 5, 6, 7]
        var a = np.array(new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
        var mask = (a > 3) & (a < 8);

        var result = a[mask];

        result.Should().BeOfValues(4, 5, 6, 7);
    }

    [Test]
    [OpenBugs]  // Boolean masking with explicit mask array returns all elements
    public void BooleanMask_FromNumPyTest_Basic()
    {
        // From NumPy test_mask: x = [1, 2, 3, 4], m = [F, T, F, F] -> [2]
        var x = np.array(new[] { 1, 2, 3, 4 });
        var m = np.array(new[] { false, true, false, false });

        var result = x[m];

        result.Should().BeOfValues(2);
    }

    [Test]
    [OpenBugs]  // 2D row mask may fail
    public void BooleanMask_FromNumPyTest_2D_RowMask()
    {
        // From NumPy test_mask2: x = [[1,2,3,4],[5,6,7,8]], m = [F, T] -> [[5,6,7,8]]
        var x = np.array(new[,] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } });
        var m = np.array(new[] { false, true });

        var result = x[m];

        Assert.AreEqual(1, result.shape[0]);
        Assert.AreEqual(4, result.shape[1]);
        result["0, :"].Should().BeOfValues(5, 6, 7, 8);
    }

    [Test]
    [OpenBugs]  // 2D element mask doesn't flatten correctly
    public void BooleanMask_FromNumPyTest_2D_ElementMask()
    {
        // From NumPy test_mask2: 2D element mask flattens
        // x = [[1,2,3,4],[5,6,7,8]], m2 = [[F,T,F,F],[T,F,F,F]] -> [2, 5]
        var x = np.array(new[,] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } });
        var m2 = np.array(new[,] { { false, true, false, false }, { true, false, false, false } });

        var result = x[m2];

        result.Should().BeOfValues(2, 5);
    }

    [Test]
    [OpenBugs]  // Boolean masking with explicit mask array returns all elements
    public void BooleanMask_UInt8()
    {
        // NumPy: uint8 dtype preserved
        var a = np.array(new byte[] { 10, 20, 30, 40, 50 });
        var mask = np.array(new[] { true, false, true, false, true });

        var result = a[mask];

        Assert.AreEqual(NPTypeCode.Byte, result.typecode);
        result.Should().BeOfValues((byte)10, (byte)30, (byte)50);
    }

    [Test]
    public void BooleanMask_Int32_Condition()
    {
        // NumPy: a[a > 250] = [300, 400, 500]
        var a = np.array(new[] { 100, 200, 300, 400, 500 });
        var mask = a > 250;

        var result = a[mask];

        result.Should().BeOfValues(300, 400, 500);
    }

    [Test]
    public void BooleanMask_Float64_Condition()
    {
        // NumPy: a[a < 3.5] = [1.1, 2.2, 3.3]
        var a = np.array(new[] { 1.1, 2.2, 3.3, 4.4, 5.5 });
        var mask = a < 3.5;

        var result = a[mask];

        result.Should().BeOfValuesApproximately(0.001, 1.1, 2.2, 3.3);
    }

    [Test]
    public void BooleanMask_EvenNumbers()
    {
        // NumPy: a[a % 2 == 0] = [2, 4, 6]
        var a = np.array(new[] { 1, 2, 3, 4, 5, 6 });

        var result = a[a % 2 == 0];

        result.Should().BeOfValues(2, 4, 6);
    }

    [Test]
    public void BooleanMask_2D_Condition_Flattens()
    {
        // NumPy: arr2d[arr2d > 5] = [6, 7, 8, 9, 10, 11] (flattened)
        var arr = np.arange(12).reshape(3, 4);

        var result = arr[arr > 5];

        Assert.AreEqual(1, result.ndim);
        Assert.AreEqual(6, result.size);
        result.Should().BeOfValues(6, 7, 8, 9, 10, 11);
    }

    #endregion
}
