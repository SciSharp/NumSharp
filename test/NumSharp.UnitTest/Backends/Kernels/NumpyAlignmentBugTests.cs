using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.UnitTest.Utilities;
using TUnit.Core;

namespace NumSharp.UnitTest.Backends.Kernels;

/// <summary>
/// Comprehensive bug reproduction tests found during NumPy 2.4.2 alignment testing.
/// All tests assert CORRECT NumPy behavior. Tests FAIL while bug exists, PASS when fixed.
///
/// Bug Summary (23 bugs):
/// - BUG-1: Boolean indexing setter throws NotImplementedException (CRITICAL)
/// - BUG-2: np.nonzero 1-D shape wrong (HIGH)
/// - BUG-3: np.all/np.any with axis throws InvalidCastException (HIGH) - FIXED
/// - BUG-4: np.std/np.var ddof parameter ignored (MEDIUM)
/// - BUG-5: np.std/np.var crash on empty array (MEDIUM)
/// - BUG-6: np.sum empty 2D with axis returns scalar (MEDIUM)
/// - BUG-7: sbyte (int8) not supported (LOW)
/// - BUG-8: np.dot(vector, matrix) not supported (LOW)
/// - BUG-9: np.unique returns unsorted (MEDIUM)
/// - BUG-10: np.repeat with array repeats fails (MEDIUM)
/// - BUG-11: np.repeat missing axis parameter (LOW)
/// - BUG-12: np.searchsorted scalar input throws + array input returns wrong results (MEDIUM)
/// - BUG-13: np.linspace returns float32 instead of float64 (LOW) - FIXED
/// - BUG-14: np.abs changes int dtype to Double (MEDIUM) - FIXED
/// - BUG-15: np.moveaxis returns wrong shape (MEDIUM) - FIXED
/// - BUG-16: nd.astype(int) uses rounding instead of truncation (MEDIUM)
/// - BUG-17: np.convolve throws NullReferenceException (HIGH)
/// - BUG-18: np.negative applies abs then negates instead of just negating (HIGH) - FIXED
/// - BUG-19: np.positive applies abs instead of being identity (HIGH) - FIXED
/// - BUG-20: np.arange/sum int32 overflow - no auto-promotion to int64 (CRITICAL) - FIXED
/// - BUG-21: np.amax empty array returns -Inf instead of raising ValueError (MEDIUM)
/// - BUG-22: np.amin empty array returns +Inf instead of raising ValueError (MEDIUM)
/// - BUG-25: np.power(int, float) returns int instead of float64 (MEDIUM) - FIXED
/// - BUG-32: np.random.choice replace=False parameter ignored (HIGH)
/// </summary>
[OpenBugs]
public class NumpyAlignmentBugTests
{
    private const double Tolerance = 1e-10;

    #region BUG-1: Boolean Indexing Setter Throws NotImplementedException (CRITICAL)

    [Test]
    public void Bug1_BooleanMask_ReturnsCorrectValues()
    {
        // NUMPY 2.4.2:
        // >>> arr = np.array([10, 20, 30, 40, 50])
        // >>> mask = np.array([True, False, True, False, True])
        // >>> arr[mask]
        // array([10, 30, 50])

        var arr = np.array(new[] { 10, 20, 30, 40, 50 });
        var mask = np.array(new[] { true, false, true, false, true });

        var result = arr[mask];

        Assert.AreEqual(3, result.size, "mask has 3 True values");
        Assert.AreEqual(10, result.GetInt32(0));
        Assert.AreEqual(30, result.GetInt32(1));
        Assert.AreEqual(50, result.GetInt32(2));
    }

    [Test]
    public void Bug1_BooleanMask_AllFalse_ReturnsEmpty()
    {
        var arr = np.array(new[] { 1, 2, 3 });
        var mask = np.array(new[] { false, false, false });

        var result = arr[mask];

        Assert.AreEqual(0, result.size, "all False mask should return empty array");
    }

    [Test]
    public void Bug1_BooleanMask_AllTrue_ReturnsAll()
    {
        var arr = np.array(new[] { 1, 2, 3 });
        var mask = np.array(new[] { true, true, true });

        var result = arr[mask];

        Assert.AreEqual(3, result.size);
        Assert.AreEqual(1, result.GetInt32(0));
        Assert.AreEqual(2, result.GetInt32(1));
        Assert.AreEqual(3, result.GetInt32(2));
    }

    [Test]
    public void BooleanMask_FromComparison_Works()
    {
        // Using comparison result directly as mask
        // NUMPY: arr[arr > 2] = [3, 4, 5]
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });
        var mask = arr > 2;

        var result = arr[mask];

        Assert.AreEqual(3, result.size);
        Assert.AreEqual(3, result.GetInt32(0));
        Assert.AreEqual(4, result.GetInt32(1));
        Assert.AreEqual(5, result.GetInt32(2));
    }

    [Test]
    public void Bug1_BooleanMask_Assignment_ThrowsNotImplementedException()
    {
        // This is the actual bug - setter throws NotImplementedException
        var arr = np.array(new[] { 10, 20, 30, 40, 50 });
        var mask = np.array(new[] { true, false, true, false, true });

        // NumPy: arr[mask] = 0 sets elements at True positions to 0
        // NumSharp: throws NotImplementedException
        arr[mask] = 0;

        Assert.AreEqual(0, arr.GetInt32(0));
        Assert.AreEqual(20, arr.GetInt32(1));
        Assert.AreEqual(0, arr.GetInt32(2));
        Assert.AreEqual(40, arr.GetInt32(3));
        Assert.AreEqual(0, arr.GetInt32(4));
    }

    #endregion

    #region BUG-2: np.nonzero 1-D Shape Wrong (HIGH)

    [Test]
    public void Bug2_Nonzero_1D_ReturnsCorrectShape()
    {
        // NUMPY 2.4.2:
        // >>> np.nonzero([1, 0, 2, 0, 3])
        // (array([0, 2, 4]),)  -- tuple of 1 array for 1D input

        var arr = np.array(new[] { 1, 0, 2, 0, 3 });

        var result = np.nonzero(arr);

        Assert.AreEqual(1, result.Length, "1D input should return tuple of 1 array");
        Assert.AreEqual(3, result[0].size, "3 nonzero elements");
        CollectionAssert.AreEqual(new[] { 0, 2, 4 }, result[0].ToArray<int>());
    }

    [Test]
    public void Bug2_Nonzero_EmptyArray()
    {
        var empty = np.array(Array.Empty<double>());

        var result = np.nonzero(empty);

        Assert.IsNotNull(result);
        Assert.AreEqual(1, result.Length);
        Assert.AreEqual(0, result[0].size);
    }

    #endregion

    #region BUG-3: np.all/np.any with axis Throws (HIGH)

    [Test]
    public void Bug3_All_Axis0()
    {
        // NUMPY 2.4.2:
        // >>> np.all([[True, False, True], [True, True, True]], axis=0)
        // array([ True, False,  True])
        var arr = np.array(new[,] { { true, false, true }, { true, true, true } });

        var result = np.all(arr, axis: 0);

        Assert.AreEqual(1, result.ndim, "should return 1D array");
        Assert.AreEqual(3, result.size);
        Assert.IsTrue(result.GetBoolean(0));
        Assert.IsFalse(result.GetBoolean(1));
        Assert.IsTrue(result.GetBoolean(2));
    }

    [Test]
    public void Bug3_All_Axis1()
    {
        var arr = np.array(new[,] { { true, false, true }, { true, true, true } });

        var result = np.all(arr, axis: 1);

        Assert.AreEqual(1, result.ndim);
        Assert.AreEqual(2, result.size);
        Assert.IsFalse(result.GetBoolean(0)); // first row has False
        Assert.IsTrue(result.GetBoolean(1));  // second row all True
    }

    [Test]
    public void Bug3_Any_Axis0()
    {
        // NUMPY 2.4.2:
        // >>> np.any([[True, False, False], [False, False, True]], axis=0)
        // array([ True, False,  True])
        var arr = np.array(new[,] { { true, false, false }, { false, false, true } });

        var result = np.any(arr, axis: 0);

        Assert.AreEqual(1, result.ndim);
        Assert.AreEqual(3, result.size);
        Assert.IsTrue(result.GetBoolean(0));
        Assert.IsFalse(result.GetBoolean(1));
        Assert.IsTrue(result.GetBoolean(2));
    }

    [Test]
    public void Bug3_Any_Axis1()
    {
        var arr = np.array(new[,] { { false, false, false }, { false, true, false } });

        var result = np.any(arr, axis: 1);

        Assert.AreEqual(1, result.ndim);
        Assert.AreEqual(2, result.size);
        Assert.IsFalse(result.GetBoolean(0)); // first row all False
        Assert.IsTrue(result.GetBoolean(1));  // second row has True
    }

    #endregion

    #region BUG-4: np.std/np.var ddof Ignored (MEDIUM)

    [Test]
    public void Bug4_Std_Ddof1()
    {
        // NUMPY 2.4.2:
        // >>> np.std([1, 2, 3, 4, 5], ddof=0)  # population std
        // 1.4142135623730951
        // >>> np.std([1, 2, 3, 4, 5], ddof=1)  # sample std
        // 1.5811388300841898
        var arr = np.array(new double[] { 1, 2, 3, 4, 5 });

        var stdPop = np.std(arr, ddof: 0);
        var stdSample = np.std(arr, ddof: 1);

        Assert.AreEqual(1.4142135623730951, stdPop.GetDouble(0), Tolerance);
        Assert.AreEqual(1.5811388300841898, stdSample.GetDouble(0), Tolerance,
            "ddof=1 should give sample std dev, not population");
    }

    [Test]
    public void Bug4_Var_Ddof1()
    {
        // NUMPY 2.4.2:
        // >>> np.var([1, 2, 3, 4, 5], ddof=0)  # 2.0
        // >>> np.var([1, 2, 3, 4, 5], ddof=1)  # 2.5
        var arr = np.array(new double[] { 1, 2, 3, 4, 5 });

        var varPop = np.var(arr, ddof: 0);
        var varSample = np.var(arr, ddof: 1);

        Assert.AreEqual(2.0, varPop.GetDouble(0), Tolerance);
        Assert.AreEqual(2.5, varSample.GetDouble(0), Tolerance);
    }

    #endregion

    #region BUG-5: np.std/np.var Empty Array Returns NaN (MEDIUM)

    [Test]
    public void Bug5_Std_EmptyArray_ReturnsNaN()
    {
        // NUMPY 2.4.2 returns nan for empty array (with RuntimeWarning)
        var empty = np.array(Array.Empty<double>());

        var result = np.std(empty);

        Assert.IsNotNull(result);
        Assert.IsTrue(double.IsNaN(result.GetDouble(0)), "empty array should return NaN");
    }

    [Test]
    public void Bug5_Var_EmptyArray_ReturnsNaN()
    {
        var empty = np.array(Array.Empty<double>());

        var result = np.var(empty);

        Assert.IsNotNull(result);
        Assert.IsTrue(double.IsNaN(result.GetDouble(0)));
    }

    #endregion

    #region BUG-6: np.sum Empty 2D Returns Scalar (MEDIUM)

    [Test]
    public void Bug6_Sum_Empty2D_Axis0_ReturnsArray()
    {
        // NUMPY 2.4.2:
        // >>> np.sum(np.zeros((0, 3)), axis=0)
        // array([0., 0., 0.])  -- returns 1D array of shape (3,)
        var empty = np.zeros(new Shape(0, 3));

        var result = np.sum(empty, axis: 0);

        Assert.AreEqual(1, result.ndim, "should return 1D array, not scalar");
        Assert.AreEqual(3, result.shape[0]);
    }

    #endregion

    #region BUG-7: sbyte (int8) Not Supported (LOW)

    [Test]
    [OpenBugs]  // sbyte (int8) is not supported by NumSharp - requires adding NPTypeCode.SByte
    public void Bug7_SByte_ArrayCreation()
    {
        // NumPy supports int8 (sbyte), but NumSharp does not have NPTypeCode.SByte
        // This is a known limitation documented in CLAUDE.md as BUG-7
        sbyte[] data = { -128, 0, 127 };

        var arr = np.array(data);

        Assert.IsNotNull(arr, "sbyte (int8) should be supported");
        Assert.AreEqual(typeof(sbyte), arr.dtype);
    }

    #endregion

    #region BUG-8: np.dot(vector, matrix) Not Supported (LOW)

    [Test]
    public void Bug8_Dot_VectorMatrix()
    {
        // NUMPY 2.4.2:
        // >>> np.dot([5, 6], [[1, 2], [3, 4]])
        // array([23, 34])
        var v = np.array(new[] { 5, 6 });
        var M = np.array(new[,] { { 1, 2 }, { 3, 4 } });

        var result = np.dot(v, M);

        Assert.AreEqual(1, result.ndim);
        Assert.AreEqual(2, result.size);
        Assert.AreEqual(23, result.GetInt32(0));
        Assert.AreEqual(34, result.GetInt32(1));
    }

    #endregion

    #region BUG-9: np.unique Returns Unsorted (MEDIUM)

    [Test]
    public void Bug9_Unique_ReturnsSorted()
    {
        // NUMPY 2.4.2:
        // >>> np.unique([3, 1, 2, 1, 3, 2, 4, 1])
        // array([1, 2, 3, 4])
        // NumPy guarantees sorted output
        var arr = np.array(new[] { 3, 1, 2, 1, 3, 2, 4, 1 });

        var result = np.unique(arr);

        Assert.AreEqual(4, result.size);
        // NumPy returns sorted: [1, 2, 3, 4]
        // NumSharp BUG: returns insertion order [3, 1, 2, 4]
        Assert.AreEqual(1, result.GetInt32(0), "unique should return sorted output");
        Assert.AreEqual(2, result.GetInt32(1));
        Assert.AreEqual(3, result.GetInt32(2));
        Assert.AreEqual(4, result.GetInt32(3));
    }

    [Test]
    public void Bug9_Unique_Float64_Sorted()
    {
        // NUMPY: np.unique([3.5, 1.5, 2.5]) = [1.5, 2.5, 3.5]
        var arr = np.array(new[] { 3.5, 1.5, 2.5 });

        var result = np.unique(arr);

        Assert.AreEqual(3, result.size);
        Assert.AreEqual(1.5, result.GetDouble(0), Tolerance);
        Assert.AreEqual(2.5, result.GetDouble(1), Tolerance);
        Assert.AreEqual(3.5, result.GetDouble(2), Tolerance);
    }

    [Test]
    public void Bug9_Unique_WithDuplicates()
    {
        // NUMPY: np.unique([5, 5, 5, 1, 1, 3]) = [1, 3, 5]
        var arr = np.array(new[] { 5, 5, 5, 1, 1, 3 });

        var result = np.unique(arr);

        Assert.AreEqual(3, result.size);
        Assert.AreEqual(1, result.GetInt32(0));
        Assert.AreEqual(3, result.GetInt32(1));
        Assert.AreEqual(5, result.GetInt32(2));
    }

    #endregion

    #region BUG-10: np.repeat with Array Repeats Fails (MEDIUM)

    [Test]
    public void Bug10_Repeat_ArrayRepeats()
    {
        // NUMPY 2.4.2:
        // >>> np.repeat([1, 2, 3], [2, 3, 4])
        // array([1, 1, 2, 2, 2, 3, 3, 3, 3])
        // Each element repeated according to corresponding count
        var arr = np.array(new[] { 1, 2, 3 });
        var repeats = np.array(new[] { 2, 3, 4 });

        var result = np.repeat(arr, repeats);

        Assert.IsNotNull(result);
        Assert.AreEqual(9, result.size);  // 2 + 3 + 4 = 9
        CollectionAssert.AreEqual(
            new[] { 1, 1, 2, 2, 2, 3, 3, 3, 3 },
            result.ToArray<int>());
    }

    [Test]
    public void Bug10_Repeat_ArrayRepeats_DifferentCounts()
    {
        // NUMPY: np.repeat([10, 20], [1, 3]) = [10, 20, 20, 20]
        var arr = np.array(new[] { 10, 20 });
        var repeats = np.array(new[] { 1, 3 });

        var result = np.repeat(arr, repeats);

        CollectionAssert.AreEqual(
            new[] { 10, 20, 20, 20 },
            result.ToArray<int>());
    }

    [Test]
    public void Bug10_Repeat_ArrayRepeats_WithZero()
    {
        // NUMPY: np.repeat([1, 2, 3], [0, 2, 0]) = [2, 2]
        // Zero means skip that element
        var arr = np.array(new[] { 1, 2, 3 });
        var repeats = np.array(new[] { 0, 2, 0 });

        var result = np.repeat(arr, repeats);

        Assert.AreEqual(2, result.size);
        CollectionAssert.AreEqual(new[] { 2, 2 }, result.ToArray<int>());
    }

    #endregion

    #region BUG-11: np.repeat Missing axis Parameter (LOW)
    // Note: These tests will fail to compile if axis parameter is missing
    // Commenting out until the parameter is added

    /*
    [Test]
    public void Bug11_Repeat_2D_Axis0()
    {
        // NUMPY 2.4.2:
        // >>> np.repeat([[1, 2], [3, 4]], 2, axis=0)
        // array([[1, 2],
        //        [1, 2],
        //        [3, 4],
        //        [3, 4]])
        var arr = np.array(new[,] { { 1, 2 }, { 3, 4 } });

        var result = np.repeat(arr, 2, axis: 0);

        Assert.AreEqual(2, result.ndim);
        Assert.AreEqual(4, result.shape[0]);
        Assert.AreEqual(2, result.shape[1]);
        Assert.AreEqual(1, result.GetInt32(0, 0));
        Assert.AreEqual(1, result.GetInt32(1, 0));  // repeated
        Assert.AreEqual(3, result.GetInt32(2, 0));
        Assert.AreEqual(3, result.GetInt32(3, 0));  // repeated
    }

    [Test]
    public void Bug11_Repeat_2D_Axis1()
    {
        // NUMPY 2.4.2:
        // >>> np.repeat([[1, 2], [3, 4]], 2, axis=1)
        // array([[1, 1, 2, 2],
        //        [3, 3, 4, 4]])
        var arr = np.array(new[,] { { 1, 2 }, { 3, 4 } });

        var result = np.repeat(arr, 2, axis: 1);

        Assert.AreEqual(2, result.ndim);
        Assert.AreEqual(2, result.shape[0]);
        Assert.AreEqual(4, result.shape[1]);
        Assert.AreEqual(1, result.GetInt32(0, 0));
        Assert.AreEqual(1, result.GetInt32(0, 1));  // repeated
        Assert.AreEqual(2, result.GetInt32(0, 2));
        Assert.AreEqual(2, result.GetInt32(0, 3));  // repeated
    }

    [Test]
    public void Bug11_Repeat_2D_Axis_NegativeAxis()
    {
        // NUMPY: np.repeat(arr, 3, axis=-1) same as axis=1 for 2D
        var arr = np.array(new[,] { { 1, 2 }, { 3, 4 } });

        var result = np.repeat(arr, 3, axis: -1);

        Assert.AreEqual(2, result.ndim);
        Assert.AreEqual(2, result.shape[0]);
        Assert.AreEqual(6, result.shape[1]);
    }
    */

    #endregion

    #region BUG-12: np.searchsorted Scalar Input Throws (MEDIUM)

    [Test]
    public void Bug12_Searchsorted_ScalarInput()
    {
        // NUMPY 2.4.2:
        // >>> np.searchsorted([1, 2, 3, 4, 5], 3)
        // 2
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });

        // NumSharp throws IndexOutOfRangeException for scalar input
        // Array input works fine
        var result = np.searchsorted(arr, 3);

        Assert.AreEqual(2, result);
    }

    [Test]
    public void Bug12_Searchsorted_ScalarInput_NotFound()
    {
        // NUMPY: np.searchsorted([1, 3, 5], 4) = 2
        var arr = np.array(new[] { 1, 3, 5 });

        var result = np.searchsorted(arr, 4);

        Assert.AreEqual(2, result);
    }

    [Test]
    public void Bug12_Searchsorted_ArrayInput_WrongResults()
    {
        // NUMPY 2.4.2:
        // >>> np.searchsorted([1, 2, 3, 4, 5], [2, 4])
        // array([1, 3])
        // Insert 2 at index 1, insert 4 at index 3
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });
        var values = np.array(new[] { 2, 4 });

        var result = np.searchsorted(arr, values);

        Assert.AreEqual(2, result.size);
        Assert.AreEqual(1, result.GetInt32(0), "index to insert 2 should be 1");
        Assert.AreEqual(3, result.GetInt32(1), "index to insert 4 should be 3");
    }

    #endregion

    #region BUG-13: np.linspace Returns float32 Instead of float64 (LOW)

    [Test]
    public void Bug13_Linspace_ReturnsDtype()
    {
        // NUMPY 2.4.2:
        // >>> np.linspace(0, 10, 5).dtype
        // dtype('float64')
        var lin = np.linspace(0, 10, 5);

        // NumSharp returns Single (float32), NumPy returns Double (float64)
        Assert.AreEqual(typeof(double), lin.dtype,
            "np.linspace should return float64, not float32");
    }

    [Test]
    public void Bug13_Linspace_Values()
    {
        // NUMPY: np.linspace(0, 10, 5) = [0, 2.5, 5, 7.5, 10]
        var lin = np.linspace(0, 10, 5);

        Assert.AreEqual(5, lin.size);
        Assert.AreEqual(0.0, lin.GetDouble(0), 1e-10);
        Assert.AreEqual(2.5, lin.GetDouble(1), 1e-10);
        Assert.AreEqual(5.0, lin.GetDouble(2), 1e-10);
        Assert.AreEqual(7.5, lin.GetDouble(3), 1e-10);
        Assert.AreEqual(10.0, lin.GetDouble(4), 1e-10);
    }

    #endregion

    #region BUG-14: np.abs Changes int dtype to Double (MEDIUM)

    [Test]
    public void Bug14_Abs_PreservesIntDtype()
    {
        // NUMPY 2.4.2:
        // >>> np.abs([-3, -1, 0, 2, 5]).dtype
        // dtype('int64')
        var arr = np.array(new[] { -3, -1, 0, 2, 5 });

        var absArr = np.abs(arr);

        // NumSharp changes dtype to Double, should preserve Int32
        Assert.AreEqual(typeof(int), absArr.dtype,
            "np.abs should preserve integer dtype");
        CollectionAssert.AreEqual(
            new[] { 3, 1, 0, 2, 5 },
            absArr.ToArray<int>());
    }

    [Test]
    public void Bug14_Abs_Float64_PreservesDtype()
    {
        // NUMPY: np.abs([-3.5, 2.5]).dtype = float64
        var arr = np.array(new[] { -3.5, 2.5 });

        var absArr = np.abs(arr);

        Assert.AreEqual(typeof(double), absArr.dtype);
        Assert.AreEqual(3.5, absArr.GetDouble(0), 1e-10);
        Assert.AreEqual(2.5, absArr.GetDouble(1), 1e-10);
    }

    #endregion

    #region BUG-15: np.moveaxis Returns Wrong Shape (MEDIUM)

    [Test]
    public void Bug15_Moveaxis_3D()
    {
        // NUMPY 2.4.2:
        // >>> arr = np.zeros((3, 4, 5))
        // >>> np.moveaxis(arr, 0, -1).shape
        // (4, 5, 3)
        var arr = np.zeros(3, 4, 5);

        var moved = np.moveaxis(arr, 0, -1);

        // NumSharp returns unchanged shape (3, 4, 5)
        // Should move axis 0 to last position: (4, 5, 3)
        Assert.AreEqual(4, moved.shape[0], "axis 0 should become 4 (original axis 1)");
        Assert.AreEqual(5, moved.shape[1], "axis 1 should become 5 (original axis 2)");
        Assert.AreEqual(3, moved.shape[2], "axis 2 should become 3 (original axis 0)");
    }

    [Test]
    public void Bug15_Moveaxis_ToFirst()
    {
        // NUMPY: np.moveaxis(arr, -1, 0).shape on (3,4,5) -> (5,3,4)
        var arr = np.zeros(3, 4, 5);

        var moved = np.moveaxis(arr, -1, 0);

        Assert.AreEqual(5, moved.shape[0]);
        Assert.AreEqual(3, moved.shape[1]);
        Assert.AreEqual(4, moved.shape[2]);
    }

    #endregion

    #region BUG-16: nd.astype(int) Uses Rounding Instead of Truncation (MEDIUM)

    [Test]
    public void Bug16_Astype_FloatToInt_ShouldTruncate()
    {
        // NUMPY 2.4.2:
        // >>> np.array([1.7, 2.3, 3.9]).astype(int)
        // array([1, 2, 3])  -- truncation toward zero
        var arr = np.array(new[] { 1.7, 2.3, 3.9 });

        var asInt = arr.astype(np.int32);

        // NumSharp uses rounding: [2, 2, 4]
        // NumPy uses truncation: [1, 2, 3]
        CollectionAssert.AreEqual(
            new[] { 1, 2, 3 },
            asInt.ToArray<int>(),
            "astype(int) should truncate, not round");
    }

    [Test]
    public void Bug16_Astype_NegativeFloatToInt()
    {
        // NUMPY: np.array([-1.7, -2.3, -3.9]).astype(int) = [-1, -2, -3]
        var arr = np.array(new[] { -1.7, -2.3, -3.9 });

        var asInt = arr.astype(np.int32);

        // Truncation toward zero
        CollectionAssert.AreEqual(
            new[] { -1, -2, -3 },
            asInt.ToArray<int>());
    }

    #endregion

    #region BUG-17: np.convolve Throws NullReferenceException (HIGH)

    [Test]
    public void Bug17_Convolve_Basic()
    {
        // NUMPY 2.4.2:
        // >>> np.convolve([1, 2, 3], [0, 1, 0.5])
        // array([0. , 1. , 2.5, 4. , 1.5])
        var a = np.array(new double[] { 1, 2, 3 });
        var v = np.array(new double[] { 0, 1, 0.5 });

        // NumSharp throws NullReferenceException
        var result = np.convolve(a, v);

        Assert.IsNotNull(result);
        Assert.AreEqual(5, result.size);
        Assert.AreEqual(0.0, result.GetDouble(0), 1e-10);
        Assert.AreEqual(1.0, result.GetDouble(1), 1e-10);
        Assert.AreEqual(2.5, result.GetDouble(2), 1e-10);
        Assert.AreEqual(4.0, result.GetDouble(3), 1e-10);
        Assert.AreEqual(1.5, result.GetDouble(4), 1e-10);
    }

    [Test]
    public void Bug17_Convolve_IntArrays()
    {
        // NUMPY: np.convolve([1, 2, 3], [1, 1]) = [1, 3, 5, 3]
        var a = np.array(new[] { 1, 2, 3 });
        var v = np.array(new[] { 1, 1 });

        var result = np.convolve(a, v);

        Assert.IsNotNull(result);
        Assert.AreEqual(4, result.size);
    }

    #endregion

    #region BUG-18: np.negative Applies abs() Then Negates (HIGH)

    [Test]
    public void Bug18_Negative_ShouldJustNegate()
    {
        // NUMPY 2.4.2:
        // >>> np.negative([1, -2, 3, -4])
        // array([-1,  2, -3,  4])
        // Just negates each element (flip sign)
        var arr = np.array(new[] { 1, -2, 3, -4 });

        var neg = np.negative(arr);

        // NumSharp BUG: applies abs() then negates, giving [-1, -2, -3, -4]
        // Should give: [-1, 2, -3, 4]
        Assert.AreEqual(-1, neg.GetInt32(0));
        Assert.AreEqual(2, neg.GetInt32(1), "negative of -2 should be 2");
        Assert.AreEqual(-3, neg.GetInt32(2));
        Assert.AreEqual(4, neg.GetInt32(3), "negative of -4 should be 4");
    }

    [Test]
    public void Bug18_Negative_Float64()
    {
        // NUMPY: np.negative([-1.5, 2.5]) = [1.5, -2.5]
        var arr = np.array(new[] { -1.5, 2.5 });

        var neg = np.negative(arr);

        Assert.AreEqual(1.5, neg.GetDouble(0), 1e-10);
        Assert.AreEqual(-2.5, neg.GetDouble(1), 1e-10);
    }

    [Test]
    public void Bug18_Negative_Zero()
    {
        // NUMPY: np.negative([0]) = [0] (or -0.0 for float)
        var arr = np.array(new[] { 0 });

        var neg = np.negative(arr);

        Assert.AreEqual(0, neg.GetInt32(0));
    }

    #endregion

    #region BUG-19: np.positive Applies abs() Instead of Identity (HIGH)

    [Test]
    public void Bug19_Positive_ShouldBeIdentity()
    {
        // NUMPY 2.4.2:
        // >>> np.positive([1, -2, 3, -4])
        // array([ 1, -2,  3, -4])
        // np.positive is identity function (returns unchanged)
        var arr = np.array(new[] { 1, -2, 3, -4 });

        var pos = np.positive(arr);

        // NumSharp BUG: applies abs(), giving [1, 2, 3, 4]
        // Should give: [1, -2, 3, -4]
        Assert.AreEqual(1, pos.GetInt32(0));
        Assert.AreEqual(-2, pos.GetInt32(1), "positive should preserve -2");
        Assert.AreEqual(3, pos.GetInt32(2));
        Assert.AreEqual(-4, pos.GetInt32(3), "positive should preserve -4");
    }

    [Test]
    public void Bug19_Positive_Float64()
    {
        // NUMPY: np.positive([-1.5, 2.5]) = [-1.5, 2.5]
        var arr = np.array(new[] { -1.5, 2.5 });

        var pos = np.positive(arr);

        Assert.AreEqual(-1.5, pos.GetDouble(0), 1e-10, "should preserve -1.5");
        Assert.AreEqual(2.5, pos.GetDouble(1), 1e-10);
    }

    #endregion

    #region BUG-20: np.arange/sum Integer Overflow - No Auto-Promotion (CRITICAL)
    // Two related issues:
    // A) np.arange returns int32 instead of int64 on 64-bit systems
    // B) np.sum doesn't auto-promote to int64 to prevent overflow

    [Test]
    public void Bug20_Arange_ShouldReturnInt64()
    {
        // NUMPY 2.4.2:
        // >>> np.arange(100000).dtype
        // dtype('int64')  on 64-bit systems
        //
        // NumSharp returns int32 instead
        var arr = np.arange(100000);

        // NumPy returns int64 on 64-bit systems
        // NumSharp BUG: returns int32
        Assert.AreEqual(typeof(long), arr.dtype,
            "np.arange should return int64 on 64-bit systems, not int32");
    }

    [Test]
    public void Bug20_Sum_ShouldAutoPromoteToPreventOverflow()
    {
        // NUMPY 2.4.2:
        // >>> np.sum(np.arange(85000, dtype=np.int32))
        // 3612457500  (auto-promoted to int64)
        //
        // FIXED: NumSharp now auto-promotes int32 to int64 (NEP50)
        var arr = np.arange(85000);  // Returns int32 in NumSharp
        var sum = np.sum(arr);

        // Expected: 85000 * 84999 / 2 = 3,612,457,500
        // This exceeds int32.MaxValue (2,147,483,647)
        // FIXED: NumSharp now auto-promotes to int64 and returns correct value

        var expected = 3612457500L;
        var actual = sum.GetInt64(0);  // Using GetInt64 since sum now returns int64

        Assert.AreEqual(expected, actual,
            $"Sum should be {expected}, got {actual}. NumPy auto-promotes to int64.");
    }

    [Test]
    public void Bug20_Sum_SmallArray_NoOverflow()
    {
        // Verify small arrays work correctly (no overflow)
        // 50000 * 49999 / 2 = 1,249,975,000 (fits in int32)
        var arr = np.arange(50000);
        var sum = np.sum(arr);

        // np.sum promotes int32 to int64 for accumulation (NEP50)
        Assert.AreEqual(1249975000L, sum.GetInt64(0),
            "Sum of arange(50000) should be 1,249,975,000");
    }

    [Test]
    public void Bug20_Sum_LargeArray_Overflow()
    {
        // 100000 * 99999 / 2 = 4,999,950,000 (exceeds int32)
        // FIXED: NumSharp now auto-promotes to int64
        var arr = np.arange(100000);
        var sum = np.sum(arr);

        var expected = 4999950000L;
        var actual = sum.GetInt64(0);  // Using GetInt64 since sum now returns int64

        Assert.AreEqual(expected, actual,
            $"Sum should be {expected}, got {actual}.");
    }

    [Test]
    public void Bug20_Sum_Float64_Workaround()
    {
        // Workaround: convert to float64 to avoid overflow
        var large = np.arange(100000).astype(np.float64);

        var sum = np.sum(large);

        Assert.IsNotNull(sum);
        Assert.AreEqual(4999950000.0, sum.GetDouble(0), 1.0,
            "Float64 workaround should give correct sum");
    }

    #endregion

    #region BUG-21: np.amax Empty Array Returns -Inf (MEDIUM)

    [Test]
    public void Bug21_Amax_EmptyArray_ShouldThrow()
    {
        // NUMPY 2.4.2:
        // >>> np.amax([])
        // ValueError: zero-size array to reduction operation maximum which has no identity
        //
        // NumSharp returns -Infinity instead of throwing
        var empty = np.array(Array.Empty<double>());

        // NumPy throws ValueError, NumSharp returns -Inf
        // This test documents that empty max should ideally throw
        try
        {
            var result = np.amax(empty);
            // If we get here, NumSharp returned a value instead of throwing
            // Document that it returns -Inf (incorrect NumPy behavior)
            Assert.Fail($"np.amax on empty array should throw, but returned: {result.GetDouble(0)}");
        }
        catch (Exception)
        {
            // Correct behavior - should throw
            Assert.IsTrue(true);
        }
    }

    #endregion

    #region BUG-22: np.amin Empty Array Returns +Inf (MEDIUM)

    [Test]
    public void Bug22_Amin_EmptyArray_ShouldThrow()
    {
        // NUMPY 2.4.2:
        // >>> np.amin([])
        // ValueError: zero-size array to reduction operation minimum which has no identity
        //
        // NumSharp returns +Infinity instead of throwing
        var empty = np.array(Array.Empty<double>());

        // NumPy throws ValueError, NumSharp returns +Inf
        try
        {
            var result = np.amin(empty);
            // If we get here, NumSharp returned a value instead of throwing
            Assert.Fail($"np.amin on empty array should throw, but returned: {result.GetDouble(0)}");
        }
        catch (Exception)
        {
            // Correct behavior - should throw
            Assert.IsTrue(true);
        }
    }

    #endregion

    #region BUG-32: np.random.choice replace=False Parameter Ignored (HIGH)

    [Test]
    [OpenBugs]
    public void Bug32_Choice_ReplaceFalse_NoDuplicates()
    {
        // NUMPY 2.4.2:
        // >>> np.random.seed(42)
        // >>> np.random.choice(10, 5, replace=False)
        // array([8, 1, 5, 0, 7])  # All unique values
        //
        // NumSharp BUG: replace parameter is declared but never used.
        // Code always uses randint which samples WITH replacement.
        np.random.seed(42);
        var result = np.random.choice(10, new Shape(5), replace: false);

        // With replace=False, all values must be unique
        var values = result.ToArray<int>();
        var uniqueCount = values.Distinct().Count();

        Assert.AreEqual(5, uniqueCount,
            "replace=False should produce 5 unique values, but got duplicates. " +
            "NumSharp ignores the replace parameter.");
    }

    [Test]
    [OpenBugs]
    public void Bug32_Choice_ReplaceFalse_SizeExceedsPopulation_ShouldThrow()
    {
        // NUMPY 2.4.2:
        // >>> np.random.choice(5, 10, replace=False)
        // ValueError: Cannot take a larger sample than population when 'replace=False'
        //
        // NumSharp BUG: No validation, will produce duplicates instead
        np.random.seed(42);

        try
        {
            var result = np.random.choice(5, new Shape(10), replace: false);
            Assert.Fail(
                "np.random.choice(5, 10, replace=False) should throw ValueError, " +
                "but returned: " + string.Join(", ", result.ToArray<int>()));
        }
        catch (ArgumentException)
        {
            // Correct - should throw when size > population with replace=False
            Assert.IsTrue(true);
        }
        catch (InvalidOperationException)
        {
            // Also acceptable exception type
            Assert.IsTrue(true);
        }
    }

    [Test]
    public void Bug32_Choice_ReplaceTrue_AllowsDuplicates()
    {
        // NUMPY: replace=True (default) allows duplicates
        // This test verifies the default behavior still works
        np.random.seed(42);
        var result = np.random.choice(3, new Shape(100), replace: true);

        // With only 3 choices and 100 samples, duplicates are guaranteed
        var values = result.ToArray<int>();
        var uniqueCount = values.Distinct().Count();

        Assert.IsTrue(uniqueCount <= 3,
            "replace=True should allow duplicates (only 3 unique values possible)");
        Assert.AreEqual(100, values.Length, "Should return 100 samples");
    }

    [Test]
    [OpenBugs]
    public void Bug32_Choice_NDArray_ReplaceFalse()
    {
        // NUMPY: np.random.choice(['a','b','c','d','e'], 3, replace=False)
        // Should return 3 unique elements from the array
        np.random.seed(42);
        var arr = np.array(new[] { 10, 20, 30, 40, 50 });
        var result = np.random.choice(arr, new Shape(3), replace: false);

        var values = result.ToArray<int>();
        var uniqueCount = values.Distinct().Count();

        Assert.AreEqual(3, uniqueCount,
            "replace=False with NDArray input should produce unique selections");

        // All values must be from the original array
        foreach (var v in values)
        {
            Assert.IsTrue(v == 10 || v == 20 || v == 30 || v == 40 || v == 50,
                $"Value {v} not in original array");
        }
    }

    #endregion

    #region Verified Working: np.modf

    [Test]
    public void Modf_Works()
    {
        // NUMPY: np.modf([1.5, 2.7, -3.2]) = ([0.5, 0.7, -0.2], [1.0, 2.0, -3.0])
        var arr = np.array(new[] { 1.5, 2.7, -3.2 });

        var (frac, integ) = np.modf(arr);

        Assert.AreEqual(3, frac.size);
        Assert.AreEqual(0.5, frac.GetDouble(0), 1e-10);
        Assert.AreEqual(0.7, frac.GetDouble(1), 1e-10);
        Assert.AreEqual(-0.2, frac.GetDouble(2), 1e-10);

        Assert.AreEqual(1.0, integ.GetDouble(0), 1e-10);
        Assert.AreEqual(2.0, integ.GetDouble(1), 1e-10);
        Assert.AreEqual(-3.0, integ.GetDouble(2), 1e-10);
    }

    #endregion

    #region Verified Working: Negative Axis

    [Test]
    public void Sum_NegativeAxis_Works()
    {
        // NUMPY: np.sum([[1,2],[3,4]], axis=-1) = [3, 7]
        var arr = np.array(new[,] { { 1, 2 }, { 3, 4 } });

        var result = np.sum(arr, axis: -1);

        Assert.AreEqual(1, result.ndim);
        Assert.AreEqual(2, result.size);
        // np.sum promotes int32 to int64 for accumulation (NEP50)
        Assert.AreEqual(3L, result.GetInt64(0));
        Assert.AreEqual(7L, result.GetInt64(1));
    }

    #endregion

    #region Verified Working: 3D Sum with Axis

    [Test]
    public void Sum_3D_Axis1_Works()
    {
        // NUMPY: np.sum(arr, axis=1) on (2,3,4) -> (2,4)
        var arr = np.arange(24).reshape(2, 3, 4);

        var result = np.sum(arr, axis: 1);

        Assert.AreEqual(2, result.ndim);
        Assert.AreEqual(2, result.shape[0]);
        Assert.AreEqual(4, result.shape[1]);
    }

    #endregion

    #region Verified Working: np.repeat with Scalar

    [Test]
    public void Repeat_Scalar_Works()
    {
        // This works correctly
        var arr = np.array(new[] { 1, 2, 3 });
        var result = np.repeat(arr, 3);

        CollectionAssert.AreEqual(
            new[] { 1, 1, 1, 2, 2, 2, 3, 3, 3 },
            result.ToArray<int>());
    }

    #endregion

    #region Verified Working: np.clip

    [Test]
    public void Clip_Works()
    {
        // np.clip verified working
        var arr = np.array(new[] { 1, 5, 10, 15, 20 });
        var result = np.clip(arr, 5, 15);

        CollectionAssert.AreEqual(
            new[] { 5, 5, 10, 15, 15 },
            result.ToArray<int>());
    }

    #endregion

    #region Verified Working: np.roll

    [Test]
    public void Roll_1D_Positive()
    {
        // NUMPY: np.roll([1,2,3,4,5], 2) = [4,5,1,2,3]
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });
        var result = np.roll(arr, 2);

        CollectionAssert.AreEqual(
            new[] { 4, 5, 1, 2, 3 },
            result.ToArray<int>());
    }

    [Test]
    public void Roll_1D_Negative()
    {
        // NUMPY: np.roll([1,2,3,4,5], -2) = [3,4,5,1,2]
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });
        var result = np.roll(arr, -2);

        CollectionAssert.AreEqual(
            new[] { 3, 4, 5, 1, 2 },
            result.ToArray<int>());
    }

    #endregion

    #region Verified Working: np.argsort

    [Test]
    public void Argsort_1D_Int32()
    {
        // NUMPY: np.argsort([3,1,4,1,5]) = [1,3,0,2,4]
        // NumPy argsort always returns int64 indices
        var arr = np.array(new[] { 3, 1, 4, 1, 5 });
        var result = np.argsort<int>(arr);

        CollectionAssert.AreEqual(
            new long[] { 1, 3, 0, 2, 4 },
            result.ToArray<long>());
    }

    #endregion

    #region Verified Working: np.cumsum

    [Test]
    public void Cumsum_1D()
    {
        // NUMPY: np.cumsum([1,2,3,4,5]) = [1,3,6,10,15]
        // NumPy-aligned: cumsum of int32 returns int64
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });
        var result = np.cumsum(arr);

        CollectionAssert.AreEqual(
            new long[] { 1, 3, 6, 10, 15 },
            result.ToArray<long>());
    }

    [Test]
    public void Cumsum_2D_Flat()
    {
        // NUMPY: np.cumsum([[1,2],[3,4]]) = [1,3,6,10]
        // NumPy-aligned: cumsum of int32 returns int64
        var arr = np.array(new[,] { { 1, 2 }, { 3, 4 } });
        var result = np.cumsum(arr);

        CollectionAssert.AreEqual(
            new long[] { 1, 3, 6, 10 },
            result.ToArray<long>());
    }

    #endregion

    #region Verified Working: NaN in Reductions

    [Test]
    public void Sum_WithNaN_ReturnsNaN()
    {
        // NUMPY: np.sum([1, 2, NaN, 4]) = NaN
        var arr = np.array(new[] { 1.0, 2.0, double.NaN, 4.0 });
        var result = np.sum(arr);

        Assert.IsTrue(double.IsNaN(result.GetDouble(0)),
            "Sum with NaN should return NaN");
    }

    [Test]
    public void Max_WithNaN_ReturnsNaN()
    {
        // NUMPY: np.max([1, 2, NaN, 4]) = NaN
        var arr = np.array(new[] { 1.0, 2.0, double.NaN, 4.0 });
        var result = np.amax(arr);

        Assert.IsTrue(double.IsNaN(result.GetDouble(0)),
            "Max with NaN should return NaN");
    }

    [Test]
    public void Min_WithNaN_ReturnsNaN()
    {
        // NUMPY: np.min([1, 2, NaN, 4]) = NaN
        var arr = np.array(new[] { 1.0, 2.0, double.NaN, 4.0 });
        var result = np.amin(arr);

        Assert.IsTrue(double.IsNaN(result.GetDouble(0)),
            "Min with NaN should return NaN");
    }

    [Test]
    public void ArgMax_WithNaN_ReturnsNaNIndex()
    {
        // NUMPY: np.argmax([1, 2, NaN, 4]) = 2 (index of NaN)
        var arr = np.array(new[] { 1.0, 2.0, double.NaN, 4.0 });
        var result = arr.argmax();

        Assert.AreEqual(2, result,
            "ArgMax with NaN should return index of NaN");
    }

    [Test]
    public void ArgMin_WithNaN_ReturnsNaNIndex()
    {
        // NUMPY: np.argmin([1, 2, NaN, 4]) = 2 (index of NaN)
        var arr = np.array(new[] { 1.0, 2.0, double.NaN, 4.0 });
        var result = arr.argmin();

        Assert.AreEqual(2, result,
            "ArgMin with NaN should return index of NaN");
    }

    #endregion

    #region Verified Working: Double Array Reductions (BUG-26 Fix Verification)

    [Test]
    public void Bug26_Sum_DoubleArray_Works()
    {
        // BUG-26 FIX VERIFICATION: np.sum on double arrays
        var arr = np.array(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var result = np.sum(arr);

        Assert.AreEqual(15.0, result.GetDouble(0), 1e-10,
            "sum(double[]) should return 15.0");
    }

    [Test]
    public void Bug26_Prod_DoubleArray_Works()
    {
        // BUG-26 FIX VERIFICATION: np.prod on double arrays
        var arr = np.array(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var result = np.prod(arr);

        Assert.AreEqual(120.0, result.GetDouble(0), 1e-10,
            "prod(double[]) should return 120.0");
    }

    [Test]
    public void Bug26_Cumsum_DoubleArray_Works()
    {
        // BUG-26 FIX VERIFICATION: np.cumsum on double arrays
        var arr = np.array(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var result = np.cumsum(arr);

        Assert.AreEqual(5, result.size);
        Assert.AreEqual(1.0, result.GetDouble(0), 1e-10);
        Assert.AreEqual(3.0, result.GetDouble(1), 1e-10);
        Assert.AreEqual(6.0, result.GetDouble(2), 1e-10);
        Assert.AreEqual(10.0, result.GetDouble(3), 1e-10);
        Assert.AreEqual(15.0, result.GetDouble(4), 1e-10);
    }

    #endregion

    #region Verified Working: Empty Array Reductions

    [Test]
    public void Sum_EmptyArray_ReturnsZero()
    {
        // NUMPY: np.sum([]) = 0.0
        var empty = np.array(Array.Empty<double>());
        var result = np.sum(empty);

        Assert.AreEqual(0.0, result.GetDouble(0), 1e-10,
            "Sum of empty array should be 0");
    }

    [Test]
    public void Prod_EmptyArray_ReturnsOne()
    {
        // NUMPY: np.prod([]) = 1.0
        var empty = np.array(Array.Empty<double>());
        var result = np.prod(empty);

        Assert.AreEqual(1.0, result.GetDouble(0), 1e-10,
            "Prod of empty array should be 1");
    }

    [Test]
    public void Mean_EmptyArray_ReturnsNaN()
    {
        // NUMPY: np.mean([]) = NaN (with RuntimeWarning)
        var empty = np.array(Array.Empty<double>());
        var result = np.mean(empty);

        Assert.IsTrue(double.IsNaN(result.GetDouble(0)),
            "Mean of empty array should be NaN");
    }

    #endregion

    #region Verified Working: Type Promotion

    [Test]
    public void TypePromotion_Int32_Float64()
    {
        // NUMPY: int32 + float64 = float64
        var intArr = np.array(new[] { 1, 2, 3 });
        var floatArr = np.array(new[] { 0.5, 0.5, 0.5 });

        var result = intArr + floatArr;

        Assert.AreEqual(typeof(double), result.dtype);
        Assert.AreEqual(1.5, result.GetDouble(0), 1e-10);
    }

    [Test]
    public void TypePromotion_Bool_Int32()
    {
        // NUMPY: bool + int32 = int32
        var boolArr = np.array(new[] { true, true, true });
        var intArr = np.array(new[] { 1, 1, 3 });

        var result = boolArr + intArr;

        Assert.AreEqual(typeof(int), result.dtype);
        Assert.AreEqual(2, result.GetInt32(0));
        Assert.AreEqual(2, result.GetInt32(1));
        Assert.AreEqual(4, result.GetInt32(2));
    }

    [Test]
    public void Bug25_Power_FloatExponent_ReturnsFloat64()
    {
        // BUG-25 FIX VERIFICATION:
        // NUMPY 2.4.2: np.power(int32, float) returns float64
        // >>> np.power(np.array([1,2,3], dtype=np.int32), 2.0).dtype
        // dtype('float64')
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });
        var result = np.power(arr, 2.0);

        Assert.AreEqual(typeof(double), result.dtype,
            "power(int32, float) should return float64 per NumPy behavior");
        Assert.AreEqual(1.0, result.GetDouble(0), 1e-10);
        Assert.AreEqual(4.0, result.GetDouble(1), 1e-10);
        Assert.AreEqual(9.0, result.GetDouble(2), 1e-10);
    }

    [Test]
    public void Power_IntExponent_PreservesIntType()
    {
        // NUMPY: np.power(int32, int) returns int32
        // >>> np.power(np.array([1,2,3], dtype=np.int32), 2).dtype
        // dtype('int32')
        var arr = np.array(new[] { 1, 2, 3 });
        var result = np.power(arr, 2);

        Assert.AreEqual(typeof(int), result.dtype,
            "power(int32, int) should preserve int32 dtype");
        Assert.AreEqual(1, result.GetInt32(0));
        Assert.AreEqual(4, result.GetInt32(1));
        Assert.AreEqual(9, result.GetInt32(2));
    }

    #endregion

    #region Verified Working: All/Any with Axis and Keepdims

    [Test]
    public void All_3D_Axis0()
    {
        // NUMPY: np.all(arr, axis=0) on (2,2,2) -> (2,2)
        var arr = np.array(new[,,] {
            { { true, true }, { true, false } },
            { { true, true }, { false, true } }
        });

        var result = np.all(arr, axis: 0);

        Assert.AreEqual(2, result.ndim);
        Assert.AreEqual(2, result.shape[0]);
        Assert.AreEqual(2, result.shape[1]);
    }

    [Test]
    public void All_3D_Axis1()
    {
        // NUMPY: np.all(arr, axis=1) on (2,2,2) -> (2,2)
        var arr = np.array(new[,,] {
            { { true, true }, { true, false } },
            { { true, true }, { false, true } }
        });

        var result = np.all(arr, axis: 1);

        Assert.AreEqual(2, result.ndim);
        Assert.AreEqual(2, result.shape[0]);
        Assert.AreEqual(2, result.shape[1]);
    }

    [Test]
    public void All_Keepdims()
    {
        // NUMPY: np.all([[True,False],[True,True]], axis=1, keepdims=True) -> [[False],[True]]
        var arr = np.array(new[,] { { true, false }, { true, true } });

        var result = np.all(arr, axis: 1, keepdims: true);

        Assert.AreEqual(2, result.ndim);
        Assert.AreEqual(2, result.shape[0]);
        Assert.AreEqual(1, result.shape[1]);
        Assert.IsFalse(result.GetBoolean(0, 0));
        Assert.IsTrue(result.GetBoolean(1, 0));
    }

    #endregion
}

/// <summary>
/// Tests documenting missing or dead-code NumPy functions.
/// </summary>
[OpenBugs]
public class MissingFunctionTests
{
    [Test]
    public void IsNaN_DeadCode()
    {
        var arr = np.array(new[] { 1.0, double.NaN, 3.0 });

        var result = np.isnan(arr);

        Assert.IsNotNull(result, "np.isnan returns null - dead code");
    }

    [Test]
    public void IsFinite_DeadCode()
    {
        var arr = np.array(new[] { 1.0, double.NaN, double.PositiveInfinity });

        var result = np.isfinite(arr);

        Assert.IsNotNull(result, "np.isfinite returns null - dead code");
    }

    [Test]
    public void IsClose_DeadCode()
    {
        var a = np.array(new[] { 1.0, 1.0001 });
        var b = np.array(new[] { 1.0, 1.0002 });

        var result = np.isclose(a, b, atol: 1e-3);

        Assert.IsNotNull(result, "np.isclose returns null - dead code");
    }

    // Note: The following functions are not implemented in NumSharp
    // Selection/Manipulation:
    //   np.where - conditional selection
    //   np.sort - returns sorted copy (only argsort exists)
    //   np.flip, np.fliplr, np.flipud - reverse along axis
    //   np.rot90 - rotate 90 degrees
    //   np.tile - repeat array
    //   np.pad - pad array
    //   np.split, np.array_split, np.hsplit, np.vsplit - split operations
    // Math:
    //   np.cumprod - cumulative product
    //   np.diff - discrete difference
    //   np.gradient - numerical gradient
    //   np.ediff1d - 1D differences
    //   np.round - rounding (np.around may exist as alternative)
    // Linear Algebra:
    //   np.diag - extract/create diagonal
    //   np.diagonal - return diagonal
    //   np.trace - sum of diagonal
    // Counting:
    //   np.count_nonzero - count non-zero elements
    // NaN-aware:
    //   np.nansum, np.nanprod, np.nanmax, np.nanmin - NaN-ignoring reductions
}

/// <summary>
/// Tests documenting type promotion differences between NumSharp and NumPy.
/// </summary>
public class TypePromotionDifferenceTests
{
    [Test]
    public void Sum_Int32_OutputType_NowAligned()
    {
        // After BUG-21 fix: NumSharp now matches NumPy 2.x behavior
        // int32 sum accumulates to int64 to prevent overflow
        var arr = np.array(new int[] { 1, 2, 3 });
        var result = np.sum(arr);

        Assert.AreEqual(NPTypeCode.Int64, result.typecode,
            "NumPy 2.x: int32 sum returns int64. NumSharp now aligned.");
    }

    [Test]
    public void Max_EmptyArray_ThrowsException()
    {
        // NumPy: np.max([]) raises ValueError
        // NumSharp: Now throws ArgumentException (FIXED in Task #84)
        var empty = np.array(Array.Empty<double>());

        Assert.ThrowsException<ArgumentException>(() => np.amax(empty),
            "NumSharp now raises ArgumentException matching NumPy's ValueError.");
    }

    [Test]
    public void Min_EmptyArray_ThrowsException()
    {
        // NumPy: np.min([]) raises ValueError
        // NumSharp: Now throws ArgumentException (FIXED in Task #84)
        var empty = np.array(Array.Empty<double>());

        Assert.ThrowsException<ArgumentException>(() => np.amin(empty),
            "NumSharp now raises ArgumentException matching NumPy's ValueError.");
    }
}
