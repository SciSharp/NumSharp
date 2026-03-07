using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.UnitTest.Utilities;
using TUnit.Core;

namespace NumSharp.UnitTest.Backends.Kernels;

/// <summary>
/// Comprehensive bug reproduction tests found during NumPy 2.4.2 alignment testing.
/// All tests assert CORRECT NumPy behavior. Tests FAIL while bug exists, PASS when fixed.
///
/// Bug Summary (11 bugs):
/// - BUG-1: Boolean indexing setter throws NotImplementedException (CRITICAL)
/// - BUG-2: np.nonzero 1-D shape wrong (HIGH)
/// - BUG-3: np.all/np.any with axis throws InvalidCastException (HIGH)
/// - BUG-4: np.std/np.var ddof parameter ignored (MEDIUM)
/// - BUG-5: np.std/np.var crash on empty array (MEDIUM)
/// - BUG-6: np.sum empty 2D with axis returns scalar (MEDIUM)
/// - BUG-7: sbyte (int8) not supported (LOW)
/// - BUG-8: np.dot(vector, matrix) not supported (LOW)
/// - BUG-9: np.unique returns unsorted (MEDIUM)
/// - BUG-10: np.repeat with array repeats fails (MEDIUM)
/// - BUG-11: np.repeat missing axis parameter (LOW)
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
    public void Bug7_SByte_ArrayCreation()
    {
        // NumPy supports int8 (sbyte)
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
}

/// <summary>
/// Tests documenting type promotion differences between NumSharp and NumPy.
/// </summary>
public class TypePromotionDifferenceTests
{
    [Test]
    [Misaligned]
    public void Sum_Int32_OutputType()
    {
        var arr = np.array(new int[] { 1, 2, 3 });
        var result = np.sum(arr);

        Assert.AreEqual(NPTypeCode.Int32, result.typecode,
            "NumSharp returns int32. NumPy returns int64.");
    }

    [Test]
    [Misaligned]
    public void Max_EmptyArray_UsesIdentity()
    {
        var empty = np.array(Array.Empty<double>());
        var result = np.amax(empty);

        Assert.IsTrue(double.IsNegativeInfinity(result.GetDouble(0)),
            "NumSharp returns -Inf. NumPy raises ValueError.");
    }

    [Test]
    [Misaligned]
    public void Min_EmptyArray_UsesIdentity()
    {
        var empty = np.array(Array.Empty<double>());
        var result = np.amin(empty);

        Assert.IsTrue(double.IsPositiveInfinity(result.GetDouble(0)),
            "NumSharp returns +Inf. NumPy raises ValueError.");
    }
}
