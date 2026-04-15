using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.Backends.Kernels;

/// <summary>
/// Tests documenting NumSharp vs NumPy misalignments in kernel operations.
/// These tests verify current NumSharp behavior which differs from NumPy.
/// </summary>
public class KernelMisalignmentTests
{
    /// <summary>
    /// NumPy: np.square(np.int32(3)) -> int32(9) (preserves dtype)
    /// NumSharp: np.square(3) -> int32(9) (preserves dtype) - FIXED
    /// </summary>
    [TestMethod]
    public void Square_IntegerInput_PreservesDtype()
    {
        // NumPy behavior: preserves int32 dtype
        // >>> np.square(np.int32(3))
        // np.int32(9)
        // >>> np.square(np.int32(3)).dtype
        // dtype('int32')

        var result = np.square(3);

        // NumSharp now preserves int dtype (matching NumPy)
        Assert.AreEqual(typeof(int), result.dtype);
        Assert.AreEqual(9, (int)result);
    }

    /// <summary>
    /// NumPy: np.square([1, 2, 3]) -> array([1, 4, 9], dtype=int32)
    /// NumSharp: np.square([1, 2, 3]) -> array([1, 4, 9], dtype=int32) - FIXED
    /// </summary>
    [TestMethod]
    public void Square_IntegerArray_PreservesDtype()
    {
        // NumPy behavior:
        // >>> np.square(np.array([1, 2, 3], dtype=np.int32)).dtype
        // dtype('int32')

        var arr = np.array(new[] { 1, 2, 3 });
        var result = np.square(arr);

        // NumSharp now preserves int dtype (matching NumPy)
        Assert.AreEqual(typeof(int), result.dtype);

        var values = result.ToArray<int>();
        Assert.AreEqual(1, values[0]);
        Assert.AreEqual(4, values[1]);
        Assert.AreEqual(9, values[2]);
    }

    /// <summary>
    /// NumPy: np.invert(True) -> False (logical NOT for boolean)
    /// NumSharp: np.invert(true) -> False (logical NOT) - FIXED
    /// </summary>
    [TestMethod]
    public void Invert_Boolean_LogicalNot()
    {
        // NumPy behavior:
        // >>> np.invert(True)
        // np.False_
        // >>> np.invert(False)
        // np.True_

        // NumSharp now uses logical NOT for booleans (matching NumPy)
        var trueResult = np.invert(true);
        Assert.IsFalse((bool)trueResult);

        var falseResult = np.invert(false);
        Assert.IsTrue((bool)falseResult);
    }

    /// <summary>
    /// NumPy: np.invert(np.array([True, False])) -> array([False, True])
    /// NumSharp: np.invert([true, false]) -> [False, True] - FIXED
    /// </summary>
    [TestMethod]
    public void Invert_BooleanArray_LogicalNot()
    {
        // NumPy behavior:
        // >>> np.invert(np.array([True, False]))
        // array([False,  True])

        var arr = np.array(new[] { true, false });
        var result = np.invert(arr);
        var values = result.ToArray<bool>();

        // NumSharp now uses logical NOT for booleans (matching NumPy)
        Assert.AreEqual(2, values.Length);
        Assert.IsFalse(values[0]);  // True -> False
        Assert.IsTrue(values[1]);   // False -> True
    }

    /// <summary>
    /// Verify np.square works correctly for floating point (no misalignment).
    /// </summary>
    [TestMethod]
    public void Square_FloatingPoint_Correct()
    {
        // Float input preserves float
        var floatResult = np.square(3.0f);
        Assert.AreEqual(typeof(float), floatResult.dtype);
        Assert.AreEqual(9.0f, (float)floatResult);

        // Double input preserves double
        var doubleResult = np.square(3.0);
        Assert.AreEqual(typeof(double), doubleResult.dtype);
        Assert.AreEqual(9.0, (double)doubleResult);
    }

    /// <summary>
    /// Verify np.invert works correctly for integers (no misalignment).
    /// </summary>
    [TestMethod]
    public void Invert_Integer_Correct()
    {
        // NumPy: np.invert(0) -> -1
        // NumPy: np.invert(1) -> -2
        var zeroResult = np.invert(0);
        var oneResult = np.invert(1);

        Assert.AreEqual(-1, (int)zeroResult);
        Assert.AreEqual(-2, (int)oneResult);
    }

    /// <summary>
    /// NumPy: np.reciprocal(2) -> 0 (integer floor division: 1/2 = 0)
    /// NumSharp: np.reciprocal(2) -> 0.5 (promotes to double, does floating point division)
    ///
    /// This is a type promotion misalignment - NumSharp promotes integer inputs
    /// to floating point for reciprocal, while NumPy preserves integer dtype
    /// and uses floor division.
    /// </summary>
    [TestMethod]
    [Misaligned]
    public void Reciprocal_Integer_TypePromotion()
    {
        // NumPy behavior:
        // >>> np.reciprocal(2)
        // 0
        // >>> np.reciprocal(np.int32(2)).dtype
        // dtype('int32')
        // >>> np.reciprocal(np.array([2, 3, 4]))
        // array([0, 0, 0])

        var result = np.reciprocal(2);

        // NumSharp promotes to double and returns 0.5
        Assert.AreEqual(typeof(double), result.dtype);
        Assert.AreEqual(0.5, (double)result);

        // Expected NumPy behavior (not implemented):
        // Assert.AreEqual(typeof(int), result.dtype);
        // Assert.AreEqual(0, (int)result);
    }

    /// <summary>
    /// Verify np.reciprocal floating point is correct.
    /// </summary>
    [TestMethod]
    public void Reciprocal_FloatingPoint_Correct()
    {
        // NumPy: np.reciprocal(2.0) -> 0.5
        var result = np.reciprocal(2.0);
        Assert.AreEqual(0.5, (double)result);

        // NumPy: np.reciprocal(0.0) -> inf
        var zeroResult = np.reciprocal(0.0);
        Assert.IsTrue(double.IsPositiveInfinity((double)zeroResult));
    }
}
