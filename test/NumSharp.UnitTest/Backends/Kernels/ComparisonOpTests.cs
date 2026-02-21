using System;
using System.Threading.Tasks;
using NumSharp;
using NumSharp.UnitTest.Utilities;
using TUnit.Core;
using TUnit.Assertions;
using TUnit.Assertions.Extensions;

namespace NumSharp.UnitTest.Backends.Kernels;

/// <summary>
/// Comprehensive tests for comparison operations (==, !=, &lt;, &gt;, &lt;=, &gt;=).
/// All expected values are verified against NumPy 2.x output.
/// </summary>
public class ComparisonOpTests
{
    #region Basic Equality Tests (Test 1)

    [Test]
    public async Task Equal_Int32_SameType()
    {
        // NumPy: np.array([1, 2, 3, 4, 5]) == np.array([1, 3, 3, 5, 5]) = [True, False, True, False, True]
        var a = np.array(new[] { 1, 2, 3, 4, 5 });
        var b = np.array(new[] { 1, 3, 3, 5, 5 });
        var result = a == b;

        await Assert.That(result.GetBoolean(0)).IsTrue();   // 1 == 1
        await Assert.That(result.GetBoolean(1)).IsFalse();  // 2 != 3
        await Assert.That(result.GetBoolean(2)).IsTrue();   // 3 == 3
        await Assert.That(result.GetBoolean(3)).IsFalse();  // 4 != 5
        await Assert.That(result.GetBoolean(4)).IsTrue();   // 5 == 5
    }

    [Test]
    public async Task NotEqual_Int32_SameType()
    {
        // NumPy: np.array([1, 2, 3, 4, 5]) != np.array([1, 3, 3, 5, 5]) = [False, True, False, True, False]
        var a = np.array(new[] { 1, 2, 3, 4, 5 });
        var b = np.array(new[] { 1, 3, 3, 5, 5 });
        var result = a != b;

        await Assert.That(result.GetBoolean(0)).IsFalse();
        await Assert.That(result.GetBoolean(1)).IsTrue();
        await Assert.That(result.GetBoolean(2)).IsFalse();
        await Assert.That(result.GetBoolean(3)).IsTrue();
        await Assert.That(result.GetBoolean(4)).IsFalse();
    }

    [Test]
    public async Task Less_Int32_SameType()
    {
        // NumPy: np.array([1, 2, 3, 4, 5]) < np.array([1, 3, 3, 5, 5]) = [False, True, False, True, False]
        var a = np.array(new[] { 1, 2, 3, 4, 5 });
        var b = np.array(new[] { 1, 3, 3, 5, 5 });
        var result = a < b;

        await Assert.That(result.GetBoolean(0)).IsFalse();  // 1 < 1 = False
        await Assert.That(result.GetBoolean(1)).IsTrue();   // 2 < 3 = True
        await Assert.That(result.GetBoolean(2)).IsFalse();  // 3 < 3 = False
        await Assert.That(result.GetBoolean(3)).IsTrue();   // 4 < 5 = True
        await Assert.That(result.GetBoolean(4)).IsFalse();  // 5 < 5 = False
    }

    [Test]
    public async Task Greater_Int32_SameType()
    {
        // NumPy: np.array([1, 2, 3, 4, 5]) > np.array([1, 3, 3, 5, 5]) = [False, False, False, False, False]
        var a = np.array(new[] { 1, 2, 3, 4, 5 });
        var b = np.array(new[] { 1, 3, 3, 5, 5 });
        var result = a > b;

        await Assert.That(result.GetBoolean(0)).IsFalse();
        await Assert.That(result.GetBoolean(1)).IsFalse();
        await Assert.That(result.GetBoolean(2)).IsFalse();
        await Assert.That(result.GetBoolean(3)).IsFalse();
        await Assert.That(result.GetBoolean(4)).IsFalse();
    }

    [Test]
    public async Task LessEqual_Int32_SameType()
    {
        // NumPy: np.array([1, 2, 3, 4, 5]) <= np.array([1, 3, 3, 5, 5]) = [True, True, True, True, True]
        var a = np.array(new[] { 1, 2, 3, 4, 5 });
        var b = np.array(new[] { 1, 3, 3, 5, 5 });
        var result = a <= b;

        await Assert.That(result.GetBoolean(0)).IsTrue();
        await Assert.That(result.GetBoolean(1)).IsTrue();
        await Assert.That(result.GetBoolean(2)).IsTrue();
        await Assert.That(result.GetBoolean(3)).IsTrue();
        await Assert.That(result.GetBoolean(4)).IsTrue();
    }

    [Test]
    public async Task GreaterEqual_Int32_SameType()
    {
        // NumPy: np.array([1, 2, 3, 4, 5]) >= np.array([1, 3, 3, 5, 5]) = [True, False, True, False, True]
        var a = np.array(new[] { 1, 2, 3, 4, 5 });
        var b = np.array(new[] { 1, 3, 3, 5, 5 });
        var result = a >= b;

        await Assert.That(result.GetBoolean(0)).IsTrue();
        await Assert.That(result.GetBoolean(1)).IsFalse();
        await Assert.That(result.GetBoolean(2)).IsTrue();
        await Assert.That(result.GetBoolean(3)).IsFalse();
        await Assert.That(result.GetBoolean(4)).IsTrue();
    }

    #endregion

    #region Mixed Types Tests (Test 2)

    [Test]
    public async Task Equal_MixedTypes_Int32_Float64()
    {
        // NumPy: np.array([1, 2, 3, 4], dtype=int32) == np.array([1.5, 2.0, 2.5, 4.0], dtype=float64) = [False, True, False, True]
        var x = np.array(new[] { 1, 2, 3, 4 });
        var y = np.array(new[] { 1.5, 2.0, 2.5, 4.0 });
        var result = x == y;

        await Assert.That(result.GetBoolean(0)).IsFalse();  // 1 != 1.5
        await Assert.That(result.GetBoolean(1)).IsTrue();   // 2 == 2.0
        await Assert.That(result.GetBoolean(2)).IsFalse();  // 3 != 2.5
        await Assert.That(result.GetBoolean(3)).IsTrue();   // 4 == 4.0
    }

    [Test]
    public async Task Less_MixedTypes_Int32_Float64()
    {
        // NumPy: np.array([1, 2, 3, 4], dtype=int32) < np.array([1.5, 2.0, 2.5, 4.0], dtype=float64) = [True, False, False, False]
        var x = np.array(new[] { 1, 2, 3, 4 });
        var y = np.array(new[] { 1.5, 2.0, 2.5, 4.0 });
        var result = x < y;

        await Assert.That(result.GetBoolean(0)).IsTrue();   // 1 < 1.5
        await Assert.That(result.GetBoolean(1)).IsFalse();  // 2 >= 2.0
        await Assert.That(result.GetBoolean(2)).IsFalse();  // 3 >= 2.5
        await Assert.That(result.GetBoolean(3)).IsFalse();  // 4 >= 4.0
    }

    [Test]
    public async Task GreaterEqual_MixedTypes_Int32_Float64()
    {
        // NumPy: np.array([1, 2, 3, 4], dtype=int32) >= np.array([1.5, 2.0, 2.5, 4.0], dtype=float64) = [False, True, True, True]
        var x = np.array(new[] { 1, 2, 3, 4 });
        var y = np.array(new[] { 1.5, 2.0, 2.5, 4.0 });
        var result = x >= y;

        await Assert.That(result.GetBoolean(0)).IsFalse();  // 1 < 1.5
        await Assert.That(result.GetBoolean(1)).IsTrue();   // 2 >= 2.0
        await Assert.That(result.GetBoolean(2)).IsTrue();   // 3 >= 2.5
        await Assert.That(result.GetBoolean(3)).IsTrue();   // 4 >= 4.0
    }

    #endregion

    #region Broadcasting Tests (Test 3)

    [Test]
    public async Task Greater_Broadcasting_2D_vs_1D()
    {
        // NumPy: arr2d = [[1, 2, 3], [4, 5, 6]], arr1d = [2, 3, 4]
        // arr2d > arr1d = [[False, False, False], [True, True, True]]
        var arr2d = np.array(new[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        var arr1d = np.array(new[] { 2, 3, 4 });
        var result = arr2d > arr1d;

        // Row 0: [1>2, 2>3, 3>4] = [False, False, False]
        await Assert.That(result.GetBoolean(0, 0)).IsFalse();
        await Assert.That(result.GetBoolean(0, 1)).IsFalse();
        await Assert.That(result.GetBoolean(0, 2)).IsFalse();

        // Row 1: [4>2, 5>3, 6>4] = [True, True, True]
        await Assert.That(result.GetBoolean(1, 0)).IsTrue();
        await Assert.That(result.GetBoolean(1, 1)).IsTrue();
        await Assert.That(result.GetBoolean(1, 2)).IsTrue();
    }

    [Test]
    public async Task LessEqual_Broadcasting_2D_vs_1D()
    {
        // NumPy: arr2d <= arr1d = [[True, True, True], [False, False, False]]
        var arr2d = np.array(new[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        var arr1d = np.array(new[] { 2, 3, 4 });
        var result = arr2d <= arr1d;

        // Row 0: [1<=2, 2<=3, 3<=4] = [True, True, True]
        await Assert.That(result.GetBoolean(0, 0)).IsTrue();
        await Assert.That(result.GetBoolean(0, 1)).IsTrue();
        await Assert.That(result.GetBoolean(0, 2)).IsTrue();

        // Row 1: [4<=2, 5<=3, 6<=4] = [False, False, False]
        await Assert.That(result.GetBoolean(1, 0)).IsFalse();
        await Assert.That(result.GetBoolean(1, 1)).IsFalse();
        await Assert.That(result.GetBoolean(1, 2)).IsFalse();
    }

    #endregion

    #region Scalar Comparison Tests (Test 4)

    [Test]
    public async Task Greater_ScalarRight()
    {
        // NumPy: arr > 3 = [False, False, False, True, True]
        var arr = np.array(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var result = arr > 3;

        await Assert.That(result.GetBoolean(0)).IsFalse();
        await Assert.That(result.GetBoolean(1)).IsFalse();
        await Assert.That(result.GetBoolean(2)).IsFalse();
        await Assert.That(result.GetBoolean(3)).IsTrue();
        await Assert.That(result.GetBoolean(4)).IsTrue();
    }

    [Test]
    public async Task LessEqual_ScalarRight()
    {
        // NumPy: arr <= 2.5 = [True, True, False, False, False]
        var arr = np.array(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var result = arr <= 2.5;

        await Assert.That(result.GetBoolean(0)).IsTrue();
        await Assert.That(result.GetBoolean(1)).IsTrue();
        await Assert.That(result.GetBoolean(2)).IsFalse();
        await Assert.That(result.GetBoolean(3)).IsFalse();
        await Assert.That(result.GetBoolean(4)).IsFalse();
    }

    [Test]
    public async Task Equal_ScalarRight()
    {
        // NumPy: arr == 3.0 = [False, False, True, False, False]
        var arr = np.array(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var result = arr == 3.0;

        await Assert.That(result.GetBoolean(0)).IsFalse();
        await Assert.That(result.GetBoolean(1)).IsFalse();
        await Assert.That(result.GetBoolean(2)).IsTrue();
        await Assert.That(result.GetBoolean(3)).IsFalse();
        await Assert.That(result.GetBoolean(4)).IsFalse();
    }

    #endregion

    #region Boolean Comparison Tests (Test 5)

    [Test]
    public async Task Equal_Boolean()
    {
        // NumPy: [True, True, False, False] == [True, False, True, False] = [True, False, False, True]
        var a = np.array(new[] { true, true, false, false });
        var b = np.array(new[] { true, false, true, false });
        var result = a == b;

        await Assert.That(result.GetBoolean(0)).IsTrue();
        await Assert.That(result.GetBoolean(1)).IsFalse();
        await Assert.That(result.GetBoolean(2)).IsFalse();
        await Assert.That(result.GetBoolean(3)).IsTrue();
    }

    [Test]
    public async Task NotEqual_Boolean()
    {
        // NumPy: [True, True, False, False] != [True, False, True, False] = [False, True, True, False]
        var a = np.array(new[] { true, true, false, false });
        var b = np.array(new[] { true, false, true, false });
        var result = a != b;

        await Assert.That(result.GetBoolean(0)).IsFalse();
        await Assert.That(result.GetBoolean(1)).IsTrue();
        await Assert.That(result.GetBoolean(2)).IsTrue();
        await Assert.That(result.GetBoolean(3)).IsFalse();
    }

    [Test]
    public async Task Less_Boolean()
    {
        // NumPy: [True, True, False, False] < [True, False, True, False] = [False, False, True, False]
        // Note: In NumPy, False < True (False=0, True=1)
        var a = np.array(new[] { true, true, false, false });
        var b = np.array(new[] { true, false, true, false });
        var result = a < b;

        await Assert.That(result.GetBoolean(0)).IsFalse();  // True < True = False
        await Assert.That(result.GetBoolean(1)).IsFalse();  // True < False = False
        await Assert.That(result.GetBoolean(2)).IsTrue();   // False < True = True
        await Assert.That(result.GetBoolean(3)).IsFalse();  // False < False = False
    }

    [Test]
    public async Task Greater_Boolean()
    {
        // NumPy: [True, True, False, False] > [True, False, True, False] = [False, True, False, False]
        var a = np.array(new[] { true, true, false, false });
        var b = np.array(new[] { true, false, true, false });
        var result = a > b;

        await Assert.That(result.GetBoolean(0)).IsFalse();  // True > True = False
        await Assert.That(result.GetBoolean(1)).IsTrue();   // True > False = True
        await Assert.That(result.GetBoolean(2)).IsFalse();  // False > True = False
        await Assert.That(result.GetBoolean(3)).IsFalse();  // False > False = False
    }

    #endregion

    #region Byte Comparison Tests (Test 6)

    [Test]
    public async Task Equal_Byte()
    {
        // NumPy: np.array([0, 128, 255], dtype=uint8) == np.array([1, 128, 254], dtype=uint8) = [False, True, False]
        var a = np.array(new byte[] { 0, 128, 255 });
        var b = np.array(new byte[] { 1, 128, 254 });
        var result = a == b;

        await Assert.That(result.GetBoolean(0)).IsFalse();
        await Assert.That(result.GetBoolean(1)).IsTrue();
        await Assert.That(result.GetBoolean(2)).IsFalse();
    }

    [Test]
    public async Task Less_Byte()
    {
        // NumPy: np.array([0, 128, 255], dtype=uint8) < np.array([1, 128, 254], dtype=uint8) = [True, False, False]
        var a = np.array(new byte[] { 0, 128, 255 });
        var b = np.array(new byte[] { 1, 128, 254 });
        var result = a < b;

        await Assert.That(result.GetBoolean(0)).IsTrue();   // 0 < 1
        await Assert.That(result.GetBoolean(1)).IsFalse();  // 128 >= 128
        await Assert.That(result.GetBoolean(2)).IsFalse();  // 255 > 254
    }

    #endregion

    #region Scalar vs Scalar Tests (Test 7)

    [Test]
    public async Task Equal_ScalarVsScalar()
    {
        // NumPy: np.array(3) == np.array(5) = False (shape: ())
        var s1 = NDArray.Scalar(3);
        var s2 = NDArray.Scalar(5);
        var result = s1 == s2;

        await Assert.That(result.Shape.IsScalar).IsTrue();
        await Assert.That(result.GetBoolean()).IsFalse();
    }

    [Test]
    public async Task Less_ScalarVsScalar()
    {
        // NumPy: np.array(3) < np.array(5) = True
        var s1 = NDArray.Scalar(3);
        var s2 = NDArray.Scalar(5);
        var result = s1 < s2;

        await Assert.That(result.Shape.IsScalar).IsTrue();
        await Assert.That(result.GetBoolean()).IsTrue();
    }

    [Test]
    public async Task GreaterEqual_ScalarVsScalar()
    {
        // NumPy: np.array(3) >= np.array(5) = False
        var s1 = NDArray.Scalar(3);
        var s2 = NDArray.Scalar(5);
        var result = s1 >= s2;

        await Assert.That(result.Shape.IsScalar).IsTrue();
        await Assert.That(result.GetBoolean()).IsFalse();
    }

    #endregion

    #region Float Edge Cases Tests (Test 8)

    [Test]
    public async Task Equal_FloatWithNaN()
    {
        // NumPy: [1.0, nan, inf, -inf, 0.0] == [1.0, nan, inf, -inf, 0.0] = [True, False, True, True, True]
        // NaN != NaN by IEEE 754 standard
        var floats = np.array(new[] { 1.0, double.NaN, double.PositiveInfinity, double.NegativeInfinity, 0.0 });
        var result = floats == floats;

        await Assert.That(result.GetBoolean(0)).IsTrue();   // 1.0 == 1.0
        await Assert.That(result.GetBoolean(1)).IsFalse();  // NaN != NaN (IEEE 754)
        await Assert.That(result.GetBoolean(2)).IsTrue();   // Inf == Inf
        await Assert.That(result.GetBoolean(3)).IsTrue();   // -Inf == -Inf
        await Assert.That(result.GetBoolean(4)).IsTrue();   // 0.0 == 0.0
    }

    [Test]
    public async Task Less_FloatWithInfinity()
    {
        // NumPy: [1.0, nan, inf, -inf, 0.0] < 1.0 = [False, False, False, True, True]
        var floats = np.array(new[] { 1.0, double.NaN, double.PositiveInfinity, double.NegativeInfinity, 0.0 });
        var result = floats < 1.0;

        await Assert.That(result.GetBoolean(0)).IsFalse();  // 1.0 < 1.0 = False
        await Assert.That(result.GetBoolean(1)).IsFalse();  // NaN < anything = False
        await Assert.That(result.GetBoolean(2)).IsFalse();  // Inf < 1.0 = False
        await Assert.That(result.GetBoolean(3)).IsTrue();   // -Inf < 1.0 = True
        await Assert.That(result.GetBoolean(4)).IsTrue();   // 0.0 < 1.0 = True
    }

    #endregion

    #region All dtypes tests

    [Test]
    public async Task Equal_Int16()
    {
        var a = np.array(new short[] { 1, 2, 3 });
        var b = np.array(new short[] { 1, 3, 3 });
        var result = a == b;

        await Assert.That(result.GetBoolean(0)).IsTrue();
        await Assert.That(result.GetBoolean(1)).IsFalse();
        await Assert.That(result.GetBoolean(2)).IsTrue();
    }

    [Test]
    public async Task Equal_UInt16()
    {
        var a = np.array(new ushort[] { 1, 2, 3 });
        var b = np.array(new ushort[] { 1, 3, 3 });
        var result = a == b;

        await Assert.That(result.GetBoolean(0)).IsTrue();
        await Assert.That(result.GetBoolean(1)).IsFalse();
        await Assert.That(result.GetBoolean(2)).IsTrue();
    }

    [Test]
    public async Task Equal_UInt32()
    {
        var a = np.array(new uint[] { 1, 2, 3 });
        var b = np.array(new uint[] { 1, 3, 3 });
        var result = a == b;

        await Assert.That(result.GetBoolean(0)).IsTrue();
        await Assert.That(result.GetBoolean(1)).IsFalse();
        await Assert.That(result.GetBoolean(2)).IsTrue();
    }

    [Test]
    public async Task Equal_Int64()
    {
        var a = np.array(new long[] { 1, 2, 3 });
        var b = np.array(new long[] { 1, 3, 3 });
        var result = a == b;

        await Assert.That(result.GetBoolean(0)).IsTrue();
        await Assert.That(result.GetBoolean(1)).IsFalse();
        await Assert.That(result.GetBoolean(2)).IsTrue();
    }

    [Test]
    public async Task Equal_UInt64()
    {
        var a = np.array(new ulong[] { 1, 2, 3 });
        var b = np.array(new ulong[] { 1, 3, 3 });
        var result = a == b;

        await Assert.That(result.GetBoolean(0)).IsTrue();
        await Assert.That(result.GetBoolean(1)).IsFalse();
        await Assert.That(result.GetBoolean(2)).IsTrue();
    }

    [Test]
    public async Task Equal_Single()
    {
        var a = np.array(new float[] { 1.0f, 2.0f, 3.0f });
        var b = np.array(new float[] { 1.0f, 3.0f, 3.0f });
        var result = a == b;

        await Assert.That(result.GetBoolean(0)).IsTrue();
        await Assert.That(result.GetBoolean(1)).IsFalse();
        await Assert.That(result.GetBoolean(2)).IsTrue();
    }

    [Test]
    public async Task Equal_Decimal()
    {
        var a = np.array(new decimal[] { 1.0m, 2.0m, 3.0m });
        var b = np.array(new decimal[] { 1.0m, 3.0m, 3.0m });
        var result = a == b;

        await Assert.That(result.GetBoolean(0)).IsTrue();
        await Assert.That(result.GetBoolean(1)).IsFalse();
        await Assert.That(result.GetBoolean(2)).IsTrue();
    }

    [Test]
    public async Task Equal_Char()
    {
        var a = np.array(new char[] { 'a', 'b', 'c' });
        var b = np.array(new char[] { 'a', 'x', 'c' });
        var result = a == b;

        await Assert.That(result.GetBoolean(0)).IsTrue();
        await Assert.That(result.GetBoolean(1)).IsFalse();
        await Assert.That(result.GetBoolean(2)).IsTrue();
    }

    #endregion
}
