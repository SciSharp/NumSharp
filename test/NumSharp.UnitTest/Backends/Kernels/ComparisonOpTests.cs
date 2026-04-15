using System;
using System.Threading.Tasks;
using NumSharp;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Backends.Kernels;

/// <summary>
/// Comprehensive tests for comparison operations (==, !=, &lt;, &gt;, &lt;=, &gt;=).
/// All expected values are verified against NumPy 2.x output.
/// </summary>
public class ComparisonOpTests
{
    #region Basic Equality Tests (Test 1)

    [TestMethod]
    public async Task Equal_Int32_SameType()
    {
        // NumPy: np.array([1, 2, 3, 4, 5]) == np.array([1, 3, 3, 5, 5]) = [True, False, True, False, True]
        var a = np.array(new[] { 1, 2, 3, 4, 5 });
        var b = np.array(new[] { 1, 3, 3, 5, 5 });
        var result = a == b;

        result.GetBoolean(0).Should().BeTrue();   // 1 == 1
        result.GetBoolean(1).Should().BeFalse();  // 2 != 3
        result.GetBoolean(2).Should().BeTrue();   // 3 == 3
        result.GetBoolean(3).Should().BeFalse();  // 4 != 5
        result.GetBoolean(4).Should().BeTrue();   // 5 == 5
    }

    [TestMethod]
    public async Task NotEqual_Int32_SameType()
    {
        // NumPy: np.array([1, 2, 3, 4, 5]) != np.array([1, 3, 3, 5, 5]) = [False, True, False, True, False]
        var a = np.array(new[] { 1, 2, 3, 4, 5 });
        var b = np.array(new[] { 1, 3, 3, 5, 5 });
        var result = a != b;

        result.GetBoolean(0).Should().BeFalse();
        result.GetBoolean(1).Should().BeTrue();
        result.GetBoolean(2).Should().BeFalse();
        result.GetBoolean(3).Should().BeTrue();
        result.GetBoolean(4).Should().BeFalse();
    }

    [TestMethod]
    public async Task Less_Int32_SameType()
    {
        // NumPy: np.array([1, 2, 3, 4, 5]) < np.array([1, 3, 3, 5, 5]) = [False, True, False, True, False]
        var a = np.array(new[] { 1, 2, 3, 4, 5 });
        var b = np.array(new[] { 1, 3, 3, 5, 5 });
        var result = a < b;

        result.GetBoolean(0).Should().BeFalse();  // 1 < 1 = False
        result.GetBoolean(1).Should().BeTrue();   // 2 < 3 = True
        result.GetBoolean(2).Should().BeFalse();  // 3 < 3 = False
        result.GetBoolean(3).Should().BeTrue();   // 4 < 5 = True
        result.GetBoolean(4).Should().BeFalse();  // 5 < 5 = False
    }

    [TestMethod]
    public async Task Greater_Int32_SameType()
    {
        // NumPy: np.array([1, 2, 3, 4, 5]) > np.array([1, 3, 3, 5, 5]) = [False, False, False, False, False]
        var a = np.array(new[] { 1, 2, 3, 4, 5 });
        var b = np.array(new[] { 1, 3, 3, 5, 5 });
        var result = a > b;

        result.GetBoolean(0).Should().BeFalse();
        result.GetBoolean(1).Should().BeFalse();
        result.GetBoolean(2).Should().BeFalse();
        result.GetBoolean(3).Should().BeFalse();
        result.GetBoolean(4).Should().BeFalse();
    }

    [TestMethod]
    public async Task LessEqual_Int32_SameType()
    {
        // NumPy: np.array([1, 2, 3, 4, 5]) <= np.array([1, 3, 3, 5, 5]) = [True, True, True, True, True]
        var a = np.array(new[] { 1, 2, 3, 4, 5 });
        var b = np.array(new[] { 1, 3, 3, 5, 5 });
        var result = a <= b;

        result.GetBoolean(0).Should().BeTrue();
        result.GetBoolean(1).Should().BeTrue();
        result.GetBoolean(2).Should().BeTrue();
        result.GetBoolean(3).Should().BeTrue();
        result.GetBoolean(4).Should().BeTrue();
    }

    [TestMethod]
    public async Task GreaterEqual_Int32_SameType()
    {
        // NumPy: np.array([1, 2, 3, 4, 5]) >= np.array([1, 3, 3, 5, 5]) = [True, False, True, False, True]
        var a = np.array(new[] { 1, 2, 3, 4, 5 });
        var b = np.array(new[] { 1, 3, 3, 5, 5 });
        var result = a >= b;

        result.GetBoolean(0).Should().BeTrue();
        result.GetBoolean(1).Should().BeFalse();
        result.GetBoolean(2).Should().BeTrue();
        result.GetBoolean(3).Should().BeFalse();
        result.GetBoolean(4).Should().BeTrue();
    }

    #endregion

    #region Mixed Types Tests (Test 2)

    [TestMethod]
    public async Task Equal_MixedTypes_Int32_Float64()
    {
        // NumPy: np.array([1, 2, 3, 4], dtype=int32) == np.array([1.5, 2.0, 2.5, 4.0], dtype=float64) = [False, True, False, True]
        var x = np.array(new[] { 1, 2, 3, 4 });
        var y = np.array(new[] { 1.5, 2.0, 2.5, 4.0 });
        var result = x == y;

        result.GetBoolean(0).Should().BeFalse();  // 1 != 1.5
        result.GetBoolean(1).Should().BeTrue();   // 2 == 2.0
        result.GetBoolean(2).Should().BeFalse();  // 3 != 2.5
        result.GetBoolean(3).Should().BeTrue();   // 4 == 4.0
    }

    [TestMethod]
    public async Task Less_MixedTypes_Int32_Float64()
    {
        // NumPy: np.array([1, 2, 3, 4], dtype=int32) < np.array([1.5, 2.0, 2.5, 4.0], dtype=float64) = [True, False, False, False]
        var x = np.array(new[] { 1, 2, 3, 4 });
        var y = np.array(new[] { 1.5, 2.0, 2.5, 4.0 });
        var result = x < y;

        result.GetBoolean(0).Should().BeTrue();   // 1 < 1.5
        result.GetBoolean(1).Should().BeFalse();  // 2 >= 2.0
        result.GetBoolean(2).Should().BeFalse();  // 3 >= 2.5
        result.GetBoolean(3).Should().BeFalse();  // 4 >= 4.0
    }

    [TestMethod]
    public async Task GreaterEqual_MixedTypes_Int32_Float64()
    {
        // NumPy: np.array([1, 2, 3, 4], dtype=int32) >= np.array([1.5, 2.0, 2.5, 4.0], dtype=float64) = [False, True, True, True]
        var x = np.array(new[] { 1, 2, 3, 4 });
        var y = np.array(new[] { 1.5, 2.0, 2.5, 4.0 });
        var result = x >= y;

        result.GetBoolean(0).Should().BeFalse();  // 1 < 1.5
        result.GetBoolean(1).Should().BeTrue();   // 2 >= 2.0
        result.GetBoolean(2).Should().BeTrue();   // 3 >= 2.5
        result.GetBoolean(3).Should().BeTrue();   // 4 >= 4.0
    }

    #endregion

    #region Broadcasting Tests (Test 3)

    [TestMethod]
    public async Task Greater_Broadcasting_2D_vs_1D()
    {
        // NumPy: arr2d = [[1, 2, 3], [4, 5, 6]], arr1d = [2, 3, 4]
        // arr2d > arr1d = [[False, False, False], [True, True, True]]
        var arr2d = np.array(new[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        var arr1d = np.array(new[] { 2, 3, 4 });
        var result = arr2d > arr1d;

        // Row 0: [1>2, 2>3, 3>4] = [False, False, False]
        result.GetBoolean(0, 0).Should().BeFalse();
        result.GetBoolean(0, 1).Should().BeFalse();
        result.GetBoolean(0, 2).Should().BeFalse();

        // Row 1: [4>2, 5>3, 6>4] = [True, True, True]
        result.GetBoolean(1, 0).Should().BeTrue();
        result.GetBoolean(1, 1).Should().BeTrue();
        result.GetBoolean(1, 2).Should().BeTrue();
    }

    [TestMethod]
    public async Task LessEqual_Broadcasting_2D_vs_1D()
    {
        // NumPy: arr2d <= arr1d = [[True, True, True], [False, False, False]]
        var arr2d = np.array(new[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        var arr1d = np.array(new[] { 2, 3, 4 });
        var result = arr2d <= arr1d;

        // Row 0: [1<=2, 2<=3, 3<=4] = [True, True, True]
        result.GetBoolean(0, 0).Should().BeTrue();
        result.GetBoolean(0, 1).Should().BeTrue();
        result.GetBoolean(0, 2).Should().BeTrue();

        // Row 1: [4<=2, 5<=3, 6<=4] = [False, False, False]
        result.GetBoolean(1, 0).Should().BeFalse();
        result.GetBoolean(1, 1).Should().BeFalse();
        result.GetBoolean(1, 2).Should().BeFalse();
    }

    #endregion

    #region Scalar Comparison Tests (Test 4)

    [TestMethod]
    public async Task Greater_ScalarRight()
    {
        // NumPy: arr > 3 = [False, False, False, True, True]
        var arr = np.array(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var result = arr > 3;

        result.GetBoolean(0).Should().BeFalse();
        result.GetBoolean(1).Should().BeFalse();
        result.GetBoolean(2).Should().BeFalse();
        result.GetBoolean(3).Should().BeTrue();
        result.GetBoolean(4).Should().BeTrue();
    }

    [TestMethod]
    public async Task LessEqual_ScalarRight()
    {
        // NumPy: arr <= 2.5 = [True, True, False, False, False]
        var arr = np.array(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var result = arr <= 2.5;

        result.GetBoolean(0).Should().BeTrue();
        result.GetBoolean(1).Should().BeTrue();
        result.GetBoolean(2).Should().BeFalse();
        result.GetBoolean(3).Should().BeFalse();
        result.GetBoolean(4).Should().BeFalse();
    }

    [TestMethod]
    public async Task Equal_ScalarRight()
    {
        // NumPy: arr == 3.0 = [False, False, True, False, False]
        var arr = np.array(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var result = arr == 3.0;

        result.GetBoolean(0).Should().BeFalse();
        result.GetBoolean(1).Should().BeFalse();
        result.GetBoolean(2).Should().BeTrue();
        result.GetBoolean(3).Should().BeFalse();
        result.GetBoolean(4).Should().BeFalse();
    }

    #endregion

    #region Boolean Comparison Tests (Test 5)

    [TestMethod]
    public async Task Equal_Boolean()
    {
        // NumPy: [True, True, False, False] == [True, False, True, False] = [True, False, False, True]
        var a = np.array(new[] { true, true, false, false });
        var b = np.array(new[] { true, false, true, false });
        var result = a == b;

        result.GetBoolean(0).Should().BeTrue();
        result.GetBoolean(1).Should().BeFalse();
        result.GetBoolean(2).Should().BeFalse();
        result.GetBoolean(3).Should().BeTrue();
    }

    [TestMethod]
    public async Task NotEqual_Boolean()
    {
        // NumPy: [True, True, False, False] != [True, False, True, False] = [False, True, True, False]
        var a = np.array(new[] { true, true, false, false });
        var b = np.array(new[] { true, false, true, false });
        var result = a != b;

        result.GetBoolean(0).Should().BeFalse();
        result.GetBoolean(1).Should().BeTrue();
        result.GetBoolean(2).Should().BeTrue();
        result.GetBoolean(3).Should().BeFalse();
    }

    [TestMethod]
    public async Task Less_Boolean()
    {
        // NumPy: [True, True, False, False] < [True, False, True, False] = [False, False, True, False]
        // Note: In NumPy, False < True (False=0, True=1)
        var a = np.array(new[] { true, true, false, false });
        var b = np.array(new[] { true, false, true, false });
        var result = a < b;

        result.GetBoolean(0).Should().BeFalse();  // True < True = False
        result.GetBoolean(1).Should().BeFalse();  // True < False = False
        result.GetBoolean(2).Should().BeTrue();   // False < True = True
        result.GetBoolean(3).Should().BeFalse();  // False < False = False
    }

    [TestMethod]
    public async Task Greater_Boolean()
    {
        // NumPy: [True, True, False, False] > [True, False, True, False] = [False, True, False, False]
        var a = np.array(new[] { true, true, false, false });
        var b = np.array(new[] { true, false, true, false });
        var result = a > b;

        result.GetBoolean(0).Should().BeFalse();  // True > True = False
        result.GetBoolean(1).Should().BeTrue();   // True > False = True
        result.GetBoolean(2).Should().BeFalse();  // False > True = False
        result.GetBoolean(3).Should().BeFalse();  // False > False = False
    }

    #endregion

    #region Byte Comparison Tests (Test 6)

    [TestMethod]
    public async Task Equal_Byte()
    {
        // NumPy: np.array([0, 128, 255], dtype=uint8) == np.array([1, 128, 254], dtype=uint8) = [False, True, False]
        var a = np.array(new byte[] { 0, 128, 255 });
        var b = np.array(new byte[] { 1, 128, 254 });
        var result = a == b;

        result.GetBoolean(0).Should().BeFalse();
        result.GetBoolean(1).Should().BeTrue();
        result.GetBoolean(2).Should().BeFalse();
    }

    [TestMethod]
    public async Task Less_Byte()
    {
        // NumPy: np.array([0, 128, 255], dtype=uint8) < np.array([1, 128, 254], dtype=uint8) = [True, False, False]
        var a = np.array(new byte[] { 0, 128, 255 });
        var b = np.array(new byte[] { 1, 128, 254 });
        var result = a < b;

        result.GetBoolean(0).Should().BeTrue();   // 0 < 1
        result.GetBoolean(1).Should().BeFalse();  // 128 >= 128
        result.GetBoolean(2).Should().BeFalse();  // 255 > 254
    }

    #endregion

    #region Scalar vs Scalar Tests (Test 7)

    [TestMethod]
    public async Task Equal_ScalarVsScalar()
    {
        // NumPy: np.array(3) == np.array(5) = False (shape: ())
        var s1 = NDArray.Scalar(3);
        var s2 = NDArray.Scalar(5);
        var result = s1 == s2;

        result.Shape.IsScalar.Should().BeTrue();
        result.GetBoolean().Should().BeFalse();
    }

    [TestMethod]
    public async Task Less_ScalarVsScalar()
    {
        // NumPy: np.array(3) < np.array(5) = True
        var s1 = NDArray.Scalar(3);
        var s2 = NDArray.Scalar(5);
        var result = s1 < s2;

        result.Shape.IsScalar.Should().BeTrue();
        result.GetBoolean().Should().BeTrue();
    }

    [TestMethod]
    public async Task GreaterEqual_ScalarVsScalar()
    {
        // NumPy: np.array(3) >= np.array(5) = False
        var s1 = NDArray.Scalar(3);
        var s2 = NDArray.Scalar(5);
        var result = s1 >= s2;

        result.Shape.IsScalar.Should().BeTrue();
        result.GetBoolean().Should().BeFalse();
    }

    #endregion

    #region Float Edge Cases Tests (Test 8)

    [TestMethod]
    public async Task Equal_FloatWithNaN()
    {
        // NumPy: [1.0, nan, inf, -inf, 0.0] == [1.0, nan, inf, -inf, 0.0] = [True, False, True, True, True]
        // NaN != NaN by IEEE 754 standard
        var floats = np.array(new[] { 1.0, double.NaN, double.PositiveInfinity, double.NegativeInfinity, 0.0 });
        var result = floats == floats;

        result.GetBoolean(0).Should().BeTrue();   // 1.0 == 1.0
        result.GetBoolean(1).Should().BeFalse();  // NaN != NaN (IEEE 754)
        result.GetBoolean(2).Should().BeTrue();   // Inf == Inf
        result.GetBoolean(3).Should().BeTrue();   // -Inf == -Inf
        result.GetBoolean(4).Should().BeTrue();   // 0.0 == 0.0
    }

    [TestMethod]
    public async Task Less_FloatWithInfinity()
    {
        // NumPy: [1.0, nan, inf, -inf, 0.0] < 1.0 = [False, False, False, True, True]
        var floats = np.array(new[] { 1.0, double.NaN, double.PositiveInfinity, double.NegativeInfinity, 0.0 });
        var result = floats < 1.0;

        result.GetBoolean(0).Should().BeFalse();  // 1.0 < 1.0 = False
        result.GetBoolean(1).Should().BeFalse();  // NaN < anything = False
        result.GetBoolean(2).Should().BeFalse();  // Inf < 1.0 = False
        result.GetBoolean(3).Should().BeTrue();   // -Inf < 1.0 = True
        result.GetBoolean(4).Should().BeTrue();   // 0.0 < 1.0 = True
    }

    #endregion

    #region All dtypes tests

    [TestMethod]
    public async Task Equal_Int16()
    {
        var a = np.array(new short[] { 1, 2, 3 });
        var b = np.array(new short[] { 1, 3, 3 });
        var result = a == b;

        result.GetBoolean(0).Should().BeTrue();
        result.GetBoolean(1).Should().BeFalse();
        result.GetBoolean(2).Should().BeTrue();
    }

    [TestMethod]
    public async Task Equal_UInt16()
    {
        var a = np.array(new ushort[] { 1, 2, 3 });
        var b = np.array(new ushort[] { 1, 3, 3 });
        var result = a == b;

        result.GetBoolean(0).Should().BeTrue();
        result.GetBoolean(1).Should().BeFalse();
        result.GetBoolean(2).Should().BeTrue();
    }

    [TestMethod]
    public async Task Equal_UInt32()
    {
        var a = np.array(new uint[] { 1, 2, 3 });
        var b = np.array(new uint[] { 1, 3, 3 });
        var result = a == b;

        result.GetBoolean(0).Should().BeTrue();
        result.GetBoolean(1).Should().BeFalse();
        result.GetBoolean(2).Should().BeTrue();
    }

    [TestMethod]
    public async Task Equal_Int64()
    {
        var a = np.array(new long[] { 1, 2, 3 });
        var b = np.array(new long[] { 1, 3, 3 });
        var result = a == b;

        result.GetBoolean(0).Should().BeTrue();
        result.GetBoolean(1).Should().BeFalse();
        result.GetBoolean(2).Should().BeTrue();
    }

    [TestMethod]
    public async Task Equal_UInt64()
    {
        var a = np.array(new ulong[] { 1, 2, 3 });
        var b = np.array(new ulong[] { 1, 3, 3 });
        var result = a == b;

        result.GetBoolean(0).Should().BeTrue();
        result.GetBoolean(1).Should().BeFalse();
        result.GetBoolean(2).Should().BeTrue();
    }

    [TestMethod]
    public async Task Equal_Single()
    {
        var a = np.array(new float[] { 1.0f, 2.0f, 3.0f });
        var b = np.array(new float[] { 1.0f, 3.0f, 3.0f });
        var result = a == b;

        result.GetBoolean(0).Should().BeTrue();
        result.GetBoolean(1).Should().BeFalse();
        result.GetBoolean(2).Should().BeTrue();
    }

    [TestMethod]
    public async Task Equal_Decimal()
    {
        var a = np.array(new decimal[] { 1.0m, 2.0m, 3.0m });
        var b = np.array(new decimal[] { 1.0m, 3.0m, 3.0m });
        var result = a == b;

        result.GetBoolean(0).Should().BeTrue();
        result.GetBoolean(1).Should().BeFalse();
        result.GetBoolean(2).Should().BeTrue();
    }

    [TestMethod]
    public async Task Equal_Char()
    {
        var a = np.array(new char[] { 'a', 'b', 'c' });
        var b = np.array(new char[] { 'a', 'x', 'c' });
        var result = a == b;

        result.GetBoolean(0).Should().BeTrue();
        result.GetBoolean(1).Should().BeFalse();
        result.GetBoolean(2).Should().BeTrue();
    }

    #endregion
}
