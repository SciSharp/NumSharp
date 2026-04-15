using System;
using System.Threading.Tasks;
using NumSharp;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Sorting;

/// <summary>
/// Tests for np.argsort behavior with NaN, Inf, and -Inf values.
/// NumPy behavior:
/// - NaN sorts to the end (after +Inf)
/// - -Inf sorts to the beginning
/// - +Inf sorts between normal values and NaN
/// </summary>
[TestClass]
public class ArgsortNaNTests
{
    #region NaN Handling

    [TestMethod]
    public async Task Argsort_NaN_SortsToEnd()
    {
        // NumPy: np.argsort([nan, 1, 2]) = [1, 2, 0]
        // NaN should be at the end
        var a = np.array(new double[] { double.NaN, 1.0, 2.0 });
        var result = np.argsort<double>(a);

        result.GetInt64(0).Should().Be(1L);
        result.GetInt64(1).Should().Be(2L);
        result.GetInt64(2).Should().Be(0L);
    }

    [TestMethod]
    public async Task Argsort_MultipleNaN_SortToEnd()
    {
        // NumPy: np.argsort([nan, 1, nan, 2]) = [1, 3, 0, 2]
        // All NaN values at the end, maintaining relative order
        var a = np.array(new double[] { double.NaN, 1.0, double.NaN, 2.0 });
        var result = np.argsort<double>(a);

        result.GetInt64(0).Should().Be(1L); // 1.0
        result.GetInt64(1).Should().Be(3L); // 2.0
        // NaN values at end
        double.IsNaN(a.GetDouble((int)result.GetInt64(2))).Should().BeTrue();
        double.IsNaN(a.GetDouble((int)result.GetInt64(3))).Should().BeTrue();
    }

    #endregion

    #region Inf Handling

    [TestMethod]
    public async Task Argsort_Inf_SortsCorrectly()
    {
        // NumPy: np.argsort([inf, -inf, 0]) = [1, 2, 0]
        // -inf first, then 0, then inf
        var a = np.array(new double[] { double.PositiveInfinity, double.NegativeInfinity, 0.0 });
        var result = np.argsort<double>(a);

        result.GetInt64(0).Should().Be(1L); // -inf
        result.GetInt64(1).Should().Be(2L); // 0
        result.GetInt64(2).Should().Be(0L); // +inf
    }

    [TestMethod]
    public async Task Argsort_InfAndNaN_SortsCorrectly()
    {
        // NumPy: np.argsort([nan, inf, -inf, 0]) = [2, 3, 1, 0]
        // Order: -inf, 0, +inf, nan
        var a = np.array(new double[] { double.NaN, double.PositiveInfinity, double.NegativeInfinity, 0.0 });
        var result = np.argsort<double>(a);

        result.GetInt64(0).Should().Be(2L); // -inf
        result.GetInt64(1).Should().Be(3L); // 0
        result.GetInt64(2).Should().Be(1L); // +inf
        result.GetInt64(3).Should().Be(0L); // nan
    }

    #endregion

    #region Float32 Tests

    [TestMethod]
    public async Task Argsort_Float32_NaN_SortsToEnd()
    {
        // Same behavior for float32
        var a = np.array(new float[] { float.NaN, 1.0f, 2.0f });
        var result = np.argsort<float>(a);

        result.GetInt64(0).Should().Be(1L);
        result.GetInt64(1).Should().Be(2L);
        result.GetInt64(2).Should().Be(0L);
    }

    [TestMethod]
    public async Task Argsort_Float32_InfAndNaN_SortsCorrectly()
    {
        var a = np.array(new float[] { float.NaN, float.PositiveInfinity, float.NegativeInfinity, 0.0f });
        var result = np.argsort<float>(a);

        result.GetInt64(0).Should().Be(2L); // -inf
        result.GetInt64(1).Should().Be(3L); // 0
        result.GetInt64(2).Should().Be(1L); // +inf
        result.GetInt64(3).Should().Be(0L); // nan
    }

    #endregion

    #region Basic Argsort (No NaN)

    [TestMethod]
    public async Task Argsort_Normal_SortsCorrectly()
    {
        // NumPy: np.argsort([3, 1, 2]) = [1, 2, 0]
        var a = np.array(new double[] { 3.0, 1.0, 2.0 });
        var result = np.argsort<double>(a);

        result.GetInt64(0).Should().Be(1L);
        result.GetInt64(1).Should().Be(2L);
        result.GetInt64(2).Should().Be(0L);
    }

    [TestMethod]
    public async Task Argsort_AlreadySorted_ReturnsSequentialIndices()
    {
        // NumPy: np.argsort([1, 2, 3]) = [0, 1, 2]
        var a = np.array(new double[] { 1.0, 2.0, 3.0 });
        var result = np.argsort<double>(a);

        result.GetInt64(0).Should().Be(0L);
        result.GetInt64(1).Should().Be(1L);
        result.GetInt64(2).Should().Be(2L);
    }

    [TestMethod]
    public async Task Argsort_ReverseSorted_ReturnsReverseIndices()
    {
        // NumPy: np.argsort([3, 2, 1]) = [2, 1, 0]
        var a = np.array(new double[] { 3.0, 2.0, 1.0 });
        var result = np.argsort<double>(a);

        result.GetInt64(0).Should().Be(2L);
        result.GetInt64(1).Should().Be(1L);
        result.GetInt64(2).Should().Be(0L);
    }

    #endregion

    #region Integer Types (No NaN)

    [TestMethod]
    public async Task Argsort_Int32_SortsCorrectly()
    {
        var a = np.array(new int[] { 3, 1, 2 });
        var result = np.argsort<int>(a);

        result.GetInt64(0).Should().Be(1L);
        result.GetInt64(1).Should().Be(2L);
        result.GetInt64(2).Should().Be(0L);
    }

    #endregion
}
