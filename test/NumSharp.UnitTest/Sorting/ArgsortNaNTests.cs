using System;
using System.Threading.Tasks;
using NumSharp;
using NumSharp.UnitTest.Utilities;
using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace NumSharp.UnitTest.Sorting;

/// <summary>
/// Tests for np.argsort behavior with NaN, Inf, and -Inf values.
/// NumPy behavior:
/// - NaN sorts to the end (after +Inf)
/// - -Inf sorts to the beginning
/// - +Inf sorts between normal values and NaN
/// </summary>
public class ArgsortNaNTests
{
    #region NaN Handling

    [Test]
    public async Task Argsort_NaN_SortsToEnd()
    {
        // NumPy: np.argsort([nan, 1, 2]) = [1, 2, 0]
        // NaN should be at the end
        var a = np.array(new double[] { double.NaN, 1.0, 2.0 });
        var result = np.argsort<double>(a);

        await Assert.That(result.GetInt64(0)).IsEqualTo(1L);
        await Assert.That(result.GetInt64(1)).IsEqualTo(2L);
        await Assert.That(result.GetInt64(2)).IsEqualTo(0L);
    }

    [Test]
    public async Task Argsort_MultipleNaN_SortToEnd()
    {
        // NumPy: np.argsort([nan, 1, nan, 2]) = [1, 3, 0, 2]
        // All NaN values at the end, maintaining relative order
        var a = np.array(new double[] { double.NaN, 1.0, double.NaN, 2.0 });
        var result = np.argsort<double>(a);

        await Assert.That(result.GetInt64(0)).IsEqualTo(1L); // 1.0
        await Assert.That(result.GetInt64(1)).IsEqualTo(3L); // 2.0
        // NaN values at end
        await Assert.That(double.IsNaN(a.GetDouble((int)result.GetInt64(2)))).IsTrue();
        await Assert.That(double.IsNaN(a.GetDouble((int)result.GetInt64(3)))).IsTrue();
    }

    #endregion

    #region Inf Handling

    [Test]
    public async Task Argsort_Inf_SortsCorrectly()
    {
        // NumPy: np.argsort([inf, -inf, 0]) = [1, 2, 0]
        // -inf first, then 0, then inf
        var a = np.array(new double[] { double.PositiveInfinity, double.NegativeInfinity, 0.0 });
        var result = np.argsort<double>(a);

        await Assert.That(result.GetInt64(0)).IsEqualTo(1L); // -inf
        await Assert.That(result.GetInt64(1)).IsEqualTo(2L); // 0
        await Assert.That(result.GetInt64(2)).IsEqualTo(0L); // +inf
    }

    [Test]
    public async Task Argsort_InfAndNaN_SortsCorrectly()
    {
        // NumPy: np.argsort([nan, inf, -inf, 0]) = [2, 3, 1, 0]
        // Order: -inf, 0, +inf, nan
        var a = np.array(new double[] { double.NaN, double.PositiveInfinity, double.NegativeInfinity, 0.0 });
        var result = np.argsort<double>(a);

        await Assert.That(result.GetInt64(0)).IsEqualTo(2L); // -inf
        await Assert.That(result.GetInt64(1)).IsEqualTo(3L); // 0
        await Assert.That(result.GetInt64(2)).IsEqualTo(1L); // +inf
        await Assert.That(result.GetInt64(3)).IsEqualTo(0L); // nan
    }

    #endregion

    #region Float32 Tests

    [Test]
    public async Task Argsort_Float32_NaN_SortsToEnd()
    {
        // Same behavior for float32
        var a = np.array(new float[] { float.NaN, 1.0f, 2.0f });
        var result = np.argsort<float>(a);

        await Assert.That(result.GetInt64(0)).IsEqualTo(1L);
        await Assert.That(result.GetInt64(1)).IsEqualTo(2L);
        await Assert.That(result.GetInt64(2)).IsEqualTo(0L);
    }

    [Test]
    public async Task Argsort_Float32_InfAndNaN_SortsCorrectly()
    {
        var a = np.array(new float[] { float.NaN, float.PositiveInfinity, float.NegativeInfinity, 0.0f });
        var result = np.argsort<float>(a);

        await Assert.That(result.GetInt64(0)).IsEqualTo(2L); // -inf
        await Assert.That(result.GetInt64(1)).IsEqualTo(3L); // 0
        await Assert.That(result.GetInt64(2)).IsEqualTo(1L); // +inf
        await Assert.That(result.GetInt64(3)).IsEqualTo(0L); // nan
    }

    #endregion

    #region Basic Argsort (No NaN)

    [Test]
    public async Task Argsort_Normal_SortsCorrectly()
    {
        // NumPy: np.argsort([3, 1, 2]) = [1, 2, 0]
        var a = np.array(new double[] { 3.0, 1.0, 2.0 });
        var result = np.argsort<double>(a);

        await Assert.That(result.GetInt64(0)).IsEqualTo(1L);
        await Assert.That(result.GetInt64(1)).IsEqualTo(2L);
        await Assert.That(result.GetInt64(2)).IsEqualTo(0L);
    }

    [Test]
    public async Task Argsort_AlreadySorted_ReturnsSequentialIndices()
    {
        // NumPy: np.argsort([1, 2, 3]) = [0, 1, 2]
        var a = np.array(new double[] { 1.0, 2.0, 3.0 });
        var result = np.argsort<double>(a);

        await Assert.That(result.GetInt64(0)).IsEqualTo(0L);
        await Assert.That(result.GetInt64(1)).IsEqualTo(1L);
        await Assert.That(result.GetInt64(2)).IsEqualTo(2L);
    }

    [Test]
    public async Task Argsort_ReverseSorted_ReturnsReverseIndices()
    {
        // NumPy: np.argsort([3, 2, 1]) = [2, 1, 0]
        var a = np.array(new double[] { 3.0, 2.0, 1.0 });
        var result = np.argsort<double>(a);

        await Assert.That(result.GetInt64(0)).IsEqualTo(2L);
        await Assert.That(result.GetInt64(1)).IsEqualTo(1L);
        await Assert.That(result.GetInt64(2)).IsEqualTo(0L);
    }

    #endregion

    #region Integer Types (No NaN)

    [Test]
    public async Task Argsort_Int32_SortsCorrectly()
    {
        var a = np.array(new int[] { 3, 1, 2 });
        var result = np.argsort<int>(a);

        await Assert.That(result.GetInt64(0)).IsEqualTo(1L);
        await Assert.That(result.GetInt64(1)).IsEqualTo(2L);
        await Assert.That(result.GetInt64(2)).IsEqualTo(0L);
    }

    #endregion
}
