using System;
using System.Threading.Tasks;
using NumSharp;
using NumSharp.UnitTest.Utilities;
using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace NumSharp.UnitTest.APIs;

/// <summary>
/// Tests for np.count_nonzero.
/// NumPy: np.count_nonzero([0, 1, 0, 2]) = 2
/// </summary>
public class CountNonzeroTests
{
    #region Element-wise (No Axis)

    [Test]
    public async Task CountNonzero_1D_ReturnsCount()
    {
        // NumPy: np.count_nonzero([0, 1, 0, 2]) = 2
        var a = np.array(new int[] { 0, 1, 0, 2 });
        var result = np.count_nonzero(a);

        await Assert.That(result).IsEqualTo(2);
    }

    [Test]
    public async Task CountNonzero_1D_AllZero_ReturnsZero()
    {
        // NumPy: np.count_nonzero([0, 0, 0]) = 0
        var a = np.array(new int[] { 0, 0, 0 });
        var result = np.count_nonzero(a);

        await Assert.That(result).IsEqualTo(0);
    }

    [Test]
    public async Task CountNonzero_1D_AllNonzero_ReturnsSize()
    {
        // NumPy: np.count_nonzero([1, 2, 3]) = 3
        var a = np.array(new int[] { 1, 2, 3 });
        var result = np.count_nonzero(a);

        await Assert.That(result).IsEqualTo(3);
    }

    [Test]
    public async Task CountNonzero_2D_ReturnsTotal()
    {
        // NumPy: np.count_nonzero([[0, 1], [2, 0]]) = 2
        var a = np.array(new int[,] { { 0, 1 }, { 2, 0 } });
        var result = np.count_nonzero(a);

        await Assert.That(result).IsEqualTo(2);
    }

    [Test]
    public async Task CountNonzero_Empty_ReturnsZero()
    {
        // NumPy: np.count_nonzero([]) = 0
        var a = np.array(new int[0]);
        var result = np.count_nonzero(a);

        await Assert.That(result).IsEqualTo(0);
    }

    #endregion

    #region Boolean Arrays

    [Test]
    public async Task CountNonzero_Boolean_TrueIsNonzero()
    {
        // NumPy: np.count_nonzero([False, True, False, True]) = 2
        var a = np.array(new bool[] { false, true, false, true });
        var result = np.count_nonzero(a);

        await Assert.That(result).IsEqualTo(2);
    }

    #endregion

    #region Float Arrays

    [Test]
    public async Task CountNonzero_Float_ZeroIsExact()
    {
        // NumPy: np.count_nonzero([0.0, 1.0, 0.0, 2.0]) = 2
        var a = np.array(new double[] { 0.0, 1.0, 0.0, 2.0 });
        var result = np.count_nonzero(a);

        await Assert.That(result).IsEqualTo(2);
    }

    [Test]
    public async Task CountNonzero_Float_NaN_IsNonzero()
    {
        // NumPy: np.count_nonzero([0.0, nan, 0.0]) = 1
        var a = np.array(new double[] { 0.0, double.NaN, 0.0 });
        var result = np.count_nonzero(a);

        await Assert.That(result).IsEqualTo(1);
    }

    #endregion

    #region Axis Reduction

    [Test]
    public async Task CountNonzero_2D_Axis0()
    {
        // NumPy: np.count_nonzero([[0, 1, 2], [3, 0, 0]], axis=0) = [1, 1, 1]
        var a = np.array(new int[,] { { 0, 1, 2 }, { 3, 0, 0 } });
        var result = np.count_nonzero(a, axis: 0);

        await Assert.That(result.shape).IsEquivalentTo(new long[] { 3 });
        await Assert.That(result.GetInt64(0)).IsEqualTo(1L);
        await Assert.That(result.GetInt64(1)).IsEqualTo(1L);
        await Assert.That(result.GetInt64(2)).IsEqualTo(1L);
    }

    [Test]
    public async Task CountNonzero_2D_Axis1()
    {
        // NumPy: np.count_nonzero([[0, 1, 2], [3, 0, 0]], axis=1) = [2, 1]
        var a = np.array(new int[,] { { 0, 1, 2 }, { 3, 0, 0 } });
        var result = np.count_nonzero(a, axis: 1);

        await Assert.That(result.shape).IsEquivalentTo(new long[] { 2 });
        await Assert.That(result.GetInt64(0)).IsEqualTo(2L);
        await Assert.That(result.GetInt64(1)).IsEqualTo(1L);
    }

    [Test]
    public async Task CountNonzero_2D_Axis0_Keepdims()
    {
        // NumPy: np.count_nonzero([[0, 1, 2], [3, 0, 0]], axis=0, keepdims=True) = [[1, 1, 1]] (shape 1,3)
        var a = np.array(new int[,] { { 0, 1, 2 }, { 3, 0, 0 } });
        var result = np.count_nonzero(a, axis: 0, keepdims: true);

        await Assert.That(result.shape).IsEquivalentTo(new long[] { 1, 3 });
        // Access via 2D indexing since keepdims preserves dimensions
        await Assert.That(result.GetAtIndex<long>(0)).IsEqualTo(1L);
        await Assert.That(result.GetAtIndex<long>(1)).IsEqualTo(1L);
        await Assert.That(result.GetAtIndex<long>(2)).IsEqualTo(1L);
    }

    [Test]
    public async Task CountNonzero_2D_NegativeAxis()
    {
        // NumPy: np.count_nonzero([[0, 1, 2], [3, 0, 0]], axis=-1) = [2, 1]
        var a = np.array(new int[,] { { 0, 1, 2 }, { 3, 0, 0 } });
        var result = np.count_nonzero(a, axis: -1);

        await Assert.That(result.shape).IsEquivalentTo(new long[] { 2 });
        await Assert.That(result.GetInt64(0)).IsEqualTo(2L);
        await Assert.That(result.GetInt64(1)).IsEqualTo(1L);
    }

    [Test]
    public async Task CountNonzero_3D_Axis1()
    {
        // 3D array with shape (2, 2, 3) reduced along axis 1 gives shape (2, 3)
        var a = np.zeros(new Shape(2, 2, 3), NPTypeCode.Int32);
        a.SetInt32(1, 0, 0, 0); // [0,0,0] = 1
        a.SetInt32(1, 0, 1, 1); // [0,1,1] = 1
        a.SetInt32(1, 1, 0, 2); // [1,0,2] = 1
        a.SetInt32(1, 1, 1, 0); // [1,1,0] = 1

        var result = np.count_nonzero(a, axis: 1);

        await Assert.That(result.shape).IsEquivalentTo(new long[] { 2, 3 });
        // [0,0]: sum axis 1 of [0,:,0] = [1,0] -> 1 nonzero
        // [0,1]: sum axis 1 of [0,:,1] = [0,1] -> 1 nonzero
        // [0,2]: sum axis 1 of [0,:,2] = [0,0] -> 0 nonzero
        await Assert.That(result.GetAtIndex<long>(0)).IsEqualTo(1L);
        await Assert.That(result.GetAtIndex<long>(1)).IsEqualTo(1L);
        await Assert.That(result.GetAtIndex<long>(2)).IsEqualTo(0L);
    }

    #endregion

    #region Empty Array Axis Reduction

    [Test]
    public async Task CountNonzero_Empty2D_Axis0_ReturnsZeros()
    {
        // NumPy: np.count_nonzero(np.zeros((0, 3)), axis=0) = [0, 0, 0]
        var a = np.zeros(new Shape(0, 3), NPTypeCode.Int32);
        var result = np.count_nonzero(a, axis: 0);

        await Assert.That(result.shape).IsEquivalentTo(new long[] { 3 });
        await Assert.That(result.GetInt64(0)).IsEqualTo(0L);
        await Assert.That(result.GetInt64(1)).IsEqualTo(0L);
        await Assert.That(result.GetInt64(2)).IsEqualTo(0L);
    }

    [Test]
    public async Task CountNonzero_Empty2D_Axis1_ReturnsEmpty()
    {
        // NumPy: np.count_nonzero(np.zeros((0, 3)), axis=1) = [] (shape (0,))
        var a = np.zeros(new Shape(0, 3), NPTypeCode.Int32);
        var result = np.count_nonzero(a, axis: 1);

        await Assert.That(result.shape).IsEquivalentTo(new long[] { 0 });
        await Assert.That(result.size).IsEqualTo(0);
    }

    #endregion
}
