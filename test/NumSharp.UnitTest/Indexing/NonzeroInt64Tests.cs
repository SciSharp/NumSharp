using System;
using System.Threading.Tasks;
using NumSharp;
using NumSharp.UnitTest.Utilities;
using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace NumSharp.UnitTest.Indexing;

/// <summary>
/// Tests for np.nonzero with int64 indexing support.
/// Verifies nonzero returns int64 indices and handles various dtypes.
/// </summary>
public class NonzeroInt64Tests
{
    #region Return Type Tests

    [Test]
    public async Task Nonzero_ReturnsInt64Indices()
    {
        // NumPy: np.nonzero([0, 1, 0, 2])[0].dtype = int64
        var a = np.array(new int[] { 0, 1, 0, 2 });
        var result = np.nonzero(a);

        await Assert.That(result[0].typecode).IsEqualTo(NPTypeCode.Int64);
    }

    [Test]
    public async Task Nonzero_2D_AllArraysAreInt64()
    {
        var a = np.array(new int[,] { { 0, 1 }, { 2, 0 } });
        var result = np.nonzero(a);

        await Assert.That(result.Length).IsEqualTo(2);
        await Assert.That(result[0].typecode).IsEqualTo(NPTypeCode.Int64);
        await Assert.That(result[1].typecode).IsEqualTo(NPTypeCode.Int64);
    }

    [Test]
    public async Task Nonzero_3D_AllArraysAreInt64()
    {
        var a = np.zeros(new Shape(2, 2, 2), NPTypeCode.Int32);
        a.SetInt32(1, 0, 0, 0);
        a.SetInt32(1, 1, 1, 1);

        var result = np.nonzero(a);

        await Assert.That(result.Length).IsEqualTo(3);
        await Assert.That(result[0].typecode).IsEqualTo(NPTypeCode.Int64);
        await Assert.That(result[1].typecode).IsEqualTo(NPTypeCode.Int64);
        await Assert.That(result[2].typecode).IsEqualTo(NPTypeCode.Int64);
    }

    #endregion

    #region Various DType Tests

    [Test]
    public async Task Nonzero_Byte_ReturnsCorrectIndices()
    {
        var a = np.array(new byte[] { 0, 1, 0, 2, 0 });
        var result = np.nonzero(a);

        await Assert.That(result[0].size).IsEqualTo(2);
        await Assert.That(result[0].GetInt64(0)).IsEqualTo(1L);
        await Assert.That(result[0].GetInt64(1)).IsEqualTo(3L);
    }

    [Test]
    public async Task Nonzero_Int16_ReturnsCorrectIndices()
    {
        var a = np.array(new short[] { 0, -1, 0, 2, 0 });
        var result = np.nonzero(a);

        await Assert.That(result[0].size).IsEqualTo(2);
        await Assert.That(result[0].GetInt64(0)).IsEqualTo(1L);
        await Assert.That(result[0].GetInt64(1)).IsEqualTo(3L);
    }

    [Test]
    public async Task Nonzero_UInt16_ReturnsCorrectIndices()
    {
        var a = np.array(new ushort[] { 0, 1, 0, 2, 0 });
        var result = np.nonzero(a);

        await Assert.That(result[0].size).IsEqualTo(2);
        await Assert.That(result[0].GetInt64(0)).IsEqualTo(1L);
        await Assert.That(result[0].GetInt64(1)).IsEqualTo(3L);
    }

    [Test]
    public async Task Nonzero_Int64_ReturnsCorrectIndices()
    {
        var a = np.array(new long[] { 0, 1, 0, 2, 0 });
        var result = np.nonzero(a);

        await Assert.That(result[0].size).IsEqualTo(2);
        await Assert.That(result[0].GetInt64(0)).IsEqualTo(1L);
        await Assert.That(result[0].GetInt64(1)).IsEqualTo(3L);
    }

    [Test]
    public async Task Nonzero_UInt64_ReturnsCorrectIndices()
    {
        var a = np.array(new ulong[] { 0, 1, 0, 2, 0 });
        var result = np.nonzero(a);

        await Assert.That(result[0].size).IsEqualTo(2);
        await Assert.That(result[0].GetInt64(0)).IsEqualTo(1L);
        await Assert.That(result[0].GetInt64(1)).IsEqualTo(3L);
    }

    [Test]
    public async Task Nonzero_Float32_ReturnsCorrectIndices()
    {
        var a = np.array(new float[] { 0f, 1.5f, 0f, -2.5f, 0f });
        var result = np.nonzero(a);

        await Assert.That(result[0].size).IsEqualTo(2);
        await Assert.That(result[0].GetInt64(0)).IsEqualTo(1L);
        await Assert.That(result[0].GetInt64(1)).IsEqualTo(3L);
    }

    [Test]
    public async Task Nonzero_Float64_ReturnsCorrectIndices()
    {
        var a = np.array(new double[] { 0.0, 1.5, 0.0, -2.5, 0.0 });
        var result = np.nonzero(a);

        await Assert.That(result[0].size).IsEqualTo(2);
        await Assert.That(result[0].GetInt64(0)).IsEqualTo(1L);
        await Assert.That(result[0].GetInt64(1)).IsEqualTo(3L);
    }

    [Test]
    public async Task Nonzero_Decimal_ReturnsCorrectIndices()
    {
        var a = np.array(new decimal[] { 0m, 1.5m, 0m, -2.5m, 0m });
        var result = np.nonzero(a);

        await Assert.That(result[0].size).IsEqualTo(2);
        await Assert.That(result[0].GetInt64(0)).IsEqualTo(1L);
        await Assert.That(result[0].GetInt64(1)).IsEqualTo(3L);
    }

    #endregion

    #region Boolean Array Tests

    [Test]
    public async Task Nonzero_Boolean_TrueIsNonzero()
    {
        var a = np.array(new bool[] { false, true, false, true, false });
        var result = np.nonzero(a);

        await Assert.That(result[0].typecode).IsEqualTo(NPTypeCode.Int64);
        await Assert.That(result[0].size).IsEqualTo(2);
        await Assert.That(result[0].GetInt64(0)).IsEqualTo(1L);
        await Assert.That(result[0].GetInt64(1)).IsEqualTo(3L);
    }

    [Test]
    public async Task Nonzero_Boolean_2D_ReturnsCorrectIndices()
    {
        var a = np.array(new bool[,] { { false, true }, { true, false } });
        var result = np.nonzero(a);

        await Assert.That(result.Length).IsEqualTo(2);
        await Assert.That(result[0].size).IsEqualTo(2);

        // (0,1) and (1,0) are true
        await Assert.That(result[0].GetInt64(0)).IsEqualTo(0L);
        await Assert.That(result[0].GetInt64(1)).IsEqualTo(1L);
        await Assert.That(result[1].GetInt64(0)).IsEqualTo(1L);
        await Assert.That(result[1].GetInt64(1)).IsEqualTo(0L);
    }

    #endregion

    #region Special Float Values

    [Test]
    public async Task Nonzero_NaN_IsNonzero()
    {
        // NumPy: NaN is considered nonzero
        var a = np.array(new double[] { 0.0, double.NaN, 0.0 });
        var result = np.nonzero(a);

        await Assert.That(result[0].size).IsEqualTo(1);
        await Assert.That(result[0].GetInt64(0)).IsEqualTo(1L);
    }

    [Test]
    public async Task Nonzero_PositiveInfinity_IsNonzero()
    {
        var a = np.array(new double[] { 0.0, double.PositiveInfinity, 0.0 });
        var result = np.nonzero(a);

        await Assert.That(result[0].size).IsEqualTo(1);
        await Assert.That(result[0].GetInt64(0)).IsEqualTo(1L);
    }

    [Test]
    public async Task Nonzero_NegativeInfinity_IsNonzero()
    {
        var a = np.array(new double[] { 0.0, double.NegativeInfinity, 0.0 });
        var result = np.nonzero(a);

        await Assert.That(result[0].size).IsEqualTo(1);
        await Assert.That(result[0].GetInt64(0)).IsEqualTo(1L);
    }

    [Test]
    public async Task Nonzero_NegativeZero_IsZero()
    {
        // NumPy: -0.0 is still zero
        var a = np.array(new double[] { 1.0, -0.0, 2.0 });
        var result = np.nonzero(a);

        await Assert.That(result[0].size).IsEqualTo(2);
        await Assert.That(result[0].GetInt64(0)).IsEqualTo(0L);
        await Assert.That(result[0].GetInt64(1)).IsEqualTo(2L);
    }

    #endregion

    #region Edge Cases

    [Test]
    public async Task Nonzero_AllZeros_ReturnsEmptyArray()
    {
        var a = np.zeros(new Shape(10), NPTypeCode.Int32);
        var result = np.nonzero(a);

        await Assert.That(result[0].size).IsEqualTo(0);
        await Assert.That(result[0].typecode).IsEqualTo(NPTypeCode.Int64);
    }

    [Test]
    public async Task Nonzero_AllNonzero_ReturnsAllIndices()
    {
        var a = np.ones(new Shape(5), NPTypeCode.Int32);
        var result = np.nonzero(a);

        await Assert.That(result[0].size).IsEqualTo(5);
        for (int i = 0; i < 5; i++)
        {
            await Assert.That(result[0].GetInt64(i)).IsEqualTo((long)i);
        }
    }

    [Test]
    public async Task Nonzero_SingleNonzero_ReturnsOneIndex()
    {
        var a = np.zeros(new Shape(10), NPTypeCode.Int32);
        a.SetInt32(1, 5);

        var result = np.nonzero(a);

        await Assert.That(result[0].size).IsEqualTo(1);
        await Assert.That(result[0].GetInt64(0)).IsEqualTo(5L);
    }

    #endregion

    #region Using Result for Indexing

    [Test]
    public async Task Nonzero_CanBeUsedForIndexing()
    {
        // NumPy pattern: a[np.nonzero(a)] extracts nonzero values
        var a = np.array(new int[] { 0, 3, 0, 1, 0, 2 });
        var indices = np.nonzero(a);

        var nonzeroValues = a[indices];

        await Assert.That(nonzeroValues.size).IsEqualTo(3);
        await Assert.That(nonzeroValues.GetInt32(0)).IsEqualTo(3);
        await Assert.That(nonzeroValues.GetInt32(1)).IsEqualTo(1);
        await Assert.That(nonzeroValues.GetInt32(2)).IsEqualTo(2);
    }

    #endregion
}
