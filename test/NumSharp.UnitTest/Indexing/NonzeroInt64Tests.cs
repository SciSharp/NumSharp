using System;
using NumSharp;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Indexing;

/// <summary>
/// Tests for np.nonzero with int64 indexing support.
/// Verifies nonzero returns int64 indices and handles various dtypes.
/// </summary>
[TestClass]
public class NonzeroInt64Tests
{
    #region Return Type Tests

    [TestMethod]
    public void Nonzero_ReturnsInt64Indices()
    {
        // NumPy: np.nonzero([0, 1, 0, 2])[0].dtype = int64
        var a = np.array(new int[] { 0, 1, 0, 2 });
        var result = np.nonzero(a);

        result[0].typecode.Should().Be(NPTypeCode.Int64);
    }

    [TestMethod]
    public void Nonzero_2D_AllArraysAreInt64()
    {
        var a = np.array(new int[,] { { 0, 1 }, { 2, 0 } });
        var result = np.nonzero(a);

        result.Length.Should().Be(2);
        result[0].typecode.Should().Be(NPTypeCode.Int64);
        result[1].typecode.Should().Be(NPTypeCode.Int64);
    }

    [TestMethod]
    public void Nonzero_3D_AllArraysAreInt64()
    {
        var a = np.zeros(new Shape(2, 2, 2), NPTypeCode.Int32);
        a.SetInt32(1, 0, 0, 0);
        a.SetInt32(1, 1, 1, 1);

        var result = np.nonzero(a);

        result.Length.Should().Be(3);
        result[0].typecode.Should().Be(NPTypeCode.Int64);
        result[1].typecode.Should().Be(NPTypeCode.Int64);
        result[2].typecode.Should().Be(NPTypeCode.Int64);
    }

    #endregion

    #region Various DType Tests

    [TestMethod]
    public void Nonzero_Byte_ReturnsCorrectIndices()
    {
        var a = np.array(new byte[] { 0, 1, 0, 2, 0 });
        var result = np.nonzero(a);

        result[0].size.Should().Be(2);
        result[0].GetInt64(0).Should().Be(1L);
        result[0].GetInt64(1).Should().Be(3L);
    }

    [TestMethod]
    public void Nonzero_Int16_ReturnsCorrectIndices()
    {
        var a = np.array(new short[] { 0, -1, 0, 2, 0 });
        var result = np.nonzero(a);

        result[0].size.Should().Be(2);
        result[0].GetInt64(0).Should().Be(1L);
        result[0].GetInt64(1).Should().Be(3L);
    }

    [TestMethod]
    public void Nonzero_UInt16_ReturnsCorrectIndices()
    {
        var a = np.array(new ushort[] { 0, 1, 0, 2, 0 });
        var result = np.nonzero(a);

        result[0].size.Should().Be(2);
        result[0].GetInt64(0).Should().Be(1L);
        result[0].GetInt64(1).Should().Be(3L);
    }

    [TestMethod]
    public void Nonzero_Int64_ReturnsCorrectIndices()
    {
        var a = np.array(new long[] { 0, 1, 0, 2, 0 });
        var result = np.nonzero(a);

        result[0].size.Should().Be(2);
        result[0].GetInt64(0).Should().Be(1L);
        result[0].GetInt64(1).Should().Be(3L);
    }

    [TestMethod]
    public void Nonzero_UInt64_ReturnsCorrectIndices()
    {
        var a = np.array(new ulong[] { 0, 1, 0, 2, 0 });
        var result = np.nonzero(a);

        result[0].size.Should().Be(2);
        result[0].GetInt64(0).Should().Be(1L);
        result[0].GetInt64(1).Should().Be(3L);
    }

    [TestMethod]
    public void Nonzero_Float32_ReturnsCorrectIndices()
    {
        var a = np.array(new float[] { 0f, 1.5f, 0f, -2.5f, 0f });
        var result = np.nonzero(a);

        result[0].size.Should().Be(2);
        result[0].GetInt64(0).Should().Be(1L);
        result[0].GetInt64(1).Should().Be(3L);
    }

    [TestMethod]
    public void Nonzero_Float64_ReturnsCorrectIndices()
    {
        var a = np.array(new double[] { 0.0, 1.5, 0.0, -2.5, 0.0 });
        var result = np.nonzero(a);

        result[0].size.Should().Be(2);
        result[0].GetInt64(0).Should().Be(1L);
        result[0].GetInt64(1).Should().Be(3L);
    }

    [TestMethod]
    public void Nonzero_Decimal_ReturnsCorrectIndices()
    {
        var a = np.array(new decimal[] { 0m, 1.5m, 0m, -2.5m, 0m });
        var result = np.nonzero(a);

        result[0].size.Should().Be(2);
        result[0].GetInt64(0).Should().Be(1L);
        result[0].GetInt64(1).Should().Be(3L);
    }

    #endregion

    #region Boolean Array Tests

    [TestMethod]
    public void Nonzero_Boolean_TrueIsNonzero()
    {
        var a = np.array(new bool[] { false, true, false, true, false });
        var result = np.nonzero(a);

        result[0].typecode.Should().Be(NPTypeCode.Int64);
        result[0].size.Should().Be(2);
        result[0].GetInt64(0).Should().Be(1L);
        result[0].GetInt64(1).Should().Be(3L);
    }

    [TestMethod]
    public void Nonzero_Boolean_2D_ReturnsCorrectIndices()
    {
        var a = np.array(new bool[,] { { false, true }, { true, false } });
        var result = np.nonzero(a);

        result.Length.Should().Be(2);
        result[0].size.Should().Be(2);

        // (0,1) and (1,0) are true
        result[0].GetInt64(0).Should().Be(0L);
        result[0].GetInt64(1).Should().Be(1L);
        result[1].GetInt64(0).Should().Be(1L);
        result[1].GetInt64(1).Should().Be(0L);
    }

    #endregion

    #region Special Float Values

    [TestMethod]
    public void Nonzero_NaN_IsNonzero()
    {
        // NumPy: NaN is considered nonzero
        var a = np.array(new double[] { 0.0, double.NaN, 0.0 });
        var result = np.nonzero(a);

        result[0].size.Should().Be(1);
        result[0].GetInt64(0).Should().Be(1L);
    }

    [TestMethod]
    public void Nonzero_PositiveInfinity_IsNonzero()
    {
        var a = np.array(new double[] { 0.0, double.PositiveInfinity, 0.0 });
        var result = np.nonzero(a);

        result[0].size.Should().Be(1);
        result[0].GetInt64(0).Should().Be(1L);
    }

    [TestMethod]
    public void Nonzero_NegativeInfinity_IsNonzero()
    {
        var a = np.array(new double[] { 0.0, double.NegativeInfinity, 0.0 });
        var result = np.nonzero(a);

        result[0].size.Should().Be(1);
        result[0].GetInt64(0).Should().Be(1L);
    }

    [TestMethod]
    public void Nonzero_NegativeZero_IsZero()
    {
        // NumPy: -0.0 is still zero
        var a = np.array(new double[] { 1.0, -0.0, 2.0 });
        var result = np.nonzero(a);

        result[0].size.Should().Be(2);
        result[0].GetInt64(0).Should().Be(0L);
        result[0].GetInt64(1).Should().Be(2L);
    }

    #endregion

    #region Edge Cases

    [TestMethod]
    public void Nonzero_AllZeros_ReturnsEmptyArray()
    {
        var a = np.zeros(new Shape(10), NPTypeCode.Int32);
        var result = np.nonzero(a);

        result[0].size.Should().Be(0);
        result[0].typecode.Should().Be(NPTypeCode.Int64);
    }

    [TestMethod]
    public void Nonzero_AllNonzero_ReturnsAllIndices()
    {
        var a = np.ones(new Shape(5), NPTypeCode.Int32);
        var result = np.nonzero(a);

        result[0].size.Should().Be(5);
        for (int i = 0; i < 5; i++)
        {
            result[0].GetInt64(i).Should().Be((long)i);
        }
    }

    [TestMethod]
    public void Nonzero_SingleNonzero_ReturnsOneIndex()
    {
        var a = np.zeros(new Shape(10), NPTypeCode.Int32);
        a.SetInt32(1, 5);

        var result = np.nonzero(a);

        result[0].size.Should().Be(1);
        result[0].GetInt64(0).Should().Be(5L);
    }

    #endregion

    #region Using Result for Indexing

    [TestMethod]
    public void Nonzero_CanBeUsedForIndexing()
    {
        // NumPy pattern: a[np.nonzero(a)] extracts nonzero values
        var a = np.array(new int[] { 0, 3, 0, 1, 0, 2 });
        var indices = np.nonzero(a);

        var nonzeroValues = a[indices];

        nonzeroValues.size.Should().Be(3);
        nonzeroValues.GetInt32(0).Should().Be(3);
        nonzeroValues.GetInt32(1).Should().Be(1);
        nonzeroValues.GetInt32(2).Should().Be(2);
    }

    #endregion
}
