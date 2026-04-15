using System;
using System.Linq;
using System.Threading.Tasks;
using NumSharp;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Sorting;

/// <summary>
/// Tests for np.argsort int64 migration.
/// Verifies argsort returns long (int64) indices, matching NumPy behavior.
/// </summary>
public class ArgsortInt64Tests
{
    #region Return Type Tests

    [TestMethod]
    public async Task Argsort_ReturnsInt64Indices()
    {
        // NumPy: np.argsort([3, 1, 2]).dtype = int64
        var a = np.array(new int[] { 3, 1, 2 });
        var result = np.argsort<int>(a);

        await Assert.That(result.typecode).IsEqualTo(NPTypeCode.Int64);
    }

    [TestMethod]
    public async Task Argsort_Float64_ReturnsInt64Indices()
    {
        var a = np.array(new double[] { 3.0, 1.0, 2.0 });
        var result = np.argsort<double>(a);

        await Assert.That(result.typecode).IsEqualTo(NPTypeCode.Int64);
    }

    [TestMethod]
    public async Task Argsort_Float32_ReturnsInt64Indices()
    {
        var a = np.array(new float[] { 3.0f, 1.0f, 2.0f });
        var result = np.argsort<float>(a);

        await Assert.That(result.typecode).IsEqualTo(NPTypeCode.Int64);
    }

    [TestMethod]
    public async Task Argsort_Byte_ReturnsInt64Indices()
    {
        var a = np.array(new byte[] { 3, 1, 2 });
        var result = np.argsort<byte>(a);

        await Assert.That(result.typecode).IsEqualTo(NPTypeCode.Int64);
    }

    [TestMethod]
    public async Task Argsort_Int64_ReturnsInt64Indices()
    {
        var a = np.array(new long[] { 3, 1, 2 });
        var result = np.argsort<long>(a);

        await Assert.That(result.typecode).IsEqualTo(NPTypeCode.Int64);
    }

    #endregion

    #region Various DType Tests

    [TestMethod]
    public async Task Argsort_Int16_SortsCorrectly()
    {
        // NumPy: np.argsort(np.array([30, 10, 20], dtype=np.int16)) = [1, 2, 0]
        var a = np.array(new short[] { 30, 10, 20 });
        var result = np.argsort<short>(a);

        await Assert.That(result.GetInt64(0)).IsEqualTo(1L);
        await Assert.That(result.GetInt64(1)).IsEqualTo(2L);
        await Assert.That(result.GetInt64(2)).IsEqualTo(0L);
    }

    [TestMethod]
    public async Task Argsort_UInt16_SortsCorrectly()
    {
        var a = np.array(new ushort[] { 30, 10, 20 });
        var result = np.argsort<ushort>(a);

        await Assert.That(result.GetInt64(0)).IsEqualTo(1L);
        await Assert.That(result.GetInt64(1)).IsEqualTo(2L);
        await Assert.That(result.GetInt64(2)).IsEqualTo(0L);
    }

    [TestMethod]
    public async Task Argsort_UInt32_SortsCorrectly()
    {
        var a = np.array(new uint[] { 30, 10, 20 });
        var result = np.argsort<uint>(a);

        await Assert.That(result.GetInt64(0)).IsEqualTo(1L);
        await Assert.That(result.GetInt64(1)).IsEqualTo(2L);
        await Assert.That(result.GetInt64(2)).IsEqualTo(0L);
    }

    [TestMethod]
    public async Task Argsort_UInt64_SortsCorrectly()
    {
        var a = np.array(new ulong[] { 30, 10, 20 });
        var result = np.argsort<ulong>(a);

        await Assert.That(result.GetInt64(0)).IsEqualTo(1L);
        await Assert.That(result.GetInt64(1)).IsEqualTo(2L);
        await Assert.That(result.GetInt64(2)).IsEqualTo(0L);
    }

    [TestMethod]
    public async Task Argsort_Decimal_SortsCorrectly()
    {
        var a = np.array(new decimal[] { 3.0m, 1.0m, 2.0m });
        var result = np.argsort<decimal>(a);

        await Assert.That(result.GetInt64(0)).IsEqualTo(1L);
        await Assert.That(result.GetInt64(1)).IsEqualTo(2L);
        await Assert.That(result.GetInt64(2)).IsEqualTo(0L);
    }

    #endregion

    #region Axis Parameter Tests

    [TestMethod]
    public async Task Argsort_2D_Axis0_SortsColumns()
    {
        // NumPy:
        // a = np.array([[3, 1], [1, 3], [2, 2]])
        // np.argsort(a, axis=0) = [[1, 0], [2, 2], [0, 1]]
        var a = np.array(new int[,] { { 3, 1 }, { 1, 3 }, { 2, 2 } });
        var result = np.argsort<int>(a, axis: 0);

        await Assert.That(result.shape[0]).IsEqualTo(3);
        await Assert.That(result.shape[1]).IsEqualTo(2);
        await Assert.That(result.typecode).IsEqualTo(NPTypeCode.Int64);

        // First column sorted: values [3,1,2] -> indices [1,2,0]
        await Assert.That(result.GetInt64(0, 0)).IsEqualTo(1L);
        await Assert.That(result.GetInt64(1, 0)).IsEqualTo(2L);
        await Assert.That(result.GetInt64(2, 0)).IsEqualTo(0L);

        // Second column sorted: values [1,3,2] -> indices [0,2,1]
        await Assert.That(result.GetInt64(0, 1)).IsEqualTo(0L);
        await Assert.That(result.GetInt64(1, 1)).IsEqualTo(2L);
        await Assert.That(result.GetInt64(2, 1)).IsEqualTo(1L);
    }

    [TestMethod]
    public async Task Argsort_2D_Axis1_SortsRows()
    {
        // NumPy:
        // a = np.array([[3, 1, 2], [6, 4, 5]])
        // np.argsort(a, axis=1) = [[1, 2, 0], [1, 2, 0]]
        var a = np.array(new int[,] { { 3, 1, 2 }, { 6, 4, 5 } });
        var result = np.argsort<int>(a, axis: 1);

        await Assert.That(result.shape[0]).IsEqualTo(2);
        await Assert.That(result.shape[1]).IsEqualTo(3);
        await Assert.That(result.typecode).IsEqualTo(NPTypeCode.Int64);

        // First row: values [3,1,2] -> indices [1,2,0]
        await Assert.That(result.GetInt64(0, 0)).IsEqualTo(1L);
        await Assert.That(result.GetInt64(0, 1)).IsEqualTo(2L);
        await Assert.That(result.GetInt64(0, 2)).IsEqualTo(0L);

        // Second row: values [6,4,5] -> indices [1,2,0]
        await Assert.That(result.GetInt64(1, 0)).IsEqualTo(1L);
        await Assert.That(result.GetInt64(1, 1)).IsEqualTo(2L);
        await Assert.That(result.GetInt64(1, 2)).IsEqualTo(0L);
    }

    [TestMethod]
    public async Task Argsort_2D_AxisMinus1_SortsLastAxis()
    {
        // NumPy: axis=-1 is equivalent to the last axis
        var a = np.array(new int[,] { { 3, 1, 2 }, { 6, 4, 5 } });
        var result = np.argsort<int>(a, axis: -1);

        await Assert.That(result.typecode).IsEqualTo(NPTypeCode.Int64);

        // Same as axis=1 for 2D
        await Assert.That(result.GetInt64(0, 0)).IsEqualTo(1L);
        await Assert.That(result.GetInt64(0, 1)).IsEqualTo(2L);
        await Assert.That(result.GetInt64(0, 2)).IsEqualTo(0L);
    }

    #endregion

    #region Edge Cases

    [TestMethod]
    public async Task Argsort_SingleElement_ReturnsZero()
    {
        var a = np.array(new int[] { 42 });
        var result = np.argsort<int>(a);

        await Assert.That(result.size).IsEqualTo(1);
        await Assert.That(result.GetInt64(0)).IsEqualTo(0L);
    }

    [TestMethod]
    public async Task Argsort_TwoElements_SortsCorrectly()
    {
        var a = np.array(new int[] { 2, 1 });
        var result = np.argsort<int>(a);

        await Assert.That(result.GetInt64(0)).IsEqualTo(1L);
        await Assert.That(result.GetInt64(1)).IsEqualTo(0L);
    }

    [TestMethod]
    public async Task Argsort_DuplicateValues_StableSort()
    {
        // NumPy uses stable sort by default (mergesort)
        // For equal values, original order is preserved
        var a = np.array(new int[] { 1, 2, 1, 2, 1 });
        var result = np.argsort<int>(a);

        // All 1s come first, then all 2s
        // Original indices of 1s: 0, 2, 4
        // Original indices of 2s: 1, 3
        await Assert.That(result.GetInt64(0)).IsEqualTo(0L);
        await Assert.That(result.GetInt64(1)).IsEqualTo(2L);
        await Assert.That(result.GetInt64(2)).IsEqualTo(4L);
        await Assert.That(result.GetInt64(3)).IsEqualTo(1L);
        await Assert.That(result.GetInt64(4)).IsEqualTo(3L);
    }

    [TestMethod]
    public async Task Argsort_LargerArray_SortsCorrectly()
    {
        // Test with a larger array to ensure no issues with indexing
        var values = new int[] { 9, 3, 7, 1, 5, 8, 2, 6, 4, 0 };
        var a = np.array(values);
        var result = np.argsort<int>(a);

        await Assert.That(result.typecode).IsEqualTo(NPTypeCode.Int64);

        // Expected order by value: 0(idx9), 1(idx3), 2(idx6), 3(idx1), 4(idx8), 5(idx4), 6(idx7), 7(idx2), 8(idx5), 9(idx0)
        var expected = new long[] { 9, 3, 6, 1, 8, 4, 7, 2, 5, 0 };
        for (int i = 0; i < expected.Length; i++)
        {
            await Assert.That(result.GetInt64(i)).IsEqualTo(expected[i]);
        }
    }

    #endregion

    #region Using Result as Index

    [TestMethod]
    public async Task Argsort_CanBeUsedForIndexing()
    {
        // NumPy pattern: a[np.argsort(a)] gives sorted array
        var a = np.array(new int[] { 3, 1, 4, 1, 5, 9, 2, 6 });
        var indices = np.argsort<int>(a);

        // Use the indices to get sorted values
        var sorted = a[indices];

        // Verify sorted order
        var sortedData = sorted.Data<int>();
        for (int i = 1; i < sortedData.Count; i++)
        {
            await Assert.That(sortedData[i]).IsGreaterThanOrEqualTo(sortedData[i - 1]);
        }
    }

    #endregion
}
