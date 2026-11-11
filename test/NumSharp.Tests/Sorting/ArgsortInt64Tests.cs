namespace NumSharp.Tests.Sorting;

/// <summary>
/// Tests for np.argsort int64 migration.
/// Verifies argsort returns long (int64) indices, matching NumPy behavior.
/// </summary>
[TestClass]
public class ArgsortInt64Tests
{
    #region Return Type Tests

    /// <summary>np.argsort of an int32 array returns int64 indices, as NumPy's (intp on a 64-bit build).</summary>
    [TestMethod]
    public void Argsort_ReturnsInt64Indices()
    {
        // NumPy: np.argsort([3, 1, 2]).dtype = int64
        var a = np.array(new int[] { 3, 1, 2 });
        var result = np.argsort(a);

        result.typecode.Should().Be(NPTypeCode.Int64);
    }

    /// <summary>np.argsort of a float64 array returns int64 indices.</summary>
    [TestMethod]
    public void Argsort_Float64_ReturnsInt64Indices()
    {
        var a = np.array(new double[] { 3.0, 1.0, 2.0 });
        var result = np.argsort(a);

        result.typecode.Should().Be(NPTypeCode.Int64);
    }

    /// <summary>np.argsort of a float32 array returns int64 indices.</summary>
    [TestMethod]
    public void Argsort_Float32_ReturnsInt64Indices()
    {
        var a = np.array(new float[] { 3.0f, 1.0f, 2.0f });
        var result = np.argsort(a);

        result.typecode.Should().Be(NPTypeCode.Int64);
    }

    /// <summary>np.argsort of a uint8 array returns int64 indices.</summary>
    [TestMethod]
    public void Argsort_Byte_ReturnsInt64Indices()
    {
        var a = np.array(new byte[] { 3, 1, 2 });
        var result = np.argsort(a);

        result.typecode.Should().Be(NPTypeCode.Int64);
    }

    /// <summary>np.argsort of an int64 array returns int64 indices.</summary>
    [TestMethod]
    public void Argsort_Int64_ReturnsInt64Indices()
    {
        var a = np.array(new long[] { 3, 1, 2 });
        var result = np.argsort(a);

        result.typecode.Should().Be(NPTypeCode.Int64);
    }

    #endregion

    #region Various DType Tests

    /// <summary>np.argsort of int16 [30, 10, 20] is NumPy's [1, 2, 0].</summary>
    [TestMethod]
    public void Argsort_Int16_SortsCorrectly()
    {
        // NumPy: np.argsort(np.array([30, 10, 20], dtype=np.int16)) = [1, 2, 0]
        var a = np.array(new short[] { 30, 10, 20 });
        var result = np.argsort(a);

        result.GetInt64(0).Should().Be(1L);
        result.GetInt64(1).Should().Be(2L);
        result.GetInt64(2).Should().Be(0L);
    }

    /// <summary>np.argsort of uint16 [30, 10, 20] is NumPy's [1, 2, 0].</summary>
    [TestMethod]
    public void Argsort_UInt16_SortsCorrectly()
    {
        var a = np.array(new ushort[] { 30, 10, 20 });
        var result = np.argsort(a);

        result.GetInt64(0).Should().Be(1L);
        result.GetInt64(1).Should().Be(2L);
        result.GetInt64(2).Should().Be(0L);
    }

    /// <summary>np.argsort of uint32 [30, 10, 20] is NumPy's [1, 2, 0].</summary>
    [TestMethod]
    public void Argsort_UInt32_SortsCorrectly()
    {
        var a = np.array(new uint[] { 30, 10, 20 });
        var result = np.argsort(a);

        result.GetInt64(0).Should().Be(1L);
        result.GetInt64(1).Should().Be(2L);
        result.GetInt64(2).Should().Be(0L);
    }

    /// <summary>np.argsort of uint64 [30, 10, 20] is NumPy's [1, 2, 0].</summary>
    [TestMethod]
    public void Argsort_UInt64_SortsCorrectly()
    {
        var a = np.array(new ulong[] { 30, 10, 20 });
        var result = np.argsort(a);

        result.GetInt64(0).Should().Be(1L);
        result.GetInt64(1).Should().Be(2L);
        result.GetInt64(2).Should().Be(0L);
    }

    /// <summary>np.argsort of decimal [3, 1, 2] is [1, 2, 0] (NumSharp's decimal dtype sorts as numbers).</summary>
    [TestMethod]
    public void Argsort_Decimal_SortsCorrectly()
    {
        var a = np.array(new decimal[] { 3.0m, 1.0m, 2.0m });
        var result = np.argsort(a);

        result.GetInt64(0).Should().Be(1L);
        result.GetInt64(1).Should().Be(2L);
        result.GetInt64(2).Should().Be(0L);
    }

    #endregion

    #region Axis Parameter Tests

    /// <summary>np.argsort(axis=0) of [[3, 1], [1, 3], [2, 2]] sorts each column: NumPy's [[1, 0], [2, 2], [0, 1]].</summary>
    [TestMethod]
    public void Argsort_2D_Axis0_SortsColumns()
    {
        // NumPy:
        // a = np.array([[3, 1], [1, 3], [2, 2]])
        // np.argsort(a, axis=0) = [[1, 0], [2, 2], [0, 1]]
        var a = np.array(new int[,] { { 3, 1 }, { 1, 3 }, { 2, 2 } });
        var result = np.argsort(a, axis: 0);

        result.shape[0].Should().Be(3);
        result.shape[1].Should().Be(2);
        result.typecode.Should().Be(NPTypeCode.Int64);

        // First column sorted: values [3,1,2] -> indices [1,2,0]
        result.GetInt64(0, 0).Should().Be(1L);
        result.GetInt64(1, 0).Should().Be(2L);
        result.GetInt64(2, 0).Should().Be(0L);

        // Second column sorted: values [1,3,2] -> indices [0,2,1]
        result.GetInt64(0, 1).Should().Be(0L);
        result.GetInt64(1, 1).Should().Be(2L);
        result.GetInt64(2, 1).Should().Be(1L);
    }

    /// <summary>np.argsort(axis=1) of [[3, 1, 2], [6, 4, 5]] sorts each row: NumPy's [[1, 2, 0], [1, 2, 0]].</summary>
    [TestMethod]
    public void Argsort_2D_Axis1_SortsRows()
    {
        // NumPy:
        // a = np.array([[3, 1, 2], [6, 4, 5]])
        // np.argsort(a, axis=1) = [[1, 2, 0], [1, 2, 0]]
        var a = np.array(new int[,] { { 3, 1, 2 }, { 6, 4, 5 } });
        var result = np.argsort(a, axis: 1);

        result.shape[0].Should().Be(2);
        result.shape[1].Should().Be(3);
        result.typecode.Should().Be(NPTypeCode.Int64);

        // First row: values [3,1,2] -> indices [1,2,0]
        result.GetInt64(0, 0).Should().Be(1L);
        result.GetInt64(0, 1).Should().Be(2L);
        result.GetInt64(0, 2).Should().Be(0L);

        // Second row: values [6,4,5] -> indices [1,2,0]
        result.GetInt64(1, 0).Should().Be(1L);
        result.GetInt64(1, 1).Should().Be(2L);
        result.GetInt64(1, 2).Should().Be(0L);
    }

    /// <summary>np.argsort(axis=-1) is the last axis, the same answer as axis=1 for a 2-D array.</summary>
    [TestMethod]
    public void Argsort_2D_AxisMinus1_SortsLastAxis()
    {
        // NumPy: axis=-1 is equivalent to the last axis
        var a = np.array(new int[,] { { 3, 1, 2 }, { 6, 4, 5 } });
        var result = np.argsort(a, axis: -1);

        result.typecode.Should().Be(NPTypeCode.Int64);

        // Same as axis=1 for 2D
        result.GetInt64(0, 0).Should().Be(1L);
        result.GetInt64(0, 1).Should().Be(2L);
        result.GetInt64(0, 2).Should().Be(0L);
    }

    #endregion

    #region Edge Cases

    /// <summary>np.argsort of a one-element array is [0].</summary>
    [TestMethod]
    public void Argsort_SingleElement_ReturnsZero()
    {
        var a = np.array(new int[] { 42 });
        var result = np.argsort(a);

        result.size.Should().Be(1);
        result.GetInt64(0).Should().Be(0L);
    }

    /// <summary>np.argsort of [2, 1] is [1, 0].</summary>
    [TestMethod]
    public void Argsort_TwoElements_SortsCorrectly()
    {
        var a = np.array(new int[] { 2, 1 });
        var result = np.argsort(a);

        result.GetInt64(0).Should().Be(1L);
        result.GetInt64(1).Should().Be(0L);
    }

    /// <summary>Equal values keep their input order (the stable permutation NumPy's default gives for these numbers).</summary>
    [TestMethod]
    public void Argsort_DuplicateValues_StableSort()
    {
        // NumPy uses stable sort by default (mergesort)
        // For equal values, original order is preserved
        var a = np.array(new int[] { 1, 2, 1, 2, 1 });
        var result = np.argsort(a);

        // All 1s come first, then all 2s
        // Original indices of 1s: 0, 2, 4
        // Original indices of 2s: 1, 3
        result.GetInt64(0).Should().Be(0L);
        result.GetInt64(1).Should().Be(2L);
        result.GetInt64(2).Should().Be(4L);
        result.GetInt64(3).Should().Be(1L);
        result.GetInt64(4).Should().Be(3L);
    }

    /// <summary>A ten-element permutation of 0..9 sorts to the indices NumPy gives, as int64.</summary>
    [TestMethod]
    public void Argsort_LargerArray_SortsCorrectly()
    {
        // Test with a larger array to ensure no issues with indexing
        var values = new int[] { 9, 3, 7, 1, 5, 8, 2, 6, 4, 0 };
        var a = np.array(values);
        var result = np.argsort(a);

        result.typecode.Should().Be(NPTypeCode.Int64);

        // Expected order by value: 0(idx9), 1(idx3), 2(idx6), 3(idx1), 4(idx8), 5(idx4), 6(idx7), 7(idx2), 8(idx5), 9(idx0)
        var expected = new long[] { 9, 3, 6, 1, 8, 4, 7, 2, 5, 0 };
        for (int i = 0; i < expected.Length; i++)
        {
            result.GetInt64(i).Should().Be(expected[i]);
        }
    }

    #endregion

    #region Using Result as Index

    /// <summary>a[np.argsort(a)] is the sorted array, NumPy's idiom.</summary>
    [TestMethod]
    public void Argsort_CanBeUsedForIndexing()
    {
        // NumPy pattern: a[np.argsort(a)] gives sorted array
        var a = np.array(new int[] { 3, 1, 4, 1, 5, 9, 2, 6 });
        var indices = np.argsort(a);

        // Use the indices to get sorted values
        var sorted = a[indices];

        // Verify sorted order
        var sortedData = sorted.Data<int>();
        for (int i = 1; i < sortedData.Count; i++)
        {
            sortedData[i].Should().BeGreaterThanOrEqualTo(sortedData[i - 1]);
        }
    }

    #endregion
}
