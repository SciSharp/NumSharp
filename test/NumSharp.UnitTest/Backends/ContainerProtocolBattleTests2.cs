using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using NumSharp;

namespace NumSharp.UnitTest.Backends;

/// <summary>
/// Second round of battle tests for container protocol.
/// Focuses on edge cases, long indexing, and complex scenarios.
/// </summary>
[TestClass]
public class ContainerProtocolBattleTests2
{
    #region __len__ Round 2

    [TestMethod]
    public void Len_LargeArray_ReturnsCorrectLength()
    {
        // Test with large first dimension
        var arr = np.zeros(new long[] { 10000, 10 });
        arr.__len__().Should().Be(10000);
    }

    [TestMethod]
    public void Len_FirstDimensionOne_ReturnsOne()
    {
        var arr = np.zeros(new long[] { 1, 100, 100 });
        arr.__len__().Should().Be(1);
    }

    [TestMethod]
    public void Len_HighDimensional_5D()
    {
        var arr = np.zeros(new long[] { 2, 3, 4, 5, 6 });
        arr.__len__().Should().Be(2);
    }

    [TestMethod]
    public void Len_AfterTranspose_ChangesFirstDimension()
    {
        var arr = np.arange(24).reshape(2, 3, 4);
        arr.__len__().Should().Be(2);

        // After transpose, dimensions are reversed
        var transposed = arr.T;
        transposed.__len__().Should().Be(4);
    }

    [TestMethod]
    public void Len_NegativeStridedSlice()
    {
        var arr = np.arange(10);
        var reversed = arr["::-2"]; // [9, 7, 5, 3, 1]
        reversed.__len__().Should().Be(5);
    }

    [TestMethod]
    public void Len_SliceOfSlice()
    {
        var arr = np.arange(100);
        var slice1 = arr["10:90"];
        var slice2 = slice1["20:60"];
        slice2.__len__().Should().Be(40);
    }

    #endregion

    #region __getitem__ Round 2 - Long Indexing

    [TestMethod]
    public void GetItem_LongIndex_PositiveWorks()
    {
        var arr = np.arange(10);
        var result = arr.__getitem__(5L);
        ((int)result).Should().Be(5);
    }

    [TestMethod]
    public void GetItem_LongIndex_NegativeWorks()
    {
        var arr = np.arange(10);
        var result = arr.__getitem__(-1L);
        ((int)result).Should().Be(9);
    }

    [TestMethod]
    public void GetItem_LongIndex_2DArray()
    {
        var arr = np.arange(12).reshape(3, 4);
        var row = arr.__getitem__(2L);
        row.size.Should().Be(4);
        ((int)row[0]).Should().Be(8);
    }

    [TestMethod]
    public void GetItem_Ellipsis_Basic()
    {
        var arr = np.arange(24).reshape(2, 3, 4);
        // "..., 0" = all dimensions except last, then index 0 on last
        var result = arr["..., 0"];
        result.shape.Should().BeEquivalentTo(new long[] { 2, 3 });
    }

    [TestMethod]
    public void GetItem_ComplexSlice_StartStopStep()
    {
        var arr = np.arange(20);
        var result = arr["2:18:3"]; // [2, 5, 8, 11, 14, 17]
        result.size.Should().Be(6);
        ((int)result[0]).Should().Be(2);
        ((int)result[5]).Should().Be(17);
    }

    [TestMethod]
    public void GetItem_2D_ColumnSlice()
    {
        var arr = np.arange(12).reshape(3, 4);
        var column = arr[":, 1"]; // All rows, column 1
        column.size.Should().Be(3);
        ((int)column[0]).Should().Be(1);
        ((int)column[1]).Should().Be(5);
        ((int)column[2]).Should().Be(9);
    }

    [TestMethod]
    public void GetItem_2D_SubMatrix()
    {
        var arr = np.arange(20).reshape(4, 5);
        var submat = arr["1:3, 2:5"];
        submat.shape.Should().BeEquivalentTo(new long[] { 2, 3 });
        ((int)submat[0, 0]).Should().Be(7);
    }

    [TestMethod]
    public void GetItem_ViewChaining_PreservesData()
    {
        var original = np.arange(100).reshape(10, 10);
        var view1 = original["2:8, :"];
        var view2 = view1[":, 3:7"];
        var view3 = view2["1:4, :"];

        // Modify original
        original[3, 5] = 999;

        // Check all views see the change
        ((int)view1[1, 5]).Should().Be(999);
        ((int)view2[1, 2]).Should().Be(999);
        ((int)view3[0, 2]).Should().Be(999);
    }

    #endregion

    #region __setitem__ Round 2

    [TestMethod]
    public void SetItem_LongIndex_Works()
    {
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });
        arr.__setitem__(3L, 99);
        ((int)arr[3]).Should().Be(99);
    }

    [TestMethod]
    public void SetItem_NegativeLongIndex_Works()
    {
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });
        arr.__setitem__(-2L, 88);
        ((int)arr[3]).Should().Be(88);
    }

    [TestMethod]
    public void SetItem_BroadcastScalarTo2D()
    {
        var arr = np.zeros(new long[] { 3, 4 }, np.int32);
        arr.__setitem__(":", 5);
        np.all(arr == 5).Should().BeTrue();
    }

    [TestMethod]
    [OpenBugs] // BUG: NumPy broadcasts row to all rows, NumSharp doesn't
    public void SetItem_BroadcastRowTo2D()
    {
        // NumPy: arr[:] = row broadcasts row to all rows
        // NumSharp: doesn't support this broadcast pattern yet
        var arr = np.zeros(new long[] { 3, 4 }, np.int32);
        var row = np.array(new[] { 1, 2, 3, 4 });
        arr.__setitem__(":", row);

        // When fixed, all rows should be [1, 2, 3, 4]
        for (int i = 0; i < 3; i++)
        {
            ((int)arr[i, 0]).Should().Be(1);
            ((int)arr[i, 3]).Should().Be(4);
        }
    }

    [TestMethod]
    public void SetItem_SliceToSlice()
    {
        var arr = np.zeros(new long[] { 10 }, np.int32);
        var source = np.arange(3) + 1; // [1, 2, 3]

        arr.__setitem__("2:5", source);

        ((int)arr[0]).Should().Be(0);
        ((int)arr[1]).Should().Be(0);
        ((int)arr[2]).Should().Be(1);
        ((int)arr[3]).Should().Be(2);
        ((int)arr[4]).Should().Be(3);
        ((int)arr[5]).Should().Be(0);
    }

    [TestMethod]
    public void SetItem_StridedSlice()
    {
        var arr = np.zeros(new long[] { 10 }, np.int32);
        arr.__setitem__("::2", 1); // Set every other element

        ((int)arr[0]).Should().Be(1);
        ((int)arr[1]).Should().Be(0);
        ((int)arr[2]).Should().Be(1);
        ((int)arr[3]).Should().Be(0);
    }

    [TestMethod]
    public void SetItem_TypePromotion_IntToDouble()
    {
        var arr = np.array(new[] { 1.5, 2.5, 3.5 });
        arr.__setitem__(1, 10); // int assigned to double array
        ((double)arr[1]).Should().Be(10.0);
    }

    [TestMethod]
    [Misaligned] // NumPy truncates (2.9 -> 2), NumSharp rounds (2.9 -> 3)
    public void SetItem_TypePromotion_DoubleToInt_Rounds()
    {
        // NumPy truncates: arr[1] = 2.9 becomes 2
        // NumSharp rounds: arr[1] = 2.9 becomes 3
        var arr = np.array(new[] { 1, 2, 3 });
        arr.__setitem__(1, 2.9);
        ((int)arr[1]).Should().Be(3); // NumSharp rounds
    }

    [TestMethod]
    public void SetItem_ViewModifiesOriginal()
    {
        var original = np.arange(10);
        var view = original["3:7"];

        view.__setitem__(1, 999);

        ((int)original[4]).Should().Be(999);
    }

    #endregion

    #region __contains__ Round 2

    [TestMethod]
    public void Contains_NaN_InFloatArray_ReturnsFalse()
    {
        // NaN != NaN in IEEE 754
        var arr = np.array(new[] { 1.0f, float.NaN, 3.0f });
        arr.Contains(float.NaN).Should().BeFalse();
    }

    [TestMethod]
    public void Contains_NaN_DoubleArray_ReturnsFalse()
    {
        var arr = np.array(new[] { 1.0, double.NaN, 3.0 });
        arr.Contains(double.NaN).Should().BeFalse();
    }

    [TestMethod]
    public void Contains_MaxValue_Int32()
    {
        var arr = np.array(new[] { int.MinValue, 0, int.MaxValue });
        arr.Contains(int.MaxValue).Should().BeTrue();
        arr.Contains(int.MinValue).Should().BeTrue();
    }

    [TestMethod]
    public void Contains_MaxValue_Int64()
    {
        var arr = np.array(new[] { long.MinValue, 0L, long.MaxValue });
        arr.Contains(long.MaxValue).Should().BeTrue();
        arr.Contains(long.MinValue).Should().BeTrue();
    }

    [TestMethod]
    public void Contains_TypePromotion_ByteInInt32Array()
    {
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });
        arr.Contains(((byte)3)).Should().BeTrue();
    }

    [TestMethod]
    public void Contains_TypePromotion_Int32InInt64Array()
    {
        var arr = np.array(new long[] { 1, 2, 3, 4, 5 });
        arr.Contains(3).Should().BeTrue();
    }

    [TestMethod]
    public void Contains_ZeroInMixedSignArray()
    {
        var arr = np.array(new[] { -5, -3, 0, 3, 5 });
        arr.Contains(0).Should().BeTrue();
    }

    [TestMethod]
    public void Contains_SpecialFloats()
    {
        var arr = np.array(new[] {
            double.NegativeInfinity,
            double.MinValue,
            0.0,
            double.MaxValue,
            double.PositiveInfinity
        });

        arr.Contains(double.NegativeInfinity).Should().BeTrue();
        arr.Contains(double.PositiveInfinity).Should().BeTrue();
        arr.Contains(double.MinValue).Should().BeTrue();
        arr.Contains(double.MaxValue).Should().BeTrue();
    }

    [TestMethod]
    public void Contains_TransposedArray()
    {
        var arr = np.arange(12).reshape(3, 4).T;
        arr.Contains(5).Should().BeTrue();
        arr.Contains(100).Should().BeFalse();
    }

    [TestMethod]
    public void Contains_ReversedArray()
    {
        var arr = np.arange(10)["::-1"];
        arr.Contains(5).Should().BeTrue();
        arr.Contains(100).Should().BeFalse();
    }

    #endregion

    #region __iter__ Round 2

    [TestMethod]
    public void Iter_ViewIteration_CorrectValues()
    {
        var arr = np.arange(20).reshape(4, 5);
        var view = arr["1:3, :"];

        var rows = new List<NDArray>();
        foreach (var item in view)
        {
            rows.Add((NDArray)item);
        }

        rows.Count.Should().Be(2);
        ((int)rows[0][0]).Should().Be(5);
        ((int)rows[1][0]).Should().Be(10);
    }

    [TestMethod]
    public void Iter_NestedIteration_3DArray()
    {
        var arr = np.arange(24).reshape(2, 3, 4);

        var allValues = new List<int>();
        foreach (var plane in arr) // 2D planes
        {
            foreach (var row in (NDArray)plane) // 1D rows
            {
                foreach (var val in (NDArray)row) // scalars
                {
                    allValues.Add(Convert.ToInt32(val));
                }
            }
        }

        allValues.Count.Should().Be(24);
        allValues[0].Should().Be(0);
        allValues[23].Should().Be(23);
    }

    [TestMethod]
    public void Iter_MultipleEnumerators_Independent()
    {
        var arr = np.array(new[] { 1, 2, 3 });

        var enum1 = arr.GetEnumerator();
        var enum2 = arr.GetEnumerator();

        enum1.MoveNext();
        enum1.MoveNext();
        enum2.MoveNext();

        Convert.ToInt32(enum1.Current).Should().Be(2);
        Convert.ToInt32(enum2.Current).Should().Be(1);
    }

    [TestMethod]
    public void Iter_ResetEnumerator_ThrowsNotSupported()
    {
        // NDIterator does not support Reset - document this behavior
        var arr = np.array(new[] { 1, 2, 3 });
        var enumerator = arr.GetEnumerator();

        enumerator.MoveNext();
        enumerator.MoveNext();
        Convert.ToInt32(enumerator.Current).Should().Be(2);

        // Reset throws NotSupportedException
        Microsoft.VisualStudio.TestTools.UnitTesting.Assert.ThrowsException<NotSupportedException>(() => enumerator.Reset());
    }

    [TestMethod]
    public void Iter_TransposedArray_CorrectOrder()
    {
        var arr = np.arange(6).reshape(2, 3).T; // Shape (3, 2)

        var rows = new List<NDArray>();
        foreach (var item in arr)
        {
            rows.Add((NDArray)item);
        }

        rows.Count.Should().Be(3);
        // First row of transposed is [0, 3]
        ((int)rows[0][0]).Should().Be(0);
        ((int)rows[0][1]).Should().Be(3);
    }

    [TestMethod]
    public void Iter_SlicedArray_CorrectElements()
    {
        var arr = np.arange(20);
        var sliced = arr["5:10"]; // [5, 6, 7, 8, 9]

        var values = new List<int>();
        foreach (var item in sliced)
        {
            values.Add(Convert.ToInt32(item));
        }

        values.Should().BeEquivalentTo(new[] { 5, 6, 7, 8, 9 });
    }

    [TestMethod]
    public void Iter_StridedArray_CorrectElements()
    {
        var arr = np.arange(10);
        var strided = arr["1::2"]; // [1, 3, 5, 7, 9]

        var values = new List<int>();
        foreach (var item in strided)
        {
            values.Add(Convert.ToInt32(item));
        }

        values.Should().BeEquivalentTo(new[] { 1, 3, 5, 7, 9 });
    }

    [TestMethod]
    public void Iter_ReversedArray_CorrectOrder()
    {
        var arr = np.arange(5)["::-1"]; // [4, 3, 2, 1, 0]

        var values = new List<int>();
        foreach (var item in arr)
        {
            values.Add(Convert.ToInt32(item));
        }

        values.Should().BeEquivalentTo(new[] { 4, 3, 2, 1, 0 });
    }

    [TestMethod]
    public void Iter_BroadcastArray_IteratesOverBroadcastDimension()
    {
        var arr = np.array(new[] { 1, 2, 3 });
        var broadcast = np.broadcast_to(arr, new Shape(4, 3));

        var rows = new List<NDArray>();
        foreach (var item in broadcast)
        {
            rows.Add((NDArray)item);
        }

        rows.Count.Should().Be(4);
        // All rows should be [1, 2, 3]
        foreach (var row in rows)
        {
            ((int)row[0]).Should().Be(1);
            ((int)row[2]).Should().Be(3);
        }
    }

    #endregion

    #region Edge Cases and Error Conditions

    [TestMethod]
    public void GetItem_EmptySlice_ReturnsEmpty()
    {
        var arr = np.arange(10);
        var empty = arr["5:5"]; // Empty slice
        empty.size.Should().Be(0);
    }

    [TestMethod]
    public void GetItem_OutOfBoundsStart_ClampsToZero()
    {
        var arr = np.arange(5);
        var result = arr["-100:3"]; // Should be equivalent to [:3]
        result.size.Should().Be(3);
    }

    [TestMethod]
    public void GetItem_OutOfBoundsStop_ClampsToEnd()
    {
        var arr = np.arange(5);
        var result = arr["2:100"]; // Should be equivalent to [2:]
        result.size.Should().Be(3);
    }

    [TestMethod]
    public void Len_EmptyHighDimensional()
    {
        var arr = np.zeros(new long[] { 0, 5, 10 });
        arr.__len__().Should().Be(0);
    }

    [TestMethod]
    public void Contains_WrongType_ReturnsFalse()
    {
        var arr = np.array(new[] { 1, 2, 3 });
        // String can't be in int array
        arr.Contains("hello").Should().BeFalse();
    }

    [TestMethod]
    public void Iter_EmptyDimension_NoIterations()
    {
        var arr = np.zeros(new long[] { 0, 5 });
        var count = 0;
        foreach (var _ in arr)
        {
            count++;
        }
        count.Should().Be(0);
    }

    [TestMethod]
    public void Hash_DifferentArrays_AllThrow()
    {
        var arrays = new[]
        {
            np.zeros(5),
            np.ones(5),
            np.arange(10).reshape(2, 5),
            np.array(new[] { 1.5, 2.5 }),
        };

        foreach (var arr in arrays)
        {
            Microsoft.VisualStudio.TestTools.UnitTesting.Assert.ThrowsException<NotSupportedException>(() => arr.GetHashCode());
        }
    }

    #endregion

    #region Python API Naming

    [TestMethod]
    public void PythonAPI_AllMethodsAccessible()
    {
        var arr = np.arange(12).reshape(3, 4);

        // All these should be callable without exception
        var len = arr.__len__();
        var contains = arr.__contains__(5);
        var item = arr.__getitem__(1);
        var iter = arr.__iter__();

        len.Should().Be(3);
        contains.Should().BeTrue();
        item.size.Should().Be(4);
        iter.Should().NotBeNull();
    }

    [TestMethod]
    public void PythonAPI_SetItem_Works()
    {
        var arr = np.zeros(new long[] { 3, 3 }, np.int32);
        arr.__setitem__(1, 5);

        ((int)arr[1, 0]).Should().Be(5);
        ((int)arr[1, 1]).Should().Be(5);
        ((int)arr[1, 2]).Should().Be(5);
    }

    [TestMethod]
    public void PythonAPI_Hash_Throws()
    {
        var arr = np.array(new[] { 1, 2, 3 });
        Microsoft.VisualStudio.TestTools.UnitTesting.Assert.ThrowsException<NotSupportedException>(() => arr.__hash__());
    }

    #endregion
}
