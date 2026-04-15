using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using NumSharp;

namespace NumSharp.UnitTest.Backends;

/// <summary>
/// Second round of battle tests for container protocol.
/// Focuses on edge cases, long indexing, and complex scenarios.
/// </summary>
public class ContainerProtocolBattleTests2
{
    #region __len__ Round 2

    [TestMethod]
    public async Task Len_LargeArray_ReturnsCorrectLength()
    {
        // Test with large first dimension
        var arr = np.zeros(new long[] { 10000, 10 });
        await Assert.That(arr.__len__()).IsEqualTo(10000);
    }

    [TestMethod]
    public async Task Len_FirstDimensionOne_ReturnsOne()
    {
        var arr = np.zeros(new long[] { 1, 100, 100 });
        await Assert.That(arr.__len__()).IsEqualTo(1);
    }

    [TestMethod]
    public async Task Len_HighDimensional_5D()
    {
        var arr = np.zeros(new long[] { 2, 3, 4, 5, 6 });
        await Assert.That(arr.__len__()).IsEqualTo(2);
    }

    [TestMethod]
    public async Task Len_AfterTranspose_ChangesFirstDimension()
    {
        var arr = np.arange(24).reshape(2, 3, 4);
        await Assert.That(arr.__len__()).IsEqualTo(2);

        // After transpose, dimensions are reversed
        var transposed = arr.T;
        await Assert.That(transposed.__len__()).IsEqualTo(4);
    }

    [TestMethod]
    public async Task Len_NegativeStridedSlice()
    {
        var arr = np.arange(10);
        var reversed = arr["::-2"]; // [9, 7, 5, 3, 1]
        await Assert.That(reversed.__len__()).IsEqualTo(5);
    }

    [TestMethod]
    public async Task Len_SliceOfSlice()
    {
        var arr = np.arange(100);
        var slice1 = arr["10:90"];
        var slice2 = slice1["20:60"];
        await Assert.That(slice2.__len__()).IsEqualTo(40);
    }

    #endregion

    #region __getitem__ Round 2 - Long Indexing

    [TestMethod]
    public async Task GetItem_LongIndex_PositiveWorks()
    {
        var arr = np.arange(10);
        var result = arr.__getitem__(5L);
        await Assert.That((int)result).IsEqualTo(5);
    }

    [TestMethod]
    public async Task GetItem_LongIndex_NegativeWorks()
    {
        var arr = np.arange(10);
        var result = arr.__getitem__(-1L);
        await Assert.That((int)result).IsEqualTo(9);
    }

    [TestMethod]
    public async Task GetItem_LongIndex_2DArray()
    {
        var arr = np.arange(12).reshape(3, 4);
        var row = arr.__getitem__(2L);
        await Assert.That(row.size).IsEqualTo(4);
        await Assert.That((int)row[0]).IsEqualTo(8);
    }

    [TestMethod]
    public async Task GetItem_Ellipsis_Basic()
    {
        var arr = np.arange(24).reshape(2, 3, 4);
        // "..., 0" = all dimensions except last, then index 0 on last
        var result = arr["..., 0"];
        await Assert.That(result.shape).IsEquivalentTo(new long[] { 2, 3 });
    }

    [TestMethod]
    public async Task GetItem_ComplexSlice_StartStopStep()
    {
        var arr = np.arange(20);
        var result = arr["2:18:3"]; // [2, 5, 8, 11, 14, 17]
        await Assert.That(result.size).IsEqualTo(6);
        await Assert.That((int)result[0]).IsEqualTo(2);
        await Assert.That((int)result[5]).IsEqualTo(17);
    }

    [TestMethod]
    public async Task GetItem_2D_ColumnSlice()
    {
        var arr = np.arange(12).reshape(3, 4);
        var column = arr[":, 1"]; // All rows, column 1
        await Assert.That(column.size).IsEqualTo(3);
        await Assert.That((int)column[0]).IsEqualTo(1);
        await Assert.That((int)column[1]).IsEqualTo(5);
        await Assert.That((int)column[2]).IsEqualTo(9);
    }

    [TestMethod]
    public async Task GetItem_2D_SubMatrix()
    {
        var arr = np.arange(20).reshape(4, 5);
        var submat = arr["1:3, 2:5"];
        await Assert.That(submat.shape).IsEquivalentTo(new long[] { 2, 3 });
        await Assert.That((int)submat[0, 0]).IsEqualTo(7);
    }

    [TestMethod]
    public async Task GetItem_ViewChaining_PreservesData()
    {
        var original = np.arange(100).reshape(10, 10);
        var view1 = original["2:8, :"];
        var view2 = view1[":, 3:7"];
        var view3 = view2["1:4, :"];

        // Modify original
        original[3, 5] = 999;

        // Check all views see the change
        await Assert.That((int)view1[1, 5]).IsEqualTo(999);
        await Assert.That((int)view2[1, 2]).IsEqualTo(999);
        await Assert.That((int)view3[0, 2]).IsEqualTo(999);
    }

    #endregion

    #region __setitem__ Round 2

    [TestMethod]
    public async Task SetItem_LongIndex_Works()
    {
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });
        arr.__setitem__(3L, 99);
        await Assert.That((int)arr[3]).IsEqualTo(99);
    }

    [TestMethod]
    public async Task SetItem_NegativeLongIndex_Works()
    {
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });
        arr.__setitem__(-2L, 88);
        await Assert.That((int)arr[3]).IsEqualTo(88);
    }

    [TestMethod]
    public async Task SetItem_BroadcastScalarTo2D()
    {
        var arr = np.zeros(new long[] { 3, 4 }, np.int32);
        arr.__setitem__(":", 5);
        await Assert.That(np.all(arr == 5)).IsTrue();
    }

    [TestMethod]
    [OpenBugs] // BUG: NumPy broadcasts row to all rows, NumSharp doesn't
    public async Task SetItem_BroadcastRowTo2D()
    {
        // NumPy: arr[:] = row broadcasts row to all rows
        // NumSharp: doesn't support this broadcast pattern yet
        var arr = np.zeros(new long[] { 3, 4 }, np.int32);
        var row = np.array(new[] { 1, 2, 3, 4 });
        arr.__setitem__(":", row);

        // When fixed, all rows should be [1, 2, 3, 4]
        for (int i = 0; i < 3; i++)
        {
            await Assert.That((int)arr[i, 0]).IsEqualTo(1);
            await Assert.That((int)arr[i, 3]).IsEqualTo(4);
        }
    }

    [TestMethod]
    public async Task SetItem_SliceToSlice()
    {
        var arr = np.zeros(new long[] { 10 }, np.int32);
        var source = np.arange(3) + 1; // [1, 2, 3]

        arr.__setitem__("2:5", source);

        await Assert.That((int)arr[0]).IsEqualTo(0);
        await Assert.That((int)arr[1]).IsEqualTo(0);
        await Assert.That((int)arr[2]).IsEqualTo(1);
        await Assert.That((int)arr[3]).IsEqualTo(2);
        await Assert.That((int)arr[4]).IsEqualTo(3);
        await Assert.That((int)arr[5]).IsEqualTo(0);
    }

    [TestMethod]
    public async Task SetItem_StridedSlice()
    {
        var arr = np.zeros(new long[] { 10 }, np.int32);
        arr.__setitem__("::2", 1); // Set every other element

        await Assert.That((int)arr[0]).IsEqualTo(1);
        await Assert.That((int)arr[1]).IsEqualTo(0);
        await Assert.That((int)arr[2]).IsEqualTo(1);
        await Assert.That((int)arr[3]).IsEqualTo(0);
    }

    [TestMethod]
    public async Task SetItem_TypePromotion_IntToDouble()
    {
        var arr = np.array(new[] { 1.5, 2.5, 3.5 });
        arr.__setitem__(1, 10); // int assigned to double array
        await Assert.That((double)arr[1]).IsEqualTo(10.0);
    }

    [TestMethod]
    [Misaligned] // NumPy truncates (2.9 -> 2), NumSharp rounds (2.9 -> 3)
    public async Task SetItem_TypePromotion_DoubleToInt_Rounds()
    {
        // NumPy truncates: arr[1] = 2.9 becomes 2
        // NumSharp rounds: arr[1] = 2.9 becomes 3
        var arr = np.array(new[] { 1, 2, 3 });
        arr.__setitem__(1, 2.9);
        await Assert.That((int)arr[1]).IsEqualTo(3); // NumSharp rounds
    }

    [TestMethod]
    public async Task SetItem_ViewModifiesOriginal()
    {
        var original = np.arange(10);
        var view = original["3:7"];

        view.__setitem__(1, 999);

        await Assert.That((int)original[4]).IsEqualTo(999);
    }

    #endregion

    #region __contains__ Round 2

    [TestMethod]
    public async Task Contains_NaN_InFloatArray_ReturnsFalse()
    {
        // NaN != NaN in IEEE 754
        var arr = np.array(new[] { 1.0f, float.NaN, 3.0f });
        await Assert.That(arr.Contains(float.NaN)).IsFalse();
    }

    [TestMethod]
    public async Task Contains_NaN_DoubleArray_ReturnsFalse()
    {
        var arr = np.array(new[] { 1.0, double.NaN, 3.0 });
        await Assert.That(arr.Contains(double.NaN)).IsFalse();
    }

    [TestMethod]
    public async Task Contains_MaxValue_Int32()
    {
        var arr = np.array(new[] { int.MinValue, 0, int.MaxValue });
        await Assert.That(arr.Contains(int.MaxValue)).IsTrue();
        await Assert.That(arr.Contains(int.MinValue)).IsTrue();
    }

    [TestMethod]
    public async Task Contains_MaxValue_Int64()
    {
        var arr = np.array(new[] { long.MinValue, 0L, long.MaxValue });
        await Assert.That(arr.Contains(long.MaxValue)).IsTrue();
        await Assert.That(arr.Contains(long.MinValue)).IsTrue();
    }

    [TestMethod]
    public async Task Contains_TypePromotion_ByteInInt32Array()
    {
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });
        await Assert.That(arr.Contains((byte)3)).IsTrue();
    }

    [TestMethod]
    public async Task Contains_TypePromotion_Int32InInt64Array()
    {
        var arr = np.array(new long[] { 1, 2, 3, 4, 5 });
        await Assert.That(arr.Contains(3)).IsTrue();
    }

    [TestMethod]
    public async Task Contains_ZeroInMixedSignArray()
    {
        var arr = np.array(new[] { -5, -3, 0, 3, 5 });
        await Assert.That(arr.Contains(0)).IsTrue();
    }

    [TestMethod]
    public async Task Contains_SpecialFloats()
    {
        var arr = np.array(new[] {
            double.NegativeInfinity,
            double.MinValue,
            0.0,
            double.MaxValue,
            double.PositiveInfinity
        });

        await Assert.That(arr.Contains(double.NegativeInfinity)).IsTrue();
        await Assert.That(arr.Contains(double.PositiveInfinity)).IsTrue();
        await Assert.That(arr.Contains(double.MinValue)).IsTrue();
        await Assert.That(arr.Contains(double.MaxValue)).IsTrue();
    }

    [TestMethod]
    public async Task Contains_TransposedArray()
    {
        var arr = np.arange(12).reshape(3, 4).T;
        await Assert.That(arr.Contains(5)).IsTrue();
        await Assert.That(arr.Contains(100)).IsFalse();
    }

    [TestMethod]
    public async Task Contains_ReversedArray()
    {
        var arr = np.arange(10)["::-1"];
        await Assert.That(arr.Contains(5)).IsTrue();
        await Assert.That(arr.Contains(100)).IsFalse();
    }

    #endregion

    #region __iter__ Round 2

    [TestMethod]
    public async Task Iter_ViewIteration_CorrectValues()
    {
        var arr = np.arange(20).reshape(4, 5);
        var view = arr["1:3, :"];

        var rows = new List<NDArray>();
        foreach (var item in view)
        {
            rows.Add((NDArray)item);
        }

        await Assert.That(rows.Count).IsEqualTo(2);
        await Assert.That((int)rows[0][0]).IsEqualTo(5);
        await Assert.That((int)rows[1][0]).IsEqualTo(10);
    }

    [TestMethod]
    public async Task Iter_NestedIteration_3DArray()
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

        await Assert.That(allValues.Count).IsEqualTo(24);
        await Assert.That(allValues[0]).IsEqualTo(0);
        await Assert.That(allValues[23]).IsEqualTo(23);
    }

    [TestMethod]
    public async Task Iter_MultipleEnumerators_Independent()
    {
        var arr = np.array(new[] { 1, 2, 3 });

        var enum1 = arr.GetEnumerator();
        var enum2 = arr.GetEnumerator();

        enum1.MoveNext();
        enum1.MoveNext();
        enum2.MoveNext();

        await Assert.That(Convert.ToInt32(enum1.Current)).IsEqualTo(2);
        await Assert.That(Convert.ToInt32(enum2.Current)).IsEqualTo(1);
    }

    [TestMethod]
    public async Task Iter_ResetEnumerator_ThrowsNotSupported()
    {
        // NDIterator does not support Reset - document this behavior
        var arr = np.array(new[] { 1, 2, 3 });
        var enumerator = arr.GetEnumerator();

        enumerator.MoveNext();
        enumerator.MoveNext();
        await Assert.That(Convert.ToInt32(enumerator.Current)).IsEqualTo(2);

        // Reset throws NotSupportedException
        await Assert.That(() => enumerator.Reset()).Throws<NotSupportedException>();
    }

    [TestMethod]
    public async Task Iter_TransposedArray_CorrectOrder()
    {
        var arr = np.arange(6).reshape(2, 3).T; // Shape (3, 2)

        var rows = new List<NDArray>();
        foreach (var item in arr)
        {
            rows.Add((NDArray)item);
        }

        await Assert.That(rows.Count).IsEqualTo(3);
        // First row of transposed is [0, 3]
        await Assert.That((int)rows[0][0]).IsEqualTo(0);
        await Assert.That((int)rows[0][1]).IsEqualTo(3);
    }

    [TestMethod]
    public async Task Iter_SlicedArray_CorrectElements()
    {
        var arr = np.arange(20);
        var sliced = arr["5:10"]; // [5, 6, 7, 8, 9]

        var values = new List<int>();
        foreach (var item in sliced)
        {
            values.Add(Convert.ToInt32(item));
        }

        await Assert.That(values).IsEquivalentTo(new[] { 5, 6, 7, 8, 9 });
    }

    [TestMethod]
    public async Task Iter_StridedArray_CorrectElements()
    {
        var arr = np.arange(10);
        var strided = arr["1::2"]; // [1, 3, 5, 7, 9]

        var values = new List<int>();
        foreach (var item in strided)
        {
            values.Add(Convert.ToInt32(item));
        }

        await Assert.That(values).IsEquivalentTo(new[] { 1, 3, 5, 7, 9 });
    }

    [TestMethod]
    public async Task Iter_ReversedArray_CorrectOrder()
    {
        var arr = np.arange(5)["::-1"]; // [4, 3, 2, 1, 0]

        var values = new List<int>();
        foreach (var item in arr)
        {
            values.Add(Convert.ToInt32(item));
        }

        await Assert.That(values).IsEquivalentTo(new[] { 4, 3, 2, 1, 0 });
    }

    [TestMethod]
    public async Task Iter_BroadcastArray_IteratesOverBroadcastDimension()
    {
        var arr = np.array(new[] { 1, 2, 3 });
        var broadcast = np.broadcast_to(arr, new Shape(4, 3));

        var rows = new List<NDArray>();
        foreach (var item in broadcast)
        {
            rows.Add((NDArray)item);
        }

        await Assert.That(rows.Count).IsEqualTo(4);
        // All rows should be [1, 2, 3]
        foreach (var row in rows)
        {
            await Assert.That((int)row[0]).IsEqualTo(1);
            await Assert.That((int)row[2]).IsEqualTo(3);
        }
    }

    #endregion

    #region Edge Cases and Error Conditions

    [TestMethod]
    public async Task GetItem_EmptySlice_ReturnsEmpty()
    {
        var arr = np.arange(10);
        var empty = arr["5:5"]; // Empty slice
        await Assert.That(empty.size).IsEqualTo(0);
    }

    [TestMethod]
    public async Task GetItem_OutOfBoundsStart_ClampsToZero()
    {
        var arr = np.arange(5);
        var result = arr["-100:3"]; // Should be equivalent to [:3]
        await Assert.That(result.size).IsEqualTo(3);
    }

    [TestMethod]
    public async Task GetItem_OutOfBoundsStop_ClampsToEnd()
    {
        var arr = np.arange(5);
        var result = arr["2:100"]; // Should be equivalent to [2:]
        await Assert.That(result.size).IsEqualTo(3);
    }

    [TestMethod]
    public async Task Len_EmptyHighDimensional()
    {
        var arr = np.zeros(new long[] { 0, 5, 10 });
        await Assert.That(arr.__len__()).IsEqualTo(0);
    }

    [TestMethod]
    public async Task Contains_WrongType_ReturnsFalse()
    {
        var arr = np.array(new[] { 1, 2, 3 });
        // String can't be in int array
        await Assert.That(arr.Contains("hello")).IsFalse();
    }

    [TestMethod]
    public async Task Iter_EmptyDimension_NoIterations()
    {
        var arr = np.zeros(new long[] { 0, 5 });
        var count = 0;
        foreach (var _ in arr)
        {
            count++;
        }
        await Assert.That(count).IsEqualTo(0);
    }

    [TestMethod]
    public async Task Hash_DifferentArrays_AllThrow()
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
            await Assert.That(() => arr.GetHashCode()).Throws<NotSupportedException>();
        }
    }

    #endregion

    #region Python API Naming

    [TestMethod]
    public async Task PythonAPI_AllMethodsAccessible()
    {
        var arr = np.arange(12).reshape(3, 4);

        // All these should be callable without exception
        var len = arr.__len__();
        var contains = arr.__contains__(5);
        var item = arr.__getitem__(1);
        var iter = arr.__iter__();

        await Assert.That(len).IsEqualTo(3);
        await Assert.That(contains).IsTrue();
        await Assert.That(item.size).IsEqualTo(4);
        await Assert.That(iter).IsNotNull();
    }

    [TestMethod]
    public async Task PythonAPI_SetItem_Works()
    {
        var arr = np.zeros(new long[] { 3, 3 }, np.int32);
        arr.__setitem__(1, 5);

        await Assert.That((int)arr[1, 0]).IsEqualTo(5);
        await Assert.That((int)arr[1, 1]).IsEqualTo(5);
        await Assert.That((int)arr[1, 2]).IsEqualTo(5);
    }

    [TestMethod]
    public async Task PythonAPI_Hash_Throws()
    {
        var arr = np.array(new[] { 1, 2, 3 });
        await Assert.That(() => arr.__hash__()).Throws<NotSupportedException>();
    }

    #endregion
}
