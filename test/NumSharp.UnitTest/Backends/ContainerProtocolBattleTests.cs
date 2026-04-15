using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using NumSharp;

namespace NumSharp.UnitTest.Backends;

/// <summary>
/// Comprehensive battle tests for container protocol implementation.
/// Tests all edge cases, dtypes, and memory layouts.
/// </summary>
public class ContainerProtocolBattleTests
{
    #region __len__ Battle Tests

    [TestMethod]
    public async Task Len_AllDtypes_1DArray()
    {
        // Test __len__ works for all 12 dtypes
        np.array(new bool[] { true, false, true }).__len__().Should().Be(3);
        np.array(new byte[] { 1, 2, 3, 4 }).__len__().Should().Be(4);
        np.array(new short[] { 1, 2 }).__len__().Should().Be(2);
        np.array(new ushort[] { 1, 2, 3 }).__len__().Should().Be(3);
        np.array(new int[] { 1, 2, 3, 4, 5 }).__len__().Should().Be(5);
        np.array(new uint[] { 1 }).__len__().Should().Be(1);
        np.array(new long[] { 1, 2, 3, 4, 5, 6 }).__len__().Should().Be(6);
        np.array(new ulong[] { 1, 2 }).__len__().Should().Be(2);
        np.array(new char[] { 'a', 'b', 'c' }).__len__().Should().Be(3);
        np.array(new float[] { 1f, 2f }).__len__().Should().Be(2);
        np.array(new double[] { 1.0, 2.0, 3.0, 4.0 }).__len__().Should().Be(4);
        np.array(new decimal[] { 1m, 2m, 3m }).__len__().Should().Be(3);
    }

    [TestMethod]
    public async Task Len_3DArray_ReturnsFirstDimension()
    {
        // NumPy: len(np.zeros((2, 3, 4))) = 2
        var arr = np.zeros(new[] { 2, 3, 4 });
        arr.__len__().Should().Be(2);
    }

    [TestMethod]
    public async Task Len_4DArray_ReturnsFirstDimension()
    {
        var arr = np.zeros(new[] { 5, 4, 3, 2 });
        arr.__len__().Should().Be(5);
    }

    [TestMethod]
    public async Task Len_EmptyArray_ReturnsZero()
    {
        // NumPy: len(np.array([])) = 0
        var arr = np.array(Array.Empty<int>());
        arr.__len__().Should().Be(0);
    }

    [TestMethod]
    public async Task Len_EmptyArray_2D_ReturnsZero()
    {
        // NumPy: len(np.zeros((0, 5))) = 0
        var arr = np.zeros(new[] { 0, 5 });
        arr.__len__().Should().Be(0);
    }

    [TestMethod]
    public async Task Len_SlicedArray_ReturnsCorrectLength()
    {
        var arr = np.arange(10);
        var sliced = arr["2:7"]; // [2, 3, 4, 5, 6]
        sliced.__len__().Should().Be(5);
    }

    [TestMethod]
    public async Task Len_SlicedArray_Strided_ReturnsCorrectLength()
    {
        var arr = np.arange(10);
        var sliced = arr["::2"]; // [0, 2, 4, 6, 8]
        sliced.__len__().Should().Be(5);
    }

    [TestMethod]
    public async Task Len_SlicedArray_Reversed_ReturnsCorrectLength()
    {
        var arr = np.arange(10);
        var sliced = arr["::-1"]; // [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        sliced.__len__().Should().Be(10);
    }

    [TestMethod]
    public async Task Len_TransposedArray_ReturnsFirstDimension()
    {
        // NumPy: len(np.zeros((3, 5)).T) = 5
        var arr = np.zeros(new[] { 3, 5 });
        var transposed = arr.T;
        transposed.__len__().Should().Be(5);
    }

    [TestMethod]
    public async Task Len_ReshapedArray_ReturnsFirstDimension()
    {
        var arr = np.arange(12).reshape(3, 4);
        arr.__len__().Should().Be(3);
    }

    [TestMethod]
    public async Task Len_BroadcastArray_ReturnsFirstDimension()
    {
        var arr = np.array(new[] { 1, 2, 3 });
        var broadcast = np.broadcast_to(arr, new Shape(4, 3));
        broadcast.__len__().Should().Be(4);
    }

    [TestMethod]
    public async Task Len_SingleElementArray_ReturnsOne()
    {
        var arr = np.array(new[] { 42 });
        arr.__len__().Should().Be(1);
    }

    [TestMethod]
    public async Task Len_ScalarAllDtypes_ThrowsTypeError()
    {
        // All scalar types should throw TypeError
        Assert.Throws<TypeError>(() => NDArray.Scalar(true).__len__());
        Assert.Throws<TypeError>(() => NDArray.Scalar((byte)1).__len__());
        Assert.Throws<TypeError>(() => NDArray.Scalar((short)1).__len__());
        Assert.Throws<TypeError>(() => NDArray.Scalar((ushort)1).__len__());
        Assert.Throws<TypeError>(() => NDArray.Scalar(1).__len__());
        Assert.Throws<TypeError>(() => NDArray.Scalar(1u).__len__());
        Assert.Throws<TypeError>(() => NDArray.Scalar(1L).__len__());
        Assert.Throws<TypeError>(() => NDArray.Scalar(1ul).__len__());
        Assert.Throws<TypeError>(() => NDArray.Scalar('a').__len__());
        Assert.Throws<TypeError>(() => NDArray.Scalar(1f).__len__());
        Assert.Throws<TypeError>(() => NDArray.Scalar(1.0).__len__());
        Assert.Throws<TypeError>(() => NDArray.Scalar(1m).__len__());
    }

    #endregion

    #region __getitem__ Battle Tests

    [TestMethod]
    public async Task GetItem_AllDtypes_IntIndex()
    {
        ((bool)np.array(new[] { true, false }).__getitem__(0)).Should().BeTrue();
        ((int)(byte)np.array(new byte[] { 1, 2 }).__getitem__(1)).Should().Be(2);
        ((int)(short)np.array(new short[] { 10, 20 }).__getitem__(0)).Should().Be(10);
        ((int)(ushort)np.array(new ushort[] { 100, 200 }).__getitem__(1)).Should().Be(200);
        ((int)np.array(new[] { 1, 2, 3 }).__getitem__(2)).Should().Be(3);
        ((long)(uint)np.array(new uint[] { 1, 2 }).__getitem__(0)).Should().Be(1L);
        ((long)np.array(new long[] { 100L, 200L }).__getitem__(1)).Should().Be(200L);
        ((long)(ulong)np.array(new ulong[] { 1, 2 }).__getitem__(0)).Should().Be(1L);
        ((char)np.array(new[] { 'x', 'y' }).__getitem__(1)).Should().Be('y');
        ((float)np.array(new[] { 1.5f, 2.5f }).__getitem__(0)).Should().Be(1.5f);
        ((double)np.array(new[] { 1.5, 2.5 }).__getitem__(1)).Should().Be(2.5);
        ((decimal)np.array(new[] { 1.5m, 2.5m }).__getitem__(0)).Should().Be(1.5m);
    }

    [TestMethod]
    public async Task GetItem_NegativeIndex_AllPositions()
    {
        var arr = np.array(new[] { 10, 20, 30, 40, 50 });

        ((int)arr.__getitem__(-1)).Should().Be(50);
        ((int)arr.__getitem__(-2)).Should().Be(40);
        ((int)arr.__getitem__(-3)).Should().Be(30);
        ((int)arr.__getitem__(-4)).Should().Be(20);
        ((int)arr.__getitem__(-5)).Should().Be(10);
    }

    [TestMethod]
    public async Task GetItem_SliceStrings_Various()
    {
        var arr = np.array(new[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });

        // Start:Stop
        arr.__getitem__("2:5").size.Should().Be(3);
        ((int)arr.__getitem__("2:5")[0]).Should().Be(2);

        // :Stop
        arr.__getitem__(":3").size.Should().Be(3);
        ((int)arr.__getitem__(":3")[0]).Should().Be(0);

        // Start:
        arr.__getitem__("7:").size.Should().Be(3);
        ((int)arr.__getitem__("7:")[0]).Should().Be(7);

        // ::Step
        arr.__getitem__("::2").size.Should().Be(5);
        ((int)arr.__getitem__("::2")[1]).Should().Be(2);

        // ::-1 (reverse)
        arr.__getitem__("::-1").size.Should().Be(10);
        ((int)arr.__getitem__("::-1")[0]).Should().Be(9);

        // Negative indices
        arr.__getitem__("-3:").size.Should().Be(3);
        ((int)arr.__getitem__("-3:")[0]).Should().Be(7);
    }

    [TestMethod]
    public async Task GetItem_2DArray_RowAccess()
    {
        var arr = np.array(new[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } });

        var row0 = arr.__getitem__(0);
        row0.ndim.Should().Be(1);
        row0.size.Should().Be(3);
        (((int)row0[0])).Should().Be(1);
        (((int)row0[2])).Should().Be(3);

        var row2 = arr.__getitem__(-1);
        (((int)row2[0])).Should().Be(7);
    }

    [TestMethod]
    public async Task GetItem_SliceReturnsView_NotCopy()
    {
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });
        var slice = arr.__getitem__("1:4");

        // Modify original
        arr[2] = 99;

        // View should reflect change
        (((int)slice[1])).Should().Be(99);
    }

    [TestMethod]
    public async Task GetItem_OutOfBounds_Throws()
    {
        var arr = np.array(new[] { 1, 2, 3 });

        Assert.Throws<Exception>(() => arr.__getitem__(10));
        Assert.Throws<Exception>(() => arr.__getitem__(-10));
    }

    [TestMethod]
    public async Task GetItem_EmptySlice_ReturnsEmptyArray()
    {
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });
        var empty = arr.__getitem__("2:2"); // Empty slice

        empty.size.Should().Be(0);
    }

    [TestMethod]
    public async Task GetItem_SlicedSourceArray()
    {
        var arr = np.arange(20);
        var sliced = arr["5:15"]; // [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        var subslice = sliced.__getitem__("2:5");

        (((int)subslice[0])).Should().Be(7);
        subslice.size.Should().Be(3);
    }

    #endregion

    #region __setitem__ Battle Tests

    [TestMethod]
    public async Task SetItem_AllDtypes_ScalarAssignment()
    {
        var boolArr = np.array(new[] { true, false });
        boolArr.__setitem__(0, false);
        (((bool)boolArr[0])).Should().BeFalse();

        var intArr = np.array(new[] { 1, 2, 3 });
        intArr.__setitem__(1, 99);
        (((int)intArr[1])).Should().Be(99);

        var doubleArr = np.array(new[] { 1.0, 2.0, 3.0 });
        doubleArr.__setitem__(2, 99.5);
        (((double)doubleArr[2])).Should().Be(99.5);
    }

    [TestMethod]
    public async Task SetItem_NegativeIndex()
    {
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });

        arr.__setitem__(-1, 99);
        (((int)arr[4])).Should().Be(99);

        arr.__setitem__(-3, 88);
        (((int)arr[2])).Should().Be(88);
    }

    [TestMethod]
    public async Task SetItem_SliceString_ScalarBroadcast()
    {
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });

        arr.__setitem__("1:4", 0);

        (((int)arr[0])).Should().Be(1);  // Unchanged
        (((int)arr[1])).Should().Be(0);
        (((int)arr[2])).Should().Be(0);
        (((int)arr[3])).Should().Be(0);
        (((int)arr[4])).Should().Be(5);  // Unchanged
    }

    [TestMethod]
    public async Task SetItem_SliceString_ArrayAssignment()
    {
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });

        arr.__setitem__("1:4", np.array(new[] { 10, 20, 30 }));

        (((int)arr[0])).Should().Be(1);
        (((int)arr[1])).Should().Be(10);
        (((int)arr[2])).Should().Be(20);
        (((int)arr[3])).Should().Be(30);
        (((int)arr[4])).Should().Be(5);
    }

    [TestMethod]
    public async Task SetItem_ViewAffectsOriginal()
    {
        var original = np.array(new[] { 1, 2, 3, 4, 5 });
        var view = original["1:4"];

        view.__setitem__(0, 99);

        (((int)original[1])).Should().Be(99);
    }

    [TestMethod]
    public async Task SetItem_2DArray_RowAssignment()
    {
        var arr = np.array(new[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } });

        arr.__setitem__(1, np.array(new[] { 10, 20, 30 }));

        (((int)arr[1, 0])).Should().Be(10);
        (((int)arr[1, 1])).Should().Be(20);
        (((int)arr[1, 2])).Should().Be(30);
    }

    [TestMethod]
    public async Task SetItem_TypePromotion()
    {
        var arr = np.array(new[] { 1.0, 2.0, 3.0 });

        // Set with int, should promote to double
        arr.__setitem__(1, 99);

        (((double)arr[1])).Should().Be(99.0);
    }

    [TestMethod]
    public async Task SetItem_AllElements_WithColon()
    {
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });

        arr.__setitem__(":", 0);

        np.all(arr == 0).Should().BeTrue();
    }

    [TestMethod]
    public async Task SetItem_ReversedSlice()
    {
        // NumPy: arr[::-1] = [10, 20, 30, 40, 50] assigns in reverse order
        // arr[4]=10, arr[3]=20, arr[2]=30, arr[1]=40, arr[0]=50
        // Result: [50, 40, 30, 20, 10]
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });

        arr.__setitem__("::-1", np.array(new[] { 10, 20, 30, 40, 50 }));

        (((int)arr[0])).Should().Be(50); // Last of source
        (((int)arr[4])).Should().Be(10); // First of source
    }

    #endregion

    #region __hash__ Battle Tests

    [TestMethod]
    public async Task Hash_AllDtypes_Throw()
    {
        Assert.Throws<NotSupportedException>(() => np.array(new[] { true, false }).GetHashCode());
        Assert.Throws<NotSupportedException>(() => np.array(new byte[] { 1, 2 }).GetHashCode());
        Assert.Throws<NotSupportedException>(() => np.array(new short[] { 1, 2 }).GetHashCode());
        Assert.Throws<NotSupportedException>(() => np.array(new ushort[] { 1, 2 }).GetHashCode());
        Assert.Throws<NotSupportedException>(() => np.array(new[] { 1, 2 }).GetHashCode());
        Assert.Throws<NotSupportedException>(() => np.array(new uint[] { 1, 2 }).GetHashCode());
        Assert.Throws<NotSupportedException>(() => np.array(new long[] { 1, 2 }).GetHashCode());
        Assert.Throws<NotSupportedException>(() => np.array(new ulong[] { 1, 2 }).GetHashCode());
        Assert.Throws<NotSupportedException>(() => np.array(new[] { 'a', 'b' }).GetHashCode());
        Assert.Throws<NotSupportedException>(() => np.array(new[] { 1f, 2f }).GetHashCode());
        Assert.Throws<NotSupportedException>(() => np.array(new[] { 1.0, 2.0 }).GetHashCode());
        Assert.Throws<NotSupportedException>(() => np.array(new[] { 1m, 2m }).GetHashCode());
    }

    [TestMethod]
    public async Task Hash_EmptyArray_Throws()
    {
        Assert.Throws<NotSupportedException>(() => np.array(Array.Empty<int>()).GetHashCode());
    }

    [TestMethod]
    public async Task Hash_ScalarArray_Throws()
    {
        Assert.Throws<NotSupportedException>(() => NDArray.Scalar(5).GetHashCode());
    }

    [TestMethod]
    public async Task Hash_SlicedArray_Throws()
    {
        var arr = np.arange(10);
        var sliced = arr["2:8"];
        Assert.Throws<NotSupportedException>(() => sliced.GetHashCode());
    }

    [TestMethod]
    public async Task Hash_BroadcastArray_Throws()
    {
        var arr = np.array(new[] { 1, 2, 3 });
        var broadcast = np.broadcast_to(arr, new Shape(4, 3));
        Assert.Throws<NotSupportedException>(() => broadcast.GetHashCode());
    }

    [TestMethod]
    public async Task Hash_HashSetUsage_Fails()
    {
        var arr = np.array(new[] { 1, 2, 3 });
        var set = new HashSet<object>();

        Assert.Throws<NotSupportedException>(() => set.Add(arr));
    }

    [TestMethod]
    public async Task Hash_ErrorMessage_IsDescriptive()
    {
        var arr = np.array(new[] { 1, 2, 3 });

        try
        {
            arr.GetHashCode();
            throw new Exception("Should have thrown NotSupportedException");
        }
        catch (NotSupportedException ex)
        {
            ex.Message.Should().Contain("unhashable");
            ex.Message.Should().Contain("mutable");
        }
    }

    #endregion

    #region __contains__ Battle Tests

    [TestMethod]
    public async Task Contains_AllDtypes()
    {
        np.array(new[] { true, false }).Contains(true).Should().BeTrue();
        np.array(new byte[] { 1, 2, 3 }).Contains(((byte)2)).Should().BeTrue();
        np.array(new short[] { 1, 2, 3 }).Contains(((short)2)).Should().BeTrue();
        np.array(new ushort[] { 1, 2, 3 }).Contains(((ushort)2)).Should().BeTrue();
        np.array(new[] { 1, 2, 3 }).Contains(2).Should().BeTrue();
        np.array(new uint[] { 1, 2, 3 }).Contains(2u).Should().BeTrue();
        np.array(new long[] { 1, 2, 3 }).Contains(2L).Should().BeTrue();
        np.array(new ulong[] { 1, 2, 3 }).Contains(2ul).Should().BeTrue();
        np.array(new[] { 'a', 'b', 'c' }).Contains('b').Should().BeTrue();
        np.array(new[] { 1f, 2f, 3f }).Contains(2f).Should().BeTrue();
        np.array(new[] { 1.0, 2.0, 3.0 }).Contains(2.0).Should().BeTrue();
        np.array(new[] { 1m, 2m, 3m }).Contains(2m).Should().BeTrue();
    }

    [TestMethod]
    public async Task Contains_LargeArray()
    {
        var arr = np.arange(10000);

        arr.Contains(0).Should().BeTrue();
        arr.Contains(5000).Should().BeTrue();
        arr.Contains(9999).Should().BeTrue();
        arr.Contains(10000).Should().BeFalse();
    }

    [TestMethod]
    public async Task Contains_Infinity()
    {
        var arr = np.array(new[] { double.NegativeInfinity, 0.0, double.PositiveInfinity });

        arr.Contains(double.PositiveInfinity).Should().BeTrue();
        arr.Contains(double.NegativeInfinity).Should().BeTrue();
        arr.Contains(0.0).Should().BeTrue();
    }

    [TestMethod]
    public async Task Contains_NegativeValues()
    {
        var arr = np.array(new[] { -5, -3, -1, 0, 1, 3, 5 });

        arr.Contains(-3).Should().BeTrue();
        arr.Contains(-10).Should().BeFalse();
    }

    [TestMethod]
    public async Task Contains_3DArray()
    {
        var arr = np.arange(24).reshape(2, 3, 4);

        arr.Contains(0).Should().BeTrue();
        arr.Contains(23).Should().BeTrue();
        arr.Contains(12).Should().BeTrue();
        arr.Contains(100).Should().BeFalse();
    }

    [TestMethod]
    public async Task Contains_SlicedArray()
    {
        var arr = np.arange(10);
        var sliced = arr["3:7"]; // [3, 4, 5, 6]

        sliced.Contains(5).Should().BeTrue();
        sliced.Contains(2).Should().BeFalse(); // Not in slice
        sliced.Contains(8).Should().BeFalse(); // Not in slice
    }

    [TestMethod]
    public async Task Contains_StridedArray()
    {
        var arr = np.arange(10);
        var strided = arr["::2"]; // [0, 2, 4, 6, 8]

        strided.Contains(4).Should().BeTrue();
        strided.Contains(3).Should().BeFalse(); // Odd numbers not included
    }

    [TestMethod]
    public async Task Contains_BroadcastArray()
    {
        var arr = np.array(new[] { 1, 2, 3 });
        var broadcast = np.broadcast_to(arr, new Shape(4, 3));

        broadcast.Contains(2).Should().BeTrue();
        broadcast.Contains(5).Should().BeFalse();
    }

    [TestMethod]
    public async Task Contains_ScalarArray()
    {
        var scalar = NDArray.Scalar(42);

        scalar.Contains(42).Should().BeTrue();
        scalar.Contains(0).Should().BeFalse();
    }

    #endregion

    #region __iter__ Battle Tests

    [TestMethod]
    public async Task Iter_AllDtypes_Enumerate()
    {
        // Just verify iteration doesn't crash for all types
        foreach (var _ in np.array(new[] { true, false })) { }
        foreach (var _ in np.array(new byte[] { 1, 2 })) { }
        foreach (var _ in np.array(new short[] { 1, 2 })) { }
        foreach (var _ in np.array(new ushort[] { 1, 2 })) { }
        foreach (var _ in np.array(new[] { 1, 2 })) { }
        foreach (var _ in np.array(new uint[] { 1, 2 })) { }
        foreach (var _ in np.array(new long[] { 1, 2 })) { }
        foreach (var _ in np.array(new ulong[] { 1, 2 })) { }
        foreach (var _ in np.array(new[] { 'a', 'b' })) { }
        foreach (var _ in np.array(new[] { 1f, 2f })) { }
        foreach (var _ in np.array(new[] { 1.0, 2.0 })) { }
        foreach (var _ in np.array(new[] { 1m, 2m })) { }

        true.Should().BeTrue(); // Passed if no exception
    }

    [TestMethod]
    public async Task Iter_1DArray_ElementsMatch()
    {
        var arr = np.array(new[] { 10, 20, 30 });
        var collected = new List<int>();

        // 1D arrays iterate over elements (int values), not NDArray
        foreach (var item in arr)
        {
            collected.Add(Convert.ToInt32(item));
        }

        collected.Should().BeEquivalentTo(new[] { 10, 20, 30 });
    }

    [TestMethod]
    public async Task Iter_2DArray_IteratesRows()
    {
        var arr = np.array(new[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
        var rowCount = 0;

        // 2D arrays iterate over rows (NDArray slices)
        foreach (var item in arr)
        {
            if (item is NDArray row)
            {
                row.ndim.Should().Be(1);
                row.size.Should().Be(2);
            }
            rowCount++;
        }

        rowCount.Should().Be(3);
    }

    [TestMethod]
    public async Task Iter_3DArray_Iterates2DSlices()
    {
        var arr = np.zeros(new[] { 2, 3, 4 });
        var sliceCount = 0;

        // 3D arrays iterate over 2D slices
        foreach (var item in arr)
        {
            if (item is NDArray slice)
            {
                slice.ndim.Should().Be(2);
                slice.shape.Should().BeEquivalentTo(new long[] { 3, 4 });
            }
            sliceCount++;
        }

        sliceCount.Should().Be(2);
    }

    [TestMethod]
    public async Task Iter_EmptyArray_NoIterations()
    {
        var arr = np.array(Array.Empty<int>());
        var count = 0;

        foreach (var _ in arr)
        {
            count++;
        }

        count.Should().Be(0);
    }

    [TestMethod]
    public async Task Iter_SingleElement_OneIteration()
    {
        var arr = np.array(new[] { 42 });
        var count = 0;
        var value = 0;

        // 1D array iterates over elements
        foreach (var item in arr)
        {
            value = Convert.ToInt32(item);
            count++;
        }

        count.Should().Be(1);
        value.Should().Be(42);
    }

    [TestMethod]
    public async Task Iter_MultipleEnumeration()
    {
        var arr = np.array(new[] { 1, 2, 3 });

        // First enumeration
        var first = new List<int>();
        foreach (var item in arr)
            first.Add(Convert.ToInt32(item));

        // Second enumeration
        var second = new List<int>();
        foreach (var item in arr)
            second.Add(Convert.ToInt32(item));

        first.Should().BeEquivalentTo(second);
    }

    [TestMethod]
    public async Task Iter_LINQ_ToList()
    {
        var arr = np.array(new[] { 1, 2, 3 });

        var list = arr.Cast<object>().ToList();

        list.Count.Should().Be(3);
    }

    [TestMethod]
    public async Task Iter_LINQ_Count()
    {
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });

        var count = arr.Cast<object>().Count();

        count.Should().Be(5);
    }

    [TestMethod]
    public async Task Iter_SlicedArray()
    {
        var arr = np.arange(10);
        var sliced = arr["2:7"];
        var count = 0;

        foreach (var _ in sliced)
            count++;

        count.Should().Be(5);
    }

    [TestMethod]
    public async Task Iter_BreakEarly()
    {
        var arr = np.arange(100);
        var count = 0;

        foreach (var _ in arr)
        {
            count++;
            if (count >= 5)
                break;
        }

        count.Should().Be(5);
    }

    [TestMethod]
    public async Task Iter_ScalarArray_ThrowsTypeError()
    {
        // NumPy: iteration over a 0-d array throws TypeError
        var scalar = NDArray.Scalar(42);

        Assert.Throws<TypeError>(() =>
        {
            foreach (var _ in scalar)
            {
                // Should not reach here
            }
        });
    }

    #endregion
}
