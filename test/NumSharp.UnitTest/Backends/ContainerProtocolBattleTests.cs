using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using NumSharp;
using TUnit.Core;
using TUnit.Assertions;
using TUnit.Assertions.Extensions;

namespace NumSharp.UnitTest.Backends;

/// <summary>
/// Comprehensive battle tests for container protocol implementation.
/// Tests all edge cases, dtypes, and memory layouts.
/// </summary>
public class ContainerProtocolBattleTests
{
    #region __len__ Battle Tests

    [Test]
    public async Task Len_AllDtypes_1DArray()
    {
        // Test __len__ works for all 12 dtypes
        await Assert.That(np.array(new bool[] { true, false, true }).__len__()).IsEqualTo(3);
        await Assert.That(np.array(new byte[] { 1, 2, 3, 4 }).__len__()).IsEqualTo(4);
        await Assert.That(np.array(new short[] { 1, 2 }).__len__()).IsEqualTo(2);
        await Assert.That(np.array(new ushort[] { 1, 2, 3 }).__len__()).IsEqualTo(3);
        await Assert.That(np.array(new int[] { 1, 2, 3, 4, 5 }).__len__()).IsEqualTo(5);
        await Assert.That(np.array(new uint[] { 1 }).__len__()).IsEqualTo(1);
        await Assert.That(np.array(new long[] { 1, 2, 3, 4, 5, 6 }).__len__()).IsEqualTo(6);
        await Assert.That(np.array(new ulong[] { 1, 2 }).__len__()).IsEqualTo(2);
        await Assert.That(np.array(new char[] { 'a', 'b', 'c' }).__len__()).IsEqualTo(3);
        await Assert.That(np.array(new float[] { 1f, 2f }).__len__()).IsEqualTo(2);
        await Assert.That(np.array(new double[] { 1.0, 2.0, 3.0, 4.0 }).__len__()).IsEqualTo(4);
        await Assert.That(np.array(new decimal[] { 1m, 2m, 3m }).__len__()).IsEqualTo(3);
    }

    [Test]
    public async Task Len_3DArray_ReturnsFirstDimension()
    {
        // NumPy: len(np.zeros((2, 3, 4))) = 2
        var arr = np.zeros(new[] { 2, 3, 4 });
        await Assert.That(arr.__len__()).IsEqualTo(2);
    }

    [Test]
    public async Task Len_4DArray_ReturnsFirstDimension()
    {
        var arr = np.zeros(new[] { 5, 4, 3, 2 });
        await Assert.That(arr.__len__()).IsEqualTo(5);
    }

    [Test]
    public async Task Len_EmptyArray_ReturnsZero()
    {
        // NumPy: len(np.array([])) = 0
        var arr = np.array(Array.Empty<int>());
        await Assert.That(arr.__len__()).IsEqualTo(0);
    }

    [Test]
    public async Task Len_EmptyArray_2D_ReturnsZero()
    {
        // NumPy: len(np.zeros((0, 5))) = 0
        var arr = np.zeros(new[] { 0, 5 });
        await Assert.That(arr.__len__()).IsEqualTo(0);
    }

    [Test]
    public async Task Len_SlicedArray_ReturnsCorrectLength()
    {
        var arr = np.arange(10);
        var sliced = arr["2:7"]; // [2, 3, 4, 5, 6]
        await Assert.That(sliced.__len__()).IsEqualTo(5);
    }

    [Test]
    public async Task Len_SlicedArray_Strided_ReturnsCorrectLength()
    {
        var arr = np.arange(10);
        var sliced = arr["::2"]; // [0, 2, 4, 6, 8]
        await Assert.That(sliced.__len__()).IsEqualTo(5);
    }

    [Test]
    public async Task Len_SlicedArray_Reversed_ReturnsCorrectLength()
    {
        var arr = np.arange(10);
        var sliced = arr["::-1"]; // [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        await Assert.That(sliced.__len__()).IsEqualTo(10);
    }

    [Test]
    public async Task Len_TransposedArray_ReturnsFirstDimension()
    {
        // NumPy: len(np.zeros((3, 5)).T) = 5
        var arr = np.zeros(new[] { 3, 5 });
        var transposed = arr.T;
        await Assert.That(transposed.__len__()).IsEqualTo(5);
    }

    [Test]
    public async Task Len_ReshapedArray_ReturnsFirstDimension()
    {
        var arr = np.arange(12).reshape(3, 4);
        await Assert.That(arr.__len__()).IsEqualTo(3);
    }

    [Test]
    public async Task Len_BroadcastArray_ReturnsFirstDimension()
    {
        var arr = np.array(new[] { 1, 2, 3 });
        var broadcast = np.broadcast_to(arr, new Shape(4, 3));
        await Assert.That(broadcast.__len__()).IsEqualTo(4);
    }

    [Test]
    public async Task Len_SingleElementArray_ReturnsOne()
    {
        var arr = np.array(new[] { 42 });
        await Assert.That(arr.__len__()).IsEqualTo(1);
    }

    [Test]
    public async Task Len_ScalarAllDtypes_ThrowsTypeError()
    {
        // All scalar types should throw TypeError
        await Assert.That(() => NDArray.Scalar(true).__len__()).Throws<TypeError>();
        await Assert.That(() => NDArray.Scalar((byte)1).__len__()).Throws<TypeError>();
        await Assert.That(() => NDArray.Scalar((short)1).__len__()).Throws<TypeError>();
        await Assert.That(() => NDArray.Scalar((ushort)1).__len__()).Throws<TypeError>();
        await Assert.That(() => NDArray.Scalar(1).__len__()).Throws<TypeError>();
        await Assert.That(() => NDArray.Scalar(1u).__len__()).Throws<TypeError>();
        await Assert.That(() => NDArray.Scalar(1L).__len__()).Throws<TypeError>();
        await Assert.That(() => NDArray.Scalar(1ul).__len__()).Throws<TypeError>();
        await Assert.That(() => NDArray.Scalar('a').__len__()).Throws<TypeError>();
        await Assert.That(() => NDArray.Scalar(1f).__len__()).Throws<TypeError>();
        await Assert.That(() => NDArray.Scalar(1.0).__len__()).Throws<TypeError>();
        await Assert.That(() => NDArray.Scalar(1m).__len__()).Throws<TypeError>();
    }

    #endregion

    #region __getitem__ Battle Tests

    [Test]
    public async Task GetItem_AllDtypes_IntIndex()
    {
        await Assert.That((bool)np.array(new[] { true, false }).__getitem__(0)).IsTrue();
        await Assert.That((int)(byte)np.array(new byte[] { 1, 2 }).__getitem__(1)).IsEqualTo(2);
        await Assert.That((int)(short)np.array(new short[] { 10, 20 }).__getitem__(0)).IsEqualTo(10);
        await Assert.That((int)(ushort)np.array(new ushort[] { 100, 200 }).__getitem__(1)).IsEqualTo(200);
        await Assert.That((int)np.array(new[] { 1, 2, 3 }).__getitem__(2)).IsEqualTo(3);
        await Assert.That((long)(uint)np.array(new uint[] { 1, 2 }).__getitem__(0)).IsEqualTo(1L);
        await Assert.That((long)np.array(new long[] { 100L, 200L }).__getitem__(1)).IsEqualTo(200L);
        await Assert.That((long)(ulong)np.array(new ulong[] { 1, 2 }).__getitem__(0)).IsEqualTo(1L);
        await Assert.That((char)np.array(new[] { 'x', 'y' }).__getitem__(1)).IsEqualTo('y');
        await Assert.That((float)np.array(new[] { 1.5f, 2.5f }).__getitem__(0)).IsEqualTo(1.5f);
        await Assert.That((double)np.array(new[] { 1.5, 2.5 }).__getitem__(1)).IsEqualTo(2.5);
        await Assert.That((decimal)np.array(new[] { 1.5m, 2.5m }).__getitem__(0)).IsEqualTo(1.5m);
    }

    [Test]
    public async Task GetItem_NegativeIndex_AllPositions()
    {
        var arr = np.array(new[] { 10, 20, 30, 40, 50 });

        await Assert.That((int)arr.__getitem__(-1)).IsEqualTo(50);
        await Assert.That((int)arr.__getitem__(-2)).IsEqualTo(40);
        await Assert.That((int)arr.__getitem__(-3)).IsEqualTo(30);
        await Assert.That((int)arr.__getitem__(-4)).IsEqualTo(20);
        await Assert.That((int)arr.__getitem__(-5)).IsEqualTo(10);
    }

    [Test]
    public async Task GetItem_SliceStrings_Various()
    {
        var arr = np.array(new[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });

        // Start:Stop
        await Assert.That(arr.__getitem__("2:5").size).IsEqualTo(3);
        await Assert.That((int)arr.__getitem__("2:5")[0]).IsEqualTo(2);

        // :Stop
        await Assert.That(arr.__getitem__(":3").size).IsEqualTo(3);
        await Assert.That((int)arr.__getitem__(":3")[0]).IsEqualTo(0);

        // Start:
        await Assert.That(arr.__getitem__("7:").size).IsEqualTo(3);
        await Assert.That((int)arr.__getitem__("7:")[0]).IsEqualTo(7);

        // ::Step
        await Assert.That(arr.__getitem__("::2").size).IsEqualTo(5);
        await Assert.That((int)arr.__getitem__("::2")[1]).IsEqualTo(2);

        // ::-1 (reverse)
        await Assert.That(arr.__getitem__("::-1").size).IsEqualTo(10);
        await Assert.That((int)arr.__getitem__("::-1")[0]).IsEqualTo(9);

        // Negative indices
        await Assert.That(arr.__getitem__("-3:").size).IsEqualTo(3);
        await Assert.That((int)arr.__getitem__("-3:")[0]).IsEqualTo(7);
    }

    [Test]
    public async Task GetItem_2DArray_RowAccess()
    {
        var arr = np.array(new[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } });

        var row0 = arr.__getitem__(0);
        await Assert.That(row0.ndim).IsEqualTo(1);
        await Assert.That(row0.size).IsEqualTo(3);
        await Assert.That((int)row0[0]).IsEqualTo(1);
        await Assert.That((int)row0[2]).IsEqualTo(3);

        var row2 = arr.__getitem__(-1);
        await Assert.That((int)row2[0]).IsEqualTo(7);
    }

    [Test]
    public async Task GetItem_SliceReturnsView_NotCopy()
    {
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });
        var slice = arr.__getitem__("1:4");

        // Modify original
        arr[2] = 99;

        // View should reflect change
        await Assert.That((int)slice[1]).IsEqualTo(99);
    }

    [Test]
    public async Task GetItem_OutOfBounds_Throws()
    {
        var arr = np.array(new[] { 1, 2, 3 });

        await Assert.That(() => arr.__getitem__(10)).Throws<Exception>();
        await Assert.That(() => arr.__getitem__(-10)).Throws<Exception>();
    }

    [Test]
    public async Task GetItem_EmptySlice_ReturnsEmptyArray()
    {
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });
        var empty = arr.__getitem__("2:2"); // Empty slice

        await Assert.That(empty.size).IsEqualTo(0);
    }

    [Test]
    public async Task GetItem_SlicedSourceArray()
    {
        var arr = np.arange(20);
        var sliced = arr["5:15"]; // [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        var subslice = sliced.__getitem__("2:5");

        await Assert.That((int)subslice[0]).IsEqualTo(7);
        await Assert.That(subslice.size).IsEqualTo(3);
    }

    #endregion

    #region __setitem__ Battle Tests

    [Test]
    public async Task SetItem_AllDtypes_ScalarAssignment()
    {
        var boolArr = np.array(new[] { true, false });
        boolArr.__setitem__(0, false);
        await Assert.That((bool)boolArr[0]).IsFalse();

        var intArr = np.array(new[] { 1, 2, 3 });
        intArr.__setitem__(1, 99);
        await Assert.That((int)intArr[1]).IsEqualTo(99);

        var doubleArr = np.array(new[] { 1.0, 2.0, 3.0 });
        doubleArr.__setitem__(2, 99.5);
        await Assert.That((double)doubleArr[2]).IsEqualTo(99.5);
    }

    [Test]
    public async Task SetItem_NegativeIndex()
    {
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });

        arr.__setitem__(-1, 99);
        await Assert.That((int)arr[4]).IsEqualTo(99);

        arr.__setitem__(-3, 88);
        await Assert.That((int)arr[2]).IsEqualTo(88);
    }

    [Test]
    public async Task SetItem_SliceString_ScalarBroadcast()
    {
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });

        arr.__setitem__("1:4", 0);

        await Assert.That((int)arr[0]).IsEqualTo(1);  // Unchanged
        await Assert.That((int)arr[1]).IsEqualTo(0);
        await Assert.That((int)arr[2]).IsEqualTo(0);
        await Assert.That((int)arr[3]).IsEqualTo(0);
        await Assert.That((int)arr[4]).IsEqualTo(5);  // Unchanged
    }

    [Test]
    public async Task SetItem_SliceString_ArrayAssignment()
    {
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });

        arr.__setitem__("1:4", np.array(new[] { 10, 20, 30 }));

        await Assert.That((int)arr[0]).IsEqualTo(1);
        await Assert.That((int)arr[1]).IsEqualTo(10);
        await Assert.That((int)arr[2]).IsEqualTo(20);
        await Assert.That((int)arr[3]).IsEqualTo(30);
        await Assert.That((int)arr[4]).IsEqualTo(5);
    }

    [Test]
    public async Task SetItem_ViewAffectsOriginal()
    {
        var original = np.array(new[] { 1, 2, 3, 4, 5 });
        var view = original["1:4"];

        view.__setitem__(0, 99);

        await Assert.That((int)original[1]).IsEqualTo(99);
    }

    [Test]
    public async Task SetItem_2DArray_RowAssignment()
    {
        var arr = np.array(new[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } });

        arr.__setitem__(1, np.array(new[] { 10, 20, 30 }));

        await Assert.That((int)arr[1, 0]).IsEqualTo(10);
        await Assert.That((int)arr[1, 1]).IsEqualTo(20);
        await Assert.That((int)arr[1, 2]).IsEqualTo(30);
    }

    [Test]
    public async Task SetItem_TypePromotion()
    {
        var arr = np.array(new[] { 1.0, 2.0, 3.0 });

        // Set with int, should promote to double
        arr.__setitem__(1, 99);

        await Assert.That((double)arr[1]).IsEqualTo(99.0);
    }

    [Test]
    public async Task SetItem_AllElements_WithColon()
    {
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });

        arr.__setitem__(":", 0);

        await Assert.That(np.all(arr == 0)).IsTrue();
    }

    [Test]
    public async Task SetItem_ReversedSlice()
    {
        // NumPy: arr[::-1] = [10, 20, 30, 40, 50] assigns in reverse order
        // arr[4]=10, arr[3]=20, arr[2]=30, arr[1]=40, arr[0]=50
        // Result: [50, 40, 30, 20, 10]
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });

        arr.__setitem__("::-1", np.array(new[] { 10, 20, 30, 40, 50 }));

        await Assert.That((int)arr[0]).IsEqualTo(50); // Last of source
        await Assert.That((int)arr[4]).IsEqualTo(10); // First of source
    }

    #endregion

    #region __hash__ Battle Tests

    [Test]
    public async Task Hash_AllDtypes_Throw()
    {
        await Assert.That(() => np.array(new[] { true, false }).GetHashCode()).Throws<NotSupportedException>();
        await Assert.That(() => np.array(new byte[] { 1, 2 }).GetHashCode()).Throws<NotSupportedException>();
        await Assert.That(() => np.array(new short[] { 1, 2 }).GetHashCode()).Throws<NotSupportedException>();
        await Assert.That(() => np.array(new ushort[] { 1, 2 }).GetHashCode()).Throws<NotSupportedException>();
        await Assert.That(() => np.array(new[] { 1, 2 }).GetHashCode()).Throws<NotSupportedException>();
        await Assert.That(() => np.array(new uint[] { 1, 2 }).GetHashCode()).Throws<NotSupportedException>();
        await Assert.That(() => np.array(new long[] { 1, 2 }).GetHashCode()).Throws<NotSupportedException>();
        await Assert.That(() => np.array(new ulong[] { 1, 2 }).GetHashCode()).Throws<NotSupportedException>();
        await Assert.That(() => np.array(new[] { 'a', 'b' }).GetHashCode()).Throws<NotSupportedException>();
        await Assert.That(() => np.array(new[] { 1f, 2f }).GetHashCode()).Throws<NotSupportedException>();
        await Assert.That(() => np.array(new[] { 1.0, 2.0 }).GetHashCode()).Throws<NotSupportedException>();
        await Assert.That(() => np.array(new[] { 1m, 2m }).GetHashCode()).Throws<NotSupportedException>();
    }

    [Test]
    public async Task Hash_EmptyArray_Throws()
    {
        await Assert.That(() => np.array(Array.Empty<int>()).GetHashCode()).Throws<NotSupportedException>();
    }

    [Test]
    public async Task Hash_ScalarArray_Throws()
    {
        await Assert.That(() => NDArray.Scalar(5).GetHashCode()).Throws<NotSupportedException>();
    }

    [Test]
    public async Task Hash_SlicedArray_Throws()
    {
        var arr = np.arange(10);
        var sliced = arr["2:8"];
        await Assert.That(() => sliced.GetHashCode()).Throws<NotSupportedException>();
    }

    [Test]
    public async Task Hash_BroadcastArray_Throws()
    {
        var arr = np.array(new[] { 1, 2, 3 });
        var broadcast = np.broadcast_to(arr, new Shape(4, 3));
        await Assert.That(() => broadcast.GetHashCode()).Throws<NotSupportedException>();
    }

    [Test]
    public async Task Hash_HashSetUsage_Fails()
    {
        var arr = np.array(new[] { 1, 2, 3 });
        var set = new HashSet<object>();

        await Assert.That(() => set.Add(arr)).Throws<NotSupportedException>();
    }

    [Test]
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
            await Assert.That(ex.Message).Contains("unhashable");
            await Assert.That(ex.Message).Contains("mutable");
        }
    }

    #endregion

    #region __contains__ Battle Tests

    [Test]
    public async Task Contains_AllDtypes()
    {
        await Assert.That(np.array(new[] { true, false }).Contains(true)).IsTrue();
        await Assert.That(np.array(new byte[] { 1, 2, 3 }).Contains((byte)2)).IsTrue();
        await Assert.That(np.array(new short[] { 1, 2, 3 }).Contains((short)2)).IsTrue();
        await Assert.That(np.array(new ushort[] { 1, 2, 3 }).Contains((ushort)2)).IsTrue();
        await Assert.That(np.array(new[] { 1, 2, 3 }).Contains(2)).IsTrue();
        await Assert.That(np.array(new uint[] { 1, 2, 3 }).Contains(2u)).IsTrue();
        await Assert.That(np.array(new long[] { 1, 2, 3 }).Contains(2L)).IsTrue();
        await Assert.That(np.array(new ulong[] { 1, 2, 3 }).Contains(2ul)).IsTrue();
        await Assert.That(np.array(new[] { 'a', 'b', 'c' }).Contains('b')).IsTrue();
        await Assert.That(np.array(new[] { 1f, 2f, 3f }).Contains(2f)).IsTrue();
        await Assert.That(np.array(new[] { 1.0, 2.0, 3.0 }).Contains(2.0)).IsTrue();
        await Assert.That(np.array(new[] { 1m, 2m, 3m }).Contains(2m)).IsTrue();
    }

    [Test]
    public async Task Contains_LargeArray()
    {
        var arr = np.arange(10000);

        await Assert.That(arr.Contains(0)).IsTrue();
        await Assert.That(arr.Contains(5000)).IsTrue();
        await Assert.That(arr.Contains(9999)).IsTrue();
        await Assert.That(arr.Contains(10000)).IsFalse();
    }

    [Test]
    public async Task Contains_Infinity()
    {
        var arr = np.array(new[] { double.NegativeInfinity, 0.0, double.PositiveInfinity });

        await Assert.That(arr.Contains(double.PositiveInfinity)).IsTrue();
        await Assert.That(arr.Contains(double.NegativeInfinity)).IsTrue();
        await Assert.That(arr.Contains(0.0)).IsTrue();
    }

    [Test]
    public async Task Contains_NegativeValues()
    {
        var arr = np.array(new[] { -5, -3, -1, 0, 1, 3, 5 });

        await Assert.That(arr.Contains(-3)).IsTrue();
        await Assert.That(arr.Contains(-10)).IsFalse();
    }

    [Test]
    public async Task Contains_3DArray()
    {
        var arr = np.arange(24).reshape(2, 3, 4);

        await Assert.That(arr.Contains(0)).IsTrue();
        await Assert.That(arr.Contains(23)).IsTrue();
        await Assert.That(arr.Contains(12)).IsTrue();
        await Assert.That(arr.Contains(100)).IsFalse();
    }

    [Test]
    public async Task Contains_SlicedArray()
    {
        var arr = np.arange(10);
        var sliced = arr["3:7"]; // [3, 4, 5, 6]

        await Assert.That(sliced.Contains(5)).IsTrue();
        await Assert.That(sliced.Contains(2)).IsFalse(); // Not in slice
        await Assert.That(sliced.Contains(8)).IsFalse(); // Not in slice
    }

    [Test]
    public async Task Contains_StridedArray()
    {
        var arr = np.arange(10);
        var strided = arr["::2"]; // [0, 2, 4, 6, 8]

        await Assert.That(strided.Contains(4)).IsTrue();
        await Assert.That(strided.Contains(3)).IsFalse(); // Odd numbers not included
    }

    [Test]
    public async Task Contains_BroadcastArray()
    {
        var arr = np.array(new[] { 1, 2, 3 });
        var broadcast = np.broadcast_to(arr, new Shape(4, 3));

        await Assert.That(broadcast.Contains(2)).IsTrue();
        await Assert.That(broadcast.Contains(5)).IsFalse();
    }

    [Test]
    public async Task Contains_ScalarArray()
    {
        var scalar = NDArray.Scalar(42);

        await Assert.That(scalar.Contains(42)).IsTrue();
        await Assert.That(scalar.Contains(0)).IsFalse();
    }

    #endregion

    #region __iter__ Battle Tests

    [Test]
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

        await Assert.That(true).IsTrue(); // Passed if no exception
    }

    [Test]
    public async Task Iter_1DArray_ElementsMatch()
    {
        var arr = np.array(new[] { 10, 20, 30 });
        var collected = new List<int>();

        // 1D arrays iterate over elements (int values), not NDArray
        foreach (var item in arr)
        {
            collected.Add(Convert.ToInt32(item));
        }

        await Assert.That(collected).IsEquivalentTo(new[] { 10, 20, 30 });
    }

    [Test]
    public async Task Iter_2DArray_IteratesRows()
    {
        var arr = np.array(new[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
        var rowCount = 0;

        // 2D arrays iterate over rows (NDArray slices)
        foreach (var item in arr)
        {
            if (item is NDArray row)
            {
                await Assert.That(row.ndim).IsEqualTo(1);
                await Assert.That(row.size).IsEqualTo(2);
            }
            rowCount++;
        }

        await Assert.That(rowCount).IsEqualTo(3);
    }

    [Test]
    public async Task Iter_3DArray_Iterates2DSlices()
    {
        var arr = np.zeros(new[] { 2, 3, 4 });
        var sliceCount = 0;

        // 3D arrays iterate over 2D slices
        foreach (var item in arr)
        {
            if (item is NDArray slice)
            {
                await Assert.That(slice.ndim).IsEqualTo(2);
                await Assert.That(slice.shape).IsEquivalentTo(new long[] { 3, 4 });
            }
            sliceCount++;
        }

        await Assert.That(sliceCount).IsEqualTo(2);
    }

    [Test]
    public async Task Iter_EmptyArray_NoIterations()
    {
        var arr = np.array(Array.Empty<int>());
        var count = 0;

        foreach (var _ in arr)
        {
            count++;
        }

        await Assert.That(count).IsEqualTo(0);
    }

    [Test]
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

        await Assert.That(count).IsEqualTo(1);
        await Assert.That(value).IsEqualTo(42);
    }

    [Test]
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

        await Assert.That(first).IsEquivalentTo(second);
    }

    [Test]
    public async Task Iter_LINQ_ToList()
    {
        var arr = np.array(new[] { 1, 2, 3 });

        var list = arr.Cast<object>().ToList();

        await Assert.That(list.Count).IsEqualTo(3);
    }

    [Test]
    public async Task Iter_LINQ_Count()
    {
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });

        var count = arr.Cast<object>().Count();

        await Assert.That(count).IsEqualTo(5);
    }

    [Test]
    public async Task Iter_SlicedArray()
    {
        var arr = np.arange(10);
        var sliced = arr["2:7"];
        var count = 0;

        foreach (var _ in sliced)
            count++;

        await Assert.That(count).IsEqualTo(5);
    }

    [Test]
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

        await Assert.That(count).IsEqualTo(5);
    }

    [Test]
    public async Task Iter_ScalarArray_ThrowsTypeError()
    {
        // NumPy: iteration over a 0-d array throws TypeError
        var scalar = NDArray.Scalar(42);

        await Assert.That(() =>
        {
            foreach (var _ in scalar)
            {
                // Should not reach here
            }
        }).Throws<TypeError>();
    }

    #endregion
}
