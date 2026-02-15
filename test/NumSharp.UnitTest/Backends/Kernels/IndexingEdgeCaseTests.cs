using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.UnitTest.Utilities;
using TUnit.Core;

namespace NumSharp.UnitTest.Backends.Kernels;

/// <summary>
/// Edge case tests for array indexing operations.
/// All expected values are verified against NumPy 2.x output.
/// </summary>
public class IndexingEdgeCaseTests
{
    #region Negative Indexing

    [Test]
    public void NegativeIndex_Last()
    {
        // NumPy: arr[-1] = 5
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });

        Assert.AreEqual(5, arr[-1].GetInt32(0));
    }

    [Test]
    public void NegativeIndex_First()
    {
        // NumPy: arr[-5] = 1 (same as arr[0] for 5-element array)
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });

        Assert.AreEqual(1, arr[-5].GetInt32(0));
    }

    [Test]
    public void NegativeSlice_LastN()
    {
        // NumPy: arr[-3:] = [3, 4, 5]
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });

        var result = arr["-3:"];

        result.Should().BeOfValues(3, 4, 5);
    }

    [Test]
    public void NegativeSlice_ExcludeLastN()
    {
        // NumPy: arr[:-3] = [1, 2]
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });

        var result = arr[":-3"];

        result.Should().BeOfValues(1, 2);
    }

    [Test]
    public void NegativeSlice_Range()
    {
        // NumPy: arr[-4:-1] = [2, 3, 4]
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });

        var result = arr["-4:-1"];

        result.Should().BeOfValues(2, 3, 4);
    }

    [Test]
    public void NegativeSlice_ReversePartial()
    {
        // NumPy: arr[-1:-4:-1] = [5, 4, 3]
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });

        var result = arr["-1:-4:-1"];

        result.Should().BeOfValues(5, 4, 3);
    }

    [Test]
    public void NegativeSlice_2D_Corner()
    {
        // NumPy: arr2d[-2:, -2:] = [[6,7], [10,11]]
        var arr = np.arange(12).reshape(3, 4);

        var result = arr["-2:, -2:"];

        Assert.AreEqual(2, result.shape[0]);
        Assert.AreEqual(2, result.shape[1]);
        Assert.AreEqual(6, result.GetInt32(0, 0));
        Assert.AreEqual(11, result.GetInt32(1, 1));
    }

    [Test]
    public void NegativeSlice_2D_FullReverse()
    {
        // NumPy: arr2d[::-1, ::-1] reverses both axes
        var arr = np.arange(12).reshape(3, 4);

        var result = arr["::-1, ::-1"];

        Assert.AreEqual(11, result.GetInt32(0, 0));
        Assert.AreEqual(0, result.GetInt32(2, 3));
    }

    #endregion

    #region Boolean Indexing

    [Test]
    [OpenBugs]  // Boolean indexing returns incorrect results
    public void BooleanIndex_Simple()
    {
        // NumPy: arr[mask] = [1, 3, 5]
        var arr = np.array(new[] { 1, 2, 3, 4, 5, 6 });
        var mask = np.array(new[] { true, false, true, false, true, false });

        var result = arr[mask];

        result.Should().BeOfValues(1, 3, 5);
    }

    [Test]
    public void BooleanIndex_Condition()
    {
        // NumPy: arr[arr > 3] = [4, 5, 6]
        var arr = np.array(new[] { 1, 2, 3, 4, 5, 6 });

        var result = arr[arr > 3];

        result.Should().BeOfValues(4, 5, 6);
    }

    [Test]
    public void BooleanIndex_EvenNumbers()
    {
        // NumPy: arr[arr % 2 == 0] = [2, 4, 6]
        var arr = np.array(new[] { 1, 2, 3, 4, 5, 6 });

        var result = arr[arr % 2 == 0];

        result.Should().BeOfValues(2, 4, 6);
    }

    [Test]
    public void BooleanIndex_2D_Flattens()
    {
        // NumPy: arr2d[arr2d > 5] = [6, 7, 8, 9, 10, 11] (flattened!)
        var arr = np.arange(12).reshape(3, 4);

        var result = arr[arr > 5];

        Assert.AreEqual(1, result.ndim);  // Flattened to 1D
        Assert.AreEqual(6, result.size);
        Assert.AreEqual(6, result.GetInt32(0));
    }

    [Test]
    [OpenBugs]  // Boolean row selection fails
    public void BooleanIndex_RowSelection()
    {
        // NumPy: arr2d[[True, False, True]] selects rows 0 and 2
        var arr = np.arange(12).reshape(3, 4);
        var rowMask = np.array(new[] { true, false, true });

        var result = arr[rowMask];

        Assert.AreEqual(2, result.shape[0]);
        Assert.AreEqual(4, result.shape[1]);
        Assert.AreEqual(0, result.GetInt32(0, 0));
        Assert.AreEqual(8, result.GetInt32(1, 0));
    }

    #endregion

    #region Fancy Indexing

    [Test]
    public void FancyIndex_Simple()
    {
        // NumPy: arr[[0, 2, 4]] = [10, 30, 50]
        var arr = np.array(new[] { 10, 20, 30, 40, 50 });
        var indices = np.array(new[] { 0, 2, 4 });

        var result = arr[indices];

        result.Should().BeOfValues(10, 30, 50);
    }

    [Test]
    public void FancyIndex_NegativeIndices()
    {
        // NumPy: arr[[-1, -3, -5]] = [50, 30, 10]
        var arr = np.array(new[] { 10, 20, 30, 40, 50 });
        var indices = np.array(new[] { -1, -3, -5 });

        var result = arr[indices];

        result.Should().BeOfValues(50, 30, 10);
    }

    [Test]
    public void FancyIndex_Repeated()
    {
        // NumPy: arr[[0, 0, 1, 1, 2]] = [10, 10, 20, 20, 30]
        var arr = np.array(new[] { 10, 20, 30, 40, 50 });
        var indices = np.array(new[] { 0, 0, 1, 1, 2 });

        var result = arr[indices];

        result.Should().BeOfValues(10, 10, 20, 20, 30);
    }

    [Test]
    public void FancyIndex_2D_Diagonal()
    {
        // NumPy: arr2d[[0,1,2], [0,1,2]] = [0, 5, 10] (diagonal)
        var arr = np.arange(12).reshape(3, 4);
        var rowIdx = np.array(new[] { 0, 1, 2 });
        var colIdx = np.array(new[] { 0, 1, 2 });

        var result = arr[rowIdx, colIdx];

        result.Should().BeOfValues(0, 5, 10);
    }

    #endregion

    #region Ellipsis

    [Test]
    public void Ellipsis_3D_Last()
    {
        // NumPy: arr3d[..., 0].shape = (2, 3)
        var arr = np.arange(24).reshape(2, 3, 4);

        var result = arr["..., 0"];

        Assert.AreEqual(2, result.ndim);
        Assert.AreEqual(2, result.shape[0]);
        Assert.AreEqual(3, result.shape[1]);
    }

    [Test]
    public void Ellipsis_3D_First()
    {
        // NumPy: arr3d[0, ...].shape = (3, 4)
        var arr = np.arange(24).reshape(2, 3, 4);

        var result = arr["0, ..."];

        Assert.AreEqual(2, result.ndim);
        Assert.AreEqual(3, result.shape[0]);
        Assert.AreEqual(4, result.shape[1]);
    }

    #endregion

    #region Combined Indexing

    [Test]
    public void Combined_SliceAndInteger()
    {
        // NumPy: arr2d[1:, 0] = [4, 8]
        var arr = np.arange(12).reshape(3, 4);

        var result = arr["1:, 0"];

        Assert.AreEqual(1, result.ndim);
        result.Should().BeOfValues(4, 8);
    }

    [Test]
    public void Combined_IntegerAndSlice()
    {
        // NumPy: arr2d[0, 1:3] = [1, 2]
        var arr = np.arange(12).reshape(3, 4);

        var result = arr["0, 1:3"];

        Assert.AreEqual(1, result.ndim);
        result.Should().BeOfValues(1, 2);
    }

    #endregion

    #region Indexing Assignment

    [Test]
    public void Assignment_SingleElement()
    {
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });

        arr[2] = 100;

        arr.Should().BeOfValues(1, 2, 100, 4, 5);
    }

    [Test]
    public void Assignment_Slice()
    {
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });

        arr["1:4"] = np.array(new[] { 10, 20, 30 });

        arr.Should().BeOfValues(1, 10, 20, 30, 5);
    }

    [Test]
    public void Assignment_SliceWithScalar()
    {
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });

        arr["1:4"] = 99;

        arr.Should().BeOfValues(1, 99, 99, 99, 5);
    }

    [Test]
    [OpenBugs]  // Fancy index assignment silently no-ops
    public void Assignment_FancyIndex()
    {
        // NumPy: arr[[0, 2, 4]] = 99 -> [99, 2, 99, 4, 99]
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });
        var indices = np.array(new[] { 0, 2, 4 });

        arr[indices] = 99;

        arr.Should().BeOfValues(99, 2, 99, 4, 99);
    }

    [Test]
    [OpenBugs]  // Boolean mask assignment silently no-ops
    public void Assignment_BooleanMask()
    {
        // NumPy: arr[arr > 3] = 0 -> [1, 2, 3, 0, 0]
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });

        arr[arr > 3] = 0;

        arr.Should().BeOfValues(1, 2, 3, 0, 0);
    }

    #endregion

    #region Edge Cases

    [Test]
    public void EmptySlice()
    {
        // NumPy: arr[2:2] = [] (empty)
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });

        var result = arr["2:2"];

        Assert.AreEqual(0, result.size);
    }

    [Test]
    public void StepGreaterThanSize()
    {
        // NumPy: arr[::10] = [1] (only first element)
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });

        var result = arr["::10"];

        Assert.AreEqual(1, result.size);
        Assert.AreEqual(1, result.GetInt32(0));
    }

    [Test]
    public void ReverseWithStep()
    {
        // NumPy: arr[4:0:-2] = [5, 3]
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });

        var result = arr["4:0:-2"];

        result.Should().BeOfValues(5, 3);
    }

    #endregion
}
