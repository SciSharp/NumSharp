using System;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using TUnit.Core;

namespace NumSharp.UnitTest.LongIndexing;

/// <summary>
/// Smoke tests for long indexing API compatibility.
///
/// These tests use smaller arrays (1 million elements) to verify that
/// the long-indexed APIs work correctly without requiring massive memory.
/// They complement the full LongIndexingMasterTest which requires 8GB+ RAM.
///
/// Purpose:
/// - Runs in CI to catch API regressions
/// - Verifies Shape, NDArray, and np.* APIs accept long indices
/// - Fast execution (~1-2 seconds total)
/// </summary>
public class LongIndexingSmokeTest
{
    /// <summary>
    /// Test size for smoke tests - large enough to be meaningful but small enough for CI.
    /// </summary>
    private const long TestSize = 1_000_000L;

    [Test]
    public async Task Shape_AcceptsLongDimensions()
    {
        var shape = new Shape(TestSize);
        Assert.AreEqual(TestSize, shape.Size);
        Assert.AreEqual(1, shape.NDim);
        Assert.AreEqual(TestSize, shape.Dimensions[0]);
    }

    [Test]
    public async Task Shape_AcceptsLongMultiDimensions()
    {
        var shape = new Shape(1000L, 1000L);
        Assert.AreEqual(1_000_000L, shape.Size);
        Assert.AreEqual(2, shape.NDim);
    }

    [Test]
    public async Task Shape_GetOffset_AcceptsLongIndices()
    {
        var shape = new Shape(TestSize);
        var offset = shape.GetOffset(TestSize - 1);
        Assert.AreEqual(TestSize - 1, offset);
    }

    [Test]
    public async Task Shape_GetCoordinates_ReturnsLongArray()
    {
        var shape = new Shape(1000L, 1000L);
        long[] coords = shape.GetCoordinates(999_999);
        Assert.AreEqual(2, coords.Length);
        Assert.AreEqual(999L, coords[0]);
        Assert.AreEqual(999L, coords[1]);
    }

    [Test]
    public async Task NDArray_Zeros_AcceptsLongShape()
    {
        var arr = np.zeros(new Shape(TestSize), np.uint8);
        Assert.AreEqual(TestSize, arr.size);
        Assert.AreEqual(0, arr.GetByte(0));
        Assert.AreEqual(0, arr.GetByte(TestSize - 1));
    }

    [Test]
    public async Task NDArray_Ones_AcceptsLongShape()
    {
        var arr = np.ones(new Shape(TestSize), np.uint8);
        Assert.AreEqual(TestSize, arr.size);
        Assert.AreEqual(1, arr.GetByte(0));
        Assert.AreEqual(1, arr.GetByte(TestSize - 1));
    }

    [Test]
    public async Task NDArray_Full_AcceptsLongShape()
    {
        var arr = np.full(new Shape(TestSize), (byte)42, np.uint8);
        Assert.AreEqual(TestSize, arr.size);
        Assert.AreEqual(42, arr.GetByte(0));
        Assert.AreEqual(42, arr.GetByte(TestSize - 1));
    }

    [Test]
    public async Task NDArray_GetSet_AcceptsLongIndex()
    {
        var arr = np.zeros(new Shape(TestSize), np.uint8);
        arr.SetByte((byte)123, TestSize - 1);
        Assert.AreEqual(123, arr.GetByte(TestSize - 1));
    }

    [Test]
    public async Task NDArray_Reshape_AcceptsLongDimensions()
    {
        var arr = np.zeros(new Shape(TestSize), np.uint8);
        var reshaped = np.reshape(arr, new Shape(1000L, 1000L));
        Assert.AreEqual(TestSize, reshaped.size);
        Assert.AreEqual(2, reshaped.ndim);
    }

    [Test]
    public async Task NDArray_Sum_ReturnsCorrectForLargeArray()
    {
        var arr = np.ones(new Shape(TestSize), np.uint8);
        var result = np.sum(arr);
        Assert.AreEqual(0, result.ndim);
    }

    [Test]
    public async Task NDArray_Mean_ReturnsCorrectForLargeArray()
    {
        var arr = np.ones(new Shape(TestSize), np.uint8);
        var result = np.mean(arr);
        Assert.AreEqual(1.0, result.GetDouble(0), 0.001);
    }

    [Test]
    public async Task NDArray_ArgMax_ReturnsLongIndex()
    {
        var arr = np.zeros(new Shape(TestSize), np.uint8);
        arr.SetByte((byte)255, TestSize - 100);
        var result = np.argmax(arr);
        Assert.AreEqual(TestSize - 100, result);
    }

    [Test]
    public async Task NDArray_ArgMin_ReturnsLongIndex()
    {
        var arr = np.ones(new Shape(TestSize), np.uint8);
        arr.SetByte((byte)0, TestSize / 2);
        var result = np.argmin(arr);
        Assert.AreEqual(TestSize / 2, result);
    }

    [Test]
    public async Task NDArray_CountNonzero_ReturnsLongCount()
    {
        var arr = np.ones(new Shape(TestSize), np.uint8);
        var count = np.count_nonzero(arr);
        Assert.AreEqual(TestSize, count);
    }

    [Test]
    public async Task NDArray_Slicing_WorksWithLargeIndices()
    {
        var arr = np.full(new Shape(TestSize), (byte)99, np.uint8);
        long start = TestSize - 100;
        long stop = TestSize;
        var slice = arr[$"{start}:{stop}"];
        Assert.AreEqual(100, slice.size);
        Assert.AreEqual(99, slice.GetByte(0));
    }

    [Test]
    public async Task NDArray_BroadcastTo_WorksWithLongTarget()
    {
        var scalar = np.full(new Shape(1L), (byte)77, np.uint8);
        var broadcast = np.broadcast_to(scalar, new Shape(TestSize));
        Assert.AreEqual(TestSize, broadcast.size);
        Assert.AreEqual(77, broadcast.GetByte(0));
        Assert.AreEqual(77, broadcast.GetByte(TestSize - 1));
    }

    [Test]
    public async Task NDArray_Add_WorksWithLargeArrays()
    {
        var a = np.ones(new Shape(TestSize), np.uint8);
        var b = np.ones(new Shape(TestSize), np.uint8);
        var result = np.add(a, b);
        Assert.AreEqual(TestSize, result.size);
        Assert.AreEqual(2, result.GetByte(0));
        Assert.AreEqual(2, result.GetByte(TestSize - 1));
    }

    [Test]
    public async Task NDArray_Cumsum_WorksWithLargeArray()
    {
        var arr = np.ones(new Shape(TestSize), np.uint8);
        var result = np.cumsum(arr);
        Assert.AreEqual(TestSize, result.size);
    }

    [Test]
    public async Task NDArray_Roll_WorksWithLargeShift()
    {
        var arr = np.zeros(new Shape(TestSize), np.uint8);
        arr.SetByte((byte)1, 0);
        var result = np.roll(arr, TestSize / 2);
        Assert.AreEqual(TestSize, result.size);
        Assert.AreEqual(1, result.GetByte(TestSize / 2));
    }

    [Test]
    public async Task NDArray_Clip_WorksWithLargeArray()
    {
        var arr = np.full(new Shape(TestSize), (byte)100, np.uint8);
        var result = np.clip(arr, 50, 75);
        Assert.AreEqual(TestSize, result.size);
        Assert.AreEqual(75, result.GetByte(0));
    }

    [Test]
    public async Task NDArray_BitwiseOps_WorkWithLargeArrays()
    {
        var a = np.full(new Shape(TestSize), (byte)0b11110000, np.uint8);
        var b = np.full(new Shape(TestSize), (byte)0b00001111, np.uint8);

        // Use operators for bitwise operations
        var andResult = a & b;
        Assert.AreEqual(0, andResult.GetByte(0));

        var orResult = a | b;
        Assert.AreEqual(255, orResult.GetByte(0));

        var xorResult = a.TensorEngine.BitwiseXor(a, b);
        Assert.AreEqual(255, xorResult.GetByte(0));
    }

    [Test]
    public async Task NDArray_ExpandDims_WorksWithLargeArray()
    {
        var arr = np.zeros(new Shape(TestSize), np.uint8);
        var result = np.expand_dims(arr, 0);
        Assert.AreEqual(TestSize, result.size);
        Assert.AreEqual(2, result.ndim);
    }

    [Test]
    public async Task NDArray_Ravel_WorksWithLargeArray()
    {
        var arr = np.zeros(new Shape(1000L, 1000L), np.uint8);
        var result = np.ravel(arr);
        Assert.AreEqual(TestSize, result.size);
        Assert.AreEqual(1, result.ndim);
    }

    [Test]
    public async Task NDArray_Flatten_WorksWithLargeArray()
    {
        var arr = np.zeros(new Shape(1000L, 1000L), np.uint8);
        var result = arr.flatten();
        Assert.AreEqual(TestSize, result.size);
        Assert.AreEqual(1, result.ndim);
    }

    [Test]
    public async Task NDArray_All_WorksWithLargeArray()
    {
        var arr = np.ones(new Shape(TestSize), np.uint8);
        var result = np.all(arr);
        Assert.IsTrue(result);
    }

    [Test]
    public async Task NDArray_Any_WorksWithLargeArray()
    {
        var arr = np.zeros(new Shape(TestSize), np.uint8);
        arr.SetByte((byte)1, TestSize - 1);
        var result = np.any(arr);
        Assert.IsTrue(result);
    }

    [Test]
    public async Task NDArray_Min_WorksWithLargeArray()
    {
        var arr = np.ones(new Shape(TestSize), np.uint8);
        var result = np.min(arr);
        Assert.AreEqual(1, result.GetByte(0));
    }

    [Test]
    public async Task NDArray_Max_WorksWithLargeArray()
    {
        var arr = np.ones(new Shape(TestSize), np.uint8);
        var result = np.max(arr);
        Assert.AreEqual(1, result.GetByte(0));
    }

    [Test]
    public async Task NDArray_Maximum_WorksWithLargeArrays()
    {
        var a = np.full(new Shape(TestSize), (byte)5, np.uint8);
        var b = np.full(new Shape(TestSize), (byte)10, np.uint8);
        var result = np.maximum(a, b);
        Assert.AreEqual(10, result.GetByte(0));
    }

    [Test]
    public async Task NDArray_Minimum_WorksWithLargeArrays()
    {
        var a = np.full(new Shape(TestSize), (byte)5, np.uint8);
        var b = np.full(new Shape(TestSize), (byte)10, np.uint8);
        var result = np.minimum(a, b);
        Assert.AreEqual(5, result.GetByte(0));
    }

    [Test]
    public async Task NDArray_LeftShift_WorksWithLargeArray()
    {
        var arr = np.full(new Shape(TestSize), (byte)1, np.uint8);
        var result = np.left_shift(arr, 4);
        Assert.AreEqual(16, result.GetByte(0));
    }

    [Test]
    public async Task NDArray_RightShift_WorksWithLargeArray()
    {
        var arr = np.full(new Shape(TestSize), (byte)16, np.uint8);
        var result = np.right_shift(arr, 4);
        Assert.AreEqual(1, result.GetByte(0));
    }

    [Test]
    public async Task NDArray_Invert_WorksWithLargeArray()
    {
        var arr = np.full(new Shape(TestSize), (byte)0b11110000, np.uint8);
        var result = np.invert(arr);
        Assert.AreEqual(0b00001111, result.GetByte(0));
    }

    [Test]
    public async Task NDArray_Square_WorksWithLargeArray()
    {
        var arr = np.full(new Shape(TestSize), (byte)3, np.uint8);
        var result = np.square(arr);
        Assert.AreEqual(9, result.GetByte(0));
    }

    [Test]
    public async Task NDArray_Multiply_WorksWithLargeArrays()
    {
        var a = np.full(new Shape(TestSize), (byte)3, np.uint8);
        var b = np.full(new Shape(TestSize), (byte)4, np.uint8);
        var result = np.multiply(a, b);
        Assert.AreEqual(12, result.GetByte(0));
    }

    [Test]
    public async Task NDArray_Subtract_WorksWithLargeArrays()
    {
        var a = np.full(new Shape(TestSize), (byte)10, np.uint8);
        var b = np.full(new Shape(TestSize), (byte)3, np.uint8);
        var result = np.subtract(a, b);
        Assert.AreEqual(7, result.GetByte(0));
    }
}
