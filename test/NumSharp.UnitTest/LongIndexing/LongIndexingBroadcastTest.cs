using System;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.LongIndexing;

/// <summary>
/// Long indexing tests using broadcast arrays to avoid massive memory allocations.
///
/// By broadcasting a scalar (single element) to a large shape (> int.MaxValue),
/// we can test all long indexing code paths without allocating 2+ GB of memory.
/// The broadcast creates a view with stride=0, so only 1 element is stored.
///
/// Memory requirements: ~8 bytes per array (just the scalar value)
/// Test coverage: Same long indexing code paths as the full master test
///
/// LIMITATIONS DISCOVERED:
/// - SliceDef is limited to int indices (cannot slice broadcast arrays with long indices)
/// - Operations that produce output (add, multiply, etc.) allocate full-size output arrays
///   even when input is broadcast, causing OutOfMemoryException
///
/// NOTE: Marked [LargeMemoryTest] because iterating over 2.36 billion elements causes
/// excessive CPU/memory pressure when MSTest runs tests in parallel.
/// </summary>
[TestClass]
public class LongIndexingBroadcastTest
{
    /// <summary>
    /// Target size for long indexing tests: 10% larger than int.MaxValue.
    /// </summary>
    private const long LargeSize = (long)(int.MaxValue * 1.1); // ~2.36 billion

    /// <summary>
    /// Creates a broadcast array of the given value with LargeSize elements.
    /// Only allocates memory for 1 element, but has LargeSize logical elements.
    /// </summary>
    private static NDArray BroadcastScalar(byte value)
    {
        var scalar = np.full(new Shape(1L), value, np.uint8);
        return np.broadcast_to(scalar, new Shape(LargeSize));
    }

    /// <summary>
    /// Creates a broadcast array of the given value with LargeSize elements.
    /// </summary>
    private static NDArray BroadcastScalar(double value)
    {
        var scalar = np.full(new Shape(1L), value, np.float64);
        return np.broadcast_to(scalar, new Shape(LargeSize));
    }

    // ================================================================
    // SHAPE AND INDEXING (all pass - no memory allocation needed)
    // ================================================================

    [TestMethod, LargeMemoryTest]
    public async Task Broadcast_HasCorrectSize()
    {
        var arr = BroadcastScalar((byte)42);
        Assert.AreEqual(LargeSize, arr.size);
        Assert.AreEqual(1, arr.ndim);
        Assert.AreEqual(LargeSize, arr.shape[0]);
    }

    [TestMethod, LargeMemoryTest]
    public async Task Broadcast_IsBroadcasted()
    {
        var arr = BroadcastScalar((byte)42);
        Assert.IsTrue(arr.Shape.IsBroadcasted);
        Assert.IsFalse(arr.Shape.IsContiguous);
    }

    [TestMethod, LargeMemoryTest]
    public async Task Broadcast_GetAtLargeIndex()
    {
        var arr = BroadcastScalar((byte)77);

        // All positions should return the same broadcast value
        Assert.AreEqual(77, arr.GetByte(0));
        Assert.AreEqual(77, arr.GetByte(LargeSize / 2));
        Assert.AreEqual(77, arr.GetByte(LargeSize - 1));
    }

    // ================================================================
    // REDUCTIONS (these iterate over all elements logically - PASS)
    // ================================================================

    [TestMethod, LargeMemoryTest]
    public async Task Broadcast_Mean()
    {
        var arr = BroadcastScalar((byte)42);
        var result = np.mean(arr);

        // Mean of all 42s = 42.0
        Assert.AreEqual(0, result.ndim);
        Assert.AreEqual(42.0, result.GetDouble(0), 0.001);
    }

    [TestMethod, LargeMemoryTest]
    public async Task Broadcast_Min()
    {
        var arr = BroadcastScalar((byte)7);
        var result = np.min(arr);

        Assert.AreEqual(7, result.GetByte(0));
    }

    [TestMethod, LargeMemoryTest]
    public async Task Broadcast_Max()
    {
        var arr = BroadcastScalar((byte)200);
        var result = np.max(arr);

        Assert.AreEqual(200, result.GetByte(0));
    }

    [TestMethod, LargeMemoryTest]
    public async Task Broadcast_ArgMax()
    {
        var arr = BroadcastScalar((byte)100);
        var result = np.argmax(arr);

        // All elements are the same, argmax returns first (0)
        Assert.AreEqual(0L, result);
    }

    [TestMethod, LargeMemoryTest]
    public async Task Broadcast_ArgMin()
    {
        var arr = BroadcastScalar((byte)50);
        var result = np.argmin(arr);

        // All elements are the same, argmin returns first (0)
        Assert.AreEqual(0L, result);
    }

    [TestMethod, LargeMemoryTest]
    public async Task Broadcast_All()
    {
        var arr = BroadcastScalar((byte)1);
        var result = np.all(arr);

        Assert.IsTrue(result);
    }

    [TestMethod, LargeMemoryTest]
    public async Task Broadcast_All_Zero()
    {
        var arr = BroadcastScalar((byte)0);
        var result = np.all(arr);

        Assert.IsFalse(result);
    }

    [TestMethod, LargeMemoryTest]
    public async Task Broadcast_Any()
    {
        var arr = BroadcastScalar((byte)1);
        var result = np.any(arr);

        Assert.IsTrue(result);
    }

    [TestMethod, LargeMemoryTest]
    public async Task Broadcast_Any_Zero()
    {
        var arr = BroadcastScalar((byte)0);
        var result = np.any(arr);

        Assert.IsFalse(result);
    }

    [TestMethod, LargeMemoryTest]
    public async Task Broadcast_CountNonzero()
    {
        var arr = BroadcastScalar((byte)1);
        var result = np.count_nonzero(arr);

        // All LargeSize elements are nonzero
        Assert.AreEqual(LargeSize, result);
    }

    [TestMethod, LargeMemoryTest]
    public async Task Broadcast_CountNonzero_Zero()
    {
        var arr = BroadcastScalar((byte)0);
        var result = np.count_nonzero(arr);

        Assert.AreEqual(0L, result);
    }

    // ================================================================
    // SHAPE MANIPULATION (most pass - only reshape with slice fails)
    // ================================================================

    [TestMethod, LargeMemoryTest]
    public async Task Broadcast_ExpandDims()
    {
        var arr = BroadcastScalar((byte)11);
        var result = np.expand_dims(arr, 0);

        Assert.AreEqual(LargeSize, result.size);
        Assert.AreEqual(2, result.ndim);
        Assert.AreEqual(1, result.shape[0]);
        Assert.AreEqual(LargeSize, result.shape[1]);
    }

    [TestMethod, LargeMemoryTest]
    public async Task Broadcast_Atleast2d()
    {
        var arr = BroadcastScalar((byte)22);
        var result = np.atleast_2d(arr);

        Assert.AreEqual(LargeSize, result.size);
        Assert.AreEqual(2, result.ndim);
    }

    [TestMethod, LargeMemoryTest]
    public async Task Broadcast_Atleast3d()
    {
        var arr = BroadcastScalar((byte)22);
        var result = np.atleast_3d(arr);

        Assert.AreEqual(LargeSize, result.size);
        Assert.AreEqual(3, result.ndim);
    }

    // ================================================================
    // 2D BROADCAST (tests multi-dimensional long indexing)
    // ================================================================

    [TestMethod, LargeMemoryTest]
    public async Task Broadcast2D_RowBroadcast()
    {
        // Broadcast a single row to many rows
        long rows = LargeSize;
        long cols = 10;

        var row = np.full(new Shape(1L, cols), (byte)77, np.uint8);
        var arr = np.broadcast_to(row, new Shape(rows, cols));

        Assert.AreEqual(rows * cols, arr.size);
        Assert.AreEqual(77, arr.GetByte(0, 0));
        Assert.AreEqual(77, arr.GetByte(rows - 1, cols - 1));
    }

    [TestMethod, LargeMemoryTest]
    public async Task Broadcast2D_ColumnBroadcast()
    {
        // Broadcast a single column to many columns
        long rows = 10;
        long cols = LargeSize;

        var col = np.full(new Shape(rows, 1L), (byte)88, np.uint8);
        var arr = np.broadcast_to(col, new Shape(rows, cols));

        Assert.AreEqual(rows * cols, arr.size);
        Assert.AreEqual(88, arr.GetByte(0, 0));
        Assert.AreEqual(88, arr.GetByte(rows - 1, cols - 1));
    }

    [TestMethod, LargeMemoryTest]
    public async Task Broadcast2D_Transpose()
    {
        // Test transpose of a broadcast array
        long rows = 10;
        long cols = LargeSize / 10;

        var row = np.full(new Shape(1L, cols), (byte)55, np.uint8);
        var arr = np.broadcast_to(row, new Shape(rows, cols));
        var transposed = np.transpose(arr);

        Assert.AreEqual(rows * cols, transposed.size);
        Assert.AreEqual(cols, transposed.shape[0]);
        Assert.AreEqual(rows, transposed.shape[1]);
        Assert.AreEqual(55, transposed.GetByte(0, 0));
    }

    // ================================================================
    // KNOWN LIMITATIONS - Tests below document current limitations
    // These are marked as OpenBugs to exclude from CI
    // ================================================================

    /// <summary>
    /// LIMITATION: SliceDef is limited to int indices.
    /// Slicing a broadcast array with indices > int.MaxValue throws OverflowException.
    /// </summary>
    [TestMethod]
    public async Task Broadcast_SliceWithLargeIndices_Limited()
    {
        var arr = BroadcastScalar((byte)99);

        // This throws: "Dimension X exceeds int.MaxValue. SliceDef indices limited to int range."
        long start = LargeSize - 100;
        long stop = LargeSize;
        var slice = arr[$"{start}:{stop}"];

        Assert.AreEqual(100, slice.size);
    }

    /// <summary>
    /// LIMITATION: Binary operations allocate full output arrays.
    /// Even with broadcast inputs, the output is allocated at full size.
    /// </summary>
    [TestMethod]
    public async Task Broadcast_Add_AllocatesFullOutput()
    {
        var a = BroadcastScalar((byte)10);
        var b = BroadcastScalar((byte)20);

        // This throws OutOfMemoryException because it allocates LargeSize output
        var result = np.add(a, b);
        Assert.AreEqual(30, result.GetByte(0));
    }

    /// <summary>
    /// LIMITATION: Unary operations allocate full output arrays.
    /// </summary>
    [TestMethod]
    public async Task Broadcast_Square_AllocatesFullOutput()
    {
        var arr = BroadcastScalar((byte)5);

        // This throws OutOfMemoryException
        var result = np.square(arr);
        Assert.AreEqual(25, result.GetByte(0));
    }

    /// <summary>
    /// LIMITATION: Sum on broadcast uses inefficient iterator.
    /// Currently throws "index < Count, Memory corruption expected" for large broadcast.
    /// </summary>
    [TestMethod]
    [OpenBugs]
    public async Task Broadcast_Sum_InternalError()
    {
        var arr = BroadcastScalar((byte)1);
        var result = np.sum(arr);

        // Sum of LargeSize ones should be LargeSize
        Assert.AreEqual(LargeSize, result.GetInt64(0));
    }

    // ================================================================
    // COPY (materializes the broadcast - tests large allocation path)
    // Note: This DOES allocate full memory, so it's marked Explicit + LongIndexing
    // ================================================================

    [TestMethod]
    [TestCategory("Explicit")] // Allocates full 2.4GB array
    [HighMemory]
    public async Task Broadcast_Copy_MaterializesFullArray()
    {
        var broadcast = BroadcastScalar((byte)123);

        // np.copy materializes the broadcast to a contiguous array
        var materialized = np.copy(broadcast);

        Assert.AreEqual(LargeSize, materialized.size);
        Assert.IsTrue(materialized.Shape.IsContiguous);
        Assert.IsFalse(materialized.Shape.IsBroadcasted);
        Assert.AreEqual(123, materialized.GetByte(0));
        Assert.AreEqual(123, materialized.GetByte(LargeSize - 1));
    }
}
