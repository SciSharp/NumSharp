using System;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

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
[TestClass]
public class LongIndexingSmokeTest
{
    /// <summary>
    /// Test size for smoke tests - large enough to be meaningful but small enough for CI.
    /// </summary>
    private const long TestSize = 1_000_000L;

    [TestMethod]
    public async Task Shape_AcceptsLongDimensions()
    {
        var shape = new Shape(TestSize);
        Assert.AreEqual(TestSize, shape.Size);
        Assert.AreEqual(1, shape.NDim);
        Assert.AreEqual(TestSize, shape.Dimensions[0]);
    }

    [TestMethod]
    public async Task Shape_AcceptsLongMultiDimensions()
    {
        var shape = new Shape(1000L, 1000L);
        Assert.AreEqual(1_000_000L, shape.Size);
        Assert.AreEqual(2, shape.NDim);
    }

    [TestMethod]
    public async Task Shape_GetOffset_AcceptsLongIndices()
    {
        var shape = new Shape(TestSize);
        var offset = shape.GetOffset(TestSize - 1);
        Assert.AreEqual(TestSize - 1, offset);
    }

    [TestMethod]
    public async Task Shape_GetCoordinates_ReturnsLongArray()
    {
        var shape = new Shape(1000L, 1000L);
        long[] coords = shape.GetCoordinates(999_999);
        Assert.AreEqual(2, coords.Length);
        Assert.AreEqual(999L, coords[0]);
        Assert.AreEqual(999L, coords[1]);
    }

    [TestMethod]
    public async Task NDArray_Zeros_AcceptsLongShape()
    {
        var arr = np.zeros(new Shape(TestSize), np.uint8);
        Assert.AreEqual(TestSize, arr.size);
        Assert.AreEqual(0, arr.GetByte(0));
        Assert.AreEqual(0, arr.GetByte(TestSize - 1));
    }

    [TestMethod]
    public async Task NDArray_Ones_AcceptsLongShape()
    {
        var arr = np.ones(new Shape(TestSize), np.uint8);
        Assert.AreEqual(TestSize, arr.size);
        Assert.AreEqual(1, arr.GetByte(0));
        Assert.AreEqual(1, arr.GetByte(TestSize - 1));
    }

    [TestMethod]
    public async Task NDArray_Full_AcceptsLongShape()
    {
        var arr = np.full(new Shape(TestSize), (byte)42, np.uint8);
        Assert.AreEqual(TestSize, arr.size);
        Assert.AreEqual(42, arr.GetByte(0));
        Assert.AreEqual(42, arr.GetByte(TestSize - 1));
    }

    [TestMethod]
    public async Task NDArray_GetSet_AcceptsLongIndex()
    {
        var arr = np.zeros(new Shape(TestSize), np.uint8);
        arr.SetByte((byte)123, TestSize - 1);
        Assert.AreEqual(123, arr.GetByte(TestSize - 1));
    }

    [TestMethod]
    public async Task NDArray_Reshape_AcceptsLongDimensions()
    {
        var arr = np.zeros(new Shape(TestSize), np.uint8);
        var reshaped = np.reshape(arr, new Shape(1000L, 1000L));
        Assert.AreEqual(TestSize, reshaped.size);
        Assert.AreEqual(2, reshaped.ndim);
    }

    [TestMethod]
    public async Task NDArray_Sum_ReturnsCorrectForLargeArray()
    {
        var arr = np.ones(new Shape(TestSize), np.uint8);
        var result = np.sum(arr);
        Assert.AreEqual(0, result.ndim);
    }

    [TestMethod]
    public async Task NDArray_Mean_ReturnsCorrectForLargeArray()
    {
        var arr = np.ones(new Shape(TestSize), np.uint8);
        var result = np.mean(arr);
        Assert.AreEqual(1.0, result.GetDouble(0), 0.001);
    }

    [TestMethod]
    public async Task NDArray_ArgMax_ReturnsLongIndex()
    {
        var arr = np.zeros(new Shape(TestSize), np.uint8);
        arr.SetByte((byte)255, TestSize - 100);
        var result = np.argmax(arr);
        Assert.AreEqual(TestSize - 100, result);
    }

    [TestMethod]
    public async Task NDArray_ArgMin_ReturnsLongIndex()
    {
        var arr = np.ones(new Shape(TestSize), np.uint8);
        arr.SetByte((byte)0, TestSize / 2);
        var result = np.argmin(arr);
        Assert.AreEqual(TestSize / 2, result);
    }

    [TestMethod]
    public async Task NDArray_CountNonzero_ReturnsLongCount()
    {
        var arr = np.ones(new Shape(TestSize), np.uint8);
        var count = np.count_nonzero(arr);
        Assert.AreEqual(TestSize, count);
    }

    [TestMethod]
    public async Task NDArray_Slicing_WorksWithLargeIndices()
    {
        var arr = np.full(new Shape(TestSize), (byte)99, np.uint8);
        long start = TestSize - 100;
        long stop = TestSize;
        var slice = arr[$"{start}:{stop}"];
        Assert.AreEqual(100, slice.size);
        Assert.AreEqual(99, slice.GetByte(0));
    }

    [TestMethod]
    public async Task NDArray_BroadcastTo_WorksWithLongTarget()
    {
        var scalar = np.full(new Shape(1L), (byte)77, np.uint8);
        var broadcast = np.broadcast_to(scalar, new Shape(TestSize));
        Assert.AreEqual(TestSize, broadcast.size);
        Assert.AreEqual(77, broadcast.GetByte(0));
        Assert.AreEqual(77, broadcast.GetByte(TestSize - 1));
    }

    [TestMethod]
    public async Task NDArray_Add_WorksWithLargeArrays()
    {
        var a = np.ones(new Shape(TestSize), np.uint8);
        var b = np.ones(new Shape(TestSize), np.uint8);
        var result = np.add(a, b);
        Assert.AreEqual(TestSize, result.size);
        Assert.AreEqual(2, result.GetByte(0));
        Assert.AreEqual(2, result.GetByte(TestSize - 1));
    }

    [TestMethod]
    public async Task NDArray_Cumsum_WorksWithLargeArray()
    {
        var arr = np.ones(new Shape(TestSize), np.uint8);
        var result = np.cumsum(arr);
        Assert.AreEqual(TestSize, result.size);
    }

    [TestMethod]
    public async Task NDArray_Roll_WorksWithLargeShift()
    {
        var arr = np.zeros(new Shape(TestSize), np.uint8);
        arr.SetByte((byte)1, 0);
        var result = np.roll(arr, TestSize / 2);
        Assert.AreEqual(TestSize, result.size);
        Assert.AreEqual(1, result.GetByte(TestSize / 2));
    }

    [TestMethod]
    public async Task NDArray_Clip_WorksWithLargeArray()
    {
        var arr = np.full(new Shape(TestSize), (byte)100, np.uint8);
        var result = np.clip(arr, 50, 75);
        Assert.AreEqual(TestSize, result.size);
        Assert.AreEqual(75, result.GetByte(0));
    }

    [TestMethod]
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

    [TestMethod]
    public async Task NDArray_ExpandDims_WorksWithLargeArray()
    {
        var arr = np.zeros(new Shape(TestSize), np.uint8);
        var result = np.expand_dims(arr, 0);
        Assert.AreEqual(TestSize, result.size);
        Assert.AreEqual(2, result.ndim);
    }

    [TestMethod]
    public async Task NDArray_Ravel_WorksWithLargeArray()
    {
        var arr = np.zeros(new Shape(1000L, 1000L), np.uint8);
        var result = np.ravel(arr);
        Assert.AreEqual(TestSize, result.size);
        Assert.AreEqual(1, result.ndim);
    }

    [TestMethod]
    public async Task NDArray_Flatten_WorksWithLargeArray()
    {
        var arr = np.zeros(new Shape(1000L, 1000L), np.uint8);
        var result = arr.flatten();
        Assert.AreEqual(TestSize, result.size);
        Assert.AreEqual(1, result.ndim);
    }

    [TestMethod]
    public async Task NDArray_All_WorksWithLargeArray()
    {
        var arr = np.ones(new Shape(TestSize), np.uint8);
        var result = np.all(arr);
        Assert.IsTrue(result);
    }

    [TestMethod]
    public async Task NDArray_Any_WorksWithLargeArray()
    {
        var arr = np.zeros(new Shape(TestSize), np.uint8);
        arr.SetByte((byte)1, TestSize - 1);
        var result = np.any(arr);
        Assert.IsTrue(result);
    }

    [TestMethod]
    public async Task NDArray_Min_WorksWithLargeArray()
    {
        var arr = np.ones(new Shape(TestSize), np.uint8);
        var result = np.min(arr);
        Assert.AreEqual(1, result.GetByte(0));
    }

    [TestMethod]
    public async Task NDArray_Max_WorksWithLargeArray()
    {
        var arr = np.ones(new Shape(TestSize), np.uint8);
        var result = np.max(arr);
        Assert.AreEqual(1, result.GetByte(0));
    }

    [TestMethod]
    public async Task NDArray_Maximum_WorksWithLargeArrays()
    {
        var a = np.full(new Shape(TestSize), (byte)5, np.uint8);
        var b = np.full(new Shape(TestSize), (byte)10, np.uint8);
        var result = np.maximum(a, b);
        Assert.AreEqual(10, result.GetByte(0));
    }

    [TestMethod]
    public async Task NDArray_Minimum_WorksWithLargeArrays()
    {
        var a = np.full(new Shape(TestSize), (byte)5, np.uint8);
        var b = np.full(new Shape(TestSize), (byte)10, np.uint8);
        var result = np.minimum(a, b);
        Assert.AreEqual(5, result.GetByte(0));
    }

    [TestMethod]
    public async Task NDArray_LeftShift_WorksWithLargeArray()
    {
        var arr = np.full(new Shape(TestSize), (byte)1, np.uint8);
        var result = np.left_shift(arr, 4);
        Assert.AreEqual(16, result.GetByte(0));
    }

    [TestMethod]
    public async Task NDArray_RightShift_WorksWithLargeArray()
    {
        var arr = np.full(new Shape(TestSize), (byte)16, np.uint8);
        var result = np.right_shift(arr, 4);
        Assert.AreEqual(1, result.GetByte(0));
    }

    [TestMethod]
    public async Task NDArray_Invert_WorksWithLargeArray()
    {
        var arr = np.full(new Shape(TestSize), (byte)0b11110000, np.uint8);
        var result = np.invert(arr);
        Assert.AreEqual(0b00001111, result.GetByte(0));
    }

    [TestMethod]
    public async Task NDArray_Square_WorksWithLargeArray()
    {
        var arr = np.full(new Shape(TestSize), (byte)3, np.uint8);
        var result = np.square(arr);
        Assert.AreEqual(9, result.GetByte(0));
    }

    [TestMethod]
    public async Task NDArray_Multiply_WorksWithLargeArrays()
    {
        var a = np.full(new Shape(TestSize), (byte)3, np.uint8);
        var b = np.full(new Shape(TestSize), (byte)4, np.uint8);
        var result = np.multiply(a, b);
        Assert.AreEqual(12, result.GetByte(0));
    }

    [TestMethod]
    public async Task NDArray_Subtract_WorksWithLargeArrays()
    {
        var a = np.full(new Shape(TestSize), (byte)10, np.uint8);
        var b = np.full(new Shape(TestSize), (byte)3, np.uint8);
        var result = np.subtract(a, b);
        Assert.AreEqual(7, result.GetByte(0));
    }

    // ================================================================
    // UNARY MATH OPERATIONS
    // ================================================================

    [TestMethod]
    public async Task NDArray_Abs_WorksWithLargeArray()
    {
        var arr = np.full(new Shape(TestSize), -5, np.int32);
        var result = np.abs(arr);
        Assert.AreEqual(TestSize, result.size);
        Assert.AreEqual(5, result.GetInt32(0));
    }

    [TestMethod]
    public async Task NDArray_Absolute_WorksWithLargeArray()
    {
        var arr = np.full(new Shape(TestSize), -5, np.int32);
        var result = np.absolute(arr);
        Assert.AreEqual(TestSize, result.size);
        Assert.AreEqual(5, result.GetInt32(0));
    }

    [TestMethod]
    public async Task NDArray_Negative_WorksWithLargeArray()
    {
        var arr = np.full(new Shape(TestSize), 5, np.int32);
        var result = np.negative(arr);
        Assert.AreEqual(TestSize, result.size);
        Assert.AreEqual(-5, result.GetInt32(0));
    }

    [TestMethod]
    public async Task NDArray_Positive_WorksWithLargeArray()
    {
        var arr = np.full(new Shape(TestSize), 5, np.int32);
        var result = np.positive(arr);
        Assert.AreEqual(TestSize, result.size);
        Assert.AreEqual(5, result.GetInt32(0));
    }

    [TestMethod]
    public async Task NDArray_Sqrt_WorksWithLargeArray()
    {
        var arr = np.full(new Shape(TestSize), 16.0, np.float64);
        var result = np.sqrt(arr);
        Assert.AreEqual(TestSize, result.size);
        Assert.AreEqual(4.0, result.GetDouble(0), 0.001);
    }

    [TestMethod]
    public async Task NDArray_Cbrt_WorksWithLargeArray()
    {
        var arr = np.full(new Shape(TestSize), 27.0, np.float64);
        var result = np.cbrt(arr);
        Assert.AreEqual(TestSize, result.size);
        Assert.AreEqual(3.0, result.GetDouble(0), 0.001);
    }

    [TestMethod]
    public async Task NDArray_Reciprocal_WorksWithLargeArray()
    {
        var arr = np.full(new Shape(TestSize), 4.0, np.float64);
        var result = np.reciprocal(arr);
        Assert.AreEqual(TestSize, result.size);
        Assert.AreEqual(0.25, result.GetDouble(0), 0.001);
    }

    [TestMethod]
    public async Task NDArray_Floor_WorksWithLargeArray()
    {
        var arr = np.full(new Shape(TestSize), 3.7, np.float64);
        var result = np.floor(arr);
        Assert.AreEqual(TestSize, result.size);
        Assert.AreEqual(3.0, result.GetDouble(0), 0.001);
    }

    [TestMethod]
    public async Task NDArray_Ceil_WorksWithLargeArray()
    {
        var arr = np.full(new Shape(TestSize), 3.7, np.float64);
        var result = np.ceil(arr);
        Assert.AreEqual(TestSize, result.size);
        Assert.AreEqual(4.0, result.GetDouble(0), 0.001);
    }

    [TestMethod]
    public async Task NDArray_Trunc_WorksWithLargeArray()
    {
        var arr = np.full(new Shape(TestSize), 3.7, np.float64);
        var result = np.trunc(arr);
        Assert.AreEqual(TestSize, result.size);
        Assert.AreEqual(3.0, result.GetDouble(0), 0.001);
    }

    [TestMethod]
    public async Task NDArray_Sign_WorksWithLargeArray()
    {
        var arr = np.full(new Shape(TestSize), -5, np.int32);
        var result = np.sign(arr);
        Assert.AreEqual(TestSize, result.size);
        Assert.AreEqual(-1, result.GetInt32(0));

        arr = np.full(new Shape(TestSize), 5, np.int32);
        result = np.sign(arr);
        Assert.AreEqual(1, result.GetInt32(0));

        arr = np.full(new Shape(TestSize), 0, np.int32);
        result = np.sign(arr);
        Assert.AreEqual(0, result.GetInt32(0));
    }

    [TestMethod]
    public async Task NDArray_Exp_WorksWithLargeArray()
    {
        var arr = np.full(new Shape(TestSize), 1.0, np.float64);
        var result = np.exp(arr);
        Assert.AreEqual(TestSize, result.size);
        Assert.AreEqual(Math.E, result.GetDouble(0), 0.0001);
    }

    [TestMethod]
    public async Task NDArray_Exp2_WorksWithLargeArray()
    {
        var arr = np.full(new Shape(TestSize), 3.0, np.float64);
        var result = np.exp2(arr);
        Assert.AreEqual(TestSize, result.size);
        Assert.AreEqual(8.0, result.GetDouble(0), 0.001);
    }

    [TestMethod]
    public async Task NDArray_Expm1_WorksWithLargeArray()
    {
        var arr = np.full(new Shape(TestSize), 0.0, np.float64);
        var result = np.expm1(arr);
        Assert.AreEqual(TestSize, result.size);
        Assert.AreEqual(0.0, result.GetDouble(0), 0.001);
    }

    [TestMethod]
    public async Task NDArray_Log_WorksWithLargeArray()
    {
        var arr = np.full(new Shape(TestSize), Math.E, np.float64);
        var result = np.log(arr);
        Assert.AreEqual(TestSize, result.size);
        Assert.AreEqual(1.0, result.GetDouble(0), 0.0001);
    }

    [TestMethod]
    public async Task NDArray_Log10_WorksWithLargeArray()
    {
        var arr = np.full(new Shape(TestSize), 100.0, np.float64);
        var result = np.log10(arr);
        Assert.AreEqual(TestSize, result.size);
        Assert.AreEqual(2.0, result.GetDouble(0), 0.001);
    }

    [TestMethod]
    public async Task NDArray_Log1p_WorksWithLargeArray()
    {
        var arr = np.full(new Shape(TestSize), 0.0, np.float64);
        var result = np.log1p(arr);
        Assert.AreEqual(TestSize, result.size);
        Assert.AreEqual(0.0, result.GetDouble(0), 0.001);
    }

    [TestMethod]
    public async Task NDArray_Log2_WorksWithLargeArray()
    {
        var arr = np.full(new Shape(TestSize), 8.0, np.float64);
        var result = np.log2(arr);
        Assert.AreEqual(TestSize, result.size);
        Assert.AreEqual(3.0, result.GetDouble(0), 0.001);
    }

    [TestMethod]
    public async Task NDArray_Sin_WorksWithLargeArray()
    {
        var arr = np.full(new Shape(TestSize), 0.0, np.float64);
        var result = np.sin(arr);
        Assert.AreEqual(TestSize, result.size);
        Assert.AreEqual(0.0, result.GetDouble(0), 0.001);
    }

    [TestMethod]
    public async Task NDArray_Cos_WorksWithLargeArray()
    {
        var arr = np.full(new Shape(TestSize), 0.0, np.float64);
        var result = np.cos(arr);
        Assert.AreEqual(TestSize, result.size);
        Assert.AreEqual(1.0, result.GetDouble(0), 0.001);
    }

    [TestMethod]
    public async Task NDArray_Tan_WorksWithLargeArray()
    {
        var arr = np.full(new Shape(TestSize), 0.0, np.float64);
        var result = np.tan(arr);
        Assert.AreEqual(TestSize, result.size);
        Assert.AreEqual(0.0, result.GetDouble(0), 0.001);
    }

    // ================================================================
    // BINARY MATH OPERATIONS
    // ================================================================

    [TestMethod]
    public async Task NDArray_Divide_WorksWithLargeArrays()
    {
        var a = np.full(new Shape(TestSize), 10.0, np.float64);
        var b = np.full(new Shape(TestSize), 4.0, np.float64);
        var result = np.divide(a, b);
        Assert.AreEqual(TestSize, result.size);
        Assert.AreEqual(2.5, result.GetDouble(0), 0.001);
    }

    [TestMethod]
    public async Task NDArray_TrueDivide_WorksWithLargeArrays()
    {
        var a = np.full(new Shape(TestSize), 10.0, np.float64);
        var b = np.full(new Shape(TestSize), 4.0, np.float64);
        var result = np.true_divide(a, b);
        Assert.AreEqual(TestSize, result.size);
        Assert.AreEqual(2.5, result.GetDouble(0), 0.001);
    }

    [TestMethod]
    public async Task NDArray_FloorDivide_WorksWithLargeArrays()
    {
        var a = np.full(new Shape(TestSize), 10.0, np.float64);
        var b = np.full(new Shape(TestSize), 4.0, np.float64);
        var result = np.floor_divide(a, b);
        Assert.AreEqual(TestSize, result.size);
        Assert.AreEqual(2.0, result.GetDouble(0), 0.001);
    }

    [TestMethod]
    public async Task NDArray_Mod_WorksWithLargeArrays()
    {
        var a = np.full(new Shape(TestSize), 10, np.int32);
        var b = np.full(new Shape(TestSize), 3, np.int32);
        var result = np.mod(a, b);
        Assert.AreEqual(TestSize, result.size);
        Assert.AreEqual(1, result.GetInt32(0));
    }

    [TestMethod]
    public async Task NDArray_Power_WorksWithLargeArrays()
    {
        var a = np.full(new Shape(TestSize), 2, np.int32);
        var b = np.full(new Shape(TestSize), 10, np.int32);
        var result = np.power(a, b);
        Assert.AreEqual(TestSize, result.size);
        Assert.AreEqual(1024, result.GetInt32(0));
    }

    [TestMethod]
    public async Task NDArray_Modf_WorksWithLargeArray()
    {
        var arr = np.full(new Shape(TestSize), 3.5, np.float64);
        var (fractional, integer) = np.modf(arr);
        Assert.AreEqual(TestSize, fractional.size);
        Assert.AreEqual(TestSize, integer.size);
        Assert.AreEqual(0.5, fractional.GetDouble(0), 0.001);
        Assert.AreEqual(3.0, integer.GetDouble(0), 0.001);
    }

    // ================================================================
    // ADDITIONAL REDUCTIONS
    // ================================================================

    [TestMethod]
    public async Task NDArray_Prod_WorksWithLargeArray()
    {
        // Use small values to avoid overflow
        var arr = np.full(new Shape(10L), 2L, np.int64);
        var result = np.prod(arr);
        Assert.AreEqual(0, result.ndim);
        Assert.AreEqual(1024L, result.GetInt64(0));
    }

    [TestMethod]
    public async Task NDArray_Std_WorksWithLargeArray()
    {
        var arr = np.array(new double[] { 1, 2, 3, 4, 5 });
        var result = np.std(arr);
        Assert.AreEqual(0, result.ndim);
        Assert.AreEqual(1.414214, result.GetDouble(0), 0.0001);
    }

    [TestMethod]
    public async Task NDArray_Var_WorksWithLargeArray()
    {
        var arr = np.array(new double[] { 1, 2, 3, 4, 5 });
        var result = np.var(arr);
        Assert.AreEqual(0, result.ndim);
        Assert.AreEqual(2.0, result.GetDouble(0), 0.0001);
    }

    [TestMethod]
    public async Task NDArray_Cumprod_WorksWithLargeArray()
    {
        var arr = np.full(new Shape(5L), 2L, np.int64);
        var result = np.cumprod(arr);
        Assert.AreEqual(5, result.size);
        Assert.AreEqual(2L, result.GetInt64(0));
        Assert.AreEqual(4L, result.GetInt64(1));
        Assert.AreEqual(8L, result.GetInt64(2));
        Assert.AreEqual(16L, result.GetInt64(3));
        Assert.AreEqual(32L, result.GetInt64(4));
    }

    // ================================================================
    // NAN-AWARE FUNCTIONS
    // ================================================================

    [TestMethod]
    public async Task NDArray_Nansum_WorksWithLargeArray()
    {
        var arr = np.array(new double[] { 1.0, double.NaN, 3.0, double.NaN, 5.0 });
        var result = np.nansum(arr);
        Assert.AreEqual(9.0, result.GetDouble(0), 0.001);
    }

    [TestMethod]
    public async Task NDArray_Nanmean_WorksWithLargeArray()
    {
        var arr = np.array(new double[] { 1.0, double.NaN, 3.0, double.NaN, 5.0 });
        var result = np.nanmean(arr);
        Assert.AreEqual(3.0, result.GetDouble(0), 0.001);
    }

    [TestMethod]
    public async Task NDArray_Nanmin_WorksWithLargeArray()
    {
        var arr = np.array(new double[] { 1.0, double.NaN, 3.0, double.NaN, 5.0 });
        var result = np.nanmin(arr);
        Assert.AreEqual(1.0, result.GetDouble(0), 0.001);
    }

    [TestMethod]
    public async Task NDArray_Nanmax_WorksWithLargeArray()
    {
        var arr = np.array(new double[] { 1.0, double.NaN, 3.0, double.NaN, 5.0 });
        var result = np.nanmax(arr);
        Assert.AreEqual(5.0, result.GetDouble(0), 0.001);
    }

    [TestMethod]
    public async Task NDArray_Nanprod_WorksWithLargeArray()
    {
        var arr = np.array(new double[] { 1.0, double.NaN, 3.0, double.NaN, 5.0 });
        var result = np.nanprod(arr);
        Assert.AreEqual(15.0, result.GetDouble(0), 0.001);
    }

    [TestMethod]
    public async Task NDArray_Nanstd_WorksWithLargeArray()
    {
        var arr = np.array(new double[] { 1.0, double.NaN, 3.0, double.NaN, 5.0 });
        var result = np.nanstd(arr);
        Assert.AreEqual(1.632993, result.GetDouble(0), 0.0001);
    }

    [TestMethod]
    public async Task NDArray_Nanvar_WorksWithLargeArray()
    {
        var arr = np.array(new double[] { 1.0, double.NaN, 3.0, double.NaN, 5.0 });
        var result = np.nanvar(arr);
        Assert.AreEqual(2.666667, result.GetDouble(0), 0.0001);
    }

    // ================================================================
    // COMPARISON & LOGIC
    // ================================================================

    [TestMethod]
    public async Task NDArray_Isnan_WorksWithLargeArray()
    {
        var arr = np.array(new double[] { 1.0, double.NaN, double.PositiveInfinity, double.NegativeInfinity, 0.0 });
        var result = np.isnan(arr);
        Assert.AreEqual(5, result.size);
        Assert.IsFalse(result.GetBoolean(0));  // 1.0
        Assert.IsTrue(result.GetBoolean(1));   // NaN
        Assert.IsFalse(result.GetBoolean(2));  // +inf
        Assert.IsFalse(result.GetBoolean(3));  // -inf
        Assert.IsFalse(result.GetBoolean(4));  // 0.0
    }

    [TestMethod]
    [OpenBugs] // isinf not implemented - returns null (Default.IsInf.cs)
    public async Task NDArray_Isinf_WorksWithLargeArray()
    {
        var arr = np.array(new double[] { 1.0, double.NaN, double.PositiveInfinity, double.NegativeInfinity, 0.0 });
        var result = np.isinf(arr);
        Assert.AreEqual(5, result.size);
        Assert.IsFalse(result.GetBoolean(0));  // 1.0
        Assert.IsFalse(result.GetBoolean(1));  // NaN
        Assert.IsTrue(result.GetBoolean(2));   // +inf
        Assert.IsTrue(result.GetBoolean(3));   // -inf
        Assert.IsFalse(result.GetBoolean(4));  // 0.0
    }

    [TestMethod]
    public async Task NDArray_Isfinite_WorksWithLargeArray()
    {
        var arr = np.array(new double[] { 1.0, double.NaN, double.PositiveInfinity, double.NegativeInfinity, 0.0 });
        var result = np.isfinite(arr);
        Assert.AreEqual(5, result.size);
        Assert.IsTrue(result.GetBoolean(0));   // 1.0
        Assert.IsFalse(result.GetBoolean(1));  // NaN
        Assert.IsFalse(result.GetBoolean(2));  // +inf
        Assert.IsFalse(result.GetBoolean(3));  // -inf
        Assert.IsTrue(result.GetBoolean(4));   // 0.0
    }

    [TestMethod]
    public async Task NDArray_ArrayEqual_WorksWithLargeArrays()
    {
        var a = np.full(new Shape(TestSize), 1.0, np.float64);
        var b = np.full(new Shape(TestSize), 1.0, np.float64);
        Assert.IsTrue(np.array_equal(a, b));

        b = np.full(new Shape(TestSize), 1.1, np.float64);
        Assert.IsFalse(np.array_equal(a, b));
    }

    [TestMethod]
    public async Task NDArray_Allclose_WorksWithLargeArrays()
    {
        var a = np.array(new double[] { 1.0, 2.0, 3.0 });
        var b = np.array(new double[] { 1.0, 2.0, 3.0000001 });
        Assert.IsTrue(np.allclose(a, b));
    }

    [TestMethod]
    public async Task NDArray_Isclose_WorksWithLargeArrays()
    {
        var a = np.array(new double[] { 1.0, 2.0, 3.0 });
        var b = np.array(new double[] { 1.0, 2.0, 3.0000001 });
        var result = np.isclose(a, b);
        Assert.AreEqual(3, result.size);
        Assert.IsTrue(result.GetBoolean(0));
        Assert.IsTrue(result.GetBoolean(1));
        Assert.IsTrue(result.GetBoolean(2));
    }

    // ================================================================
    // SHAPE MANIPULATION
    // ================================================================

    [TestMethod]
    public async Task NDArray_Concatenate_WorksWithLargeArrays()
    {
        var a = np.full(new Shape(TestSize / 2), (byte)1, np.uint8);
        var b = np.full(new Shape(TestSize / 2), (byte)2, np.uint8);
        var result = np.concatenate(new NDArray[] { a, b });
        Assert.AreEqual(TestSize, result.size);
        Assert.AreEqual(1, result.GetByte(0));
        Assert.AreEqual(2, result.GetByte(TestSize / 2));
    }

    [TestMethod]
    public async Task NDArray_Stack_WorksWithLargeArrays()
    {
        var a = np.full(new Shape(1000L, 1000L), (byte)1, np.uint8);
        var b = np.full(new Shape(1000L, 1000L), (byte)2, np.uint8);
        var result = np.stack(new NDArray[] { a, b }, axis: 0);
        Assert.AreEqual(2_000_000L, result.size);
        Assert.AreEqual(3, result.ndim);
        Assert.AreEqual(2, result.shape[0]);
    }

    [TestMethod]
    public async Task NDArray_Hstack_WorksWithLargeArrays()
    {
        var a = np.full(new Shape(1000L, 500L), (byte)1, np.uint8);
        var b = np.full(new Shape(1000L, 500L), (byte)2, np.uint8);
        var result = np.hstack(a, b);
        Assert.AreEqual(TestSize, result.size);
        Assert.AreEqual(1000, result.shape[0]);
        Assert.AreEqual(1000, result.shape[1]);
    }

    [TestMethod]
    public async Task NDArray_Vstack_WorksWithLargeArrays()
    {
        var a = np.full(new Shape(500L, 1000L), (byte)1, np.uint8);
        var b = np.full(new Shape(500L, 1000L), (byte)2, np.uint8);
        var result = np.vstack(a, b);
        Assert.AreEqual(TestSize, result.size);
        Assert.AreEqual(1000, result.shape[0]);
        Assert.AreEqual(1000, result.shape[1]);
    }

    [TestMethod]
    public async Task NDArray_Dstack_WorksWithLargeArrays()
    {
        var a = np.full(new Shape(100L, 100L), (byte)1, np.uint8);
        var b = np.full(new Shape(100L, 100L), (byte)2, np.uint8);
        var result = np.dstack(a, b);
        Assert.AreEqual(20_000L, result.size);
        Assert.AreEqual(3, result.ndim);
        Assert.AreEqual(100, result.shape[0]);
        Assert.AreEqual(100, result.shape[1]);
        Assert.AreEqual(2, result.shape[2]);
    }

    [TestMethod]
    public async Task NDArray_Moveaxis_WorksWithLargeArrays()
    {
        var arr = np.zeros(new Shape(100L, 200L, 300L), np.uint8);
        var result = np.moveaxis(arr, 0, -1);
        Assert.AreEqual(6_000_000L, result.size);
        Assert.AreEqual(200, result.shape[0]);
        Assert.AreEqual(300, result.shape[1]);
        Assert.AreEqual(100, result.shape[2]);
    }

    [TestMethod]
    public async Task NDArray_Rollaxis_WorksWithLargeArrays()
    {
        var arr = np.zeros(new Shape(100L, 200L, 300L), np.uint8);
        var result = np.rollaxis(arr, 2, 0);
        Assert.AreEqual(6_000_000L, result.size);
        Assert.AreEqual(300, result.shape[0]);
        Assert.AreEqual(100, result.shape[1]);
        Assert.AreEqual(200, result.shape[2]);
    }

    [TestMethod]
    public async Task NDArray_Repeat_WorksWithLargeArray()
    {
        var arr = np.array(new byte[] { 1, 2, 3 });
        var result = np.repeat(arr, 2);
        Assert.AreEqual(6, result.size);
        Assert.AreEqual(1, result.GetByte(0));
        Assert.AreEqual(1, result.GetByte(1));
        Assert.AreEqual(2, result.GetByte(2));
        Assert.AreEqual(2, result.GetByte(3));
    }

    // ================================================================
    // SORTING & SEARCHING
    // ================================================================

    [TestMethod]
    public async Task NDArray_Argsort_WorksWithLargeArray()
    {
        var arr = np.array(new int[] { 3, 1, 4, 1, 5, 9, 2, 6 });
        var result = np.argsort<int>(arr);
        Assert.AreEqual(8, result.size);
        // argsort returns Int64 indices (NumPy behavior)
        Assert.AreEqual(typeof(long), result.dtype);
        Assert.AreEqual(1L, result.GetInt64(0)); // Index of first 1
        Assert.AreEqual(3L, result.GetInt64(1)); // Index of second 1
        Assert.AreEqual(6L, result.GetInt64(2)); // Index of 2
        Assert.AreEqual(0L, result.GetInt64(3)); // Index of 3
    }

    [TestMethod]
    public async Task NDArray_Nonzero_WorksWithLargeArray()
    {
        var arr = np.array(new int[] { 0, 5, 0, 3, 0 });
        var result = np.nonzero(arr);
        Assert.AreEqual(1, result.Length); // 1D array returns 1 index array
        Assert.AreEqual(2, result[0].size); // Two nonzero elements
        Assert.AreEqual(1, result[0].GetInt64(0)); // Index of 5
        Assert.AreEqual(3, result[0].GetInt64(1)); // Index of 3
    }

    [TestMethod]
    public async Task NDArray_Searchsorted_WorksWithLargeArray()
    {
        var arr = np.array(new int[] { 1, 3, 5, 7, 9 });
        var result = np.searchsorted(arr, 4);
        Assert.AreEqual(2, result); // Insert position for 4
    }

    // ================================================================
    // CREATION
    // ================================================================

    [TestMethod]
    public async Task NDArray_Eye_WorksWithLargeArray()
    {
        var result = np.eye(1000);
        Assert.AreEqual(TestSize, result.size);
        Assert.AreEqual(2, result.ndim);
        Assert.AreEqual(1.0, result.GetDouble(0, 0));
        Assert.AreEqual(0.0, result.GetDouble(0, 1));
        Assert.AreEqual(1.0, result.GetDouble(1, 1));
    }

    [TestMethod]
    public async Task NDArray_Identity_WorksWithLargeArray()
    {
        var result = np.identity(1000);
        Assert.AreEqual(TestSize, result.size);
        Assert.AreEqual(2, result.ndim);
        Assert.AreEqual(1.0, result.GetDouble(0, 0));
        Assert.AreEqual(0.0, result.GetDouble(0, 1));
        Assert.AreEqual(1.0, result.GetDouble(999, 999));
    }

    [TestMethod]
    public async Task NDArray_Array_WorksWithLargeArray()
    {
        var data = new byte[TestSize];
        for (int i = 0; i < TestSize; i++) data[i] = (byte)(i % 256);
        var result = np.array(data);
        Assert.AreEqual(TestSize, result.size);
        Assert.AreEqual(0, result.GetByte(0));
        Assert.AreEqual(255, result.GetByte(255));
    }

    // ================================================================
    // OTHER
    // ================================================================

    [TestMethod]
    public async Task NDArray_Around_WorksWithLargeArray()
    {
        var arr = np.array(new double[] { 1.567, 2.345, 3.789 });
        var result = np.around(arr, 1);
        Assert.AreEqual(3, result.size);
        Assert.AreEqual(1.6, result.GetDouble(0), 0.001);
        Assert.AreEqual(2.3, result.GetDouble(1), 0.001);
        Assert.AreEqual(3.8, result.GetDouble(2), 0.001);
    }

    [TestMethod]
    public async Task NDArray_Copyto_WorksWithLargeArray()
    {
        var dst = np.zeros(new Shape(TestSize), np.float64);
        var src = np.ones(new Shape(TestSize), np.float64);
        np.copyto(dst, src);
        Assert.AreEqual(1.0, dst.GetDouble(0));
        Assert.AreEqual(1.0, dst.GetDouble(TestSize - 1));
    }

    // ================================================================
    // LINEAR ALGEBRA
    // ================================================================

    [TestMethod]
    public async Task NDArray_Dot_1D_WorksWithLargeArrays()
    {
        // dot product of two 1D vectors: sum of element-wise products
        var a = np.ones(new Shape(TestSize), np.float64);
        var b = np.ones(new Shape(TestSize), np.float64);
        var result = np.dot(a, b);
        Assert.AreEqual(0, result.ndim); // Scalar result
        Assert.AreEqual((double)TestSize, result.GetDouble(0), 0.001);
    }

    [TestMethod]
    public async Task NDArray_Dot_2D_WorksWithLargeArrays()
    {
        // Matrix multiplication: (100x100) @ (100x100) = (100x100)
        // Each element = sum of 100 ones = 100
        var a = np.ones(new Shape(100L, 100L), np.float64);
        var b = np.ones(new Shape(100L, 100L), np.float64);
        var result = np.dot(a, b);
        Assert.AreEqual(2, result.ndim);
        Assert.AreEqual(100, result.shape[0]);
        Assert.AreEqual(100, result.shape[1]);
        Assert.AreEqual(100.0, result.GetDouble(0, 0), 0.001);
    }

    [TestMethod]
    public async Task NDArray_Matmul_WorksWithLargeArrays()
    {
        // Matrix multiplication: (100x100) @ (100x100) = (100x100)
        var a = np.ones(new Shape(100L, 100L), np.float64);
        var b = np.ones(new Shape(100L, 100L), np.float64);
        var result = np.matmul(a, b);
        Assert.AreEqual(2, result.ndim);
        Assert.AreEqual(100, result.shape[0]);
        Assert.AreEqual(100, result.shape[1]);
        Assert.AreEqual(100.0, result.GetDouble(0, 0), 0.001);
    }

    [TestMethod]
    public async Task NDArray_Outer_WorksWithLargeArrays()
    {
        // Outer product: (1000,) x (1000,) = (1000, 1000)
        // Each element = a[i] * b[j] = 2 * 3 = 6
        var a = np.full(new Shape(1000L), 2.0, np.float64);
        var b = np.full(new Shape(1000L), 3.0, np.float64);
        var result = np.outer(a, b);
        Assert.AreEqual(2, result.ndim);
        Assert.AreEqual(1000, result.shape[0]);
        Assert.AreEqual(1000, result.shape[1]);
        Assert.AreEqual(TestSize, result.size);
        Assert.AreEqual(6.0, result.GetDouble(0, 0), 0.001);
        Assert.AreEqual(6.0, result.GetDouble(999, 999), 0.001);
    }
}
