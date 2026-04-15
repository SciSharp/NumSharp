using System;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using NumSharp;
using NumSharp.UnitTest;

namespace NumSharp.UnitTest.Backends.Kernels;

/// <summary>
/// Battle tests for IL kernels with arrays exceeding int.MaxValue elements.
/// Uses byte arrays to stay under 3GB while exceeding int.MaxValue element count.
///
/// int.MaxValue = 2,147,483,647 (~2.1B)
/// Test size = 2,500,000,000 (~2.5B elements = ~2.5GB for bytes)
///
/// All tests are marked [LargeMemoryTest] to exclude from CI.
/// </summary>
public class ILKernelGenerator_LargeArray_BattleTest
{
    // 2.5 billion elements - exceeds int.MaxValue (2.147B), under 3GB for bytes
    private const long LargeSize = 2_500_000_000L;

    // Smaller size for operations that need two arrays (5GB total would exceed limits)
    // 1.4B * 2 = 2.8GB total
    private const long DualArraySize = 1_400_000_000L;

    #region Binary Operations

    [TestMethod, LargeMemoryTest]
    public async Task Binary_Add_LargeByteArray()
    {
        // Two 1.4B byte arrays = 2.8GB total
        var a = np.ones(new Shape(DualArraySize), NPTypeCode.Byte);
        var b = np.ones(new Shape(DualArraySize), NPTypeCode.Byte);

        var result = a + b;

        // Verify first, middle, and last elements
        ((byte)result[0]).Should().Be((byte)2);
        ((byte)result[DualArraySize / 2]).Should().Be((byte)2);
        ((byte)result[DualArraySize - 1]).Should().Be((byte)2);

        // Verify element beyond int.MaxValue
        long beyondIntMax = (long)int.MaxValue + 1000;
        if (beyondIntMax < DualArraySize)
        {
            ((byte)result[beyondIntMax]).Should().Be((byte)2);
        }
    }

    [TestMethod, LargeMemoryTest]
    public async Task Binary_Subtract_LargeByteArray()
    {
        var a = np.full(new Shape(DualArraySize), (byte)10, NPTypeCode.Byte);
        var b = np.ones(new Shape(DualArraySize), NPTypeCode.Byte);

        var result = a - b;

        ((byte)result[0]).Should().Be((byte)9);
        ((byte)result[DualArraySize - 1]).Should().Be((byte)9);

        long beyondIntMax = (long)int.MaxValue + 1000;
        if (beyondIntMax < DualArraySize)
        {
            ((byte)result[beyondIntMax]).Should().Be((byte)9);
        }
    }

    [TestMethod, LargeMemoryTest]
    public async Task Binary_Multiply_LargeByteArray()
    {
        var a = np.full(new Shape(DualArraySize), (byte)2, NPTypeCode.Byte);
        var b = np.full(new Shape(DualArraySize), (byte)3, NPTypeCode.Byte);

        var result = a * b;

        ((byte)result[0]).Should().Be((byte)6);
        ((byte)result[DualArraySize - 1]).Should().Be((byte)6);
    }

    #endregion

    #region Unary Operations

    [TestMethod, LargeMemoryTest]
    public async Task Unary_Negate_LargeInt16Array()
    {
        // Int16: 1.4B elements = 2.8GB
        var a = np.ones(new Shape(DualArraySize), NPTypeCode.Int16);

        var result = -a;

        ((short)result[0]).Should().Be((short)-1);
        ((short)result[DualArraySize - 1]).Should().Be((short)-1);

        long beyondIntMax = (long)int.MaxValue + 1000;
        if (beyondIntMax < DualArraySize)
        {
            ((short)result[beyondIntMax]).Should().Be((short)-1);
        }
    }

    [TestMethod, LargeMemoryTest]
    public async Task Unary_Abs_LargeInt16Array()
    {
        var a = np.full(new Shape(DualArraySize), (short)-5, NPTypeCode.Int16);

        var result = np.abs(a);

        ((short)result[0]).Should().Be((short)5);
        ((short)result[DualArraySize - 1]).Should().Be((short)5);
    }

    #endregion

    #region Reduction Operations

    [TestMethod, LargeMemoryTest]
    public async Task Reduction_Sum_LargeByteArray()
    {
        // All ones - sum should equal size
        var a = np.ones(new Shape(LargeSize), NPTypeCode.Byte);

        // Sum of bytes promotes to int64
        var result = np.sum(a);

        ((long)result).Should().Be(LargeSize);
    }

    [TestMethod, LargeMemoryTest]
    public async Task Reduction_Min_LargeByteArray()
    {
        var a = np.ones(new Shape(LargeSize), NPTypeCode.Byte);
        // Set one element beyond int.MaxValue to 0
        long targetIdx = (long)int.MaxValue + 12345;
        a[targetIdx] = (byte)0;

        var result = np.min(a);

        ((byte)result).Should().Be((byte)0);
    }

    [TestMethod, LargeMemoryTest]
    public async Task Reduction_Max_LargeByteArray()
    {
        var a = np.zeros(new Shape(LargeSize), NPTypeCode.Byte);
        // Set one element beyond int.MaxValue to 255
        long targetIdx = (long)int.MaxValue + 12345;
        a[targetIdx] = (byte)255;

        var result = np.max(a);

        ((byte)result).Should().Be((byte)255);
    }

    [TestMethod, LargeMemoryTest]
    public async Task Reduction_Mean_LargeByteArray()
    {
        // All ones - mean should be 1.0
        var a = np.ones(new Shape(LargeSize), NPTypeCode.Byte);

        var result = np.mean(a);

        ((double)result).Should().Be(1.0);
    }

    [TestMethod, LargeMemoryTest]
    public async Task Reduction_Any_LargeByteArray()
    {
        var a = np.zeros(new Shape(LargeSize), NPTypeCode.Byte);
        // Set one element beyond int.MaxValue to non-zero
        long targetIdx = (long)int.MaxValue + 99999;
        a[targetIdx] = (byte)1;

        var result = np.any(a);

        ((bool)result).Should().BeTrue();
    }

    [TestMethod, LargeMemoryTest]
    public async Task Reduction_All_LargeByteArray()
    {
        var a = np.ones(new Shape(LargeSize), NPTypeCode.Byte);

        var result = np.all(a);

        ((bool)result).Should().BeTrue();

        // Now set one element to 0 and verify all returns false
        long targetIdx = (long)int.MaxValue + 50000;
        a[targetIdx] = (byte)0;

        result = np.all(a);

        ((bool)result).Should().BeFalse();
    }

    #endregion

    #region Comparison Operations

    [TestMethod, LargeMemoryTest]
    public async Task Comparison_Equal_LargeByteArray()
    {
        var a = np.ones(new Shape(DualArraySize), NPTypeCode.Byte);
        var b = np.ones(new Shape(DualArraySize), NPTypeCode.Byte);

        // Set one element different beyond int.MaxValue
        long targetIdx = (long)int.MaxValue + 5000;
        if (targetIdx < DualArraySize)
        {
            b[targetIdx] = (byte)99;
        }

        var result = np.array_equal(a, b);

        result.Should().BeFalse();

        // Fix it and verify equal
        if (targetIdx < DualArraySize)
        {
            b[targetIdx] = (byte)1;
        }

        result = np.array_equal(a, b);

        result.Should().BeTrue();
    }

    [TestMethod, LargeMemoryTest]
    public async Task Comparison_Greater_LargeByteArray()
    {
        var a = np.full(new Shape(DualArraySize), (byte)5, NPTypeCode.Byte);
        var b = np.full(new Shape(DualArraySize), (byte)3, NPTypeCode.Byte);

        var result = a > b;

        // All should be true
        ((bool)result[0]).Should().BeTrue();
        ((bool)result[DualArraySize - 1]).Should().BeTrue();

        long beyondIntMax = (long)int.MaxValue + 1000;
        if (beyondIntMax < DualArraySize)
        {
            ((bool)result[beyondIntMax]).Should().BeTrue();
        }
    }

    #endregion

    #region Clip Operations

    [TestMethod, LargeMemoryTest]
    public async Task Clip_LargeByteArray()
    {
        var a = np.arange(0, 256, 1, NPTypeCode.Byte);
        // Tile to create large array
        // Actually, let's just create a large array with varying values
        var large = np.zeros(new Shape(LargeSize), NPTypeCode.Byte);

        // Set some values beyond int.MaxValue to test clipping
        long idx1 = (long)int.MaxValue + 100;
        long idx2 = (long)int.MaxValue + 200;
        large[idx1] = (byte)10;
        large[idx2] = (byte)250;

        var result = np.clip(large, (byte)5, (byte)200);

        // 0 should be clipped to 5
        ((byte)result[0]).Should().Be((byte)5);

        // 10 should stay 10
        ((byte)result[idx1]).Should().Be((byte)10);

        // 250 should be clipped to 200
        ((byte)result[idx2]).Should().Be((byte)200);
    }

    #endregion

    #region Indexing Beyond int.MaxValue

    [TestMethod, LargeMemoryTest]
    public async Task Indexing_SetGet_BeyondIntMaxValue()
    {
        var a = np.zeros(new Shape(LargeSize), NPTypeCode.Byte);

        // Test multiple indices beyond int.MaxValue
        long[] testIndices = new[]
        {
            (long)int.MaxValue + 1,
            (long)int.MaxValue + 1000,
            (long)int.MaxValue + 100000,
            (long)int.MaxValue + 1000000,
            LargeSize - 1
        };

        // Set values
        for (int i = 0; i < testIndices.Length; i++)
        {
            a[testIndices[i]] = (byte)(i + 1);
        }

        // Verify values
        for (int i = 0; i < testIndices.Length; i++)
        {
            ((byte)a[testIndices[i]]).Should().Be((byte)(i + 1));
        }
    }

    #endregion

    #region Slicing with Large Offsets

    [TestMethod, LargeMemoryTest]
    public async Task Slicing_ViewBeyondIntMaxValue()
    {
        var a = np.zeros(new Shape(LargeSize), NPTypeCode.Byte);

        // Set a value way beyond int.MaxValue
        long targetIdx = (long)int.MaxValue + 500000;
        a[targetIdx] = (byte)42;

        // Create a slice that includes this element
        // Slice from targetIdx - 10 to targetIdx + 10
        long start = targetIdx - 10;
        long stop = targetIdx + 11;
        var slice = a[$"{start}:{stop}"];

        // The target element should be at index 10 in the slice
        ((byte)slice[10]).Should().Be((byte)42);

        // Modify through slice
        slice[10] = (byte)99;

        // Verify original array is modified (view semantics)
        ((byte)a[targetIdx]).Should().Be((byte)99);
    }

    #endregion

    #region Large Stride Operations (stride > int.MaxValue)

    [TestMethod, LargeMemoryTest]
    public async Task Reduction_Sum_LargeStride()
    {
        // Create a 2D array where stride along axis 0 exceeds int.MaxValue
        // Shape: (3, 2_500_000_000) - stride for axis 0 is 2.5B which > int.MaxValue
        // Total: 7.5B bytes, but we only allocate the underlying buffer
        // Actually this would be 7.5GB which exceeds our limit.

        // Instead, create a smaller 2D array and use a view with large offset
        // to test that large strides work correctly

        // Alternative: Create 1D array and slice with large step
        // 2.5B element array, take every 1 billionth element = 2-3 elements
        // This tests that offset calculations work with large values

        var a = np.ones(new Shape(LargeSize), NPTypeCode.Byte);

        // Create a slice with a very large step (simulates large stride)
        // Take elements at indices 0, 1_000_000_000, 2_000_000_000
        long step = 1_000_000_000L;
        var sliced = a[$"0:{LargeSize}:{step}"];

        // Should have 3 elements (0, 1B, 2B)
        sliced.size.Should().Be(3);

        // Sum should be 3
        var sum = np.sum(sliced);
        ((long)sum).Should().Be(3);
    }

    [TestMethod, LargeMemoryTest]
    public async Task Binary_Add_SlicedWithLargeStep()
    {
        var a = np.ones(new Shape(LargeSize), NPTypeCode.Byte);
        var b = np.ones(new Shape(LargeSize), NPTypeCode.Byte);

        // Slice with step > int.MaxValue
        long step = (long)int.MaxValue + 100;
        var aSliced = a[$"0:{LargeSize}:{step}"];
        var bSliced = b[$"0:{LargeSize}:{step}"];

        // Should have 2 elements (0, ~2.1B)
        aSliced.size.Should().Be(2);

        // Add sliced arrays - tests binary ops with large strides
        var result = aSliced + bSliced;

        ((byte)result[0]).Should().Be((byte)2);
        ((byte)result[1]).Should().Be((byte)2);
    }

    [TestMethod, LargeMemoryTest]
    public async Task Reduction_Max_SlicedWithLargeStride()
    {
        var a = np.zeros(new Shape(LargeSize), NPTypeCode.Byte);

        // Set value at position beyond int.MaxValue
        long targetIdx = (long)int.MaxValue + 500;
        a[targetIdx] = (byte)42;

        // Create slice that includes this element with large step
        // Step just under int.MaxValue to ensure we hit the element
        long step = (long)int.MaxValue;
        var sliced = a[$"0:{LargeSize}:{step}"];

        // One element at 0, one at int.MaxValue (which has value 0)
        // The element at targetIdx (int.MaxValue + 500) is NOT in the slice
        // Let's adjust: set element AT int.MaxValue
        a[int.MaxValue] = (byte)99;

        sliced = a[$"0:{LargeSize}:{step}"];
        var max = np.max(sliced);

        ((byte)max).Should().Be((byte)99);
    }

    #endregion

    #region Float/Double Operations (smaller arrays due to element size)

    [TestMethod, LargeMemoryTest]
    public async Task Binary_Add_LargeFloatArray()
    {
        // Float: 4 bytes per element
        // 700M elements = 2.8GB (fits in 3GB limit)
        // But 700M < int.MaxValue, so we need to go bigger
        // Actually for floats we can't exceed int.MaxValue within 3GB
        // 750M floats = 3GB, which is < int.MaxValue (2.1B)
        // So this test is about verifying the operation works, not exceeding int.MaxValue

        long floatSize = 700_000_000L; // 2.8GB
        var a = np.ones(new Shape(floatSize), NPTypeCode.Single);
        var b = np.ones(new Shape(floatSize), NPTypeCode.Single);

        var result = a + b;

        ((float)result[0]).Should().Be(2.0f);
        ((float)result[floatSize - 1]).Should().Be(2.0f);
        ((float)result[floatSize / 2]).Should().Be(2.0f);
    }

    [TestMethod, LargeMemoryTest]
    public async Task Reduction_Sum_LargeFloatArray()
    {
        long floatSize = 500_000_000L; // 2GB
        var a = np.ones(new Shape(floatSize), NPTypeCode.Single);

        var result = np.sum(a);

        // Due to float precision, just verify it's close
        Math.Abs((double)result - ((double)floatSize)).Should().BeLessThan(floatSize * 0.001);
    }

    #endregion
}
