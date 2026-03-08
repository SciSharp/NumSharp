#:project ../src/NumSharp.Core
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true

using NumSharp;
using System;

Console.WriteLine("=== LogicalNot Battle Test ===\n");

int passed = 0, failed = 0;

void Test(string name, Func<bool> test)
{
    try
    {
        if (test())
        {
            Console.WriteLine($"[PASS] {name}");
            passed++;
        }
        else
        {
            Console.WriteLine($"[FAIL] {name}");
            failed++;
        }
    }
    catch (Exception ex)
    {
        Console.WriteLine($"[ERROR] {name}: {ex.GetType().Name}: {ex.Message}");
        failed++;
    }
}

// ============================================================================
// TEST 1: Boolean array input -> should return Boolean dtype
// ============================================================================
Console.WriteLine("\n--- Test Group 1: Boolean array input ---");

Test("Boolean array returns Boolean dtype", () =>
{
    var arr = np.array(new[] { true, false, true });
    var result = np.logical_not(arr);
    return result.dtype == typeof(bool);
});

Test("Boolean array values are inverted", () =>
{
    var arr = np.array(new[] { true, false, true, false });
    var result = np.logical_not(arr);
    var expected = new[] { false, true, false, true };
    for (int i = 0; i < expected.Length; i++)
    {
        if (result.GetBoolean(i) != expected[i]) return false;
    }
    return true;
});

Test("All True -> All False", () =>
{
    var arr = np.array(new[] { true, true, true });
    var result = np.logical_not(arr);
    return !result.GetBoolean(0) && !result.GetBoolean(1) && !result.GetBoolean(2);
});

Test("All False -> All True", () =>
{
    var arr = np.array(new[] { false, false, false });
    var result = np.logical_not(arr);
    return result.GetBoolean(0) && result.GetBoolean(1) && result.GetBoolean(2);
});

// ============================================================================
// TEST 2: Integer array input -> should return Boolean dtype
// ============================================================================
Console.WriteLine("\n--- Test Group 2: Integer array input ---");

Test("Int32 array returns Boolean dtype", () =>
{
    var arr = np.array(new[] { 0, 1, 2, -1 });
    var result = np.logical_not(arr);
    return result.dtype == typeof(bool);
});

Test("Int32 values: 0 becomes True, non-zero becomes False", () =>
{
    var arr = np.array(new[] { 0, 1, 2, -1, 100, 0 });
    var result = np.logical_not(arr);
    // 0 -> True, nonzero -> False
    var expected = new[] { true, false, false, false, false, true };
    for (int i = 0; i < expected.Length; i++)
    {
        if (result.GetBoolean(i) != expected[i])
        {
            Console.WriteLine($"    Mismatch at index {i}: expected {expected[i]}, got {result.GetBoolean(i)}");
            return false;
        }
    }
    return true;
});

Test("Int64 array returns Boolean dtype", () =>
{
    var arr = np.array(new long[] { 0, 1, -1, long.MaxValue });
    var result = np.logical_not(arr);
    return result.dtype == typeof(bool);
});

Test("Byte array returns Boolean dtype", () =>
{
    var arr = np.array(new byte[] { 0, 1, 255 });
    var result = np.logical_not(arr);
    return result.dtype == typeof(bool);
});

// ============================================================================
// TEST 3: Float array input -> should return Boolean dtype
// ============================================================================
Console.WriteLine("\n--- Test Group 3: Float array input ---");

Test("Float32 array returns Boolean dtype", () =>
{
    var arr = np.array(new float[] { 0.0f, 1.0f, -1.5f, 0.0001f });
    var result = np.logical_not(arr);
    return result.dtype == typeof(bool);
});

Test("Float32 values: 0.0 becomes True, non-zero becomes False", () =>
{
    var arr = np.array(new float[] { 0.0f, 1.0f, -1.5f, 0.0001f, 0.0f });
    var result = np.logical_not(arr);
    var expected = new[] { true, false, false, false, true };
    for (int i = 0; i < expected.Length; i++)
    {
        if (result.GetBoolean(i) != expected[i])
        {
            Console.WriteLine($"    Mismatch at index {i}: expected {expected[i]}, got {result.GetBoolean(i)}");
            return false;
        }
    }
    return true;
});

Test("Float64 array returns Boolean dtype", () =>
{
    var arr = np.array(new double[] { 0.0, 1.0, -1.5, 0.0001 });
    var result = np.logical_not(arr);
    return result.dtype == typeof(bool);
});

// ============================================================================
// TEST 4: Empty arrays
// ============================================================================
Console.WriteLine("\n--- Test Group 4: Empty arrays ---");

Test("Empty Boolean array returns empty Boolean array", () =>
{
    var arr = np.array(Array.Empty<bool>());
    var result = np.logical_not(arr);
    return result.dtype == typeof(bool) && result.size == 0;
});

// NOTE: These tests fail due to a SEPARATE BUG in the comparison operators.
// Empty array == 0 returns scalar True instead of empty boolean array.
// This is not a bug in LogicalNot itself, but in NDArray.Equals operator.
Test("Empty Int32 array returns empty Boolean array (KNOWN BUG: comparison ops)", () =>
{
    var arr = np.array(Array.Empty<int>());
    var result = np.logical_not(arr);
    // Expected: size == 0, Actual: size == 1 due to (arr == 0) bug
    Console.WriteLine($"    Empty Int32: dtype={result.dtype}, size={result.size} (expected 0)");
    return result.dtype == typeof(bool) && result.size == 0;
});

Test("Empty Float64 array returns empty Boolean array (KNOWN BUG: comparison ops)", () =>
{
    var arr = np.array(Array.Empty<double>());
    var result = np.logical_not(arr);
    // Expected: size == 0, Actual: size == 1 due to (arr == 0) bug
    Console.WriteLine($"    Empty Float64: dtype={result.dtype}, size={result.size} (expected 0)");
    return result.dtype == typeof(bool) && result.size == 0;
});

// ============================================================================
// TEST 5: Scalar inputs
// ============================================================================
Console.WriteLine("\n--- Test Group 5: Scalar inputs ---");

Test("Scalar True returns False with Boolean dtype", () =>
{
    var arr = NDArray.Scalar(true);
    var result = np.logical_not(arr);
    return result.dtype == typeof(bool) && result.GetBoolean() == false;
});

Test("Scalar False returns True with Boolean dtype", () =>
{
    var arr = NDArray.Scalar(false);
    var result = np.logical_not(arr);
    return result.dtype == typeof(bool) && result.GetBoolean() == true;
});

Test("Scalar 0 (int) returns True with Boolean dtype", () =>
{
    var arr = NDArray.Scalar(0);
    var result = np.logical_not(arr);
    return result.dtype == typeof(bool) && result.GetBoolean() == true;
});

Test("Scalar 1 (int) returns False with Boolean dtype", () =>
{
    var arr = NDArray.Scalar(1);
    var result = np.logical_not(arr);
    return result.dtype == typeof(bool) && result.GetBoolean() == false;
});

Test("Scalar 0.0 (double) returns True with Boolean dtype", () =>
{
    var arr = NDArray.Scalar(0.0);
    var result = np.logical_not(arr);
    return result.dtype == typeof(bool) && result.GetBoolean() == true;
});

Test("Scalar 1.5 (double) returns False with Boolean dtype", () =>
{
    var arr = NDArray.Scalar(1.5);
    var result = np.logical_not(arr);
    return result.dtype == typeof(bool) && result.GetBoolean() == false;
});

// ============================================================================
// TEST 6: Multi-dimensional arrays
// ============================================================================
Console.WriteLine("\n--- Test Group 6: Multi-dimensional arrays ---");

Test("2D Boolean array returns 2D Boolean result with correct shape", () =>
{
    var arr = np.array(new[,] { { true, false }, { false, true } });
    var result = np.logical_not(arr);
    return result.dtype == typeof(bool) &&
           result.ndim == 2 &&
           result.shape[0] == 2 &&
           result.shape[1] == 2;
});

Test("2D Boolean array values are inverted correctly", () =>
{
    var arr = np.array(new[,] { { true, false }, { false, true } });
    var result = np.logical_not(arr);
    return result.GetBoolean(0, 0) == false &&
           result.GetBoolean(0, 1) == true &&
           result.GetBoolean(1, 0) == true &&
           result.GetBoolean(1, 1) == false;
});

Test("2D Int32 array returns Boolean dtype", () =>
{
    var arr = np.array(new[,] { { 0, 1 }, { 2, 0 } });
    var result = np.logical_not(arr);
    return result.dtype == typeof(bool);
});

Test("3D Boolean array preserves shape", () =>
{
    var arr = np.zeros(new Shape(2, 3, 4), NPTypeCode.Boolean);
    var result = np.logical_not(arr);
    return result.dtype == typeof(bool) &&
           result.ndim == 3 &&
           result.shape[0] == 2 &&
           result.shape[1] == 3 &&
           result.shape[2] == 4;
});

// ============================================================================
// TEST 7: Non-contiguous arrays (slices)
// ============================================================================
Console.WriteLine("\n--- Test Group 7: Non-contiguous arrays (slices) ---");

Test("Sliced Boolean array returns correct dtype", () =>
{
    var arr = np.array(new[] { true, false, true, false, true, false });
    var sliced = arr["::2"]; // Every other element: true, true, true
    var result = np.logical_not(sliced);
    return result.dtype == typeof(bool);
});

Test("Sliced Boolean array values are correct", () =>
{
    var arr = np.array(new[] { true, false, true, false, true, false });
    var sliced = arr["::2"]; // Every other element: true, true, true
    var result = np.logical_not(sliced);
    // All true becomes all false
    return result.size == 3 &&
           result.GetBoolean(0) == false &&
           result.GetBoolean(1) == false &&
           result.GetBoolean(2) == false;
});

Test("Sliced Int32 array returns Boolean dtype", () =>
{
    var arr = np.array(new[] { 0, 1, 0, 2, 0, 3 });
    var sliced = arr["::2"]; // Every other element: 0, 0, 0
    var result = np.logical_not(sliced);
    return result.dtype == typeof(bool);
});

Test("Sliced Int32 array values are correct", () =>
{
    var arr = np.array(new[] { 0, 1, 0, 2, 0, 3 });
    var sliced = arr["::2"]; // Every other element: 0, 0, 0
    var result = np.logical_not(sliced);
    // All zeros become all true
    return result.size == 3 &&
           result.GetBoolean(0) == true &&
           result.GetBoolean(1) == true &&
           result.GetBoolean(2) == true;
});

// ============================================================================
// TEST 8: Double application (logical_not(logical_not(x)) == x for bool)
// ============================================================================
Console.WriteLine("\n--- Test Group 8: Double application ---");

Test("logical_not(logical_not(bool_array)) equals original", () =>
{
    var arr = np.array(new[] { true, false, true, false });
    var result = np.logical_not(np.logical_not(arr));
    for (int i = 0; i < arr.size; i++)
    {
        if (arr.GetBoolean(i) != result.GetBoolean(i)) return false;
    }
    return true;
});

// ============================================================================
// TEST 9: All supported dtypes
// ============================================================================
Console.WriteLine("\n--- Test Group 9: All supported dtypes return Boolean ---");

Test("UInt16 array returns Boolean", () =>
{
    var arr = np.array(new ushort[] { 0, 1, 65535 });
    var result = np.logical_not(arr);
    return result.dtype == typeof(bool);
});

Test("Int16 array returns Boolean", () =>
{
    var arr = np.array(new short[] { 0, 1, -1, 32767 });
    var result = np.logical_not(arr);
    return result.dtype == typeof(bool);
});

Test("UInt32 array returns Boolean", () =>
{
    var arr = np.array(new uint[] { 0, 1, uint.MaxValue });
    var result = np.logical_not(arr);
    return result.dtype == typeof(bool);
});

Test("UInt64 array returns Boolean", () =>
{
    var arr = np.array(new ulong[] { 0, 1, ulong.MaxValue });
    var result = np.logical_not(arr);
    return result.dtype == typeof(bool);
});

// ============================================================================
// TEST 10: np.invert on boolean arrays (NumPy's ~ operator equivalent)
// ============================================================================
Console.WriteLine("\n--- Test Group 10: np.invert on boolean arrays ---");

Test("np.invert(boolean_array) returns Boolean dtype", () =>
{
    var arr = np.array(new[] { true, false, true });
    var result = np.invert(arr);
    return result.dtype == typeof(bool);
});

Test("np.invert(boolean_array) inverts values correctly", () =>
{
    var arr = np.array(new[] { true, false, true, false });
    var result = np.invert(arr);
    return result.GetBoolean(0) == false &&
           result.GetBoolean(1) == true &&
           result.GetBoolean(2) == false &&
           result.GetBoolean(3) == true;
});

// ============================================================================
// TEST 10b: ! operator on boolean arrays
// ============================================================================
Console.WriteLine("\n--- Test Group 10b: ! operator on arrays ---");

Test("!boolean_array returns Boolean dtype", () =>
{
    var arr = np.array(new[] { true, false, true });
    var result = !arr;
    return result.dtype == typeof(bool);
});

Test("!boolean_array inverts values correctly", () =>
{
    var arr = np.array(new[] { true, false, true, false });
    var result = !arr;
    return result.GetBoolean(0) == false &&
           result.GetBoolean(1) == true &&
           result.GetBoolean(2) == false &&
           result.GetBoolean(3) == true;
});

Test("!int_array returns Boolean dtype", () =>
{
    var arr = np.array(new[] { 0, 1, 2 });
    var result = !arr;
    return result.dtype == typeof(bool);
});

// ============================================================================
// TEST 11: Large arrays (for SIMD edge cases)
// ============================================================================
Console.WriteLine("\n--- Test Group 11: Large arrays ---");

Test("Large Boolean array (1000 elements) returns Boolean dtype", () =>
{
    var arr = np.ones(new Shape(1000), NPTypeCode.Boolean);
    var result = np.logical_not(arr);
    return result.dtype == typeof(bool) && result.size == 1000;
});

Test("Large Boolean array values are all inverted", () =>
{
    var arr = np.ones(new Shape(1000), NPTypeCode.Boolean);
    var result = np.logical_not(arr);
    // All ones (true) should become all zeros (false)
    for (int i = 0; i < 1000; i++)
    {
        if (result.GetBoolean(i) != false) return false;
    }
    return true;
});

Test("Large Int32 array returns Boolean dtype", () =>
{
    var arr = np.arange(1000);
    var result = np.logical_not(arr);
    return result.dtype == typeof(bool);
});

// ============================================================================
// Summary
// ============================================================================
Console.WriteLine($"\n=== Summary ===");
Console.WriteLine($"Passed: {passed}");
Console.WriteLine($"Failed: {failed}");
Console.WriteLine($"Total:  {passed + failed}");

Environment.Exit(failed > 0 ? 1 : 0);
