using System;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.LongIndexing;

/// <summary>
/// Master test for long indexing support (arrays with size > int.MaxValue).
///
/// This test exercises all np.* functions with arrays that have more than
/// int.MaxValue elements (specifically int.MaxValue * 1.1 = ~2.36 billion elements).
///
/// Memory requirements:
/// - Byte arrays: ~2.4 GB per array
/// - Int32 arrays: ~9.4 GB per array (avoid)
/// - Double arrays: ~18.8 GB per array (avoid)
///
/// We use byte arrays exclusively to minimize memory pressure while still
/// testing the long indexing code paths.
///
/// Tests are run sequentially within a single test method to ensure:
/// 1. Only one large array is allocated at a time
/// 2. GC can reclaim memory between operations
/// 3. We don't exceed available system memory
/// </summary>
[HighMemory]
[TestClass]
public class LongIndexingMasterTest
{
    /// <summary>
    /// Target size for long indexing tests: 10% larger than int.MaxValue.
    /// This forces all internal loops to use long indices instead of int.
    /// </summary>
    private const long LargeSize = (long)(int.MaxValue * 1.1); // ~2.36 billion

    /// <summary>
    /// Small size for comparison testing (ensures normal code paths still work).
    /// </summary>
    private const int SmallSize = 1000;

    /// <summary>
    /// Master test that exercises all np.* functions with large arrays.
    /// Runs sequentially to manage memory usage.
    /// </summary>
    [TestMethod]
    [TestCategory("Explicit")] // Requires ~8GB+ RAM. Run manually with: dotnet test --filter "TestCategory=LongIndexing"
    public void AllNpFunctions_WithLargeArrays()
    {
        var results = new System.Collections.Generic.List<(string Operation, bool Success, string Error, TimeSpan Duration)>();

        Console.WriteLine($"=== Long Indexing Master Test ===");
        Console.WriteLine($"Target size: {LargeSize:N0} elements ({LargeSize / 1e9:F2} billion)");
        Console.WriteLine($"int.MaxValue: {int.MaxValue:N0}");
        Console.WriteLine($"Size exceeds int.MaxValue by: {(LargeSize - int.MaxValue):N0} elements");
        Console.WriteLine();

        // ================================================================
        // CREATION FUNCTIONS
        // ================================================================
        Console.WriteLine("=== CREATION FUNCTIONS ===");

        // np.zeros (large)
        RunTest(results, "np.zeros(large)", () =>
        {
            var arr = np.zeros(new Shape(LargeSize), np.uint8);
            Assert.AreEqual(LargeSize, arr.size);
            Assert.AreEqual(0, arr.GetByte(0));
            Assert.AreEqual(0, arr.GetByte(LargeSize - 1));
            return arr;
        });

        // np.ones (large)
        RunTest(results, "np.ones(large)", () =>
        {
            var arr = np.ones(new Shape(LargeSize), np.uint8);
            Assert.AreEqual(LargeSize, arr.size);
            Assert.AreEqual(1, arr.GetByte(0));
            Assert.AreEqual(1, arr.GetByte(LargeSize - 1));
            return arr;
        });

        // np.full (large)
        RunTest(results, "np.full(large)", () =>
        {
            var arr = np.full(new Shape(LargeSize), (byte)42, np.uint8);
            Assert.AreEqual(LargeSize, arr.size);
            Assert.AreEqual(42, arr.GetByte(0));
            Assert.AreEqual(42, arr.GetByte(LargeSize - 1));
            return arr;
        });

        // np.empty (large)
        RunTest(results, "np.empty(large)", () =>
        {
            var arr = np.empty(new Shape(LargeSize), np.uint8);
            Assert.AreEqual(LargeSize, arr.size);
            // Can't check values - uninitialized memory
            return arr;
        });

        // np.arange (small - byte range limited)
        RunTest(results, "np.arange(small)", () =>
        {
            var arr = np.arange(0, 256, 1);  // Returns int64 by default
            Assert.AreEqual(256, arr.size);
            Assert.AreEqual(0L, arr.GetInt64(0));
            Assert.AreEqual(255L, arr.GetInt64(255));
            return arr;
        });

        // np.linspace (small)
        RunTest(results, "np.linspace(small)", () =>
        {
            var arr = np.linspace(0.0, 1.0, SmallSize);
            Assert.AreEqual(SmallSize, arr.size);
            return arr;
        });

        // np.zeros_like (large)
        RunTest(results, "np.zeros_like(large)", () =>
        {
            var original = np.ones(new Shape(LargeSize), np.uint8);
            var arr = np.zeros_like(original);
            Assert.AreEqual(LargeSize, arr.size);
            Assert.AreEqual(0, arr.GetByte(0));
            Assert.AreEqual(0, arr.GetByte(LargeSize - 1));
            return arr;
        });

        // np.ones_like (large)
        RunTest(results, "np.ones_like(large)", () =>
        {
            var original = np.zeros(new Shape(LargeSize), np.uint8);
            var arr = np.ones_like(original);
            Assert.AreEqual(LargeSize, arr.size);
            Assert.AreEqual(1, arr.GetByte(0));
            Assert.AreEqual(1, arr.GetByte(LargeSize - 1));
            return arr;
        });

        // np.full_like (large)
        RunTest(results, "np.full_like(large)", () =>
        {
            var original = np.zeros(new Shape(LargeSize), np.uint8);
            var arr = np.full_like(original, (byte)99);
            Assert.AreEqual(LargeSize, arr.size);
            Assert.AreEqual(99, arr.GetByte(0));
            Assert.AreEqual(99, arr.GetByte(LargeSize - 1));
            return arr;
        });

        // np.empty_like (large)
        RunTest(results, "np.empty_like(large)", () =>
        {
            var original = np.zeros(new Shape(LargeSize), np.uint8);
            var arr = np.empty_like(original);
            Assert.AreEqual(LargeSize, arr.size);
            return arr;
        });

        // np.copy (large)
        RunTest(results, "np.copy(large)", () =>
        {
            var original = np.full(new Shape(LargeSize), (byte)77, np.uint8);
            var arr = np.copy(original);
            Assert.AreEqual(LargeSize, arr.size);
            Assert.AreEqual(77, arr.GetByte(0));
            Assert.AreEqual(77, arr.GetByte(LargeSize - 1));
            return arr;
        });

        ForceGC();

        // ================================================================
        // SHAPE MANIPULATION FUNCTIONS
        // ================================================================
        Console.WriteLine("\n=== SHAPE MANIPULATION FUNCTIONS ===");

        // np.reshape (large)
        RunTest(results, "np.reshape(large)", () =>
        {
            // Use a size that's evenly divisible
            long rows = 1000000L;
            long cols = LargeSize / rows;
            long actualSize = rows * cols;

            var original = np.zeros(new Shape(actualSize), np.uint8);
            var arr = np.reshape(original, new Shape(rows, cols));
            Assert.AreEqual(actualSize, arr.size);
            Assert.AreEqual(2, arr.ndim);
            return arr;
        });

        // np.ravel (large)
        RunTest(results, "np.ravel(large)", () =>
        {
            var original = np.zeros(new Shape(LargeSize), np.uint8);
            var arr = np.ravel(original);
            Assert.AreEqual(LargeSize, arr.size);
            Assert.AreEqual(1, arr.ndim);
            return arr;
        });

        // np.flatten (large)
        RunTest(results, "np.flatten(large)", () =>
        {
            var original = np.zeros(new Shape(LargeSize), np.uint8);
            var arr = original.flatten();
            Assert.AreEqual(LargeSize, arr.size);
            Assert.AreEqual(1, arr.ndim);
            return arr;
        });

        // np.squeeze (small)
        RunTest(results, "np.squeeze(small)", () =>
        {
            var original = np.zeros(new Shape(1, SmallSize, 1));
            var arr = np.squeeze(original);
            Assert.AreEqual(SmallSize, arr.size);
            Assert.AreEqual(1, arr.ndim);
            return arr;
        });

        // np.expand_dims (large)
        RunTest(results, "np.expand_dims(large)", () =>
        {
            var original = np.zeros(new Shape(LargeSize), np.uint8);
            var arr = np.expand_dims(original, 0);
            Assert.AreEqual(LargeSize, arr.size);
            Assert.AreEqual(2, arr.ndim);
            return arr;
        });

        // np.transpose (2D - smaller due to memory)
        RunTest(results, "np.transpose(2D)", () =>
        {
            var original = np.zeros(new Shape(1000, 1000));
            var arr = np.transpose(original);
            Assert.AreEqual(1000000, arr.size);
            Assert.AreEqual(1000, arr.shape[0]);
            Assert.AreEqual(1000, arr.shape[1]);
            return arr;
        });

        // np.swapaxes (2D)
        RunTest(results, "np.swapaxes(2D)", () =>
        {
            var original = np.zeros(new Shape(1000, 1000));
            var arr = np.swapaxes(original, 0, 1);
            Assert.AreEqual(1000000, arr.size);
            return arr;
        });

        // np.atleast_1d (large)
        RunTest(results, "np.atleast_1d(large)", () =>
        {
            var original = np.zeros(new Shape(LargeSize), np.uint8);
            var arr = np.atleast_1d(original);
            Assert.AreEqual(LargeSize, arr.size);
            return arr;
        });

        // np.atleast_2d (large)
        RunTest(results, "np.atleast_2d(large)", () =>
        {
            var original = np.zeros(new Shape(LargeSize), np.uint8);
            var arr = np.atleast_2d(original);
            Assert.AreEqual(LargeSize, arr.size);
            Assert.AreEqual(2, arr.ndim);
            return arr;
        });

        // np.atleast_3d (large)
        RunTest(results, "np.atleast_3d(large)", () =>
        {
            var original = np.zeros(new Shape(LargeSize), np.uint8);
            var arr = np.atleast_3d(original);
            Assert.AreEqual(LargeSize, arr.size);
            Assert.AreEqual(3, arr.ndim);
            return arr;
        });

        ForceGC();

        // ================================================================
        // ARITHMETIC OPERATIONS (element-wise)
        // ================================================================
        Console.WriteLine("\n=== ARITHMETIC OPERATIONS ===");

        // np.add (large)
        RunTest(results, "np.add(large)", () =>
        {
            var a = np.ones(new Shape(LargeSize), np.uint8);
            var b = np.ones(new Shape(LargeSize), np.uint8);
            var arr = np.add(a, b);
            Assert.AreEqual(LargeSize, arr.size);
            Assert.AreEqual(2, arr.GetByte(0));
            Assert.AreEqual(2, arr.GetByte(LargeSize - 1));
            return arr;
        });

        ForceGC();

        // np.subtract (large)
        RunTest(results, "np.subtract(large)", () =>
        {
            var a = np.full(new Shape(LargeSize), (byte)10, np.uint8);
            var b = np.ones(new Shape(LargeSize), np.uint8);
            var arr = np.subtract(a, b);
            Assert.AreEqual(LargeSize, arr.size);
            Assert.AreEqual(9, arr.GetByte(0));
            return arr;
        });

        ForceGC();

        // np.multiply (large)
        RunTest(results, "np.multiply(large)", () =>
        {
            var a = np.full(new Shape(LargeSize), (byte)3, np.uint8);
            var b = np.full(new Shape(LargeSize), (byte)4, np.uint8);
            var arr = np.multiply(a, b);
            Assert.AreEqual(LargeSize, arr.size);
            Assert.AreEqual(12, arr.GetByte(0));
            return arr;
        });

        ForceGC();

        // np.square (large)
        RunTest(results, "np.square(large)", () =>
        {
            var a = np.full(new Shape(LargeSize), (byte)3, np.uint8);
            var arr = np.square(a);
            Assert.AreEqual(LargeSize, arr.size);
            Assert.AreEqual(9, arr.GetByte(0));
            return arr;
        });

        ForceGC();

        // np.clip (large)
        RunTest(results, "np.clip(large)", () =>
        {
            var a = np.full(new Shape(LargeSize), (byte)100, np.uint8);
            var arr = np.clip(a, 50, 75);
            Assert.AreEqual(LargeSize, arr.size);
            Assert.AreEqual(75, arr.GetByte(0)); // Clipped to max
            return arr;
        });

        ForceGC();

        // ================================================================
        // REDUCTION OPERATIONS
        // ================================================================
        Console.WriteLine("\n=== REDUCTION OPERATIONS ===");

        // np.sum (large)
        RunTest(results, "np.sum(large)", () =>
        {
            var a = np.ones(new Shape(LargeSize), np.uint8);
            var result = np.sum(a);
            Assert.AreEqual(0, result.ndim); // Scalar result
            return result;
        });

        ForceGC();

        // np.mean (large)
        RunTest(results, "np.mean(large)", () =>
        {
            var a = np.ones(new Shape(LargeSize), np.uint8);
            var result = np.mean(a);
            Assert.AreEqual(0, result.ndim);
            Assert.AreEqual(1.0, result.GetDouble(0), 0.001);
            return result;
        });

        ForceGC();

        // np.min (large)
        RunTest(results, "np.min(large)", () =>
        {
            var a = np.ones(new Shape(LargeSize), np.uint8);
            var result = np.min(a);
            Assert.AreEqual(0, result.ndim);
            Assert.AreEqual(1, result.GetByte(0));
            return result;
        });

        ForceGC();

        // np.max (large)
        RunTest(results, "np.max(large)", () =>
        {
            var a = np.ones(new Shape(LargeSize), np.uint8);
            var result = np.max(a);
            Assert.AreEqual(0, result.ndim);
            Assert.AreEqual(1, result.GetByte(0));
            return result;
        });

        ForceGC();

        // np.argmax (large)
        RunTest(results, "np.argmax(large)", () =>
        {
            var a = np.zeros(new Shape(LargeSize), np.uint8);
            a.SetByte((byte)255, LargeSize / 2);
            var result = np.argmax(a);
            Assert.AreEqual(LargeSize / 2, result);
            return (object)result;
        });

        ForceGC();

        // np.argmin (large)
        RunTest(results, "np.argmin(large)", () =>
        {
            var a = np.ones(new Shape(LargeSize), np.uint8);
            a.SetByte((byte)0, LargeSize / 3);
            var result = np.argmin(a);
            Assert.AreEqual(LargeSize / 3, result);
            return (object)result;
        });

        ForceGC();

        // np.all (large)
        RunTest(results, "np.all(large)", () =>
        {
            var a = np.ones(new Shape(LargeSize), np.uint8);
            var result = np.all(a);
            Assert.IsTrue(result);
            return (object)result;
        });

        ForceGC();

        // np.any (large)
        RunTest(results, "np.any(large)", () =>
        {
            var a = np.zeros(new Shape(LargeSize), np.uint8);
            a.SetByte((byte)1, LargeSize - 1);
            var result = np.any(a);
            Assert.IsTrue(result);
            return (object)result;
        });

        ForceGC();

        // np.count_nonzero (large)
        RunTest(results, "np.count_nonzero(large)", () =>
        {
            var a = np.zeros(new Shape(LargeSize), np.uint8);
            a.SetByte((byte)1, 0);
            a.SetByte((byte)1, LargeSize / 2);
            a.SetByte((byte)1, LargeSize - 1);
            var result = np.count_nonzero(a);
            Assert.AreEqual(3L, result);
            return (object)result;
        });

        ForceGC();

        // ================================================================
        // CUMULATIVE OPERATIONS
        // ================================================================
        Console.WriteLine("\n=== CUMULATIVE OPERATIONS ===");

        // np.cumsum (large)
        RunTest(results, "np.cumsum(large)", () =>
        {
            var a = np.ones(new Shape(LargeSize), np.uint8);
            var arr = np.cumsum(a);
            Assert.AreEqual(LargeSize, arr.size);
            return arr;
        });

        ForceGC();

        // ================================================================
        // COMPARISON OPERATIONS
        // ================================================================
        Console.WriteLine("\n=== COMPARISON OPERATIONS ===");

        // np.maximum (large)
        RunTest(results, "np.maximum(large)", () =>
        {
            var a = np.full(new Shape(LargeSize), (byte)5, np.uint8);
            var b = np.full(new Shape(LargeSize), (byte)10, np.uint8);
            var arr = np.maximum(a, b);
            Assert.AreEqual(LargeSize, arr.size);
            Assert.AreEqual(10, arr.GetByte(0));
            return arr;
        });

        ForceGC();

        // np.minimum (large)
        RunTest(results, "np.minimum(large)", () =>
        {
            var a = np.full(new Shape(LargeSize), (byte)5, np.uint8);
            var b = np.full(new Shape(LargeSize), (byte)10, np.uint8);
            var arr = np.minimum(a, b);
            Assert.AreEqual(LargeSize, arr.size);
            Assert.AreEqual(5, arr.GetByte(0));
            return arr;
        });

        ForceGC();

        // ================================================================
        // BITWISE OPERATIONS
        // ================================================================
        Console.WriteLine("\n=== BITWISE OPERATIONS ===");

        // Bitwise AND (large) - using & operator
        RunTest(results, "bitwise_and(large)", () =>
        {
            var a = np.full(new Shape(LargeSize), (byte)0b11110000, np.uint8);
            var b = np.full(new Shape(LargeSize), (byte)0b10101010, np.uint8);
            var arr = a & b;
            Assert.AreEqual(LargeSize, arr.size);
            Assert.AreEqual(0b10100000, arr.GetByte(0));
            return arr;
        });

        ForceGC();

        // Bitwise OR (large) - using | operator
        RunTest(results, "bitwise_or(large)", () =>
        {
            var a = np.full(new Shape(LargeSize), (byte)0b11110000, np.uint8);
            var b = np.full(new Shape(LargeSize), (byte)0b00001111, np.uint8);
            var arr = a | b;
            Assert.AreEqual(LargeSize, arr.size);
            Assert.AreEqual(0b11111111, arr.GetByte(0));
            return arr;
        });

        ForceGC();

        // Bitwise XOR (large) - using TensorEngine.BitwiseXor
        RunTest(results, "bitwise_xor(large)", () =>
        {
            var a = np.full(new Shape(LargeSize), (byte)0b11110000, np.uint8);
            var b = np.full(new Shape(LargeSize), (byte)0b10101010, np.uint8);
            var arr = a.TensorEngine.BitwiseXor(a, b);
            Assert.AreEqual(LargeSize, arr.size);
            Assert.AreEqual(0b01011010, arr.GetByte(0));
            return arr;
        });

        ForceGC();

        // np.invert (large)
        RunTest(results, "np.invert(large)", () =>
        {
            var a = np.full(new Shape(LargeSize), (byte)0b11110000, np.uint8);
            var arr = np.invert(a);
            Assert.AreEqual(LargeSize, arr.size);
            Assert.AreEqual(0b00001111, arr.GetByte(0));
            return arr;
        });

        ForceGC();

        // np.left_shift (large)
        RunTest(results, "np.left_shift(large)", () =>
        {
            var a = np.full(new Shape(LargeSize), (byte)0b00000001, np.uint8);
            var arr = np.left_shift(a, 4);
            Assert.AreEqual(LargeSize, arr.size);
            Assert.AreEqual(0b00010000, arr.GetByte(0));
            return arr;
        });

        ForceGC();

        // np.right_shift (large)
        RunTest(results, "np.right_shift(large)", () =>
        {
            var a = np.full(new Shape(LargeSize), (byte)0b11110000, np.uint8);
            var arr = np.right_shift(a, 4);
            Assert.AreEqual(LargeSize, arr.size);
            Assert.AreEqual(0b00001111, arr.GetByte(0));
            return arr;
        });

        ForceGC();

        // ================================================================
        // INDEXING/SLICING
        // ================================================================
        Console.WriteLine("\n=== INDEXING/SLICING ===");

        // Basic indexing (large)
        RunTest(results, "indexing[large_index]", () =>
        {
            var a = np.full(new Shape(LargeSize), (byte)42, np.uint8);
            var value = a.GetByte(LargeSize - 1);
            Assert.AreEqual(42, value);
            return (object)value;
        });

        ForceGC();

        // Slicing with start/stop (large)
        RunTest(results, "slicing[start:stop](large)", () =>
        {
            var a = np.full(new Shape(LargeSize), (byte)99, np.uint8);
            long start = LargeSize - 100;
            long stop = LargeSize;
            var slice = a[$"{start}:{stop}"];
            Assert.AreEqual(100, slice.size);
            Assert.AreEqual(99, slice.GetByte(0));
            return slice;
        });

        ForceGC();

        // Setting values at large indices
        RunTest(results, "set_value[large_index]", () =>
        {
            var a = np.zeros(new Shape(LargeSize), np.uint8);
            a.SetByte((byte)123, LargeSize - 1);
            Assert.AreEqual(123, a.GetByte(LargeSize - 1));
            return a;
        });

        ForceGC();

        // ================================================================
        // BROADCASTING (limited testing due to memory)
        // ================================================================
        Console.WriteLine("\n=== BROADCASTING ===");

        // np.broadcast_to (large - from small source)
        RunTest(results, "np.broadcast_to(large)", () =>
        {
            var a = np.full(new Shape(1L), (byte)77, np.uint8);
            var arr = np.broadcast_to(a, new Shape(LargeSize));
            Assert.AreEqual(LargeSize, arr.size);
            Assert.AreEqual(77, arr.GetByte(0));
            Assert.AreEqual(77, arr.GetByte(LargeSize - 1));
            return arr;
        });

        ForceGC();

        // ================================================================
        // ROLL (large)
        // ================================================================
        Console.WriteLine("\n=== ROLL ===");

        RunTest(results, "np.roll(large)", () =>
        {
            var a = np.zeros(new Shape(LargeSize), np.uint8);
            a.SetByte((byte)1, 0);
            var arr = np.roll(a, LargeSize / 2);
            Assert.AreEqual(LargeSize, arr.size);
            Assert.AreEqual(1, arr.GetByte(LargeSize / 2));
            return arr;
        });

        ForceGC();

        // ================================================================
        // SUMMARY
        // ================================================================
        Console.WriteLine("\n" + new string('=', 60));
        Console.WriteLine("=== SUMMARY ===");
        Console.WriteLine(new string('=', 60));

        int passed = results.Count(r => r.Success);
        int failed = results.Count(r => !r.Success);
        var totalTime = TimeSpan.FromTicks(results.Sum(r => r.Duration.Ticks));

        Console.WriteLine($"\nTotal: {results.Count} tests");
        Console.WriteLine($"Passed: {passed}");
        Console.WriteLine($"Failed: {failed}");
        Console.WriteLine($"Total time: {totalTime.TotalSeconds:F2}s");

        if (failed > 0)
        {
            Console.WriteLine("\n=== FAILED TESTS ===");
            foreach (var (op, success, error, duration) in results.Where(r => !r.Success))
            {
                Console.WriteLine($"  {op}: {error}");
            }
        }

        Assert.AreEqual(0, failed, $"{failed} tests failed. See output for details.");
    }

    /// <summary>
    /// Runs a single test operation with timing and error handling.
    /// </summary>
    private void RunTest<T>(
        System.Collections.Generic.List<(string Operation, bool Success, string Error, TimeSpan Duration)> results,
        string operationName,
        Func<T> testAction)
    {
        var sw = Stopwatch.StartNew();
        try
        {
            Console.Write($"  {operationName}... ");
            testAction();
            sw.Stop();
            Console.WriteLine($"OK ({sw.ElapsedMilliseconds}ms)");
            results.Add((operationName, true, null, sw.Elapsed));
        }
        catch (Exception ex)
        {
            sw.Stop();
            Console.WriteLine($"FAILED ({sw.ElapsedMilliseconds}ms)");
            Console.WriteLine($"    Error: {ex.Message}");
            results.Add((operationName, false, ex.Message, sw.Elapsed));
        }
        finally
        {
            // Force GC after each test to reclaim memory
            ForceGC();
        }
    }

    /// <summary>
    /// Forces garbage collection to reclaim memory between large array operations.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void ForceGC()
    {
        GC.Collect(2, GCCollectionMode.Forced, true, true);
        GC.WaitForPendingFinalizers();
        GC.Collect(2, GCCollectionMode.Forced, true, true);
    }
}
