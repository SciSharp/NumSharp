using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using NumSharp;
using NumSharp.Backends.Kernels;

Console.WriteLine("=== SIMD Reduction Benchmark: Baseline vs Optimized ===");
Console.WriteLine($"Vector256 Accelerated: {Vector256.IsHardwareAccelerated}");
Console.WriteLine($"Vector256<double>.Count: {Vector256<double>.Count}");
Console.WriteLine();

// Helper to get pointer from NDArray
unsafe double* GetDoublePtr(NDArray arr)
{
    var slice = arr.GetData<double>();
    return (double*)slice.Address;
}

unsafe float* GetFloatPtr(NDArray arr)
{
    var slice = arr.GetData<float>();
    return (float*)slice.Address;
}

unsafe long* GetLongPtr(NDArray arr)
{
    var slice = arr.GetData<long>();
    return (long*)slice.Address;
}

// Test array sizes
int[] sizes = { 1000, 10_000, 100_000, 1_000_000, 10_000_000 };

foreach (var size in sizes)
{
    Console.WriteLine($"--- Array Size: {size:N0} doubles ---");

    // Allocate aligned memory
    var arr = np.random.rand(size).astype(np.float64);

    unsafe
    {
        double* ptr = GetDoublePtr(arr);

        // Warmup
        for (int w = 0; w < 5; w++)
        {
            _ = SimdReductionOptimized.SumDouble_Baseline(ptr, size);
            _ = SimdReductionOptimized.SumDouble_Optimized(ptr, size);
        }

        // Verify correctness
        double baseline = SimdReductionOptimized.SumDouble_Baseline(ptr, size);
        double optimized = SimdReductionOptimized.SumDouble_Optimized(ptr, size);
        double expected = np.sum(arr).GetDouble();

        Console.WriteLine($"  np.sum result:    {expected:F6}");
        Console.WriteLine($"  Baseline result:  {baseline:F6}");
        Console.WriteLine($"  Optimized result: {optimized:F6}");

        // Benchmark
        int iterations = size < 100_000 ? 10000 : (size < 1_000_000 ? 1000 : 100);

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
        {
            _ = SimdReductionOptimized.SumDouble_Baseline(ptr, size);
        }
        sw.Stop();
        double baselineMs = sw.Elapsed.TotalMilliseconds / iterations;

        sw.Restart();
        for (int i = 0; i < iterations; i++)
        {
            _ = SimdReductionOptimized.SumDouble_Optimized(ptr, size);
        }
        sw.Stop();
        double optimizedMs = sw.Elapsed.TotalMilliseconds / iterations;

        double speedup = baselineMs / optimizedMs;
        Console.WriteLine($"  Baseline:  {baselineMs:F4} ms");
        Console.WriteLine($"  Optimized: {optimizedMs:F4} ms");
        Console.WriteLine($"  Speedup:   {speedup:F2}x");
        Console.WriteLine();
    }
}

// Also test Max
Console.WriteLine("=== Max Double Benchmark ===");
int maxSize = 1_000_000;
var maxArr = np.random.rand(maxSize).astype(np.float64);

unsafe
{
    double* ptr = GetDoublePtr(maxArr);

    // Warmup
    for (int w = 0; w < 5; w++)
    {
        _ = SimdReductionOptimized.MaxDouble_Baseline(ptr, maxSize);
        _ = SimdReductionOptimized.MaxDouble_Optimized(ptr, maxSize);
    }

    int iterations = 1000;

    var sw = Stopwatch.StartNew();
    for (int i = 0; i < iterations; i++)
    {
        _ = SimdReductionOptimized.MaxDouble_Baseline(ptr, maxSize);
    }
    sw.Stop();
    double baselineMs = sw.Elapsed.TotalMilliseconds / iterations;

    sw.Restart();
    for (int i = 0; i < iterations; i++)
    {
        _ = SimdReductionOptimized.MaxDouble_Optimized(ptr, maxSize);
    }
    sw.Stop();
    double optimizedMs = sw.Elapsed.TotalMilliseconds / iterations;

    Console.WriteLine($"Max {maxSize:N0} doubles:");
    Console.WriteLine($"  Baseline:  {baselineMs:F4} ms");
    Console.WriteLine($"  Optimized: {optimizedMs:F4} ms");
    Console.WriteLine($"  Speedup:   {baselineMs / optimizedMs:F2}x");
}

Console.WriteLine();
Console.WriteLine("=== Int64 Sum Benchmark ===");
int longSize = 1_000_000;
var longArr = np.arange(longSize).astype(np.int64);

unsafe
{
    long* ptr = GetLongPtr(longArr);

    // Warmup
    for (int w = 0; w < 5; w++)
    {
        _ = SimdReductionOptimized.SumInt64_Baseline(ptr, longSize);
        _ = SimdReductionOptimized.SumInt64_Optimized(ptr, longSize);
    }

    int iterations = 1000;

    var sw = Stopwatch.StartNew();
    for (int i = 0; i < iterations; i++)
    {
        _ = SimdReductionOptimized.SumInt64_Baseline(ptr, longSize);
    }
    sw.Stop();
    double baselineMs = sw.Elapsed.TotalMilliseconds / iterations;

    sw.Restart();
    for (int i = 0; i < iterations; i++)
    {
        _ = SimdReductionOptimized.SumInt64_Optimized(ptr, longSize);
    }
    sw.Stop();
    double optimizedMs = sw.Elapsed.TotalMilliseconds / iterations;

    Console.WriteLine($"Sum {longSize:N0} longs:");
    Console.WriteLine($"  Baseline:  {baselineMs:F4} ms");
    Console.WriteLine($"  Optimized: {optimizedMs:F4} ms");
    Console.WriteLine($"  Speedup:   {baselineMs / optimizedMs:F2}x");
}

// Compare with actual NumSharp np.sum
Console.WriteLine();
Console.WriteLine("=== NumSharp np.sum vs Optimized SIMD ===");
int npSize = 1_000_000;
var npArr = np.random.rand(npSize).astype(np.float64);

unsafe
{
    double* ptr = GetDoublePtr(npArr);

    // Warmup
    for (int w = 0; w < 5; w++)
    {
        _ = np.sum(npArr);
        _ = SimdReductionOptimized.SumDouble_Optimized(ptr, npSize);
    }

    int iterations = 1000;

    var sw = Stopwatch.StartNew();
    for (int i = 0; i < iterations; i++)
    {
        _ = np.sum(npArr);
    }
    sw.Stop();
    double npSumMs = sw.Elapsed.TotalMilliseconds / iterations;

    sw.Restart();
    for (int i = 0; i < iterations; i++)
    {
        _ = SimdReductionOptimized.SumDouble_Optimized(ptr, npSize);
    }
    sw.Stop();
    double optimizedMs = sw.Elapsed.TotalMilliseconds / iterations;

    Console.WriteLine($"Sum {npSize:N0} doubles:");
    Console.WriteLine($"  np.sum (current): {npSumMs:F4} ms");
    Console.WriteLine($"  Optimized SIMD:   {optimizedMs:F4} ms");
    Console.WriteLine($"  Potential speedup if integrated: {npSumMs / optimizedMs:F2}x");
}

// 4x vs 8x unrolling comparison
Console.WriteLine();
Console.WriteLine("=== 4x vs 8x Unrolling Comparison ===");
foreach (var size in new[] { 10_000, 100_000, 1_000_000 })
{
    var arr = np.random.rand(size).astype(np.float64);

    unsafe
    {
        double* ptr = GetDoublePtr(arr);

        // Warmup
        for (int w = 0; w < 10; w++)
        {
            _ = SimdReductionOptimized.SumDouble_Baseline(ptr, size);
            _ = SimdReductionOptimized.SumDouble_Optimized(ptr, size);
            _ = SimdReductionOptimized.SumDouble_8x(ptr, size);
        }

        int iterations = size < 100_000 ? 10000 : (size < 1_000_000 ? 1000 : 500);

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
        {
            _ = SimdReductionOptimized.SumDouble_Baseline(ptr, size);
        }
        sw.Stop();
        double baselineMs = sw.Elapsed.TotalMilliseconds / iterations;

        sw.Restart();
        for (int i = 0; i < iterations; i++)
        {
            _ = SimdReductionOptimized.SumDouble_Optimized(ptr, size);
        }
        sw.Stop();
        double x4Ms = sw.Elapsed.TotalMilliseconds / iterations;

        sw.Restart();
        for (int i = 0; i < iterations; i++)
        {
            _ = SimdReductionOptimized.SumDouble_8x(ptr, size);
        }
        sw.Stop();
        double x8Ms = sw.Elapsed.TotalMilliseconds / iterations;

        Console.WriteLine($"Sum {size:N0} doubles:");
        Console.WriteLine($"  Baseline (1x): {baselineMs:F4}ms");
        Console.WriteLine($"  4x unroll:     {x4Ms:F4}ms  ({baselineMs/x4Ms:F2}x vs baseline)");
        Console.WriteLine($"  8x unroll:     {x8Ms:F4}ms  ({baselineMs/x8Ms:F2}x vs baseline, {x4Ms/x8Ms:F2}x vs 4x)");
    }
}
