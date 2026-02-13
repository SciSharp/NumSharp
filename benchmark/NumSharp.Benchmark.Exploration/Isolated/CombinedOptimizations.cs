using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NumSharp.Benchmark.Exploration.Infrastructure;

namespace NumSharp.Benchmark.Exploration.Isolated;

/// <summary>
/// Test combinations of single-threaded optimizations (NO PARALLEL).
/// C1: Scalar baseline
/// C2: SIMD only
/// C3: SIMD + ArrayPool (chained ops)
/// C4: In-place (+=)
/// C5: SIMD + In-place
/// C6: SIMD + Buffer reuse
/// </summary>
public static unsafe class CombinedOptimizations
{
    private const string Suite = "CombinedOptimizations";

    /// <summary>
    /// Run all combined optimization tests.
    /// </summary>
    public static List<BenchResult> RunAll(bool quick = false)
    {
        var results = new List<BenchResult>();
        var warmup = quick ? 2 : BenchFramework.DefaultWarmup;
        var measure = quick ? 5 : BenchFramework.DefaultMeasure;

        var sizes = quick
            ? new[] { 100_000, 10_000_000 }
            : new[] { 1_000, 100_000, 1_000_000, 10_000_000 };

        BenchFramework.PrintHeader($"{Suite}: Single-Threaded Optimization Combinations");

        // Test 1: Single operation optimizations
        foreach (var size in sizes)
        {
            BenchFramework.PrintDivider($"Single Operation: Size = {size:N0}");
            results.AddRange(TestSingleOperation(size, warmup, measure));
        }

        // Test 2: Chained operations (a + b + c + d)
        foreach (var size in sizes)
        {
            BenchFramework.PrintDivider($"Chained Operations (a+b+c+d): Size = {size:N0}");
            results.AddRange(TestChainedOperations(size, warmup, measure));
        }

        // Test 3: In-place operations (+=)
        foreach (var size in sizes)
        {
            BenchFramework.PrintDivider($"In-Place Operations (+=): Size = {size:N0}");
            results.AddRange(TestInPlaceOperations(size, warmup, measure));
        }

        OutputFormatters.PrintSummary(results);
        return results;
    }

    /// <summary>
    /// Test single operation optimizations.
    /// </summary>
    public static List<BenchResult> TestSingleOperation(int size, int warmup, int measure)
    {
        var results = new List<BenchResult>();
        int elementBytes = 8;

        var lhs = SimdImplementations.AllocateAligned<double>(size);
        var rhs = SimdImplementations.AllocateAligned<double>(size);
        var result = SimdImplementations.AllocateAligned<double>(size);
        SimdImplementations.FillRandom(lhs, size, 42);
        SimdImplementations.FillRandom(rhs, size, 43);

        // C1: Scalar baseline
        var c1 = BenchFramework.Run(
            () => SimdImplementations.AddScalarLoop_Float64(lhs, rhs, result, size),
            "Single", "C1_Scalar", "float64", size, elementBytes, warmup, measure, Suite);
        results.Add(c1);
        BenchFramework.PrintResult(c1);

        // C2: SIMD only
        var c2 = BenchFramework.Run(
            () => SimdImplementations.AddFull_Float64(lhs, rhs, result, size),
            "Single", "C2_SIMD", "float64", size, elementBytes, warmup, measure, Suite);
        c2 = c2 with { SpeedupVsBaseline = c1.MeanUs / c2.MeanUs };
        results.Add(c2);
        BenchFramework.PrintResult(c2);

        SimdImplementations.FreeAligned(lhs);
        SimdImplementations.FreeAligned(rhs);
        SimdImplementations.FreeAligned(result);

        return results;
    }

    /// <summary>
    /// Test chained operations: a + b + c + d
    /// Compare naive (4 allocations) vs pooled (1 allocation reused).
    /// </summary>
    public static List<BenchResult> TestChainedOperations(int size, int warmup, int measure)
    {
        var results = new List<BenchResult>();
        int elementBytes = 8;

        var a = SimdImplementations.AllocateAligned<double>(size);
        var b = SimdImplementations.AllocateAligned<double>(size);
        var c = SimdImplementations.AllocateAligned<double>(size);
        var d = SimdImplementations.AllocateAligned<double>(size);
        SimdImplementations.FillRandom(a, size, 42);
        SimdImplementations.FillRandom(b, size, 43);
        SimdImplementations.FillRandom(c, size, 44);
        SimdImplementations.FillRandom(d, size, 45);

        // Baseline: Scalar with allocations
        var baseline = BenchFramework.RunWithSetup(
            () => { }, // No setup needed
            () =>
            {
                var t1 = new double[size];
                var t2 = new double[size];
                var result = new double[size];

                fixed (double* pt1 = t1, pt2 = t2, pr = result)
                {
                    SimdImplementations.AddScalarLoop_Float64(a, b, pt1, size);
                    SimdImplementations.AddScalarLoop_Float64(pt1, c, pt2, size);
                    SimdImplementations.AddScalarLoop_Float64(pt2, d, pr, size);
                }
            },
            "Chained", "0_ScalarAlloc", "float64", size, elementBytes, warmup, measure, Suite);
        results.Add(baseline);
        BenchFramework.PrintResult(baseline);

        // C1: SIMD with new allocations each time
        var c1 = BenchFramework.RunWithSetup(
            () => { },
            () =>
            {
                var t1 = new double[size];
                var t2 = new double[size];
                var result = new double[size];

                fixed (double* pt1 = t1, pt2 = t2, pr = result)
                {
                    SimdImplementations.AddFull_Float64(a, b, pt1, size);
                    SimdImplementations.AddFull_Float64(pt1, c, pt2, size);
                    SimdImplementations.AddFull_Float64(pt2, d, pr, size);
                }
            },
            "Chained", "C1_SimdAlloc", "float64", size, elementBytes, warmup, measure, Suite);
        c1 = c1 with { SpeedupVsBaseline = baseline.MeanUs / c1.MeanUs };
        results.Add(c1);
        BenchFramework.PrintResult(c1);

        // C2: SIMD with pre-allocated buffers (simulates NumSharp output allocation)
        var t1Pre = SimdImplementations.AllocateAligned<double>(size);
        var t2Pre = SimdImplementations.AllocateAligned<double>(size);
        var resultPre = SimdImplementations.AllocateAligned<double>(size);

        var c2 = BenchFramework.Run(
            () =>
            {
                SimdImplementations.AddFull_Float64(a, b, t1Pre, size);
                SimdImplementations.AddFull_Float64(t1Pre, c, t2Pre, size);
                SimdImplementations.AddFull_Float64(t2Pre, d, resultPre, size);
            },
            "Chained", "C2_SimdPreAlloc", "float64", size, elementBytes, warmup, measure, Suite);
        c2 = c2 with { SpeedupVsBaseline = baseline.MeanUs / c2.MeanUs };
        results.Add(c2);
        BenchFramework.PrintResult(c2);

        // C3: SIMD with ArrayPool
        var c3 = BenchFramework.Run(
            () =>
            {
                var pool = ArrayPool<double>.Shared;
                var t1Arr = pool.Rent(size);
                var t2Arr = pool.Rent(size);
                var resultArr = pool.Rent(size);

                fixed (double* pt1 = t1Arr, pt2 = t2Arr, pr = resultArr)
                {
                    SimdImplementations.AddFull_Float64(a, b, pt1, size);
                    SimdImplementations.AddFull_Float64(pt1, c, pt2, size);
                    SimdImplementations.AddFull_Float64(pt2, d, pr, size);
                }

                pool.Return(t1Arr);
                pool.Return(t2Arr);
                pool.Return(resultArr);
            },
            "Chained", "C3_SimdPool", "float64", size, elementBytes, warmup, measure, Suite);
        c3 = c3 with { SpeedupVsBaseline = baseline.MeanUs / c3.MeanUs };
        results.Add(c3);
        BenchFramework.PrintResult(c3);

        // C4: SIMD with single buffer reuse (ping-pong)
        var buf1 = SimdImplementations.AllocateAligned<double>(size);
        var buf2 = SimdImplementations.AllocateAligned<double>(size);

        var c4 = BenchFramework.Run(
            () =>
            {
                SimdImplementations.AddFull_Float64(a, b, buf1, size);
                SimdImplementations.AddFull_Float64(buf1, c, buf2, size);
                SimdImplementations.AddFull_Float64(buf2, d, buf1, size);
                // Result is in buf1
            },
            "Chained", "C4_PingPong", "float64", size, elementBytes, warmup, measure, Suite);
        c4 = c4 with { SpeedupVsBaseline = baseline.MeanUs / c4.MeanUs };
        results.Add(c4);
        BenchFramework.PrintResult(c4);

        SimdImplementations.FreeAligned(a);
        SimdImplementations.FreeAligned(b);
        SimdImplementations.FreeAligned(c);
        SimdImplementations.FreeAligned(d);
        SimdImplementations.FreeAligned(t1Pre);
        SimdImplementations.FreeAligned(t2Pre);
        SimdImplementations.FreeAligned(resultPre);
        SimdImplementations.FreeAligned(buf1);
        SimdImplementations.FreeAligned(buf2);

        return results;
    }

    /// <summary>
    /// Test in-place operations (a += b).
    /// </summary>
    public static List<BenchResult> TestInPlaceOperations(int size, int warmup, int measure)
    {
        var results = new List<BenchResult>();
        int elementBytes = 8;

        var a = SimdImplementations.AllocateAligned<double>(size);
        var b = SimdImplementations.AllocateAligned<double>(size);
        var result = SimdImplementations.AllocateAligned<double>(size);
        SimdImplementations.FillRandom(b, size, 43);

        // Baseline: Out-of-place scalar
        var baseline = BenchFramework.RunWithSetup(
            () => SimdImplementations.FillRandom(a, size, 42),
            () => SimdImplementations.AddScalarLoop_Float64(a, b, result, size),
            "InPlace", "0_OutOfPlace", "float64", size, elementBytes, warmup, measure, Suite);
        results.Add(baseline);
        BenchFramework.PrintResult(baseline);

        // C1: In-place scalar (a += b)
        var c1 = BenchFramework.RunWithSetup(
            () => SimdImplementations.FillRandom(a, size, 42),
            () => AddInPlace_Scalar(a, b, size),
            "InPlace", "C1_InPlace", "float64", size, elementBytes, warmup, measure, Suite);
        c1 = c1 with { SpeedupVsBaseline = baseline.MeanUs / c1.MeanUs };
        results.Add(c1);
        BenchFramework.PrintResult(c1);

        // C2: Out-of-place SIMD
        var c2 = BenchFramework.RunWithSetup(
            () => SimdImplementations.FillRandom(a, size, 42),
            () => SimdImplementations.AddFull_Float64(a, b, result, size),
            "InPlace", "C2_SimdOut", "float64", size, elementBytes, warmup, measure, Suite);
        c2 = c2 with { SpeedupVsBaseline = baseline.MeanUs / c2.MeanUs };
        results.Add(c2);
        BenchFramework.PrintResult(c2);

        // C3: In-place SIMD (a += b)
        var c3 = BenchFramework.RunWithSetup(
            () => SimdImplementations.FillRandom(a, size, 42),
            () => AddInPlace_Simd(a, b, size),
            "InPlace", "C3_SimdIn", "float64", size, elementBytes, warmup, measure, Suite);
        c3 = c3 with { SpeedupVsBaseline = baseline.MeanUs / c3.MeanUs };
        results.Add(c3);
        BenchFramework.PrintResult(c3);

        SimdImplementations.FreeAligned(a);
        SimdImplementations.FreeAligned(b);
        SimdImplementations.FreeAligned(result);

        return results;
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void AddInPlace_Scalar(double* a, double* b, int count)
    {
        for (int i = 0; i < count; i++)
        {
            a[i] += b[i];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AddInPlace_Simd(double* a, double* b, int count)
    {
        // In-place: a += b, so result goes back to a
        SimdImplementations.AddFull_Float64(a, b, a, count);
    }

    /// <summary>
    /// Test ArrayPool overhead specifically.
    /// </summary>
    public static List<BenchResult> TestPoolOverhead(int size, int iterations, int warmup, int measure)
    {
        var results = new List<BenchResult>();

        BenchFramework.PrintDivider($"ArrayPool Overhead (size={size:N0}, iterations={iterations})");

        // Baseline: New allocation each time
        var baseline = BenchFramework.Run(
            () =>
            {
                for (int i = 0; i < iterations; i++)
                {
                    var arr = new double[size];
                    // Touch array to ensure allocation
                    arr[0] = 1.0;
                    arr[size - 1] = 1.0;
                }
            },
            "PoolOverhead", "New", "float64", size, 8, warmup, measure, Suite);
        results.Add(baseline);
        BenchFramework.PrintResult(baseline);

        // ArrayPool
        var pool = ArrayPool<double>.Shared;
        var pooled = BenchFramework.Run(
            () =>
            {
                for (int i = 0; i < iterations; i++)
                {
                    var arr = pool.Rent(size);
                    arr[0] = 1.0;
                    arr[size - 1] = 1.0;
                    pool.Return(arr);
                }
            },
            "PoolOverhead", "ArrayPool", "float64", size, 8, warmup, measure, Suite);
        pooled = pooled with { SpeedupVsBaseline = baseline.MeanUs / pooled.MeanUs };
        results.Add(pooled);
        BenchFramework.PrintResult(pooled);

        return results;
    }
}
