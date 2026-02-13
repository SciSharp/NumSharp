using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NumSharp.Benchmark.Exploration.Infrastructure;

namespace NumSharp.Benchmark.Exploration.Isolated;

/// <summary>
/// Test memory access patterns and cache behavior.
/// Investigates when gather/scatter and prefetch help.
/// </summary>
public static unsafe class MemoryPatterns
{
    private const string Suite = "MemoryPatterns";

    /// <summary>
    /// Run all memory pattern tests.
    /// </summary>
    public static List<BenchResult> RunAll(bool quick = false)
    {
        var results = new List<BenchResult>();
        var warmup = quick ? 2 : BenchFramework.DefaultWarmup;
        var measure = quick ? 5 : BenchFramework.DefaultMeasure;

        var sizes = quick
            ? new[] { 100_000, 1_000_000 }
            : new[] { 10_000, 100_000, 1_000_000, 10_000_000 };

        BenchFramework.PrintHeader($"{Suite}: Memory Access Patterns & Cache Behavior");

        // Test 1: Strided access patterns
        foreach (var size in sizes)
        {
            results.AddRange(TestStridedAccess(size, warmup, measure));
        }

        // Test 2: Gather vs Copy-then-SIMD
        foreach (var size in sizes)
        {
            results.AddRange(TestGatherVsCopy(size, warmup, measure));
        }

        OutputFormatters.PrintSummary(results);
        return results;
    }

    /// <summary>
    /// Test strided access performance degradation.
    /// </summary>
    public static List<BenchResult> TestStridedAccess(int totalElements, int warmup, int measure)
    {
        var results = new List<BenchResult>();
        int elementBytes = 8;

        BenchFramework.PrintDivider($"Strided Access: {totalElements:N0} total elements");

        var strides = new[] { 1, 2, 4, 8, 16, 32, 64 };

        foreach (var stride in strides)
        {
            // Allocate enough to hold strided access
            int actualSize = totalElements * stride;
            if (actualSize > 100_000_000) continue; // Skip if too large

            var src = SimdImplementations.AllocateAligned<double>(actualSize);
            var dst = SimdImplementations.AllocateAligned<double>(totalElements);
            SimdImplementations.FillRandom(src, actualSize, 42);

            // Measure strided read + sequential write
            var result = BenchFramework.Run(
                () => StridedRead(src, dst, totalElements, stride),
                "Strided", $"Stride{stride}", "float64", totalElements, elementBytes, warmup, measure, Suite);
            results.Add(result);

            // Calculate effective bandwidth
            var seqBandwidth = results.FirstOrDefault(r => r.Strategy == "Stride1")?.GBps ?? result.GBps;
            var efficiency = result.GBps / seqBandwidth;
            Console.WriteLine($"  Stride {stride,2}: {result.MeanUs,10:F2} us | {result.GBps,8:F2} GB/s | Efficiency: {efficiency * 100:F1}%");

            SimdImplementations.FreeAligned(src);
            SimdImplementations.FreeAligned(dst);
        }

        return results;
    }

    /// <summary>
    /// Compare Gather intrinsic vs Copy-to-contiguous-then-SIMD.
    /// </summary>
    public static List<BenchResult> TestGatherVsCopy(int totalElements, int warmup, int measure)
    {
        var results = new List<BenchResult>();
        int elementBytes = 8;

        BenchFramework.PrintDivider($"Gather vs Copy: {totalElements:N0} elements");

        // Test with stride=2 (every other element)
        int stride = 2;
        int actualSize = totalElements * stride;

        var src = SimdImplementations.AllocateAligned<double>(actualSize);
        var other = SimdImplementations.AllocateAligned<double>(totalElements);
        var result = SimdImplementations.AllocateAligned<double>(totalElements);
        var tempContiguous = SimdImplementations.AllocateAligned<double>(totalElements);
        SimdImplementations.FillRandom(src, actualSize, 42);
        SimdImplementations.FillRandom(other, totalElements, 43);

        // Method 1: Scalar strided loop
        var scalar = BenchFramework.Run(
            () => StridedAdd_Scalar(src, other, result, totalElements, stride),
            "GatherVsCopy", "1_Scalar", "float64", totalElements, elementBytes, warmup, measure, Suite);
        results.Add(scalar);
        BenchFramework.PrintResult(scalar);

        // Method 2: Copy to contiguous, then SIMD
        var copySimd = BenchFramework.Run(
            () =>
            {
                // Copy strided to contiguous
                for (int i = 0; i < totalElements; i++)
                {
                    tempContiguous[i] = src[i * stride];
                }
                // SIMD add
                SimdImplementations.AddFull_Float64(tempContiguous, other, result, totalElements);
            },
            "GatherVsCopy", "2_CopySimd", "float64", totalElements, elementBytes, warmup, measure, Suite);
        copySimd = copySimd with { SpeedupVsBaseline = scalar.MeanUs / copySimd.MeanUs };
        results.Add(copySimd);
        BenchFramework.PrintResult(copySimd);

        // Method 3: AVX2 Gather (if supported)
        if (Avx2.IsSupported && totalElements >= 4)
        {
            var gather = BenchFramework.Run(
                () => StridedAdd_Gather(src, other, result, totalElements, stride),
                "GatherVsCopy", "3_Gather", "float64", totalElements, elementBytes, warmup, measure, Suite);
            gather = gather with { SpeedupVsBaseline = scalar.MeanUs / gather.MeanUs };
            results.Add(gather);
            BenchFramework.PrintResult(gather);
        }
        else
        {
            Console.WriteLine("  3_Gather: AVX2 not supported");
        }

        SimdImplementations.FreeAligned(src);
        SimdImplementations.FreeAligned(other);
        SimdImplementations.FreeAligned(result);
        SimdImplementations.FreeAligned(tempContiguous);

        return results;
    }

    #region Implementation

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void StridedRead(double* src, double* dst, int count, int stride)
    {
        for (int i = 0; i < count; i++)
        {
            dst[i] = src[i * stride];
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void StridedAdd_Scalar(double* strided, double* contiguous, double* result, int count, int stride)
    {
        for (int i = 0; i < count; i++)
        {
            result[i] = strided[i * stride] + contiguous[i];
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void StridedAdd_Gather(double* strided, double* contiguous, double* result, int count, int stride)
    {
        // Using AVX2 GatherVector256 for double
        // Each gather loads 4 doubles at specified indices

        int i = 0;
        int vecSize = Vector256<double>.Count; // 4

        if (Avx2.IsSupported && count >= vecSize)
        {
            // Create index vector: [0, stride, 2*stride, 3*stride] * sizeof(double)
            var indices = Vector256.Create(0, stride, stride * 2, stride * 3);

            int vectorCount = count - vecSize + 1;
            for (; i < vectorCount; i += vecSize)
            {
                // Gather from strided source
                var gathered = Avx2.GatherVector256(strided + i * stride, indices, 8);

                // Load contiguous
                var cont = Avx.LoadVector256(contiguous + i);

                // Add
                var sum = Avx.Add(gathered, cont);

                // Store
                Avx.Store(result + i, sum);
            }
        }

        // Scalar tail
        for (; i < count; i++)
        {
            result[i] = strided[i * stride] + contiguous[i];
        }
    }

    #endregion

    /// <summary>
    /// Test cache line effects.
    /// </summary>
    public static List<BenchResult> TestCacheLines(int warmup, int measure)
    {
        var results = new List<BenchResult>();

        BenchFramework.PrintDivider("Cache Line Effects");

        // Test accessing elements at different cache line boundaries
        // Typical cache line is 64 bytes = 8 doubles
        var sizes = new[] { 8, 16, 32, 64, 128, 256, 512 }; // Elements per "chunk"

        int totalIterations = 1_000_000;

        foreach (var chunkSize in sizes)
        {
            int totalElements = chunkSize * 1000; // 1000 chunks
            var data = SimdImplementations.AllocateAligned<double>(totalElements);
            SimdImplementations.FillRandom(data, totalElements, 42);

            double sum = 0;
            var result = BenchFramework.Run(
                () =>
                {
                    sum = 0;
                    for (int chunk = 0; chunk < totalElements / chunkSize; chunk++)
                    {
                        // Access first element of each chunk
                        sum += data[chunk * chunkSize];
                    }
                },
                "CacheLine", $"Chunk{chunkSize}", "float64", totalIterations / chunkSize, 8, warmup, measure, Suite);
            results.Add(result);
            Console.WriteLine($"  Chunk size {chunkSize,4}: {result.MeanUs,10:F2} us | sum={sum:F2}");

            SimdImplementations.FreeAligned(data);
        }

        return results;
    }

    /// <summary>
    /// Test non-temporal (streaming) stores.
    /// </summary>
    public static List<BenchResult> TestNonTemporalStores(int size, int warmup, int measure)
    {
        var results = new List<BenchResult>();
        int elementBytes = 8;

        BenchFramework.PrintDivider($"Non-Temporal Stores: {size:N0} elements");

        var src = SimdImplementations.AllocateAligned<double>(size);
        var dst = SimdImplementations.AllocateAligned<double>(size);
        SimdImplementations.FillRandom(src, size, 42);

        // Normal stores
        var normal = BenchFramework.Run(
            () => CopyNormal(src, dst, size),
            "NTStore", "Normal", "float64", size, elementBytes, warmup, measure, Suite);
        results.Add(normal);
        BenchFramework.PrintResult(normal);

        // Non-temporal stores (streaming)
        if (Avx2.IsSupported)
        {
            var nt = BenchFramework.Run(
                () => CopyNonTemporal(src, dst, size),
                "NTStore", "NonTemp", "float64", size, elementBytes, warmup, measure, Suite);
            nt = nt with { SpeedupVsBaseline = normal.MeanUs / nt.MeanUs };
            results.Add(nt);
            BenchFramework.PrintResult(nt);
        }

        SimdImplementations.FreeAligned(src);
        SimdImplementations.FreeAligned(dst);

        return results;
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void CopyNormal(double* src, double* dst, int count)
    {
        int i = 0;
        if (Avx2.IsSupported && count >= Vector256<double>.Count)
        {
            int vectorCount = count - Vector256<double>.Count + 1;
            for (; i < vectorCount; i += Vector256<double>.Count)
            {
                var v = Avx.LoadVector256(src + i);
                Avx.Store(dst + i, v);
            }
        }
        for (; i < count; i++)
        {
            dst[i] = src[i];
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void CopyNonTemporal(double* src, double* dst, int count)
    {
        int i = 0;
        if (Avx2.IsSupported && count >= Vector256<double>.Count)
        {
            int vectorCount = count - Vector256<double>.Count + 1;
            for (; i < vectorCount; i += Vector256<double>.Count)
            {
                var v = Avx.LoadVector256(src + i);
                Avx.StoreAlignedNonTemporal(dst + i, v);
            }
        }
        for (; i < count; i++)
        {
            dst[i] = src[i];
        }
    }
}
