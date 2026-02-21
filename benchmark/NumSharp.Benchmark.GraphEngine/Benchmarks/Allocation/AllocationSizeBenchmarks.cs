using System.Runtime.InteropServices;
using BenchmarkDotNet.Attributes;
using NumSharp.Benchmark.GraphEngine.Infrastructure;

namespace NumSharp.Benchmark.GraphEngine.Benchmarks.Allocation;

/// <summary>
/// Benchmarks allocation performance across a wide range of sizes.
/// Helps identify:
/// - Small allocation overhead (where dispatch cost dominates)
/// - Crossover points between allocation strategies
/// - Large allocation behavior (OS page allocation)
/// </summary>
[BenchmarkCategory("Allocation", "SizeScaling")]
public class AllocationSizeBenchmarks : BenchmarkBase
{
    /// <summary>
    /// Byte counts covering common array sizes:
    /// - 64B: Single cache line
    /// - 256B: Small scalar arrays
    /// - 1KB: Tiny arrays
    /// - 4KB: Page size boundary
    /// - 64KB: L1 cache size
    /// - 256KB: L2 cache size
    /// - 1MB: L3 cache boundary
    /// - 4MB: Large arrays
    /// - 16MB: Very large arrays
    /// - 64MB: Memory-bound
    /// </summary>
    [Params(64, 256, 1024, 4096, 65536, 262144, 1048576, 4194304, 16777216, 67108864)]
    public int Bytes { get; set; }

    // ========================================================================
    // Marshal vs NativeMemory comparison across sizes
    // ========================================================================

    [Benchmark(Baseline = true, Description = "Marshal.AllocHGlobal")]
    [BenchmarkCategory("Compare")]
    public void MarshalAlloc()
    {
        var ptr = Marshal.AllocHGlobal(Bytes);
        Marshal.FreeHGlobal(ptr);
    }

    [Benchmark(Description = "NativeMemory.Alloc")]
    [BenchmarkCategory("Compare")]
    public unsafe void NativeAlloc()
    {
        var ptr = NativeMemory.Alloc((nuint)Bytes);
        NativeMemory.Free(ptr);
    }

    [Benchmark(Description = "NativeMemory.AlignedAlloc(32)")]
    [BenchmarkCategory("Compare")]
    public unsafe void NativeAlignedAlloc()
    {
        var ptr = NativeMemory.AlignedAlloc((nuint)Bytes, 32);
        NativeMemory.AlignedFree(ptr);
    }

    [Benchmark(Description = "NativeMemory.AllocZeroed")]
    [BenchmarkCategory("Compare")]
    public unsafe void NativeAllocZeroed()
    {
        var ptr = NativeMemory.AllocZeroed((nuint)Bytes);
        NativeMemory.Free(ptr);
    }
}
