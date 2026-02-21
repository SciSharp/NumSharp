using System.Runtime.InteropServices;
using BenchmarkDotNet.Attributes;
using NumSharp.Benchmark.GraphEngine.Infrastructure;

namespace NumSharp.Benchmark.GraphEngine.Benchmarks.Allocation;

/// <summary>
/// Micro-benchmarks comparing allocation primitives:
/// - Marshal.AllocHGlobal (current)
/// - NativeMemory.Alloc (proposed)
/// - NativeMemory.AlignedAlloc (for SIMD)
///
/// These benchmarks inform issue #528: NativeMemory modernization.
/// </summary>
[BenchmarkCategory("Allocation", "Micro")]
public class AllocationMicroBenchmarks : BenchmarkBase
{
    /// <summary>
    /// Byte counts to allocate (matching typical NumSharp array sizes).
    /// </summary>
    [Params(64, 1_000, 100_000, 10_000_000)]
    public int Bytes { get; set; }

    // ========================================================================
    // Allocation Only (no free) - measures allocation overhead
    // ========================================================================

    [Benchmark(Baseline = true, Description = "Marshal.AllocHGlobal")]
    [BenchmarkCategory("AllocOnly")]
    public nint MarshalAllocHGlobal() => Marshal.AllocHGlobal(Bytes);

    [Benchmark(Description = "NativeMemory.Alloc")]
    [BenchmarkCategory("AllocOnly")]
    public unsafe void* NativeMemoryAlloc() => NativeMemory.Alloc((nuint)Bytes);

    [Benchmark(Description = "NativeMemory.AlignedAlloc(32)")]
    [BenchmarkCategory("AllocOnly")]
    public unsafe void* NativeMemoryAlignedAlloc32() => NativeMemory.AlignedAlloc((nuint)Bytes, 32);

    [Benchmark(Description = "NativeMemory.AlignedAlloc(64)")]
    [BenchmarkCategory("AllocOnly")]
    public unsafe void* NativeMemoryAlignedAlloc64() => NativeMemory.AlignedAlloc((nuint)Bytes, 64);

    [Benchmark(Description = "NativeMemory.AllocZeroed")]
    [BenchmarkCategory("AllocOnly")]
    public unsafe void* NativeMemoryAllocZeroed() => NativeMemory.AllocZeroed((nuint)Bytes);

    // ========================================================================
    // Round-Trip (alloc + free) - measures full lifecycle
    // ========================================================================

    [Benchmark(Description = "Marshal alloc+free")]
    [BenchmarkCategory("RoundTrip")]
    public void MarshalRoundTrip()
    {
        var ptr = Marshal.AllocHGlobal(Bytes);
        Marshal.FreeHGlobal(ptr);
    }

    [Benchmark(Description = "NativeMemory alloc+free")]
    [BenchmarkCategory("RoundTrip")]
    public unsafe void NativeMemoryRoundTrip()
    {
        var ptr = NativeMemory.Alloc((nuint)Bytes);
        NativeMemory.Free(ptr);
    }

    [Benchmark(Description = "NativeMemory aligned alloc+free")]
    [BenchmarkCategory("RoundTrip")]
    public unsafe void NativeMemoryAlignedRoundTrip()
    {
        var ptr = NativeMemory.AlignedAlloc((nuint)Bytes, 32);
        NativeMemory.AlignedFree(ptr);
    }

    [Benchmark(Description = "NativeMemory zeroed alloc+free")]
    [BenchmarkCategory("RoundTrip")]
    public unsafe void NativeMemoryZeroedRoundTrip()
    {
        var ptr = NativeMemory.AllocZeroed((nuint)Bytes);
        NativeMemory.Free(ptr);
    }
}
