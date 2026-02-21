using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using BenchmarkDotNet.Attributes;
using NumSharp.Benchmark.GraphEngine.Infrastructure;

namespace NumSharp.Benchmark.GraphEngine.Benchmarks.Allocation;

/// <summary>
/// Benchmarks comparing zero-initialization approaches for np.zeros:
/// - Current: Alloc + Unsafe.InitBlock
/// - Proposed: NativeMemory.AllocZeroed (OS may have fast paths)
/// - Alternative: Alloc + Span.Clear
///
/// These benchmarks inform the np.zeros optimization decision in issue #528.
/// </summary>
[BenchmarkCategory("Allocation", "ZeroInit")]
public class ZeroInitBenchmarks : BenchmarkBase
{
    /// <summary>
    /// Element counts (int32 = 4 bytes each).
    /// Small: 4KB, Medium: 400KB, Large: 40MB
    /// </summary>
    [Params(1_000, 100_000, 10_000_000)]
    public override int N { get; set; }

    private int _bytes;

    [GlobalSetup]
    public void Setup()
    {
        _bytes = N * sizeof(int);
    }

    // ========================================================================
    // Current NumSharp approach: Alloc + Unsafe.InitBlock
    // Used in UnmanagedMemoryBlock when filling with zeros
    // ========================================================================

    [Benchmark(Baseline = true, Description = "Alloc + InitBlock (current)")]
    [BenchmarkCategory("ZeroInit")]
    public unsafe void AllocThenInitBlock()
    {
        var ptr = NativeMemory.Alloc((nuint)_bytes);
        Unsafe.InitBlock(ptr, 0, (uint)_bytes);
        NativeMemory.Free(ptr);
    }

    // ========================================================================
    // Proposed: NativeMemory.AllocZeroed
    // OS may use page zeroing, copy-on-write, or hardware optimizations
    // ========================================================================

    [Benchmark(Description = "AllocZeroed (proposed)")]
    [BenchmarkCategory("ZeroInit")]
    public unsafe void AllocZeroed()
    {
        var ptr = NativeMemory.AllocZeroed((nuint)_bytes);
        NativeMemory.Free(ptr);
    }

    // ========================================================================
    // Alternative: Alloc + Span.Clear
    // Span.Clear may use SIMD internally
    // ========================================================================

    [Benchmark(Description = "Alloc + Span.Clear")]
    [BenchmarkCategory("ZeroInit")]
    public unsafe void AllocThenSpanClear()
    {
        var ptr = NativeMemory.Alloc((nuint)_bytes);
        new Span<byte>(ptr, _bytes).Clear();
        NativeMemory.Free(ptr);
    }

    // ========================================================================
    // Aligned variants - test if alignment helps zero-init
    // ========================================================================

    [Benchmark(Description = "AlignedAlloc(32) + InitBlock")]
    [BenchmarkCategory("ZeroInit", "Aligned")]
    public unsafe void AlignedAllocThenInitBlock()
    {
        var ptr = NativeMemory.AlignedAlloc((nuint)_bytes, 32);
        Unsafe.InitBlock(ptr, 0, (uint)_bytes);
        NativeMemory.AlignedFree(ptr);
    }

    [Benchmark(Description = "AlignedAlloc(32) + Span.Clear")]
    [BenchmarkCategory("ZeroInit", "Aligned")]
    public unsafe void AlignedAllocThenSpanClear()
    {
        var ptr = NativeMemory.AlignedAlloc((nuint)_bytes, 32);
        new Span<byte>(ptr, _bytes).Clear();
        NativeMemory.AlignedFree(ptr);
    }

    // ========================================================================
    // Marshal.AllocHGlobal baseline for comparison
    // ========================================================================

    [Benchmark(Description = "Marshal.AllocHGlobal + InitBlock")]
    [BenchmarkCategory("ZeroInit", "Marshal")]
    public unsafe void MarshalAllocThenInitBlock()
    {
        var ptr = Marshal.AllocHGlobal(_bytes);
        Unsafe.InitBlock((void*)ptr, 0, (uint)_bytes);
        Marshal.FreeHGlobal(ptr);
    }
}
