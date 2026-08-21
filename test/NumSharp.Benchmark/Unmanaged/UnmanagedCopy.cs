using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using NumSharp.Backends.Unmanaged;
using NumSharp.Unmanaged.Memory;

namespace NumSharp.Benchmark.Unmanaged
{
    // Revived 2026-08 (was fully commented out). Compares four ways to allocate + copy a small
    // unmanaged block (length = 100 ints). The outer loop repeats one alloc+copy; OperationsPerInvoke
    // = iterations, so the reported Mean is the cost of ONE alloc+copy (and Allocated is per-op).
    //   * copy  (baseline) — new block + Span<T>.CopyTo
    //   * copy1            — new block + Buffer.MemoryCopy
    //   * copy2            — StackedMemoryPool.Take() block + Buffer.MemoryCopy (pooled allocation)
    //   * copy3            — new block + msvcrt memcpy P/Invoke   (Windows only)
    //
    // The StackedMemoryPool arm was ported to the current pool API (Take()/Return(IntPtr)/Dispose());
    // the old TakeBuffer(len)/ReturnBuffer(byte[])/Clear() no longer exist. Inputs are immutable across
    // invocations (each Copy* allocates and frees its OWN result; Copy2's Take is balanced by the
    // returned block's Free-triggered Return), so setup is [GlobalSetup] — NOT [IterationSetup], which
    // would force UnrollFactor=1 and needlessly re-allocate the pool before every iteration.

    // ── Latest results (net10.0 Release, InProcessEmit Throughput, per-op via OperationsPerInvoke; 2026-08-21).
    //    Mean = cost of ONE alloc+copy of 100 ints. Pooled allocation (copy2) is fastest (0.62x); fresh
    //    block + Span.CopyTo / Buffer.MemoryCopy / msvcrt memcpy are all ~parity. ──
    //   | Method            | Mean      | Ratio | Rank | Allocated |
    //   |------------------ |----------:|------:|-----:|----------:|
    //   | copy  (baseline)  | 114.94 ns |  1.00 |    2 |      96 B |   new block + Span.CopyTo
    //   | copy1             | 113.97 ns |  0.99 |    2 |      96 B |   new block + Buffer.MemoryCopy
    //   | copy2             |  71.17 ns |  0.62 |    1 |     152 B |   pooled block + Buffer.MemoryCopy
    //   | copy3             | 117.93 ns |  1.03 |    2 |      96 B |   new block + msvcrt memcpy
    [Config(typeof(UnmanagedThroughputConfig))]
    public unsafe class UnmanagedCopy
    {
        private const int length = 100;
        private const int iterations = 800_000;

        private UnmanagedMemoryBlock<int> @from;
        private static StackedMemoryPool pool;

        [GlobalSetup]
        public void Setup()
        {
            @from = new UnmanagedMemoryBlock<int>(length);
            for (int i = 0; i < length; i++)
                @from[i] = i + 1;
            pool = new StackedMemoryPool(1_677_721, 19);
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            // Untimed correctness guard: the copy must reproduce the source.
            var c = Copy<int>(@from);
            for (long i = 0; i < length; i++)
                if (c.Address[i] != @from.Address[i]) throw new Exception("copy verification failed");
            c.Free();

            @from.Free();
            pool?.Dispose();
            pool = null;
        }

        [Benchmark(OperationsPerInvoke = iterations)]
        public void copy3()
        {
            for (int j = 0; j < iterations; j++)
                Copy3<int>(@from).Free();
        }

        [Benchmark(OperationsPerInvoke = iterations)]
        public void copy2()
        {
            for (int j = 0; j < iterations; j++)
                Copy2<int>(@from).Free();
        }

        [Benchmark(OperationsPerInvoke = iterations)]
        public void copy1()
        {
            for (int j = 0; j < iterations; j++)
                Copy1<int>(@from).Free();
        }

        [Benchmark(Baseline = true, OperationsPerInvoke = iterations)]
        public void copy()
        {
            for (int j = 0; j < iterations; j++)
                Copy<int>(@from).Free();
        }

        [DllImport("msvcrt.dll", EntryPoint = "memcpy", CallingConvention = CallingConvention.Cdecl, SetLastError = false)]
        public static extern int MemCopy(void* dest, void* src, UIntPtr count);

        [MethodImpl(OptimizeAndInline)]
        public static UnmanagedMemoryBlock<T> Copy3<T>(UnmanagedMemoryBlock<T> source) where T : unmanaged
        {
            var len = source.Count * sizeof(T);
            var ret = new UnmanagedMemoryBlock<T>(source.Count);
            MemCopy(ret.Address, source.Address, (UIntPtr)len);
            return ret;
        }

        [MethodImpl(OptimizeAndInline)]
        public static UnmanagedMemoryBlock<T> Copy2<T>(UnmanagedMemoryBlock<T> source) where T : unmanaged
        {
            var len = source.Count * sizeof(T);
            var buffer = pool.Take();
            var ret = new UnmanagedMemoryBlock<T>((T*)buffer.ToPointer(), source.Count, () => pool.Return(buffer));
            Buffer.MemoryCopy(source.Address, ret.Address, len, len);
            return ret;
        }

        [MethodImpl(OptimizeAndInline)]
        public static UnmanagedMemoryBlock<T> Copy1<T>(UnmanagedMemoryBlock<T> source) where T : unmanaged
        {
            var ret = new UnmanagedMemoryBlock<T>(source.Count);
            var len = ret.Count * sizeof(T);
            Buffer.MemoryCopy(source.Address, ret.Address, len, len);
            return ret;
        }

        [MethodImpl(OptimizeAndInline)]
        public static UnmanagedMemoryBlock<T> Copy<T>(UnmanagedMemoryBlock<T> source) where T : unmanaged
        {
            var ret = new UnmanagedMemoryBlock<T>(source.Count);
            new Span<T>(source.Address, (int)source.Count).CopyTo(new Span<T>(ret.Address, (int)ret.Count));
            return ret;
        }
    }
}
