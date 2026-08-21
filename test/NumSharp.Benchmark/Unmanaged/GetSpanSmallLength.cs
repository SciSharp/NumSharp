using System;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Benchmark.Unmanaged
{
    // Revived 2026-08 (was fully commented out). length = 100, 20_000 repetitions — the small-block
    // twin of GetSpanLargeLength, where per-call overhead dominates over raw bandwidth.
    // OperationsPerInvoke = iterations, so the reported Mean is the cost of ONE get/copy.
    //
    // NOTE: the NDArray arm creates a zero-copy VIEW; the copy arms MOVE `length` ints — different
    // operations grouped for reference (read the NDArray row as view-object allocation cost).
    //   * NDArray                 — nd[3] (zero-copy view of row 3); returned so BenchmarkDotNet
    //                               consumes it (prevents dead-code elimination of the view)
    //   * NDArrayCopy             — nd[3].copy() (materialising copy; replaces the old
    //                               UnmanagedByteStorage<T> "VectorNonCopy"/"VectorWithCopy" arms)
    //   * SimpleArray (baseline)  — Span<int>.CopyTo between two unmanaged blocks
    //   * ManualCopy              — element-by-element raw int* loop
    //   * MemoryCopy              — Buffer.MemoryCopy over the whole block

    // ── Latest results (net10.0 Release, InProcessEmit Throughput, per-op via OperationsPerInvoke; 2026-08-21).
    //    Mean = cost of ONE get/copy. At length 100 the FIXED view-object cost (~758 ns) dwarfs a tiny
    //    memcpy (~4 ns), so the view is 176x SLOWER here — the mirror image of GetSpanLargeLength. ──
    //   | Method                  | Mean         | Ratio  | Rank | Allocated |
    //   |------------------------ |-------------:|-------:|-----:|----------:|
    //   | NDArray                 |   758.310 ns | 176.15 |    3 |    1480 B |
    //   | NDArrayCopy             | 2,614.478 ns | 607.31 |    4 |    2903 B |
    //   | SimpleArray  (baseline) |     4.305 ns |   1.00 |    1 |         - |
    //   | ManualCopy              |    32.464 ns |   7.54 |    2 |         - |
    //   | MemoryCopy              |     4.273 ns |   0.99 |    1 |         - |
    [Config(typeof(UnmanagedThroughputConfig))]
    public unsafe class GetSpanSmallLength
    {
        private const int length = 100;
        private const int iterations = 20_000;

        private UnmanagedMemoryBlock<int> fromsimple;
        private UnmanagedMemoryBlock<int> tosimple;

        NDArray nd;

        [GlobalSetup]
        public void Setup()
        {
            nd = np.arange(length * 10).reshape(10, length);
            fromsimple = new UnmanagedMemoryBlock<int>(length);
            tosimple = new UnmanagedMemoryBlock<int>(length);
            for (int i = 0; i < length; i++)
                fromsimple[i] = i + 1;
        }

        [GlobalCleanup]
        public void Verify()
        {
            // Untimed correctness guards (guard against a JIT eliding the copy / view work).
            for (int i = 0; i < length; i++) tosimple.Address[i] = 0;
            new Span<int>(fromsimple.Address, length).CopyTo(new Span<int>(tosimple.Address, length));
            for (int i = 0; i < length; i++)
                if (tosimple.Address[i] != fromsimple.Address[i]) throw new Exception("copy verification failed");

            if (nd[3].copy().GetInt32(0) != 3 * length) throw new Exception("nd[3] verification failed");
        }

        [Benchmark(OperationsPerInvoke = iterations)]
        public NDArray NDArray()
        {
            NDArray last = null;
            for (int j = 0; j < iterations; j++)
            {
                last = nd[3];
            }
            return last;
        }

        [Benchmark(OperationsPerInvoke = iterations)]
        public NDArray NDArrayCopy()
        {
            NDArray last = null;
            for (int j = 0; j < iterations; j++)
            {
                last = nd[3].copy();
            }
            return last;
        }

        [Benchmark(Baseline = true, OperationsPerInvoke = iterations)]
        public void SimpleArray()
        {
            var fromspan = new Span<int>(fromsimple.Address, length);
            var tospan = new Span<int>(tosimple.Address, length);
            for (int i = 0; i < iterations; i++)
            {
                fromspan.CopyTo(tospan);
            }
        }

        [Benchmark(OperationsPerInvoke = iterations)]
        public void ManualCopy()
        {
            int* frm = fromsimple.Address;
            int* to = tosimple.Address;
            for (int j = 0; j < iterations; j++)
            {
                for (int i = 0; i < length; i++)
                {
                    *(to + i) = *(frm + i);
                }
            }
        }

        [Benchmark(OperationsPerInvoke = iterations)]
        public void MemoryCopy()
        {
            int* frm = fromsimple.Address;
            int* to = tosimple.Address;
            var single = sizeof(int);
            var len = length * single;
            for (int j = 0; j < iterations; j++)
            {
                Buffer.MemoryCopy(frm, to, len, len);
            }
        }
    }
}
