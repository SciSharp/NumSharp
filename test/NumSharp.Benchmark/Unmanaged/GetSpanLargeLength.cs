using System;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Benchmark.Unmanaged
{
    // Revived 2026-08 (was fully commented out). length = 100_000, 100 repetitions.
    // OperationsPerInvoke = iterations, so the reported Mean is the cost of ONE get/copy.
    //
    // NOTE: the NDArray arm creates a zero-copy VIEW (no data movement); the SimpleArray/ManualCopy/
    // MemoryCopy/NDArrayCopy arms each MOVE `length` ints. They are grouped for reference but measure
    // DIFFERENT operations — read the NDArray row as "cost to allocate a view object", not a copy.
    //   * NDArray                 — nd[3] (zero-copy view of row 3); returned so BenchmarkDotNet
    //                               consumes it (prevents dead-code elimination of the view)
    //   * NDArrayCopy             — nd[3].copy() (materialising copy; replaces the old
    //                               UnmanagedByteStorage<T> "VectorWithCopy" arm, whose type is gone)
    //   * SimpleArray (baseline)  — Span<int>.CopyTo between two unmanaged blocks
    //   * ManualCopy              — element-by-element raw int* loop
    //   * MemoryCopy              — Buffer.MemoryCopy over the whole block
    //
    // Historical numbers (BenchmarkDotNet 0.12.x, out-of-process, whole-method time), reference only:
    //   |      Method | RunStrategy |       Mean |  Ratio |
    //   |------------ |------------ |-----------:|-------:|
    //   |     NDArray |   ColdStart |  78.610 ms |  51.58 |
    //   | SimpleArray |   ColdStart |   1.551 ms |   1.00 |
    //   |  ManualCopy |   ColdStart |   7.641 ms |   5.01 |
    //   |  MemoryCopy |   ColdStart |   1.394 ms |   0.91 |

    // ── Latest results (net10.0 Release, InProcessEmit Throughput, per-op via OperationsPerInvoke; 2026-08-21).
    //    Mean = cost of ONE get/copy. At length 100k the O(1) VIEW (NDArray) is 7.7x FASTER than a memcpy;
    //    contrast GetSpanSmallLength, where at length 100 the fixed view-object cost makes it 176x slower. ──
    //   | Method                  | Mean        | Ratio | Rank | Allocated |
    //   |------------------------ |------------:|------:|-----:|----------:|
    //   | NDArray                 |    738.9 ns |  0.13 |    1 |    1480 B |
    //   | NDArrayCopy             | 40,730.4 ns |  7.39 |    4 |    3016 B |
    //   | SimpleArray  (baseline) |  5,508.6 ns |  1.00 |    2 |         - |
    //   | ManualCopy              | 24,613.3 ns |  4.47 |    3 |         - |
    //   | MemoryCopy              |  5,499.5 ns |  1.00 |    2 |         - |
    [Config(typeof(UnmanagedThroughputConfig))]
    public unsafe class GetSpanLargeLength
    {
        private const int length = 100_000;
        private const int iterations = 100;

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
