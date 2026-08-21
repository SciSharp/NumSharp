using System;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Benchmark.Unmanaged
{
    // Revived 2026-08 (was fully commented out). Compares three ways to bulk-set a length-N block. The
    // outer `iterations` loop repeats one bulk copy; OperationsPerInvoke = iterations so the reported
    // Mean is the cost of ONE length-N copy. All three arms move `length` ints — comparable.
    //   * SliceSet                — ArraySlice<int>.CopyTo into a sub-range (the modern stand-in for
    //                               the deleted UnmanagedByteStorage<T>.Set(row) this arm originally used)
    //   * SimpleArray (baseline)  — Span<int>.CopyTo between two unmanaged blocks
    //   * ManualCopy              — element-by-element raw int* loop
    //
    // Historical numbers (BenchmarkDotNet 0.12.x) for the original storage-based arm are omitted
    // because that arm's type no longer exists; the copy baselines are unchanged in spirit.

    // ── Latest results (net10.0 Release, InProcessEmit Throughput, per-op via OperationsPerInvoke; 2026-08-21).
    //    Mean = cost of ONE length-100k bulk set. ArraySlice.CopyTo matches Span.CopyTo (both vectorized,
    //    no allocation); the naive element loop is 4.6x slower. ──
    //   | Method                  | Mean      | Ratio | Rank |
    //   |------------------------ |----------:|------:|-----:|
    //   | SliceSet                |  5.489 us |  1.00 |    1 |
    //   | SimpleArray  (baseline) |  5.506 us |  1.00 |    1 |
    //   | ManualCopy              | 25.424 us |  4.62 |    2 |
    [Config(typeof(UnmanagedThroughputConfig))]
    public unsafe class SetSpan
    {
        private const int length = 100_000;
        private const int iterations = 20_000;

        private ArraySlice<int> bigSlice;   // a (10*length) buffer; row 3 is the write target
        private ArraySlice<int> setSlice;   // the length-N source row
        private ArraySlice<int> destRow;    // a view over row 3 of bigSlice

        private UnmanagedMemoryBlock<int> fromsimple;
        private UnmanagedMemoryBlock<int> tosimple;

        [GlobalSetup]
        public void Setup()
        {
            bigSlice = ArraySlice<int>.Allocate(10L * length);
            setSlice = ArraySlice<int>.Allocate(length);
            for (int i = 0; i < length; i++)
                setSlice.SetIndex(i, i + 1);
            destRow = bigSlice.Slice(3L * length, length);

            fromsimple = new UnmanagedMemoryBlock<int>(length);
            tosimple = new UnmanagedMemoryBlock<int>(length);
            for (int i = 0; i < length; i++)
                fromsimple[i] = i + 1;
        }

        [GlobalCleanup]
        public void Verify()
        {
            // Untimed correctness guard: the slice copy must reproduce the source row.
            setSlice.CopyTo((IntPtr)destRow.Address);
            for (int i = 0; i < length; i++)
                if (destRow[i] != setSlice[i]) throw new Exception("SliceSet copy verification failed");
        }

        [Benchmark(OperationsPerInvoke = iterations)]
        public void SliceSet()
        {
            for (int j = 0; j < iterations; j++)
            {
                setSlice.CopyTo((IntPtr)destRow.Address);
            }
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
    }
}
