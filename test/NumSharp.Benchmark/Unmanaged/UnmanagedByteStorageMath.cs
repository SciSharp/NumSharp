using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Benchmark.Unmanaged
{
    // Revived 2026-08 (was fully commented out). Compares three implementations of an element-wise int
    // add across two sizes. The size is a [Params] axis, so BenchmarkDotNet gives each N its OWN
    // baseline instead of dividing a 100k/500k arm by a 25-element one (the original mixed a 25-element
    // baseline with 100k/500k "Large" arms — and even labelled 100k and 500k arms both "Large", a
    // straight unequal-work bug). Each invocation performs `Repeats` adds; OperationsPerInvoke =
    // Repeats, so the reported Mean is the cost of ONE add.
    //   * UnmanagedBlockAdd — allocate an UnmanagedMemoryBlock<int> result and add via raw int*.
    //                         (Replaces the old UnmanagedByteStorage<int> `a + b` arm — that
    //                          shape-aware storage type with its own operator+ was deleted.)
    //   * Direct (baseline) — managed int[] result + add loop (returned so BDN consumes it)
    //   * NDArray_          — NDArray operator+ (returned so BDN consumes it)
    //
    // All three arms operate on int32 data of the SAME length N. The NDArray operands are cast to int32
    // (np.arange is int64) so they are element-for-element comparable with the int[] / block arms —
    // otherwise the NDArray arm would add 8-byte elements against the others' 4-byte, a size confound.
    // Inputs are immutable, so setup is [GlobalSetup].

    // ── Latest results (net10.0 Release, InProcessEmit Throughput, per-op via OperationsPerInvoke; 2026-08-21).
    //    Mean = cost of ONE add; Direct is the per-N baseline. At N=25 the scalar int[] loop wins (alloc
    //    overhead dominates tiny work); at N=100000 the SIMD NDArray add and the raw-block add are ~2x
    //    FASTER than the naive scalar loop. (Allocated is managed-only, so the NDArray/block arms'
    //    unmanaged result buffers are invisible — Direct's int[] is fully counted.) ──
    //   | Method            | N      | Mean         | Ratio | Rank | Allocated |
    //   |------------------ |------- |-------------:|------:|-----:|----------:|
    //   | UnmanagedBlockAdd | 25     |    106.89 ns |  7.60 |    2 |      96 B |
    //   | Direct (baseline) | 25     |     14.07 ns |  1.00 |    1 |     128 B |
    //   | NDArray_          | 25     |    955.32 ns | 67.92 |    3 |    1365 B |
    //   | UnmanagedBlockAdd | 100000 | 25,197.75 ns |  0.45 |    1 |     163 B |
    //   | Direct (baseline) | 100000 | 55,916.63 ns |  1.00 |    3 |  400252 B |
    //   | NDArray_          | 100000 | 27,706.40 ns |  0.50 |    2 |    1427 B |
    [Config(typeof(UnmanagedThroughputConfig))]
    public unsafe class DArrayMath
    {
        [Params(25, 100_000)]
        public int N { get; set; }

        private const int Repeats = 5;

        UnmanagedMemoryBlock<int> a;
        UnmanagedMemoryBlock<int> b;
        NDArray nd1;
        NDArray nd2;
        int[] arr_a;
        int[] arr_b;

        [GlobalSetup]
        public void Setup()
        {
            nd1 = np.arange(N).astype(NPTypeCode.Int32);
            nd2 = np.arange(N).astype(NPTypeCode.Int32);

            a = new UnmanagedMemoryBlock<int>(N);
            b = new UnmanagedMemoryBlock<int>(N);

            arr_a = new int[N];
            arr_b = new int[N];

            for (int i = 0; i < N; i++)
            {
                a.Address[i] = i;
                b.Address[i] = 2 * i;
                arr_a[i] = i;
                arr_b[i] = 2 * i;
            }
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            // Untimed correctness guard: a[i]=i, b[i]=2i  =>  sum must be 3i.
            var ret = new UnmanagedMemoryBlock<int>(a.Count);
            int* rp = ret.Address, xp = a.Address, yp = b.Address;
            for (long i = 0; i < a.Count; i++)
                rp[i] = xp[i] + yp[i];
            for (long i = 0; i < a.Count; i++)
                if (rp[i] != 3 * i) throw new System.Exception("add verification failed");
            ret.Free();

            a.Free();
            b.Free();
        }

        [MethodImpl(OptimizeAndInline)]
        private static void AddBlocks(UnmanagedMemoryBlock<int> x, UnmanagedMemoryBlock<int> y)
        {
            var n = x.Count;
            var ret = new UnmanagedMemoryBlock<int>(n);
            int* rp = ret.Address, xp = x.Address, yp = y.Address;
            for (long i = 0; i < n; i++)
                rp[i] = xp[i] + yp[i];
            ret.Free();
        }

        [Benchmark(OperationsPerInvoke = Repeats)]
        public void UnmanagedBlockAdd()
        {
            for (int rep = 0; rep < Repeats; rep++)
                AddBlocks(a, b);
        }

        [Benchmark(Baseline = true, OperationsPerInvoke = Repeats)]
        public int[] Direct()
        {
            int[] ret = null;
            for (int rep = 0; rep < Repeats; rep++)
            {
                var len = arr_a.Length;
                ret = new int[len];
                for (int i = 0; i < len; i++)
                {
                    ret[i] = arr_a[i] + arr_b[i];
                }
            }
            return ret;
        }

        [Benchmark(OperationsPerInvoke = Repeats)]
        public NDArray NDArray_()
        {
            NDArray ret = null;
            for (int rep = 0; rep < Repeats; rep++)
                ret = nd1 + nd2;
            return ret;
        }
    }
}
