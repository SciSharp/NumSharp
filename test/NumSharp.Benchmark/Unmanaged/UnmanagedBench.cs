using System;
using System.Runtime.InteropServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Benchmark.Unmanaged
{
    // Revived 2026-08 (was fully commented out). Compares four ways to write element i := i over 100k
    // ints. The outer `iterations` loop repeats the whole-array fill; OperationsPerInvoke = iterations
    // so BenchmarkDotNet reports the cost of ONE 100k fill (per-op), and the MemoryDiagnoser allocation
    // column is likewise per-fill. All four arms perform identical work — this is a clean,
    // apples-to-apples accessor comparison.
    //   * UnmanagedArray          — UnmanagedMemoryBlock<int> indexer
    //   * Vector                  — ArraySlice<int>.SetIndex  (the modern stand-in for the deleted
    //                               shape-aware UnmanagedByteStorage<T>, which this arm originally used)
    //   * SimpleArray  (baseline) — managed int[]
    //   * SimpleArray_DirectAccess— raw int* into the pinned managed array
    //
    // Historical numbers (BenchmarkDotNet 0.12.x, .NET Core, out-of-process ClrJob/CoreJob/MonoJob —
    // all three job kinds no longer exist), kept for reference only (whole-method time, NOT per-op):
    //   |                   Method | RunStrategy |      Mean | Ratio |
    //   |------------------------- |------------ |----------:|------:|
    //   |           UnmanagedArray |   ColdStart | 110.51 ms |  1.08 |
    //   |                   Vector |   ColdStart | 105.41 ms |  1.03 |
    //   |              SimpleArray |   ColdStart | 102.15 ms |  1.00 |
    //   | SimpleArray_DirectAccess |   ColdStart |  87.79 ms |  0.86 |

    // ── Latest results (net10.0 Release, InProcessEmit Throughput, per-op via OperationsPerInvoke; 2026-08-21).
    //    Mean = cost of ONE 100k fill. NOTE: the Vector arm (ArraySlice.SetIndex) BOXES each element
    //    (~24 B x 100k = 2.4 MB/op) — hence 10x slower; the other three do not allocate. ──
    //   | Method                   | Mean      | Ratio | Rank | Allocated |
    //   |------------------------- |----------:|------:|-----:|----------:|
    //   | UnmanagedArray           |  27.02 us |  1.45 |    3 |         - |
    //   | Vector                   | 194.79 us | 10.45 |    4 | 2399988 B |
    //   | SimpleArray  (baseline)  |  18.65 us |  1.00 |    1 |         - |
    //   | SimpleArray_DirectAccess |  25.42 us |  1.36 |    2 |         - |
    [Config(typeof(UnmanagedThroughputConfig))]
    public unsafe class UnmanagedBench
    {
        private const int length = 100_000;
        private const int iterations = 2_000;
        int[] arr;
        UnmanagedMemoryBlock<int> mem;
        ArraySlice<int> vec;
        private unsafe int* arrayAddress;

        [GlobalSetup]
        public void Setup()
        {
            mem = new UnmanagedMemoryBlock<int>(length);
            vec = ArraySlice<int>.Allocate(length);
            arr = new int[length];

            var handle = GCHandle.Alloc(arr, GCHandleType.Pinned);
            arrayAddress = (int*)handle.AddrOfPinnedObject();
        }

        [GlobalCleanup]
        public void Verify()
        {
            // Untimed correctness guard: each accessor under test must actually write (guards against a
            // JIT eliding the loop body).
            mem[3] = 3; if (mem[3] != 3) throw new Exception("UnmanagedMemoryBlock indexer failed");
            vec.SetIndex(3, 3); if (vec[3] != 3) throw new Exception("ArraySlice.SetIndex failed");
            arr[3] = 3; if (arr[3] != 3) throw new Exception("int[] write failed");
            arrayAddress[3] = 3; if (arrayAddress[3] != 3) throw new Exception("pinned int* write failed");
        }

        [Benchmark(OperationsPerInvoke = iterations)]
        public void UnmanagedArray()
        {
            for (int j = 0; j < iterations; j++)
            {
                for (int i = 0; i < length; i++)
                {
                    mem[i] = i;
                }
            }
        }

        [Benchmark(OperationsPerInvoke = iterations)]
        public void Vector()
        {
            for (int j = 0; j < iterations; j++)
            {
                for (int i = 0; i < length; i++)
                {
                    vec.SetIndex(i, i);
                }
            }
        }

        [Benchmark(Baseline = true, OperationsPerInvoke = iterations)]
        public void SimpleArray()
        {
            for (int j = 0; j < iterations; j++)
            {
                for (int i = 0; i < length; i++)
                {
                    arr[i] = i;
                }
            }
        }

        [Benchmark(OperationsPerInvoke = iterations)]
        public void SimpleArray_DirectAccess()
        {
            int* ptr = arrayAddress;
            for (int j = 0; j < iterations; j++)
            {
                for (int i = 0; i < length; i++)
                {
                    *(ptr + i) = i;
                }
            }
        }
    }
}
