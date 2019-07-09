using System.Runtime.InteropServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using OOMath;

namespace NumSharp.Benchmark.Unmanaged
{
    //|                   Method |          Toolchain | IterationCount | RunStrategy | UnrollFactor |      Mean |      Error |     StdDev |       Min |       Max |    Median | Ratio | RatioSD |
    //|------------------------- |------------------- |--------------- |------------ |------------- |----------:|-----------:|-----------:|----------:|----------:|----------:|------:|--------:|
    //|           UnmanagedArray |            Default |             20 |   ColdStart |            1 | 110.51 ms |  7.2478 ms |  8.3465 ms | 101.13 ms | 131.91 ms | 109.51 ms |  1.08 |    0.08 |
    //|                   Vector |            Default |             20 |   ColdStart |            1 | 105.41 ms |  2.5535 ms |  2.9406 ms | 102.32 ms | 111.92 ms | 104.39 ms |  1.03 |    0.04 |
    //|              SimpleArray |            Default |             20 |   ColdStart |            1 | 102.15 ms |  1.3386 ms |  1.5415 ms | 100.53 ms | 107.19 ms | 101.92 ms |  1.00 |    0.00 |
    //| SimpleArray_DirectAccess |            Default |             20 |   ColdStart |            1 |  87.79 ms |  1.8597 ms |  2.1417 ms |  85.89 ms |  95.26 ms |  87.23 ms |  0.86 |    0.02 |
    //|                          |                    |                |             |              |           |            |            |           |           |           |       |         |
    //|           UnmanagedArray |            Default |             20 |  Throughput |           16 | 110.53 ms |  4.6098 ms |  5.3087 ms | 102.73 ms | 121.18 ms | 109.59 ms |  1.00 |    0.05 |
    //|                   Vector |            Default |             20 |  Throughput |           16 | 110.17 ms |  3.9187 ms |  4.3557 ms | 104.04 ms | 121.25 ms | 109.57 ms |  1.00 |    0.05 |
    //|              SimpleArray |            Default |             20 |  Throughput |           16 | 110.29 ms |  5.1693 ms |  5.7457 ms | 102.51 ms | 121.31 ms | 107.99 ms |  1.00 |    0.00 |
    //| SimpleArray_DirectAccess |            Default |             20 |  Throughput |           16 |  86.74 ms |  0.5962 ms |  0.6866 ms |  86.03 ms |  88.55 ms |  86.66 ms |  0.79 |    0.04 |
    //|                          |                    |                |             |              |           |            |            |           |           |           |       |         |


    [SimpleJob(RunStrategy.ColdStart, targetCount: 20)]
    [SimpleJob(RunStrategy.Throughput, targetCount: 20)]
    [MinColumn, MaxColumn, MeanColumn, MedianColumn]
    [HtmlExporter]
    [ClrJob(baseline: true)]
    [MonoJob]
    [CoreJob]
    public unsafe class UnmanagedBench
    {
        private const int length = 100_000;
        private const int iterations = 2_000;
        int[] arr;
        UnmanagedArray<int> mem;
        UnmanagedByteStorage<int> vec;
        private unsafe int* arrayAddress;

        [GlobalSetup]
        public void Setup()
        {
            mem = new UnmanagedArray<int>(length);
            vec = new UnmanagedByteStorage<int>(mem, new Shape(length));
            arr = new int[length];

            var ret = new UnmanagedArray<int>();
            var handle = GCHandle.Alloc(arr, GCHandleType.Pinned);
            arrayAddress = (int*)handle.AddrOfPinnedObject();
        }

        [Benchmark]
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

        [Benchmark]
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

        [Benchmark(Baseline = true)]
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

        [Benchmark]
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
