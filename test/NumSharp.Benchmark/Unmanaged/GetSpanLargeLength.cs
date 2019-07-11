using System;
using System.Linq;
using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Benchmark.Unmanaged
{
    //|         Method |          Toolchain | IterationCount | RunStrategy | UnrollFactor |       Mean |      Error |     StdDev |        Min |        Max |     Median |  Ratio | RatioSD |
    //|--------------- |------------------- |--------------- |------------ |------------- |-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|-------:|--------:|
    //|  VectorNonCopy |            Default |             20 |   ColdStart |            1 |   9.535 ms |  1.1402 ms |  1.3130 ms |   8.724 ms |  13.096 ms |   8.995 ms |   6.23 |    0.99 |
    //| VectorWithCopy |            Default |             20 |   ColdStart |            1 | 351.689 ms | 11.8808 ms | 13.6819 ms | 330.815 ms | 376.969 ms | 352.871 ms | 231.10 |   30.29 |
    //|        NDArray |            Default |             20 |   ColdStart |            1 |  78.610 ms |  3.0417 ms |  3.5028 ms |  75.879 ms |  88.297 ms |  77.575 ms |  51.58 |    6.39 |
    //|    SimpleArray |            Default |             20 |   ColdStart |            1 |   1.551 ms |  0.2153 ms |  0.2480 ms |   1.411 ms |   2.243 ms |   1.439 ms |   1.00 |    0.00 |
    //|     ManualCopy |            Default |             20 |   ColdStart |            1 |   7.641 ms |  0.1747 ms |  0.2012 ms |   7.538 ms |   8.430 ms |   7.580 ms |   5.01 |    0.57 |
    //|     MemoryCopy |            Default |             20 |   ColdStart |            1 |   1.394 ms |  0.1777 ms |  0.2047 ms |   1.302 ms |   2.173 ms |   1.329 ms |   0.91 |    0.11 |
    //|                |                    |                |             |              |            |            |            |            |            |            |        |         |
    //|  VectorNonCopy |            Default |             20 |  Throughput |           16 |   9.159 ms |  0.1123 ms |  0.1293 ms |   8.992 ms |   9.457 ms |   9.139 ms |   6.71 |    0.10 |
    //| VectorWithCopy |            Default |             20 |  Throughput |           16 | 356.050 ms | 12.3996 ms | 13.2675 ms | 336.000 ms | 381.900 ms | 356.417 ms | 260.50 |   10.13 |
    //|        NDArray |            Default |             20 |  Throughput |           16 |  73.999 ms |  1.7034 ms |  1.8227 ms |  71.696 ms |  77.212 ms |  73.811 ms |  54.14 |    1.40 |
    //|    SimpleArray |            Default |             20 |  Throughput |           16 |   1.366 ms |  0.0074 ms |  0.0082 ms |   1.356 ms |   1.385 ms |   1.366 ms |   1.00 |    0.00 |
    //|     ManualCopy |            Default |             20 |  Throughput |           16 |   7.345 ms |  0.0397 ms |  0.0408 ms |   7.286 ms |   7.423 ms |   7.348 ms |   5.37 |    0.04 |
    //|     MemoryCopy |            Default |             20 |  Throughput |           16 |   1.348 ms |  0.0261 ms |  0.0268 ms |   1.323 ms |   1.416 ms |   1.343 ms |   0.99 |    0.02 |

    [SimpleJob(RunStrategy.ColdStart, targetCount: 20)]
    [SimpleJob(RunStrategy.Throughput, targetCount: 20)]
    [MinColumn, MaxColumn, MeanColumn, MedianColumn]
    [HtmlExporter]
    public unsafe class GetSpanLargeLength
    {
        private const int length = 100_000;
        private const int iterations = 100;

        private UnmanagedMemoryBlock<int> from;
        private UnmanagedByteStorage<int> fromvec;
        private UnmanagedMemoryBlock<int> to;
        private UnmanagedByteStorage<int> setvec;

        private UnmanagedMemoryBlock<int> fromsimple;
        private UnmanagedMemoryBlock<int> tosimple;

        NDArray nd;

        [GlobalSetup]
        public void Setup()
        {
            @from = new UnmanagedMemoryBlock<int>(length);
            fromvec = new UnmanagedByteStorage<int>(new int[10 * length], new Shape(10, length));
            to = new UnmanagedMemoryBlock<int>(length);
            setvec = new UnmanagedByteStorage<int>(Enumerable.Range(0, length).ToArray(), new Shape(length));
            nd = np.arange(length * 10).reshape(10, length);

            fromsimple = new UnmanagedMemoryBlock<int>(length);
            tosimple = new UnmanagedMemoryBlock<int>(length);
        }

        [BenchmarkDotNet.Attributes.IterationCleanup()]
        public void Cleanup()
        {
            //if (memoryTrash != null)
            //    for (var i = 0; i < memoryTrash.Length; i++) {
            //        var vector = memoryTrash[i];
            //        if (vector == null)
            //            break;
            //        vector.Free();
            //        memoryTrash[i] = null;
            //    }
        }

        [Benchmark]
        public void VectorNonCopy()
        {
            for (int j = 0; j < iterations; j++)
            {
                var _ = fromvec.Get(3);
            }
        }

        UnmanagedByteStorage<int>[] memoryTrash;

        [Benchmark]
        public void VectorWithCopy()
        {
            memoryTrash = new UnmanagedByteStorage<int>[iterations];
            for (int j = 0; j < iterations; j++)
            {
                //memoryTrash[j] = fromvec.Get(3).Clone();
                fromvec.GetCopy(3).Clone();
            }
        }

        [Benchmark]
        public void NDArray()
        {
            for (int j = 0; j < iterations; j++)
            {
                var _ = nd[3];
            }
        }

        [Benchmark(Baseline = true)]
        public void SimpleArray()
        {
            var fromspan = fromsimple.AsSpan();
            var tospan = tosimple.AsSpan();
            for (int i = 0; i < iterations; i++)
            {
                fromspan.CopyTo(tospan);
            }
        }

        [Benchmark]
        public void ManualCopy()
        {
            int* frm = (int*)Unsafe.AsPointer(ref fromsimple.GetPinnableReference());
            int* to = (int*)Unsafe.AsPointer(ref tosimple.GetPinnableReference());
            for (int j = 0; j < iterations; j++)
            {
                for (int i = 0; i < length; i++)
                {
                    *(to + i) = *(frm + i);
                }
            }
        }

        [Benchmark]
        public void MemoryCopy()
        {
            int* frm = (int*)Unsafe.AsPointer(ref fromsimple.GetPinnableReference());
            int* to = (int*)Unsafe.AsPointer(ref tosimple.GetPinnableReference());
            var single = sizeof(int);
            var len = length * single;
            for (int j = 0; j < iterations; j++)
            {
                Buffer.MemoryCopy(frm, to, len, len);
            }
        }
    }
}
