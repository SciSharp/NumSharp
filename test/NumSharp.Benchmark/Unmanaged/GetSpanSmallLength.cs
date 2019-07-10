using System;
using System.Linq;
using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Benchmark.Unmanaged
{
    //|         Method | RunStrategy |         Mean |          Error |         StdDev |       Median |         Min |          Max |  Ratio | RatioSD |
    //|--------------- |------------ |-------------:|---------------:|---------------:|-------------:|------------:|-------------:|-------:|--------:|
    //|  VectorNonCopy |   ColdStart |   1,355.8 us |    774.3699 us |    891.7655 us |   1,134.7 us |  1,071.9 us |   5,124.9 us |   8.26 |    1.12 |
    //| VectorWithCopy |   ColdStart | 116,122.1 us | 44,224.4978 us | 50,928.9974 us |  99,734.1 us | 93,455.4 us | 327,380.6 us | 812.20 |  411.45 |
    //|        NDArray |   ColdStart |   9,481.6 us |  1,878.5694 us |  2,163.3633 us |   8,922.2 us |  8,176.8 us |  17,232.6 us |  63.68 |   14.33 |
    //|    SimpleArray |   ColdStart |     185.2 us |    186.2988 us |    214.5420 us |     136.0 us |    135.5 us |   1,096.4 us |   1.00 |    0.00 |
    //|     ManualCopy |   ColdStart |   1,158.0 us |    194.1728 us |    223.6097 us |   1,103.5 us |  1,099.4 us |   2,107.2 us |   7.78 |    1.41 |
    //|     MemoryCopy |   ColdStart |     189.5 us |    219.4025 us |    252.6643 us |     130.4 us |    130.2 us |   1,262.5 us |   0.98 |    0.07 |
    //|                |             |              |                |                |              |             |              |        |         |
    //|  VectorNonCopy |  Throughput |   1,174.8 us |     39.6416 us |     42.4160 us |   1,161.7 us |  1,117.3 us |   1,294.7 us |   8.69 |    0.32 |
    //| VectorWithCopy |  Throughput | 107,212.1 us | 12,329.5115 us | 13,704.2193 us | 102,322.9 us | 95,017.6 us | 138,940.6 us | 798.46 |  104.51 |
    //|        NDArray |  Throughput |   8,386.5 us |    139.0418 us |    154.5446 us |   8,415.6 us |  8,142.1 us |   8,641.9 us |  61.92 |    1.21 |
    //|    SimpleArray |  Throughput |     135.4 us |      0.3412 us |      0.3504 us |     135.3 us |    135.2 us |     136.0 us |   1.00 |    0.00 |
    //|     ManualCopy |  Throughput |   1,106.2 us |      9.3761 us |     10.4215 us |   1,102.8 us |  1,097.2 us |   1,127.6 us |   8.17 |    0.07 |
    //|     MemoryCopy |  Throughput |     132.9 us |      3.1830 us |      3.5379 us |     130.8 us |    130.1 us |     140.3 us |   0.98 |    0.03 |

    [SimpleJob(RunStrategy.ColdStart, targetCount: 20)]
    [SimpleJob(RunStrategy.Throughput, targetCount: 20)]
    [MinColumn, MaxColumn, MeanColumn, MedianColumn]
    [HtmlExporter]
    public unsafe class GetSpanSmallLength
    {
        private const int length = 100;
        private const int iterations = 20_000;

        private UnmanagedArray<int> from;
        private UnmanagedByteStorage<int> fromvec;
        private UnmanagedArray<int> to;
        private UnmanagedByteStorage<int> setvec;

        private UnmanagedArray<int> fromsimple;
        private UnmanagedArray<int> tosimple;

        NDArray nd;

        [GlobalSetup]
        public void Setup()
        {
            @from = new UnmanagedArray<int>(length);
            fromvec = new UnmanagedByteStorage<int>(new int[10 * length], new Shape(10, length));
            to = new UnmanagedArray<int>(length);
            setvec = new UnmanagedByteStorage<int>(Enumerable.Range(0, length).ToArray(), new Shape(length));
            nd = np.arange(length * 10).reshape(10, length);

            fromsimple = new UnmanagedArray<int>(length);
            tosimple = new UnmanagedArray<int>(length);
        }

        [BenchmarkDotNet.Attributes.IterationCleanup()]
        public void Cleanup()
        {
            if (memoryTrash != null)
                for (var i = 0; i < memoryTrash.Length; i++)
                {
                    var vector = memoryTrash[i];
                    if (vector == null)
                        break;
                    vector.Free();
                    memoryTrash[i] = null;
                }
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
                var _ = fromvec.GetCopy(3);
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
