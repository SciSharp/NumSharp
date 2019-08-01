//using System.Linq;
//using System.Runtime.CompilerServices;
//using BenchmarkDotNet.Attributes;
//using BenchmarkDotNet.Engines;
//using NumSharp.Backends.Unmanaged;

//namespace NumSharp.Benchmark.Unmanaged
//{
//    //|         Method |     Job | Runtime |          Toolchain | IterationCount | RunStrategy | UnrollFactor |       Mean |     Error |    StdDev |        Min |        Max |     Median | Ratio | RatioSD |
//    //|--------------- |-------- |-------- |------------------- |--------------- |------------ |------------- |-----------:|----------:|----------:|-----------:|-----------:|-----------:|------:|--------:|
//    //| UnmanagedArray | Default |    Core |            Default |             20 |   ColdStart |            1 |   241.9 ms |  1.880 ms |  2.165 ms |   235.4 ms |   247.8 ms |   241.5 ms |     ? |       ? |
//    //|    SimpleArray | Default |    Core |            Default |             20 |   ColdStart |            1 |   215.9 ms |  2.711 ms |  3.122 ms |   211.9 ms |   222.3 ms |   215.8 ms |     ? |       ? |
//    //|     ManualCopy | Default |    Core |            Default |             20 |   ColdStart |            1 |   681.1 ms |  7.453 ms |  8.583 ms |   664.9 ms |   706.0 ms |   680.5 ms |     ? |       ? |
//    //| UnmanagedArray | Default |    Core |            Default |             20 |  Throughput |           16 |   238.2 ms |  2.640 ms |  3.040 ms |   234.6 ms |   243.8 ms |   237.5 ms |     ? |       ? |
//    //|    SimpleArray | Default |    Core |            Default |             20 |  Throughput |           16 |   220.1 ms |  1.986 ms |  2.039 ms |   215.3 ms |   223.5 ms |   220.3 ms |     ? |       ? |
//    //|     ManualCopy | Default |    Core |            Default |             20 |  Throughput |           16 |   851.2 ms | 58.912 ms | 67.844 ms |   741.0 ms |   934.0 ms |   877.9 ms |     ? |       ? |

//    [SimpleJob(RunStrategy.ColdStart, targetCount: 20)]
//    [SimpleJob(RunStrategy.Throughput, targetCount: 20)]
//    [MinColumn, MaxColumn, MeanColumn, MedianColumn]
//    [HtmlExporter]
//    public unsafe class SetSpan
//    {
//        private const int length = 100_000;
//        private const int iterations = 20_000;

//        UnmanagedMemoryBlock<int> from;
//        UnmanagedByteStorage<int> fromvec;
//        UnmanagedMemoryBlock<int> to;
//        UnmanagedByteStorage<int> setvec;

//        private UnmanagedMemoryBlock<int> fromsimple;
//        private UnmanagedMemoryBlock<int> tosimple;

//        [GlobalSetup]
//        public void Setup()
//        {
//            @from = new UnmanagedMemoryBlock<int>(length);
//            fromvec = new UnmanagedByteStorage<int>(new int[10 * length], new Shape(10, length));
//            to = new UnmanagedMemoryBlock<int>(length);
//            setvec = new UnmanagedByteStorage<int>(Enumerable.Range(0, length).ToArray(), new Shape(length));


//            fromsimple = new UnmanagedMemoryBlock<int>(length);
//            tosimple = new UnmanagedMemoryBlock<int>(length);
//        }

//        [Benchmark]
//        public void UnmanagedArray()
//        {
//            for (int j = 0; j < iterations; j++)
//            {
//                fromvec.Set(setvec, 3);
//            }
//        }

//        [Benchmark(Baseline = true)]
//        public void SimpleArray()
//        {
//            var fromspan = fromsimple.AsSpan();
//            var tospan = tosimple.AsSpan();
//            for (int i = 0; i < iterations; i++)
//            {
//                fromspan.CopyTo(tospan);
//            }
//        }

//        [Benchmark]
//        public void ManualCopy()
//        {
//            int* frm = (int*)Unsafe.AsPointer(ref fromsimple.GetPinnableReference());
//            int* to = (int*)Unsafe.AsPointer(ref tosimple.GetPinnableReference());
//            for (int j = 0; j < iterations; j++)
//            {
//                for (int i = 0; i < length; i++)
//                {
//                    *(to + i) = *(frm + i);
//                }
//            }
//        }
//    }
//}
