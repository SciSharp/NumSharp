//using System;
//using System.Collections.Generic;
//using System.Linq;
//using System.Threading.Tasks;
//using BenchmarkDotNet.Attributes;
//using NumSharp.Backends.Unmanaged;

//namespace NumSharp.Benchmark.Unmanaged
//{
//    //|              Method | Iterations |            Mean |         Error |        StdDev |          Median |             Min |             Max | Ratio | RatioSD |
//    //|-------------------- |----------- |----------------:|--------------:|--------------:|----------------:|----------------:|----------------:|------:|--------:|
//    //| DefaultHigherDegree |        100 |        35.05 us |      6.150 us |      5.135 us |        35.60 us |        23.40 us |        42.80 us |  0.86 |    0.23 |
//    //|             Chunked |        100 |        44.38 us |     14.139 us |     13.226 us |        44.40 us |        26.50 us |        71.20 us |  1.11 |    0.41 |
//    //|             Default |        100 |        41.23 us |      8.003 us |      7.486 us |        41.20 us |        29.50 us |        56.60 us |  1.00 |    0.00 |
//    //|              Linear |        100 |        33.87 us |     18.341 us |     17.156 us |        25.50 us |        17.00 us |        71.80 us |  0.86 |    0.47 |
//    //|                     |            |                 |               |               |                 |                 |                 |       |         |
//    //| DefaultHigherDegree |      50000 |     3,315.75 us |    320.059 us |    299.383 us |     3,358.80 us |     2,888.00 us |     3,881.30 us |  1.01 |    0.14 |
//    //|             Chunked |      50000 |     3,235.83 us |    398.624 us |    372.873 us |     3,139.90 us |     2,881.80 us |     4,070.30 us |  0.99 |    0.18 |
//    //|             Default |      50000 |     3,325.63 us |    380.863 us |    356.259 us |     3,283.10 us |     2,841.60 us |     3,937.90 us |  1.00 |    0.00 |
//    //|              Linear |      50000 |    10,506.26 us |    365.836 us |    342.204 us |    10,481.90 us |     9,801.70 us |    11,078.90 us |  3.20 |    0.38 |
//    //|                     |            |                 |               |               |                 |                 |                 |       |         |
//    //| DefaultHigherDegree |     100000 |     6,225.46 us |    405.006 us |    378.843 us |     6,282.30 us |     5,449.10 us |     6,745.50 us |  0.98 |    0.08 |
//    //|             Chunked |     100000 |     6,459.21 us |    620.486 us |    580.403 us |     6,367.40 us |     5,361.60 us |     7,310.20 us |  1.02 |    0.11 |
//    //|             Default |     100000 |     6,363.70 us |    387.224 us |    362.210 us |     6,336.40 us |     5,889.80 us |     6,902.90 us |  1.00 |    0.00 |
//    //|              Linear |     100000 |    20,096.70 us |    269.536 us |    252.124 us |    20,027.00 us |    19,725.50 us |    20,553.20 us |  3.17 |    0.20 |
//    //|                     |            |                 |               |               |                 |                 |                 |       |         |
//    //| DefaultHigherDegree |    1000000 |    54,332.82 us |  1,051.753 us |    983.810 us |    54,203.25 us |    52,923.85 us |    55,812.55 us |  0.97 |    0.05 |
//    //|             Chunked |    1000000 |    52,661.57 us |    826.620 us |    690.265 us |    52,591.70 us |    51,600.90 us |    54,156.10 us |  0.93 |    0.04 |
//    //|             Default |    1000000 |    56,406.96 us |  2,563.616 us |  2,398.008 us |    56,596.35 us |    53,468.35 us |    59,842.65 us |  1.00 |    0.00 |
//    //|              Linear |    1000000 |   203,360.73 us |  3,474.078 us |  3,249.655 us |   202,893.80 us |   199,156.30 us |   211,793.00 us |  3.61 |    0.16 |
//    //|                     |            |                 |               |               |                 |                 |                 |       |         |
//    //| DefaultHigherDegree |   10000000 |   620,079.34 us | 47,974.548 us | 42,528.187 us |   611,766.60 us |   573,894.00 us |   709,457.20 us |  1.07 |    0.06 |
//    //|             Chunked |   10000000 |   538,294.63 us | 20,631.564 us | 19,298.778 us |   534,506.70 us |   506,161.00 us |   571,052.20 us |  0.93 |    0.03 |
//    //|             Default |   10000000 |   576,942.29 us | 16,614.468 us | 14,728.293 us |   577,812.15 us |   553,668.80 us |   598,920.90 us |  1.00 |    0.00 |
//    //|              Linear |   10000000 | 2,133,232.02 us | 36,133.767 us | 33,799.548 us | 2,138,216.80 us | 2,074,787.80 us | 2,187,277.80 us |  3.69 |    0.07 |

//    [MinColumn, MaxColumn, MeanColumn, MedianColumn]
//    [SimpleJob(launchCount: 1, warmupCount: 3, targetCount: 15)]
//    public class ParallelFor
//    {
//        [Params(100, 50_000, 100_000, 1_000_000, 10_000_000)]
//        public int Iterations { get; set; }

//        UnmanagedByteStorage<int> a;
//        UnmanagedByteStorage<int> b;

//        //private ParallelOptions settingsDistributed;
//        private ParallelOptions settingsConcurrent;

//        [GlobalSetup]
//        public void GlobalSetup()
//        {
//            //if (DistributedScheduler.Created == false) {
//            //    DistributedScheduler.Configure(1, 1);
//            //}
//            //
//            //var dist = DistributedScheduler.Default;
//            //settingsDistributed = new ParallelOptions() {TaskScheduler = DistributedScheduler.Default, MaxDegreeOfParallelism = dist.MaximumConcurrencyLevel};
//            settingsConcurrent = new ParallelOptions() {TaskScheduler = TaskScheduler.Default, MaxDegreeOfParallelism = Environment.ProcessorCount * 2};
//        }

//        [IterationSetup]
//        public void Setup()
//        {
//            a = new UnmanagedByteStorage<int>(new Shape(5, 5), 0);
//            b = new UnmanagedByteStorage<int>(Enumerable.Range(0, 25).ToArray(), new Shape(5, 5));

//            var _ = UnmanagedByteStorage<int>.TypeCode;

//            var chunks = Chunk(1000, Environment.ProcessorCount);
//            Parallel.ForEach(chunks, i =>
//                { });
//        }


//        //[Benchmark]
//        //public void DistributedScheduler_() {
//        //    Parallel.For(0, largeSize, settingsDistributed, i => {
//        //        var c = a + b;
//        //    });
//        //}

//        [Benchmark]
//        public void DefaultHigherDegree()
//        {
//            Parallel.For(0, Iterations, settingsConcurrent, i =>
//            {
//                var c = a + b;
//            });
//        }

//        [Benchmark()]
//        public void Chunked()
//        {
//            var chunks = Chunk(Iterations, Environment.ProcessorCount);
//            Parallel.ForEach(chunks, i =>
//            {
//                for (int j = i.from; j <= i.to; j++)
//                {
//                    var c = a + b;
//                }
//            });
//        }

//        public static (int from, int to)[] Chunk(int len, int divby)
//        {
//            if (divby >= len || divby <= 1)
//                return new (int @from, int to)[] {(0, len - 1)};

//            var chunkSize = len / divby;
//            var lower = (len / chunkSize);
//            var ret = new (int, int)[lower + 1];
//            var index = 0;
//            for (var i = 0; i < len; i += chunkSize, index++)
//            {
//                ret[index] = (i, Math.Min(i + chunkSize, len) - 1);
//            }

//            if (index == lower)
//                Array.Resize(ref ret, lower);
//            return ret;
//        }

//        [Benchmark(Baseline = true)]
//        public void Default()
//        {
//            Parallel.For(0, Iterations, i =>
//            {
//                var c = a + b;
//            });
//        }

//        [Benchmark()]
//        public void Linear()
//        {
//            for (var i = 0; i < Iterations; i++)
//            {
//                var c = a + b;
//            }
//        }
//    }
//}
