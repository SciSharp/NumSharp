using System;
using System.Linq;
using System.Threading.Tasks;
using BenchmarkDotNet.Attributes;
using OOMath;

namespace NumSharp.Benchmark.Unmanaged
{
    //RELEASE-OPTIMIZE:
    //|                      Method | RunStrategy | UnrollFactor |        Mean |       Error |      StdDev |      Median |         Min |         Max | Ratio | RatioSD |
    //|---------------------------- |------------ |------------- |------------:|------------:|------------:|------------:|------------:|------------:|------:|--------:|
    //|    '(double) (object) Tval' |   ColdStart |            1 |   125.96 us | 186.4836 us | 123.3473 us |    75.20 us |    75.10 us |   468.90 us |  0.08 |    0.03 |
    //| '(double) (ValueType) Tval' |   ColdStart |            1 |   114.82 us | 180.4753 us | 119.3732 us |    75.20 us |    75.10 us |   454.30 us |  0.07 |    0.02 |
    //| '(double) (IComparab) Tval' |   ColdStart |            1 |   115.18 us | 188.9671 us | 124.9900 us |    75.25 us |    75.10 us |   470.90 us |  0.07 |    0.03 |
    //|     'FastCast((TOut) @in;)' |   ColdStart |            1 | 1,491.73 us | 985.5014 us | 651.8481 us | 1,285.55 us | 1,163.00 us | 3,326.30 us |  1.00 |    0.00 |
    //|                             |             |              |             |             |             |             |             |             |       |         |
    //|    '(double) (object) Tval' |  Throughput |           16 |    75.44 us |   0.1344 us |   0.0703 us |    75.41 us |    75.38 us |    75.58 us |  0.07 |    0.00 |
    //| '(double) (ValueType) Tval' |  Throughput |           16 |    76.64 us |   1.5446 us |   1.0216 us |    76.31 us |    75.48 us |    78.39 us |  0.07 |    0.00 |
    //| '(double) (IComparab) Tval' |  Throughput |           16 |    75.89 us |   0.6555 us |   0.4335 us |    75.79 us |    75.48 us |    76.70 us |  0.07 |    0.00 |
    //|     'FastCast((TOut) @in;)' |  Throughput |           16 | 1,126.62 us |  20.0607 us |  11.9378 us | 1,125.58 us | 1,113.78 us | 1,150.95 us |  1.00 |    0.00 |

    [MinColumn, MaxColumn, MeanColumn, MedianColumn]
    [SimpleJob(launchCount: 1, warmupCount: 3, targetCount: 15)]
    public class ParallelFor
    {
        private const int iterations = 10;
        private int largeSize = 250;
        UnmanagedByteStorage<int> a;
        UnmanagedByteStorage<int> b;

        //private ParallelOptions settingsDistributed;
        private ParallelOptions settingsConcurrent;

        [GlobalSetup]
        public void GlobalSetup()
        {
            //if (DistributedScheduler.Created == false) {
            //    DistributedScheduler.Configure(1, 1);
            //}
            //
            //var dist = DistributedScheduler.Default;
            //settingsDistributed = new ParallelOptions() {TaskScheduler = DistributedScheduler.Default, MaxDegreeOfParallelism = dist.MaximumConcurrencyLevel};
            settingsConcurrent = new ParallelOptions() {TaskScheduler = TaskScheduler.Default, MaxDegreeOfParallelism = Environment.ProcessorCount * 2};
        }

        [IterationSetup]
        public void Setup()
        {
            a = new UnmanagedByteStorage<int>(new Shape(5, 5), 0);
            b = new UnmanagedByteStorage<int>(Enumerable.Range(0, 25).ToArray(), new Shape(5, 5));

            var _ = UnmanagedByteStorage<int>.TypeCode;
        }


        //[Benchmark]
        //public void DistributedScheduler_() {
        //    Parallel.For(0, largeSize, settingsDistributed, i => {
        //        var c = a + b;
        //    });
        //}

        [Benchmark]
        public void DefaultHigherDegree()
        {
            Parallel.For(0, largeSize, settingsConcurrent, i =>
            {
                var c = a + b;
            });
        }

        [Benchmark(Baseline = true)]
        public void Default()
        {
            Parallel.For(0, largeSize, i =>
            {
                var c = a + b;
            });
        }

        [Benchmark()]
        public void Linear()
        {
            for (var i = 0; i < largeSize; i++)
            {
                var c = a + b;
            }
        }
    }
}
