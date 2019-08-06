using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using NumSharp.Backends.Unmanaged;
using NumSharp.Memory.Pooling;
using NumSharp.Unmanaged.Memory;
using NumSharp.Utilities;

namespace NumSharp.Benchmark.Unmanaged
{

    //|            Method |     Mean |    Error |   StdDev |   Median |      Min |        Max | Ratio | RatioSD |
    //|------------------ |---------:|---------:|---------:|---------:|---------:|-----------:|------:|--------:|
    //| OffsetIncrementor | 898.2 us | 30.68 us | 416.4 us | 747.5 us | 682.0 us | 3,483.5 us |  1.41 |    0.70 |
    //|   GetOffset | 658.4 us | 10.43 us | 141.6 us | 612.6 us | 542.0 us | 2,724.8 us |  1.00 |    0.00 |

    [SimpleJob(RunStrategy.ColdStart, targetCount: 2000)]
    [MinColumn, MaxColumn, MeanColumn, MedianColumn]
    [HtmlExporter]
    public class Incrementors
    {
        private Shape shape;
        private NDOffsetIncrementor iter;

        [GlobalSetup]
        public void Setup()
        {
            var _ = ScalarMemoryPool.Instance;
            var __ = InfoOf<byte>.Size;
            shape = new Shape(1, 1, 100_000);
            iter = new NDOffsetIncrementor(shape);
        }

        [IterationCleanup]
        public void Reset()
        {
            iter.Reset();
        }

        [Benchmark]
        public void OffsetIncrementor()
        {
            for (int i = 0; i < 100_000; i++)
            {
                iter.Next();
            }
        }

        [Benchmark(Baseline = true)]
        public void GetOffset()
        {
            for (int i = 0; i < 100_000; i++)
            {
                shape.GetOffset(0, 0, i);
            }
        }
    }
}
