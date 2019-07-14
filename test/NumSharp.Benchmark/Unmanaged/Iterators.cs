using System.Linq;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using NumSharp.Backends.Unmanaged;
using NumSharp.Memory.Pooling;
using NumSharp.Utilities;

namespace NumSharp.Benchmark.Unmanaged
{
    //|            Method |       Mean |    Error |   StdDev |     Median |        Min |        Max | Ratio | RatioSD |
    //|------------------ |-----------:|---------:|---------:|-----------:|-----------:|-----------:|------:|--------:|
    //| OffsetIncrementor | 1,049.5 us | 6.060 us | 82.25 us | 1,030.6 us | 1,006.9 us | 2,119.3 us |  1.42 |    0.15 |
    //|   GetIndexInShape |   745.3 us | 6.163 us | 83.64 us |   728.0 us |   685.1 us | 2,746.7 us |  1.00 |    0.00 |

    [SimpleJob(RunStrategy.ColdStart, targetCount: 2000)]
    [MinColumn, MaxColumn, MeanColumn, MedianColumn]
    [HtmlExporter]
    public class Iterators
    {
        private Shape shape;
        private NDIterator<int> iter;
        private NDArray ndarray;

        [GlobalSetup]
        public void Setup()
        {
            var _ = ScalarMemoryPool.Instance;
            var __ = InfoOf<byte>.Size;
            shape = new Shape(2, 1, 50_000);
            ndarray = np.array(Enumerable.Range(0, 100_000).ToArray()).reshape(ref shape);
            iter = new NDIterator<int>((IMemoryBlock<int>)ndarray.Array, shape);
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
                iter.MoveNext();
            }
        }

        [Benchmark(Baseline = true)]
        public void GetIndexInShape()
        {
            for (int j = 0; j < 2; j++)
            {
                for (int i = 0; i < 50_000; i++)
                {
                    ndarray.GetInt32(j, 0, i);
                }
            }
        }
    }
}
