using System.Linq;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

using NumSharp.Unmanaged.Memory;
using NumSharp.Utilities;

namespace NumSharp.Benchmark.Unmanaged
{
    //|            Method |       Mean |    Error |   StdDev |     Median |        Min |        Max | Ratio | RatioSD |
    //|------------------ |-----------:|---------:|---------:|-----------:|-----------:|-----------:|------:|--------:|
    //|        GetAtIndex |        ... |      ... |      ... |        ... |        ... |        ... |   ... |     ... |
    //|         GetOffset |   745.3 us | 6.163 us | 83.64 us |   728.0 us |   685.1 us | 2,746.7 us |  1.00 |    0.00 |

    // NDIterator (the legacy per-element offset incrementor benchmarked here as
    // "OffsetIncrementor") has been removed in favor of NDIter/NDIterRef. NDIterRef
    // is a ref struct and cannot be held in a [GlobalSetup] field, so the flat walk is
    // now measured through NDArray.GetAtIndex — the public C-order element accessor that
    // replaced AsIterator in ToString and np.broadcast(...).iters — against coordinate
    // GetOffset access.
    [SimpleJob(RunStrategy.ColdStart, targetCount: 2000)]
    [MinColumn, MaxColumn, MeanColumn, MedianColumn]
    [HtmlExporter]
    public class Iterators
    {
        private Shape shape;
        private NDArray ndarray;

        [GlobalSetup]
        public void Setup()
        {
            var _ = ScalarMemoryPool.Instance;
            var __ = InfoOf<byte>.Size;
            shape = new Shape(2, 1, 50_000);
            ndarray = np.array(Enumerable.Range(0, 100_000).ToArray()).reshape(ref shape);
        }

        [Benchmark]
        public void GetAtIndex()
        {
            for (long i = 0; i < 100_000; i++)
            {
                ndarray.GetAtIndex(i);
            }
        }

        [Benchmark(Baseline = true)]
        public void GetOffset()
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
