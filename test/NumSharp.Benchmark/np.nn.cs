using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using NumSharp.Backends.Unmanaged;
using NumSharp.Memory.Pooling;
using NumSharp.Unmanaged.Memory;
using NumSharp.Utilities;

namespace NumSharp.Benchmark
{
    [SimpleJob(RunStrategy.ColdStart, targetCount: 10)]
    [MinColumn, MaxColumn, MeanColumn, MedianColumn]
    [HtmlExporter]
    public class npnn
    {
        NDArray x;
        [GlobalSetup]
        public void Setup()
        {
            x = np.arange(1.0d, 1000.0d);
        }

        [Benchmark]
        public void Sigmoid()
        {
            for (int i = 0; i < 1000; i++)
            {
                var sigmoid = 1.0d / (1.0d + np.exp(-x));
            }
        }
    }
}
