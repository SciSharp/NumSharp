using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using NumSharp.Backends.Unmanaged;
using NumSharp.Memory.Pooling;
using NumSharp.Unmanaged.Memory;
using NumSharp.Utilities;

namespace NumSharp.Benchmark
{
    [SimpleJob(RunStrategy.ColdStart, targetCount: 50)]
    [MinColumn, MaxColumn, MeanColumn, MedianColumn]
    [HtmlExporter]
    public class npadd
    {
        double a;
        double b;
        double c;

        [GlobalSetup]
        public void Setup()
        {
            var _ = ScalarMemoryPool.Instance;
            var __ = InfoOf<byte>.Size;
            var ___ = InfoOf<double>.Size;
            b = 3;
            a = 0; 
            c = 5;
        }

        [Benchmark]
        public void NDArray_()
        {
            NDArray a;
            for (double i = 0; i < 10000; i++)
            {
                a = NDArray.Scalar(b) + NDArray.Scalar(c);
            }
        }

        [Benchmark(Baseline = true)]
        public void Direct()
        {
            for (double i = 0; i < 10000; i++)
            {
                a = b + c;
            }
        }
    }
}
