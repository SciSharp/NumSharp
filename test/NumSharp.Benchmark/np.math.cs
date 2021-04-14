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
    public class npmath
    {
        double b;
        double c;
        NDArray x;
        NDArray y;
        double[] b1;

        [GlobalSetup]
        public void Setup()
        {
            var _ = ScalarMemoryPool.Instance;
            var __ = InfoOf<byte>.Size;
            var ___ = InfoOf<double>.Size;
            b = 3;
            c = 5;
            x = np.arange(1000.0d);
            y = np.arange(1.0d, 1001.0d);
            b1 = x.ToArray<double>();
        }

        [Benchmark(Baseline = true)]
        public void Direct()
        {
            var z = new double[1000];

            for (int i = 0; i < 1000; i++)
            {
                for (int j = 0; j < 1000; j++)
                {
                    z[j] = b1[j] - c;
                }
            }
        }

        [Benchmark]
        public void Add()
        {
            for (int i = 0; i < 1000; i++)
            {
                var z = x + y;
            }
        }

        [Benchmark]
        public void Subtract()
        {
            for (int i = 0; i < 1000; i++)
            {
                var z = x - y;
            }
        }

        [Benchmark]
        public void SubtractScalar()
        {
            for (int i = 0; i < 1000; i++)
            {
                var z = x - b;
            }
        }

        [Benchmark]
        public void Multiply()
        {
            for (int i = 0; i < 1000; i++)
            {
                var z = x * y;
            }
        }

        [Benchmark]
        public void Divide()
        {
            for (int i = 0; i < 1000; i++)
            {
                var z = x / y;
            }
        }
    }
}
