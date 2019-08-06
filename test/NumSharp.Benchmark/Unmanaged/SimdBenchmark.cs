using System.Numerics;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;

namespace NumSharp.Benchmark
{
    [SimpleJob(RunStrategy.ColdStart, targetCount: 20)]
    [SimpleJob(RunStrategy.Throughput, targetCount: 20)]
    [MinColumn, MaxColumn, MeanColumn, MedianColumn]
    [HtmlExporter]
    public class SimdBenchmark
    {
        private const int _items = 10000;
        private double[] genericArray;
        private double[] genericArray2;

        [GlobalSetup]
        public void Setup()
        {
            var rnd = new Randomizer(42);
            // first array
            genericArray = new double[_items];
            genericArray2 = new double[_items];

            for (int i = 0; i < _items; i++)
            {
                genericArray[i] = rnd.NextDouble();
                genericArray2[i] = rnd.NextDouble();
            }
        }


        [Benchmark(Baseline = true)]
        public double DotDouble()
        {
            double returnVal = 0.0;
            for (int i = 0; i < _items; i++)
            {
                returnVal += genericArray[i] * genericArray2[i];
            }

            return returnVal;
        }

        [Benchmark]
        public double DotDoubleVectorNaive()
        {
            double returnVal = 0.0;
            for (int i = 0; i < _items - 4; i += 1)
            {
                returnVal += Vector.Dot(new Vector<double>(genericArray, i), new Vector<double>(genericArray2, i));
            }

            return returnVal;
        }

        [Benchmark]
        public double DotDoubleVectorBetter()
        {
            Vector<double> sumVect = Vector<double>.Zero;
            for (int i = 0; i < _items - 4; i += 1)
            {
                sumVect += new Vector<double>(genericArray, i) * new Vector<double>(genericArray2, i);
            }

            return Vector.Dot(sumVect, Vector<double>.One);
        }
    }
}
