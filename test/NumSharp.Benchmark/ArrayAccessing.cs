using System;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;

namespace NumSharp.Benchmark
{
    //[RPlotExporter, RankColumn]
    [SimpleJob(RunStrategy.ColdStart, targetCount: 20)]
    [SimpleJob(RunStrategy.Throughput, targetCount: 20)]
    [MinColumn, MaxColumn, MeanColumn, MedianColumn]
    [HtmlExporter]
    public class ArrayAccessing
    {
        public double[] genericArray;
        public Array nonGenericArray;

        [GlobalSetup]
        public void Setup()
        {
            var rnd = new Randomizer(42);
            // first array
            nonGenericArray = genericArray = new double[10_000_000];

            for (int i = 0; i < genericArray.Length; i++)
            {
                genericArray[i] = rnd.NextDouble();
            }
        }

        [Benchmark(Baseline = true)]
        public void GenericAccess()
        {
            var length = genericArray.Length;
            for (int i = 0; i < length; i++)
            {
                var a = genericArray[i];
            }
        }

        [Benchmark]
        public void NonGenericAccess()
        {
            double sum = 0;
            var length = nonGenericArray.Length;
            for (int i = 0; i < length; i++)
            {
                var a = nonGenericArray.GetValue(i);
            }
        }
    }
}
