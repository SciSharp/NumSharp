
using System;
using System.Numerics;
using System.Linq;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using BenchmarkDotNet.Running;

namespace NumSharp.Benchmark
{
    //[RPlotExporter, RankColumn]
    [SimpleJob(RunStrategy.ColdStart, targetCount: 20)]
    [MinColumn, MaxColumn, MeanColumn, MedianColumn]
    [HtmlExporter]
    public class LinqTesterDouble
    {
        public double[] np1;
        public double[] np2;
        public double scalar;
        public double[] result;
        [GlobalSetup]
        public void Setup()
        {
            // first array
            np1 = new double[100 * 100];
            np1 = np1.Select((x,idx) => (double) idx).ToArray();

            // second array
            np2 = new double[100 * 100];
            np2 = np2.Select((x,idx) => (double) (idx + 1)).ToArray();
            
            // scalar 
            scalar = 25;
        }
        [Benchmark(Baseline = true)]
        public void NormalForLoopWithPreDefineResultArray()
        {
            result = new double[np1.Length];

            // 2 arrays
            for (int idx = 0; idx < np1.Length;idx++)
            {
                result[idx] = np1[idx] + np2[idx];
            }
            // 1 array and scalar
            for (int idx = 0; idx < np1.Length;idx++)
            {
                result[idx] = np1[idx] + scalar;
            }
        }
        [Benchmark]
        public void LinqSelectWithPreDefineResultArray()
        {
            result = new double[np1.Length];

            // 2 arrays
            result = np1.Select((x,idx) => x + np2[idx]).ToArray();
            // 1 array and scalar
            result = np1.Select((x,idx) => x + scalar).ToArray();
        }
        [Benchmark]
        public void LinqSelectWithoutPreDefineResultArray()
        {
            // 2 arrays
            result = np1.Select((x,idx) => x + np2[idx]).ToArray();
            // 1 array and scalar
            result = np1.Select((x,idx) => x + scalar).ToArray();
        }
    }
}
