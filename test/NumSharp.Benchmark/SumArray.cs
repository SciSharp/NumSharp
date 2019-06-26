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
    [SimpleJob(RunStrategy.Throughput, targetCount: 20)]
    [MinColumn, MaxColumn, MeanColumn, MedianColumn]
    [HtmlExporter]
    public class SumArray
    {
        public double[] np1;
        public double[] np2;

        [GlobalSetup]
        public void Setup()
        {
            var rnd = new Random(42);
            // first array
            np1 = new double[10_000_000];

            for (int i = 0; i < np1.Length; i++)
            {
                np1[i] = rnd.NextDouble();
            }
        }

        [Benchmark(Baseline = true)]
        public void ForLoop()
        {
            double sum = 0;
            for (int i = 0; i < np1.Length; i++)
            {
                sum += np1[i];
            }
        }        
        
        [Benchmark]
        public void Foreach()
        {
            double sum = 0;
            foreach (var v in np1) {
                sum += v;
            }
        }

        [Benchmark]
        public void ForLoopPredefined()
        {
            double sum = 0;
            int till = np1.Length;
            for (int i = 0; i < till; i++)
            {
                sum += np1[i];
            }
        }

        [Benchmark]
        public void ForLoopCheckedOuter()
        {
            double sum = 0;
            checked
            {
                for (int i = 0; i < np1.Length; i++)
                {
                    sum += np1[i];
                }
            }
        }

        [Benchmark]
        public void ForLoopCheckedInner()
        {
            double sum = 0;

            for (int i = 0; i < np1.Length; i++)
            {
                checked
                {
                    sum += np1[i];
                }
            }
        }

        [Benchmark]
        public void ForLoopUncheckedInner()
        {
            double sum = 0;

            for (int i = 0; i < np1.Length; i++)
            {
                unchecked
                {
                    sum += np1[i];
                }
            }
        }

        [Benchmark]
        public void ForLoopUncheckedOuter()
        {
            double sum = 0;
            unchecked
            {
                for (int i = 0; i < np1.Length; i++)
                {
                    sum += np1[i];
                }
            }
        }

        [Benchmark]
        public void Linq()
        {
            var sum = np1.Sum();
        }

        [Benchmark]
        public void Agregate()
        {
            var sum = np1.Sum();
        }
    }
}
