
using System;
using System.Numerics;
using System.Linq;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using BenchmarkDotNet.Running;
using NumSharp.Core;

namespace NumSharp.Benchmark
{
    //[RPlotExporter, RankColumn]
    [SimpleJob(RunStrategy.ColdStart, targetCount: 20)]
    [MinColumn, MaxColumn, MeanColumn, MedianColumn]
    [HtmlExporter]
    public class NDArrayTester1D
    {
        public NDArrayGeneric<double> np1;
        public NDArrayGeneric<double> np2;
        public NDArrayGeneric<double> np3;
        public double[] np1Double;
        public double[] np2Double;
        public double[] np3Double;
        [GlobalSetup]
        public void Setup()
        {
            // first array
            np1 = new NDArrayGeneric<double>();
            np1.Data = new double[100000].Select((x,idx) => x + idx ).ToArray();
            np1.Shape = new Shape(np1.Data.Length);

            np1Double = np1.Data;
            
            // second array
            np2 = new NDArrayGeneric<double>();
            np2.Data = new double[100000].Select((x,idx) => x + idx  ).ToArray();
            np2.Shape = new Shape(np2.Data.Length);

            np2Double = np2.Data;

            // result array
            np3 = new NDArrayGeneric<double>();
            np3.Data = new double[100000];

            np3Double = np3.Data;

            // result array
            np3 = new NDArrayGeneric<double>();
            np3.Data = new double[100000];

            np3Double = np3.Data;
        }
        [Benchmark]
        public void DirectAddition1D()
        {
            for(int idx = 0; idx < np3Double.Length;idx++)
                np3Double[idx] = np1Double[idx] + np2Double[idx];
        } 
        [Benchmark]
        public void NDArrayAddition1D()
        {
            np3 = np1 + np2;
        }
        [Benchmark]
        public void DirectSubstration1D()
        {
            for(int idx = 0; idx < np3Double.Length;idx++)
                np3Double[idx] = np1Double[idx] - np2Double[idx];
        } 
        [Benchmark]
        public void NDArraySubstraction1D()
        {
            np3 = np1 - np2;
        }
    }
}
