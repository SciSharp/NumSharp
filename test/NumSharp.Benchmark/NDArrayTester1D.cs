using System;
using System.Numerics;
using System.Linq;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using BenchmarkDotNet.Running;
using NumSharp;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Benchmark
{
    //[RPlotExporter, RankColumn]
    [SimpleJob(RunStrategy.ColdStart, targetCount: 20)]
    [MinColumn, MaxColumn, MeanColumn, MedianColumn]
    [HtmlExporter]
    public class NDArrayTester1D
    {
        public NDArray np1;
        public NDArray np2;
        public NDArray np3;
        public ArraySlice<double> np1Double;
        public ArraySlice<double> np2Double;
        public ArraySlice<double> np3Double;

        [GlobalSetup]
        public void Setup()
        {
            // first array
            np1 = new NDArray(new double[100000].Select((x, idx) => x + idx).ToArray());
            np1Double = np1.Data<double>();

            // second array
            np2 = new NDArray(new double[100000].Select((x, idx) => x + idx).ToArray());
            np2Double = np2.Data<double>();

            // result array
            np3 = new NDArray(new double[100000]);
            np3Double = np3.Data<double>();

            // result array
            np3 = new NDArray(new double[100000]);
            np3Double = np3.Data<double>();
        }

        [Benchmark]
        public void DirectAddition1D()
        {
            for (int idx = 0; idx < np3Double.Length; idx++)
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
            for (int idx = 0; idx < np3Double.Length; idx++)
                np3Double[idx] = np1Double[idx] - np2Double[idx];
        }

        [Benchmark]
        public void NDArraySubstraction1D()
        {
            np3 = np1 - np2;
        }
    }
}
