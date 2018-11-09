using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NumSharp.Extensions;

namespace NumSharp.Benchmark
{
    [SimpleJob(RunStrategy.ColdStart, targetCount: 5)]
    [MinColumn, MaxColumn, MeanColumn, MedianColumn]
    public class npamin
    {
        private NumPy<double> np;
        private NDArray<double> nd;

        [GlobalSetup]
        public void Setup()
        {
            np = new NumPy<double>();
            nd = np.arange(240).reshape(20, 3, 2, 2);
        }

        [Benchmark]
        public void amin()
        {
            np.amin(nd);
        }

        [Benchmark]
        public void amin0axis()
        {
            np.amin(nd, 0);
        }

        [Benchmark]
        public void amin3axis()
        {
            np.amin(nd, 3);
        }
    }
}
