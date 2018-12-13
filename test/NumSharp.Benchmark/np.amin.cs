using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using NumSharp.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using np = NumSharp.Core.NumPy;

namespace NumSharp.Benchmark
{
    [SimpleJob(RunStrategy.ColdStart, targetCount: 10)]
    [MinColumn, MaxColumn, MeanColumn, MedianColumn]
    public class npamin
    {
        private NDArray nd;

        [GlobalSetup]
        public void Setup()
        {
            nd = np.arange(1000 * 8 * 8 * 8).reshape(1000, 8, 8, 8);
        }

        [Benchmark]
        public void min()
        {
            np.amin(nd);
        }

        [Benchmark]
        public void amin0axis()
        {
            var nd2 = new NDArray(typeof(double));
            nd2 = np.arange(1000 * 8 * 8 * 8).reshape(1000, 8, 8, 8);
            var nd3 = nd2.amin(0);
        }

        [Benchmark]
        public void amin2axis()
        {
            np.amin(nd, 2);
        }
    }
}
