using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using NumSharp.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Benchmark
{
    [SimpleJob(RunStrategy.ColdStart, targetCount: 10)]
    [MinColumn, MaxColumn, MeanColumn, MedianColumn]
    public class npamin
    {
        private NumPy<double> np;
        private NDArray<double> nd;

        [GlobalSetup]
        public void Setup()
        {
            np = new NumPy<double>();
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
            var nd2 = new NDArray<double>();
            nd2 = np.arange(1000 * 8 * 8 * 8).reshape(1000, 8, 8, 8);
            //var nd3 = nd2.AMin(0);
        }

        [Benchmark]
        public void amin0axisWithDType()
        {
            var nd2 = new NDArrayWithDType(NumPyWithDType.double8);
            nd2 = nd2.arange(1000 * 8 * 8 * 8, 0, 1).reshape(1000, 8, 8, 8);
            var nd3 = nd2.AMin(0);
        }

        [Benchmark]
        public void amin2axis()
        {
            np.amin(nd, 2);
        }
    }
}
