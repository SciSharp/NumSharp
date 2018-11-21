using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NumSharp.Core;

namespace NumSharp.Benchmark
{
    [SimpleJob(RunStrategy.ColdStart, targetCount: 10)]
    [MinColumn, MaxColumn, MeanColumn, MedianColumn]
    public class nparange
    {
        private NumPy<int> np;
        private NDArray<int> nd;

        private int start;
        private int step;
        private int length;

        [GlobalSetup]
        public void Setup()
        {
            np = new NumPy<int>();
            nd = new NDArray<int>();
            start = 0;
            step = 1;
            length = 100 * 100;
        }

        [Benchmark]
        public void arange()
        {
            np.arange(length);
        }

        [Benchmark]
        public void arange_ndarraywithdtype()
        {
            var nd2 = new NDArrayWithDType(Core.NumSharp.int16);
            var nd3 = nd2.arange(length, start, step);
        }

        [Benchmark]
        public void arange_ndarray()
        {
            var nd2 = new NDArray<int>();
            var nd3 = nd2.arange(length, start, step);
        }

        [Benchmark]
        public void arange_raw()
        {
            int index = 0;

            Enumerable.Range(start, length)
                .Where(x => index++ % step == 0)
                .ToArray();
        }
    }
}
