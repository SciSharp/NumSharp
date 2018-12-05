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
        private NumPy np;
        private NDArray nd;

        private int start;
        private int step;
        private int length;

        private NumPy np2;

        [GlobalSetup]
        public void Setup()
        {
            np = new NumPy();
            nd = new NDArray(typeof(int));
            start = 0;
            step = 1;
            length = 100 * 100;

            np2 = new NumPy();
        }

        [Benchmark]
        public void arange()
        {
            np.arange(length);
        }

        [Benchmark]
        public void arange_ndarray()
        {
            var nd3 = nd.arange(length, start, step);
        }

        [Benchmark]
        public void arange_ndarray2()
        {
            var nd3 = np2.arange(start, length, step);
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
