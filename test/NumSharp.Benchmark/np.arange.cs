using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NumSharp;

namespace NumSharp.Benchmark
{
    [SimpleJob(RunStrategy.ColdStart, targetCount: 10)]
    [MinColumn, MaxColumn, MeanColumn, MedianColumn]
    public class nparange
    {
        private NDArray nd;

        private int start;
        private int step;
        private int length;

        [GlobalSetup]
        public void Setup()
        {
            nd = new NDArray(typeof(int));
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
        public void arange_ndarray()
        {
            var nd3 = np.arange(start, length, step);
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
