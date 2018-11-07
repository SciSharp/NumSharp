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
    public class nparrange
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
        public void arange_ndarray()
        {
            var n = new NDArray<int>();
            n.ARange(length, start, step);
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
