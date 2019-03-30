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
    public class npindex
    {
        private NDArray nd;

        private Shape shape;

        [GlobalSetup]
        public void Setup()
        {
            shape = new Shape(1000, 1000);
            nd = np.arange(1000 * 1000 * 1.0).reshape(shape);
        }

        [Benchmark]
        public void accessInDtype()
        {
            double a = 0;

            for (int d1 = 0; d1 < shape.Dimensions[0]; d1++)
            {
                for (int d2 = 0; d2 < shape.Dimensions[1]; d2++)
                {
                    a = (double)nd[d1, d2];
                }
            }
        }

        [Benchmark]
        public void accessInGeneric()
        {
            double a = 0;

            for (int d1 = 0; d1 < shape.Dimensions[0]; d1++)
            {
                for (int d2 = 0; d2 < shape.Dimensions[1]; d2++)
                {
                    a = nd.Storage.GetData<double>(d1, d2);
                }
            }
        }

        [Benchmark]
        public void accessInObject()
        {
            object a = 0;

            for (int d1 = 0; d1 < shape.Dimensions[0]; d1++)
            {
                for (int d2 = 0; d2 < shape.Dimensions[1]; d2++)
                {
                    a = nd[d1, d2];
                }
            }
        }
    }
}
