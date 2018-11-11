
using System;
using System.Numerics;
using System.Linq;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using BenchmarkDotNet.Running;
using NumSharp.Extensions;

namespace NumSharp.Benchmark
{
    //[RPlotExporter, RankColumn]
    [SimpleJob(RunStrategy.ColdStart, targetCount: 20)]
    [MinColumn, MaxColumn, MeanColumn, MedianColumn]
    [HtmlExporter]
    public class NDArrayTester2D
    {
        public NDArray<double> np1Matrix;
        public NDArray<double> np2Matrix;
        public NDArray<double> np3Matrix;
        public double[][] np1DoubleMatrix;
        public double[][] np2DoubleMatrix;
        public double[][] np3DoubleMatrix;
        [GlobalSetup]
        public void Setup()
        {
            // first array
            np1Matrix = new NDArray<double>().arange(100 * 100+2,2).reshape(100,100);

            np1DoubleMatrix = np1Matrix.ToDotNetArray<double[][]>();

            // second array
            np2Matrix = new NDArray<double>().arange(100*100+1,1).reshape(100,100);;
            
            np2DoubleMatrix = np2Matrix.ToDotNetArray<double[][]>();
        }
        [Benchmark]
        public void DirectAddition1D()
        {
            for(int idx = 0; idx < np1DoubleMatrix.Length;idx++)
                for(int jdx = 0; jdx < np1DoubleMatrix[0].Length;jdx++)
                    np3DoubleMatrix[idx][jdx] = np1DoubleMatrix[idx][jdx] + np2DoubleMatrix[idx][jdx];
        } 
        [Benchmark]
        public void NDArrayAddition1D()
        {
            np3Matrix = np1Matrix + np2Matrix;
        }
        [Benchmark]
        public void DirectSubstration1D()
        {
            for(int idx = 0; idx < np1DoubleMatrix.Length;idx++)
                for(int jdx = 0; jdx < np1DoubleMatrix[0].Length;jdx++)
                    np3DoubleMatrix[idx][jdx] = np1DoubleMatrix[idx][jdx] - np2DoubleMatrix[idx][jdx];
        } 
        [Benchmark]
        public void NDArraySubstraction1D()
        {
            np3Matrix = np1Matrix - np2Matrix;
        }
        [Benchmark]
        public void DirectMatrixMultiplication()
        {
            np3DoubleMatrix = new double[np1DoubleMatrix.Length][];

            for (int idx = 0; idx < np3DoubleMatrix.Length;idx++)
            {
                np3DoubleMatrix[idx] = new double[np1DoubleMatrix[0].Length];
            }

            for (int idx = 0; idx < np1DoubleMatrix.Length; idx++)
            {
                for (int jdx = 0; jdx < np1DoubleMatrix[0].Length;jdx++)
                {
                    np3DoubleMatrix[idx][jdx] = 0;
                    for (int kdx = 0; kdx < np2DoubleMatrix.Length;kdx++)
                    {
                        np3DoubleMatrix[idx][jdx] += np1DoubleMatrix[idx][kdx] * np2DoubleMatrix[kdx][jdx];
                    }
                }
            }
        }
        [Benchmark]
        public void NDArrayMatrixMultilication()
        {
            np3Matrix = np1Matrix.dot(np2Matrix);
        }
    }
}
