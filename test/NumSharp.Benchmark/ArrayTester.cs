
using System;
using System.Linq;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using BenchmarkDotNet.Running;

namespace NumSharp.Benchmark
{
    //[RPlotExporter, RankColumn]
    [SimpleJob(RunStrategy.ColdStart, targetCount: 5)]
    [MinColumn, MaxColumn, MeanColumn, MedianColumn]
    public class ArrayTester
    {
        public int[,] A;

        [Params(10, 100)]//0, 10000)]
        public int N { get; set; }

        [GlobalSetup]
        public void Setup()
        {
            A = new int[N, N];
        }

        [Benchmark]
        public void Access1()
        {
            //var A = new int[N,N];
            int ALength = N * N;

            for (int idx = 0; idx < ALength;idx++)
            {
                int dim0 = idx / N;
                int dim1 = idx % N;
                A[dim0,dim1] = idx;
            }
        }

        [Benchmark]
        public void Access2()
        {
            //var A = new int[N, N];
            int ALength = N * N;

            for (int idx = 0; idx < ALength; idx++)
            {
                int dim0 = idx / N;
                int dim1 = idx % N;
                int puffer = A[dim0, dim1];
                A[dim0, dim1] = puffer;
            }
        }

        [Benchmark]
        public void Access3()
        {
            //var A = new int[N, N];
            int ALength = N * N;

            A = null;

            var B = new int[N][];

            B = B.Select(x => new int[N]).ToArray();

            for (int idx = 0; idx < ALength; idx++)
            {
                int dim0 = idx / N;
                int dim1 = idx % N;
                B[dim0][dim1] = idx;
            }
        }

        [Benchmark]
        public void Access4()
        {
            //var A = new int[N, N];
            int ALength = N * N;

            A = null;

            var B = new int[N][];

            B = B.Select(x => new int[N]).ToArray();

            for (int idx = 0; idx < ALength; idx++)
            {
                int dim0 = idx / N;
                int dim1 = idx % N;
                int puffer = B[dim0][dim1];
                B[dim0][dim1] = puffer;
            }
        }

        [Benchmark]
        public void Access5()
        {
            //var A = new int[N, N];
            int ALength = N * N;

            var C = new int[ALength];

            C = C.Select((x, idx) => idx).ToArray();

            for (int idx = 0; idx < ALength; idx++)
            {
                int dim0 = idx / N;
                int dim1 = idx % N;
                int puffer = C[idx];
                C[idx] = puffer;
            }
        }


        [Benchmark]
        public void CheckPlusOperation1()
        {
            var A = new double[N][];
            A = A.Select(x => new double[N]).Select((x, idx) => x.Select((y, jdx) => (double)(idx + jdx)).ToArray()).ToArray();

            var B = new double[N][];
            B = B.Select(x => new double[N]).Select((x, idx) => x.Select((y, jdx) => (double)(idx + jdx)).ToArray()).ToArray();

            var C = A.Select((x, idx) => x.Select((y, jdx) => y + A[idx][jdx]).ToArray()).ToArray();
        }

        [Benchmark]
        public void CheckPlusOperation2()
        {
            var a = new double[N * N];
            a = a.Select((x, idx) => (double)idx).ToArray();

            var b = new double[N * N];
            b = b.Select((x, idx) => (double)idx).ToArray();

            var c = a.Select((x, idx) => x + b[idx]).ToArray();
        }

        [Benchmark]
        public void CheckMatrixMultiplication1()
        {
            var A = new double[N][];
            A = A.Select(x => new double[N]).Select((x, idx) => x.Select((y, jdx) => (double)(idx + jdx)).ToArray()).ToArray();

            var B = new double[N][];
            B = B.Select(x => new double[N]).Select((x, idx) => x.Select((y, jdx) => (double)(idx + jdx)).ToArray()).ToArray();

            int numOfLines = A.GetLength(0);
            int numOfColumns = A[0].GetLength(0);

            int iterator = B[0].GetLength(0);

            var result = new double[numOfLines][];
            result = result.Select(x => new double[numOfColumns]).ToArray();

            for (int idx = 0; idx < numOfLines; idx++)
            {
                for (int jdx = 0; jdx < numOfColumns; jdx++)
                {
                    result[idx][jdx] = 0;
                    for (int kdx = 0; kdx < iterator; kdx++)
                    {
                        result[idx][jdx] += A[idx][kdx] * B[kdx][jdx];
                    }
                }
            }
        }

        [Benchmark]
        public void CheckMatrixMultiplication2()
        {
            var B = new double[N][];
            B = B.Select(x => new double[N]).Select((x, idx) => x.Select((y, jdx) => (double)(idx + jdx)).ToArray()).ToArray();
            int iterator = B[0].GetLength(0);

            var a = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            var b = new double[] { 8, 0, 4, 2, 1, 0, 8, 1, 0 };

            var c = new double[9];

            for (int idx = 0; idx < 9; idx++)
            {
                int line = idx % 3;
                int column = idx / 3;

                c[idx] = 0;
                for (int kdx = 0; kdx < 3; kdx++)
                {
                    c[idx] += a[line + kdx * 3] * b[3 * column + kdx];
                }
            }

            a = new double[N * N];
            a = a.Select((x, idx) => (double)idx).ToArray();

            b = new double[N * N];
            b = b.Select((x, idx) => (double)idx).ToArray();

            var resultArr = new double[a.Length];

            for (int idx = 0; idx < a.Length; idx++)
            {
                int line = idx % N;
                int column = idx / N;

                resultArr[idx] = 0;
                for (int kdx = 0; kdx < iterator; kdx++)
                {
                    resultArr[idx] += a[line + kdx * N] * b[N * column + kdx];
                }
            }
        }
    }
}
