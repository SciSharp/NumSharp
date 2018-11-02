
using System;
using System.Linq;

namespace NumSharp.Benchmark
{
    public static class ArrayTester
    {
        public static void Access()
        {
            var A = new int[10000,10000];
            int ALength = 10000 * 10000;

            for (int idx = 0; idx < ALength;idx++)
            {
                int dim0 = idx / 10000;
                int dim1 = idx % 10000;
                A[dim0,dim1] = idx;
            }

            var watcher1 = new System.Diagnostics.Stopwatch();

            watcher1.Start();

            for (int idx = 0; idx < ALength;idx++)
            {
                int dim0 = idx / 10000;
                int dim1 = idx % 10000;
                int puffer = A[dim0,dim1];
                A[dim0,dim1] = puffer;
            }

            watcher1.Stop();

            A = null;

            var B = new int[10000][];

            B = B.Select(x => new int[10000]).ToArray();

            for (int idx = 0; idx < ALength;idx++)
            {
                int dim0 = idx / 10000;
                int dim1 = idx % 10000;
                B[dim0][dim1] = idx;
            }

            for (int idx = 0; idx < ALength;idx++)
            {
                int dim0 = idx / 10000;
                int dim1 = idx % 10000;
                B[dim0][dim1] = idx;
            }
            
            var watcher2 = new System.Diagnostics.Stopwatch();

            watcher2.Start();

            for (int idx = 0; idx < ALength;idx++)
            {
                int dim0 = idx / 10000;
                int dim1 = idx % 10000;
                int puffer = B[dim0][dim1];
                B[dim0][dim1] = puffer;
            }

            watcher2.Stop();

            B = null;

            var C = new int[ALength];

            C = C.Select((x,idx) => idx).ToArray();

            var watcher3 = new System.Diagnostics.Stopwatch();

            watcher3.Start();

            for (int idx = 0; idx < ALength;idx++)
            {
                int dim0 = idx / 10000;
                int dim1 = idx % 10000;
                int puffer = C[idx];
                C[idx] = puffer;
            }

            watcher3.Stop();

        }
        public static void CheckPlusOperation()
        {
            var A = new double[1000][];
            A = A.Select(x => new double[1000]).Select((x,idx) => x.Select((y,jdx) => (double)(idx + jdx) ).ToArray()).ToArray();
            
            var B = new double[1000][];
            B = B.Select(x => new double[1000]).Select((x,idx) => x.Select((y,jdx) => (double)(idx + jdx) ).ToArray()).ToArray();
            
            var watch1 = new System.Diagnostics.Stopwatch();
            watch1.Start();

            var C = A.Select((x,idx) => x.Select((y,jdx) => y + A[idx][jdx]).ToArray()).ToArray();

            watch1.Stop();

            A = null;
            B = null;

            var a = new double[1000*1000];
            a = a.Select((x,idx) => (double)idx).ToArray();

            var b = new double[1000*1000];
            b = b.Select((x,idx) => (double)idx).ToArray();

            var watch2 = new System.Diagnostics.Stopwatch();
            watch2.Start();

            var c = a.Select((x,idx) => x + b[idx]).ToArray();

            watch2.Stop();

            a = null;
            b = null;


        }
        public static void CheckMatrixMultiplication()
        {
            var A = new double[1000][];
            A = A.Select(x => new double[1000]).Select((x,idx) => x.Select((y,jdx) => (double)(idx + jdx) ).ToArray()).ToArray();
            
            var B = new double[1000][];
            B = B.Select(x => new double[1000]).Select((x,idx) => x.Select((y,jdx) => (double)(idx + jdx) ).ToArray()).ToArray();
           
            int numOfLines = A.GetLength(0);
            int numOfColumns = A[0].GetLength(0);

            int iterator = B[0].GetLength(0);

            var result = new double[numOfLines][];
            result = result.Select(x => new double[numOfColumns]).ToArray();

            var watch1 = new System.Diagnostics.Stopwatch();
            watch1.Start();

            for (int idx = 0; idx < numOfLines;idx++)
            {
                for( int jdx = 0; jdx < numOfColumns; jdx++)
                {
                    result[idx][jdx] = 0;
                    for (int kdx = 0; kdx < iterator; kdx++)
                    {
                        result[idx][jdx] += A[idx][kdx] * B[kdx][jdx];
                    }
                }
            }

            watch1.Stop();

            var a = new double[]{1,2,3,4,5,6,7,8,9};
            var b = new double[]{8,0,4,2,1,0,8,1,0};

            var c = new double[9];

            for (int idx = 0; idx < 9;idx++)
            {
                int line = idx % 3;
                int column = idx / 3;

                c[idx] = 0;
                for (int kdx = 0; kdx < 3;kdx++)
                {
                    c[idx] += a[line + kdx * 3] * b[3 * column + kdx];    
                }
            }
            
            a = new double[1000*1000];
            a = a.Select((x,idx) => (double)idx).ToArray();

            b = new double[1000*1000];
            b = b.Select((x,idx) => (double)idx).ToArray();

            var resultArr = new double[a.Length];

            var watch2 = new System.Diagnostics.Stopwatch();
            watch2.Start();


            for (int idx = 0; idx < a.Length;idx++)
            {
                int line = idx % 1000;
                int column = idx / 1000;

                resultArr[idx] = 0;
                for (int kdx = 0; kdx < iterator;kdx++)
                {
                    resultArr[idx] += a[line + kdx * 1000] * b[1000 * column + kdx];
                }
            }
            
            watch2.Stop();

        }
    }
}