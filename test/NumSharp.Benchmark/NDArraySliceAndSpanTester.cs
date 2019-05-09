
using System;
using System.Numerics;
using System.Linq;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using BenchmarkDotNet.Running;
using NumSharp;
using NumSharp.Generic;

namespace NumSharp.Benchmark
{
    //[RPlotExporter, RankColumn]
    [SimpleJob(RunStrategy.ColdStart, targetCount: 20)]
    [MinColumn, MaxColumn, MeanColumn, MedianColumn]
    [HtmlExporter]
    public class NDArraySliceAndSpanTester
    {

        public double[] A1;
        public Memory<double> M1;
        public Memory<double> M2;
        public NDArray N1;
        public NDArray<double> ND1;
        public NDArray NS1;
        public NDArray<double> NDS1;
        public NDArray NS2;
        public NDArray<double> NDS2;

        private Span<double> GetFullA1Span()
        {
            return new Span<double>(A1);
        }
        private Span<double> GetPartialA1Span()
        {
            return new Span<double>(A1, 1, A1.Length-2);
        }

        [GlobalSetup]
        public void Setup()
        {
            var arraySize = 100000;
            A1 = new double[arraySize].Select((x, idx) => x + idx).ToArray();
            M1 = new Memory<double>(A1);
            M2 = new Memory<double>(A1, 2, A1.Length - 2);
            N1 = new NDArray(new double[arraySize].Select((x, idx) => x + idx).ToArray());
            ND1 = new NDArray(new double[arraySize].Select((x, idx) => x + idx).ToArray()).MakeGeneric<double>();
            NS1 = N1["1:" + (N1.size-2)];
            NDS1 = ND1["1:" + (ND1.size - 2)];
            NS2 = N1["1:" + (N1.size - 2) + ":2"];
            NDS2 = ND1["1:" + (ND1.size - 2) + ":2"];
        }
        [Benchmark(Description = "C# double[] initialized in a for loop")]
        public void Array1InitInForLoop()
        {
            //unchecked
            {
                var aLength = A1.Length;
                for (int idx = 0; idx < aLength; idx++)
                    A1[idx] = 1.0;
            }
        }
        [Benchmark(Description = "C# double[] initialized in a fixed for loop")]
        public unsafe void Array1InitInFixedForLoop()
        {
            fixed (double *pA1 = A1)
            {
                var aLength = A1.Length;
                for (int idx = 0; idx < aLength; idx++)
                    pA1[idx] = 1.0;
            }
        }

        [Benchmark(Description = "C# Memory<double> initialized in a for loop")]
        public void Memory1InitInForLoop()
        {
            var MS1 = M1.Span;
            //unchecked
            {
                for (int idx = 0; idx < M1.Length; idx++)
                    MS1[idx] = 1.0;
            }
        }
        [Benchmark(Description = "C# Memory<double>(1:-1).Span initialized in a for loop")]
        public void Memory2InitInForLoop()
        {
            var MS2 = M2.Span;
            //unchecked
            {
                for (int idx = 0; idx < M2.Length; idx++)
                    MS2[idx] = 1.0;
            }
        }

        [Benchmark(Description = "C# Span<double> initialized in a for loop")]
        public void Span1InitInForLoop()
        {
            var S1 = GetFullA1Span();
            //unchecked
            {
                for (int idx = 0; idx < S1.Length; idx++)
                    S1[idx] = 1.0;
            }
        }
        [Benchmark(Description = "C# Span<double>(1:-1) initialized in a for loop")]
        public void Span2InitInForLoop()
        {
            var S2 = GetPartialA1Span();
            //unchecked
            {
                for (int idx = 0; idx < S2.Length; idx++)
                    S2[idx] = 1.0;
            }
        }

        [Benchmark(Description = "C# Span<double>(1:-1) initialized in a fixed for loop")]
        public unsafe void Span2InitInFixedForLoop()
        {
            var S2 = GetPartialA1Span();
            fixed (double *pS2 = S2)
            {
                var aLength = S2.Length;
                for (int idx = 0; idx < aLength; idx++)
                    pS2[idx] = 1.0;
            }
        }



        [Benchmark(Description = "C# Span<double> initialized in a for loop via array copy")]
        public void Span1AsArrayInitInForLoop()
        {
            var S1 = GetFullA1Span();
            var S1Copy = S1.ToArray();
            //unchecked
            {
                for (int idx = 0; idx < S1Copy.Length; idx++)
                    S1Copy[idx] = 1.0;
                for (int idx = 0; idx < S1Copy.Length; idx++)
                    S1[idx] = S1Copy[idx];
            }
        }
        [Benchmark(Description = "C# Span<double>(1:-1) initialized in a for loop via array copy")]
        public void Span2AsArrayInitInForLoop()
        {
            var S2 = GetPartialA1Span();
            var S2Copy = S2.ToArray();
            //unchecked
            {
                for (int idx = 0; idx < S2Copy.Length; idx++)
                    S2Copy[idx] = 1.0;
                for (int idx = 0; idx < S2Copy.Length; idx++)
                    S2[idx] = S2Copy[idx];
            }
        }
        [Benchmark(Description = "NDArray initialized in a for loop")]
        public void NDArray1InitInForLoop()
        {
            //var A1 = N1.Array;
            //unchecked
            {
                for (int idx = 0; idx < N1.size; idx++)
                    N1[idx] = 1.0;
            }
        }
        [Benchmark(Description = "NDArray<double> initialized in a for loop")]
        public void NDArrayDouble1InitInForLoop()
        {
            //var A1 = N1.Array;
            //unchecked
            {
                for (int idx = 0; idx < ND1.size; idx++)
                    ND1[idx] = 1.0;
            }
        }
        [Benchmark(Description = "NDArray[1:-1] initialized in a for loop")]
        public void NDArraySlice1InitInForLoop()
        {
            //var A1 = N1.Array;
            //unchecked
            {
                for (int idx = 0; idx < NS1.size; idx++)
                    NS1[idx] = 1.0;
            }
        }


        [Benchmark(Description = "NDArray<double>[1:-1] initialized in a for loop")]
        public void NDArrayDoubleSlice1InitInForLoop()
        {
            //var A1 = N1.Array;
            //unchecked
            {
                for (int idx = 0; idx < NDS1.size; idx++)
                    NDS1[idx] = 1.0;
            }
        }

        [Benchmark(Description = "NDArray[1:-1:2] initialized in a for loop")]
        public void NDArraySlice2InitInForLoop()
        {
            //var A1 = N1.Array;
            //unchecked
            {
                for (int idx = 0; idx < NS2.size; idx++)
                    NS2[idx] = 1.0;
            }
        }


        [Benchmark(Description = "NDArray<double>[1:-1:2] initialized in a for loop")]
        public void NDArrayDoubleSlice2InitInForLoop()
        {
            //var A1 = N1.Array;
            //unchecked
            {
                for (int idx = 0; idx < NDS2.size; idx++)
                    NDS2[idx] = 1.0;
            }
        }

        [Benchmark(Description = "NDArray[1:-1].ToArray() and assign index 0")]
        public void NDArraySlice1ToArray()
        {
            var A1 = NS1.Array;
            A1.SetValue(1.1, 0);
        }
        [Benchmark(Description = "NDArray<double>[1:-1].ToArray() and assign index 0")]
        public void NDArrayDoubleSlice1ToArray()
        {
            var A1 = NDS1.Array;
            A1.SetValue(1.1, 0);
        }

        [Benchmark(Description = "NDArray[1:-1:2].ToArray() and assign index 0")]
        public void NDArraySlice2ToArray()
        {
            var A2 = NS2.Array;
            A2.SetValue(1.1, 0);
        }
        [Benchmark(Description = "NDArray<double>[1:-1:2].ToArray() and assign index 0")]
        public void NDArrayDoubleSlice2ToArray()
        {
            var A2 = NDS2.Array;
            A2.SetValue(1.1, 0);
        }


    }
}
