using System;
using System.Numerics;
using System.Linq;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using BenchmarkDotNet.Running;
using NumSharp;
using NumSharp.Backends.Unmanaged;
using NumSharp.Generic;

namespace NumSharp.Benchmark
{
    //[RPlotExporter, RankColumn]
    [SimpleJob(RunStrategy.ColdStart, targetCount: 20)]
    [MinColumn, MaxColumn, MeanColumn, MedianColumn]
    [HtmlExporter]
    public class NDArraySliceAndSpanTester
    {
        public ArraySlice<double> A1;
        public ArraySlice<double> ACopy1;
        public ArraySlice<double> ASlice1;
        public ArraySlice<double> ASliceStep2;
        public Memory<double> M1;
        public Memory<double> M2;
        public NDArray ND1;
        public NDArray NDSlice1;
        public NDArray NDSliceStep2;

        public NDArray<double> NDDouble1;
        public NDArray<double> NDDoubleSlice1;
        public NDArray<double> NDDoubleSliceStep2;

        private Span<double> GetFullA1Span()
        {
            return A1.AsSpan;
        }

        private Span<double> GetPartialA1Span()
        {
            unsafe {
                return new Span<double>(A1.Address + 1, A1.Count - 2);
            }
        }

        [GlobalSetup]
        public void Setup()
        {
            var arraySize = 100000;
            A1 = ArraySlice.FromArray(new double[arraySize].Select((x, idx) => x + idx).ToArray());
            ACopy1 = ArraySlice.FromArray(new double[arraySize].Select((x, idx) => x + idx).ToArray());
            ASlice1 = ArraySlice.FromArray(new double[arraySize - 2].Select((x, idx) => x + idx).ToArray());
            ASliceStep2 = ArraySlice.FromArray(new double[((arraySize - 2) + 1) / 2].Select((x, idx) => x + idx).ToArray());

            M1 = new Memory<double>(A1.ToArray());
            M2 = new Memory<double>(A1.ToArray(), 2, A1.Count - 2);
            ND1 = new NDArray(new double[arraySize].Select((x, idx) => x + idx).ToArray());
            NDSlice1 = ND1["1:" + (ND1.size - 2)];
            NDSliceStep2 = ND1["1:" + (ND1.size - 2) + ":2"];

            NDDouble1 = new NDArray(new double[arraySize].Select((x, idx) => x + idx).ToArray()).MakeGeneric<double>();
            NDDoubleSlice1 = NDDouble1["1:" + (NDDouble1.size - 2)];
            NDDoubleSliceStep2 = NDDouble1["1:" + (NDDouble1.size - 2) + ":2"];
        }

        [Benchmark(Description = "C# double[] initialized in a for loop")]
        public void Array1InitInForLoop()
        {
            //unchecked
            {
                var aLength = A1.Count;
                for (int idx = 0; idx < aLength; idx++)
                    A1[idx] = 1.0;
            }
        }

        [Benchmark(Description = "C# double[] initialized in a fixed for loop")]
        public unsafe void Array1InitInFixedForLoop()
        {
            fixed (double* pA1 = A1)
            {
                var aLength = A1.Count;
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
            fixed (double* pS2 = S2)
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
                for (int idx = 0; idx < ND1.size; idx++)
                    ND1[idx] = 1.0;
            }
        }

        [Benchmark(Description = "NDArray<double> initialized in a for loop")]
        public void NDArrayDouble1InitInForLoop()
        {
            //var A1 = N1.Array;
            //unchecked
            {
                for (int idx = 0; idx < NDDouble1.size; idx++)
                    NDDouble1[idx] = 1.0;
            }
        }

        [Benchmark(Description = "NDArray[1:-1] initialized in a for loop")]
        public void NDArraySlice1InitInForLoop()
        {
            //var A1 = N1.Array;
            //unchecked
            {
                for (int idx = 0; idx < NDSlice1.size; idx++)
                    NDSlice1[idx] = 1.0;
            }
        }


        [Benchmark(Description = "NDArray<double>[1:-1] initialized in a for loop")]
        public void NDArrayDoubleSlice1InitInForLoop()
        {
            //var A1 = N1.Array;
            //unchecked
            {
                for (int idx = 0; idx < NDDoubleSlice1.size; idx++)
                    NDDoubleSlice1[idx] = 1.0;
            }
        }

        [Benchmark(Description = "NDArray[1:-1:2] initialized in a for loop")]
        public void NDArraySlice2InitInForLoop()
        {
            //var A1 = N1.Array;
            //unchecked
            {
                for (int idx = 0; idx < NDSliceStep2.size; idx++)
                    NDSliceStep2[idx] = 1.0;
            }
        }


        [Benchmark(Description = "NDArray<double>[1:-1:2] initialized in a for loop")]
        public void NDArrayDoubleSlice2InitInForLoop()
        {
            //var A1 = N1.Array;
            //unchecked
            {
                for (int idx = 0; idx < NDDoubleSliceStep2.size; idx++)
                    NDDoubleSliceStep2[idx] = 1.0;
            }
        }

        #region Array Assign

        [Benchmark(Description = "C# double[] assigned a double[]")]
        public void NDArray1AssignArray()
        {
            //unchecked
            {
                for (int idx = 0; idx < ND1.size; idx++)
                    A1[idx] = ACopy1[idx];
            }
        }

        /*
         * Not implemented yet
        [Benchmark(Description = "NDArray assigned a double[]")]
        public void NDArray1AssignArray()
        {
            ND1.Array = ACopy1;
        }
        */
        [Benchmark(Description = "NDArray<double> assigned a double[]")]
        public void NDArrayDouble1AssignArrayClone()
        {
            //unchecked
            NDDouble1.Array = ACopy1; // Is this implemented correctly, should it be cloned by the storage ?
        }

        [Benchmark(Description = "NDArray<double> assigned a clone of double[]")]
        public void NDArrayDouble1AssignArray()
        {
            //unchecked
            NDDouble1.Array = (ArraySlice<double>)ACopy1.Clone(); // Is this implemented correctly, should it be cloned by the storage ?
        }

        /*
         * Not implemented yet
        [Benchmark(Description = "NDArray[1:-1] assigned a double[]")]
        public void NDArraySlice1AssignArray()
        {
            NDSlice1.Array = ASlice1;
        }
        */

        [Benchmark(Description = "NDArray<double>[1:-1] assigned a double[]")]
        public void NDArrayDoubleSliceAssignArrayInitInForLoop()
        {
            NDDoubleSlice1.Array = ASlice1;
        }

        /*
         * Not implemented yet
        [Benchmark(Description = "NDArray[1:-1:2] assigned a double[]")]
        public void NDArraySlice2AssignArray()
        {
            NDSliceStep2.Array = ASliceStep2;
        }
        */
        [Benchmark(Description = "NDArray<double>[1:-1:2] assigned a double[]")]
        public void NDArrayDoubleSlice2AssignArray()
        {
            NDDoubleSliceStep2.Array = ASliceStep2;
        }

        #endregion

        #region ToArray

        [Benchmark(Description = "C# double[].ToArray() and assign index 0")]
        public void Array1ToArray()
        {
            var A = A1.ToArray();
            A[0] = 1.1;
        }

        [Benchmark(Description = "NDArray.ToArray() and assign index 0")]
        public void NDArrayToArray()
        {
            var A1 = ND1.Array;
            A1.SetIndex(0, 1.1);
        }

        [Benchmark(Description = "NDArray<double>.ToArray() and assign index 0")]
        public void NDArrayDoubleToArray()
        {
            var A1 = NDDouble1.Array;
            A1.SetIndex(0, 1.1);
        }

        [Benchmark(Description = "NDArray[1:-1].ToArray() and assign index 0")]
        public void NDArraySlice1ToArray()
        {
            var A1 = NDSlice1.Array;
            A1.SetIndex(0, 1.1);
        }

        [Benchmark(Description = "NDArray<double>[1:-1].ToArray() and assign index 0")]
        public void NDArrayDoubleSlice1ToArray()
        {
            var A1 = NDDoubleSlice1.Array;
            A1.SetIndex(0, 1.1);
        }

        [Benchmark(Description = "NDArray[1:-1:2].ToArray() and assign index 0")]
        public void NDArraySlice2ToArray()
        {
            var A2 = NDSliceStep2.Array;
            A2.SetIndex(0, 1.1);
        }

        [Benchmark(Description = "NDArray<double>[1:-1:2].ToArray() and assign index 0")]
        public void NDArrayDoubleSlice2ToArray()
        {
            var A2 = NDDoubleSliceStep2.Array;
            A2.SetIndex(0, 1.1);
        }

        #endregion
    }
}
