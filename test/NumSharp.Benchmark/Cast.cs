using System;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;

namespace NumSharp.Benchmark
{
    //BenchmarkDotNet=v0.11.5, OS=Windows 10.0.17763.557 (1809/October2018Update/Redstone5)
    //Intel Core i7-6700K CPU 4.00GHz(Skylake), 1 CPU, 8 logical and 4 physical cores
    //.NET Core SDK = 3.0.100-preview-010184
    //
    //[Host]     : .NET Core 2.1.8 (CoreCLR 4.6.27317.03, CoreFX 4.6.27317.03), 64bit RyuJIT
    //Job-CBCCQS : .NET Core 2.1.8 (CoreCLR 4.6.27317.03, CoreFX 4.6.27317.03), 64bit RyuJIT
    //
    //IterationCount=10  LaunchCount=1  WarmupCount=3
    //
    //|     Method |         Mean |      Error |     StdDev |          Min |          Max |       Median |  Ratio | RatioSD |
    //|----------- |-------------:|-----------:|-----------:|-------------:|-------------:|-------------:|-------:|--------:|
    //|         As |    94.378 us |  3.7468 us |  2.4783 us |    91.947 us |    98.740 us |    93.336 us |  33.47 |    2.23 |
    //|   MakeRef1 | 1,626.859 us | 27.6512 us | 16.4548 us | 1,610.047 us | 1,658.922 us | 1,620.051 us | 575.01 |   29.54 |
    //|   MakeRef2 |   156.498 us |  5.5628 us |  3.3103 us |   152.471 us |   162.297 us |   156.659 us |  55.29 |    2.55 |
    //|  ClassicIs |   120.593 us |  4.3033 us |  2.5608 us |   117.739 us |   125.350 us |   119.360 us |  42.63 |    2.54 |
    //|   ModernIs |   100.975 us |  5.2199 us |  3.4526 us |    95.176 us |   105.901 us |   102.073 us |  35.59 |    1.84 |
    //| DirectCast |     7.990 us |  0.1783 us |  0.1061 us |     7.883 us |     8.235 us |     7.974 us |   2.83 |    0.17 |
    //|   Baseline |     2.837 us |  0.2768 us |  0.1647 us |     2.638 us |     3.159 us |     2.790 us |   1.00 |    0.00 |
    [SimpleJob(launchCount: 1, warmupCount: 3, targetCount: 10)]
    [MinColumn, MaxColumn, MeanColumn, MedianColumn]
    [HtmlExporter]
    public class Cast
    {
        private object input = "woof";
        private string input2 = "woof";

        [Benchmark]
        public int As()
        {
            for (int i = 0; i < 100_000; i++)
            {
                string text = input as string;
                if (text != null)
                {
                    var txt = text.Length;
                }
            }

            return 0;
        }

        [Benchmark]
        public int MakeRef1()
        {
            for (int i = 0; i < 100_000; i++)
            {
                var txt = ReinterpretCast<object, string>(input);
            }

            return 0;
        }

        [MethodImpl((MethodImplOptions)512)]
        static unsafe TDest ReinterpretCast<TSource, TDest>(TSource source)
        {
            var tr = __makeref(source);
            TDest w = default(TDest);
            var trw = __makeref(w);
            *((IntPtr*)&trw) = *((IntPtr*)&tr);
            return __refvalue(trw, TDest);
        }

        [Benchmark]
        public unsafe int MakeRef2()
        {
            for (int i = 0; i < 100_000; i++)
            {
                var x = __makeref(input);
                var txt = Unsafe.AsRef<string>((void*)*(IntPtr*)*(IntPtr*)&x);
            }

            return 0;
        }

        [Benchmark]
        public int ClassicIs()
        {
            for (int i = 0; i < 100_000; i++)
            {
                if (input is string)
                {
                    var txt = ((string)input).Length;
                }
            }

            return 0;
        }

        [Benchmark]
        public int ModernIs()
        {
            for (int i = 0; i < 100_000; i++)
            {
                if (input is string text)
                {
                    var txt = text.Length;
                }
            }

            return 0;
        }


        [Benchmark]
        public int DirectCast()
        {
            for (int i = 0; i < 100_00; i++)
            {
                var txt = ((string)input).Length;
            }

            return 0;
        }

        [Benchmark(Baseline = true)]
        public int Baseline()
        {
            for (int i = 0; i < 100_00; i++)
            {
                if (input2 != null)
                {
                    var txt = input2.Length;
                }
            }

            return 0;
        }
    }
}
