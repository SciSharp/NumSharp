using System;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;

namespace NumSharp.Benchmark
{

    // BenchmarkDotNet=v0.11.5, OS=Windows 10.0.17763.557 (1809/October2018Update/Redstone5)
    // Intel Core i7-6700K CPU 4.00GHz(Skylake), 1 CPU, 8 logical and 4 physical cores
    //     .NET Core SDK = 3.0.100-preview-010184
    // 
    // [Host]     : .NET Core 2.1.8 (CoreCLR 4.6.27317.03, CoreFX 4.6.27317.03), 64bit RyuJIT DEBUG
    //     Job-AKGLWC : .NET Core 2.1.8 (CoreCLR 4.6.27317.03, CoreFX 4.6.27317.03), 64bit RyuJIT
    // Job-YFUEUR : .NET Core 2.1.8 (CoreCLR 4.6.27317.03, CoreFX 4.6.27317.03), 64bit RyuJIT
    //
    //|     Method |          Toolchain | IterationCount | RunStrategy | UnrollFactor |       Mean |       Error |      StdDev |     Median |        Min |        Max | Ratio | RatioSD |
    //|----------- |------------------- |--------------- |------------ |------------- |-----------:|------------:|------------:|-----------:|-----------:|-----------:|------:|--------:|
    //|         As |            Default |             10 |   ColdStart |            1 | 118.880 us | 201.7495 us | 133.4448 us |  75.250 us |  75.200 us | 498.600 us | 22.25 |   10.50 |
    //|  ClassicIs |            Default |             10 |   ColdStart |            1 | 184.900 us | 243.7026 us | 161.1942 us | 127.250 us | 125.200 us | 642.500 us | 38.78 |   18.50 |
    //|   ModernIs |            Default |             10 |   ColdStart |            1 | 122.770 us | 201.7733 us | 133.4605 us |  75.200 us |  75.200 us | 500.100 us | 22.50 |    9.70 |
    //| DirectCast |            Default |             10 |   ColdStart |            1 |  81.940 us | 325.7037 us | 215.4328 us |  12.550 us |  11.800 us | 695.000 us |  4.11 |    2.02 |
    //|   Baseline |            Default |             10 |   ColdStart |            1 |  56.730 us | 253.5501 us | 167.7077 us |   2.700 us |   2.600 us | 534.000 us |  1.00 |    0.00 |
    //|            |                    |                |             |              |            |             |             |            |            |            |       |         |
    //|         As |            Default |             10 |  Throughput |           16 |  85.273 us |   2.3262 us |   1.3843 us |  85.363 us |  83.100 us |  88.028 us | 16.50 |    0.34 |
    //|  ClassicIs |            Default |             10 |  Throughput |           16 | 122.488 us |   5.7124 us |   3.7784 us | 121.037 us | 118.710 us | 130.459 us | 23.70 |    0.92 |
    //|   ModernIs |            Default |             10 |  Throughput |           16 |  87.205 us |   3.4896 us |   2.3081 us |  87.142 us |  83.199 us |  90.525 us | 16.87 |    0.49 |
    //| DirectCast |            Default |             10 |  Throughput |           16 |   7.717 us |   0.2031 us |   0.1062 us |   7.714 us |   7.578 us |   7.863 us |  1.49 |    0.05 |
    //|   Baseline |            Default |             10 |  Throughput |           16 |   5.169 us |   0.1490 us |   0.0985 us |   5.136 us |   5.058 us |   5.310 us |  1.00 |    0.00 |
    //|            |                    |                |             |              |            |             |             |            |            |            |       |         |
    //|         As | InProcessToolchain |        Default |     Default |           16 | 319.148 us |   6.5959 us |  12.7080 us | 314.129 us | 304.627 us | 349.252 us | 12.27 |    0.73 |
    //|  ClassicIs | InProcessToolchain |        Default |     Default |           16 | 407.405 us |   5.0519 us |   4.4783 us | 406.397 us | 400.882 us | 417.268 us | 15.56 |    0.50 |
    //|   ModernIs | InProcessToolchain |        Default |     Default |           16 | 331.574 us |   7.4056 us |   6.9272 us | 327.957 us | 326.833 us | 347.776 us | 12.65 |    0.44 |
    //| DirectCast | InProcessToolchain |        Default |     Default |           16 |  25.409 us |   0.0815 us |   0.0722 us |  25.395 us |  25.325 us |  25.553 us |  0.97 |    0.03 |
    //|   Baseline | InProcessToolchain |        Default |     Default |           16 |  26.152 us |   0.7129 us |   0.8486 us |  25.824 us |  25.436 us |  29.045 us |  1.00 |    0.00 |

    //[RPlotExporter, RankColumn]
    [SimpleJob(RunStrategy.ColdStart, targetCount: 10)]
    [SimpleJob(RunStrategy.Throughput, targetCount: 10)]
    [MinColumn, MaxColumn, MeanColumn, MedianColumn]
    [HtmlExporter]
    public class Cast
    {
        private readonly object input = "woof";
        private readonly string input2 = "woof";

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
