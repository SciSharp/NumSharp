using System;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;

namespace NumSharp.Benchmark
{
    //[RPlotExporter, RankColumn]
    [SimpleJob(RunStrategy.ColdStart, targetCount: 10)]
    [SimpleJob(RunStrategy.Throughput, targetCount: 10)]
    [MinColumn, MaxColumn, MeanColumn, MedianColumn]
    [HtmlExporter]
    public class CastArray
    {

        // BenchmarkDotNet=v0.11.5, OS=Windows 10.0.17763.557 (1809/October2018Update/Redstone5)
        // Intel Core i7-6700K CPU 4.00GHz(Skylake), 1 CPU, 8 logical and 4 physical cores
        //     .NET Core SDK = 3.0.100-preview-010184
        // 
        // [Host]     : .NET Core 2.1.8 (CoreCLR 4.6.27317.03, CoreFX 4.6.27317.03), 64bit RyuJIT DEBUG
        //     Job-AKGLWC : .NET Core 2.1.8 (CoreCLR 4.6.27317.03, CoreFX 4.6.27317.03), 64bit RyuJIT
        // Job-YFUEUR : .NET Core 2.1.8 (CoreCLR 4.6.27317.03, CoreFX 4.6.27317.03), 64bit RyuJIT
        //
        //|     Method |          Toolchain | IterationCount | RunStrategy | UnrollFactor |       Mean |       Error |      StdDev |     Median |        Min |          Max | Ratio | RatioSD |
        //|----------- |------------------- |--------------- |------------ |------------- |-----------:|------------:|------------:|-----------:|-----------:|-------------:|------:|--------:|
        //|         As |            Default |             10 |   ColdStart |            1 | 170.720 us | 356.2498 us | 235.6372 us |  96.750 us |  92.000 us |   841.300 us | 16.59 |    5.49 |
        //|  ClassicIs |            Default |             10 |   ColdStart |            1 | 380.740 us | 455.6083 us | 301.3567 us | 278.250 us | 275.200 us | 1,237.100 us | 48.94 |   17.40 |
        //|   ModernIs |            Default |             10 |   ColdStart |            1 | 106.970 us | 223.7155 us | 147.9740 us |  56.600 us |  56.400 us |   527.700 us | 10.39 |    3.66 |
        //| DirectCast |            Default |             10 |   ColdStart |            1 |  47.780 us | 200.3409 us | 132.5131 us |   5.350 us |   5.200 us |   424.900 us |  1.10 |    0.27 |
        //|   Baseline |            Default |             10 |   ColdStart |            1 |  46.350 us | 195.9936 us | 129.6376 us |   5.100 us |   5.100 us |   415.300 us |  1.00 |    0.00 |
        //|            |                    |                |             |              |            |             |             |            |            |              |       |         |
        //|         As |            Default |             10 |  Throughput |           16 |  52.104 us |   0.8234 us |   0.5446 us |  52.151 us |  51.285 us |    53.051 us | 10.26 |    0.13 |
        //|  ClassicIs |            Default |             10 |  Throughput |           16 | 279.535 us |   4.6577 us |   2.7717 us | 277.743 us | 277.167 us |   284.671 us | 54.98 |    0.61 |
        //|   ModernIs |            Default |             10 |  Throughput |           16 |  51.522 us |   0.8216 us |   0.5434 us |  51.581 us |  50.742 us |    52.268 us | 10.12 |    0.14 |
        //| DirectCast |            Default |             10 |  Throughput |           16 |   5.078 us |   0.0735 us |   0.0486 us |   5.059 us |   5.029 us |     5.153 us |  1.00 |    0.01 |
        //|   Baseline |            Default |             10 |  Throughput |           16 |   5.084 us |   0.0676 us |   0.0402 us |   5.092 us |   5.036 us |     5.150 us |  1.00 |    0.00 |
        //|            |                    |                |             |              |            |             |             |            |            |              |       |         |
        //|         As | InProcessToolchain |        Default |     Default |           16 | 520.785 us |  10.4007 us |  15.5673 us | 518.490 us | 503.174 us |   556.868 us | 18.49 |    0.58 |
        //|  ClassicIs | InProcessToolchain |        Default |     Default |           16 | 783.473 us |   3.6723 us |   3.4351 us | 783.799 us | 778.711 us |   791.667 us | 27.63 |    0.74 |
        //|   ModernIs | InProcessToolchain |        Default |     Default |           16 | 577.359 us |   5.5165 us |   4.6065 us | 575.059 us | 572.856 us |   585.135 us | 20.32 |    0.55 |
        //| DirectCast | InProcessToolchain |        Default |     Default |           16 |  47.752 us |   1.7846 us |   5.2620 us |  45.664 us |  42.760 us |    64.004 us |  1.64 |    0.20 |
        //|   Baseline | InProcessToolchain |        Default |     Default |           16 |  28.231 us |   0.5631 us |   0.6703 us |  27.972 us |  27.533 us |    29.989 us |  1.00 |    0.00 |

        private string[] inputTyped;
        private Array input;
        private string[] input2;

        public CastArray()
        {
            inputTyped = new String[] {"woof"};
            input2 = new string[] {"woof"};
            input = inputTyped;
        }

        [Benchmark]
        public int As()
        {
            for (int i = 0; i < 100_000; i++)
            {
                var text = (input as string[])[0].Length;
            }

            return 0;
        }

        [Benchmark]
        public int ClassicIs()
        {
            for (int i = 0; i < 100_000; i++)
            {
                if (input is string[])
                {
                    var txt = ((string[])input)[0].Length;
                }
            }

            return 0;
        }

        [Benchmark]
        public int ModernIs()
        {
            for (int i = 0; i < 100_000; i++)
            {
                if (input is string[] text)
                {
                    var txt = text[0].Length;
                }
            }

            return 0;
        }


        [Benchmark]
        public int DirectCast()
        {
            for (int i = 0; i < 100_00; i++)
            {
                var txt = ((string[])input)[0].Length;
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
                    var txt = input2[0].Length;
                }
            }

            return 0;
        }
    }
}
