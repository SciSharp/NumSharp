using System;
using System.Linq;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using NumSharp.Extensions;

namespace NumSharp.Benchmark
{
    class Program
    {
        static void Main(string[] args)
        {
            BenchmarkDotNet.Reports.Summary accessSummary;

            if (args.Contains("Linq"))
            {
                accessSummary = BenchmarkRunner.Run<LinqTesterInt>();
                accessSummary = BenchmarkRunner.Run<LinqTesterDouble>();
                accessSummary = BenchmarkRunner.Run<LinqTesterQuaternion>();

            }
            if (args.Contains("NDArray"))
            {
                accessSummary = BenchmarkRunner.Run<NDArrayTester1D>();
                accessSummary = BenchmarkRunner.Run<NDArrayTester2D>();
            }
            if (args.Contains("amin"))
            {
                accessSummary = BenchmarkRunner.Run<npamin>();
            }
                     
            Console.WriteLine("Please press any key to continue.");
            Console.ReadKey();
        }
    }
}
