using System;
using System.Linq;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;

namespace NumSharp.Benchmark
{
    class Program
    {
        static void Main(string[] args)
        {
            var accessSummary = BenchmarkRunner.Run<npamin>();
            accessSummary = BenchmarkRunner.Run<LinqTesterInt>();
            accessSummary = BenchmarkRunner.Run<LinqTesterDouble>();
            accessSummary = BenchmarkRunner.Run<LinqTesterQuaternion>();
            Console.WriteLine("Please press any key to continue.");
            Console.ReadKey();
        }
    }
}
