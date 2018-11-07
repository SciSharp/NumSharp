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
            var accessSummary = BenchmarkRunner.Run<ArrayTester>();
            Console.WriteLine("Please press any key to continue.");
            Console.ReadKey();
        }
    }
}
