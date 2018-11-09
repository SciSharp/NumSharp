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
            var accessSummary = BenchmarkRunner.Run<npamin>();
<<<<<<< HEAD
            accessSummary = BenchmarkRunner.Run<NDArrayTester1D>();
            accessSummary = BenchmarkRunner.Run<NDArrayTester2D>();
            accessSummary = BenchmarkRunner.Run<LinqTesterInt>();
            accessSummary = BenchmarkRunner.Run<LinqTesterDouble>();
            accessSummary = BenchmarkRunner.Run<LinqTesterQuaternion>();
=======
>>>>>>> upstream/master
            Console.WriteLine("Please press any key to continue.");
            Console.ReadKey();
        }
    }
}
