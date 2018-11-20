using System;
using System.Linq;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using NumSharp.Extensions;

namespace NumSharp.Benchmark
{
    class Program
    {
        /// <summary>
        /// dotnet NumSharp.Benchmark.dll (Benchmark Class Name)
        /// dotnet NumSharp.Benchmark.dll nparange
        /// </summary>
        /// <param name="args"></param>
        static void Main(string[] args)
        {
            string method = $"NumSharp.Benchmark.{args[0]}";
            Console.WriteLine(method);
            Type type = Type.GetType(method);
            var accessSummary = BenchmarkRunner.Run(type);
                     
            Console.WriteLine("Please press any key to continue.");
            Console.ReadKey();
        }
    }
}
