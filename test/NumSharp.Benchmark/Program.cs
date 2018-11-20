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
            for(int i = 0; i < args.Length; i++)
            {
                string method = $"NumSharp.Benchmark.{args[i]}";
                Type type = Type.GetType(method);
                BenchmarkRunner.Run(type);
            }

            Console.WriteLine("Please press any key to continue.");
            Console.ReadKey();
        }
    }
}
