using System;
using System.Reflection;
using BenchmarkDotNet.Running;

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
            if (args?.Length > 0)
            {
                for (int i = 0; i < args.Length; i++)
                {
                    string method = $"NumSharp.Benchmark.{args[i]}";
                    var type = Type.GetType(method);
                    BenchmarkRunner.Run(type);
                }
            }
            else
            {
                BenchmarkSwitcher.FromAssembly(Assembly.GetExecutingAssembly()).Run();
            }

            Console.ReadLine();
        }
    }
}
