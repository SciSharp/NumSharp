using System;
using System.Reflection;
using BenchmarkDotNet.Configs;
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
#if DEBUG
                IConfig config = new DebugInProcessConfig();
#else
                IConfig config = null;
#endif
                BenchmarkSwitcher.FromAssembly(Assembly.GetExecutingAssembly()).Run(args, config);
                // BenchmarkSwitcher.FromAssembly(Assembly.GetExecutingAssembly()).Run(args, ManualConfig.Create(DefaultConfig.Instance).With(ConfigOptions.DisableOptimizationsValidator));
            }

            if (args?.Length > 0)
            {
                for (int i = 0; i < args.Length; i++)
                {
                    string method = $"OMath.Benchmarks.{args[i]}";
                    var type = Type.GetType(method);
                    BenchmarkRunner.Run(type);
                }
            }
            else
            {
            }

            Console.ReadLine();
        }
    }
}
