using BenchmarkDotNet.Running;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Environments;
using BenchmarkDotNet.Columns;
using BenchmarkDotNet.Reports;
using NumSharp.Benchmark.GraphEngine;

// Run all benchmarks or specific ones based on command line args
if (args.Length == 0)
{
    // Interactive menu
    Console.WriteLine("NumSharp Performance Benchmarks");
    Console.WriteLine("================================");
    Console.WriteLine();
    Console.WriteLine("=== Original Benchmarks ===");
    Console.WriteLine("1. Dispatch Mechanism Comparison (DynamicMethod vs Static vs Struct)");
    Console.WriteLine("2. Fusion Pattern Benchmarks (fused vs multi-pass)");
    Console.WriteLine("3. NumSharp Current Performance");
    Console.WriteLine("4. DynamicMethod Emission (#544) - NumSharp vs DynMethod per-op");
    Console.WriteLine();
    Console.WriteLine("=== Comprehensive Benchmarks ===");
    Console.WriteLine("5. Arithmetic Operations (add, sub, mul, div, mod)");
    Console.WriteLine("6. Unary Operations (math, exp/log, trig, power)");
    Console.WriteLine("7. Reduction Operations (sum, mean, var/std, min/max)");
    Console.WriteLine("8. Broadcasting Operations");
    Console.WriteLine("9. Array Creation Operations");
    Console.WriteLine("10. Shape Manipulation (reshape, transpose, stack)");
    Console.WriteLine("11. Slicing Operations");
    Console.WriteLine("12. Multi-dimensional Arrays");
    Console.WriteLine();
    Console.WriteLine("=== Meta Options ===");
    Console.WriteLine("A. All Benchmarks");
    Console.WriteLine("Q. Quick smoke test (dry run)");
    Console.WriteLine();
    Console.Write("Select benchmark suite: ");

    var choice = Console.ReadLine()?.Trim().ToUpperInvariant();
    args = choice switch
    {
        "1" => ["--filter", "*Dispatch*"],
        "2" => ["--filter", "*Fusion*"],
        "3" => ["--filter", "*NumSharpBenchmarks*"],
        "4" => ["--filter", "*DynamicEmission*"],
        "5" => ["--filter", "*Arithmetic*"],
        "6" => ["--filter", "*Unary*,*Math*,*ExpLog*,*Trig*,*Power*"],
        "7" => ["--filter", "*Reduction*,*Sum*,*Mean*,*VarStd*,*MinMax*,*Prod*"],
        "8" => ["--filter", "*Broadcast*"],
        "9" => ["--filter", "*Creation*"],
        "10" => ["--filter", "*Manipulation*,*Reshape*,*Stack*,*Dims*"],
        "11" => ["--filter", "*Slice*"],
        "12" => ["--filter", "*MultiDim*"],
        "A" => [],
        "Q" => ["--job", "Dry"],
        _ => []
    };
}

BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(args);
