using BenchmarkDotNet.Running;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Environments;
using BenchmarkDotNet.Columns;
using BenchmarkDotNet.Reports;
using NumSharp.Benchmark.Core;

// Run all benchmarks or specific ones based on command line args
if (args.Length == 0)
{
    // Interactive menu
    Console.WriteLine("NumSharp Performance Benchmarks");
    Console.WriteLine("================================");
    Console.WriteLine();
    Console.WriteLine("=== NumPy Comparison Benchmarks ===");
    Console.WriteLine("1. Arithmetic Operations (add, sub, mul, div, mod)");
    Console.WriteLine("2. Unary Operations (math, exp/log, trig, power)");
    Console.WriteLine("3. Reduction Operations (sum, mean, var/std, min/max)");
    Console.WriteLine("4. Broadcasting Operations");
    Console.WriteLine("5. Array Creation Operations");
    Console.WriteLine("6. Shape Manipulation (reshape, transpose, stack)");
    Console.WriteLine("7. Slicing Operations");
    Console.WriteLine("8. Multi-dimensional Arrays");
    Console.WriteLine();
    Console.WriteLine("=== Experimental Benchmarks (C# internals, not for NumPy comparison) ===");
    Console.WriteLine("D. Dispatch Mechanism Comparison (DynamicMethod vs Static vs Struct)");
    Console.WriteLine("F. Fusion Pattern Benchmarks (fused vs multi-pass)");
    Console.WriteLine("N. NumSharp Current Performance");
    Console.WriteLine("E. DynamicMethod Emission (#544) - NumSharp vs DynMethod per-op");
    Console.WriteLine();
    Console.WriteLine("=== Meta Options ===");
    Console.WriteLine("A. All Benchmarks (NumPy comparison only, excludes Experimental)");
    Console.WriteLine("X. All Experimental Benchmarks");
    Console.WriteLine("*. Everything (all benchmarks)");
    Console.WriteLine("Q. Quick smoke test (dry run)");
    Console.WriteLine();
    Console.Write("Select benchmark suite: ");

    var choice = Console.ReadLine()?.Trim().ToUpperInvariant();
    args = choice switch
    {
        // NumPy comparison benchmarks
        "1" => ["--filter", "*Arithmetic*"],
        "2" => ["--filter", "*Unary*,*Math*,*ExpLog*,*Trig*,*Power*"],
        "3" => ["--filter", "*Reduction*,*Sum*,*Mean*,*VarStd*,*MinMax*,*Prod*"],
        "4" => ["--filter", "*Broadcast*"],
        "5" => ["--filter", "*Creation*"],
        "6" => ["--filter", "*Manipulation*,*Reshape*,*Stack*,*Dims*"],
        "7" => ["--filter", "*Slice*"],
        "8" => ["--filter", "*MultiDim*"],

        // Experimental benchmarks
        "D" => ["--filter", "*Experimental*Dispatch*"],
        "F" => ["--filter", "*Experimental*Fusion*"],
        "N" => ["--filter", "*Experimental*NumSharpBenchmarks*"],
        "E" => ["--filter", "*Experimental*DynamicEmission*"],

        // Meta options
        "A" => ["--filter", "*Benchmarks*", "--filter", "!*Experimental*"],  // All except Experimental
        "X" => ["--filter", "*Experimental*"],  // All Experimental
        "*" => [],  // Everything
        "Q" => ["--job", "Dry"],
        _ => []
    };
}

BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(args);
