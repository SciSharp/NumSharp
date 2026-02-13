using System.Diagnostics;
using System.Runtime;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace NumSharp.Benchmark.Exploration.Infrastructure;

/// <summary>
/// Lightweight Stopwatch-based benchmark runner for fast iteration during exploration.
/// Provides consistent measurement methodology with warmup, GC control, and statistics.
/// </summary>
public static class BenchFramework
{
    /// <summary>Default warmup iterations to trigger JIT compilation.</summary>
    public const int DefaultWarmup = 5;

    /// <summary>Default measurement iterations.</summary>
    public const int DefaultMeasure = 20;

    /// <summary>Quick mode: still enough iterations for stable results.</summary>
    public const int QuickMeasure = 10;

    /// <summary>
    /// Run a benchmark with specified warmup and measurement iterations.
    /// </summary>
    /// <param name="action">The action to benchmark (should be self-contained).</param>
    /// <param name="scenario">Scenario identifier (e.g., "S1_contiguous").</param>
    /// <param name="strategy">Strategy identifier (e.g., "SIMD-FULL").</param>
    /// <param name="dtype">Data type name (e.g., "float64").</param>
    /// <param name="size">Number of elements processed.</param>
    /// <param name="elementBytes">Size of each element in bytes.</param>
    /// <param name="warmup">Number of warmup iterations (default: 3).</param>
    /// <param name="measure">Number of measurement iterations (default: 10).</param>
    /// <param name="suite">Suite name for grouping.</param>
    /// <returns>Benchmark result with statistics.</returns>
    public static BenchResult Run(
        Action action,
        string scenario,
        string strategy,
        string dtype,
        int size,
        int elementBytes,
        int warmup = DefaultWarmup,
        int measure = DefaultMeasure,
        string suite = "")
    {
        // Force full GC before test to minimize interference
        ForceGC();

        // Warmup phase - trigger JIT, fill caches
        for (int i = 0; i < warmup; i++)
        {
            action();
        }

        // Let any background GC settle
        Thread.Sleep(1);

        // Measurement phase
        var times = new double[measure];
        for (int i = 0; i < measure; i++)
        {
            var sw = Stopwatch.StartNew();
            action();
            sw.Stop();
            times[i] = sw.Elapsed.TotalMicroseconds;
        }

        // Calculate statistics
        var mean = times.Average();
        var variance = times.Select(t => (t - mean) * (t - mean)).Average();
        var stddev = Math.Sqrt(variance);
        var min = times.Min();
        var max = times.Max();
        var gbps = BenchResult.CalculateGBps(size, elementBytes, mean);

        return new BenchResult
        {
            Scenario = scenario,
            Strategy = strategy,
            Dtype = dtype,
            Size = size,
            MeanUs = mean,
            StdDevUs = stddev,
            MinUs = min,
            MaxUs = max,
            GBps = gbps,
            Reps = measure,
            Timestamp = DateTime.UtcNow.ToString("O"),
            Suite = suite
        };
    }

    /// <summary>
    /// Run a benchmark that returns a result (to prevent dead code elimination).
    /// </summary>
    public static BenchResult Run<T>(
        Func<T> func,
        string scenario,
        string strategy,
        string dtype,
        int size,
        int elementBytes,
        int warmup = DefaultWarmup,
        int measure = DefaultMeasure,
        string suite = "")
    {
        T result = default!;
        // Use block lambda to ensure it resolves as Action, not Func<T>
        // (assignment expressions return a value, which would cause infinite recursion)
        return Run(
            () => { result = func(); },
            scenario, strategy, dtype, size, elementBytes, warmup, measure, suite);
    }

    /// <summary>
    /// Run a benchmark with a setup action that runs before each measurement.
    /// Useful when the benchmark modifies its inputs (e.g., in-place operations).
    /// </summary>
    public static BenchResult RunWithSetup(
        Action setup,
        Action benchmark,
        string scenario,
        string strategy,
        string dtype,
        int size,
        int elementBytes,
        int warmup = DefaultWarmup,
        int measure = DefaultMeasure,
        string suite = "")
    {
        ForceGC();

        // Warmup
        for (int i = 0; i < warmup; i++)
        {
            setup();
            benchmark();
        }

        Thread.Sleep(1);

        // Measurement
        var times = new double[measure];
        for (int i = 0; i < measure; i++)
        {
            setup();

            var sw = Stopwatch.StartNew();
            benchmark();
            sw.Stop();

            times[i] = sw.Elapsed.TotalMicroseconds;
        }

        var mean = times.Average();
        var variance = times.Select(t => (t - mean) * (t - mean)).Average();
        var stddev = Math.Sqrt(variance);

        return new BenchResult
        {
            Scenario = scenario,
            Strategy = strategy,
            Dtype = dtype,
            Size = size,
            MeanUs = mean,
            StdDevUs = stddev,
            MinUs = times.Min(),
            MaxUs = times.Max(),
            GBps = BenchResult.CalculateGBps(size, elementBytes, mean),
            Reps = measure,
            Timestamp = DateTime.UtcNow.ToString("O"),
            Suite = suite
        };
    }

    /// <summary>
    /// Run a comparative benchmark with baseline and optimized versions.
    /// </summary>
    public static (BenchResult Baseline, BenchResult Optimized, double Speedup) RunComparison(
        Action baseline,
        Action optimized,
        string scenario,
        string dtype,
        int size,
        int elementBytes,
        int warmup = DefaultWarmup,
        int measure = DefaultMeasure,
        string suite = "")
    {
        var baseResult = Run(baseline, scenario, "Baseline", dtype, size, elementBytes, warmup, measure, suite);
        var optResult = Run(optimized, scenario, "Optimized", dtype, size, elementBytes, warmup, measure, suite);

        var speedup = baseResult.MeanUs / optResult.MeanUs;

        return (
            baseResult,
            optResult with { SpeedupVsBaseline = speedup },
            speedup
        );
    }

    /// <summary>
    /// Force a full garbage collection to reduce interference.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    public static void ForceGC()
    {
        GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true);
        GC.WaitForPendingFinalizers();
        GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true);
    }

    /// <summary>
    /// Print a formatted header for console output.
    /// </summary>
    public static void PrintHeader(string title)
    {
        Console.WriteLine();
        Console.WriteLine(new string('=', 80));
        Console.WriteLine($"  {title}");
        Console.WriteLine(new string('=', 80));
        Console.WriteLine();
        Console.WriteLine($"{"Scenario",-12} {"Strategy",-10} {"Dtype",-8} {"Size",12} | {"Mean (us)",12} ± {"StdDev",8} | {"GB/s",8}");
        Console.WriteLine(new string('-', 80));
    }

    /// <summary>
    /// Print a result line.
    /// </summary>
    public static void PrintResult(BenchResult result)
    {
        Console.WriteLine(result.ToString());
    }

    /// <summary>
    /// Print a section divider.
    /// </summary>
    public static void PrintDivider(string? label = null)
    {
        if (label != null)
        {
            Console.WriteLine($"\n--- {label} ---");
        }
        else
        {
            Console.WriteLine(new string('-', 80));
        }
    }

    /// <summary>
    /// Print environment information.
    /// </summary>
    public static void PrintEnvironment()
    {
        Console.WriteLine("Environment:");
        Console.WriteLine($"  .NET Version: {RuntimeInformation.FrameworkDescription}");
        Console.WriteLine($"  OS: {RuntimeInformation.OSDescription}");
        Console.WriteLine($"  Architecture: {RuntimeInformation.ProcessArchitecture}");
        Console.WriteLine($"  Processor Count: {Environment.ProcessorCount}");
        Console.WriteLine($"  SIMD Enabled: {System.Numerics.Vector.IsHardwareAccelerated}");
        Console.WriteLine($"  Vector256 Supported: {System.Runtime.Intrinsics.X86.Avx2.IsSupported}");
        Console.WriteLine();
    }
}

/// <summary>
/// Standard array sizes for benchmarking.
/// </summary>
public static class ArraySizes
{
    /// <summary>Fits in L1 cache (32 KB).</summary>
    public const int Tiny = 32;

    /// <summary>Fits in L1 cache (~8 KB for float64).</summary>
    public const int Small = 1_000;

    /// <summary>Fits in L2/L3 cache (~800 KB for float64).</summary>
    public const int Medium = 100_000;

    /// <summary>Exceeds L3 cache (~8 MB for float64).</summary>
    public const int Large = 1_000_000;

    /// <summary>Memory-bound (~80 MB for float64).</summary>
    public const int Huge = 10_000_000;

    /// <summary>Large memory-bound (~800 MB for float64).</summary>
    public const int Massive = 100_000_000;

    /// <summary>Standard sizes for systematic testing.</summary>
    public static readonly int[] Standard = [Small, Medium, Large, Huge];

    /// <summary>Quick sizes for fast iteration.</summary>
    public static readonly int[] Quick = [Medium, Huge];

    /// <summary>Threshold discovery sizes (powers of 2).</summary>
    public static readonly int[] Thresholds = [
        8, 16, 32, 64, 128, 256, 512,
        1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072
    ];

    /// <summary>All sizes including edge cases.</summary>
    public static readonly int[] All = [Tiny, Small, Medium, Large, Huge, Massive];
}

/// <summary>
/// Standard dtypes for benchmarking.
/// </summary>
public static class Dtypes
{
    /// <summary>All numeric types to test.</summary>
    public static readonly string[] All = ["byte", "int16", "int32", "int64", "float32", "float64"];

    /// <summary>Common types for quick tests.</summary>
    public static readonly string[] Common = ["int32", "float64"];

    /// <summary>All integer types.</summary>
    public static readonly string[] Integer = ["byte", "int16", "int32", "int64"];

    /// <summary>All floating types.</summary>
    public static readonly string[] Float = ["float32", "float64"];

    /// <summary>Types most benefiting from SIMD (more elements per vector).</summary>
    public static readonly string[] SmallElements = ["byte", "int16"];

    /// <summary>Types where SIMD benefit is smaller.</summary>
    public static readonly string[] LargeElements = ["int64", "float64"];
}
