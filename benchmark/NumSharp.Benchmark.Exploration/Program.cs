using System.Diagnostics;
using System.Runtime.InteropServices;
using BenchmarkDotNet.Running;
using NumSharp.Benchmark.Exploration.Infrastructure;
using NumSharp.Benchmark.Exploration.Isolated;
using NumSharp.Benchmark.Exploration.Integration;

namespace NumSharp.Benchmark.Exploration;

/// <summary>
/// NumSharp SIMD & Broadcasting Performance Exploration Suite.
///
/// Usage:
///   dotnet run -c Release -- --suite broadcast-scenarios
///   dotnet run -c Release -- --all
///   dotnet run -c Release -- --quick
///   dotnet run -c Release -- --bdn BroadcastBenchmarks
/// </summary>
public class Program
{
    public static int Main(string[] args)
    {
        Console.WriteLine("╔═══════════════════════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║        NumSharp SIMD & Broadcasting Performance Exploration Suite             ║");
        Console.WriteLine("╚═══════════════════════════════════════════════════════════════════════════════╝");
        Console.WriteLine();

        BenchFramework.PrintEnvironment();

        // Parse arguments
        var options = ParseArgs(args);

        if (options.ShowHelp)
        {
            PrintHelp();
            return 0;
        }

        if (options.Interactive)
        {
            return RunInteractive(options);
        }

        return RunSuites(options);
    }

    private static int RunInteractive(Options options)
    {
        Console.WriteLine("Select benchmark suite:");
        Console.WriteLine();
        Console.WriteLine("=== Isolated Benchmarks (Raw SIMD) ===");
        Console.WriteLine("1. Broadcast Scenarios (S1-S7)");
        Console.WriteLine("2. Size Thresholds (crossover discovery)");
        Console.WriteLine("3. SIMD Strategies (row broadcast comparison)");
        Console.WriteLine("4. Dispatch Overhead");
        Console.WriteLine("5. Combined Optimizations (SIMD+Pool+InPlace)");
        Console.WriteLine("6. Memory Patterns (strided, gather)");
        Console.WriteLine();
        Console.WriteLine("=== Integration Benchmarks (NumSharp) ===");
        Console.WriteLine("7. NumSharp Broadcasting");
        Console.WriteLine("8. NumSharp vs Raw SIMD Overhead");
        Console.WriteLine();
        Console.WriteLine("=== Meta Options ===");
        Console.WriteLine("A. All Isolated Benchmarks");
        Console.WriteLine("I. All Integration Benchmarks");
        Console.WriteLine("Q. Quick smoke test");
        Console.WriteLine("B. BenchmarkDotNet (full validation)");
        Console.WriteLine();
        Console.Write("Select: ");

        var choice = Console.ReadLine()?.Trim().ToUpperInvariant();

        switch (choice)
        {
            case "1": options.Suite = "broadcast-scenarios"; break;
            case "2": options.Suite = "size-thresholds"; break;
            case "3": options.Suite = "simd-strategies"; break;
            case "4": options.Suite = "dispatch"; break;
            case "5": options.Suite = "combined"; break;
            case "6": options.Suite = "memory"; break;
            case "7": options.Suite = "numsharp-broadcast"; break;
            case "8": options.Suite = "numsharp-overhead"; break;
            case "A": options.Suite = "all-isolated"; break;
            case "I": options.Suite = "all-integration"; break;
            case "Q": options.Suite = "all-isolated"; options.Quick = true; break;
            case "B": options.UseBenchmarkDotNet = true; break;
            default: options.Suite = "all-isolated"; break;
        }

        return RunSuites(options);
    }

    private static int RunSuites(Options options)
    {
        if (options.UseBenchmarkDotNet)
        {
            // Run BenchmarkDotNet
            BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(options.RemainingArgs);
            return 0;
        }

        var allResults = new List<BenchResult>();
        var timestamp = DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss");

        try
        {
            switch (options.Suite.ToLowerInvariant())
            {
                case "broadcast-scenarios":
                    allResults.AddRange(BroadcastScenarios.RunAll(
                        options.GetSizes(), options.GetDtypes(), options.Quick));
                    break;

                case "size-thresholds":
                    allResults.AddRange(SizeThresholds.RunAll(options.GetDtypes(), options.Quick));
                    break;

                case "simd-strategies":
                    allResults.AddRange(SimdStrategies.RunAll(options.Quick));
                    break;

                case "dispatch":
                    allResults.AddRange(DispatchOverhead.RunAll(options.Quick));
                    break;

                case "combined":
                    allResults.AddRange(CombinedOptimizations.RunAll(options.Quick));
                    break;

                case "memory":
                    allResults.AddRange(MemoryPatterns.RunAll(options.Quick));
                    break;

                case "numsharp-broadcast":
                    allResults.AddRange(NumSharpBroadcast.RunAll(options.Quick));
                    break;

                case "numsharp-overhead":
                    foreach (var size in options.GetSizes())
                    {
                        allResults.AddRange(NumSharpBroadcast.CompareOverhead(size, 3, 10));
                    }
                    break;

                case "all-isolated":
                    Console.WriteLine("Running ALL isolated benchmarks...\n");
                    allResults.AddRange(BroadcastScenarios.RunAll(options.GetSizes(), options.GetDtypes(), options.Quick));
                    allResults.AddRange(SizeThresholds.RunAll(options.GetDtypes(), options.Quick));
                    allResults.AddRange(SimdStrategies.RunAll(options.Quick));
                    allResults.AddRange(DispatchOverhead.RunAll(options.Quick));
                    allResults.AddRange(CombinedOptimizations.RunAll(options.Quick));
                    allResults.AddRange(MemoryPatterns.RunAll(options.Quick));
                    break;

                case "all-integration":
                    Console.WriteLine("Running ALL integration benchmarks...\n");
                    allResults.AddRange(NumSharpBroadcast.RunAll(options.Quick));
                    foreach (var size in options.GetSizes())
                    {
                        allResults.AddRange(NumSharpBroadcast.CompareOverhead(size, 3, 10));
                    }
                    break;

                case "all":
                    Console.WriteLine("Running ALL benchmarks...\n");
                    allResults.AddRange(BroadcastScenarios.RunAll(options.GetSizes(), options.GetDtypes(), options.Quick));
                    allResults.AddRange(SizeThresholds.RunAll(options.GetDtypes(), options.Quick));
                    allResults.AddRange(SimdStrategies.RunAll(options.Quick));
                    allResults.AddRange(DispatchOverhead.RunAll(options.Quick));
                    allResults.AddRange(CombinedOptimizations.RunAll(options.Quick));
                    allResults.AddRange(MemoryPatterns.RunAll(options.Quick));
                    allResults.AddRange(NumSharpBroadcast.RunAll(options.Quick));
                    break;

                default:
                    Console.WriteLine($"Unknown suite: {options.Suite}");
                    PrintHelp();
                    return 1;
            }

            // Export results
            if (allResults.Any())
            {
                var resultsDir = Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "Results");
                Directory.CreateDirectory(resultsDir);

                var csvPath = Path.Combine(resultsDir, $"{timestamp}_{options.Suite}.csv");
                var jsonPath = Path.Combine(resultsDir, $"{timestamp}_{options.Suite}.json");
                var mdPath = Path.Combine(resultsDir, $"{timestamp}_{options.Suite}.md");

                OutputFormatters.ExportCsv(allResults, csvPath);
                OutputFormatters.ExportJson(allResults, jsonPath);

                var resultSet = new BenchResultSet
                {
                    SuiteName = options.Suite,
                    Description = $"SIMD & Broadcasting exploration - {options.Suite}",
                    StartTime = DateTime.UtcNow.AddSeconds(-allResults.Count * 0.5), // Approximate
                    CpuModel = GetCpuModel(),
                    DotNetVersion = RuntimeInformation.FrameworkDescription
                };
                foreach (var r in allResults) resultSet.Add(r);
                resultSet.EndTime = DateTime.UtcNow;

                OutputFormatters.ExportMarkdown(resultSet, mdPath);

                // If thresholds were run, generate threshold constants
                if (options.Suite.Contains("threshold") || options.Suite == "all" || options.Suite == "all-isolated")
                {
                    var thresholdResults = allResults.Where(r => r.Suite == "SizeThresholds").ToList();
                    if (thresholdResults.Any())
                    {
                        var constants = SizeThresholds.GenerateThresholdConstants(thresholdResults);
                        var constantsPath = Path.Combine(resultsDir, $"{timestamp}_thresholds.cs");
                        File.WriteAllText(constantsPath, constants);
                        Console.WriteLine($"\nThreshold constants exported to: {constantsPath}");
                    }
                }
            }

            Console.WriteLine("\n✓ Benchmark suite completed successfully.");
            return 0;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"\n✗ Error: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
            return 1;
        }
    }

    private static Options ParseArgs(string[] args)
    {
        var options = new Options();
        var remaining = new List<string>();

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i].ToLowerInvariant())
            {
                case "--help" or "-h":
                    options.ShowHelp = true;
                    break;

                case "--suite" or "-s":
                    if (i + 1 < args.Length)
                        options.Suite = args[++i];
                    break;

                case "--quick" or "-q":
                    options.Quick = true;
                    break;

                case "--all" or "-a":
                    options.Suite = "all";
                    break;

                case "--bdn":
                    options.UseBenchmarkDotNet = true;
                    remaining.AddRange(args.Skip(i + 1));
                    i = args.Length; // Stop parsing
                    break;

                case "--dtypes":
                    if (i + 1 < args.Length)
                        options.Dtypes = args[++i];
                    break;

                case "--sizes":
                    if (i + 1 < args.Length)
                        options.Sizes = args[++i];
                    break;

                case "--output" or "-o":
                    if (i + 1 < args.Length)
                        options.OutputPath = args[++i];
                    break;

                default:
                    if (!args[i].StartsWith("-"))
                        options.Suite = args[i];
                    else
                        remaining.Add(args[i]);
                    break;
            }
        }

        options.RemainingArgs = remaining.ToArray();

        // If no suite specified and no args, go interactive
        if (string.IsNullOrEmpty(options.Suite) && args.Length == 0)
        {
            options.Interactive = true;
        }

        return options;
    }

    private static void PrintHelp()
    {
        Console.WriteLine(@"
Usage: dotnet run -c Release -- [options] [suite]

Options:
  --help, -h         Show this help
  --suite, -s NAME   Run specific suite
  --quick, -q        Quick mode (fewer iterations, smaller sizes)
  --all, -a          Run all suites
  --bdn              Use BenchmarkDotNet (followed by BDN args)
  --dtypes TYPE      Dtype filter: common (default), all
  --sizes SIZE       Size filter: quick, standard (default), all
  --output, -o PATH  Output file path

Suites:
  broadcast-scenarios  All 7 broadcasting scenarios (S1-S7)
  size-thresholds      SIMD crossover point discovery
  simd-strategies      Row broadcast strategy comparison
  dispatch             Dispatch mechanism overhead
  combined             SIMD + Pool + InPlace combinations
  memory               Memory access patterns
  numsharp-broadcast   NumSharp integration tests
  numsharp-overhead    NumSharp vs raw SIMD overhead
  all-isolated         All isolated (raw SIMD) benchmarks
  all-integration      All NumSharp integration benchmarks
  all                  Everything

Examples:
  dotnet run -c Release                          # Interactive menu
  dotnet run -c Release -- broadcast-scenarios   # Specific suite
  dotnet run -c Release -- --quick --all         # Quick full run
  dotnet run -c Release -- --bdn *Broadcast*     # BenchmarkDotNet filter
");
    }

    private static string GetCpuModel()
    {
        try
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                var psi = new ProcessStartInfo
                {
                    FileName = "wmic",
                    Arguments = "cpu get name",
                    RedirectStandardOutput = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };
                using var proc = Process.Start(psi);
                var output = proc?.StandardOutput.ReadToEnd() ?? "";
                var lines = output.Split('\n', StringSplitOptions.RemoveEmptyEntries);
                if (lines.Length >= 2)
                    return lines[1].Trim();
            }
        }
        catch { }

        return "Unknown CPU";
    }

    private class Options
    {
        public string Suite { get; set; } = "";
        public bool Quick { get; set; } = false;
        public bool ShowHelp { get; set; } = false;
        public bool Interactive { get; set; } = false;
        public bool UseBenchmarkDotNet { get; set; } = false;
        public string Dtypes { get; set; } = "common";
        public string Sizes { get; set; } = "standard";
        public string OutputPath { get; set; } = "";
        public string[] RemainingArgs { get; set; } = [];

        public string[] GetDtypes() => Dtypes.ToLowerInvariant() switch
        {
            "all" => Exploration.Infrastructure.Dtypes.All,
            _ => Exploration.Infrastructure.Dtypes.Common
        };

        public int[] GetSizes() => (Sizes.ToLowerInvariant(), Quick) switch
        {
            ("quick", _) or (_, true) => ArraySizes.Quick,
            ("all", _) => ArraySizes.All,
            _ => ArraySizes.Standard
        };

        public Options Clone() => new Options
        {
            Suite = Suite,
            Quick = Quick,
            ShowHelp = ShowHelp,
            Interactive = Interactive,
            UseBenchmarkDotNet = UseBenchmarkDotNet,
            Dtypes = Dtypes,
            Sizes = Sizes,
            OutputPath = OutputPath,
            RemainingArgs = RemainingArgs
        };
    }
}
