using System.Collections.Immutable;
using System.Globalization;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.ConsoleArguments;
using BenchmarkDotNet.Environments;
using BenchmarkDotNet.EventProcessors;
using BenchmarkDotNet.Exporters.Json;
using BenchmarkDotNet.Filters;
using BenchmarkDotNet.Loggers;
using BenchmarkDotNet.Reports;
using BenchmarkDotNet.Running;
using BenchmarkDotNet.Validators;

namespace NumSharp.Benchmark.CSharp.Infrastructure;

/// <summary>
/// Stable case identities, no-workload discovery, and durable per-case BDN reports.
/// Job IDs deliberately do not form part of identity: the orchestrator pins the run's
/// depth/runtime/backend separately and an official run has exactly one job per case.
/// </summary>
public static class BenchmarkCheckpoint
{
    private static readonly IReadOnlyDictionary<string, string> Suites =
        new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["Arithmetic"] = "arithmetic", ["Unary"] = "unary", ["Reduction"] = "reduction",
            ["Broadcasting"] = "broadcast", ["Creation"] = "creation",
            ["Manipulation"] = "manipulation", ["Slicing"] = "slicing",
            ["Comparison"] = "comparison", ["Bitwise"] = "bitwise", ["Logic"] = "logic",
            ["Statistics"] = "statistics", ["Sorting"] = "sorting", ["LinearAlgebra"] = "linalg",
            ["Selection"] = "selection", ["Fourier"] = "fft", ["Random"] = "random",
            ["NDArrayApi"] = "ndarray", ["ApiSurface"] = "api",
        };

    internal static IReadOnlySet<string>? RequestedCases { get; } = ReadRequestedCases();

    public static string CaseId(BenchmarkCase benchmarkCase) =>
        $"{benchmarkCase.Descriptor.Type.FullName}|{benchmarkCase.Descriptor.WorkloadMethod.Name}|" +
        string.Join("&", benchmarkCase.Parameters.Items.OrderBy(p => p.Name, StringComparer.Ordinal)
            .Select(p => $"{p.Name}={Convert.ToString(p.Value, CultureInfo.InvariantCulture)}"));

    /// <summary>
    /// Handles --benchmark-plan-json PATH, using BDN's own parameter expansion and CLI
    /// filters. No setup or benchmark method executes. Returns false for ordinary runs.
    /// </summary>
    public static bool TryWritePlan(Assembly assembly, string[] args)
    {
        int index = Array.FindIndex(args, arg => arg.Equals("--benchmark-plan-json", StringComparison.OrdinalIgnoreCase));
        if (index < 0)
            return false;
        if (index + 1 >= args.Length || args[index + 1].StartsWith("--", StringComparison.Ordinal))
            throw new ArgumentException("--benchmark-plan-json requires an output path.");

        string path = args[index + 1];
        var bdnArgs = args.Where((_, i) => i != index && i != index + 1).ToArray();
        var official = new OfficialBenchmarkConfig();
        var parsed = ConfigParser.Parse(bdnArgs, ConsoleLogger.Default, official);
        if (!parsed.Item1)
            throw new ArgumentException("Invalid BenchmarkDotNet arguments for benchmark discovery.");
        var config = ManualConfig.Union(official, parsed.Item2);
        var cases = assembly.GetTypes()
            .Where(type => !type.IsAbstract && Suite(type) is not null
                && type.GetMethods().Any(method => method.GetCustomAttribute<BenchmarkAttribute>() is not null))
            .SelectMany(type => BenchmarkConverter.TypeToBenchmarks(type, config).BenchmarksCases)
            .Where(item => item.Config.GetFilters().All(filter => filter.Predicate(item)))
            .OrderBy(CaseId, StringComparer.Ordinal)
            .ToArray();
        if (cases.Select(CaseId).Distinct(StringComparer.Ordinal).Count() != cases.Length)
            throw new InvalidOperationException("Checkpointed benchmarks require exactly one job per case.");

        var rows = cases.Select(item =>
        {
            var parameters = item.Parameters.Items.ToDictionary(p => p.Name,
                p => Convert.ToString(p.Value, CultureInfo.InvariantCulture));
            var operation = item.Descriptor.WorkloadMethod.GetCustomAttribute<BenchmarkAttribute>()?.Description
                ?? item.Descriptor.WorkloadMethod.Name;
            return new
            {
                id = CaseId(item), suite = Suite(item.Descriptor.Type), operation, name = operation,
                dtype = BenchmarkRunSelection.CanonicalDtype(item.Parameters.Items
                    .FirstOrDefault(p => p.Name.Equals("DType", StringComparison.OrdinalIgnoreCase))?.Value) ?? "float64",
                n = item.Parameters.Items.FirstOrDefault(p => p.Name == "N") is { } size
                    ? Convert.ToInt64(size.Value, CultureInfo.InvariantCulture) : 10_000_000L,
                type = item.Descriptor.Type.FullName, method = item.Descriptor.WorkloadMethod.Name,
                filter = item.Descriptor.GetFilterName(), parameters,
            };
        });
        WriteAtomic(path, JsonSerializer.Serialize(rows, new JsonSerializerOptions { WriteIndented = true }));
        Console.WriteLine($"Benchmark plan: {cases.Length} cases -> {Path.GetFullPath(path)}");
        return true;
    }

    private static string? Suite(Type type)
    {
        const string prefix = "NumSharp.Benchmark.CSharp.Benchmarks.";
        string? ns = type.Namespace;
        return ns is not null && ns.StartsWith(prefix, StringComparison.Ordinal)
            && Suites.TryGetValue(ns[prefix.Length..], out string? suite) ? suite : null;
    }

    private static IReadOnlySet<string>? ReadRequestedCases()
    {
        string? path = Environment.GetEnvironmentVariable("NUMSHARP_BENCHMARK_CASES_FILE");
        if (string.IsNullOrWhiteSpace(path))
            return null;
        var ids = JsonSerializer.Deserialize<string[]>(File.ReadAllText(path))
            ?? throw new InvalidOperationException("NUMSHARP_BENCHMARK_CASES_FILE must contain a JSON string array.");
        if (ids.Any(string.IsNullOrWhiteSpace))
            throw new InvalidOperationException("Benchmark case IDs must be nonempty strings.");
        return new HashSet<string>(ids, StringComparer.Ordinal);
    }

    internal static void WriteAtomic(string path, string content)
    {
        path = Path.GetFullPath(path);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        string temporary = path + "." + Guid.NewGuid().ToString("N") + ".tmp";
        try
        {
            using (var stream = new FileStream(temporary, FileMode.CreateNew, FileAccess.Write, FileShare.None))
            {
                byte[] bytes = Encoding.UTF8.GetBytes(content);
                stream.Write(bytes);
                stream.Flush(flushToDisk: true);
            }
            File.Move(temporary, path, overwrite: true);
        }
        finally
        {
            if (File.Exists(temporary))
                File.Delete(temporary);
        }
    }
}

internal sealed class RequestedCaseFilter : IFilter
{
    public bool Predicate(BenchmarkCase benchmarkCase) =>
        BenchmarkCheckpoint.RequestedCases is not { } requested || requested.Contains(BenchmarkCheckpoint.CaseId(benchmarkCase));
}

/// <summary>
/// BDN normally exports only when a complete benchmark class finishes. Exporting its
/// own full format here preserves statistics AND every actual/warmup measurement if
/// a later case crashes or the run is interrupted. Completion is announced only after
/// the replacement checkpoint has been flushed and atomically published.
/// </summary>
internal sealed class BenchmarkCheckpointProcessor : EventProcessor
{
    private readonly string directory;
    private readonly HashSet<string> completed = new(StringComparer.Ordinal);
    private HostEnvironmentInfo? host;

    public BenchmarkCheckpointProcessor(string directory) => this.directory = Path.GetFullPath(directory);

    public override void OnStartRunBenchmarksInType(Type type, IReadOnlyList<BenchmarkCase> benchmarks)
    {
        if (benchmarks.Select(BenchmarkCheckpoint.CaseId).Distinct(StringComparer.Ordinal).Count() != benchmarks.Count)
            throw new InvalidOperationException("Checkpointed benchmarks require exactly one job per case.");
    }

    public override void OnEndRunBenchmarksInType(Type type, Summary summary)
    {
        // A build failure produces a report without entering OnStart/OnEndRunBenchmark.
        // Preserve those failures too, instead of making them look like cases never attempted.
        foreach (var report in summary.Reports)
            if (!completed.Contains(BenchmarkCheckpoint.CaseId(report.BenchmarkCase)))
                OnEndRunBenchmark(report.BenchmarkCase, report);
    }

    public override void OnStartRunBenchmark(BenchmarkCase benchmarkCase) =>
        Emit(new { @event = "start", id = BenchmarkCheckpoint.CaseId(benchmarkCase), status = "running" });

    public override void OnEndRunBenchmark(BenchmarkCase benchmarkCase, BenchmarkReport report)
    {
        string id = BenchmarkCheckpoint.CaseId(benchmarkCase);
        string hash = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(id))).ToLowerInvariant();
        string path = Path.Combine(directory, hash + "-report-full-compressed.json");
        var summary = new Summary(hash, ImmutableArray.Create(report), host ??= HostEnvironmentInfo.GetCurrent(),
            directory, string.Empty, TimeSpan.Zero, CultureInfo.InvariantCulture,
            ImmutableArray<ValidationError>.Empty, ImmutableArray<BenchmarkDotNet.Columns.IColumnHidingRule>.Empty,
            SummaryStyle.Default);
        var logger = new AccumulationLogger();
        JsonExporter.FullCompressed.ExportToLog(summary, logger);
        var payload = JsonNode.Parse(logger.GetLog())!;
        string status = report.Success && report.ResultStatistics is not null ? "ok" : "failed";
        payload["Benchmarks"]![0]!["CaseId"] = id;
        payload["Benchmarks"]![0]!["CheckpointStatus"] = status;
        BenchmarkCheckpoint.WriteAtomic(path, payload.ToJsonString());
        completed.Add(id);
        Emit(new
        {
            @event = "complete", id,
            status,
            checkpoint = path,
        });
    }

    private static void Emit(object message)
    {
        Console.WriteLine("@@BENCHMARK " + JsonSerializer.Serialize(message));
        Console.Out.Flush();
    }
}
