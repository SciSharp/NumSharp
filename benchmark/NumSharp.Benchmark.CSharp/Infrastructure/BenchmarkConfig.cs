using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Exporters;
using BenchmarkDotNet.Exporters.Json;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Diagnosers;
using BenchmarkDotNet.Columns;
using BenchmarkDotNet.Reports;
using BenchmarkDotNet.Filters;
using BenchmarkDotNet.Validators;
using BenchmarkDotNet.Toolchains.InProcess.Emit;
using BenchmarkDotNet.Running;
using Perfolizer.Horology;
using NumSharp;

namespace NumSharp.Benchmark.CSharp.Infrastructure;

/// <summary>
/// Custom BenchmarkDotNet configuration for NumSharp benchmarks.
/// Provides consistent settings across all benchmark classes.
/// </summary>
public class NumSharpBenchmarkConfig : ManualConfig
{
    public NumSharpBenchmarkConfig()
    {
        // Job configuration
        AddJob(Job.Default
            .WithWarmupCount(3)
            .WithIterationCount(20)
            .AsDefault());

        // Diagnosers
        AddDiagnoser(MemoryDiagnoser.Default);

        // Exporters - JSON for automated comparison
        AddExporter(JsonExporter.FullCompressed);
        AddExporter(MarkdownExporter.GitHub);

        // Columns
        AddColumn(StatisticColumn.Mean);
        AddColumn(StatisticColumn.StdDev);
        AddColumn(StatisticColumn.Median);
        AddColumn(StatisticColumn.Min);
        AddColumn(StatisticColumn.Max);
        AddColumn(BaselineRatioColumn.RatioMean);
        AddColumn(RankColumn.Arabic);

        // Summary style
        WithSummaryStyle(SummaryStyle.Default.WithMaxParameterColumnWidth(40));
    }
}

/// <summary>
/// Quick benchmark configuration for fast iterations during development.
/// </summary>
public class QuickBenchmarkConfig : ManualConfig
{
    public QuickBenchmarkConfig()
    {
        AddJob(Job.Dry
            .WithWarmupCount(1)
            .WithIterationCount(3));

        AddDiagnoser(MemoryDiagnoser.Default);
        AddExporter(JsonExporter.Brief);
    }
}

/// <summary>
/// Official benchmark configuration for the NumSharp-vs-NumPy comparison report.
///
/// Uses the <see cref="InProcessEmitToolchain"/> rather than the default out-of-process
/// CsProj toolchain. The out-of-process toolchain searches the whole repository tree for
/// the benchmark project by name and fails here ("Benchmark project names need to be
/// unique") because sibling git worktrees under <c>.claude/worktrees/</c> contain
/// same-named copies of this project. In-process emission sidesteps that search entirely;
/// it also (a) removes per-benchmark build/launch overhead — material for the full run —
/// and (b) measures C# in a warm long-lived process, matching how the Python/NumPy side is
/// measured (a single warmed interpreter), which makes the cross-language ratio fairer.
///
/// Rigor: warmup 5 / 50 measured iterations (the "Full" tier), with BenchmarkDotNet's
/// standard statistical engine (outlier removal, margin-of-error). JSON export feeds
/// scripts/merge-results.py.
/// </summary>
public class OfficialBenchmarkConfig : ManualConfig
{
    public OfficialBenchmarkConfig()
    {
        var depth = BenchmarkRunSelection.Depth;
        // IterationTime is capped at 25 ms. BenchmarkDotNet's default Throughput strategy
        // ramps the per-iteration invocation count until each iteration takes ~0.5 s — which
        // is right for nanosecond microbenchmarks but catastrophic for array ops in the
        // µs–ms range: a 10M-element op (~60 µs) gets 8192 invocations/iteration, so 50
        // iterations = ~25 s PER case, and the full matrix would take days. Capping the
        // iteration time makes the pilot pick a per-op invocation count that fits 25 ms:
        // fast ops still get hundreds–thousands of invocations (tight mean), while slow ops
        // (e.g. argsort@10M) drop to 1 invocation/iteration (bounded). 50 measured
        // iterations are preserved, so the statistical rigor the report wants is intact.
        var job = depth switch
        {
            BenchmarkDepth.Pass => Job.Dry
                .WithToolchain(InProcessEmitToolchain.Instance)
                .WithLaunchCount(1)
                .WithWarmupCount(0)
                .WithIterationCount(1)
                .WithInvocationCount(1)
                .WithUnrollFactor(1),
            BenchmarkDepth.Light => Job.Default
                .WithToolchain(InProcessEmitToolchain.Instance)
                .WithIterationTime(TimeInterval.FromMilliseconds(25))
                .WithWarmupCount(3)
                .WithIterationCount(8),
            _ => Job.Default
                .WithToolchain(InProcessEmitToolchain.Instance)
                .WithIterationTime(TimeInterval.FromMilliseconds(25))
                .WithWarmupCount(5)
                .WithIterationCount(50),
        };
        AddJob(job.AsDefault());

        // MemoryDiagnoser performs auxiliary workload executions. Pass mode is an execution gate:
        // exactly one workload invocation, with no warmup or diagnostic rerun.
        if (depth != BenchmarkDepth.Pass)
            AddDiagnoser(MemoryDiagnoser.Default);

        if (BenchmarkRunSelection.RequestedDtypes.Count > 0)
            AddFilter(new RequestedDtypeFilter());

        AddExporter(JsonExporter.FullCompressed);
        AddExporter(MarkdownExporter.GitHub);

        // ManualConfig starts with no logger; without one BenchmarkDotNet prints
        // "No loggers defined, you will not see any progress!" and the orchestrator sees
        // nothing during the (long) run. Restore the console logger.
        AddLogger(BenchmarkDotNet.Loggers.ConsoleLogger.Default);

        AddColumn(StatisticColumn.Mean);
        AddColumn(StatisticColumn.StdDev);
        AddColumn(StatisticColumn.Median);
        AddColumn(StatisticColumn.Min);
        AddColumn(StatisticColumn.Max);
        AddColumn(RankColumn.Arabic);

        WithSummaryStyle(SummaryStyle.Default.WithMaxParameterColumnWidth(40));
    }
}

public enum BenchmarkDepth { Pass, Light, Measure }

public static class BenchmarkRunSelection
{
    private static readonly IReadOnlyDictionary<string, string> Aliases =
        new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            ["bool"] = "bool", ["boolean"] = "bool",
            ["byte"] = "uint8", ["u8"] = "uint8", ["uint8"] = "uint8",
            ["sbyte"] = "int8", ["i8"] = "int8", ["int8"] = "int8",
            ["short"] = "int16", ["i16"] = "int16", ["int16"] = "int16",
            ["ushort"] = "uint16", ["u16"] = "uint16", ["uint16"] = "uint16",
            ["int"] = "int32", ["i32"] = "int32", ["int32"] = "int32",
            ["uint"] = "uint32", ["u32"] = "uint32", ["uint32"] = "uint32",
            ["long"] = "int64", ["i64"] = "int64", ["int64"] = "int64",
            ["ulong"] = "uint64", ["u64"] = "uint64", ["uint64"] = "uint64",
            ["char"] = "char", ["half"] = "float16", ["f16"] = "float16", ["float16"] = "float16",
            ["single"] = "float32", ["float"] = "float32", ["f32"] = "float32", ["float32"] = "float32",
            ["double"] = "float64", ["f64"] = "float64", ["float64"] = "float64",
            ["decimal"] = "decimal", ["dec"] = "decimal",
            ["complex"] = "complex128", ["c128"] = "complex128", ["complex128"] = "complex128",
        };

    public static BenchmarkDepth Depth =>
        (Environment.GetEnvironmentVariable("NUMSHARP_BENCHMARK_DEPTH") ?? "measure").ToLowerInvariant() switch
        {
            "pass" => BenchmarkDepth.Pass,
            "light" => BenchmarkDepth.Light,
            _ => BenchmarkDepth.Measure,
        };

    public static IReadOnlySet<string> RequestedDtypes { get; } = ParseRequestedDtypes();

    public static string? CanonicalDtype(object? value)
    {
        if (value is NPTypeCode code)
            return code switch
            {
                NPTypeCode.Boolean => "bool", NPTypeCode.Byte => "uint8", NPTypeCode.SByte => "int8",
                NPTypeCode.Int16 => "int16", NPTypeCode.UInt16 => "uint16", NPTypeCode.Int32 => "int32",
                NPTypeCode.UInt32 => "uint32", NPTypeCode.Int64 => "int64", NPTypeCode.UInt64 => "uint64",
                NPTypeCode.Char => "char", NPTypeCode.Half => "float16", NPTypeCode.Single => "float32",
                NPTypeCode.Double => "float64", NPTypeCode.Decimal => "decimal",
                NPTypeCode.Complex => "complex128", _ => null,
            };
        var text = value?.ToString();
        return text is not null && Aliases.TryGetValue(text, out var canonical) ? canonical : null;
    }

    private static IReadOnlySet<string> ParseRequestedDtypes()
    {
        var raw = Environment.GetEnvironmentVariable("NUMSHARP_BENCHMARK_DTYPES");
        if (string.IsNullOrWhiteSpace(raw))
            return new HashSet<string>(StringComparer.Ordinal);
        var result = new HashSet<string>(StringComparer.Ordinal);
        foreach (var token in raw.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            if (!Aliases.TryGetValue(token, out var canonical))
                throw new InvalidOperationException($"Unknown NUMSHARP_BENCHMARK_DTYPES value: {token}");
            result.Add(canonical);
        }
        return result;
    }
}

internal sealed class RequestedDtypeFilter : IFilter
{
    public bool Predicate(BenchmarkCase benchmarkCase)
    {
        var parameterDtypes = benchmarkCase.Parameters.Items
            .Where(parameter => parameter.Name.Contains("dtype", StringComparison.OrdinalIgnoreCase)
                || parameter.Name.Contains("source", StringComparison.OrdinalIgnoreCase)
                || parameter.Name.Contains("destination", StringComparison.OrdinalIgnoreCase)
                || parameter.Name.Equals("from", StringComparison.OrdinalIgnoreCase)
                || parameter.Name.Equals("to", StringComparison.OrdinalIgnoreCase))
            .Select(parameter => BenchmarkRunSelection.CanonicalDtype(parameter.Value))
            .Where(dtype => dtype is not null)
            .Cast<string>()
            .ToArray();

        // Multi-dtype cases (src→dst, pairwise, etc.) match when ANY dtype side was requested.
        // Non-parameterized official cases are historically keyed as float64 by merge-results.py.
        return parameterDtypes.Length > 0
            ? parameterDtypes.Any(BenchmarkRunSelection.RequestedDtypes.Contains)
            : BenchmarkRunSelection.RequestedDtypes.Contains("float64");
    }
}

/// <summary>
/// Full benchmark configuration for comprehensive analysis.
/// </summary>
public class FullBenchmarkConfig : ManualConfig
{
    public FullBenchmarkConfig()
    {
        AddJob(Job.Default
            .WithWarmupCount(5)
            .WithIterationCount(50));

        AddDiagnoser(MemoryDiagnoser.Default);
        AddExporter(JsonExporter.FullCompressed);
        AddExporter(MarkdownExporter.GitHub);
        AddExporter(HtmlExporter.Default);

        AddColumn(StatisticColumn.Mean);
        AddColumn(StatisticColumn.StdDev);
        AddColumn(StatisticColumn.Median);
        AddColumn(StatisticColumn.P95);
        AddColumn(BaselineRatioColumn.RatioMean);
        AddColumn(RankColumn.Arabic);
    }
}
