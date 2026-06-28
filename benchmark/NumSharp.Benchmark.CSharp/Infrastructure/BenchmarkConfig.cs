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
using Perfolizer.Horology;

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
        // IterationTime is capped at 25 ms. BenchmarkDotNet's default Throughput strategy
        // ramps the per-iteration invocation count until each iteration takes ~0.5 s — which
        // is right for nanosecond microbenchmarks but catastrophic for array ops in the
        // µs–ms range: a 10M-element op (~60 µs) gets 8192 invocations/iteration, so 50
        // iterations = ~25 s PER case, and the full matrix would take days. Capping the
        // iteration time makes the pilot pick a per-op invocation count that fits 25 ms:
        // fast ops still get hundreds–thousands of invocations (tight mean), while slow ops
        // (e.g. argsort@10M) drop to 1 invocation/iteration (bounded). 50 measured
        // iterations are preserved, so the statistical rigor the report wants is intact.
        AddJob(Job.Default
            .WithToolchain(InProcessEmitToolchain.Instance)
            .WithIterationTime(TimeInterval.FromMilliseconds(25))
            .WithWarmupCount(5)
            .WithIterationCount(50)
            .AsDefault());

        AddDiagnoser(MemoryDiagnoser.Default);

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
