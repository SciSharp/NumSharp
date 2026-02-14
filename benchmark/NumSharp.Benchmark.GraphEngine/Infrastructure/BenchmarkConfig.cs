using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Exporters;
using BenchmarkDotNet.Exporters.Json;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Diagnosers;
using BenchmarkDotNet.Columns;
using BenchmarkDotNet.Reports;
using BenchmarkDotNet.Filters;
using BenchmarkDotNet.Validators;

namespace NumSharp.Benchmark.GraphEngine.Infrastructure;

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
