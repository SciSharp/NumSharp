using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace NumSharp.Benchmark.Exploration.Infrastructure;

/// <summary>
/// Output formatters for benchmark results.
/// Supports CSV, JSON, and Markdown table formats.
/// </summary>
public static class OutputFormatters
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
    };

    /// <summary>
    /// Export results to a CSV file.
    /// </summary>
    public static void ExportCsv(IEnumerable<BenchResult> results, string filePath)
    {
        var sb = new StringBuilder();

        // Header
        sb.AppendLine("Scenario,Strategy,Dtype,Size,MeanUs,StdDevUs,MinUs,MaxUs,GBps,Reps,Timestamp,Suite,Notes,SpeedupVsBaseline");

        // Data rows
        foreach (var r in results)
        {
            sb.AppendLine(string.Join(",",
                Escape(r.Scenario),
                Escape(r.Strategy),
                Escape(r.Dtype),
                r.Size.ToString(CultureInfo.InvariantCulture),
                r.MeanUs.ToString("F4", CultureInfo.InvariantCulture),
                r.StdDevUs.ToString("F4", CultureInfo.InvariantCulture),
                r.MinUs.ToString("F4", CultureInfo.InvariantCulture),
                r.MaxUs.ToString("F4", CultureInfo.InvariantCulture),
                r.GBps.ToString("F4", CultureInfo.InvariantCulture),
                r.Reps.ToString(CultureInfo.InvariantCulture),
                Escape(r.Timestamp),
                Escape(r.Suite),
                Escape(r.Notes),
                r.SpeedupVsBaseline?.ToString("F4", CultureInfo.InvariantCulture) ?? ""
            ));
        }

        Directory.CreateDirectory(Path.GetDirectoryName(filePath)!);
        File.WriteAllText(filePath, sb.ToString());
        Console.WriteLine($"CSV exported to: {filePath}");
    }

    /// <summary>
    /// Export results to a JSON file.
    /// </summary>
    public static void ExportJson(BenchResultSet resultSet, string filePath)
    {
        var json = JsonSerializer.Serialize(resultSet, JsonOptions);
        Directory.CreateDirectory(Path.GetDirectoryName(filePath)!);
        File.WriteAllText(filePath, json);
        Console.WriteLine($"JSON exported to: {filePath}");
    }

    /// <summary>
    /// Export results to a JSON file (simple array format).
    /// </summary>
    public static void ExportJson(IEnumerable<BenchResult> results, string filePath)
    {
        var json = JsonSerializer.Serialize(results.ToList(), JsonOptions);
        Directory.CreateDirectory(Path.GetDirectoryName(filePath)!);
        File.WriteAllText(filePath, json);
        Console.WriteLine($"JSON exported to: {filePath}");
    }

    /// <summary>
    /// Generate a Markdown table from results.
    /// </summary>
    public static string ToMarkdownTable(IEnumerable<BenchResult> results, string title = "")
    {
        var sb = new StringBuilder();

        if (!string.IsNullOrEmpty(title))
        {
            sb.AppendLine($"## {title}");
            sb.AppendLine();
        }

        // Header
        sb.AppendLine("| Scenario | Strategy | Dtype | Size | Mean (us) | StdDev | GB/s | Speedup |");
        sb.AppendLine("|----------|----------|-------|-----:|----------:|-------:|-----:|--------:|");

        // Data rows
        foreach (var r in results)
        {
            var speedup = r.SpeedupVsBaseline.HasValue ? $"{r.SpeedupVsBaseline:F2}x" : "-";
            sb.AppendLine($"| {r.Scenario} | {r.Strategy} | {r.Dtype} | {r.Size:N0} | {r.MeanUs:F2} | {r.StdDevUs:F2} | {r.GBps:F2} | {speedup} |");
        }

        return sb.ToString();
    }

    /// <summary>
    /// Generate a compact comparison table (baseline vs optimized).
    /// </summary>
    public static string ToComparisonTable(IEnumerable<BenchResult> results, string title = "")
    {
        var sb = new StringBuilder();

        if (!string.IsNullOrEmpty(title))
        {
            sb.AppendLine($"## {title}");
            sb.AppendLine();
        }

        // Group by scenario/dtype/size, show baseline and optimized side-by-side
        var grouped = results
            .GroupBy(r => (r.Scenario, r.Dtype, r.Size))
            .OrderBy(g => g.Key.Scenario)
            .ThenBy(g => g.Key.Dtype)
            .ThenBy(g => g.Key.Size);

        sb.AppendLine("| Scenario | Dtype | Size | Baseline (us) | Optimized (us) | Speedup |");
        sb.AppendLine("|----------|-------|-----:|--------------:|---------------:|--------:|");

        foreach (var g in grouped)
        {
            var baseline = g.FirstOrDefault(r => r.Strategy == "Baseline");
            var optimized = g.FirstOrDefault(r => r.Strategy != "Baseline" && r.SpeedupVsBaseline.HasValue);

            if (baseline != null && optimized != null)
            {
                var icon = optimized.SpeedupVsBaseline switch
                {
                    > 2.0 => " ✅",
                    > 1.2 => " 🟡",
                    > 1.0 => " 🟢",
                    _ => " 🔴"
                };

                sb.AppendLine($"| {g.Key.Scenario} | {g.Key.Dtype} | {g.Key.Size:N0} | {baseline.MeanUs:F2} | {optimized.MeanUs:F2} | {optimized.SpeedupVsBaseline:F2}x{icon} |");
            }
        }

        return sb.ToString();
    }

    /// <summary>
    /// Generate a pivot table with dtypes as columns and sizes as rows.
    /// </summary>
    public static string ToPivotTable(IEnumerable<BenchResult> results, string scenario, string metric = "speedup")
    {
        var filtered = results.Where(r => r.Scenario == scenario && r.SpeedupVsBaseline.HasValue).ToList();
        if (!filtered.Any()) return "";

        var sb = new StringBuilder();
        sb.AppendLine($"### {scenario} - Speedup Matrix");
        sb.AppendLine();

        var dtypes = filtered.Select(r => r.Dtype).Distinct().OrderBy(d => d).ToList();
        var sizes = filtered.Select(r => r.Size).Distinct().OrderBy(s => s).ToList();

        // Header
        sb.Append("| Size |");
        foreach (var d in dtypes) sb.Append($" {d} |");
        sb.AppendLine();

        sb.Append("|-----:|");
        foreach (var _ in dtypes) sb.Append("------:|");
        sb.AppendLine();

        // Rows
        foreach (var size in sizes)
        {
            sb.Append($"| {size:N0} |");
            foreach (var dtype in dtypes)
            {
                var result = filtered.FirstOrDefault(r => r.Dtype == dtype && r.Size == size);
                if (result != null)
                {
                    var value = metric switch
                    {
                        "speedup" => $"{result.SpeedupVsBaseline:F2}x",
                        "mean" => $"{result.MeanUs:F1}",
                        "gbps" => $"{result.GBps:F1}",
                        _ => "-"
                    };
                    sb.Append($" {value} |");
                }
                else
                {
                    sb.Append(" - |");
                }
            }
            sb.AppendLine();
        }

        return sb.ToString();
    }

    /// <summary>
    /// Export results to Markdown file.
    /// </summary>
    public static void ExportMarkdown(BenchResultSet resultSet, string filePath)
    {
        var sb = new StringBuilder();

        sb.AppendLine($"# {resultSet.SuiteName}");
        sb.AppendLine();
        sb.AppendLine($"_{resultSet.Description}_");
        sb.AppendLine();
        sb.AppendLine("## Environment");
        sb.AppendLine();
        sb.AppendLine($"- **CPU**: {resultSet.CpuModel}");
        sb.AppendLine($"- **.NET**: {resultSet.DotNetVersion}");
        sb.AppendLine($"- **Start**: {resultSet.StartTime:u}");
        sb.AppendLine($"- **Duration**: {resultSet.Duration:hh\\:mm\\:ss}");
        sb.AppendLine();

        // Group results by suite
        var grouped = resultSet.Results.GroupBy(r => r.Suite).OrderBy(g => g.Key);

        foreach (var group in grouped)
        {
            var suiteName = string.IsNullOrEmpty(group.Key) ? "Results" : group.Key;
            sb.AppendLine(ToMarkdownTable(group, suiteName));
            sb.AppendLine();
        }

        Directory.CreateDirectory(Path.GetDirectoryName(filePath)!);
        File.WriteAllText(filePath, sb.ToString());
        Console.WriteLine($"Markdown exported to: {filePath}");
    }

    /// <summary>
    /// Print a summary of results to console.
    /// </summary>
    public static void PrintSummary(IEnumerable<BenchResult> results)
    {
        var list = results.ToList();

        Console.WriteLine();
        Console.WriteLine("=== Summary ===");
        Console.WriteLine($"Total benchmarks: {list.Count}");

        var withSpeedup = list.Where(r => r.SpeedupVsBaseline.HasValue).ToList();
        if (withSpeedup.Any())
        {
            var avgSpeedup = withSpeedup.Average(r => r.SpeedupVsBaseline!.Value);
            var maxSpeedup = withSpeedup.Max(r => r.SpeedupVsBaseline!.Value);
            var minSpeedup = withSpeedup.Min(r => r.SpeedupVsBaseline!.Value);
            var wins = withSpeedup.Count(r => r.SpeedupVsBaseline > 1.0);

            Console.WriteLine($"Comparisons: {withSpeedup.Count}");
            Console.WriteLine($"  Wins (speedup > 1.0): {wins} ({100.0 * wins / withSpeedup.Count:F1}%)");
            Console.WriteLine($"  Avg speedup: {avgSpeedup:F2}x");
            Console.WriteLine($"  Max speedup: {maxSpeedup:F2}x");
            Console.WriteLine($"  Min speedup: {minSpeedup:F2}x");

            // Best result
            var best = withSpeedup.OrderByDescending(r => r.SpeedupVsBaseline).First();
            Console.WriteLine($"  Best: {best.Scenario} {best.Dtype} {best.Size:N0} = {best.SpeedupVsBaseline:F2}x");

            // Worst result
            var worst = withSpeedup.OrderBy(r => r.SpeedupVsBaseline).First();
            Console.WriteLine($"  Worst: {worst.Scenario} {worst.Dtype} {worst.Size:N0} = {worst.SpeedupVsBaseline:F2}x");
        }
    }

    private static string Escape(string s)
    {
        if (s.Contains(',') || s.Contains('"') || s.Contains('\n'))
        {
            return $"\"{s.Replace("\"", "\"\"")}\"";
        }
        return s;
    }
}
