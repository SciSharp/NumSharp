using System.Text.Json.Serialization;

namespace NumSharp.Benchmark.Exploration.Infrastructure;

/// <summary>
/// Result of a single benchmark run.
/// </summary>
public record BenchResult
{
    /// <summary>Broadcasting scenario identifier (S1-S7).</summary>
    public required string Scenario { get; init; }

    /// <summary>SIMD strategy used (FULL, SCALAR, CHUNK, GATHER, LOOP).</summary>
    public required string Strategy { get; init; }

    /// <summary>Data type tested (byte, int16, int32, int64, float32, float64).</summary>
    public required string Dtype { get; init; }

    /// <summary>Number of elements in the array.</summary>
    public required int Size { get; init; }

    /// <summary>Mean execution time in microseconds.</summary>
    public required double MeanUs { get; init; }

    /// <summary>Standard deviation in microseconds.</summary>
    public required double StdDevUs { get; init; }

    /// <summary>Minimum execution time in microseconds.</summary>
    public required double MinUs { get; init; }

    /// <summary>Maximum execution time in microseconds.</summary>
    public required double MaxUs { get; init; }

    /// <summary>Throughput in GB/s (based on read+write bytes).</summary>
    public required double GBps { get; init; }

    /// <summary>Number of measurement repetitions.</summary>
    public required int Reps { get; init; }

    /// <summary>ISO 8601 timestamp when benchmark was run.</summary>
    public required string Timestamp { get; init; }

    /// <summary>Suite name for grouping.</summary>
    public string Suite { get; init; } = "";

    /// <summary>Additional notes or flags.</summary>
    public string Notes { get; init; } = "";

    /// <summary>Baseline comparison ratio (this / baseline).</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? SpeedupVsBaseline { get; init; }

    /// <summary>
    /// Calculate throughput given element count and element size.
    /// Assumes read LHS + read RHS + write result = 3 * elements * elementSize bytes.
    /// </summary>
    public static double CalculateGBps(int elements, int elementBytes, double microseconds)
    {
        // Binary op: read 2 inputs + write 1 output = 3 arrays
        var totalBytes = (long)elements * elementBytes * 3;
        var seconds = microseconds / 1_000_000.0;
        return seconds > 0 ? (totalBytes / 1e9) / seconds : 0;
    }

    /// <summary>
    /// Create a formatted one-line summary for console output.
    /// </summary>
    public override string ToString()
    {
        var speedup = SpeedupVsBaseline.HasValue ? $" ({SpeedupVsBaseline:F2}x)" : "";
        return $"{Scenario,-12} {Strategy,-10} {Dtype,-8} {Size,12:N0} | {MeanUs,12:F2} us ± {StdDevUs,8:F2} | {GBps,8:F2} GB/s{speedup}";
    }
}

/// <summary>
/// Aggregated results for multiple scenarios/dtypes/sizes.
/// </summary>
public class BenchResultSet
{
    public required string SuiteName { get; init; }
    public required string Description { get; init; }
    public required DateTime StartTime { get; init; }
    public DateTime EndTime { get; set; }
    public required string CpuModel { get; init; }
    public required string DotNetVersion { get; init; }
    public List<BenchResult> Results { get; } = new();

    public TimeSpan Duration => EndTime - StartTime;

    /// <summary>
    /// Add a result and return this for fluent chaining.
    /// </summary>
    public BenchResultSet Add(BenchResult result)
    {
        Results.Add(result);
        return this;
    }
}

/// <summary>
/// Element size lookup for each dtype.
/// </summary>
public static class DtypeInfo
{
    public static int GetElementSize(string dtype) => dtype.ToLowerInvariant() switch
    {
        "byte" or "uint8" or "bool" or "boolean" => 1,
        "int16" or "short" or "uint16" or "ushort" => 2,
        "int32" or "int" or "uint32" or "uint" or "float32" or "single" or "float" => 4,
        "int64" or "long" or "uint64" or "ulong" or "float64" or "double" => 8,
        "decimal" => 16,
        _ => throw new ArgumentException($"Unknown dtype: {dtype}")
    };

    /// <summary>
    /// Get the number of elements that fit in a Vector256.
    /// </summary>
    public static int GetVector256Count(string dtype) => 32 / GetElementSize(dtype);

    /// <summary>
    /// Get the number of elements that fit in a Vector128.
    /// </summary>
    public static int GetVector128Count(string dtype) => 16 / GetElementSize(dtype);
}
