using System.Runtime.InteropServices;
using NumSharp.Benchmark.Exploration.Infrastructure;

namespace NumSharp.Benchmark.Exploration.Isolated;

/// <summary>
/// Discover size thresholds where SIMD starts helping.
/// Tests fine-grained sizes to find crossover points per dtype.
/// </summary>
public static unsafe class SizeThresholds
{
    private const string Suite = "SizeThresholds";

    /// <summary>
    /// Size range optimized for threshold discovery.
    /// </summary>
    public static readonly int[] ThresholdSizes = [
        8, 16, 24, 32, 48, 64, 96, 128,
        192, 256, 384, 512, 768, 1024,
        1536, 2048, 3072, 4096, 6144, 8192,
        12288, 16384, 24576, 32768, 49152, 65536,
        98304, 131072
    ];

    /// <summary>
    /// Run threshold discovery for all dtypes.
    /// </summary>
    public static List<BenchResult> RunAll(string[]? dtypes = null, bool quick = false)
    {
        dtypes ??= quick ? Dtypes.Common : Dtypes.All;
        var sizes = quick ? ThresholdSizes.Where(s => s >= 32 && (s <= 1024 || s >= 16384)).ToArray() : ThresholdSizes;
        var warmup = quick ? 2 : BenchFramework.DefaultWarmup;
        var measure = quick ? 5 : BenchFramework.DefaultMeasure;

        var results = new List<BenchResult>();

        BenchFramework.PrintHeader($"{Suite}: SIMD Threshold Discovery");

        foreach (var dtype in dtypes)
        {
            BenchFramework.PrintDivider($"dtype={dtype}");
            results.AddRange(FindThreshold(dtype, sizes, warmup, measure));
        }

        // Summary: find crossover points
        PrintCrossoverSummary(results);

        return results;
    }

    /// <summary>
    /// Find the threshold where SIMD becomes faster for a specific dtype.
    /// </summary>
    public static List<BenchResult> FindThreshold(string dtype, int[] sizes, int warmup, int measure)
    {
        var results = new List<BenchResult>();
        var elementBytes = DtypeInfo.GetElementSize(dtype);

        switch (dtype.ToLowerInvariant())
        {
            case "float64" or "double":
            {
                foreach (var size in sizes)
                {
                    var lhs = SimdImplementations.AllocateAligned<double>(size);
                    var rhs = SimdImplementations.AllocateAligned<double>(size);
                    var result = SimdImplementations.AllocateAligned<double>(size);
                    SimdImplementations.FillRandom(lhs, size, 42);
                    SimdImplementations.FillRandom(rhs, size, 43);

                    var baseline = BenchFramework.Run(
                        () => SimdImplementations.AddScalarLoop_Float64(lhs, rhs, result, size),
                        "Threshold", "Scalar", dtype, size, elementBytes, warmup, measure, Suite);

                    var simd = BenchFramework.Run(
                        () => SimdImplementations.AddFull_Float64(lhs, rhs, result, size),
                        "Threshold", "SIMD", dtype, size, elementBytes, warmup, measure, Suite);
                    simd = simd with { SpeedupVsBaseline = baseline.MeanUs / simd.MeanUs };

                    results.Add(baseline);
                    results.Add(simd);

                    // Compact output for threshold discovery
                    var winner = simd.SpeedupVsBaseline > 1.0 ? "SIMD" : "Scalar";
                    var icon = simd.SpeedupVsBaseline > 1.0 ? "✓" : " ";
                    Console.WriteLine($"  {size,8:N0} | Scalar: {baseline.MeanUs,10:F2} us | SIMD: {simd.MeanUs,10:F2} us | {simd.SpeedupVsBaseline,6:F2}x {icon}");

                    SimdImplementations.FreeAligned(lhs);
                    SimdImplementations.FreeAligned(rhs);
                    SimdImplementations.FreeAligned(result);
                }
                break;
            }

            case "float32" or "single" or "float":
            {
                foreach (var size in sizes)
                {
                    var lhs = SimdImplementations.AllocateAligned<float>(size);
                    var rhs = SimdImplementations.AllocateAligned<float>(size);
                    var result = SimdImplementations.AllocateAligned<float>(size);
                    SimdImplementations.FillRandom(lhs, size, 42);
                    SimdImplementations.FillRandom(rhs, size, 43);

                    var baseline = BenchFramework.Run(
                        () => SimdImplementations.AddScalarLoop_Float32(lhs, rhs, result, size),
                        "Threshold", "Scalar", dtype, size, elementBytes, warmup, measure, Suite);

                    var simd = BenchFramework.Run(
                        () => SimdImplementations.AddFull_Float32(lhs, rhs, result, size),
                        "Threshold", "SIMD", dtype, size, elementBytes, warmup, measure, Suite);
                    simd = simd with { SpeedupVsBaseline = baseline.MeanUs / simd.MeanUs };

                    results.Add(baseline);
                    results.Add(simd);

                    var icon = simd.SpeedupVsBaseline > 1.0 ? "✓" : " ";
                    Console.WriteLine($"  {size,8:N0} | Scalar: {baseline.MeanUs,10:F2} us | SIMD: {simd.MeanUs,10:F2} us | {simd.SpeedupVsBaseline,6:F2}x {icon}");

                    SimdImplementations.FreeAligned(lhs);
                    SimdImplementations.FreeAligned(rhs);
                    SimdImplementations.FreeAligned(result);
                }
                break;
            }

            case "int32" or "int":
            {
                foreach (var size in sizes)
                {
                    var lhs = SimdImplementations.AllocateAligned<int>(size);
                    var rhs = SimdImplementations.AllocateAligned<int>(size);
                    var result = SimdImplementations.AllocateAligned<int>(size);
                    SimdImplementations.FillRandom(lhs, size, 42);
                    SimdImplementations.FillRandom(rhs, size, 43);

                    var baseline = BenchFramework.Run(
                        () => SimdImplementations.AddScalarLoop_Int32(lhs, rhs, result, size),
                        "Threshold", "Scalar", dtype, size, elementBytes, warmup, measure, Suite);

                    var simd = BenchFramework.Run(
                        () => SimdImplementations.AddFull_Int32(lhs, rhs, result, size),
                        "Threshold", "SIMD", dtype, size, elementBytes, warmup, measure, Suite);
                    simd = simd with { SpeedupVsBaseline = baseline.MeanUs / simd.MeanUs };

                    results.Add(baseline);
                    results.Add(simd);

                    var icon = simd.SpeedupVsBaseline > 1.0 ? "✓" : " ";
                    Console.WriteLine($"  {size,8:N0} | Scalar: {baseline.MeanUs,10:F2} us | SIMD: {simd.MeanUs,10:F2} us | {simd.SpeedupVsBaseline,6:F2}x {icon}");

                    SimdImplementations.FreeAligned(lhs);
                    SimdImplementations.FreeAligned(rhs);
                    SimdImplementations.FreeAligned(result);
                }
                break;
            }

            case "int64" or "long":
            {
                foreach (var size in sizes)
                {
                    var lhs = SimdImplementations.AllocateAligned<long>(size);
                    var rhs = SimdImplementations.AllocateAligned<long>(size);
                    var result = SimdImplementations.AllocateAligned<long>(size);
                    SimdImplementations.FillRandom(lhs, size, 42);
                    SimdImplementations.FillRandom(rhs, size, 43);

                    var baseline = BenchFramework.Run(
                        () => SimdImplementations.AddScalarLoop_Int64(lhs, rhs, result, size),
                        "Threshold", "Scalar", dtype, size, elementBytes, warmup, measure, Suite);

                    var simd = BenchFramework.Run(
                        () => SimdImplementations.AddFull_Int64(lhs, rhs, result, size),
                        "Threshold", "SIMD", dtype, size, elementBytes, warmup, measure, Suite);
                    simd = simd with { SpeedupVsBaseline = baseline.MeanUs / simd.MeanUs };

                    results.Add(baseline);
                    results.Add(simd);

                    var icon = simd.SpeedupVsBaseline > 1.0 ? "✓" : " ";
                    Console.WriteLine($"  {size,8:N0} | Scalar: {baseline.MeanUs,10:F2} us | SIMD: {simd.MeanUs,10:F2} us | {simd.SpeedupVsBaseline,6:F2}x {icon}");

                    SimdImplementations.FreeAligned(lhs);
                    SimdImplementations.FreeAligned(rhs);
                    SimdImplementations.FreeAligned(result);
                }
                break;
            }

            case "int16" or "short":
            {
                foreach (var size in sizes)
                {
                    var lhs = SimdImplementations.AllocateAligned<short>(size);
                    var rhs = SimdImplementations.AllocateAligned<short>(size);
                    var result = SimdImplementations.AllocateAligned<short>(size);
                    SimdImplementations.FillRandom(lhs, size, 42);
                    SimdImplementations.FillRandom(rhs, size, 43);

                    var baseline = BenchFramework.Run(
                        () => SimdImplementations.AddScalarLoop_Int16(lhs, rhs, result, size),
                        "Threshold", "Scalar", dtype, size, elementBytes, warmup, measure, Suite);

                    var simd = BenchFramework.Run(
                        () => SimdImplementations.AddFull_Int16(lhs, rhs, result, size),
                        "Threshold", "SIMD", dtype, size, elementBytes, warmup, measure, Suite);
                    simd = simd with { SpeedupVsBaseline = baseline.MeanUs / simd.MeanUs };

                    results.Add(baseline);
                    results.Add(simd);

                    var icon = simd.SpeedupVsBaseline > 1.0 ? "✓" : " ";
                    Console.WriteLine($"  {size,8:N0} | Scalar: {baseline.MeanUs,10:F2} us | SIMD: {simd.MeanUs,10:F2} us | {simd.SpeedupVsBaseline,6:F2}x {icon}");

                    SimdImplementations.FreeAligned(lhs);
                    SimdImplementations.FreeAligned(rhs);
                    SimdImplementations.FreeAligned(result);
                }
                break;
            }

            case "byte" or "uint8":
            {
                foreach (var size in sizes)
                {
                    var lhs = SimdImplementations.AllocateAligned<byte>(size);
                    var rhs = SimdImplementations.AllocateAligned<byte>(size);
                    var result = SimdImplementations.AllocateAligned<byte>(size);
                    SimdImplementations.FillRandom(lhs, size, 42);
                    SimdImplementations.FillRandom(rhs, size, 43);

                    var baseline = BenchFramework.Run(
                        () => SimdImplementations.AddScalarLoop_Byte(lhs, rhs, result, size),
                        "Threshold", "Scalar", dtype, size, elementBytes, warmup, measure, Suite);

                    var simd = BenchFramework.Run(
                        () => SimdImplementations.AddFull_Byte(lhs, rhs, result, size),
                        "Threshold", "SIMD", dtype, size, elementBytes, warmup, measure, Suite);
                    simd = simd with { SpeedupVsBaseline = baseline.MeanUs / simd.MeanUs };

                    results.Add(baseline);
                    results.Add(simd);

                    var icon = simd.SpeedupVsBaseline > 1.0 ? "✓" : " ";
                    Console.WriteLine($"  {size,8:N0} | Scalar: {baseline.MeanUs,10:F2} us | SIMD: {simd.MeanUs,10:F2} us | {simd.SpeedupVsBaseline,6:F2}x {icon}");

                    SimdImplementations.FreeAligned(lhs);
                    SimdImplementations.FreeAligned(rhs);
                    SimdImplementations.FreeAligned(result);
                }
                break;
            }
        }

        return results;
    }

    /// <summary>
    /// Analyze results to find crossover points.
    /// </summary>
    private static void PrintCrossoverSummary(List<BenchResult> results)
    {
        Console.WriteLine();
        Console.WriteLine("=== Crossover Summary ===");
        Console.WriteLine();

        var byDtype = results
            .Where(r => r.Strategy == "SIMD" && r.SpeedupVsBaseline.HasValue)
            .GroupBy(r => r.Dtype);

        foreach (var group in byDtype.OrderBy(g => g.Key))
        {
            var ordered = group.OrderBy(r => r.Size).ToList();

            // Find first size where SIMD is consistently faster
            int? threshold = null;
            int consecutiveWins = 0;
            const int requiredWins = 2; // Require N consecutive wins to call it a threshold

            foreach (var r in ordered)
            {
                if (r.SpeedupVsBaseline > 1.05) // 5% margin to avoid noise
                {
                    consecutiveWins++;
                    if (consecutiveWins >= requiredWins && threshold == null)
                    {
                        // Backtrack to find the first win in this streak
                        var idx = ordered.IndexOf(r) - requiredWins + 1;
                        threshold = ordered[idx].Size;
                    }
                }
                else
                {
                    consecutiveWins = 0;
                }
            }

            // Find maximum speedup
            var maxSpeedup = ordered.MaxBy(r => r.SpeedupVsBaseline);
            var minSpeedup = ordered.MinBy(r => r.SpeedupVsBaseline);

            Console.WriteLine($"{group.Key,-10}: Threshold = {threshold?.ToString("N0") ?? "N/A",-8} | " +
                $"Max speedup = {maxSpeedup?.SpeedupVsBaseline:F2}x at {maxSpeedup?.Size:N0} | " +
                $"Min speedup = {minSpeedup?.SpeedupVsBaseline:F2}x at {minSpeedup?.Size:N0}");
        }

        Console.WriteLine();
        Console.WriteLine("Legend: Threshold = smallest size where SIMD is consistently >5% faster");
    }

    /// <summary>
    /// Generate recommended threshold constants based on results.
    /// </summary>
    public static string GenerateThresholdConstants(List<BenchResult> results)
    {
        var lines = new List<string>
        {
            "/// <summary>",
            "/// SIMD threshold constants discovered empirically.",
            "/// Use SIMD when element count >= threshold.",
            "/// </summary>",
            "public static class SimdThresholds",
            "{"
        };

        var byDtype = results
            .Where(r => r.Strategy == "SIMD" && r.SpeedupVsBaseline.HasValue)
            .GroupBy(r => r.Dtype);

        foreach (var group in byDtype.OrderBy(g => g.Key))
        {
            var ordered = group.OrderBy(r => r.Size).ToList();
            int? threshold = null;
            int consecutiveWins = 0;

            foreach (var r in ordered)
            {
                if (r.SpeedupVsBaseline > 1.05)
                {
                    consecutiveWins++;
                    if (consecutiveWins >= 2 && threshold == null)
                    {
                        var idx = ordered.IndexOf(r) - 1;
                        threshold = ordered[Math.Max(0, idx)].Size;
                    }
                }
                else
                {
                    consecutiveWins = 0;
                }
            }

            var dtypeName = group.Key switch
            {
                "byte" => "Byte",
                "int16" => "Int16",
                "int32" => "Int32",
                "int64" => "Int64",
                "float32" => "Float32",
                "float64" => "Float64",
                _ => group.Key
            };

            lines.Add($"    public const int MinSizeForSimd_{dtypeName} = {threshold ?? 32};");
        }

        lines.Add("}");

        return string.Join(Environment.NewLine, lines);
    }
}
