using System.Diagnostics;

namespace NumSharp.Benchmark.CSharp.Infrastructure;

/// <summary>
/// Host-stability controls for the C# side. The orchestrator (benchmark/run_benchmark.py, via
/// benchmark/scripts/benchmark_host.py) pins BOTH languages to the same single performance core and
/// locks the CPU clock (turbo boost off) for the whole run; this is the C# half of that contract.
///
/// Why: on a hybrid, turbo-boosting host (the i9-13900K this suite runs on) an unpinned benchmark
/// thread migrates between P-cores of differing boost residency or lands on an E-core (2–3× slower),
/// and the High-performance plan's Aggressive boost swings the clock ~1.7× — measured as a 30–50 %
/// faster excursion in 2 of 50 BenchmarkDotNet iterations that never reproduced. BenchmarkDotNet
/// keeps fast outliers (OutlierMode.RemoveUpper), so that lucky window became Statistics.Min, the
/// harness's best-window basis. The clock lock is system-wide (powercfg) so it needs no C# hook; the
/// core pin is per process and is applied here, at the one seam BOTH runners pass through
/// (<see cref="OfficialBenchmarkConfig"/>'s constructor). InProcessEmit runs the benchmarks inside
/// this process, so the process affinity IS the benchmark thread's affinity — the same mechanism the
/// nditer probes validated with NS_PROBE_AFFINITY.
/// </summary>
public static class BenchmarkHost
{
    /// <summary>The orchestrator's contract; a hex affinity mask such as 0x4 (logical CPU 2).</summary>
    public const string AffinityVariable = "NUMSHARP_BENCHMARK_AFFINITY";

    /// <summary>The nditer probes' older spelling, accepted as a fallback.</summary>
    public const string LegacyAffinityVariable = "NS_PROBE_AFFINITY";

    private static long? _applied;

    /// <summary>
    /// Pins the current process to the mask named by <see cref="AffinityVariable"/> (or the legacy
    /// variable). Idempotent; a missing variable leaves the process unpinned and a failure is logged
    /// rather than thrown — a host control must never fail a benchmark run.
    /// </summary>
    public static long? PinToBenchmarkCore()
    {
        if (_applied is { } already)
            return already;

        var raw = Environment.GetEnvironmentVariable(AffinityVariable);
        if (string.IsNullOrWhiteSpace(raw))
            raw = Environment.GetEnvironmentVariable(LegacyAffinityVariable);
        if (string.IsNullOrWhiteSpace(raw))
            return null;

        try
        {
            var text = raw.Trim();
            if (text.StartsWith("0x", StringComparison.OrdinalIgnoreCase))
                text = text.Substring(2);
            var mask = Convert.ToInt64(text, 16);
            if (mask <= 0)
                return null;

            Process.GetCurrentProcess().ProcessorAffinity = (IntPtr)mask;
            _applied = mask;
            Console.WriteLine($"// Host: pinned process affinity to 0x{mask:x} (logical CPUs {DescribeMask(mask)})");
            return mask;
        }
        catch (Exception error)
        {
            Console.WriteLine($"// Host: could not pin process affinity from {AffinityVariable}='{raw}' ({error.Message}); continuing unpinned");
            return null;
        }
    }

    private static string DescribeMask(long mask)
    {
        var cpus = new List<int>();
        for (int i = 0; i < 64; i++)
            if (((mask >> i) & 1) != 0)
                cpus.Add(i);
        return string.Join(",", cpus);
    }
}
