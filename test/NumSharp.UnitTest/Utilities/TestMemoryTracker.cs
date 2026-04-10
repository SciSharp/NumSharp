using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Threading.Tasks;
using TUnit.Core;

namespace NumSharp.UnitTest.Utilities;

/// <summary>
/// Tracks currently running tests and their memory usage.
/// Logs active tests when memory pressure is detected.
/// </summary>
public static class TestMemoryTracker
{
    private static readonly ConcurrentDictionary<string, TestInfo> _runningTests = new();
    private static long _peakMemoryBytes = 0;
    private static readonly object _lock = new();

    /// <summary>
    /// Records a test starting execution.
    /// </summary>
    public static void TestStarted(string testName)
    {
        var info = new TestInfo
        {
            TestName = testName,
            StartTime = DateTime.UtcNow,
            MemoryAtStart = GC.GetTotalMemory(forceFullCollection: false)
        };
        _runningTests[testName] = info;
    }

    /// <summary>
    /// Records a test completing execution.
    /// </summary>
    public static void TestCompleted(string testName)
    {
        _runningTests.TryRemove(testName, out _);
    }

    /// <summary>
    /// Gets currently running tests with their memory info.
    /// </summary>
    public static string GetRunningTestsReport()
    {
        var currentMemory = GC.GetTotalMemory(forceFullCollection: false);
        var currentMemoryMB = currentMemory / (1024.0 * 1024.0);

        lock (_lock)
        {
            if (currentMemory > _peakMemoryBytes)
                _peakMemoryBytes = currentMemory;
        }

        var tests = _runningTests.Values
            .OrderBy(t => t.StartTime)
            .ToList();

        if (tests.Count == 0)
            return $"[MemoryTracker] No tests currently running. Memory: {currentMemoryMB:F1} MB";

        var lines = new System.Text.StringBuilder();
        lines.AppendLine($"[MemoryTracker] Current memory: {currentMemoryMB:F1} MB, Peak: {_peakMemoryBytes / (1024.0 * 1024.0):F1} MB");
        lines.AppendLine($"[MemoryTracker] {tests.Count} test(s) currently running:");

        foreach (var test in tests)
        {
            var elapsed = DateTime.UtcNow - test.StartTime;
            var memoryAtStartMB = test.MemoryAtStart / (1024.0 * 1024.0);
            lines.AppendLine($"  [{elapsed.TotalSeconds:F1}s] {test.TestName} (started at {memoryAtStartMB:F1} MB)");
        }

        return lines.ToString();
    }

    /// <summary>
    /// Gets available system memory in GB.
    /// </summary>
    public static double GetAvailableMemoryGB()
    {
        try
        {
            var gcInfo = GC.GetGCMemoryInfo();
            return gcInfo.TotalAvailableMemoryBytes / (1024.0 * 1024.0 * 1024.0);
        }
        catch
        {
            return double.MaxValue;
        }
    }

    /// <summary>
    /// Logs running tests if memory is below threshold.
    /// </summary>
    public static void LogIfMemoryLow(double thresholdGB)
    {
        var availableGB = GetAvailableMemoryGB();
        if (availableGB < thresholdGB)
        {
            Console.Error.WriteLine($"\n[MemoryTracker] WARNING: Low memory detected! Available: {availableGB:F1} GB (threshold: {thresholdGB} GB)");
            Console.Error.WriteLine(GetRunningTestsReport());
            Console.Error.WriteLine();
        }
    }

    private class TestInfo
    {
        public string TestName { get; set; } = "";
        public DateTime StartTime { get; set; }
        public long MemoryAtStart { get; set; }
    }
}
