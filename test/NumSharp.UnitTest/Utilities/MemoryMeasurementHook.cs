using System;
using System.Collections.Concurrent;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using TUnit.Core;

namespace NumSharp.UnitTest.Utilities;

/// <summary>
/// Tracks test execution and memory usage.
/// - Registers tests with TestMemoryTracker for active test monitoring
/// - Logs high memory allocations during test run
/// - Produces summary report at end of test run
/// </summary>
public class MemoryMeasurementHook
{
    private static readonly ConcurrentDictionary<string, long> _beforeMemory = new();
    private static readonly ConcurrentDictionary<string, MemoryResult> _results = new();
    private static readonly string _logFile = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "memory-profile.log");
    private static bool _initialized = false;
    private static readonly object _initLock = new();

    private static void Initialize()
    {
        lock (_initLock)
        {
            if (_initialized) return;
            _initialized = true;

            try
            {
                File.WriteAllText(_logFile, $"Memory Profile Started: {DateTime.Now}\n");
            }
            catch { }

            Console.Error.WriteLine("[MemoryHook] Memory tracking initialized");
        }
    }

    private static void Log(string message)
    {
        try
        {
            lock (_logFile)
            {
                File.AppendAllText(_logFile, $"[{DateTime.Now:HH:mm:ss.fff}] {message}\n");
            }
        }
        catch { }
    }

    [BeforeEvery(HookType.Test)]
    public static async Task BeforeEachTest(TestContext context)
    {
        Initialize();

        var testName = GetTestName(context);

        // Register with tracker for active test monitoring
        TestMemoryTracker.TestStarted(testName);

        // Force GC to get accurate baseline
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        var memoryBefore = GC.GetTotalMemory(forceFullCollection: false);
        _beforeMemory[testName] = memoryBefore;

        // Check memory and log if getting low (< 4GB available)
        TestMemoryTracker.LogIfMemoryLow(4.0);

        await Task.CompletedTask;
    }

    [AfterEvery(HookType.Test)]
    public static async Task AfterEachTest(TestContext context)
    {
        var testName = GetTestName(context);

        // Unregister from tracker
        TestMemoryTracker.TestCompleted(testName);

        if (!_beforeMemory.TryRemove(testName, out var memoryBefore))
        {
            memoryBefore = 0;
        }

        // Measure after (don't force GC to see actual allocation)
        var memoryAfter = GC.GetTotalMemory(forceFullCollection: false);
        var allocated = memoryAfter - memoryBefore;

        var result = new MemoryResult
        {
            TestName = testName,
            MemoryBefore = memoryBefore,
            MemoryAfter = memoryAfter,
            Allocated = allocated
        };

        _results[testName] = result;

        // Log if significant allocation (> 10 MB)
        var allocatedMB = allocated / (1024.0 * 1024.0);
        if (allocatedMB > 10.0)
        {
            var msg = $"{testName}: +{allocatedMB:F1} MB allocated";
            Log(msg);
            Console.Error.WriteLine($"[MemoryHook] {msg}");
        }

        // Check memory after test completes
        TestMemoryTracker.LogIfMemoryLow(4.0);

        await Task.CompletedTask;
    }

    [After(HookType.Assembly)]
    public static async Task AfterAllTests(AssemblyHookContext context)
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("\n" + new string('=', 100));
        sb.AppendLine("MEMORY USAGE SUMMARY (sorted by allocation, descending)");
        sb.AppendLine(new string('=', 100));

        var sorted = _results.Values
            .OrderByDescending(r => r.Allocated)
            .Take(100);

        foreach (var result in sorted)
        {
            var allocatedMB = result.Allocated / (1024.0 * 1024.0);

            if (allocatedMB > 1.0)
            {
                sb.AppendLine($"  {allocatedMB,10:F2} MB | {result.TestName}");
            }
        }

        sb.AppendLine(new string('=', 100));
        sb.AppendLine($"Total tests measured: {_results.Count}");

        var totalAllocated = _results.Values.Sum(r => Math.Max(0, r.Allocated));
        sb.AppendLine($"Total allocated: {totalAllocated / (1024.0 * 1024.0 * 1024.0):F2} GB");

        var testsOver100MB = _results.Values.Count(r => r.Allocated > 100 * 1024 * 1024);
        var testsOver1GB = _results.Values.Count(r => r.Allocated > 1024L * 1024 * 1024);
        sb.AppendLine($"Tests > 100 MB: {testsOver100MB}");
        sb.AppendLine($"Tests > 1 GB: {testsOver1GB}");
        sb.AppendLine(new string('=', 100));

        var summary = sb.ToString();
        Log(summary);
        Console.WriteLine(summary);

        await Task.CompletedTask;
    }

    private static string GetTestName(TestContext context)
    {
        var className = context.Metadata.TestDetails?.Class?.ClassType?.Name ?? "Unknown";
        var testName = context.Metadata.TestName;
        return $"{className}.{testName}";
    }

    private class MemoryResult
    {
        public string TestName { get; set; } = "";
        public long MemoryBefore { get; set; }
        public long MemoryAfter { get; set; }
        public long Allocated { get; set; }
    }
}
