using System;
using System.Threading.Tasks;
using TUnit.Core;

namespace NumSharp.UnitTest.Utilities;

/// <summary>
/// Skips the test when available system memory is below the required threshold.
/// When skipping, logs all currently running tests to help diagnose memory pressure.
///
/// This is a runtime skip that works regardless of treenode-filter limitations.
/// Use together with [HighMemory] for documentation and filtering attempts:
/// <code>
/// [HighMemory]              // CategoryAttribute for documentation/filtering
/// [SkipOnLowMemory(8)]      // Runtime skip if less than 8GB available
/// public void LargeAllocationTest() { }
/// </code>
/// </summary>
public class SkipOnLowMemoryAttribute : SkipAttribute
{
    private readonly long _requiredMemoryGB;

    /// <summary>
    /// Creates a skip attribute that skips when available memory is below the threshold.
    /// </summary>
    /// <param name="requiredMemoryGB">Minimum required available memory in gigabytes.</param>
    public SkipOnLowMemoryAttribute(int requiredMemoryGB = 8)
        : base($"Requires {requiredMemoryGB}GB+ available memory")
    {
        _requiredMemoryGB = requiredMemoryGB;
    }

    public override Task<bool> ShouldSkip(TestRegisteredContext context)
    {
        // DISABLED FOR TESTING - always run, never skip
        // This tests if CI can pass without memory-based skipping
        var availableMemoryGB = TestMemoryTracker.GetAvailableMemoryGB();
        Console.Error.WriteLine($"[SkipOnLowMemory] DISABLED - Running test. Available: {availableMemoryGB:F1}GB, Would need: {_requiredMemoryGB}GB");

        return Task.FromResult(false); // Never skip

        /* ORIGINAL CODE - re-enable after testing
        var shouldSkip = availableMemoryGB < _requiredMemoryGB;

        if (shouldSkip)
        {
            // Get test name from TestRegisteredContext.TestDetails
            var className = context.TestDetails?.Class?.ClassType?.Name ?? "Unknown";
            var methodName = context.TestDetails?.TestName ?? context.TestName ?? "Unknown";
            var testName = $"{className}.{methodName}";

            // Log skip reason with available memory
            Console.Error.WriteLine($"\n[SkipOnLowMemory] SKIPPING {testName}");
            Console.Error.WriteLine($"[SkipOnLowMemory] Available: {availableMemoryGB:F1}GB, Required: {_requiredMemoryGB}GB");

            // Log what tests are currently running that might be consuming memory
            Console.Error.WriteLine(TestMemoryTracker.GetRunningTestsReport());
            Console.Error.WriteLine();
        }

        return Task.FromResult(shouldSkip);
        */
    }
}
