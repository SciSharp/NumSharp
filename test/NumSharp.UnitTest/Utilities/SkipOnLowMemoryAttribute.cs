using System;
using System.Threading.Tasks;
using TUnit.Core;

namespace NumSharp.UnitTest.Utilities;

/// <summary>
/// Skips the test when available system memory is below the required threshold.
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
        var availableMemoryGB = GetAvailableMemoryGB();
        var shouldSkip = availableMemoryGB < _requiredMemoryGB;

        if (shouldSkip)
        {
            // Update skip reason with actual available memory
            Console.WriteLine($"[SkipOnLowMemory] Skipping: {availableMemoryGB:F1}GB available, need {_requiredMemoryGB}GB");
        }

        return Task.FromResult(shouldSkip);
    }

    private static double GetAvailableMemoryGB()
    {
        try
        {
            // Use GC to get approximate available memory
            // This is a rough estimate but works cross-platform
            var gcInfo = GC.GetGCMemoryInfo();
            var availableBytes = gcInfo.TotalAvailableMemoryBytes;
            return availableBytes / (1024.0 * 1024.0 * 1024.0);
        }
        catch
        {
            // If we can't determine memory, assume we have enough
            return double.MaxValue;
        }
    }
}
