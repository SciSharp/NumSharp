using System;
using System.Threading.Tasks;
using TUnit.Core;

namespace NumSharp.UnitTest.Utilities;

/// <summary>
/// Skips the test on non-Windows platforms (Linux, macOS).
/// Used for tests requiring GDI+/System.Drawing.Common.
///
/// This is separate from <see cref="WindowsOnlyAttribute"/> which is a CategoryAttribute
/// for CI filtering. Use both together:
/// <code>
/// [WindowsOnly]           // CategoryAttribute for CI filtering
/// [SkipOnNonWindows]      // SkipAttribute for runtime skip
/// public class BitmapTests { }
/// </code>
/// </summary>
public class SkipOnNonWindowsAttribute : SkipAttribute
{
    public SkipOnNonWindowsAttribute() : base("Requires Windows (GDI+/System.Drawing.Common)") { }

    public override Task<bool> ShouldSkip(TestRegisteredContext context)
        => Task.FromResult(!OperatingSystem.IsWindows());
}
