using System;
using System.Threading.Tasks;
using TUnit.Core;

namespace NumSharp.UnitTest.Utilities;

/// <summary>
/// Skips the test on non-Windows platforms (Linux, macOS).
/// Used for tests requiring GDI+/System.Drawing.Common.
/// </summary>
public class WindowsOnlyAttribute : SkipAttribute
{
    public WindowsOnlyAttribute() : base("Requires Windows (GDI+/System.Drawing.Common)") { }

    public override Task<bool> ShouldSkip(TestRegisteredContext context)
        => Task.FromResult(!OperatingSystem.IsWindows());
}
