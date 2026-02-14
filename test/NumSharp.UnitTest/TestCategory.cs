using TUnit.Core;

namespace NumSharp.UnitTest;

/// <summary>
/// Defines test categories used for filtering tests in CI and local development.
///
/// <para><b>How Categories Work in CI:</b></para>
/// <para>
/// The CI pipeline (<c>.github/workflows/build-and-release.yml</c>) uses TUnit's
/// <c>--treenode-filter</c> to exclude certain categories from test runs:
/// </para>
/// <code>
/// TEST_FILTER: '/*/*/*/*[Category!=OpenBugs]'
/// </code>
/// <para>
/// This filter excludes all tests marked with <c>[Category("OpenBugs")]</c> from CI runs.
/// Tests in other categories (like <see cref="Misaligned"/>) still run and can fail CI.
/// </para>
///
/// <para><b>Local Development Filtering:</b></para>
/// <code>
/// # Exclude OpenBugs (same as CI)
/// dotnet test -- --treenode-filter "/*/*/*/*[Category!=OpenBugs]"
///
/// # Run ONLY OpenBugs tests (to verify fixes)
/// dotnet test -- --treenode-filter "/*/*/*/*[Category=OpenBugs]"
///
/// # Run ONLY Misaligned tests
/// dotnet test -- --treenode-filter "/*/*/*/*[Category=Misaligned]"
/// </code>
///
/// <para><b>Usage:</b></para>
/// <para>Apply at class level for entire test classes:</para>
/// <code>
/// [OpenBugs]  // or [Category(TestCategory.OpenBugs)]
/// public class MyBugReproTests { ... }
/// </code>
/// <para>Or at individual test level:</para>
/// <code>
/// [Test]
/// [OpenBugs]  // or [Category(TestCategory.OpenBugs)]
/// public async Task ReproducesIssue123() { ... }
/// </code>
/// </summary>
public static class TestCategory
{
    /// <summary>
    /// Known-failing bug reproductions that document broken behavior.
    ///
    /// <para><b>Purpose:</b></para>
    /// <list type="bullet">
    ///   <item>Tests that FAIL because they document a real bug in NumSharp</item>
    ///   <item>Expected values are CORRECT (based on NumPy behavior)</item>
    ///   <item>NumSharp produces WRONG results, hence the test fails</item>
    /// </list>
    ///
    /// <para><b>CI Behavior:</b></para>
    /// <para>
    /// Excluded from CI runs via <c>--treenode-filter "/*/*/*/*[Category!=OpenBugs]"</c>.
    /// This prevents known bugs from blocking PRs while keeping them documented.
    /// </para>
    ///
    /// <para><b>Lifecycle:</b></para>
    /// <list type="number">
    ///   <item>Add <c>[OpenBugs]</c> when creating a failing bug reproduction</item>
    ///   <item>Fix the bug in NumSharp code</item>
    ///   <item>Test starts passing</item>
    ///   <item>REMOVE the <c>[OpenBugs]</c> attribute</item>
    ///   <item>Move test to appropriate permanent test class if needed</item>
    /// </list>
    ///
    /// <para><b>Files:</b></para>
    /// <list type="bullet">
    ///   <item><c>OpenBugs.cs</c> - General bug reproductions</item>
    ///   <item><c>OpenBugs.Bitmap.cs</c> - Bitmap/image-related bugs</item>
    ///   <item><c>OpenBugs.ApiAudit.cs</c> - API audit failures</item>
    /// </list>
    /// </summary>
    public const string OpenBugs = "OpenBugs";

    /// <summary>
    /// Tests documenting behavioral differences between NumSharp and NumPy.
    ///
    /// <para><b>Purpose:</b></para>
    /// <list type="bullet">
    ///   <item>Documents intentional or unavoidable differences from NumPy</item>
    ///   <item>Tests PASS but behavior differs from NumPy 2.x</item>
    ///   <item>Useful for tracking alignment progress toward NumPy 2.x compatibility</item>
    /// </list>
    ///
    /// <para><b>Examples:</b></para>
    /// <list type="bullet">
    ///   <item>Slicing a broadcast array materializes data in NumSharp (NumPy keeps view)</item>
    ///   <item>Type promotion rules that differ from NumPy</item>
    ///   <item>Edge cases where NumSharp behavior is acceptable but not identical</item>
    /// </list>
    ///
    /// <para><b>CI Behavior:</b></para>
    /// <para>
    /// Runs in CI (NOT excluded). These tests pass - they just document differences.
    /// </para>
    ///
    /// <para><b>Usage:</b></para>
    /// <code>
    /// [Test]
    /// [Category(TestCategory.Misaligned)]
    /// public void BroadcastSlice_MaterializesInNumSharp()
    /// {
    ///     // Document: NumSharp copies on slice, NumPy keeps view
    ///     var c = broadcast["1:4, :"];
    ///     c.@base.Should().NotBeSameAs(original.Storage);  // NumSharp: materialized
    ///     // NumPy would have: c.base is original
    /// }
    /// </code>
    /// </summary>
    public const string Misaligned = "Misaligned";

    /// <summary>
    /// Tests requiring GDI+/System.Drawing.Common (Windows-only).
    ///
    /// <para><b>Purpose:</b></para>
    /// <list type="bullet">
    ///   <item>Bitmap creation and manipulation tests</item>
    ///   <item>Image format conversion tests</item>
    ///   <item>Any test using <c>System.Drawing.Bitmap</c></item>
    /// </list>
    ///
    /// <para><b>CI Behavior:</b></para>
    /// <para>
    /// Automatically excluded on non-Windows runners (Ubuntu, macOS) via CI workflow.
    /// The workflow computes the filter at runtime based on <c>RUNNER_OS</c>.
    /// </para>
    ///
    /// <para><b>Local Development:</b></para>
    /// <para>
    /// When running tests locally on non-Windows, use the filter:
    /// <c>dotnet test -- --treenode-filter "/*/*/*/*[Category!=WindowsOnly]"</c>
    /// </para>
    ///
    /// <para><b>Note:</b></para>
    /// <para>
    /// System.Drawing.Common is Windows-only starting from .NET 6.
    /// Cross-platform alternatives: ImageSharp, SkiaSharp.
    /// </para>
    /// </summary>
    public const string WindowsOnly = "WindowsOnly";
}

/// <summary>
/// Attribute for tests documenting known bugs that are expected to fail.
/// Shorthand for <c>[Category("OpenBugs")]</c>.
///
/// <para>See <see cref="TestCategory.OpenBugs"/> for full documentation.</para>
/// </summary>
/// <example>
/// <code>
/// // Basic usage
/// [Test]
/// [OpenBugs]
/// public async Task BroadcastArrayWriteThrows() { ... }
///
/// // With GitHub issue URL (clickable in IDE)
/// [Test]
/// [OpenBugs(IssueUrl = "https://github.com/SciSharp/NumSharp/issues/396")]
/// public async Task OddWidthBitmapCorruption() { ... }
/// </code>
/// </example>
public class OpenBugsAttribute : CategoryAttribute
{
    /// <summary>
    /// URL to the GitHub issue tracking this bug.
    /// Clickable in most IDEs (Rider, VS, VS Code).
    /// </summary>
    /// <example>
    /// <code>
    /// [OpenBugs(IssueUrl = "https://github.com/SciSharp/NumSharp/issues/396")]
    /// </code>
    /// </example>
    public string? IssueUrl { get; set; }

    public OpenBugsAttribute() : base(TestCategory.OpenBugs) { }
}

/// <summary>
/// Attribute for tests documenting behavioral differences between NumSharp and NumPy.
/// Shorthand for <c>[Category("Misaligned")]</c>.
///
/// <para>See <see cref="TestCategory.Misaligned"/> for full documentation.</para>
/// </summary>
/// <example>
/// <code>
/// [Test]
/// [Misaligned]
/// public void SlicingBroadcast_MaterializesData()
/// {
///     // Document: NumSharp materializes, NumPy keeps view
///     var slice = broadcast["1:3"];
///     slice.@base.Should().NotBeSameAs(original.Storage);
/// }
/// </code>
/// </example>
public class MisalignedAttribute : CategoryAttribute
{
    public MisalignedAttribute() : base(TestCategory.Misaligned) { }
}

/// <summary>
/// Attribute for tests requiring Windows (GDI+/System.Drawing.Common).
/// Shorthand for <c>[Category("WindowsOnly")]</c>.
///
/// <para>See <see cref="TestCategory.WindowsOnly"/> for full documentation.</para>
/// </summary>
public class WindowsOnlyAttribute : CategoryAttribute
{
    public WindowsOnlyAttribute() : base(TestCategory.WindowsOnly) { }
}
