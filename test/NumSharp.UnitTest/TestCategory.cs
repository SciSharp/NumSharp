using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest;

/// <summary>
/// Defines test categories used for filtering tests in CI and local development.
///
/// <para><b>How Categories Work in CI:</b></para>
/// <para>
/// The CI pipeline (<c>.github/workflows/build-and-release.yml</c>) uses MSTest's
/// <c>--filter</c> to exclude certain categories from test runs:
/// </para>
/// <code>
/// --filter "TestCategory!=OpenBugs"
/// </code>
/// <para>
/// This filter excludes all tests marked with <c>[TestCategory("OpenBugs")]</c> from CI runs.
/// Tests in other categories (like <see cref="Misaligned"/>) still run and can fail CI.
/// </para>
///
/// <para><b>Local Development Filtering:</b></para>
/// <code>
/// # Exclude OpenBugs (same as CI)
/// dotnet test --filter "TestCategory!=OpenBugs"
///
/// # Run ONLY OpenBugs tests (to verify fixes)
/// dotnet test --filter "TestCategory=OpenBugs"
///
/// # Run ONLY Misaligned tests
/// dotnet test --filter "TestCategory=Misaligned"
/// </code>
///
/// <para><b>Usage:</b></para>
/// <para>Apply at class or method level:</para>
/// <code>
/// [TestMethod]
/// [OpenBugs]  // or [TestCategory(TestCategory.OpenBugs)]
/// public void ReproducesIssue123() { ... }
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
    /// Excluded from CI runs via <c>--filter "TestCategory!=OpenBugs"</c>.
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
    /// [TestMethod]
    /// [TestCategory(TestCategory.Misaligned)]
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
    /// <c>dotnet test --filter "TestCategory!=WindowsOnly"</c>
    /// </para>
    ///
    /// <para><b>Note:</b></para>
    /// <para>
    /// System.Drawing.Common is Windows-only starting from .NET 6.
    /// Cross-platform alternatives: ImageSharp, SkiaSharp.
    /// </para>
    /// </summary>
    public const string WindowsOnly = "WindowsOnly";

    /// <summary>
    /// Tests for long indexing support (arrays with size > int.MaxValue).
    ///
    /// <para><b>Purpose:</b></para>
    /// <list type="bullet">
    ///   <item>Tests that verify operations work correctly with > 2 billion elements</item>
    ///   <item>Most use broadcast arrays or small allocations (run in CI)</item>
    ///   <item>High-memory tests should also use <see cref="HighMemory"/></item>
    /// </list>
    ///
    /// <para><b>CI Behavior:</b></para>
    /// <para>
    /// Runs in CI. Only tests also marked with <see cref="HighMemory"/> are excluded.
    /// </para>
    /// </summary>
    public const string LongIndexing = "LongIndexing";

    /// <summary>
    /// Tests requiring high memory allocation (8GB+ RAM).
    ///
    /// <para><b>Purpose:</b></para>
    /// <list type="bullet">
    ///   <item>Tests that allocate > 2GB arrays (e.g., int.MaxValue * 1.1 byte arrays)</item>
    ///   <item>Requires 8GB+ RAM to run without OutOfMemoryException</item>
    ///   <item>Too resource-intensive for standard CI runners</item>
    /// </list>
    ///
    /// <para><b>CI Behavior:</b></para>
    /// <para>
    /// Excluded from CI runs via <c>--filter "TestCategory!=HighMemory"</c>.
    /// </para>
    ///
    /// <para><b>Local Development:</b></para>
    /// <code>
    /// # Run ONLY HighMemory tests (requires 8GB+ RAM)
    /// dotnet test --filter "TestCategory=HighMemory"
    /// </code>
    /// </summary>
    public const string HighMemory = "HighMemory";
}

/// <summary>
/// Attribute for tests documenting known bugs that are expected to fail.
/// Shorthand for <c>[TestCategory("OpenBugs")]</c>.
///
/// <para>See <see cref="TestCategory.OpenBugs"/> for full documentation.</para>
/// </summary>
/// <example>
/// <code>
/// // Basic usage
/// [TestMethod]
/// [OpenBugs]
/// public void BroadcastArrayWriteThrows() { ... }
///
/// // With GitHub issue URL (clickable in IDE)
/// [TestMethod]
/// [OpenBugs(IssueUrl = "https://github.com/SciSharp/NumSharp/issues/396")]
/// public void OddWidthBitmapCorruption() { ... }
/// </code>
/// </example>
public class OpenBugsAttribute : TestCategoryBaseAttribute
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

    public override IList<string> TestCategories => [TestCategory.OpenBugs];
}

/// <summary>
/// Attribute for tests documenting behavioral differences between NumSharp and NumPy.
/// Shorthand for <c>[TestCategory("Misaligned")]</c>.
///
/// <para>See <see cref="TestCategory.Misaligned"/> for full documentation.</para>
/// </summary>
/// <example>
/// <code>
/// [TestMethod]
/// [Misaligned]
/// public void SlicingBroadcast_MaterializesData()
/// {
///     // Document: NumSharp materializes, NumPy keeps view
///     var slice = broadcast["1:3"];
///     slice.@base.Should().NotBeSameAs(original.Storage);
/// }
/// </code>
/// </example>
public class MisalignedAttribute : TestCategoryBaseAttribute
{
    public override IList<string> TestCategories => [TestCategory.Misaligned];
}

/// <summary>
/// Attribute for tests requiring Windows (GDI+/System.Drawing.Common).
/// Shorthand for <c>[TestCategory("WindowsOnly")]</c>.
///
/// <para>See <see cref="TestCategory.WindowsOnly"/> for full documentation.</para>
/// </summary>
public class WindowsOnlyAttribute : TestCategoryBaseAttribute
{
    public override IList<string> TestCategories => [TestCategory.WindowsOnly];
}

/// <summary>
/// Attribute for tests verifying long indexing (> int.MaxValue elements).
/// Shorthand for <c>[TestCategory("LongIndexing")]</c>.
///
/// <para>See <see cref="TestCategory.LongIndexing"/> for full documentation.</para>
/// </summary>
public class LongIndexingAttribute : TestCategoryBaseAttribute
{
    public override IList<string> TestCategories => [TestCategory.LongIndexing];
}

/// <summary>
/// Attribute for tests requiring high memory (8GB+ RAM).
/// Shorthand for <c>[TestCategory("HighMemory")]</c>.
///
/// <para>See <see cref="TestCategory.HighMemory"/> for full documentation.</para>
/// </summary>
/// <example>
/// <code>
/// [TestMethod]
/// [HighMemory]
/// public void LargeArraySum()
/// {
///     // Allocates ~2.4GB
///     using var arr = np.ones&lt;byte&gt;((long)(int.MaxValue * 1.1));
///     var sum = np.sum(arr);
/// }
/// </code>
/// </example>
public class HighMemoryAttribute : TestCategoryBaseAttribute
{
    public override IList<string> TestCategories => [TestCategory.HighMemory];
}

/// <summary>
/// Attribute for tests that allocate large amounts of memory and crash CI runners.
/// Combines OpenBugs category so tests are automatically excluded from CI.
///
/// <para>Use this instead of [OpenBugs] for memory-intensive tests that aren't actually bugs,
/// just too heavy for CI runners.</para>
/// </summary>
/// <example>
/// <code>
/// [TestMethod]
/// [LargeMemoryTest]  // Auto-excluded from CI
/// public void Allocate_4GB()
/// {
///     var arr = np.ones&lt;int&gt;((4L * 1024 * 1024 * 1024 / 4));  // 4GB
/// }
/// </code>
/// </example>
public class LargeMemoryTestAttribute : TestCategoryBaseAttribute
{
    public override IList<string> TestCategories => [TestCategory.OpenBugs, TestCategory.HighMemory];
}
