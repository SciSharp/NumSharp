namespace NumSharp.Tests.Issues
{
    /// <summary>
    ///     Backs the answer that closed GitHub #498, "Is there an example on how to use it with IronPython?"
    ///     (https://github.com/SciSharp/NumSharp/issues/498): since 0.80.0 NumSharp targets .NET 8 / .NET 10 only, with no
    ///     .NET Framework build, so it cannot stand in for numpy inside IronPython on .NET Framework 4.x.
    /// </summary>
    /// <remarks>
    ///     The answer's other half - real CPython with numpy from C#, sharing memory with an <c>NDArray</c> through
    ///     NumSharp.Interop.pythonnet - is backed live in NumSharp.Tests.Interop (<c>ClosedInteropIssuesTests.Issue498_*</c>).
    /// </remarks>
    [TestClass]
    public class Issue498
    {
        /// <summary>The assembly targets .NET 8 or .NET 10 (no .NET Framework build).</summary>
        [TestMethod]
        public void NoDotNetFrameworkBuild() => IssueAnswerSupport.AssertNet8Or10_WithRuntimeEmittedKernels();
    }
}
