namespace NumSharp.Tests.Issues
{
    /// <summary>
    ///     Backs the answer that closed GitHub #424, "The type or namespace name 'NumSharp' could not be found ...
    ///     [Assembly-CSharp]" (https://github.com/SciSharp/NumSharp/issues/424, a Unity project) as stale: NumSharp 0.80.0
    ///     ships only net8.0 / net10.0 assemblies and emits its kernels with <c>System.Reflection.Emit</c> at run time, so it
    ///     needs a .NET 8+ runtime with a JIT.
    /// </summary>
    [TestClass]
    public class Issue424
    {
        /// <summary>The assembly targets .NET 8 or .NET 10, and its kernels are DynamicMethods emitted at run time.</summary>
        [TestMethod]
        public void Net8Or10Only_WithRuntimeEmittedKernels() => IssueAnswerSupport.AssertNet8Or10_WithRuntimeEmittedKernels();
    }
}
