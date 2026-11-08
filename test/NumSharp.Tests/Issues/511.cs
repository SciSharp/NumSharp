namespace NumSharp.Tests.Issues
{
    /// <summary>
    ///     Backs the answer that closed GitHub #511, "Does Numsharp work in Unity with the IL2CPP backend?"
    ///     (https://github.com/SciSharp/NumSharp/issues/511) as not planned: no - NumSharp generates its kernels at run time
    ///     with <c>DynamicMethod</c>, which IL2CPP cannot run, and it ships only net8.0 / net10.0 assemblies.
    /// </summary>
    [TestClass]
    public class Issue511
    {
        /// <summary>The assembly targets .NET 8 or .NET 10 and its kernels are DynamicMethods emitted at run time.</summary>
        [TestMethod]
        public void Net8Or10Only_WithRuntimeEmittedKernels() => IssueAnswerSupport.AssertNet8Or10_WithRuntimeEmittedKernels();
    }
}
