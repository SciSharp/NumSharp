namespace NumSharp.Tests.Issues
{
    /// <summary>
    ///     Backs the answer that closed GitHub #430, "NumSharp.Backends.Unmanaged.UnmanagedMemoryBlock`1 fails on Mono on
    ///     Linux" (https://github.com/SciSharp/NumSharp/issues/430) as not planned: NumSharp 0.80.0 targets .NET 8 and .NET 10
    ///     only, so Mono is not a supported runtime.
    /// </summary>
    [TestClass]
    public class Issue430
    {
        /// <summary>The assembly targets .NET 8 or .NET 10 (no Mono / netstandard build), with run-time-emitted kernels.</summary>
        [TestMethod]
        public void Net8Or10Only() => IssueAnswerSupport.AssertNet8Or10_WithRuntimeEmittedKernels();
    }
}
