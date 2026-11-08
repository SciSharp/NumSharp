using System.Linq;

namespace NumSharp.Tests.Issues
{
    /// <summary>
    ///     Backs the answer that closed GitHub #455, "NumSharp does not allow building with IL2CPP via Unity"
    ///     (https://github.com/SciSharp/NumSharp/issues/455) as not planned: NumSharp's kernels are <c>DynamicMethod</c>s
    ///     emitted at run time, which an ahead-of-time compiler such as IL2CPP cannot run, and the
    ///     <c>LAPACKProviderType</c> enum the report failed on no longer exists.
    /// </summary>
    /// <remarks>
    ///     The answer's third claim, that a <c>PublishAot</c> build gets warning NDW021, is held by
    ///     <c>NumSharp.Tests.Build.Analyzer</c>'s <c>Ndw021BuildTests</c>.
    /// </remarks>
    [TestClass]
    public class Issue455
    {
        /// <summary>The kernels are emitted at run time, which needs a JIT, in a .NET 8 / .NET 10 assembly.</summary>
        [TestMethod]
        public void Kernels_AreEmittedAtRunTime() => IssueAnswerSupport.AssertNet8Or10_WithRuntimeEmittedKernels();

        /// <summary>No type named <c>LAPACKProviderType</c> is left in NumSharp (the report's IL2CPP error named it).</summary>
        [TestMethod]
        public void LapackProviderType_NoLongerExists()
        {
            typeof(np).Assembly.GetTypes().Where(t => t.Name == "LAPACKProviderType").Should().BeEmpty();
        }
    }
}
