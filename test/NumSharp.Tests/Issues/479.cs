using System;

namespace NumSharp.Tests.Issues
{
    /// <summary>
    ///     Backs the answer that closed GitHub #479, "Lacking/Outdated Documentation"
    ///     (https://github.com/SciSharp/NumSharp/issues/479): NumSharp is maintained (0.80.0 or later is what runs here) and
    ///     the docs the answer links, guides and the Supported Features Dashboard, are in the site source.
    /// </summary>
    [TestClass]
    public class Issue479
    {
        /// <summary>The running NumSharp is 0.80.0 or later, and the linked dashboard and guides exist.</summary>
        [TestMethod]
        public void Maintained_AndDocumented()
        {
            typeof(np).Assembly.GetName().Version.Should().BeGreaterThanOrEqualTo(new Version(0, 80, 0));
            IssueAnswerSupport.ReadRepoFile("docs/website-src/docs/coverage-support-dashboard.md").Should().NotBeNullOrWhiteSpace();
            IssueAnswerSupport.ReadRepoFile("docs/website-src/docs/fundamentals/index.md").Should().NotBeNullOrWhiteSpace();
        }
    }
}
