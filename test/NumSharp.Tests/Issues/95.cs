namespace NumSharp.Tests.Issues
{
    /// <summary>
    ///     Backs the answer that closed GitHub #95, "Extend the guidelines" (https://github.com/SciSharp/NumSharp/issues/95):
    ///     the guidelines live in CONTRIBUTING.md ("match NumPy exactly"), ARCHITECTURE.md and the site's Advanced usage section.
    /// </summary>
    [TestClass]
    public class Issue95
    {
        /// <summary>The three documents the answer links exist, and CONTRIBUTING.md states the "match NumPy exactly" rule.</summary>
        [TestMethod]
        public void Guidelines_LiveInTheRepository()
        {
            IssueAnswerSupport.ReadRepoFile("CONTRIBUTING.md").Should().Contain("Match NumPy Exactly");
            IssueAnswerSupport.ReadRepoFile("ARCHITECTURE.md").Should().NotBeNullOrWhiteSpace();
            IssueAnswerSupport.ReadRepoFile("docs/website-src/docs/advanced/index.md").Should().NotBeNullOrWhiteSpace();
        }
    }
}
