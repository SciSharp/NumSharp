namespace NumSharp.Tests.Issues
{
    /// <summary>
    ///     Backs the answer that closed GitHub #70, "Let more people know about NumSharp"
    ///     (https://github.com/SciSharp/NumSharp/issues/70): the DocFX site's guides the answer links exist in the repository
    ///     the site is built from.
    /// </summary>
    [TestClass]
    public class Issue70
    {
        /// <summary>
        ///     The pages the answer links (The absolute basics, Quickstart, Fundamentals, Advanced usage and the Supported
        ///     Features Dashboard) are present in <c>docs/website-src</c>, each with a title.
        /// </summary>
        [TestMethod]
        public void LinkedGuides_ExistInTheSiteSource()
        {
            foreach (var page in new[]
                     {
                         "docs/website-src/docs/absolute-basics.md", "docs/website-src/docs/quickstart.md",
                         "docs/website-src/docs/fundamentals/index.md", "docs/website-src/docs/advanced/index.md",
                         "docs/website-src/docs/coverage-support-dashboard.md",
                     })
                IssueAnswerSupport.ReadRepoFile(page).Should().NotBeNullOrWhiteSpace(page);
        }
    }
}
