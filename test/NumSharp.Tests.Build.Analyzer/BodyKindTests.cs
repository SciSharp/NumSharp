using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Build.Analyzer
{
    /// <summary>
    ///     Every METHOD-BODY KIND is analyzed: async methods, iterators, instance/static constructors,
    ///     finalizers, expression-bodied members, unscoped property getters, and user-defined operators
    ///     all draw NDW012 on a leak — while a lambda RETURNING its temp is a legitimate escape.
    /// </summary>
    [TestClass]
    public class BodyKindTests
    {
        [TestMethod]
        public Task BodyKindScenarios_MatchTagsExactly()
            => AnalyzerTestHarness.AssertExactAsync("BodyKindScenarios.cs");

        [TestMethod]
        public async Task EveryBodyKind_IsAnalyzed()
        {
            var flagged = await FixtureFacts.FlaggedLineTexts("BodyKindScenarios.cs");
            FixtureFacts.AnyStartsWith(flagged, "public BodyKindHost(", "an instance constructor");
            FixtureFacts.AnyStartsWith(flagged, "static BodyKindHost()", "a static constructor");
            FixtureFacts.AnyStartsWith(flagged, "~BodyKindHost()", "a finalizer");
            FixtureFacts.AnyStartsWith(flagged, "var t = _x + 1.0;", "a user-defined operator body");
            FixtureFacts.AnyStartsWith(flagged, "var t = a + b;", "an async / iterator body");
            FixtureFacts.AnyStartsWith(flagged, "public static void ExprBodiedDiscard", "an expression-bodied discard");
            FixtureFacts.AnyStartsWith(flagged, "public static NDArray LeakyGetter", "an unscoped leaky getter");
        }

        [TestMethod]
        public async Task LambdaReturningTemp_IsClean()
        {
            var flagged = await FixtureFacts.FlaggedLineTexts("BodyKindScenarios.cs");
            FixtureFacts.NoneContains(flagged, "Func<NDArray> f = () => a + b;", "a lambda returning its temp");
            FixtureFacts.NoneContains(flagged, "return f();", "invoking that lambda");
        }
    }
}
