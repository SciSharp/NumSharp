using System.Linq;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Build.Analyzer
{
    /// <summary>
    ///     NDW017 — a disposable type never disposes an NDArray-holding member on any path from its
    ///     Dispose / Dispose(bool) / DisposeAsync / DisposeAsyncCore. The exact-match gate pins every
    ///     tagged member; the content facts pin the "forgotten" shapes and every recognised spelling of
    ///     "disposed on the Dispose path" — direct, conditional, cast, Dispose(bool), explicit interface,
    ///     same-type helpers (chains, accessors, delegates, local functions), hand-off to any method,
    ///     local copies, Interlocked.Exchange, using, foreach / for / ForEach / Parallel over
    ///     collections, dictionaries (values, pairs, deconstructed), tuples, carriers, records, structs,
    ///     ref structs, partial types, async patterns and NumSharp's <c>close()</c>.
    /// </summary>
    [TestClass]
    public class DisposePathTests
    {
        [TestMethod]
        public Task DisposePathScenarios_MatchTagsExactly() => AnalyzerTestHarness.AssertExactAsync("DisposePathScenarios.cs");

        [TestMethod]
        public async Task ForgottenMembers_AreFlagged()
        {
            var flagged = await FixtureFacts.FlaggedLineTexts("DisposePathScenarios.cs", "NDW017");
            FixtureFacts.AnyStartsWith(flagged, "private NDArray _forgotten;", "the one of two that Dispose skips");
            FixtureFacts.AnyStartsWith(flagged, "private NDArray _a;", "a member only nulled / logged / finalizer-disposed / reset-disposed");
            FixtureFacts.AnyStartsWith(flagged, "private readonly List<NDArray> _items", "a list only cleared or only read");
            FixtureFacts.AnyStartsWith(flagged, "private NDArray _own;", "a member the inherited base Dispose cannot see");
            FixtureFacts.AnyStartsWith(flagged, "private DisposableOwnerForPath _owner;", "a contagious member never disposed");
            FixtureFacts.AnyStartsWith(flagged, "private NonDisposableInner _inner;", "a member whose type is not yet disposable");
            FixtureFacts.AnyStartsWith(flagged, "public NDArray Value { get; set; }", "an auto-property never disposed");
            FixtureFacts.AnyStartsWith(flagged, "private NDArray _other;", "the sibling of a handled tuple");

            // Exactly the tagged count of each forgotten shape — no over-reporting of the clean twins.
            Assert.AreEqual(1, flagged.Count(t => t.StartsWith("private NDArray _forgotten;")));
            Assert.AreEqual(1, flagged.Count(t => t.StartsWith("private DisposableOwnerForPath _owner;")));
        }

        [TestMethod]
        public async Task EveryDisposeSpelling_IsClean()
        {
            var r = await AnalyzerTestHarness.RunFileAsync("DisposePathScenarios.cs");
            Assert.IsTrue(r.CompileErrors.IsEmpty, "fixture must compile");
            var lines = System.IO.File.ReadAllLines(AnalyzerTestHarness.FixturePath("DisposePathScenarios.cs"));

            // Every NDW017 must land inside the "MUST warn" half of the file (the divider line, not
            // the header comment that also mentions the phrase).
            int cleanStart = System.Array.FindIndex(lines, l => l.TrimStart().StartsWith("// ---") && l.Contains("MUST stay clean")) + 1;
            Assert.IsTrue(cleanStart > 0, "the fixture carries a MUST-stay-clean section");
            var inCleanHalf = r.Ndw.Where(d => d.Id == "NDW017" && AnalyzerTestHarness.LineOf(d) > cleanStart)
                               .Select(d => $"L{AnalyzerTestHarness.LineOf(d)}: {lines[AnalyzerTestHarness.LineOf(d) - 1].Trim()}")
                               .ToList();
            Assert.AreEqual(0, inCleanHalf.Count,
                "no member in the clean half may draw NDW017: " + string.Join(" | ", inCleanHalf));
        }

        [TestMethod]
        public async Task Message_NamesTheContract_TheMember_AndTheKind()
        {
            var r = await AnalyzerTestHarness.RunFileAsync("DisposePathScenarios.cs");
            var m = r.Ndw.Single(d => d.Id == "NDW017" && d.GetMessage().StartsWith("'ForgetsOneOfTwo'"));
            StringAssert.Contains(m.GetMessage(), "implements IDisposable", "the contract is named");
            StringAssert.Contains(m.GetMessage(), "'_forgotten' (a field holding NDArrays)", "the member and its kind are named");

            var p = r.Ndw.Single(d => d.Id == "NDW017" && d.GetMessage().StartsWith("'AutoPropertyForgotten'"));
            StringAssert.Contains(p.GetMessage(), "'Value' (a property holding NDArrays)", "a property is reported as a property");

            var a = r.Ndw.Single(d => d.Id == "NDW017" && d.GetMessage().StartsWith("'AsyncForgets'"));
            StringAssert.Contains(a.GetMessage(), "implements IAsyncDisposable", "the async contract is named");
        }

        [TestMethod]
        public async Task Message_HintsAtTheInheritedDispose_AndTheNonDisposableInner()
        {
            var r = await AnalyzerTestHarness.RunFileAsync("DisposePathScenarios.cs");
            var inherited = r.Ndw.Single(d => d.Id == "NDW017" && d.GetMessage().StartsWith("'InheritsDisposeWithoutOverride'"));
            StringAssert.Contains(inherited.GetMessage(), "inherited from 'DisposableBase'", "the hint names the base that owns Dispose");

            var blocked = r.Ndw.Single(d => d.Id == "NDW017" && d.GetMessage().StartsWith("'HoldsNonDisposableInner'"));
            StringAssert.Contains(blocked.GetMessage(), "make 'NonDisposableInner' disposable first", "the hint points one level down");

            var plain = r.Ndw.Single(d => d.Id == "NDW017" && d.GetMessage().StartsWith("'ForgetsOneOfTwo'"));
            Assert.IsFalse(plain.GetMessage().Contains("inherited from"), "a type with its own Dispose gets no inheritance hint");
        }

        [TestMethod]
        public async Task DisposePathFixture_DrawsNoLeakDiagnostics()
        {
            var r = await AnalyzerTestHarness.RunFileAsync("DisposePathScenarios.cs");
            Assert.AreEqual(0, r.CountOf("NDW012"), "no Dispose body in the fixture creates and drops an NDArray");
            Assert.AreEqual(1, r.CountOf("NDW016"), "exactly the deliberately non-disposable inner type draws NDW016");
        }
    }
}
