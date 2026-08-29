using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Build.Analyzer
{
    /// <summary>Asserts NDW012 (the NDArray leak warning) fires on exactly the tagged fixture lines.</summary>
    [TestClass]
    public class LeakAnalyzerTests
    {
        // The authoritative gates: the analyzer's NDW output must match the `// [NDW012]` tags exactly
        // (a missing warning AND an unexpected one both fail).
        [TestMethod]
        public Task LeakScenarios_MatchTagsExactly() => AnalyzerTestHarness.AssertExactAsync("LeakScenarios.cs");

        [TestMethod]
        public Task CarrierScenarios_MatchTagsExactly() => AnalyzerTestHarness.AssertExactAsync("CarrierScenarios.cs");

        // ---- readable scenario pins (content-based, robust to line shifts) ----

        [TestMethod]
        public async Task LeakingScenarios_AreFlagged()
        {
            var flagged = await FlaggedLineTexts("LeakScenarios.cs");
            AssertAnyStartsWith(flagged, "var t = a + b;", "dead local");
            AssertAnyStartsWith(flagged, "return np.sum(a * b);", "temp handed to np.sum");
            AssertAnyStartsWith(flagged, "np.add(a, b);", "discarded result");
            AssertAnyStartsWith(flagged, "var t = a + 1.0;", "local fed into np.sum");
            AssertAnyStartsWith(flagged, "var t = new NDArray(", "constructed and dropped");
        }

        [TestMethod]
        public async Task CleanScenarios_AreNotFlagged()
        {
            var flagged = await FlaggedLineTexts("LeakScenarios.cs");
            AssertNoneContains(flagged, "return a + b;", "a returned value");
            AssertNoneContains(flagged, "using var t = a + 1.0;", "a disposed value");
            AssertNoneContains(flagged, "ConsumingSink(t);", "a value handed to a foreign API");
            AssertNoneContains(flagged, "r = a + 1.0;", "an out-parameter assignment");
            AssertNoneContains(flagged, "_field = a + 1.0;", "a field store");
            AssertNoneContains(flagged, "var t = a;", "an alias of the input");
            AssertNoneContains(flagged, ".reshape(1, -1)", "a fluent view chain");
            AssertNoneContains(flagged, "NDScope.Attach(t);", "a hand-scoped method");
        }

        [TestMethod]
        public async Task TupleArrayAndCarrierDrops_AreFlagged()
        {
            var flagged = await FlaggedLineTexts("CarrierScenarios.cs");
            AssertAnyStartsWith(flagged, "var t = SplitTuple(a);", "dropped tuple");
            AssertAnyStartsWith(flagged, "var (q, r) = SplitTuple(a);", "deconstruction with an unused side");
            AssertAnyStartsWith(flagged, "var g = SplitArray(a);", "dropped NDArray[]");
            AssertAnyStartsWith(flagged, "var p = MakePair(a);", "dropped INDArrayCarrier");
            // literal carriers own what their elements own — dropping one drops the produced temps
            AssertAnyStartsWith(flagged, "var t = (a + 1.0, b - 1.0);", "dropped tuple LITERAL of temps");
            AssertAnyStartsWith(flagged, "var t = (a + 1.0, b);", "dropped MIXED tuple literal (its temp leaks)");
            AssertAnyStartsWith(flagged, "var t = ((a + 1.0, b - 1.0), 5);", "dropped NESTED tuple literal");
            AssertAnyStartsWith(flagged, "var g = new[] { a + 1.0 };", "dropped array LITERAL of a temp");
            AssertAnyStartsWith(flagged, "var (q, r) = (a + 1.0, b - 1.0);", "deconstructed literal with an unused side");
        }

        [TestMethod]
        public async Task AliasCarriersAndEscapingLiterals_AreNotFlagged()
        {
            var flagged = await FlaggedLineTexts("CarrierScenarios.cs");
            // R2: a tuple/array of plain inputs owns nothing
            AssertNoneContains(flagged, "var t = (a, b);", "a tuple of input aliases");
            AssertNoneContains(flagged, "var g = new[] { a, b };", "an array of input aliases");
            AssertNoneContains(flagged, "new NDArray[5]", "an empty array allocation");
            // escaping literal forms
            AssertNoneContains(flagged, "Sink(new[] { a + 1.0 });", "an array literal handed to a sink");
            AssertNoneContains(flagged, "_cache = g;", "an array literal stored to a field");
            AssertNoneContains(flagged, "return t;", "a tuple literal returned via a local");
        }

        private static async Task<List<string>> FlaggedLineTexts(string fileName)
        {
            var result = await AnalyzerTestHarness.RunFileAsync(fileName);
            Assert.IsTrue(result.CompileErrors.IsEmpty, $"fixture '{fileName}' must compile");
            var lines = File.ReadAllLines(AnalyzerTestHarness.FixturePath(fileName));
            return result.Ndw
                .Where(d => d.Id == "NDW012")
                .Select(d => lines[AnalyzerTestHarness.LineOf(d) - 1].Trim())
                .ToList();
        }

        private static void AssertAnyStartsWith(List<string> flagged, string prefix, string what)
            => Assert.IsTrue(flagged.Any(t => t.StartsWith(prefix, System.StringComparison.Ordinal)),
                             $"expected NDW012 on {what} ('{prefix}'); flagged lines: [{string.Join(" | ", flagged)}]");

        private static void AssertNoneContains(List<string> flagged, string needle, string what)
            => Assert.IsFalse(flagged.Any(t => t.Contains(needle)),
                              $"{what} ('{needle}') must NOT draw NDW012; flagged lines: [{string.Join(" | ", flagged)}]");
    }
}
