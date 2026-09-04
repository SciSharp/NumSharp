using System.Linq;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Build.Analyzer
{
    /// <summary>
    ///     NDW012 under CONTAGION: an instance of an NDArray-owning disposable (a consumer's holder
    ///     class, NumSharp's own <c>np.nditer</c> iterator) is an owned value the method must reclaim,
    ///     and a <c>foreach</c> over a produced NDArray or such an instance never disposes it. The
    ///     exact-match gate pins every line; the facts pin each drop and each reclaim / escape / exemption.
    /// </summary>
    [TestClass]
    public class ContagionTests
    {
        [TestMethod]
        public Task ContagionScenarios_MatchTagsExactly() => AnalyzerTestHarness.AssertExactAsync("ContagionScenarios.cs");

        [TestMethod]
        public async Task DroppedOwningInstances_AreFlagged()
        {
            var flagged = await FixtureFacts.FlaggedLineTexts("ContagionScenarios.cs");
            FixtureFacts.AnyStartsWith(flagged, "new Batch(a + b);", "a dropped holder construction");
            FixtureFacts.AnyStartsWith(flagged, "var batch = new Batch(a + b);", "a dead holder local");
            FixtureFacts.AnyStartsWith(flagged, "var batch = MakeBatch(a);", "a dropped factory result");
            FixtureFacts.AnyStartsWith(flagged, "_ = new Batch(a + b);", "an explicitly discarded holder");
            FixtureFacts.AnyStartsWith(flagged, "var batches = new[] { new Batch(a + 1.0) };", "a dropped array of holders");
            FixtureFacts.AnyStartsWith(flagged, "var it = np.nditer(a);", "a never-closed np.nditer");
            FixtureFacts.AnyStartsWith(flagged, "foreach (var x in np.nditer(a))", "an np.nditer walked by foreach");
            FixtureFacts.AnyStartsWith(flagged, "foreach (var row in a + b)", "a produced NDArray walked by foreach");
            FixtureFacts.AnyStartsWith(flagged, "using var it = np.nditer(a + b);", "a temp handed to np.nditer");
        }

        [TestMethod]
        public async Task ReclaimedEscapedAndExemptInstances_AreNotFlagged()
        {
            var flagged = await FixtureFacts.FlaggedLineTexts("ContagionScenarios.cs");
            FixtureFacts.NoneContains(flagged, "using var batch = new Batch(a + b);", "a using-declared holder");
            FixtureFacts.NoneContains(flagged, "batch.Dispose();", "an explicitly disposed holder");
            FixtureFacts.NoneContains(flagged, "return new Batch(a + b);", "a returned holder");
            FixtureFacts.NoneContains(flagged, "_kept = new Batch(a + b);", "a stored holder");
            FixtureFacts.NoneContains(flagged, "Sink(new Batch(a + b));", "a holder handed to a sink");
            FixtureFacts.NoneContains(flagged, "using (var batch = new Batch(a + b))", "a holder in a using statement");
            FixtureFacts.NoneContains(flagged, "it.close();", "an iterator closed with NumSharp's close()");
            FixtureFacts.NoneContains(flagged, "foreach (var x in it)", "a using-held iterator walked by foreach");
            FixtureFacts.NoneContains(flagged, "new BorrowingCursor(a)", "an [NDBorrowed] disposable");
            FixtureFacts.NoneContains(flagged, "new PlainWrapper(a)", "a non-disposable wrapper (nothing to reclaim)");
            FixtureFacts.NoneContains(flagged, "foreach (var x in SplitArray(a))", "a produced NDArray[] walked by foreach");
            FixtureFacts.NoneContains(flagged, "foreach (var row in a)", "an input alias walked by foreach");
            // the [NDScoped] twin of the dead-holder local
            Assert.AreEqual(1, flagged.Count(t => t.StartsWith("var batch = new Batch(a + b);")),
                "only the UN-scoped dead holder local warns; the [NDScoped] twin is exempt");
        }

        [TestMethod]
        public async Task Message_DescribesTheOwningInstance()
        {
            var r = await AnalyzerTestHarness.RunFileAsync("ContagionScenarios.cs");
            var lines = System.IO.File.ReadAllLines(AnalyzerTestHarness.FixturePath("ContagionScenarios.cs"));
            var holder = r.Ndw.Single(d => d.Id == "NDW012" && lines[AnalyzerTestHarness.LineOf(d) - 1].Contains("new Batch(a + b);") && lines[AnalyzerTestHarness.LineOf(d) - 1].TrimStart().StartsWith("new Batch"));
            StringAssert.Contains(holder.GetMessage(), "'Batch' instance (a disposable that stores NDArrays)", "a holder instance is described as such");

            var iterator = r.Ndw.Single(d => d.Id == "NDW012" && lines[AnalyzerTestHarness.LineOf(d) - 1].Contains("var it = np.nditer(a);"));
            StringAssert.Contains(iterator.GetMessage(), "'NDIterator' instance", "NumSharp's iterator is described by its type");

            var array = r.Ndw.Single(d => d.Id == "NDW012" && lines[AnalyzerTestHarness.LineOf(d) - 1].Contains("var batches ="));
            StringAssert.Contains(array.GetMessage(), "'Batch' array", "an array of holders is described as such");

            var temp = r.Ndw.Single(d => d.Id == "NDW012" && lines[AnalyzerTestHarness.LineOf(d) - 1].Contains("using var it = np.nditer(a + b);"));
            StringAssert.Contains(temp.GetMessage(), "This NDArray is never", "the a+b temp is still described as an NDArray");
        }

        [TestMethod]
        public async Task ContagionFixture_TypeDiagnostics_AreExactlyTheWrapper()
        {
            var r = await AnalyzerTestHarness.RunFileAsync("ContagionScenarios.cs");
            Assert.AreEqual(1, r.CountOf("NDW016"), "only the deliberately non-disposable PlainWrapper draws NDW016");
            Assert.AreEqual(0, r.CountOf("NDW017"), "Batch disposes what it holds; BorrowingCursor is [NDBorrowed]");
        }
    }
}
