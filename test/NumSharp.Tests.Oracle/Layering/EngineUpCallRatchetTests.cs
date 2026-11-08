using System;
using System.IO;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Layering
{
    /// <summary>
    ///     Ratchet R2 of <c>docs/plans/np-layer-dtype-logic.md</c>: no engine file (every file under <c>Backends/</c>)
    ///     gains a call into the np.* facade, and every call the plan removes is recorded by lowering the allowlist
    ///     <c>engine_upcall_allowlist.tsv</c>.
    /// </summary>
    /// <remarks>
    ///     <para><b>Why.</b> Code that the plan moves down a layer is only deeper if it stops calling back up: an engine
    ///     routine that calls <c>np.sum</c> runs the facade again (its class-boundary test, its conversions, its error
    ///     texts naming another function). The plan's phase P6 removes the computations among these calls and moves the
    ///     structural ones (allocation, <c>copyto</c>, <c>broadcast_to</c>, the buffer size) to internal helpers.</para>
    ///     <para><b>How.</b> <see cref="LayeringScanner.CountEngineUpCalls"/> counts the np.* calls per file, computations
    ///     and structural calls apart, comments and literals excluded. The comparison rules are R1's
    ///     (<see cref="LayeringScanner.Compare"/>): a rise fails and quotes the file's calls, a drop fails until the
    ///     allowlist is tightened with <c>NUMSHARP_WRITE_LAYERING_ALLOWLIST=1</c>.</para>
    /// </remarks>
    [TestClass]
    public class EngineUpCallRatchetTests
    {
        /// <summary>The allowlist's header, rewritten by the writer.</summary>
        private static readonly string[] Header =
        {
            "Ratchet R2 (docs/plans/np-layer-dtype-logic.md): np.* calls per engine file (Backends/**), computations then",
            "structural calls. Counts may only go down. A rise is a new call from the engine up into the facade: call the",
            "engine member or the family core instead, or pin it here by hand with a reason in a fourth column. After a drop,",
            "rerun the test with NUMSHARP_WRITE_LAYERING_ALLOWLIST=1 and commit.",
            "path<TAB>computations<TAB>structural[<TAB># reason]",
        };

        /// <summary>
        ///     Every engine file's np.* call counts equal its allowlist pin: none rose, none dropped without the allowlist
        ///     being tightened, no new file appeared.
        /// </summary>
        [TestMethod]
        [TestCategory("Layering")]
        public void NoEngineFileGainsAnNpCall()
        {
            string core = LayeringScanner.FindCoreSourceDir();
            if (core == null)
            {
                Assert.Inconclusive("src/NumSharp.Core not found relative to the test sources: the ratchet needs a checkout.");
                return;
            }

            var found = LayeringScanner.CountEngineUpCalls(core, out var calls);

            // Non-vacuity on the real tree: the scan must see the engine files. The counts may reach zero as P6
            // completes; the call pattern itself is covered by LayeringScannerTests over a synthetic tree.
            Assert.IsTrue(LayeringScanner.EngineFiles(core).Count() > 100,
                "the engine scan found fewer than 100 files under Backends/: path regression?");

            string allowlist = Path.Combine(LayeringScanner.LayeringSourceDir(), "engine_upcall_allowlist.tsv");
            var problems = LayeringScanner.Compare("R2", allowlist, Header, found, calls, "computations", "structural");
            if (problems.Count > 0)
                Assert.Fail($"{problems.Count - 1} engine file(s) differ from the R2 allowlist:\n  " +
                            string.Join("\n  ", problems));
            if (Environment.GetEnvironmentVariable(LayeringScanner.WriteAllowlistVariable) == "1")
                Assert.Inconclusive($"R2 allowlist written: {allowlist}. Commit it; this run verified nothing.");
        }
    }
}
