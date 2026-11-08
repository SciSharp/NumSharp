using System;
using System.IO;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Layering
{
    /// <summary>
    ///     Ratchet R1 of <c>docs/plans/np-layer-dtype-logic.md</c>: no facade file (<c>np.*.cs</c>, <c>NDArray.*.cs</c>
    ///     outside <c>Backends/</c>, <c>Matrixlib/NDMatrix*.cs</c>) gains a dtype test, and every move out of the facade
    ///     is recorded by lowering the allowlist <c>facade_dtype_allowlist.tsv</c>.
    /// </summary>
    /// <remarks>
    ///     <para><b>Why.</b> The plan moves every dtype decision (numeric or text) from the public np.* methods into the
    ///     operation that implements them (a <c>TensorEngine</c> member, an internal operation class, a DType slot). The
    ///     facade keeps the public signature, the conversions C# forces, the class statement and one call. Without a
    ///     gate, new functions would keep adding dtype branches to the facade while the plan removes them.</para>
    ///     <para><b>How.</b> <see cref="LayeringScanner.CountFacadeDtypeTests"/> counts, per file, the lines that test a
    ///     dtype, text and numeric apart. A count above the allowlist fails and quotes every counted line of the file; a
    ///     count below it fails too and asks for the allowlist to be tightened (rerun with
    ///     <c>NUMSHARP_WRITE_LAYERING_ALLOWLIST=1</c>, which records progress and refuses to record a rise). A legitimate
    ///     rise (a C#-only conversion, say) is pinned by hand with a reason in the allowlist's fourth column.</para>
    /// </remarks>
    [TestClass]
    public class FacadeDtypeRatchetTests
    {
        /// <summary>The allowlist's header, rewritten by the writer.</summary>
        private static readonly string[] Header =
        {
            "Ratchet R1 (docs/plans/np-layer-dtype-logic.md): dtype-test lines per facade file, text then numeric.",
            "Counts may only go down. A rise is new dtype logic in the facade: move it down a layer, or pin it here by hand",
            "with a reason in a fourth column. After a drop, rerun the test with NUMSHARP_WRITE_LAYERING_ALLOWLIST=1 and commit.",
            "path<TAB>text<TAB>numeric[<TAB># reason]",
        };

        /// <summary>
        ///     Every facade file's dtype-test counts equal its allowlist pin: none rose, none dropped without the
        ///     allowlist being tightened, no new file appeared.
        /// </summary>
        [TestMethod]
        [TestCategory("Layering")]
        public void NoFacadeFileGainsADtypeTest()
        {
            string core = LayeringScanner.FindCoreSourceDir();
            if (core == null)
            {
                Assert.Inconclusive("src/NumSharp.Core not found relative to the test sources: the ratchet needs a checkout.");
                return;
            }

            var found = LayeringScanner.CountFacadeDtypeTests(core, out var lines);

            // Non-vacuity on the real tree. The counts themselves may reach zero as the plan completes, so the checks are
            // on what the scan can see: the facade files, and the text backend's type names (the patterns' own logic is
            // covered by LayeringScannerTests over a synthetic tree).
            Assert.IsTrue(LayeringScanner.FacadeFiles(core).Count() > 300,
                "the facade scan found fewer than 300 files: path or naming regression?");
            Assert.IsTrue(LayeringScanner.TextBackendReference(core).IsMatch("NDStringUfuncs.IsText(a)"),
                "the text-backend type list lost NDStringUfuncs: did Backends/Strings move?");

            string allowlist = Path.Combine(LayeringScanner.LayeringSourceDir(), "facade_dtype_allowlist.tsv");
            var problems = LayeringScanner.Compare("R1", allowlist, Header, found, lines, "text", "numeric");
            if (problems.Count > 0)
                Assert.Fail($"{problems.Count - 1} facade file(s) differ from the R1 allowlist:\n  " +
                            string.Join("\n  ", problems));
            if (Environment.GetEnvironmentVariable(LayeringScanner.WriteAllowlistVariable) == "1")
                Assert.Inconclusive($"R1 allowlist written: {allowlist}. Commit it; this run verified nothing.");
        }
    }
}
