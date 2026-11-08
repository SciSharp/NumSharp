using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     The relayout tiers (<c>test/oracle/gen_relayout_oracle.py</c>): existing cases re-emitted with one operand moved
    ///     into an ndarray KIND its (op, operand position, dtype class) cell had never seen — a C-contiguous, F, permuted or
    ///     stepped view at an offset, an F or permuted view, a reversed or stepped-and-reversed view, a read-only owner, a 0-d
    ///     view at an offset, and for the elementwise and reduction families a broadcast operand (also at an offset), a
    ///     one-element 1-D array, a 0-d owner, an empty array and an empty view inside a real buffer — with NumPy 2.4.2's
    ///     answer recomputed for the new operands.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The variants keep their source's op key, params and valueclass, so they replay through the same
    ///     <see cref="OpRegistry"/> entry under the same <see cref="MisalignedRegistry"/> rules as the tier they came from;
    ///     a source qualifies only when a plain NumPy call reproduces its recorded result byte for byte (the generator's
    ///     self-validating dispatcher), so no variant guesses a call. The view buffers hold the sentinel 7 wherever the view
    ///     does not reach, so a kernel that reads outside the view shows a value the operand never holds.
    ///     </para>
    ///     <para>
    ///     Inheriting the source tier's excuses is also a hole: an unbounded excuse ("[known bug]" value divergences of a
    ///     whole op family) swallows a layout misread as readily as the divergence it was written for. The
    ///     <see cref="RelayoutLayoutInvariance"/> contract closes it without any excuse table: where NumPy's answer does not
    ///     depend on the moved operand's layout, NumSharp's must not either.
    ///     </para>
    ///     <para>
    ///     <c>relayout_host.jsonl</c> holds the variants of the host-libm tiers and replays through
    ///     <see cref="RunHostLibmCorpus"/> like them. Known divergences are carved into <c>gaps/relayout*_gaps.jsonl</c> and
    ///     replayed by the <c>[TestCategory("OpenBugs")]</c> tests below.
    ///     </para>
    /// </remarks>
    public partial class FuzzCorpusTests
    {
        /// <summary>The portable relayout variants: every case bit-exact with NumPy or documented by its tier's rules.</summary>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void Relayout() => RunCorpus("relayout.jsonl");

        /// <summary>The relayout variants of the host-libm tiers (hard-gated on win-amd64, inconclusive elsewhere).</summary>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void RelayoutHost() => RunHostLibmCorpus("relayout_host.jsonl");

        /// <summary>
        ///     The layout-invariance contract over both relayout tiers: for every variant whose NumPy outcome is the same
        ///     with the moved operand as a fresh C-ordered copy (no <c>layout_sensitive</c> mark), NumSharp's outcome on the
        ///     variant must equal its own outcome on that copy — same dtype, shape and bytes (NaN tokenized), or a throw on
        ///     both sides. No <see cref="MisalignedRegistry"/> excuse applies, so a layout misread cannot hide behind a
        ///     divergence NumSharp also has on C-ordered input.
        /// </summary>
        /// <exception cref="AssertFailedException">A variant's answer changes with the layout where NumPy's does not.</exception>
        /// <remarks>
        ///     Host-independent: both sides run on this host's libm, so a transcendental that differs from NumPy by a ULP
        ///     differs the same way on both layouts and is not reported. Known layout bugs are carved out of the tiers
        ///     (<c>gaps/relayout*_gaps.jsonl</c>) and checked by the <c>*_KnownGaps</c> tests with this same verdict.
        /// </remarks>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void RelayoutLayoutInvariance()
        {
            var failures = new List<string>();
            int compared = 0, sensitive = 0;
            foreach (var file in new[] { "relayout.jsonl", "relayout_host.jsonl" })
            {
                using var cases = FuzzCorpus.Open(file);
                foreach (var c in cases)
                {
                    if (c.Layout_Sensitive)
                    {
                        sensitive++;
                        continue;
                    }
                    compared++;
                    string verdict = LayoutInvarianceVerdict(c);
                    if (verdict != null)
                        failures.Add(verdict);
                }
            }
            DumpFailures("relayout-invariance", failures);
            Console.WriteLine($"[relayout invariance] {compared} variants compared with their C-ordered copy, " +
                              $"{sensitive} skipped (NumPy's own answer follows the layout), {failures.Count} layout-caused");
            Assert.IsTrue(compared >= MinLayoutInvarianceCases,
                $"only {compared} relayout variants compared (floor {MinLayoutInvarianceCases}): the corpus shrank or failed to copy");
            Assert.AreEqual(0, failures.Count,
                $"{failures.Count} relayout variants answer differently on the layout than on a C-ordered copy of the same " +
                "values, where NumPy does not:\n" + string.Join("\n", failures.Take(60)));
        }

        /// <summary>
        ///     The fewest variants <see cref="RelayoutLayoutInvariance"/> must compare: a floor under the committed count
        ///     (about 16,000 across both tiers), so an emptied or uncopied corpus cannot pass vacuously.
        /// </summary>
        private const int MinLayoutInvarianceCases = 14000;

        /// <summary>The carved gaps of <see cref="Relayout"/>: fails while any of them still diverges from NumPy.</summary>
        [TestMethod]
        [TestCategory("OpenBugs")]
        public void Relayout_KnownGaps() => RunGapCorpus("relayout_gaps.jsonl", LayoutInvarianceVerdict);

        /// <summary>The carved gaps of <see cref="RelayoutHost"/>: fails while any of them still diverges from NumPy.</summary>
        [TestMethod]
        [TestCategory("OpenBugs")]
        public void RelayoutHost_KnownGaps() => RunGapCorpus("relayout_host_gaps.jsonl", LayoutInvarianceVerdict);

        /// <summary>
        ///     Does NumSharp's outcome on the variant differ from its outcome on a C-ordered copy of the moved operand?
        /// </summary>
        /// <param name="c">A relayout case (its <see cref="FuzzCorpus.Case.Layout"/> is <c>relayout:&lt;kind&gt;@&lt;pos&gt;</c>).</param>
        /// <returns>
        ///     Null when both outcomes agree (or the case is <c>layout_sensitive</c>: NumPy's own answer follows the layout,
        ///     so the contract does not apply); otherwise one failure line naming the case and the first differences.
        /// </returns>
        /// <remarks>
        ///     Each side reconstructs every operand afresh, because an in-place op (np.place, np.put, …) mutates its first
        ///     operand. The copy comes from <see cref="FuzzCorpus.ReconstructCanonical"/>, which gathers the values without
        ///     any NumSharp kernel, so a copy routine that shares the bug under test cannot hide it.
        /// </remarks>
        private static string LayoutInvarianceVerdict(FuzzCorpus.Case c)
        {
            if (c.Layout_Sensitive)
                return null;
            int pos = MovedOperand(c);
            var (variant, variantError) = OutcomeOf(c, pos, canonical: false);
            var (copy, copyError) = OutcomeOf(c, pos, canonical: true);
            if (variantError != null || copyError != null)
            {
                // A throw on both sides is the same outcome here (NumPy threw or returned the same on both layouts, and
                // the error TEXT is the ordinary replay's business); a throw on one side only is the layout's doing.
                if (variantError != null && copyError != null)
                    return null;
                return $"{c.Id} [{c.Layout}]: " + (variantError != null
                    ? $"THREW {variantError} on the layout, returned a result on the C-ordered copy"
                    : $"returned a result on the layout, THREW {copyError} on the C-ordered copy");
            }
            if (variant.typecode != copy.typecode)
                return $"{c.Id} [{c.Layout}]: dtype {variant.typecode} on the layout, {copy.typecode} on the C-ordered copy";
            if (!ShapeEquals(variant.Shape.dimensions, copy.Shape.dimensions))
                return $"{c.Id} [{c.Layout}]: shape [{string.Join(",", variant.Shape.dimensions)}] on the layout, " +
                       $"[{string.Join(",", copy.Shape.dimensions)}] on the C-ordered copy";
            var diffs = BitDiff.Compare(FuzzCorpus.ResultBytes(copy), FuzzCorpus.ResultBytes(variant), copy.typecode, false);
            if (diffs.Count == 0)
                return null;
            return $"{c.Id} [{c.Layout}]: " +
                   string.Join(", ", diffs.Take(3).Select(d => $"@{d.Index} copy {d.Expected} layout {d.Actual}")) +
                   (diffs.Count > 3 ? $" (+{diffs.Count - 3} more)" : "");
        }

        /// <summary>The position of the operand a relayout case moved, read from its layout tag <c>relayout:kind@pos</c>.</summary>
        /// <param name="c">The relayout case.</param>
        /// <returns>The zero-based operand position.</returns>
        /// <exception cref="FormatException">The layout tag is not a relayout tag (a corpus or generator bug).</exception>
        private static int MovedOperand(FuzzCorpus.Case c)
        {
            int at = c.Layout?.LastIndexOf('@') ?? -1;
            if (at < 0 || !c.Layout.StartsWith("relayout:", StringComparison.Ordinal))
                throw new FormatException($"{c.Id}: layout '{c.Layout}' is not a relayout tag");
            return int.Parse(c.Layout.AsSpan(at + 1), System.Globalization.CultureInfo.InvariantCulture);
        }

        /// <summary>Run a relayout case's op on freshly reconstructed operands, the moved one as given or as a C-ordered copy.</summary>
        /// <param name="c">The case.</param>
        /// <param name="pos">The moved operand's position.</param>
        /// <param name="canonical">True to replace the moved operand by <see cref="FuzzCorpus.ReconstructCanonical"/>.</param>
        /// <returns>(result, null) on success; (null, "ExceptionType: message") when the call threw.</returns>
        private static (NDArray result, string error) OutcomeOf(FuzzCorpus.Case c, int pos, bool canonical)
        {
            try
            {
                var operands = new NDArray[c.Operands.Length];
                for (int i = 0; i < operands.Length; i++)
                    operands[i] = canonical && i == pos
                        ? FuzzCorpus.ReconstructCanonical(c.Operands[i])
                        : FuzzCorpus.Reconstruct(c.Operands[i]);
                return (DispatchApply(c.Op, c.Params, operands), null);
            }
            catch (Exception e)
            {
                return (null, $"{e.GetType().Name}: {e.Message}");
            }
        }
    }
}
