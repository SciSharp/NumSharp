using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     The array-OBJECT oracle (<c>test/oracle/gen_viewflags_oracle.py</c>): what NumPy 2.4.2 hands back, not only its
    ///     values — view versus copy versus identity of the returned array (also through <c>out=</c>), the flags and strides
    ///     of the returned array, and the forms a shape or a secondary array parameter can take.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Every other tier compares a result as (dtype, shape, C-order bytes), which cannot tell a view from a copy, the
    ///     operand itself from a new object, an F result from a C one, or a read-only array from a writeable one. These
    ///     tiers carry the memory facet (<see cref="FuzzCorpus.MemFacet"/>) and the replay compares it
    ///     (<c>CompareMem</c>): the five flags, the normalized byte strides, and per operand whether the result shares its
    ///     memory and whether it IS the operand. All four tiers are held to zero excuses (<see cref="MisalignedRegistry"/>).
    ///     </para>
    ///     <para>
    ///     <b>Known gaps are carved, not excused.</b> The cells NumSharp gets wrong are written by the generator into
    ///     <c>gaps/&lt;tier&gt;_gaps.jsonl</c>, each case naming its gap id (<c>gen_viewflags_oracle.GAPS</c>); the
    ///     <c>[TestCategory("OpenBugs")]</c> tests below (excluded from CI like every [OpenBugs] test) replay them and fail
    ///     while any gap is open, printing per gap how many cases still diverge and how many now match NumPy. When a gap is
    ///     fixed, drop its carve rule and regenerate: its cases then join the gate.
    ///     </para>
    /// </remarks>
    public partial class FuzzCorpusTests
    {
        /// <summary>
        ///     Returned arrays: views, copies and identities of 194 APIs over 21 source layouts (owners in C and F order,
        ///     transposed / permuted / strided / reversed / offset views, read-only owners and views of them, broadcast
        ///     views, 0-d owners and views, empty arrays), plus the layout and flags of fresh results (ufuncs, reductions,
        ///     the copy family, the like-creators, triangles, partitions, the FFT family, creators, ndarray methods).
        /// </summary>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void ViewFlags() => RunCorpus("view_flags.jsonl");

        /// <summary>
        ///     <c>out=</c>: every out-capable API returns the out array ITSELF and writes into it, for out arrays that own
        ///     their memory (C or F) and views of every layout; NumPy's refusals of an out it cannot use are held verbatim.
        /// </summary>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void OutFlags() => RunCorpus("out_flags.jsonl");

        /// <summary>
        ///     Shape parameters: the shape-taking APIs with the same dims passed as a fresh Shape, a view's Shape (every
        ///     view layout), a <c>long[]</c>, an <c>int[]</c> and an <c>int</c>; NumPy's refusals of invalid dims verbatim.
        /// </summary>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void ShapeArgs() => RunCorpus("shape_args.jsonl");

        /// <summary>
        ///     Secondary array parameters (indices, conditions, sorters, bins, weights, values, masks, second operands) in
        ///     non-C layouts — F owners, transposed / strided / reversed / offset views, read-only owners, broadcast views and
        ///     0-d arrays — through the existing op keys, with the result's memory facet.
        /// </summary>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void ArrayArgs() => RunCorpus("array_args.jsonl");

        /// <summary>The carved gaps of <see cref="ViewFlags"/>: fails while any of them still diverges from NumPy.</summary>
        [TestMethod]
        [TestCategory("OpenBugs")]
        public void ViewFlags_KnownGaps() => RunGapCorpus("view_flags_gaps.jsonl");

        /// <summary>The carved gaps of <see cref="OutFlags"/>: fails while any of them still diverges from NumPy.</summary>
        [TestMethod]
        [TestCategory("OpenBugs")]
        public void OutFlags_KnownGaps() => RunGapCorpus("out_flags_gaps.jsonl");

        /// <summary>The carved gaps of <see cref="ShapeArgs"/>: fails while any of them still diverges from NumPy.</summary>
        [TestMethod]
        [TestCategory("OpenBugs")]
        public void ShapeArgs_KnownGaps() => RunGapCorpus("shape_args_gaps.jsonl");

        /// <summary>The carved gaps of <see cref="ArrayArgs"/>: fails while any of them still diverges from NumPy.</summary>
        [TestMethod]
        [TestCategory("OpenBugs")]
        public void ArrayArgs_KnownGaps() => RunGapCorpus("array_args_gaps.jsonl");

        /// <summary>
        ///     Replays one carved-gaps file (<c>Fuzz/corpus/gaps/</c>) and reports, per gap id, how many of its cases still
        ///     diverge from NumPy and how many now match (a gap that may be fixed: drop its carve rule and regenerate).
        /// </summary>
        /// <param name="fileName">The gaps file name inside <c>Fuzz/corpus/gaps/</c>.</param>
        /// <param name="extraVerdict">
        ///     An optional second verdict per case (null: none): a non-null line keeps the case open even when the ordinary
        ///     replay passes it. The relayout gaps pass the layout-invariance verdict here, because a carved layout bug can
        ///     be excused by the ordinary replay's registry (the source tier's rules travel with a variant) and would
        ///     otherwise read as "now matches" while NumSharp still misreads the layout.
        /// </param>
        /// <exception cref="AssertFailedException">Any carved case still diverges (the gap is open), or the file names a
        ///     case without a gap id (a generator bug: a carved case must say which gap it pins).</exception>
        /// <remarks>
        ///     An empty gaps file passes: every carve rule has been retired. The per-gap summary is printed whether the test
        ///     passes or fails, so the size of each open gap stays visible.
        /// </remarks>
        private static void RunGapCorpus(string fileName, Func<FuzzCorpus.Case, string> extraVerdict = null)
        {
            using var cases = FuzzCorpus.Open(System.IO.Path.Combine("gaps", fileName));
            var byGap = new SortedDictionary<string, (int total, int open, string sample)>(StringComparer.Ordinal);
            var orphans = new List<string>();
            foreach (var c in cases)
            {
                if (string.IsNullOrEmpty(c.Gap))
                {
                    orphans.Add(c.Id);
                    continue;
                }
                var failures = new List<string>();
                ReplayOne(c, failures, new Dictionary<string, int>());
                string extra = extraVerdict?.Invoke(c);
                if (extra != null)
                    failures.Add(extra);
                var (total, open, sample) = byGap.TryGetValue(c.Gap, out var e) ? e : (0, 0, null);
                byGap[c.Gap] = (total + 1, open + (failures.Count > 0 ? 1 : 0), sample ?? (failures.Count > 0 ? failures[0] : null));
            }

            foreach (var (gap, (total, open, sample)) in byGap)
                Console.WriteLine($"[{fileName}] {gap}: {open}/{total} carved cases still diverge" +
                                  (open < total ? $", {total - open} now match NumPy" : "") +
                                  (sample != null ? $"\n    e.g. {sample}" : ""));

            Assert.AreEqual(0, orphans.Count, "carved cases without a gap id: " + string.Join(", ", orphans.Take(10)));
            int stillOpen = byGap.Values.Sum(v => v.open);
            Assert.AreEqual(0, stillOpen,
                $"{stillOpen} carved cases still diverge from NumPy across {byGap.Count(kv => kv.Value.open > 0)} open gaps " +
                $"({string.Join(", ", byGap.Where(kv => kv.Value.open > 0).Select(kv => kv.Key))})");
        }
    }
}
