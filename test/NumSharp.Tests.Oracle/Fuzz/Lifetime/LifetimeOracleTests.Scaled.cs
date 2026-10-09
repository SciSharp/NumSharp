using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Fuzz.Lifetime
{
    /// <summary>
    ///     The scaled lifetime gate (<see cref="LifetimeScaled"/>): every op key of the ordinary and masked-array corpus,
    ///     its operands enlarged temporaries, replayed under the stress thread against the same call over held operands.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Gate versus known gaps.</b> A case whose op key reaches the site of an OPEN lifetime gap (an <c>LT</c> id of
    ///     <c>lifetime_gaps.jsonl</c>) is carved through <see cref="KnownScaledGaps"/>: the gate replays every other case at
    ///     zero divergences, and <see cref="LifetimeScaled_KnownGaps"/> ([OpenBugs]) replays the carved ones. The scaled
    ///     tier does not keep gap ids of its own: a site it finds that no gap names gets an <c>LT</c> id and dedicated
    ///     cases in the lifetime corpus (as LT28 did), so every gap has a pin that does not depend on how far a corpus
    ///     case scales.
    ///     </para>
    ///     <para>
    ///     <b>Retiring a gap.</b> When a gap's site is fixed, its lifetime cases leave <c>lifetime_gaps.jsonl</c>;
    ///     <see cref="LifetimeScaled_KnownGapsContract"/> then fails until the entries naming that gap are removed here, and
    ///     the gate replays their op keys again. An op key whose call reaches the site through another path than the one
    ///     fixed turns the gate red at that point: <see cref="ScaledGap.Path"/> says which path the entry assumed.
    ///     </para>
    ///     <para>
    ///     <b>State (2026-10-09).</b> <see cref="KnownScaledGaps"/> is empty: the 21 op keys it carved (35 cases, reaching
    ///     LT01, LT02, LT05, LT07, LT09, LT12, LT14, LT18, LT20, LT21, LT22 and LT28) went back into the gate when the
    ///     weaver's operand roots fixed every lifetime gap, and replayed clean.
    ///     </para>
    /// </remarks>
    public partial class LifetimeOracleTests
    {
        /// <summary>
        ///     Stressed runs per scaled case in the gate (CI budget: about a thousand cases per framework, about a minute).
        ///     <c>NUMSHARP_LIFETIME_SCALED_REPS</c> overrides it for a local discovery run (more runs reach the rarer windows).
        /// </summary>
        private static int ScaledGateReps =>
            int.TryParse(Environment.GetEnvironmentVariable("NUMSHARP_LIFETIME_SCALED_REPS"), out int n) && n > 0 ? n : 4;

        /// <summary>Stressed runs per carved scaled case (local only; a sharper verdict per gap).</summary>
        private const int ScaledGapReps = 24;

        /// <summary>
        ///     The fewest gate cases that must scale (1,242 did on 2026-10-09, ordinary and masked): fewer means a selection
        ///     or scaling regression — a corpus copy that lost files, a strategy that raises everywhere — and a gate that
        ///     tests far less than it says.
        /// </summary>
        private const int MinScaledGateCases = 1100;

        /// <summary>
        ///     A known scaled gap: the open lifetime gap whose site an op key's scaled cases reach.
        /// </summary>
        /// <param name="Gap">The lifetime gap id (<c>LT01</c> …). It must stay an OPEN gap of <c>lifetime_gaps.jsonl</c>
        ///     (<see cref="LifetimeScaled_KnownGapsContract"/>).</param>
        /// <param name="Path">How the op key's call reaches the gap's site: what the entry assumes, and what to re-check
        ///     when the gap is retired.</param>
        /// <param name="Applies">Which of the op key's cases reach the site (null: all of them); the others stay in the
        ///     gate. It reads the streamed case (<see cref="LifetimeScaled.Select"/>).</param>
        internal sealed record ScaledGap(string Gap, string Path, Func<FuzzCorpus.Case, bool> Applies = null);

        /// <summary>
        ///     The op keys whose scaled cases lose an operand mid-kernel, each mapped to the open lifetime gap whose site
        ///     its call reaches (map an entry only after a code read and, where the call path is not obvious, a standalone
        ///     probe of the call with a temporary operand under the same stress). Empty: no lifetime gap is open, so every
        ///     selected case is gated (the 21 op keys mapped here until 2026-10-09 are listed in the class remarks).
        /// </summary>
        internal static readonly Dictionary<string, ScaledGap> KnownScaledGaps = new(StringComparer.Ordinal)
        {
        };

        /// <summary>
        ///     The selection, shared by the gate, the contract and the known-gaps replay: <see cref="LifetimeScaled.Select"/>
        ///     parses every ordinary tier, which is too slow to repeat per test.
        /// </summary>
        private static readonly Lazy<List<LifetimeScaled.ScaledCase>> ScaledSelection = new(() => LifetimeScaled.Select(ScaledGapOf));

        /// <summary>
        ///     The scaled gate: no selected case outside <see cref="KnownScaledGaps"/> answers differently, raises or crashes
        ///     when its operands are temporaries.
        /// </summary>
        /// <exception cref="AssertFailedException">A lost anchor in a case not known as a gap (map it to the open gap whose
        ///     site it reaches, or give the site a new lifetime gap), or too few cases scaled (a selection or scaling
        ///     regression).</exception>
        /// <exception cref="AssertInconclusiveException">A Debug build, or the run's control was not caught.</exception>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void LifetimeScaled_EveryOpKey()
        {
            RequireOptimizedBuild();
            var all = ScaledSelection.Value;
            var gate = Reindex(all.Where(s => s.Gap == null));
            var replay = LifetimeChildProcess.RunScaled(gate, ScaledGateReps);
            RequireDetectionPower(replay);

            var byIndex = gate.ToDictionary(s => s.Index);
            int scaled = replay.Outcomes.Count(o => o.Reference == "ok");
            int skipped = replay.Outcomes.Count(o => o.Reference.StartsWith("skipped", StringComparison.Ordinal));
            var broken = replay.Outcomes.Where(o => o.Reference == "ok" && o.LifetimeBroken).ToList();
            Console.WriteLine($"scaled lifetime gate: {gate.Count} cases over {gate.Select(s => s.Op).Distinct().Count()} op keys, " +
                              $"{scaled} scaled, {skipped} skipped, control {replay.ControlHits}/{replay.ControlReps}, " +
                              $"{replay.ChildRuns} child run(s); {all.Count - gate.Count} cases carved under known gaps");
            Assert.AreEqual(0, broken.Count, DescribeScaled("scaled lifetime failures (map each to its lifetime gap in KnownScaledGaps, " +
                                                            "or give its site a new gap)", broken, byIndex));
            Assert.IsTrue(scaled >= MinScaledGateCases,
                $"only {scaled} scaled gate cases (at least {MinScaledGateCases} expected): a selection or scaling regression?");
        }

        /// <summary>
        ///     The registry contract, in-process (no child): every <see cref="KnownScaledGaps"/> entry names a lifetime gap
        ///     that is still open, and carves at least one selected case.
        /// </summary>
        /// <exception cref="AssertFailedException">An entry names a retired or unknown gap (remove it: the gate replays the
        ///     op key again), or carves nothing (the op key left the corpus, or its predicate selects none).</exception>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void LifetimeScaled_KnownGapsContract()
        {
            var open = new HashSet<string>(LifetimeCase.ReadAll(LifetimeCase.CorpusPath("lifetime_gaps.jsonl")).Select(c => c.Gap),
                StringComparer.Ordinal);
            var selection = ScaledSelection.Value;
            var problems = new List<string>();
            foreach (var (op, g) in KnownScaledGaps.OrderBy(kv => kv.Key, StringComparer.Ordinal))
            {
                if (!open.Contains(g.Gap))
                    problems.Add($"{op}: {g.Gap} is not an open lifetime gap (retired, or a typo): remove the entry");
                if (!selection.Any(s => s.Op == op && s.Gap == g.Gap))
                    problems.Add($"{op}: carves no selected case (the op key left the corpus, or its predicate selects none)");
            }
            Assert.AreEqual(0, problems.Count, "KnownScaledGaps entries out of date:\n  " + string.Join("\n  ", problems));
        }

        /// <summary>
        ///     The known scaled gaps: replays every carved case and fails while any still loses an operand, printing per gap
        ///     how many do. <c>NUMSHARP_LIFETIME_SCALED_GAP</c> (a gap id) narrows the replay to one gap.
        /// </summary>
        /// <exception cref="AssertFailedException">A carved case still diverges, raises or crashes (a gap is open).</exception>
        /// <exception cref="AssertInconclusiveException">A Debug build, the control was not caught, or no case is carved
        ///     (for the requested gap).</exception>
        /// <remarks>
        ///     Detection is a race: a short window (a key copy, a cast) is caught in some runs only, so a gap whose cases all
        ///     pass once may still be open. Its dedicated lifetime test (<c>Lifetime_&lt;id&gt;_*</c>) is the reliable pin.
        /// </remarks>
        [TestMethod]
        [TestCategory("OpenBugs")]
        public void LifetimeScaled_KnownGaps()
        {
            RequireOptimizedBuild();
            string only = Environment.GetEnvironmentVariable("NUMSHARP_LIFETIME_SCALED_GAP");
            var cases = Reindex(ScaledSelection.Value.Where(s => s.Gap != null && (string.IsNullOrEmpty(only) || s.Gap == only)));
            if (cases.Count == 0)
                Assert.Inconclusive($"no scaled case is carved{(string.IsNullOrEmpty(only) ? "" : " under " + only)}");
            var replay = LifetimeChildProcess.RunScaled(cases, ScaledGapReps);
            RequireDetectionPower(replay);

            var byIndex = cases.ToDictionary(s => s.Index);
            var open = replay.Outcomes.Where(o => o.Reference == "ok" && o.LifetimeBroken).ToList();
            var sb = new StringBuilder();
            foreach (var g in replay.Outcomes.GroupBy(o => byIndex[o.Index].Gap).OrderBy(g => g.Key, StringComparer.Ordinal))
            {
                int stillOpen = g.Count(o => o.Reference == "ok" && o.LifetimeBroken);
                var ops = g.Select(o => byIndex[o.Index].Op).Distinct().OrderBy(o => o, StringComparer.Ordinal);
                sb.AppendLine($"{g.Key}: {stillOpen}/{g.Count()} carved scaled cases still lose an operand ({string.Join(", ", ops)})");
            }
            Console.WriteLine(sb.ToString());
            Assert.AreEqual(0, open.Count, DescribeScaled("carved scaled cases still lose an operand mid-kernel", open, byIndex));
        }

        /// <summary>Names the known gap a case reaches (<see cref="KnownScaledGaps"/>), or null for a gate case.</summary>
        /// <param name="c">A streamed corpus case (its parameters are still readable).</param>
        /// <returns>The gap id, or null.</returns>
        private static string ScaledGapOf(FuzzCorpus.Case c)
            => KnownScaledGaps.TryGetValue(c.Op, out var g) && (g.Applies?.Invoke(c) ?? true) ? g.Gap : null;

        /// <summary>
        ///     True when a case has no <c>axis</c> parameter or a null one (a flat scan or reduction) — the usual predicate of
        ///     a <see cref="ScaledGap"/> whose site only the flat call reaches (LT12's scans were mapped with it).
        /// </summary>
        /// <param name="c">The case.</param>
        /// <returns>Whether the call runs without an axis.</returns>
        internal static bool IsNullAxis(FuzzCorpus.Case c)
            => c.Params == null || !c.Params.TryGetValue("axis", out var axis) || axis.ValueKind == JsonValueKind.Null;

        /// <summary>Renumbers a filtered selection so its indices are positions again (the child's resume key).</summary>
        /// <param name="cases">The cases, in selection order.</param>
        /// <returns>The renumbered list.</returns>
        private static List<LifetimeScaled.ScaledCase> Reindex(IEnumerable<LifetimeScaled.ScaledCase> cases)
            => cases.Select((s, i) => s with { Index = i }).ToList();

        /// <summary>Formats failures grouped by op key (the first 60 op keys).</summary>
        /// <param name="title">The heading.</param>
        /// <param name="failures">The failing outcomes.</param>
        /// <param name="byIndex">The selection by index (for the op keys).</param>
        /// <returns>The text.</returns>
        private static string DescribeScaled(string title, List<LifetimeOutcome> failures, Dictionary<int, LifetimeScaled.ScaledCase> byIndex)
        {
            var sb = new StringBuilder();
            var groups = failures.GroupBy(o => byIndex.TryGetValue(o.Index, out var s) ? s.Op : "?").OrderBy(g => g.Key, StringComparer.Ordinal).ToList();
            sb.AppendLine($"{failures.Count} {title}, {groups.Count} op keys:");
            foreach (var g in groups.Take(60))
            {
                var o = g.First();
                sb.Append("  ").Append(g.Key).Append(" (").Append(g.Count()).Append("): ").Append(o.Id).Append(": ");
                if (o.Crashed) sb.Append("CRASHED ");
                if (o.Wrong > 0 || o.Threw > 0) sb.Append($"wrong {o.Wrong} threw {o.Threw}; ");
                sb.AppendLine(o.First);
            }
            return sb.ToString();
        }
    }
}
