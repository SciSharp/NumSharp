using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Fuzz.Lifetime
{
    /// <summary>
    ///     The LIFETIME oracle (<c>test/oracle/gen_lifetime_oracle.py</c>): NumPy 2.4.2's answers for calls whose operands
    ///     are temporaries, replayed so that a kernel still using an operand the GC has already found unreachable is
    ///     caught — a lost GC anchor, which frees ("over-disposes") memory while it is in use.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>The bug class.</b> A NumSharp method reads an operand's raw buffer pointer and runs a kernel over it. If the
    ///     method (and every caller frame) never touches the operand again, optimized code stops reporting it to the GC
    ///     at the pointer read; a collection during the kernel then finds the operand unreachable, and its memory block's
    ///     finalizer (<c>UnmanagedMemoryBlock.Disposer</c>) returns the buffer to the pool — or to the OS — while the
    ///     kernel still reads or writes it. A caller passing a temporary
    ///     (<c>np.take(a, np.arange(n)[::-1])</c>) is all it takes; a second thread allocating (any multi-threaded
    ///     application) or the pool's own GC pacing supplies the collection. The answer is a silently wrong value, a
    ///     spurious <c>IndexError</c> from poisoned indices, or a fatal access violation. Every other tier replays over
    ///     operands the harness holds, so none of them can see it. The fix pattern is <c>GC.KeepAlive(operand)</c>
    ///     after the kernel (as np.nonzero, np.argwhere and np.unravel_index already do).
    ///     </para>
    ///     <para>
    ///     <b>How it is replayed</b> (<see cref="LifetimeChild"/>): in a child process with tiered compilation off,
    ///     each case runs once over held operands (the reference, compared with NumPy) and then repeatedly with its
    ///     <c>temp</c> operands built afresh inside the call expression while <see cref="LifetimeThief"/> forces full
    ///     collections and poisons the buffers the finalizers return. A repetition that differs from the reference,
    ///     throws, or kills the child is a lost anchor. A control (the bug pattern written by hand) must be caught in the
    ///     same run, or the run is inconclusive — it could not have seen a lost anchor.
    ///     </para>
    ///     <para>
    ///     <b>Gate versus gaps.</b> <c>lifetime.jsonl</c> holds the calls NumSharp keeps alive correctly — held to zero
    ///     divergences, so a regression that drops an anchor turns the gate red (with high probability: the detection is
    ///     a race the stress thread wins most of the time, never a false positive). <c>lifetime_gaps.jsonl</c> holds the
    ///     carved cases of the documented gaps; the [OpenBugs] tests replay them and fail while a gap is open.
    ///     </para>
    ///     <para>
    ///     <b>Release only.</b> The tests mean something only in an optimized build (CI runs Release): a Debug build
    ///     keeps every local live until its method returns, so they are inconclusive there at once
    ///     (<see cref="RequireOptimizedBuild"/>).
    ///     </para>
    /// </remarks>
    [TestClass]
    public partial class LifetimeOracleTests
    {
        /// <summary>Stressed repetitions per gate case (CI budget: the gate runs on every push, on three OSes).</summary>
        private const int GateReps = 8;

        /// <summary>
        ///     Stressed repetitions per carved case (local only, so a sharper verdict per gap: at 48 the short-window gaps —
        ///     a cast, a mask scan, a factorization's input copy — are caught in most runs).
        /// </summary>
        private const int GapReps = 48;

        /// <summary>
        ///     The gate: every call in <c>lifetime.jsonl</c> survives temporaries under the stress thread, and its
        ///     reference run matches NumPy.
        /// </summary>
        /// <exception cref="AssertFailedException">A gate case lost an operand mid-kernel (a red gate is a real lost
        ///     anchor: carve the case under a new gap id), or a reference run disagreed with NumPy.</exception>
        /// <exception cref="AssertInconclusiveException">A Debug build, or the run's control was not caught: the run
        ///     could not have seen a lost anchor.</exception>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void Lifetime()
        {
            RequireOptimizedBuild();
            var replay = LifetimeChildProcess.Run("lifetime.jsonl", GateReps);
            RequireDetectionPower(replay);
            var failures = replay.Outcomes.Where(o => o.LifetimeBroken || o.ReferenceBroken).ToList();
            Console.WriteLine($"lifetime gate: {replay.Outcomes.Count} cases, control {replay.ControlHits}/{replay.ControlReps}, " +
                              $"{replay.Outcomes.Sum(o => o.Collections)} collections, {replay.ChildRuns} child run(s)");
            Assert.AreEqual(0, failures.Count, Describe("lifetime gate failures", failures));
        }

        /// <summary>
        ///     The corpus contract, in-process (no child): both corpus files were copied to the test output and are not
        ///     empty, every case's op key has a body in <see cref="LifetimeOps"/>, every carved case names a gap id, and
        ///     every gap id has its own <c>Lifetime_&lt;id&gt;_*</c> [OpenBugs] test.
        /// </summary>
        /// <exception cref="AssertFailedException">A corpus file is missing or empty, an op key has no body, a carved case
        ///     names no gap, or a gap id has no [OpenBugs] test (a generator change the harness was not updated for).</exception>
        /// <remarks>
        ///     Without it a key the generator adds without a C# body fails only inside the child, as a reference run that
        ///     threw, and a new gap without a test is replayed only by the aggregate <see cref="Lifetime_KnownGaps"/>.
        /// </remarks>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void Lifetime_CorpusContract()
        {
            var gate = LifetimeCase.ReadAll(LifetimeCase.CorpusPath("lifetime.jsonl"));
            var gaps = LifetimeCase.ReadAll(LifetimeCase.CorpusPath("lifetime_gaps.jsonl"));
            Assert.IsTrue(gate.Count >= 200, $"only {gate.Count} lifetime gate cases: a corpus copy regression?");
            Assert.IsTrue(gaps.Count > 0, "the lifetime gaps file is empty: every gap retired? Then drop its tests too.");

            var keys = new HashSet<string>(LifetimeOps.Keys, StringComparer.Ordinal);
            var missing = gate.Concat(gaps).Select(c => c.Op).Where(op => !keys.Contains(op)).Distinct().ToList();
            Assert.AreEqual(0, missing.Count, "lifetime op keys without a body in LifetimeOps: " + string.Join(", ", missing));

            var unnamed = gaps.Where(c => string.IsNullOrEmpty(c.Gap)).Select(c => c.Id).ToList();
            Assert.AreEqual(0, unnamed.Count, "carved lifetime cases without a gap id: " + string.Join(", ", unnamed.Take(10)));
            var named = gate.Where(c => !string.IsNullOrEmpty(c.Gap)).Select(c => c.Id).ToList();
            Assert.AreEqual(0, named.Count, "gate cases carrying a gap id: " + string.Join(", ", named.Take(10)));

            var tests = typeof(LifetimeOracleTests).GetMethods().Select(m => m.Name).ToList();
            var untested = gaps.Select(c => c.Gap).Distinct()
                .Where(g => !tests.Any(t => t.StartsWith("Lifetime_" + g + "_", StringComparison.Ordinal))).ToList();
            Assert.AreEqual(0, untested.Count, "lifetime gaps without a Lifetime_<id>_* [OpenBugs] test: " + string.Join(", ", untested));
        }

        /// <summary>
        ///     The carved gaps: fails while any carved case still diverges, printing per gap how many cases are open and
        ///     how many no longer diverge (a gap that may be fixed: drop its carve rule and regenerate).
        /// </summary>
        /// <exception cref="AssertFailedException">A carved case still diverges (some gap is open).</exception>
        [TestMethod]
        [TestCategory("OpenBugs")]
        public void Lifetime_KnownGaps() => AssertGapsClosed(null);

        /// <summary>
        ///     Replays the carved cases (all of them, or one gap's) and fails while any of them diverges.
        /// </summary>
        /// <param name="gap">The gap id to replay, or null for every gap.</param>
        /// <exception cref="AssertFailedException">A carved case still diverges, the reference run broke (the case
        ///     no longer tests what it should), or a carved case names no gap.</exception>
        /// <exception cref="AssertInconclusiveException">A Debug build, or the run's control was not caught (no detection
        ///     power).</exception>
        internal static void AssertGapsClosed(string gap)
        {
            RequireOptimizedBuild();
            var replay = LifetimeChildProcess.Run("lifetime_gaps.jsonl", GapReps, gap);
            RequireDetectionPower(replay);
            var byGap = replay.Outcomes.GroupBy(o => o.Gap ?? "(none)").OrderBy(g => g.Key, StringComparer.Ordinal);
            var sb = new StringBuilder();
            int open = 0;
            foreach (var g in byGap)
            {
                int stillOpen = g.Count(o => o.LifetimeBroken);
                open += stillOpen;
                var sample = g.FirstOrDefault(o => o.LifetimeBroken);
                sb.AppendLine($"{g.Key}: {stillOpen}/{g.Count()} carved cases still diverge" +
                              (stillOpen < g.Count() ? $", {g.Count() - stillOpen} did not diverge in this run" : "") +
                              (sample != null ? $"\n    e.g. {sample.Id}: {(sample.Crashed ? "CRASHED " : "")}{sample.First}" : ""));
            }
            Console.WriteLine(sb.ToString());
            var orphans = replay.Outcomes.Where(o => string.IsNullOrEmpty(o.Gap)).ToList();
            Assert.AreEqual(0, orphans.Count, "carved cases without a gap id: " + string.Join(", ", orphans.Take(10).Select(o => o.Id)));
            var brokenRef = replay.Outcomes.Where(o => o.ReferenceBroken && !o.Crashed).ToList();
            Assert.AreEqual(0, brokenRef.Count, Describe("carved cases whose REFERENCE run broke", brokenRef));
            Assert.AreEqual(0, open, $"{open} carved lifetime case(s) still lose an operand mid-kernel:\n{sb}");
        }

        /// <summary>
        ///     Marks the test inconclusive, before any child process starts, when NumSharp.Core or the Oracle assembly is
        ///     a Debug build (its JIT optimizer disabled).
        /// </summary>
        /// <exception cref="AssertInconclusiveException">An assembly the replay runs is a Debug build.</exception>
        /// <remarks>
        ///     A Debug assembly carries <see cref="DebuggableAttribute"/> with
        ///     <see cref="DebuggableAttribute.IsJITOptimizerDisabled"/> set, and the JIT compiles each of its methods as
        ///     debuggable code — <c>DOTNET_TieredCompilation=0</c> does not change that — which reports every local and
        ///     every spilled temporary live until the method returns. No operand can lose its anchor there: NumSharp's
        ///     kernels keep their operands, and the replay's call expressions keep the temporaries. The control would
        ///     never be caught, so without this check a Debug run spends a minute per framework replaying and then ends
        ///     inconclusive anyway. Both assemblies matter: the lost anchors are in NumSharp.Core, the temporaries and the
        ///     control are in the Oracle assembly.
        /// </remarks>
        private static void RequireOptimizedBuild()
        {
            foreach (var assembly in new[] { typeof(np).Assembly, typeof(LifetimeOracleTests).Assembly })
            {
                // A Release build carries the attribute too (IgnoreSymbolStoreSequencePoints only); only the
                // DisableOptimizations mode a Debug build adds turns the optimizer off.
                var debuggable = assembly.GetCustomAttribute<DebuggableAttribute>();
                if (debuggable is { IsJITOptimizerDisabled: true })
                    Assert.Inconclusive($"{assembly.GetName().Name} is a Debug build (its JIT optimizer is disabled): " +
                                        "debuggable code keeps every local live until its method returns, so no operand " +
                                        "can lose its GC anchor and the lifetime replay could prove nothing. Run it in " +
                                        "Release (dotnet test -c Release), as CI does.");
            }
        }

        /// <summary>Marks the test inconclusive when the run's control was never caught.</summary>
        /// <param name="replay">The finished replay.</param>
        /// <exception cref="AssertInconclusiveException">The control had no hit.</exception>
        private static void RequireDetectionPower(LifetimeReplay replay)
        {
            if (replay.ControlHits <= 0)
                Assert.Inconclusive($"the lifetime control was not caught in {replay.ControlReps} repetitions: this host's " +
                                    "run could not have seen a lost GC anchor, so a clean result would prove nothing.");
        }

        /// <summary>Formats a failure list for an assertion message (the first 40 entries).</summary>
        private static string Describe(string title, List<LifetimeOutcome> failures)
        {
            var sb = new StringBuilder();
            sb.AppendLine($"{failures.Count} {title}:");
            foreach (var o in failures.Take(40))
            {
                sb.Append("  ").Append(o.Id).Append(": ");
                if (o.Crashed) sb.Append("CRASHED ");
                if (o.ReferenceBroken) sb.Append("reference ").Append(o.Reference).Append("; ");
                if (o.Wrong > 0 || o.Threw > 0) sb.Append($"wrong {o.Wrong} threw {o.Threw}; ");
                sb.AppendLine(o.First);
            }
            return sb.ToString();
        }
    }
}
