using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp.Backends.Unmanaged.Pooling;

namespace NumSharp.Tests.Lifetime
{
    /// <summary>
    ///     Sweeps the np.* surface for <b>deferred release</b>: an operation that strands an
    ///     unmanaged buffer on the finalizer queue instead of freeing it when it is done with it.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     This is the defect the repo's <c>using</c>-on-intermediate refactors exist to fix, and
    ///     the one its deleted <c>WorkingSet64</c> tests could not see (see <see cref="LeakGuards"/>).
    ///     It is not a leak in the unbounded sense — <c>~Disposer()</c> is a safety net that frees
    ///     the buffer eventually — which is precisely why a memory-growth probe cannot detect it.
    ///     It is a latency and peak-footprint defect: under load, released-late buffers accumulate
    ///     until the GC happens to run.
    ///     </para>
    ///     <para><b>The instrument.</b> <see cref="SizeBucketedBufferPool"/> already counts every
    ///     acquisition and release of an unmanaged buffer, exactly and at no added cost:</para>
    ///     <code>
    ///     acquisitions = Hits + Misses + ZeroedAllocs   // Take / TakeZeroed
    ///     releases     = Returns + ReturnsFreed         // Return, whether pooled or freed outright
    ///     </code>
    ///     <para>
    ///     Run an operation N times, disposing what it returns, with no GC in the window. Whatever
    ///     is still outstanding was left for the finalizer. The counters are exact — the clean ops
    ///     in the catalogue measure a deficit of precisely 0 at every N.
    ///     </para>
    ///     <para><b>Why the assertion is on the slope, not a threshold.</b> A deferred release
    ///     strands one buffer per call, so its deficit grows with N. Everything else — a lazily
    ///     initialised cache, a stray finalizer from an earlier test — is O(1) and does not. The
    ///     sweep therefore measures at N and 10N and divides: the per-call rate is the assertion and
    ///     there is no magic number to tune. One-time kernel/cache setup can add the same constant
    ///     offset at both sample sizes and therefore correctly reads as clean.
    ///     </para>
    ///     <para><b>Sequencing.</b> The pool counters are process-wide, so each window is opened
    ///     with the <see cref="GcQuiescence"/> protocol: a young-generation collection finalizes
    ///     what the previous window stranded, the finalizer queue is drained after the epoch is
    ///     taken, and the window is re-run whenever a collection started — or a background
    ///     collection finished — while it was open. No finalizer can therefore land inside an
    ///     accepted window, which makes every accepted reading exact. (A finalizer landing
    ///     mid-window could only ADD releases, masking a defect rather than inventing one, so even
    ///     the fallback reading below fails safe.) The class first runs one full drain for the
    ///     backlog earlier test classes left. <c>[DoNotParallelize]</c> is mandatory for the same reason.
    ///     </para>
    ///     <para><b>Why not a full collection per window.</b> Until 2026-09-24 every window was
    ///     drained with <c>GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect()</c>. In a full
    ///     suite run that drain marks ~120 MiB of the test framework's own per-test bookkeeping and
    ///     cost ~78 ms, so the 840 windows of <see cref="NoOperationDefersItsBufferRelease"/> took
    ///     68 s of a 262 s run — and grew with every test added to the suite. Exactness never needed
    ///     the full collection; see <see cref="GcQuiescence"/> for the argument.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class BufferReleaseSweepTests
    {
        /// <summary>
        ///     How many times one window is re-run when a collection disturbed it, before the last
        ///     reading is accepted as-is (see <see cref="Deficit"/>). The young collection that opens
        ///     every attempt resets the gen-0 budget, so a second disturbance in a row is already rare.
        /// </summary>
        private const int MaxWindowAttempts = 5;

        /// <summary>
        ///     Runs ONE full drain before the class's sweeps so the backlog earlier test classes left on
        ///     the finalizer queue (or as unvisited finalizable garbage) is cleared up front, instead of
        ///     surfacing as a disturbed window later. Once per CLASS: the per-window hygiene is the young
        ///     collection in <see cref="Deficit"/>, and the sweeps that follow one another here leave only
        ///     garbage that hygiene handles.
        /// </summary>
        /// <param name="context">The MSTest class context (unused).</param>
        /// <remarks>
        ///     Formerly a <c>[TestInitialize]</c>: a full drain costs ~80 ms in a full run (it marks the whole
        ///     test host's heap) and bought nothing after the first sweep — exactness never depended on it,
        ///     since every window is re-run when <see cref="GcQuiescence.Undisturbed"/> reports a collection
        ///     inside it; the drain only makes such re-runs rarer, and the backlog it clears is the one
        ///     OTHER classes left, which exists only before this class's first test.
        /// </remarks>
        [ClassInitialize]
        public static void DrainEarlierClasses(TestContext context) => GcQuiescence.CollectFull();

        /// <summary>
        ///     The two sample sizes. Both sit far below the knee where the GC starts finalizing
        ///     stranded buffers mid-loop — see <see cref="PerCallDeficit"/> for the measured curve.
        /// </summary>
        private const int SmallN = 10;

        private const int LargeN = 20;

        /// <summary>
        ///     One stranded buffer per call reads as 1.0 and the clean ops read 0.0, so half a
        ///     buffer per call sits well below any real defect and above the counters' (zero) noise.
        /// </summary>
        private const double MaxPerCall = 0.5;

        /// <summary>
        ///     The operations that defer a release today, with the rate each was measured at.
        ///     Every one is re-run by <see cref="KnownDeferredReleases_StillDeferring"/> so the debt
        ///     stays visible; when one is fixed that test goes green and the entry comes off this
        ///     list, at which point the main sweep guards it permanently.
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     EMPTY as of the 2026-08-23 disposal sweep, since carried by <see cref="NDScope"/>
        ///     boundary scopes (see <c>DISPOSAL-GUIDELINES.md</c> → "The standard instrument"): all
        ///     15 tracked sites reclaim eagerly — convolve's materialized reversed kernel,
        ///     the fancy get/set <c>computedOffsets</c> buffer, the 15-way
        ///     <c>MakeGeneric&lt;T&gt;()</c> alias drops in the getter/setter dispatch and the typed
        ///     <c>&amp;</c>/<c>|</c>/<c>^</c> operators, nonzero's column-view base matrix, the sort
        ///     driver's 1-D <c>expand_dims</c> promotion views, roll's flat-path handoff, clip's
        ///     bound casts, logical_*'s conversion temps, array_equal's comparison buffer, and the
        ///     quantile engine's staging copies. <c>np.where 3-arg</c> / <c>np.extract</c> were
        ///     HARNESS artifacts — their catalogue entries built the mask inside <c>Run</c>
        ///     (violating the operand rule above), now fixed in <see cref="LifetimeCases"/>;
        ///     <c>np.linspace</c> had already gone clean. The main sweep guards all of them now.
        ///     </para>
        ///     <para>
        ///     The former <c>np.isclose</c>/<c>np.allclose</c> 2-buffer slope was traced to the two
        ///     predicate calls inside isclose: IsFinite returned a typed alias while leaving the
        ///     freshly produced untyped owner for finalization. Owning that result through the alias
        ///     handoff fixed both APIs; the main sweep now guards them together with isnan/isfinite.
        ///     </para>
        /// </remarks>
        private static readonly HashSet<string> KnownDeferred = new();

        /// <summary>
        ///     Buffers acquired but not released across <paramref name="iterations"/> runs, measured
        ///     in a window no collection (and so no finalizer) touched.
        /// </summary>
        /// <param name="op">The operation under test.</param>
        /// <param name="operands">Its operands, built outside the window and reused by every run.</param>
        /// <param name="iterations">How many times the window runs the operation.</param>
        /// <returns>
        ///     Acquisitions minus releases over the window: 0 for an operation that releases everything
        ///     before returning, <c>strandRate × iterations</c> (+ any one-time retained buffers) otherwise.
        /// </returns>
        /// <remarks>
        ///     The window is re-run (up to <see cref="MaxWindowAttempts"/> times) whenever
        ///     <see cref="GcQuiescence.Undisturbed"/> reports a collection inside it — the pool's own
        ///     gen-0 pacing collection included, which fires once per 16 MiB handed out and so lands in
        ///     some window every few dozen. Re-running is sound because the window is re-executable: the
        ///     operands persist and every run of the operation produces the same pool traffic.
        /// </remarks>
        private static long Deficit(LifetimeCase op, NDArray[] operands, int iterations)
        {
            long deficit = 0;
            for (int attempt = 1; attempt <= MaxWindowAttempts; attempt++)
            {
                // Hygiene, not correctness: finalize what the previous window stranded (young garbage)
                // so those buffers are home again, and start the window with a fresh gen-0 budget.
                GcQuiescence.CollectYoung();

                // Exactness: everything queued before this point has run once OpenWindow returns, and
                // Undisturbed below proves no collection queued anything while the window was open.
                var window = GcQuiescence.OpenWindow();
                SizeBucketedBufferPool.ResetCounters();

                for (int i = 0; i < iterations; i++)
                    LifetimeCase.DisposeResult(op.Run(operands));

                long acquired = SizeBucketedBufferPool.Hits
                              + SizeBucketedBufferPool.Misses
                              + SizeBucketedBufferPool.ZeroedAllocs;
                long released = SizeBucketedBufferPool.Returns
                              + SizeBucketedBufferPool.ReturnsFreed;
                deficit = acquired - released;

                if (GcQuiescence.Undisturbed(window))
                    return deficit;
            }

            // A collection landed in every attempt. Accept the last reading, exactly as the sweep did
            // before it could detect interference at all: a mid-window finalizer can only ADD releases,
            // so this reading errs toward innocence and never invents a defect.
            return deficit;
        }

        /// <summary>
        ///     Stranded buffers per call: the slope between the two sample sizes, taken as the
        ///     MAXIMUM over several independent rounds.
        /// </summary>
        /// <remarks>
        ///     <para><b>Why two points and not one.</b> A one-off cost — a lazily built kernel, a
        ///     cache filled on first use — shows up as a constant offset at every N. Subtracting two
        ///     sample sizes cancels it, leaving only the part that grows per call. That is what makes
        ///     the assertion threshold-free: the defect is defined by scaling with the call count.
        ///     </para>
        ///     <para><b>Why both samples are small.</b> The deficit is
        ///     <c>strandRate*N</c> minus whatever the GC finalized mid-loop, so it is linear only
        ///     until the GC first intervenes, and then bends over. Measured across N for
        ///     np.allclose before its typed-alias ownership fix (formerly a confirmed 2/call site):
        ///     </para>
        ///     <code>
        ///     N        5   10   20   40   80  160  320  640
        ///     deficit 10   20   40   80  160  142  104   24     &lt;- linear to 80, then the GC bends it
        ///     control  0    0    0    0    0    0    0    0     &lt;- a clean op is 0 everywhere
        ///     </code>
        ///     <para>
        ///     An earlier draft sampled at N=20 and N=200, straddling that knee, and so reported
        ///     the old np.allclose defect as clean in one process and 2/call in another — the plateau flattens the
        ///     slope and reads as innocence. N=10 and N=20 sit far below the knee for every op in
        ///     the catalogue.
        ///     </para>
        ///     <para>
        ///     Since <see cref="Deficit"/> re-runs any window a collection touched, an ACCEPTED window
        ///     is linear by construction — the knee can only reach the fallback reading taken after
        ///     every attempt was disturbed. The small sample sizes still matter: they keep a window
        ///     short enough that disturbance stays rare.
        ///     </para>
        ///     <para><b>Why the maximum, not the minimum.</b> The perturbation here is asymmetric,
        ///     and in the opposite direction to a timing or allocation measurement: the GC finalizing
        ///     a stranded buffer mid-window ADDS a release and pushes the reading DOWN, toward
        ///     innocence. Only a stray acquisition pushes it up, and at these sample sizes one costs
        ///     0.1/call against a 0.5 threshold. So the ceiling across rounds is the estimate that
        ///     does not let a real defect hide. (This is why the repo's usual best-of-rounds MINIMUM
        ///     rule — right for <c>MinAllocated</c> and for timings, where noise only adds — is
        ///     inverted here.)
        ///     </para>
        ///     <para>Operands are rebuilt per round so no round inherits the previous one's state.</para>
        /// </remarks>
        private static double PerCallDeficit(LifetimeCase op, int rounds = 3)
        {
            double max = double.MinValue;

            for (int round = 0; round < rounds; round++)
            {
                var operands = op.MakeOperands();
                try
                {
                    LifetimeCase.DisposeResult(op.Run(operands)); // warm: JIT, kernel emission, caches

                    long small = Deficit(op, operands, SmallN);
                    long large = Deficit(op, operands, LargeN);

                    double perCall = (large - small) / (double)(LargeN - SmallN);
                    if (perCall > max)
                        max = perCall;
                }
                finally
                {
                    foreach (var o in operands)
                        o.Dispose();
                }
            }

            return max;
        }

        [TestMethod]
        [DoNotParallelize]
        public void NoOperationDefersItsBufferRelease()
        {
            var offenders = new List<string>();

            foreach (var op in LifetimeCases.All())
            {
                if (KnownDeferred.Contains(op.Name))
                    continue;

                double perCall = PerCallDeficit(op);
                if (perCall >= MaxPerCall)
                    offenders.Add($"{op.Name}: {perCall:0.##} buffers/call stranded");
            }

            offenders.Should().BeEmpty(
                "every operation must release its intermediates before returning; these left them "
                + "for the finalizer:\n  " + string.Join("\n  ", offenders));
        }

        /// <summary>
        ///     Regression gate for the typed-alias ownership handoff used by the three predicates.
        ///     isclose invokes isfinite twice, so the same fix also closes isclose/allclose's former
        ///     two-buffer-per-call slope.
        /// </summary>
        [TestMethod]
        [DoNotParallelize]
        public void PredicateAliases_ReleaseTheirUntypedOwners()
        {
            var names = new HashSet<string>
            {
                "np.isnan", "np.isfinite", "np.isinf", "np.isclose", "np.allclose"
            };

            var offenders = LifetimeCases.All()
                .Where(op => names.Contains(op.Name))
                .Select(op => (op.Name, Rate: PerCallDeficit(op)))
                .Where(result => result.Rate >= MaxPerCall)
                .Select(result => $"{result.Name}: {result.Rate:0.##} buffers/call stranded")
                .ToList();

            offenders.Should().BeEmpty(
                "predicate results must transfer ownership to their typed aliases synchronously");
        }

        /// <summary>
        ///     Proves the sweep can fail. Without this, a catalogue that silently ran zero cases, or
        ///     counters that stopped moving, would leave the sweep passing vacuously — the exact
        ///     failure mode of the working-set tests this suite replaces.
        /// </summary>
        [TestMethod]
        [DoNotParallelize]
        public void Sweep_IsNotVacuous()
        {
            LifetimeCases.All().Count().Should().BeGreaterThan(80,
                "the sweep is only as good as its coverage");

            // A deliberately stranded buffer must register at ~1 per call.
            var leaky = new LifetimeCase(
                "deliberate leak",
                () => new NDArray[0],
                _ => { GC.KeepAlive(np.zeros(new Shape(1000), NPTypeCode.Double)); return null; });

            PerCallDeficit(leaky).Should().BeGreaterThanOrEqualTo(MaxPerCall,
                "if a deliberate leak does not register, the sweep cannot detect anything");

            // ...and a known-clean op must read ~0, so the check is not simply always-positive.
            var clean = new LifetimeCase(
                "control astype",
                () => new[] { np.arange(2000).reshape(50, 40).astype(NPTypeCode.Int32) },
                o => o[0].astype(NPTypeCode.Double));

            PerCallDeficit(clean).Should().BeLessThan(MaxPerCall,
                "a correct operation must read as clean, or the sweep is just measuring noise");
        }

        /// <summary>
        ///     The documented backlog: fails until each entry is fixed. When one goes green, remove
        ///     it from <see cref="KnownDeferred"/> so the main sweep starts guarding it.
        /// </summary>
        [TestMethod]
        [DoNotParallelize]
        [OpenBugs]
        public void KnownDeferredReleases_StillDeferring()
        {
            var stillBroken = new List<string>();

            foreach (var op in LifetimeCases.All().Where(c => KnownDeferred.Contains(c.Name)))
            {
                double perCall = PerCallDeficit(op);
                if (perCall >= MaxPerCall)
                    stillBroken.Add($"{op.Name}: {perCall:0.##} buffers/call stranded");
            }

            stillBroken.Should().BeEmpty(
                "known deferred-release sites, tracked so they are not forgotten:\n  "
                + string.Join("\n  ", stillBroken));
        }
    }
}
