using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends.Unmanaged.Pooling;

namespace NumSharp.UnitTest.Lifetime
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
    ///     there is no magic number to tune. <c>np.allclose</c>, for one, shows a constant ~40-buffer
    ///     offset at both sample sizes; that is one-time setup and correctly reads as clean.
    ///     </para>
    ///     <para><b>Sequencing.</b> The pool counters are process-wide, so the window is drained
    ///     with a full GC + finalizer pass before <c>ResetCounters</c>. A neighbouring test's
    ///     finalizer landing mid-window can only ADD releases, which masks a defect rather than
    ///     inventing one — the sweep fails safe. <c>[DoNotParallelize]</c> is mandatory for the
    ///     same reason.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class BufferReleaseSweepTests
    {
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
        ///     The 16 operations that defer a release today, with the rate each was measured at.
        ///     Every one is re-run by <see cref="KnownDeferredReleases_StillDeferring"/> so the debt
        ///     stays visible; when one is fixed that test goes green and the entry comes off this
        ///     list, at which point the main sweep guards it permanently.
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     These are pre-existing, not regressions — the sweep is simply the first instrument
        ///     able to see them. All 16 were confirmed by the ground-truth discriminator: force
        ///     finalizers and the outstanding count collapses to exactly zero, proving the buffers
        ///     really were sitting on the finalizer queue rather than freed through some path the
        ///     pool counters do not observe.
        ///     </para>
        ///     <para>One is diagnosed to a line:</para>
        ///     <list type="bullet">
        ///       <item><b>convolve same / valid</b> — <c>NdArray.Convolve.cs</c> ends with
        ///             <c>return full["a:b"].copy();</c>. That slice is a VIEW, it is never disposed,
        ///             and it holds an ARC ref to <c>full</c>, so the <c>using var full</c> directly
        ///             above it cannot free the buffer. Mode 'full' takes no slice and measures 0 —
        ///             which is exactly what localises the defect to the slice.</item>
        ///     </list>
        ///     <para>
        ///     Worth noting for context: <c>np.allclose</c> is the operation whose <c>using</c> the
        ///     deleted working-set test claimed to guard. That <c>using</c> is real and correct, and
        ///     <c>np.isclose</c> underneath it still strands 2 buffers per call — a defect that test
        ///     never had the resolution to see.
        ///     </para>
        /// </remarks>
        private static readonly HashSet<string> KnownDeferred = new()
        {
            "convolve same",    // 1/call — the undisposed centre slice, see above
            "convolve valid",   // 1/call — same shape of defect
            "boolean mask",     // 1/call
            "fancy index",      // 2/call
            "np.where 3-arg",   // 1/call
            "np.sort",          // 1/call
            "np.argsort",       // 1/call
            "np.roll",          // 1/call
            "np.clip",          // 2/call
            "logical_and",      // 3/call — the heaviest in the catalogue
            "np.isnan",         // 1/call
            "np.isfinite",      // 1/call
            "np.extract",       // 1/call
            "np.allclose",      // 2/call
            "np.isclose",       // 2/call
            "np.array_equal",   // 1/call
        };

        /// <summary>
        ///     Buffers acquired but not released across <paramref name="iterations"/> runs, with no
        ///     GC inside the window.
        /// </summary>
        private static long Deficit(LifetimeCase op, NDArray[] operands, int iterations)
        {
            // Drain anything an earlier test left pending so it cannot land inside our window.
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            SizeBucketedBufferPool.ResetCounters();

            for (int i = 0; i < iterations; i++)
                LifetimeCase.DisposeResult(op.Run(operands));

            long acquired = SizeBucketedBufferPool.Hits
                          + SizeBucketedBufferPool.Misses
                          + SizeBucketedBufferPool.ZeroedAllocs;
            long released = SizeBucketedBufferPool.Returns
                          + SizeBucketedBufferPool.ReturnsFreed;

            return acquired - released;
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
        ///     np.allclose (a confirmed 2/call site):
        ///     </para>
        ///     <code>
        ///     N        5   10   20   40   80  160  320  640
        ///     deficit 10   20   40   80  160  142  104   24     &lt;- linear to 80, then the GC bends it
        ///     control  0    0    0    0    0    0    0    0     &lt;- a clean op is 0 everywhere
        ///     </code>
        ///     <para>
        ///     An earlier draft sampled at N=20 and N=200, straddling that knee, and so reported
        ///     np.allclose as clean in one process and 2/call in another — the plateau flattens the
        ///     slope and reads as innocence. N=10 and N=20 sit far below the knee for every op in
        ///     the catalogue.
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
