using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Interop.OpenBLAS;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     The DIRECT half of the leak audit for members no corpus row reaches: every
    ///     <see cref="LeakCatalogue"/> entry, measured with the corpus sweep's exact protocol (one warm
    ///     invocation, then <see cref="ScopeAudit.MeasureConfirmedTraffic"/>) and judged by the same
    ///     verdict (<see cref="AssertNoUnclassifiedEscapes"/>) — so a catalogue member that strands a
    ///     buffer, or hands back a fresh result allocated outside the pool, fails exactly like a corpus op.
    /// </summary>
    public partial class UndisposedIntermediateTests
    {
        /// <summary>
        ///     What one direct runner (the catalogue, or the property/field read gate) observed.
        /// </summary>
        /// <param name="Sweep">The verdict input: escape/bypass families keyed by surface id, and the per-id
        /// measured counts the completeness gate credits.</param>
        /// <param name="HarnessErrors">Entries that could not run at all (the warm invocation threw where
        /// the catalogue expected success) — a HARNESS defect, never a leak verdict, and always a failure.</param>
        /// <param name="BackendSkipReason">Why the backend-only measurements were skipped (no CBLAS/LAPACK
        /// library loads on this host), or null when they ran.</param>
        /// <param name="BackendSkipped">Surface ids whose backend-only measurements were skipped for
        /// <paramref name="BackendSkipReason"/> — credited by the completeness gate only on such a host,
        /// exactly as the host-pinned parity tiers go Inconclusive there instead of red.</param>
        internal sealed record DirectRunResult(
            SweepResult Sweep, IReadOnlyList<string> HarnessErrors, string BackendSkipReason, IReadOnlySet<string> BackendSkipped);

        /// <summary>
        ///     The one catalogue run of the process, shared by <see cref="Catalogue_EveryEntry_LeavesNoUndisposedIntermediates"/>
        ///     and the <see cref="LeakSurfaceCoverageTests"/> completeness gate (which credits only the
        ///     entries that were actually MEASURED, not merely declared).
        /// </summary>
        internal static readonly Lazy<DirectRunResult> SharedCatalogue =
            new(RunCatalogue, System.Threading.LazyThreadSafetyMode.ExecutionAndPublication);

        /// <summary>
        ///     The catalogue gate: every entry runs (a throwing entry is a harness defect and fails), and
        ///     every measured entry balances pool takes and returns with no unclassified pool bypass.
        /// </summary>
        /// <exception cref="AssertFailedException">An entry could not run, or an escape/bypass family is unclassified.</exception>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        [TestCategory("ScopeAudit")]
        public void Catalogue_EveryEntry_LeavesNoUndisposedIntermediates()
        {
            var run = SharedCatalogue.Value;
            var r = run.Sweep;

            Console.WriteLine($"[scope-audit/catalogue] entries={LeakCatalogue.All.Count} measured={r.DirectMeasured} " +
                              $"apis={r.MeasuredByOp.Count} gcInconclusive={r.GcInconclusive} harnessErrors={run.HarnessErrors.Count}");
            if (run.BackendSkipReason != null)
                Console.WriteLine($"[scope-audit/catalogue] {run.BackendSkipped.Count} backend-only ids skipped: {run.BackendSkipReason}");
            PrintPerOpRollup(r.Groups);
            PrintBypassRollup(r.Bypasses);

            // A catalogue entry is a coverage CLAIM: one that cannot run leaves its member unaudited,
            // so it fails here instead of silently shrinking the audited surface.
            if (run.HarnessErrors.Count > 0)
                Assert.Fail($"{run.HarnessErrors.Count} catalogue entries could not run (fix the entry — an unrunnable " +
                            "entry audits nothing):\n  " + string.Join("\n  ", run.HarnessErrors));

            // Non-vacuity: the catalogue measured (nearly) every entry it declares.
            Assert.IsTrue(r.DirectMeasured + r.GcInconclusive + run.BackendSkipped.Count > 0,
                          "the catalogue measured nothing — LeakCatalogue.All empty or the runner is broken");

            AssertNoUnclassifiedEscapes(r, "scope-audit/catalogue");
        }

        /// <summary>
        ///     Runs every catalogue entry: the managed ones first, then the backend-only ones with OpenBLAS
        ///     installed (threads=1) when a library loads, skipped and reported otherwise.
        /// </summary>
        /// <returns>The run's observations.</returns>
        private static DirectRunResult RunCatalogue()
        {
            var acc = new SweepAccumulator();
            var errors = new List<string>();
            var skipped = new HashSet<string>(StringComparer.Ordinal);
            string backendSkip = null;

            using var fx = new LeakFixture();
            ScopeAudit.Settle();   // drain the finalizer backlog earlier (undisposing) tests left behind

            foreach (var entry in LeakCatalogue.All.Where(e => !e.RequiresBackend))
                MeasureCatalogueEntry(entry, fx, acc, errors);

            var backendEntries = LeakCatalogue.All.Where(e => e.RequiresBackend).ToList();
            if (backendEntries.Count > 0)
            {
                backendSkip = TryEnableLeakBackend();
                if (backendSkip == null)
                {
                    try
                    {
                        foreach (var entry in backendEntries)
                            MeasureCatalogueEntry(entry, fx, acc, errors);
                    }
                    finally
                    {
                        // Every other test in this assembly asserts the managed kernels — never leave the
                        // backend installed past this block.
                        OpenBlasEngine.Disable();
                    }
                }
                else
                {
                    foreach (var entry in backendEntries)
                        skipped.Add(entry.Api);
                }
            }

            return new DirectRunResult(acc.ToResult(0), errors, backendSkip, skipped);
        }

        /// <summary>
        ///     Measures one catalogue entry: warm (a throw is recorded as a harness error and the entry is
        ///     skipped), then the confirmed measurement with the result's freshness computed against the
        ///     fixture's buffers so a pool bypass is detected, then recorded under the entry's surface id.
        /// </summary>
        /// <param name="entry">The entry.</param>
        /// <param name="fx">The shared fixture (never disposed by the region — see <see cref="LeakFixture.Keep"/>).</param>
        /// <param name="acc">The run's tallies.</param>
        /// <param name="errors">Where an unrunnable entry is reported.</param>
        private static void MeasureCatalogueEntry(LeakCase entry, LeakFixture fx, SweepAccumulator acc, List<string> errors)
        {
            long freshBytes = 0;
            void Region()
            {
                freshBytes = 0;   // re-executed on retry — recompute, don't accumulate
                object res = entry.Run(fx);
                // Freshness BEFORE disposal: a result that is neither a fixture view nor a scalar slot is
                // a fresh allocation, and one with zero pool traffic is the full-bypass signature.
                freshBytes = FreshBytesAny(res, fx.Keep, fx.Ranges);
                DisposeAny(res, fx.Keep);
            }

            if (entry.Throws)
            {
                // An always-throwing member: its error path is what exists to audit. The warm run must
                // throw — an entry flagged Throws that returns is stale (the member was fixed or changed)
                // and is reported, so the flag never silently hides a now-reachable success path.
                bool threw = false;
                try
                {
                    Region();
                }
                catch
                {
                    threw = true;
                }
                if (!threw)
                {
                    errors.Add($"{entry.Api} [{entry.Label}]: flagged Throws but returned — the member now succeeds; " +
                               "turn the entry into an ordinary E(...) so its success path is measured");
                    return;
                }

                void ErrorRegion()
                {
                    try
                    {
                        Region();
                    }
                    catch
                    {
                        // expected: this member always raises
                    }
                }

                var errTraffic = ScopeAudit.MeasureConfirmedTraffic(ErrorRegion);
                if (errTraffic == null)
                {
                    acc.GcInconclusive++;
                    return;
                }
                acc.Record(entry.Api, null, entry.Label, errTraffic.Value, 0, errorPath: true, entry.Label, "catalogue");
                return;
            }

            try
            {
                Region();   // warm: one-time caches (emitted kernels, FFT plans, printing tables)
            }
            catch (Exception e)
            {
                errors.Add($"{entry.Api} [{entry.Label}]: {e.GetType().Name}: {FirstLine(e.Message)}");
                return;
            }

            var traffic = ScopeAudit.MeasureConfirmedTraffic(Region);
            if (traffic == null)
            {
                acc.GcInconclusive++;   // a GC landed inside every attempt — indistinguishable, never red
                return;
            }

            acc.Direct++;
            acc.Record(entry.Api, null, entry.Label, traffic.Value, freshBytes, errorPath: false, entry.Label, "catalogue");
        }

        /// <summary>
        ///     Installs the OpenBLAS backend for a backend-only measurement block, single-threaded (the
        ///     deterministic configuration; a threaded BLAS also allocates per-thread scratch outside the
        ///     pool, which is irrelevant to the balance but slows the block down).
        /// </summary>
        /// <returns>Null when the backend is installed (the caller MUST <see cref="OpenBlasEngine.Disable"/>
        /// it afterwards); otherwise why no library loaded.</returns>
        internal static string TryEnableLeakBackend()
        {
            try
            {
                OpenBlasEngine.Enable(threads: 1);
                return null;
            }
            catch (Exception e)
            {
                // Enable is a no-op on failure (it never half-installs), so there is nothing to undo.
                return $"no CBLAS/LAPACK library loads here ({e.GetType().Name}: {FirstLine(e.Message)}); stage one " +
                       "with tools/fetch_openblas.py or point NUMSHARP_OPENBLAS_LIBRARY at a build";
            }
        }

        /// <summary>
        ///     Total bytes of the result's arrays that are FRESH allocations: not a fixture array, not a
        ///     view into any fixture's base buffer, and larger than a scalar-pool slot (the
        ///     StackedMemoryPool, which these counters cannot see). The catalogue twin of
        ///     <see cref="FreshResultBytes"/>, generalised to every result shape via
        ///     <see cref="CollectDisposables"/>.
        /// </summary>
        /// <param name="result">The entry's result, any shape.</param>
        /// <param name="keep">The fixture's never-dispose set (a fixture array is never fresh).</param>
        /// <param name="ranges">The fixture arrays' base-buffer byte ranges (a pointer inside one is a view).</param>
        /// <returns>Fresh bytes; 0 when the result holds no fresh array.</returns>
        private static long FreshBytesAny(object result, HashSet<object> keep, List<(ulong lo, ulong hi)> ranges)
        {
            long total = 0;
            foreach (var o in CollectDisposables(result))
            {
                if (o is not NDArray nd || keep.Contains(nd))
                    continue;
                long bytes = nd.size * nd.typecode.SizeOf();
                if (bytes <= ScalarSlotBytes)
                    continue;
                ulong a = Addr(nd);
                bool view = false;
                foreach (var (lo, hi) in ranges)
                {
                    if (a >= lo && a < hi)
                    {
                        view = true;
                        break;
                    }
                }
                if (!view)
                    total += bytes;
            }
            return total;
        }

        /// <summary>The base-buffer byte range an array's storage aliases (its internal slice), the
        /// span inside which any view of it must point.</summary>
        /// <param name="nd">A non-empty, live array.</param>
        /// <returns>The half-open range <c>[lo, hi)</c> as unsigned addresses.</returns>
        internal static unsafe (ulong lo, ulong hi) BaseRange(NDArray nd)
        {
            var slice = nd.Storage.InternalArray;
            ulong lo = (ulong)(byte*)slice.Address;
            return (lo, lo + (ulong)slice.BytesLength);
        }

        /// <summary>The first line of an exception message (multi-line messages would flood a failure list).</summary>
        /// <param name="message">The message (may be null).</param>
        /// <returns>The first line, or an empty string.</returns>
        private static string FirstLine(string message)
            => message?.Split('\n')[0].TrimEnd('\r') ?? string.Empty;
    }
}
