using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Interop.OpenBLAS;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     The BACKEND half of the leak sweep: the whole ordinary corpus replayed with the OpenBLAS
    ///     backend installed, so the code paths only a backend reaches — the LAPACK factorisation family
    ///     (Core ships no managed QR/SVD/eigensolver: without a backend those ops raise
    ///     <see cref="NotSupportedException"/> and the managed sweep can only count them as threw-skipped),
    ///     the BLAS product seams (dot/matmul/inner/vdot/vecdot/matvec/vecmat through cblas, einsum's
    ///     products, correlate/convolve's sliding-dot seam, cov with the Gram path gated off) and the
    ///     Interop assembly's own glue — are gated at zero escapes too.
    /// </summary>
    /// <remarks>
    ///     Values are not compared here (the host-pinned <c>matmul_parity</c>/<c>linalg_parity</c> tiers
    ///     do that against one exact binary), so ANY loadable CBLAS/LAPACK serves — the pin's content
    ///     hash is deliberately not required. A host where no library loads (CI's test job stages no
    ///     native binary) reports <c>Inconclusive</c>, never red, exactly like the parity tiers.
    /// </remarks>
    public partial class UndisposedIntermediateTests
    {
        /// <summary>
        ///     The op keys whose EVERY corpus case needs a matrix backend (the LAPACK family and the
        ///     polynomial ops composed on it). Without a backend they are threw-skipped by the managed
        ///     sweep; this backend pass must measure each of them, and <see cref="LeakSurfaceCoverageTests"/>
        ///     accepts their inventory members as covered by THIS pass.
        /// </summary>
        internal static readonly HashSet<string> BackendOnlyOpKeys = new(StringComparer.Ordinal)
        {
            "cholesky", "cond", "eig", "eigh", "eigvals", "eigvalsh", "lstsq", "matrix_rank",
            "norm", "pinv", "polyfit", "qr", "roots", "svd", "svdvals",
        };

        /// <summary>
        ///     What the one backend pass of the process observed.
        /// </summary>
        /// <param name="Sweep">The replay's observations; null when the pass could not run.</param>
        /// <param name="SkipReason">Why the pass could not run (no CBLAS/LAPACK library loads on this host), or
        /// null when it ran.</param>
        /// <param name="BackendInfo">The loaded backend's description (<see cref="OpenBlasEngine.Info"/>), captured
        /// while it was installed — it is uninstalled again before anyone reads this record.</param>
        internal sealed record BackendSweepResult(SweepResult Sweep, string SkipReason, string BackendInfo)
        {
            /// <summary>
            ///     Whether the pass actually EXERCISED op key <paramref name="op"/> on a success path (a confirmed
            ///     measurement, or one a GC made inconclusive — never red). False when the pass did not run: the
            ///     caller decides what a skipped pass means (<see cref="SkipReason"/>).
            /// </summary>
            /// <param name="op">A corpus op key.</param>
            /// <returns>True when a success-path measurement of <paramref name="op"/> was attempted.</returns>
            public bool Attempted(string op)
                => Sweep != null && (Sweep.MeasuredByOp.GetValueOrDefault(op) > 0 || Sweep.InconclusiveIds.Contains(op));
        }

        /// <summary>
        ///     The one backend pass of the process, shared by <see cref="Corpus_BackendOps_LeaveNoUndisposedIntermediates"/>
        ///     and the <see cref="LeakSurfaceCoverageTests"/> completeness gate, which credits a LAPACK-only member
        ///     only when THIS pass measured it — not merely because its op key is listed in
        ///     <see cref="BackendOnlyOpKeys"/>. Lazy + thread-safe like <see cref="SharedSweep"/>: the pass replays
        ///     the whole ordinary corpus, so running it twice would double the gate's wall time for no evidence.
        /// </summary>
        internal static readonly Lazy<BackendSweepResult> SharedBackendSweep =
            new(RunBackendSweep, System.Threading.LazyThreadSafetyMode.ExecutionAndPublication);

        /// <summary>
        ///     Replays every ordinary corpus tier with the OpenBLAS backend installed (threads=1, the
        ///     deterministic configuration) and applies the sweep's verdict. Also asserts non-vacuity
        ///     for the backend: every <see cref="BackendOnlyOpKeys"/> op must now have measured cases.
        /// </summary>
        /// <exception cref="AssertInconclusiveException">No CBLAS/LAPACK library loads on this host.</exception>
        /// <exception cref="AssertFailedException">An escape/bypass family is unclassified, or a
        /// backend-only op was still not measured with the backend installed.</exception>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        [TestCategory("ScopeAudit")]
        public void Corpus_BackendOps_LeaveNoUndisposedIntermediates()
        {
            var run = SharedBackendSweep.Value;
            if (run.SkipReason != null)
            {
                Assert.Inconclusive("the backend leak pass needs a loadable CBLAS/LAPACK library: " + run.SkipReason);
                return;
            }

            var r = run.Sweep;
            Console.WriteLine($"[scope-audit/backend] {run.BackendInfo} :: measured={r.OrdinaryMeasured} " +
                              $"errorPathsMeasured={r.ErrorPathsMeasured} gcInconclusive={r.GcInconclusive} " +
                              $"threwSkipped={r.ThrewSkipped} files={r.Files}");
            PrintThrewRollup(r.ThrewByOp);
            PrintPerOpRollup(r.Groups);

            // Non-vacuity for the backend itself: with a backend installed, the LAPACK family must
            // run — a backend that loaded yet declined these operands would leave them unaudited.
            var unmeasured = BackendOnlyOpKeys.Where(op => r.MeasuredByOp.GetValueOrDefault(op) == 0)
                                              .OrderBy(op => op, StringComparer.Ordinal).ToArray();
            Assert.AreEqual(0, unmeasured.Length,
                "backend-only ops still unmeasured WITH the backend installed: " + string.Join(", ", unmeasured));

            // Non-vacuity for the replay as a whole, 5% under the 2026-09-23 counts (identical on net10.0
            // and net8.0: 138,936 success + 2,441 error paths over 77 files): the per-op check above cannot
            // see the pass silently dropping whole tiers (a file filter or schema regression).
            Assert.IsTrue(r.OrdinaryMeasured > 131_900,
                $"backend pass measured only {r.OrdinaryMeasured} cases — corpus filter or schema regression?");
            Assert.IsTrue(r.ErrorPathsMeasured > 2_300,
                $"backend pass measured only {r.ErrorPathsMeasured} error paths — expects_throw replay regression?");

            AssertNoUnclassifiedEscapes(r, "scope-audit/backend");
        }

        /// <summary>
        ///     Runs the backend pass: installs OpenBLAS (threads=1), replays the ordinary corpus tiers, and
        ///     uninstalls the backend again whatever happens — every other test in this assembly asserts the
        ///     managed kernels.
        /// </summary>
        /// <returns>The pass's observations, or a skipped result when no library loads.</returns>
        private static BackendSweepResult RunBackendSweep()
        {
            string skip = TryEnableLeakBackend();
            if (skip != null)
                return new BackendSweepResult(null, skip, null);

            try
            {
                // Read while installed: Info describes the LOADED library and reports nothing afterwards.
                string info = OpenBlasEngine.Info;
                var corpusDir = Path.Combine(AppContext.BaseDirectory, "Fuzz", "corpus");
                var files = Directory.GetFiles(corpusDir, "*.jsonl")
                    .Select(Path.GetFileName)
                    .Where(f => !f.StartsWith("index_", StringComparison.Ordinal))   // the indexer never routes through BLAS
                    .Where(f => !f.StartsWith("ma_", StringComparison.Ordinal))      // np.ma delegates to np.*: covered via the ordinary tiers
                    .Where(f => !f.EndsWith(".host.jsonl", StringComparison.Ordinal))
                    .OrderBy(f => f, StringComparer.Ordinal)
                    .ToArray();

                var acc = new SweepAccumulator();
                ScopeAudit.Settle();
                foreach (var file in files)
                    SweepOrdinaryFile(file, null, acc);

                return new BackendSweepResult(acc.ToResult(files.Length), null, info);
            }
            finally
            {
                OpenBlasEngine.Disable();
            }
        }
    }
}
