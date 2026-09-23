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
            try
            {
                OpenBlasEngine.Enable(threads: 1);
            }
            catch (Exception e)
            {
                Assert.Inconclusive("the backend leak pass needs a loadable CBLAS/LAPACK library and none loaded here " +
                                    $"({e.GetType().Name}: {e.Message.Split('\n')[0]}). Stage one with " +
                                    "tools/fetch_openblas.py or point NUMSHARP_OPENBLAS_LIBRARY at a build.");
                return;
            }

            try
            {
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

                var r = acc.ToResult(files.Length);
                Console.WriteLine($"[scope-audit/backend] {OpenBlasEngine.Info} :: measured={r.OrdinaryMeasured} " +
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

                AssertNoUnclassifiedEscapes(r, "scope-audit/backend");
            }
            finally
            {
                OpenBlasEngine.Disable();
            }
        }
    }
}
