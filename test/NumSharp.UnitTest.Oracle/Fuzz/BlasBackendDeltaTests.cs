using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.OpenBLAS;

namespace NumSharp.UnitTest.Fuzz
{
    /// <summary>
    ///     Variation B of the differential oracle: the SAME committed corpus, replayed with
    ///     <c>NumSharp.Interop.OpenBLAS</c> installed, DEDUPLICATED against the managed run
    ///     (Variation A) so the gate surfaces only the cases the backend actually changes — and
    ///     never re-runs the tiers it cannot touch.
    ///
    ///     <para>
    ///     Why a two-pass RUNNER rather than a second whole-corpus test. Enabling the backend sets
    ///     exactly one thing — <c>engine.Blas = new OpenBlasBackend()</c> on the cached singleton
    ///     (see <see cref="OpenBlasEngine"/>) — and that property is consulted by ONLY the matrix
    ///     products (<c>dot</c>/<c>matmul</c> + the five gufuncs <c>inner</c>/<c>vdot</c>/
    ///     <c>vecdot</c>/<c>matvec</c>/<c>vecmat</c>), their compositions (<c>tensordot</c>/
    ///     <c>multi_dot</c>/<c>matrix_power</c>), the <c>ISlidingDotBackend</c> pair
    ///     (<c>correlate</c>/<c>convolve</c>), and the LAPACK factorisations — and ONLY for
    ///     float32/float64/complex128 (integer/bool are modular-exact; Decimal/Half/Char always
    ///     fall through the managed kernel). Every OTHER op in every OTHER tier runs the identical
    ///     <c>DefaultEngine</c> code in both variations, so replaying it under the backend is pure
    ///     noise. This runner loads only the three tiers that CONTAIN an affected op, keeps only the
    ///     affected cases, and reports each once — labelled by whether the backend changed its bytes.
    ///     </para>
    ///
    ///     <para>
    ///     Three claims, layered by what a host can honour:
    ///     <list type="number">
    ///       <item><b>Dedup (portable).</b> managed bytes == backend bytes ⇒ the backend changed
    ///             nothing here ⇒ the case is already gated by its managed tier and is NOT
    ///             re-reported. This is the "avoid too much noise" property: ~99% of affected cases
    ///             (every integer/bool product, every small-exact float product where summation
    ///             order cannot matter) land here.</item>
    ///       <item><b>No unexpected flips (portable).</b> The managed-vs-backend comparison is
    ///             NumSharp-vs-NumSharp, so it needs no host pin: any case whose bytes DO change must
    ///             carry a float32/float64/complex128 operand. A byte change on any other dtype means
    ///             the backend intercepted an op/dtype it must not — a hard failure on every host
    ///             that can load a CBLAS at all.</item>
    ///       <item><b>Backend ≡ NumPy on the flips (host-pinned).</b> For the changed cases the
    ///             backend must reproduce NumPy's committed bytes exactly — but those bytes came out
    ///             of a specific BLAS build at a specific thread count on a specific CPU kernel, so
    ///             this is asserted only on the host the corpus was pinned to (via
    ///             <see cref="MatmulParityPin"/>) and is Inconclusive elsewhere, exactly like
    ///             <see cref="FuzzCorpusTests.MatmulParity"/>.</item>
    ///     </list>
    ///     </para>
    ///
    ///     <para>
    ///     This is additive: it does not touch the managed tiers (<c>matmul</c>/<c>products</c>/
    ///     <c>groupa</c> keep running Variation A) nor the host-pinned <c>matmul_parity</c> tier,
    ///     which already gates the deep-float dot/matmul + gufunc byte-parity. The value here is the
    ///     dedup mechanism itself, the portable "no unexpected flips" safety net across the affected
    ///     tiers, and extending the backend byte-check to <c>tensordot</c>/<c>multi_dot</c>/
    ///     <c>matrix_power</c>/<c>convolve</c>/<c>correlate</c> wherever a case genuinely flips.
    ///     </para>
    /// </summary>
    public partial class FuzzCorpusTests
    {
        /// <summary>
        ///     The corpus ops that consult <c>TensorEngine.Blas</c> / <c>ISlidingDotBackend</c>, so
        ///     the backend can change their result. The <c>*_aat</c>/<c>*_ata</c> spellings are the
        ///     <c>a@a.T</c> / <c>a.T@a</c> shortcuts (syrk route). <c>outer</c> and <c>kron</c> are
        ///     deliberately ABSENT — they are a broadcast multiply / a tiling, not a cblas product.
        /// </summary>
        private static readonly HashSet<string> BlasAffectedOps = new()
        {
            "dot", "dot_aat", "dot_ata",
            "matmul", "matmul_aat", "matmul_ata",
            "inner", "vdot", "vecdot", "matvec", "vecmat",
            "tensordot", "multi_dot", "matrix_power",
            "convolve", "correlate",
        };

        /// <summary>
        ///     The three dtypes NumPy itself routes through cblas (matmul.c.src's
        ///     <c>#USEBLAS = 1,1,0,0,1,1</c> over FLOAT/DOUBLE/CFLOAT/CDOUBLE). Every other dtype
        ///     falls through the managed kernel under the backend too, so it can never flip.
        /// </summary>
        private static readonly HashSet<string> BlasEligibleDtypes = new()
        {
            "float32", "float64", "complex128",
        };

        /// <summary>
        ///     The ONLY committed tiers that contain a BLAS-affected op. The other ~45 are 100%
        ///     managed in both variations; loading them here would be the noise this runner exists
        ///     to avoid. (<c>matmul_parity</c> is excluded on purpose — it is already replayed
        ///     backend-on by <see cref="FuzzCorpusTests.MatmulParity"/>.)
        /// </summary>
        private static readonly string[] BlasAffectedTiers = { "matmul.jsonl", "products.jsonl", "groupa.jsonl" };

        [TestMethod]
        [TestCategory("FuzzMatrix")]
        [DoNotParallelize]   // toggles the process-global engine.Blas; must not overlap other tiers
        public void BlasBackendDelta()
        {
            // Gather the affected, comparable-as-one-array cases across the three affected tiers.
            var cases = new List<FuzzCorpus.Case>();
            foreach (var tier in BlasAffectedTiers)
                foreach (var c in FuzzCorpus.Load(tier))
                    if (c.Op != null && BlasAffectedOps.Contains(c.Op) && !c.Expects_Throw
                        && (c.Expected?.KindOrArray == "array" || c.Expected?.KindOrArray == "scalar"))
                        cases.Add(c);

            const int floor = 800;   // matmul dot/matmul (~1356) + products (~287) + groupa conv/corr (~36)
            Assert.IsTrue(cases.Count >= floor,
                $"only {cases.Count} BLAS-affected array cases across {string.Join(", ", BlasAffectedTiers)} " +
                $"(committed floor {floor}) — a truncated corpus, or the affected ops moved tiers");

            // --- Pass 1: the managed variation (backend OFF). Capture each case's NumSharp bytes. ---
            OpenBlasEngine.Disable();
            Assert.IsFalse(OpenBlasEngine.Enabled,
                "the backend leaked in from a prior test — pass 1 must run on NumSharp's managed kernels");
            var managed = new byte[cases.Count][];
            for (int i = 0; i < cases.Count; i++)
                managed[i] = TryComputeBytes(cases[i], out _);

            // Turn the backend on. The pin decides ONLY whether the vs-NumPy byte check (claim 3) is
            // valid on this host; the managed-vs-backend dedup (claims 1-2) is NumSharp-vs-NumSharp
            // and needs no pin, so a wrong-host-but-loadable CBLAS still runs them.
            var pin = MatmulParityPin.Load();
            string mismatch = pin.TryEnableParityBackend();
            bool hostPinned = mismatch == null;
            if (!OpenBlasEngine.Enabled)
            {
                try { OpenBlasEngine.Enable(); } catch { /* handled by the Enabled re-check below */ }
                if (!OpenBlasEngine.Enabled)
                    Assert.Inconclusive("NumSharp.Interop.OpenBLAS could not load a CBLAS here, so " +
                        "Variation B cannot run at all: " + mismatch);
            }

            int deduped = 0, flipped = 0;
            var failures = new List<string>();
            var documented = new Dictionary<string, int>();
            try
            {
                // --- Pass 2: the backend variation (backend ON). Dedup, then adjudicate the flips. ---
                for (int i = 0; i < cases.Count; i++)
                {
                    var c = cases[i];
                    byte[] b1 = managed[i];
                    byte[] b2 = TryComputeBytes(c, out NDArray r2);

                    // "threw" is a comparable outcome: both sides throwing (e.g. dot(int8), which the
                    // backend also declines) is a dedup, not a change.
                    bool same = (b1 == null && b2 == null)
                                || (b1 != null && b2 != null && b1.AsSpan().SequenceEqual(b2));
                    if (same) { deduped++; continue; }

                    flipped++;

                    // Claim 2 (portable): only a cblas dtype may differ between the two variations.
                    if (!c.Operands.Any(o => BlasEligibleDtypes.Contains(o.Dtype)))
                    {
                        failures.Add($"{c.Id} [{c.Layout}] {c.Op}: result CHANGED under the backend for a " +
                            $"non-cblas dtype ({string.Join(",", c.Operands.Select(o => o.Dtype))}) — the " +
                            "backend must serve only float32/float64/complex128");
                        continue;
                    }

                    // Claim 3 (host-pinned): on the corpus's own host the backend must reproduce
                    // NumPy's bytes exactly. A different BLAS build rounds differently, so elsewhere we
                    // assert only the structural flip above and leave the byte comparison to that host.
                    if (!hostPinned) continue;
                    if (b2 == null)
                    {
                        failures.Add($"{c.Id} [{c.Layout}] {c.Op}: backend THREW where the managed kernel " +
                            "produced a result");
                        continue;
                    }
                    CompareArray(c, r2, c.Expected, "[blas]", failures, documented);
                }
            }
            finally { OpenBlasEngine.Disable(); }

            // Never silent: the dedup/flip accounting is printed so a run's shape is auditable.
            Console.WriteLine($"[BlasBackendDelta] {cases.Count} BLAS-affected cases: {deduped} deduped " +
                $"(managed ≡ backend — not re-reported), {flipped} changed under the backend" +
                (hostPinned ? " (byte-checked vs NumPy)" : " — vs-NumPy byte check SKIPPED (host BLAS ≠ corpus pin)"));
            if (documented.Count > 0)
                Console.WriteLine("  documented divergences on the backend path: " +
                    string.Join("; ", documented.Select(kv => $"{kv.Value}x {kv.Key}")));

            if (failures.Count > 0)
                Assert.Fail($"{failures.Count} backend-delta failures:\n  " +
                    string.Join("\n  ", failures.Take(60)));

            if (!hostPinned)
                Assert.Inconclusive($"dedup verified — {deduped} identical, {flipped} changed, 0 unexpected — " +
                    "but this host's BLAS is not the corpus pin, so backend-vs-NumPy byte parity was not " +
                    $"asserted: {mismatch}");
        }

        /// <summary>
        ///     Reconstruct a case's operands and run its op, returning the C-contiguous result bytes,
        ///     or null if it threw / produced no comparable array. Null is a first-class outcome for
        ///     the dedup: two null passes (both threw) are identical, one null (a flip that enables or
        ///     breaks the op) is surfaced.
        /// </summary>
        private static byte[] TryComputeBytes(FuzzCorpus.Case c, out NDArray result)
        {
            result = null;
            try
            {
                var operands = new NDArray[c.Operands.Length];
                for (int i = 0; i < operands.Length; i++)
                    operands[i] = FuzzCorpus.Reconstruct(c.Operands[i]);
                if (c.Alias && operands.Length == 1)
                    operands = new[] { operands[0], operands[0] };
                result = OpRegistry.Apply(c.Op, c.Params, operands);
                return FuzzCorpus.ResultBytes(result);
            }
            catch
            {
                result = null;
                return null;
            }
        }
    }
}
