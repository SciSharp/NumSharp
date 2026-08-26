using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends.Unmanaged.Pooling;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     ORACLE-FREE scope gate (the MetamorphicTests pattern — no NumPy): replay the committed
    ///     corpus through <see cref="OpRegistry"/> with EVERY result disposed, and assert the
    ///     buffer pool's takes/returns balance. A surplus take is an UNDISPOSED INTERMEDIATE —
    ///     a buffer the [NDScoped] weaver / library scoping failed to dispose, reclaimable only
    ///     by a future GC + finalizer pass (the exact traffic class behind the pre-160ecbba
    ///     benchmark collapse); a surplus return is a result buffer allocated OUTSIDE the pool
    ///     (no warm reuse) whose Dispose nevertheless returns it into the pool. Both directions
    ///     fail. Value correctness is NOT asserted here — that is FuzzCorpusTests' job; this
    ///     gate only exercises the same op surface and counts buffers (see ScopeAudit).
    ///
    ///     LANDING STATE (2026-08-26): the first full sweep found pre-existing leaks in 91 ops /
    ///     ~9,900 of 102,785 measured cases — the whole axis-reduction family (all/any/sum/mean/
    ///     nan*/cum*), the product family (matmul/dot/vecdot/matvec/vecmat/vdot), fft, tri/tril/
    ///     triu/diag*, trim_zeros (up to 19 buffers per call), np.empty itself, ufunc out= paths,
    ///     and the NEP50 scalar-operand binary/comparison cells. Those were documented in
    ///     <see cref="KnownEscapes"/> (surfaced green, per-op leak CEILING enforced — a worsening
    ///     leak still fails) and worked down to ZERO across four fix waves (see the ledger on
    ///     <see cref="KnownEscapes"/>); the registry is now EMPTY, so every op is gated at zero and
    ///     the former tracking pin has been retired.
    /// </summary>
    [TestClass]
    [DoNotParallelize]   // ScopeAudit reads process-global pool counters; nothing may run beside it.
    public class UndisposedIntermediateTests
    {
        // ---------------------------------------------------------------------------------
        // The corpus sweep (FuzzMatrix gate).
        // ---------------------------------------------------------------------------------

        [TestMethod]
        [TestCategory("FuzzMatrix")]
        [TestCategory("ScopeAudit")]
        public void Corpus_AllOps_LeaveNoUndisposedIntermediates()
        {
            var r = RunSweep(includeOp: null);

            Console.WriteLine($"[scope-audit] measured={r.Measured} gcInconclusive={r.GcInconclusive} " +
                              $"threwSkipped={r.ThrewSkipped} errorParitySkipped={r.ErrorParitySkipped} " +
                              $"files={r.Files}");
            PrintPerOpRollup(r.Groups);
            PrintBypassRollup(r.Bypasses);

            // Non-vacuity: a schema/skip regression must not silently gate nothing.
            Assert.IsTrue(r.Measured > 50_000,
                $"scope audit measured only {r.Measured} cases — corpus schema or skip-logic regression?");

            // Classify each escape family: documented (a KnownEscapes entry within its recorded
            // per-op ceiling) is surfaced but green — the RunCorpus "documented divergences"
            // pattern; anything unclassified — a new leaking op, OR a known op leaking MORE
            // buffers per call than it did at landing — fails.
            var documented = new List<string>();
            var undocumented = new List<string>();
            long documentedCases = 0, escapedCases = r.Groups.Sum(g => g.Value.count);
            foreach (var kv in r.Groups.OrderByDescending(k => k.Value.count))
            {
                string line = $"{kv.Value.count}x {kv.Key.op} [{kv.Key.layout}] escaped={kv.Key.escaped}" +
                              $"  e.g. {kv.Value.file}/{kv.Value.sampleId}";
                string reason = KnownEscapes.Classify(kv.Key.op, kv.Key.escaped);
                if (reason != null)
                {
                    documentedCases += kv.Value.count;
                    documented.Add($"{line}  — {reason}");
                }
                else
                    undocumented.Add(line);
            }

            // Bypass families: fresh result, zero bucketed-pool traffic — allocated and freed
            // outside the pool. Same classify-or-fail treatment via KnownBypass.
            long bypassCases = r.Bypasses.Sum(b => b.Value.count), bypassDocumentedCases = 0;
            var bypassDocumented = new List<string>();
            foreach (var kv in r.Bypasses.OrderByDescending(k => k.Value.count))
            {
                string line = $"{kv.Value.count}x {kv.Key.op} [{kv.Key.layout}] POOL-BYPASS " +
                              $"({kv.Value.sampleBytes} B fresh result, zero pool traffic)  " +
                              $"e.g. {kv.Value.file}/{kv.Value.sampleId}";
                if (KnownBypassByDesign.TryGetValue(kv.Key.op, out var why) ||
                    KnownBypassDebt.TryGetValue(kv.Key.op, out why))
                {
                    bypassDocumentedCases += kv.Value.count;
                    bypassDocumented.Add($"{line}  — {why}");
                }
                else
                    undocumented.Add(line);
            }

            if (documented.Count > 0)
                Console.WriteLine($"[scope-audit] documented known escapes ({documentedCases} cases across " +
                                  $"{documented.Count} families still in the KnownEscapes registry — " +
                                  $"remove each op's entry as its leak is fixed):\n  " +
                                  string.Join("\n  ", documented.Take(20)) +
                                  (documented.Count > 20 ? $"\n  … {documented.Count - 20} more families" : ""));
            if (bypassDocumented.Count > 0)
                Console.WriteLine($"[scope-audit] documented known pool bypasses ({bypassDocumentedCases} cases):\n  " +
                                  string.Join("\n  ", bypassDocumented.Take(20)) +
                                  (bypassDocumented.Count > 20 ? $"\n  … {bypassDocumented.Count - 20} more families" : ""));

            if (undocumented.Count > 0)
                Assert.Fail($"{undocumented.Count} families " +
                            $"({escapedCases - documentedCases + bypassCases - bypassDocumentedCases} cases) not covered " +
                            $"by KnownEscapes/KnownBypass — NEW undisposed intermediates or pool bypasses:\n  " +
                            string.Join("\n  ", undocumented.Take(40)));
        }

        // ---------------------------------------------------------------------------------
        // Harness teeth — prove the detector detects, in both directions, and reads zero on
        // a clean op. Without these, a counters-accounting bug reads as "everything clean".
        // ---------------------------------------------------------------------------------

        [TestMethod]
        [TestCategory("FuzzMatrix")]
        [TestCategory("ScopeAudit")]
        public void Harness_Detects_AbandonedTake()
        {
            ScopeAudit.Settle();
            var taken = new List<IntPtr>();   // each attempt escapes exactly one buffer
            long? escaped = ScopeAudit.Measure(() => taken.Add(SizeBucketedBufferPool.Take(1024)));
            foreach (var p in taken)
                SizeBucketedBufferPool.Return(p, 1024);   // balance the pool after the verdict
            Assert.AreEqual(1L, escaped, "a Take with no Return inside the region must read as escaped=+1");
        }

        [TestMethod]
        [TestCategory("FuzzMatrix")]
        [TestCategory("ScopeAudit")]
        public unsafe void Harness_Detects_ForeignReturn()
        {
            ScopeAudit.Settle();
            long? escaped = ScopeAudit.Measure(() =>
            {
                // A NativeMemory buffer the pool never handed out, returned into it — the
                // pool-bypass direction (it pools or frees it; either way it counts a return).
                var p = (IntPtr)NativeMemory.Alloc(1024);
                SizeBucketedBufferPool.Return(p, 1024);
            });
            Assert.AreEqual(-1L, escaped, "a Return with no Take inside the region must read as escaped=-1");
        }

        [TestMethod]
        [TestCategory("FuzzMatrix")]
        [TestCategory("ScopeAudit")]
        public void Harness_Detects_FullPoolBypass()
        {
            // np.frombuffer wraps caller-owned managed memory: a fresh (non-operand) NDArray with
            // ZERO bucketed-pool traffic — bypass BY DESIGN, which makes it the deterministic
            // teeth for the bypass signature (takes==0, returns==0, fresh result > a scalar slot).
            ScopeAudit.Settle();
            var buf = new byte[8192];
            long freshBytes = 0;
            var traffic = ScopeAudit.MeasureTraffic(() =>
            {
                freshBytes = 0;
                var r = np.frombuffer(buf, NumSharp.NPTypeCode.Double);
                freshBytes = FreshResultBytes(r, Array.Empty<NumSharp.NDArray>(),
                                              new List<(ulong, ulong)>());
                r.Dispose();
            });
            Assert.IsNotNull(traffic, "GC interfered on every attempt — rerun");
            Assert.AreEqual(0L, traffic.Value.Takes, "frombuffer must not touch the bucketed pool");
            Assert.AreEqual(0L, traffic.Value.Returns, "frombuffer must not return into the bucketed pool");
            Assert.AreEqual(8192L, freshBytes, "the wrap must read as a fresh non-pool result");
        }

        [TestMethod]
        [TestCategory("FuzzMatrix")]
        [TestCategory("ScopeAudit")]
        public void Harness_CleanDisposedOp_ReadsZero()
        {
            ScopeAudit.Settle();
            var tmp = np.arange(1000);
            var a = tmp.astype(np.float64);
            var b = a.copy();
            try
            {
                long? escaped = ScopeAudit.Measure(() =>
                {
                    using var r = np.add(a, b);
                });
                Assert.AreEqual(0L, escaped, "a fully-disposed np.add must balance takes and returns");
            }
            finally
            {
                b.Dispose();
                a.Dispose();
                tmp.Dispose();
            }
        }

        // ---------------------------------------------------------------------------------
        // Mechanism regression pins. The whole KnownEscapes inventory has now been worked down to
        // ZERO across four fix waves (see KnownEscapes below), so its tracking pin
        // KnownEscapeFamilies_AreFixed has been RETIRED — Corpus_AllOps gates every op at zero, and
        // the registry is empty. What remains is the one root-caused MECHANISM pin, kept as a live
        // regression guard now that its leak is fixed.
        // ---------------------------------------------------------------------------------

        /// <summary>
        ///     The engine's NEP50 scalar fast path drops its dtype-cast temp:
        ///     DefaultEngine.BinaryOp.cs does `rhs = Cast(rhs, resultType, copy: true)`,
        ///     REASSIGNING THE PARAMETER, so the fresh 0-d temp (pooled buffer + finalizable
        ///     graph) is unreachable on method exit — one escape per `array op scalar` call
        ///     whose scalar needs a dtype cast (float64[] + int 5; same-dtype scalar ops are
        ///     clean). The fix captures the temp and disposes it at every exit. The operand here
        ///     is a HELD prewrapped scalar (harness-owned) precisely so the region contains ONLY
        ///     the library's own drop — a literal `a + 5` would also count the CALL-SITE's
        ///     implicit int→NDArray wrapper, which no library fix can dispose.
        /// </summary>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        [TestCategory("ScopeAudit")]
        public void BinaryScalarCastTemp_IsDisposed()
        {
            ScopeAudit.Settle();
            var tmp = np.arange(1000);
            var a = tmp.astype(np.float64);
            var five = NumSharp.NDArray.Scalar(5);   // int32 0-d: forces the scalar-cast path
            try
            {
                long? escaped = ScopeAudit.Measure(() =>
                {
                    using var r = np.add(a, five);
                });
                Assert.AreEqual(0L, escaped,
                    "the binary scalar fast path must dispose its dtype-cast 0-d temp (regression pin)");
            }
            finally
            {
                five.Dispose();
                a.Dispose();
                tmp.Dispose();
            }
        }

        // ---------------------------------------------------------------------------------
        // Known escape families. Each op below LEAKED at gate landing (2026-08-26, full-corpus
        // sweep; per-op ceiling = the worst confirmed buffers-escaped-per-call observed then).
        // Entries are surfaced green by the sweep while listed. Remove an op's entry when its leak
        // is fixed — from then on the sweep gates it at zero, and an op leaking ABOVE its recorded
        // ceiling fails immediately even while listed. The registry has been worked down to EMPTY
        // (four fix waves, ledger below), so Corpus_AllOps now gates every op at zero.
        // ---------------------------------------------------------------------------------

        private static class KnownEscapes
        {
            private static readonly Dictionary<string, long> CeilingByOp = new(StringComparer.Ordinal)
            {
                // NOTE (2026-08-26, fix wave): the composition families whose intermediates the
                // [NDScoped] weaver now reclaims have been REMOVED as they landed at zero (verified by
                // the full-corpus sweep, which from then gates each at zero forever): the whole
                // reduction/scan family (all/any/sum/mean/nan*/cum*), the product family
                // (matmul/dot/vdot/vecdot/matvec/vecmat/vector_norm), the FULL fft family
                // (fft/ifft/rfft/irfft — the 1-D transforms simply weren't marked — plus the
                // fft*shift/*freq helpers), the diagonal/triangular constructors (tri/tril/triu/diag/
                // diagflat/fill_diagonal), r_/c_, trim_zeros, angle/angle_deg, poly1d_coeffs, and the
                // NEP50 scalar-operand elementwise cells (add/subtract/multiply/divide/floor_divide/
                // mod/power/bitwise_*/maximum/minimum/fmax/fmin/clip — the engine now disposes the
                // reassigned scalar-cast temp AND the copy('F') C-contig original, mechanisms 1 & 3).
                // What remains below is the still-leaking inventory.
                //
                // SECOND fix wave (2026-08-26): the sorting/manipulation/index-generator families
                // cleared. The MULTI-OUTPUT ones were HARNESS compositions dropping un-tested
                // intermediates, disposed in OpRegistry — partition/argpartition (the `part`/kth-index
                // gather litter), modf_frac/int (the un-selected tuple half), and the tuple-selection
                // generators tril/triu_indices(_from)/mask_indices/unravel_index/ix_ (the un-picked
                // coordinate arrays, via PickAndDisposeRest). The SINGLE-CALL ones are woven
                // [NDScoped]: repeat/reshape/rot90/sort_complex/searchsorted/digitize/bincount/indices/
                // indices_sparse/argwhere/lexsort/diag_indices(_from)/reciprocal/conj/conjugate, and
                // out_unary cleared collaterally. What remains needs deeper per-op forensics — the
                // ambient scope cannot see it (raw kernel scratch, or a dtype/shape-specific temp), or
                // it is a caller-side lifetime the library op does not own:
                //
                // THIRD fix wave (2026-08-26): the raw-scratch and field-egress families cleared where
                // NDScope provably could not reach, via targeted disposal (and one hand-written scope):
                //   * copyto_overlap + pad — the shared NDIter.Copy(UnmanagedStorage) overlap clone
                //     (`src = src.Clone()`) is a bare storage no [NDScoped] scope tracks; its buffer is
                //     now returned to the pool in a finally (pad's reflect/wrap/edge modes copy
                //     padded[dst]=padded[src], so they hit the same clone).
                //   * empty + empty_like — the leak was in ndarray.fill: CoerceFillValue's integer path
                //     dropped the 1-element source + its astype cast; both now disposed.
                //   * poly1d_fromroots — the ctor's np.poly→trim_zeros(view) chain dropped np.poly's
                //     base; a hand-written NDScope yields the field via Returns and reclaims the rest
                //     (the [NDScoped] weaver can't express a ctor's field egress).
                //   * less / less_equal / greater_equal / not_equal — the residual copy('F')-path drop
                //     (mixed-dtype + mixed-layout operands force an F-contig result): the engine's
                //     `fResult.AsGeneric<bool>()` wraps the copy('F') output in a NEW NDArray<bool>,
                //     orphaning fResult's own ARC reference. Now ReferenceEquals-gate-disposed at both
                //     compare sites (ExecuteComparisonOp + TryExecuteComparisonOpViaNDIter).
                //
                // FOURTH fix wave (2026-08-26): the random samplers cleared — the last families.
                //   * rnd / grnd — the leaking distributions are LIBRARY compositions: gamma
                //     (scale*Marsaglia), lognormal (exp(normal)), f, and RandomState+Generator choice
                //     (cumsum/searchsorted chain) are now woven [NDScoped]; dirichlet's alpha copy is a
                //     RAW UnmanagedMemoryBlock (a bare pooled buffer no [NDScoped] scope can track),
                //     freed in a finally. The HARNESS scaffolding — the draws>1 multi-draw loops and
                //     grnd's per-iteration probability array — is disposed in OpRegistry (NDScope is a
                //     LIBRARY mechanism and cannot reach the test harness's own loop; the modf/partition
                //     precedent).
                //   * get_state — the harness discards a warm-up random_sample(draws); now disposed in
                //     OpRegistry.
                //
                // The registry is now EMPTY: every op is gated at zero by Corpus_AllOps
                // (KnownEscapes.Classify returns null for all, so any surviving escape fails as an
                // undocumented NEW leak).
            };

            public static bool Contains(string op) => CeilingByOp.ContainsKey(op);

            public static string Classify(string op, long escaped)
                => CeilingByOp.TryGetValue(op, out var ceiling) && escaped > 0 && escaped <= ceiling
                    ? $"pre-existing leak at gate landing (2026-08-26), ceiling {ceiling}/call — " +
                      "still in the KnownEscapes registry (fix and remove its entry)"
                    : null;
        }

        // ---------------------------------------------------------------------------------
        // Known pool-BYPASS families: ops whose fresh result reaches the caller with zero
        // bucketed-pool traffic. Two classes with different lifecycles:
        //   BY DESIGN — the result wraps caller memory or a parsed managed array, so no native
        //   allocation exists to route through a pool; documented green, never pin-tracked.
        //   DEBT — a real native alloc outside the pool (cold alloc + first-touch faults per
        //   call, no warm reuse); documented green in the sweep until routed through the pool (a
        //   non-empty KnownBypassDebt would fail Corpus_AllOps as an undocumented bypass).
        // Unlisted ops are gated at zero-bypass from day one. The landing sweep found ONLY the
        // by-design I/O wrap class (their source files carry zero raw-alloc sites — see
        // NativeAllocationChokepointTests — so by elimination the results are managed-backed);
        // internal alloc+free scratch (NDIter buffers, bincount's table) is invisible to this
        // runtime check by construction and is gated statically instead.
        // ---------------------------------------------------------------------------------

        private static readonly Dictionary<string, string> KnownBypassByDesign = new(StringComparer.Ordinal)
        {
            ["frombuffer"] = "by design — zero-copy wrap of the caller's buffer; no allocation occurs",
            ["fromfile"] = "by design — result constructed from the parsed managed array (no native alloc)",
            ["loadtxt"] = "by design — result constructed from the parsed managed array (no native alloc)",
        };

        private static readonly Dictionary<string, string> KnownBypassDebt = new(StringComparer.Ordinal)
        {
            // none at landing — the runtime sweep found no native-allocating bypass reaching a result
        };

        // ---------------------------------------------------------------------------------
        // The sweep core. Driven by the FuzzMatrix gate with includeOp: null (everything;
        // KnownEscapes families excused within their ceiling). The includeOp filter is retained for
        // a future focused re-sweep; the KnownEscapes-only tracking pin that used it has been retired
        // now that the registry is empty.
        // ---------------------------------------------------------------------------------

        private sealed record SweepResult(
            long Measured, long GcInconclusive, long ThrewSkipped, long ErrorParitySkipped, int Files,
            Dictionary<(string op, string layout, long escaped), (long count, string sampleId, string file)> Groups,
            Dictionary<(string op, string layout), (long count, string sampleId, string file, long sampleBytes)> Bypasses);

        private static SweepResult RunSweep(Func<string, bool> includeOp)
        {
            var corpusDir = Path.Combine(AppContext.BaseDirectory, "Fuzz", "corpus");
            var files = Directory.GetFiles(corpusDir, "*.jsonl")
                .Select(Path.GetFileName)
                .Where(f => !f.StartsWith("index_", StringComparison.Ordinal))    // index oracle: different case schema (IndexOracleTests)
                .Where(f => !f.EndsWith(".host.jsonl", StringComparison.Ordinal)) // host pins: not case files
                .OrderBy(f => f, StringComparer.Ordinal)
                .ToArray();
            Assert.IsTrue(files.Length >= 40, $"only {files.Length} corpus files found — corpus copy regression?");

            long measured = 0, gcInconclusive = 0, threwSkipped = 0, errorParitySkipped = 0;
            var groups = new Dictionary<(string op, string layout, long escaped),
                                        (long count, string sampleId, string file)>();
            var bypasses = new Dictionary<(string op, string layout),
                                          (long count, string sampleId, string file, long sampleBytes)>();

            ScopeAudit.Settle();   // drain finalizer backlog left by earlier (undisposing) test classes

            foreach (var file in files)
            foreach (var c in FuzzCorpus.Load(file))
            {
                if (includeOp != null && !includeOp(c.Op))
                    continue;

                // Error-parity cases: NumPy raised here and NumSharp must throw. A throwing op's
                // scope hygiene is worth gating eventually, but a thrown path has no result to
                // dispose and its verdict belongs to FuzzCorpusTests.CheckError — skip.
                if (c.Expects_Throw)
                {
                    errorParitySkipped++;
                    continue;
                }

                var operands = new NumSharp.NDArray[c.Operands.Length];
                var ranges = new List<(ulong lo, ulong hi)>(c.Operands.Length);
                try
                {
                    for (int i = 0; i < operands.Length; i++)
                    {
                        operands[i] = FuzzCorpus.Reconstruct(c.Operands[i]);
                        // The operand's whole base-buffer byte range, for result-freshness checks:
                        // a result whose data pointer lands inside any of these is a VIEW, not a
                        // fresh allocation.
                        var o = c.Operands[i];
                        if (operands[i].size > 0 && o.BufferSize > 0)
                        {
                            int isz = FuzzCorpus.DtypeToTC(o.Dtype).SizeOf();
                            ulong lo = Addr(operands[i]) - (ulong)(o.Offset * isz);
                            ranges.Add((lo, lo + (ulong)(o.BufferSize * isz)));
                        }
                    }
                }
                catch
                {
                    threwSkipped++;
                    continue;
                }

                if (c.Alias && operands.Length == 1)
                    operands = new[] { operands[0], operands[0] };

                try
                {
                    // Warm invocation, un-measured: absorbs one-time retained allocations
                    // (FFT plan/twiddle caches, emitted-kernel warmup) that would otherwise
                    // read as escapes on the first use of an (op, size).
                    try
                    {
                        DisposeResult(Invoke(c, operands), operands);
                    }
                    catch
                    {
                        threwSkipped++;   // value/throw divergences are FuzzCorpusTests' verdict, not ours
                        continue;
                    }

                    // SCREEN, then CONFIRM. The sweep runs over a library with known leaks, so
                    // escaped buffers accumulate; pacing/natural GCs collect them and the
                    // finalizer thread drains RETURNS asynchronously across later regions —
                    // invisible to GC-count detection (a drain is not a collection) and capable
                    // of huge spurious negatives / masked positives. A non-zero screen is
                    // therefore re-measured after a Settle: with the queue drained and no GC
                    // inside the confirming region, that verdict is trustworthy. (The bypass
                    // verdict needs no confirm: drain interference adds RETURNS, which makes
                    // escaped negative and routes through the confirm path; takes==0 && escaped==0
                    // implies returns==0 — arithmetically drain-free.)
                    long freshBytes = 0;
                    void Region()
                    {
                        freshBytes = 0;   // re-executed on retry — recompute, don't accumulate
                        object res = Invoke(c, operands);
                        freshBytes = FreshResultBytes(res, operands, ranges);
                        DisposeResult(res, operands);
                    }
                    var traffic = ScopeAudit.MeasureTraffic(Region);
                    if (traffic is not null && traffic.Value.Escaped != 0)
                    {
                        ScopeAudit.Settle();
                        traffic = ScopeAudit.MeasureTraffic(Region);
                    }
                    if (traffic == null)
                    {
                        gcInconclusive++;   // a GC landed inside every attempt — indistinguishable, never red
                        continue;
                    }

                    measured++;
                    // Hygiene: bound the backlog the sweep's own (known-leak) drops build up.
                    if (measured % 1024 == 0)
                        ScopeAudit.Settle();

                    long escaped = traffic.Value.Escaped;
                    if (escaped != 0)
                    {
                        var key = (c.Op, c.Layout ?? "?", escaped);
                        groups[key] = groups.TryGetValue(key, out var g)
                            ? (g.count + 1, g.sampleId, g.file)
                            : (1, c.Id, file);
                    }
                    else if (traffic.Value.Takes == 0 && freshBytes > 0)
                    {
                        // FULL POOL BYPASS: the op handed back a fresh result (not a view of any
                        // operand, larger than a scalar-pool slot) yet the bucketed pool saw
                        // ZERO traffic — the buffer was allocated AND freed outside it, paying a
                        // cold NativeMemory alloc + first-touch faults on every call with no
                        // warm reuse (the allocator tax the pool exists to remove).
                        var key = (c.Op, c.Layout ?? "?");
                        bypasses[key] = bypasses.TryGetValue(key, out var b)
                            ? (b.count + 1, b.sampleId, b.file, b.sampleBytes)
                            : (1, c.Id, file, freshBytes);
                    }
                }
                finally
                {
                    DisposeOperands(operands);
                }
            }

            return new SweepResult(measured, gcInconclusive, threwSkipped, errorParitySkipped,
                                   files.Length, groups, bypasses);
        }

        /// <summary>Layout multiplies families, but a leak is a property of an op's code path —
        /// the op-level view is the actionable inventory.</summary>
        private static void PrintPerOpRollup(
            Dictionary<(string op, string layout, long escaped), (long count, string sampleId, string file)> groups)
        {
            if (groups.Count == 0)
                return;
            Console.WriteLine("[scope-audit] escapes by op:\n  " + string.Join("\n  ",
                groups.GroupBy(g => g.Key.op)
                      .OrderByDescending(g => g.Sum(x => x.Value.count))
                      .Select(g => $"{g.Key}: {g.Sum(x => x.Value.count)} cases / {g.Count()} families / " +
                                   $"escaped {g.Min(x => x.Key.escaped)}..{g.Max(x => x.Key.escaped)}")));
        }

        private static void PrintBypassRollup(
            Dictionary<(string op, string layout), (long count, string sampleId, string file, long sampleBytes)> bypasses)
        {
            if (bypasses.Count == 0)
                return;
            Console.WriteLine("[scope-audit] pool bypasses by op:\n  " + string.Join("\n  ",
                bypasses.GroupBy(g => g.Key.op)
                        .OrderByDescending(g => g.Sum(x => x.Value.count))
                        .Select(g => $"{g.Key}: {g.Sum(x => x.Value.count)} cases / {g.Count()} families / " +
                                     $"e.g. {g.First().Value.sampleBytes} B fresh result")));
        }

        // ---------------------------------------------------------------------------------
        // Result-freshness inspection (the pool-bypass side).
        // ---------------------------------------------------------------------------------

        /// <summary>
        ///     Total bytes of result arrays that are FRESH allocations: not an operand instance,
        ///     not a view into any operand's base buffer (data pointer inside its byte range), and
        ///     larger than a scalar-pool slot (<see cref="ScalarSlotBytes"/> — 0-d/tiny results
        ///     ride the separate StackedMemoryPool, which these counters cannot see and which IS
        ///     a pool). Fresh bytes with zero bucketed-pool takes = the full-bypass signature.
        /// </summary>
        private static long FreshResultBytes(object result, NumSharp.NDArray[] operands,
                                             List<(ulong lo, ulong hi)> operandRanges)
        {
            switch (result)
            {
                case null:
                    return 0;
                case NumSharp.NDArray nd:
                    return FreshBytesOf(nd, operands, operandRanges);
                case NumSharp.NDArray[] tuple:
                {
                    long total = 0;
                    var seen = new HashSet<object>(ReferenceEqualityComparer.Instance);
                    foreach (var slot in tuple)
                        if (slot is not null && seen.Add(slot))
                            total += FreshBytesOf(slot, operands, operandRanges);
                    return total;
                }
                default:
                    return 0;
            }
        }

        /// <summary>StackedMemoryPool.SingleSize — the max dtype width (Complex/Decimal, 16 B).</summary>
        private const int ScalarSlotBytes = 16;

        private static long FreshBytesOf(NumSharp.NDArray nd, NumSharp.NDArray[] operands,
                                         List<(ulong lo, ulong hi)> operandRanges)
        {
            foreach (var op in operands)
                if (ReferenceEquals(op, nd))
                    return 0;
            long bytes = nd.size * nd.typecode.SizeOf();
            if (bytes <= ScalarSlotBytes)
                return 0;
            ulong a = Addr(nd);
            foreach (var (lo, hi) in operandRanges)
                if (a >= lo && a < hi)
                    return 0;   // a view into an operand's buffer — took nothing
            return bytes;
        }

        private static unsafe ulong Addr(NumSharp.NDArray nd) => (ulong)(byte*)nd.Address;

        // ---------------------------------------------------------------------------------
        // Plumbing.
        // ---------------------------------------------------------------------------------

        private static object Invoke(FuzzCorpus.Case c, NumSharp.NDArray[] operands)
            => c.Op == "grnd"
                ? OpRegistry.GeneratorDraw(c.Params)
                : OpRegistry.Invoke(c.Expected.KindOrArray, c.Op, c.Params, operands);

        /// <summary>
        ///     Dispose every distinct NDArray in an op result exactly once — EXCEPT any that IS an
        ///     operand instance (asanyarray-style same-reference returns; the operand pass owns
        ///     those). Views of operands dispose safely (ARC: the base keeps the buffer alive, so
        ///     no return is counted — correct, a view took nothing). Text/dtype results carry
        ///     nothing disposable.
        /// </summary>
        private static void DisposeResult(object result, NumSharp.NDArray[] operands)
        {
            switch (result)
            {
                case null:
                    return;
                case NumSharp.NDArray nd:
                    DisposeUnlessOperand(nd, operands);
                    return;
                case NumSharp.NDArray[] tuple:
                    var seen = new HashSet<object>(ReferenceEqualityComparer.Instance);
                    foreach (var slot in tuple)
                        if (slot is not null && seen.Add(slot))
                            DisposeUnlessOperand(slot, operands);
                    return;
                default:
                    return;   // string (text kind) / NPTypeCode (dtype kind)
            }
        }

        private static void DisposeUnlessOperand(NumSharp.NDArray nd, NumSharp.NDArray[] operands)
        {
            foreach (var op in operands)
                if (ReferenceEquals(op, nd))
                    return;
            nd.Dispose();
        }

        private static void DisposeOperands(NumSharp.NDArray[] operands)
        {
            var seen = new HashSet<object>(ReferenceEqualityComparer.Instance);   // Alias cases repeat a reference
            foreach (var op in operands)
                if (op is not null && seen.Add(op))
                    op.Dispose();
        }
    }
}
