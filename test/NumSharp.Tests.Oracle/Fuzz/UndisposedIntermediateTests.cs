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
    ///     and the NEP50 scalar-operand binary/comparison cells. Those are documented in
    ///     <see cref="KnownEscapes"/> (surfaced green, per-op leak CEILING enforced — a worsening
    ///     leak still fails) and tracked red by the [OpenBugs] pin
    ///     <see cref="KnownEscapeFamilies_AreFixed"/>; every op NOT in the registry is gated at
    ///     zero from day one.
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

            if (documented.Count > 0)
                Console.WriteLine($"[scope-audit] documented known escapes ({documentedCases} cases across " +
                                  $"{documented.Count} families; tracked by the KnownEscapeFamilies_AreFixed " +
                                  $"[OpenBugs] pin — remove KnownEscapes entries as leaks get fixed):\n  " +
                                  string.Join("\n  ", documented.Take(20)) +
                                  (documented.Count > 20 ? $"\n  … {documented.Count - 20} more families" : ""));

            if (undocumented.Count > 0)
                Assert.Fail($"{undocumented.Count} escape families ({escapedCases - documentedCases} cases) " +
                            $"not covered by KnownEscapes — NEW undisposed intermediates:\n  " +
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
        // Known escapes, pinned. [TestCategory("OpenBugs")] = known-failing repro, excluded in
        // CI, delete the category when fixed (house convention).
        // ---------------------------------------------------------------------------------

        /// <summary>
        ///     The tracking pin for the WHOLE <see cref="KnownEscapes"/> inventory: re-sweeps only
        ///     the corpus cases of registered ops and demands ZERO escapes — red until every
        ///     documented leak is fixed. Working the list down: fix an op, remove its
        ///     KnownEscapes entry (the FuzzMatrix sweep then gates it at zero forever), and when
        ///     the registry empties delete this pin.
        /// </summary>
        [TestMethod]
        [TestCategory("OpenBugs")]
        [TestCategory("ScopeAudit")]
        public void KnownEscapeFamilies_AreFixed()
        {
            var r = RunSweep(includeOp: KnownEscapes.Contains);
            PrintPerOpRollup(r.Groups);
            Assert.IsTrue(r.Measured > 5_000,
                $"known-escape pin measured only {r.Measured} cases — did the corpus or registry shrink?");
            long escapedCases = r.Groups.Sum(g => g.Value.count);
            Assert.AreEqual(0L, escapedCases,
                $"{r.Groups.Count} known escape families ({escapedCases} cases) still leak — " +
                "see the per-op rollup above; remove fixed ops from KnownEscapes as they land");
        }

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
        [TestCategory("OpenBugs")]
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
                    "the binary scalar fast path drops its dtype-cast 0-d temp (currently escaped=+1)");
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
        // Entries are surfaced green by the sweep and held red by KnownEscapeFamilies_AreFixed.
        // Remove an op's entry when its leak is fixed — from then on the sweep gates it at zero,
        // and an op leaking ABOVE its recorded ceiling fails immediately even while listed.
        // ---------------------------------------------------------------------------------

        private static class KnownEscapes
        {
            private static readonly Dictionary<string, long> CeilingByOp = new(StringComparer.Ordinal)
            {
                // reductions / scans (axis + nan + flat-with-out variants)
                ["all"] = 1, ["any"] = 1, ["sum"] = 1, ["mean"] = 1, ["nanmean"] = 1,
                ["nanstd"] = 1, ["nanvar"] = 1, ["cumsum"] = 1, ["cumprod"] = 1,
                // products
                ["matmul"] = 1, ["dot"] = 1, ["vdot"] = 2, ["vecdot"] = 3, ["matvec"] = 1,
                ["vecmat"] = 2, ["vector_norm"] = 1,
                // fft family
                ["fft"] = 1, ["ifft"] = 1, ["rfft"] = 1, ["irfft"] = 1,
                ["fftshift"] = 2, ["ifftshift"] = 2, ["fftfreq"] = 3, ["rfftfreq"] = 1,
                // diagonal / triangular / index generators
                ["tri"] = 1, ["tril"] = 3, ["triu"] = 3, ["diag"] = 1, ["diagflat"] = 2,
                ["fill_diagonal"] = 2, ["diag_indices"] = 1, ["diag_indices_from"] = 1,
                ["tril_indices"] = 3, ["triu_indices"] = 3, ["tril_indices_from"] = 3,
                ["triu_indices_from"] = 3, ["mask_indices"] = 5,
                ["indices"] = 1, ["indices_sparse"] = 3, ["ix_"] = 2, ["r_"] = 3, ["c_"] = 2,
                // manipulation
                ["trim_zeros"] = 19, ["repeat"] = 1, ["reshape"] = 1, ["rot90"] = 1, ["pad"] = 2,
                ["unravel_index"] = 2, ["argwhere"] = 1,
                // sorting / searching
                ["partition"] = 2, ["argpartition"] = 3, ["lexsort"] = 3, ["sort_complex"] = 1,
                ["searchsorted"] = 2, ["digitize"] = 1 + 2, ["bincount"] = 1,
                // elementwise with NEP50 scalar operands (0-d second operand casts dropped)
                ["add"] = 1, ["subtract"] = 1, ["multiply"] = 1, ["divide"] = 1,
                ["floor_divide"] = 1, ["mod"] = 1, ["power"] = 1,
                ["bitwise_and"] = 1, ["bitwise_or"] = 1, ["bitwise_xor"] = 1,
                ["maximum"] = 1, ["minimum"] = 1, ["fmax"] = 1, ["fmin"] = 1, ["clip"] = 1,
                ["less"] = 2, ["less_equal"] = 2, ["greater_equal"] = 2, ["not_equal"] = 2,
                // unary odds and ends
                ["reciprocal"] = 1, ["conj"] = 1, ["conjugate"] = 1, ["angle"] = 2, ["angle_deg"] = 3,
                ["modf_frac"] = 1, ["modf_int"] = 1, ["out_unary"] = 1, ["copyto_overlap"] = 1,
                // creation
                ["empty"] = 2, ["empty_like"] = 2,
                // polynomial / random
                ["poly1d_coeffs"] = 1, ["poly1d_fromroots"] = 2,
                ["rnd"] = 4, ["grnd"] = 7, ["get_state"] = 1,
            };

            public static bool Contains(string op) => CeilingByOp.ContainsKey(op);

            public static string Classify(string op, long escaped)
                => CeilingByOp.TryGetValue(op, out var ceiling) && escaped > 0 && escaped <= ceiling
                    ? $"pre-existing leak at gate landing (2026-08-26), ceiling {ceiling}/call — " +
                      "tracked by KnownEscapeFamilies_AreFixed"
                    : null;
        }

        // ---------------------------------------------------------------------------------
        // The sweep core, shared by the FuzzMatrix gate (includeOp: null = everything;
        // KnownEscapes families excused) and the OpenBugs pin (includeOp: KnownEscapes.Contains;
        // nothing excused).
        // ---------------------------------------------------------------------------------

        private sealed record SweepResult(
            long Measured, long GcInconclusive, long ThrewSkipped, long ErrorParitySkipped, int Files,
            Dictionary<(string op, string layout, long escaped), (long count, string sampleId, string file)> Groups);

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
                try
                {
                    for (int i = 0; i < operands.Length; i++)
                        operands[i] = FuzzCorpus.Reconstruct(c.Operands[i]);
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
                    // inside the confirming region, that verdict is trustworthy.
                    void Region() => DisposeResult(Invoke(c, operands), operands);
                    long? escaped = ScopeAudit.Measure(Region);
                    if (escaped is not null && escaped != 0)
                    {
                        ScopeAudit.Settle();
                        escaped = ScopeAudit.Measure(Region);
                    }
                    if (escaped == null)
                    {
                        gcInconclusive++;   // a GC landed inside every attempt — indistinguishable, never red
                        continue;
                    }

                    measured++;
                    // Hygiene: bound the backlog the sweep's own (known-leak) drops build up.
                    if (measured % 1024 == 0)
                        ScopeAudit.Settle();

                    if (escaped != 0)
                    {
                        var key = (c.Op, c.Layout ?? "?", escaped.Value);
                        groups[key] = groups.TryGetValue(key, out var g)
                            ? (g.count + 1, g.sampleId, g.file)
                            : (1, c.Id, file);
                    }
                }
                finally
                {
                    DisposeOperands(operands);
                }
            }

            return new SweepResult(measured, gcInconclusive, threwSkipped, errorParitySkipped,
                                   files.Length, groups);
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
