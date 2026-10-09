using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using NumSharp.Backends.Unmanaged.Pooling;
using NumSharp.Tests.Fuzz.Ownership;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     The OWNERSHIP gates: no call may dispose, release, under-count or swap an array it does not own — its caller's
    ///     operands, masks, index and value arrays, the leak fixtures, the process-wide shared arrays. Every corpus family,
    ///     the backend pass, the catalogue and the property reads check each case's arrays after each region
    ///     (<see cref="SweepAccumulator.CheckOwnership"/>); these tests turn those findings into verdicts.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Why the leak verdict cannot see it.</b> <see cref="NDArray.Dispose"/> is idempotent and every family runs one
    ///     unmeasured warm call before it measures. An op that disposes its caller's operand returns the operand's buffer
    ///     during that warm call; the measured calls then balance their own takes and returns over the disposed operand, and
    ///     the harness's own <c>Dispose</c> of the operand does nothing. <see cref="PoolBalance_CannotSeeADisposedOperand"/>
    ///     proves the blind spot; <see cref="OwnershipLedger"/> closes it by reading the ownership state itself.
    ///     </para>
    ///     <para>
    ///     <b>The GC-driven twin</b> — an operand the GC finalizes while a kernel still reads it (a lost anchor) — needs a
    ///     temporary operand and a collection inside the kernel, which no replay over held operands can produce; the lifetime
    ///     oracle (<c>Fuzz/Lifetime/</c>) gates it.
    ///     </para>
    ///     <para>
    ///     <b>Retentions</b> (a counted reference to a caller's memory that outlived the call, see
    ///     <see cref="OwnershipViolation.ReferenceRetained"/>) are the opposite direction — the caller's memory lives longer,
    ///     never shorter — and are printed, not judged: a non-array result object the harness does not dispose (an iterator,
    ///     a result struct holding views) reads the same way.
    ///     </para>
    /// </remarks>
    public partial class UndisposedIntermediateTests
    {
        /// <summary>
        ///     Over-disposal families that are known and pinned elsewhere (op key or surface id → why). Empty: every family is
        ///     gated at zero. An entry names the [OpenBugs] test that reproduces it; remove the entry when the op is fixed.
        /// </summary>
        private static readonly Dictionary<string, string> KnownOverDisposals = new(StringComparer.Ordinal)
        {
        };

        /// <summary>
        ///     The corpus gate: no case of any family (ordinary, masked-array, index — success and error paths) disposed,
        ///     released, under-counted or swapped an array it does not own, or any shared array.
        /// </summary>
        /// <exception cref="AssertFailedException">An unclassified over-disposal family, or too few cases were checked (a
        ///     replay regression that would make the gate vacuous).</exception>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        [TestCategory("ScopeAudit")]
        public void Corpus_AllOps_LeaveTheCallersArraysIntact()
        {
            var r = SharedSweep.Value;
            // Every measured, inconclusive or threw-skipped case is checked at least once. The floor sits 5% under the
            // 2026-10-09 count (669,232 cases over 126 files: 599,570 success + 67,977 error paths + 1,685 threw), so a
            // family or a stage that stops being checked trips it while corpus growth never does.
            Assert.IsTrue(r.OwnershipCases > 635_000,
                $"only {r.OwnershipCases} corpus cases had their arrays checked — the ownership checks stopped running?");
            AssertNoOverDisposal(r, "ownership");
        }

        /// <summary>
        ///     The backend twin of <see cref="Corpus_AllOps_LeaveTheCallersArraysIntact"/>: the ordinary tiers with OpenBLAS
        ///     installed, so the LAPACK family and the BLAS product seams are checked too.
        /// </summary>
        /// <exception cref="AssertInconclusiveException">No CBLAS/LAPACK library loads on this host.</exception>
        /// <exception cref="AssertFailedException">An unclassified over-disposal family.</exception>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        [TestCategory("ScopeAudit")]
        public void Corpus_BackendOps_LeaveTheCallersArraysIntact()
        {
            var run = SharedBackendSweep.Value;
            if (run.SkipReason != null)
            {
                Assert.Inconclusive("the backend ownership pass needs a loadable CBLAS/LAPACK library: " + run.SkipReason);
                return;
            }
            // The backend pass replays the ordinary tiers only (no index_ or ma_ files), so its floor is the ordinary share of
            // the corpus floor above.
            Assert.IsTrue(run.Sweep.OwnershipCases > 450_000,
                $"only {run.Sweep.OwnershipCases} backend cases had their arrays checked — the ownership checks stopped running?");
            AssertNoOverDisposal(run.Sweep, "ownership/backend");
        }

        /// <summary>
        ///     The catalogue gate: no catalogue entry (the members no corpus row reaches) damaged a fixture array or a shared
        ///     array.
        /// </summary>
        /// <exception cref="AssertFailedException">An unclassified over-disposal family, or no entry was checked.</exception>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        [TestCategory("ScopeAudit")]
        public void Catalogue_EveryEntry_LeavesTheFixturesIntact()
        {
            var r = SharedCatalogue.Value.Sweep;
            Assert.IsTrue(r.OwnershipCases > 0, "no catalogue entry had the fixture checked — the ownership checks stopped running?");
            AssertNoOverDisposal(r, "ownership/catalogue");
        }

        /// <summary>
        ///     The property-read gate: no property or field read (or round-trip write) damaged a fixture array or a shared
        ///     array.
        /// </summary>
        /// <exception cref="AssertFailedException">An unclassified over-disposal family, or no read was checked.</exception>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        [TestCategory("ScopeAudit")]
        public void EveryPropertyAndField_Read_LeavesTheFixturesIntact()
        {
            var r = SharedPropertyReads.Value.Sweep;
            Assert.IsTrue(r.OwnershipCases > 0, "no property read had the fixture checked — the ownership checks stopped running?");
            AssertNoOverDisposal(r, "ownership/property-read");
        }

        /// <summary>
        ///     The shared verdict: every over-disposal family must be in <see cref="KnownOverDisposals"/>; retentions are
        ///     printed for review.
        /// </summary>
        /// <param name="r">A runner's result.</param>
        /// <param name="label">Console tag of the gate.</param>
        /// <exception cref="AssertFailedException">An over-disposal family is not classified.</exception>
        private static void AssertNoOverDisposal(SweepResult r, string label)
        {
            var overDisposals = r.OverDisposals ?? new();
            var retentions = r.Retentions ?? new();
            Console.WriteLine($"[{label}] cases checked={r.OwnershipCases} over-disposal families={overDisposals.Count} " +
                              $"retention families={retentions.Count}");
            if (retentions.Count > 0)
                Console.WriteLine($"[{label}] retentions (a counted reference to a caller's memory outlived the call; not " +
                                  "judged):\n  " + string.Join("\n  ", retentions.OrderByDescending(k => k.Value.count).Take(40)
                                      .Select(k => $"{k.Value.count}x {k.Key}  e.g. {k.Value.file}/{k.Value.sampleId}: {k.Value.detail}")));

            var known = new List<string>();
            var unknown = new List<string>();
            foreach (var kv in overDisposals.OrderByDescending(k => k.Value.count))
            {
                string line = $"{kv.Value.count}x {kv.Key.op} [{kv.Key.kind}]  e.g. {kv.Value.file}/{kv.Value.sampleId}: {kv.Value.detail}";
                if (KnownOverDisposals.TryGetValue(kv.Key.op, out var why))
                    known.Add($"{line}  — {why}");
                else
                    unknown.Add(line);
            }
            if (known.Count > 0)
                Console.WriteLine($"[{label}] documented over-disposals:\n  " + string.Join("\n  ", known));
            if (unknown.Count > 0)
                Assert.Fail($"{unknown.Count} over-disposal families — a call disposed, released, under-counted or swapped an " +
                            "array it does not own:\n  " + string.Join("\n  ", unknown.Take(60)));
        }

        // ---------------------------------------------------------------------------------
        // Harness teeth: every violation kind is detected, a clean call reads clean, and the pool balance alone is blind
        // to a disposed operand (the reason the ledger exists).
        // ---------------------------------------------------------------------------------

        /// <summary>A fresh pooled float64 operand of 4096 elements (a whole bucket buffer, never a scalar slot).</summary>
        /// <returns>The operand; the caller disposes it.</returns>
        private static NDArray OwnershipOperand() => np.arange(4096.0);

        /// <summary>
        ///     An op that disposes its operand is caught as <see cref="OwnershipViolation.Disposed"/> and, because the operand
        ///     held the only reference, as <see cref="OwnershipViolation.Released"/> too.
        /// </summary>
        /// <exception cref="AssertFailedException">The ledger missed it.</exception>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        [TestCategory("ScopeAudit")]
        public void Ownership_Detects_DisposedOperand()
        {
            var a = OwnershipOperand();
            var ledger = OwnershipLedger.OfPositional(new[] { a });
            a.Dispose();   // the planted over-disposal
            var kinds = ledger.Check().Select(f => f.Kind).ToList();
            CollectionAssert.Contains(kinds, OwnershipViolation.Disposed);
            CollectionAssert.Contains(kinds, OwnershipViolation.Released);
        }

        /// <summary>
        ///     A reference released without a matching take, while a second reference keeps the memory alive, is caught as
        ///     <see cref="OwnershipViolation.ReferenceDropped"/> — the buffer would be freed when the caller drops its own.
        /// </summary>
        /// <exception cref="AssertFailedException">The ledger missed it.</exception>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        [TestCategory("ScopeAudit")]
        public void Ownership_Detects_DroppedReference()
        {
            var a = OwnershipOperand();
            var view = a["::2"];   // a second counted reference on the same memory
            try
            {
                var ledger = OwnershipLedger.OfPositional(new[] { a });
                a.Storage.InternalArray.Release();   // the planted over-release
                var f = ledger.Check();
                Assert.IsTrue(f.Any(x => x.Kind == OwnershipViolation.ReferenceDropped), string.Join("; ", f));
                a.Storage.InternalArray.TryAddRef();  // restore the count the plant took, so both disposes below balance
            }
            finally
            {
                view.Dispose();
                a.Dispose();
            }
        }

        /// <summary>
        ///     A forced free (<see cref="IArraySlice.DangerousFree"/>, which ignores the count) is caught as
        ///     <see cref="OwnershipViolation.Released"/>.
        /// </summary>
        /// <exception cref="AssertFailedException">The ledger missed it.</exception>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        [TestCategory("ScopeAudit")]
        public void Ownership_Detects_ForcedFree()
        {
            var a = OwnershipOperand();
            var ledger = OwnershipLedger.OfPositional(new[] { a });
            a.Storage.InternalArray.DangerousFree();   // the planted free
            var f = ledger.Check();
            Assert.IsTrue(f.Any(x => x.Kind == OwnershipViolation.Released), string.Join("; ", f));
            a.Dispose();   // a stray release on a freed block is absorbed by the ARC (the sentinel stays -1)
        }

        /// <summary>
        ///     An array moved onto other memory (its buffer swapped under it) is caught as
        ///     <see cref="OwnershipViolation.StorageReplaced"/>.
        /// </summary>
        /// <exception cref="AssertFailedException">The ledger missed it.</exception>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        [TestCategory("ScopeAudit")]
        public void Ownership_Detects_ReplacedStorage()
        {
            var a = OwnershipOperand();
            var other = OwnershipOperand();
            try
            {
                var ledger = OwnershipLedger.OfPositional(new[] { a });
                a.ReplaceData(other.Storage.InternalArray);   // the planted swap (moves a's reference onto other's memory)
                var f = ledger.Check();
                Assert.IsTrue(f.Any(x => x.Kind == OwnershipViolation.StorageReplaced), string.Join("; ", f));
            }
            finally
            {
                a.Dispose();
                other.Dispose();
            }
        }

        /// <summary>
        ///     Memory a live array reads, parked in the buffer pool (here by returning it directly, as a block adopted over
        ///     the caller's memory would on its disposal), is caught as <see cref="OwnershipViolation.PooledWhileHeld"/> even
        ///     though the array's own block is untouched.
        /// </summary>
        /// <exception cref="AssertFailedException">The pool scan missed it.</exception>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        [TestCategory("ScopeAudit")]
        public unsafe void Ownership_Detects_PooledWhileHeld()
        {
            // An odd size: its bucket is otherwise empty, so the planted return parks (a full bucket would free it instead).
            const long bytes = 4096 * 8 + 24;
            var memory = SizeBucketedBufferPool.Take(bytes);
            // The caller's array over that memory, through a NON-owning block (the pool owns the buffer in this test).
            var slice = new ArraySlice<byte>(new UnmanagedMemoryBlock<byte>((byte*)memory, bytes));
            var a = new NDArray(new UnmanagedStorage(slice, Shape.Vector(bytes)));
            try
            {
                var ledger = OwnershipLedger.OfPositional(new[] { a });
                SizeBucketedBufferPool.Return(memory, bytes);   // the planted over-disposal
                var f = ledger.Check(scanPool: true);
                Assert.IsTrue(f.Any(x => x.Kind == OwnershipViolation.PooledWhileHeld), string.Join("; ", f));
                Assert.AreEqual(0, ledger.Check().Count, "only the pool scan may see it: the array's own block is intact");
            }
            finally
            {
                // Take the buffer back before anything else can (the pool's stacks are last-in first-out), then return it
                // for good once the array over it is gone.
                var back = SizeBucketedBufferPool.Take(bytes);
                a.Dispose();
                SizeBucketedBufferPool.Return(back, bytes);
            }
        }

        /// <summary>
        ///     A view of the operand left undisposed by the call is reported as <see cref="OwnershipViolation.ReferenceRetained"/>,
        ///     which is NOT an over-disposal.
        /// </summary>
        /// <exception cref="AssertFailedException">The ledger missed it or judged it an over-disposal.</exception>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        [TestCategory("ScopeAudit")]
        public void Ownership_Reports_RetainedReference_AsNoOverDisposal()
        {
            var a = OwnershipOperand();
            NDArray leaked = null;
            try
            {
                var ledger = OwnershipLedger.OfPositional(new[] { a });
                leaked = a["1:"];   // the planted retention (a call that forgets to dispose its view)
                var f = ledger.Check();
                Assert.AreEqual(1, f.Count, string.Join("; ", f));
                Assert.AreEqual(OwnershipViolation.ReferenceRetained, f[0].Kind);
                Assert.IsFalse(f[0].IsOverDisposal);
            }
            finally
            {
                leaked?.Dispose();
                a.Dispose();
            }
        }

        /// <summary>A call that disposes only its own result reads clean.</summary>
        /// <exception cref="AssertFailedException">The ledger reported a finding for a clean call.</exception>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        [TestCategory("ScopeAudit")]
        public void Ownership_CleanCall_ReadsClean()
        {
            var a = OwnershipOperand();
            var b = OwnershipOperand();
            try
            {
                var ledger = OwnershipLedger.OfPositional(new[] { a, b });
                using (var r = np.add(a, b))
                {
                    // the result is the call's own and is disposed here
                }
                using (var v = np.transpose(a))
                {
                    // a view result takes a reference and gives it back on dispose
                }
                Assert.AreEqual(0, ledger.Check().Count);
            }
            finally
            {
                a.Dispose();
                b.Dispose();
            }
        }

        /// <summary>
        ///     The blind spot the ledger closes: an op that disposes its operand, replayed the leak sweep's way (an unmeasured
        ///     warm call, then the measured window), BALANCES the pool — the warm call already returned the operand's buffer,
        ///     the measured calls take and return only their own result, and the operand's second dispose is a no-op. The
        ///     ledger flags the same call.
        /// </summary>
        /// <exception cref="AssertFailedException">The pool balance moved (the protocol changed: re-read the class remarks),
        ///     or the ledger missed the disposal.</exception>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        [TestCategory("ScopeAudit")]
        public void PoolBalance_CannotSeeADisposedOperand()
        {
            ScopeAudit.Settle();
            var a = OwnershipOperand();
            var ledger = OwnershipLedger.OfPositional(new[] { a });

            // The planted op: computes a fresh result, then disposes the operand it was given.
            NDArray OverDisposingNegative(NDArray x)
            {
                var r = np.negative(x);
                x.Dispose();
                return r;
            }

            OverDisposingNegative(a).Dispose();   // the sweep's warm call
            var traffic = ScopeAudit.MeasureConfirmedTraffic(() => OverDisposingNegative(a).Dispose());
            if (traffic == null)
                Assert.Inconclusive("a GC landed inside every attempt — rerun");
            Assert.AreEqual(0L, traffic.Value.Escaped,
                "the pool balance moved for an op that disposes its operand: the leak sweep may now see the class itself");
            Assert.IsTrue(ledger.Check().Any(f => f.Kind == OwnershipViolation.Disposed),
                          "the ownership ledger missed the disposed operand");
        }

        /// <summary>
        ///     The shared-array watch list resolves every member it names (np.ma's three singletons and the 24 polynomial
        ///     constants) and reads clean in a clean process state.
        /// </summary>
        /// <exception cref="AssertFailedException">A shared array is missing, or one reads damaged.</exception>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        [TestCategory("ScopeAudit")]
        public void SharedArrays_AreWatched()
        {
            var all = SharedArrays.All().ToList();
            Assert.IsTrue(all.Count >= 26, $"only {all.Count} shared arrays are watched");
            Assert.IsTrue(all.All(x => x.Array is not null), "a shared array resolved to null");
            Assert.IsNull(SharedArrays.CheckOverDisposal(), "a shared array reads damaged before any op ran");
        }
    }
}
