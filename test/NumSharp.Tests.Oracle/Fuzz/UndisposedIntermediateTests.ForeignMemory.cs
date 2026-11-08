using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using NumSharp.Backends.Unmanaged.Pooling;
using NumSharp.Tests.Fuzz.Ownership;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     The kinds of memory an operand can live in. The corpus rebuilds every operand as a pinned managed copy that
    ///     NumSharp owns; real callers hand NumSharp memory it does NOT own, and every kind reaches a different release path
    ///     (pool return, unpin, a caller's release action, nothing at all).
    /// </summary>
    internal enum ForeignKind
    {
        /// <summary>NumSharp's own pooled buffer (<c>np.empty</c>): the kind every library-built array has. Over-releasing it
        ///     returns the caller's buffer to the pool, where the next allocation of its size takes it.</summary>
        Pooled,

        /// <summary>A caller's managed <c>T[]</c> wrapped without a copy (<c>new NDArray(T[], Shape)</c>): a pin NumSharp
        ///     takes and must drop exactly when the last array over it goes (early, and the GC may move the array).</summary>
        PinnedArray,

        /// <summary>A caller's <c>byte[]</c> laid out by <c>np.ndarray(shape, dtype, buffer: bytes)</c>: a block whose release
        ///     action unpins the bytes.</summary>
        ByteBuffer,

        /// <summary>A caller's <c>byte[]</c> wrapped by <c>np.frombuffer</c>.</summary>
        FromBuffer,

        /// <summary>A caller's unmanaged memory wrapped WITHOUT ownership (a non-owning block): NumSharp must never free,
        ///     pool or adopt it — the caller frees it.</summary>
        RawPointer,

        /// <summary>A caller's unmanaged memory handed over with a release action (an owning external block): released
        ///     exactly once, after the last array over it.</summary>
        ExternalRelease,
    }

    /// <summary>
    ///     The FOREIGN-MEMORY ownership gate: corpus cases replayed with every operand rebuilt over memory NumSharp does not
    ///     own (<see cref="ForeignKind"/>) — same bytes, shape, strides and offset — checked by the ownership ledger, by the
    ///     pool scan, and by each kind's own release accounting.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>What it adds to <see cref="Corpus_AllOps_LeaveTheCallersArraysIntact"/>.</b> That gate checks every corpus case,
    ///     but its operands are all one kind (pinned managed copies NumSharp itself made). An op that treats foreign memory
    ///     as its own — returns a caller's unmanaged block to the pool, wraps a caller's pinned array in a NEW pooled block,
    ///     drops a pin or a memory map early, calls a caller's release action twice — is invisible there.
    ///     </para>
    ///     <para>
    ///     <b>Sampling.</b> One case per (op, layout, operand-0 dtype) of every ordinary tier, each under every kind: the
    ///     release paths depend on the memory kind, the op and the layout, not on the values. Error paths are replayed too
    ///     (a raising op must not have released anything either).
    ///     </para>
    ///     <para>
    ///     <b>What each case asserts.</b> After the call and the disposal of its result: the ledger finds no over-disposal
    ///     and the pool holds no buffer inside any operand's memory. After the harness disposed the operands: a kind with
    ///     a release action was released at most once (an External block's action), and the pool still holds no buffer inside
    ///     a caller's unmanaged memory (only then does the harness free it).
    ///     </para>
    /// </remarks>
    public partial class UndisposedIntermediateTests
    {
        /// <summary>What the foreign-memory pass observed.</summary>
        /// <param name="Cases">Case runs (one per sampled case and kind).</param>
        /// <param name="ByKind">Case runs per kind.</param>
        /// <param name="Findings">Over-disposal families keyed (op, kind of memory, violation) → (count, sample, detail).</param>
        /// <param name="Skipped">Sampled cases a kind could not rebuild (an operand dtype the kind does not take), per kind.</param>
        internal sealed record ForeignSweepResult(
            long Cases, Dictionary<ForeignKind, long> ByKind,
            Dictionary<(string op, ForeignKind kind, string violation), (long count, string sampleId, string detail)> Findings,
            Dictionary<ForeignKind, long> Skipped);

        /// <summary>The one foreign-memory pass of the process (it replays a sample of the corpus under six kinds).</summary>
        internal static readonly Lazy<ForeignSweepResult> SharedForeignSweep =
            new(RunForeignSweep, System.Threading.LazyThreadSafetyMode.ExecutionAndPublication);

        /// <summary>
        ///     The foreign-memory gate: no sampled corpus case, under any memory kind, disposed, released, under-counted,
        ///     swapped or pooled an operand's memory, or released a caller's memory twice.
        /// </summary>
        /// <exception cref="AssertFailedException">A finding, or the pass replayed too few cases (vacuous).</exception>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        [TestCategory("ScopeAudit")]
        public void Corpus_ForeignOperands_StayTheCallers()
        {
            var r = SharedForeignSweep.Value;
            Console.WriteLine($"[ownership/foreign] case runs={r.Cases} " +
                              string.Join(" ", r.ByKind.OrderBy(k => k.Key).Select(k => $"{k.Key}={k.Value}")) +
                              " skipped: " + string.Join(" ", r.Skipped.OrderBy(k => k.Key).Select(k => $"{k.Key}={k.Value}")));

            // Every kind must have replayed a real share of the sample: a kind whose builder started failing would
            // otherwise gate nothing.
            foreach (ForeignKind kind in Enum.GetValues(typeof(ForeignKind)))
                Assert.IsTrue(r.ByKind.GetValueOrDefault(kind) > 5_000,
                    $"only {r.ByKind.GetValueOrDefault(kind)} case runs used {kind} operands — the builder or the sample broke?");

            if (r.Findings.Count > 0)
                Assert.Fail($"{r.Findings.Count} foreign-memory ownership families:\n  " + string.Join("\n  ",
                    r.Findings.OrderByDescending(k => k.Value.count).Take(60)
                     .Select(k => $"{k.Value.count}x {k.Key.op} [{k.Key.kind}, {k.Key.violation}]  e.g. {k.Value.sampleId}: {k.Value.detail}")));
        }

        /// <summary>A float64 (2, 3) operand holding 0..5, as a corpus row would describe it.</summary>
        /// <returns>The descriptor.</returns>
        private static FuzzCorpus.Operand ForeignTestOperand()
        {
            var bytes = new byte[6 * sizeof(double)];
            for (int i = 0; i < 6; i++)
                BitConverter.GetBytes((double)i).CopyTo(bytes, i * sizeof(double));
            return new FuzzCorpus.Operand
            {
                Dtype = "float64", Shape = new long[] { 2, 3 }, Strides = new long[] { 3, 1 }, Offset = 0, BufferSize = 6,
                Buffer = Convert.ToHexString(bytes),
            };
        }

        /// <summary>
        ///     Non-vacuity of the foreign tier: every kind really builds the memory it names (pooled, a pin, an external block
        ///     with a release action, a non-owning wrap) and the operand reads the corpus bytes through the corpus layout.
        /// </summary>
        /// <exception cref="AssertFailedException">A kind built another kind of memory, or the values are wrong.</exception>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        [TestCategory("ScopeAudit")]
        public void ForeignMemory_EveryKindIsTheMemoryItNames()
        {
            var expected = new Dictionary<ForeignKind, string[]>
            {
                [ForeignKind.Pooled] = new[] { "Native" },
                [ForeignKind.PinnedArray] = new[] { "GCHandle" },
                [ForeignKind.ByteBuffer] = new[] { "External" },
                // np.frombuffer pins the bytes; either block shape (a pin, or a pin with a release action) is foreign memory.
                [ForeignKind.FromBuffer] = new[] { "GCHandle", "External" },
                [ForeignKind.RawPointer] = new[] { "Wrap" },
                [ForeignKind.ExternalRelease] = new[] { "External" },
            };
            foreach (ForeignKind kind in Enum.GetValues(typeof(ForeignKind)))
            {
                var f = ForeignOperand.TryBuild(ForeignTestOperand(), kind);
                try
                {
                    string got = ArcProbe.Read(f.Array.Storage.InternalArray).Kind;
                    CollectionAssert.Contains(expected[kind], got, $"{kind} built {got} memory");
                    CollectionAssert.AreEqual(new double[] { 0, 1, 2, 3, 4, 5 }, f.Array.ToArray<double>(), $"{kind} reads wrong values");
                }
                finally
                {
                    f.Release();
                    f.FreeCallerMemory(safe: true);
                }
            }
        }

        /// <summary>
        ///     The pool check of the foreign tier sees caller memory parked in the pool (here returned directly, as a block a
        ///     buggy op laid over the caller's memory would on its disposal).
        /// </summary>
        /// <exception cref="AssertFailedException">The scan missed the parked caller memory.</exception>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        [TestCategory("ScopeAudit")]
        public unsafe void ForeignMemory_PoolScan_FindsCallerMemoryParkedInThePool()
        {
            // The caller's unmanaged memory, of an odd size whose bucket is otherwise empty (the planted return then parks
            // instead of freeing), from the plain allocator (a parking pool frees with NativeMemory.Free when it must).
            const long bytes = 4096 * 8 + 40;
            var memory = (IntPtr)NativeMemory.Alloc((nuint)bytes);
            var range = ((ulong)memory.ToInt64(), (ulong)memory.ToInt64() + (ulong)bytes);
            SizeBucketedBufferPoolReturn(memory, bytes);   // the planted over-disposal
            var found = PoolScan.FirstInside(new[] { range }, new[] { bytes });
            // Take it back out of the pool (last-in first-out) before anything else can be handed the caller's memory.
            var back = SizeBucketedBufferPool.Take(bytes);
            Assert.AreEqual(memory, back, "the pool handed back another buffer: the planted return did not park");
            NativeMemory.Free((void*)memory);
            Assert.AreEqual(memory, found, "the pool scan missed caller memory parked in the pool");
        }

        /// <summary>Returns a buffer to the pool (a seam so the planted return reads as one statement).</summary>
        /// <param name="p">The buffer.</param>
        /// <param name="bytes">Its size.</param>
        private static void SizeBucketedBufferPoolReturn(IntPtr p, long bytes)
            => SizeBucketedBufferPool.Return(p, bytes);

        /// <summary>
        ///     A caller's release action that runs while the caller still holds an array over its memory (an over-release of
        ///     the last counted reference) is caught by the ledger as <see cref="OwnershipViolation.Released"/>, and the
        ///     action is not run again by the harness's own disposal (the block's single-shot guard).
        /// </summary>
        /// <exception cref="AssertFailedException">The early release went unseen, or the action ran twice.</exception>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        [TestCategory("ScopeAudit")]
        public void ForeignMemory_EarlyRelease_IsCaught()
        {
            var f = ForeignOperand.TryBuild(ForeignTestOperand(), ForeignKind.ExternalRelease);
            var ledger = OwnershipLedger.OfPositional(new[] { f.Array });
            f.Array.Storage.InternalArray.Release();   // the planted over-release: the last reference, while the operand lives
            var findings = ledger.Check();
            Assert.IsTrue(findings.Any(x => x.Kind == OwnershipViolation.Released), string.Join("; ", findings));
            Assert.IsNull(f.Release(), "the caller's release action ran twice");
            f.FreeCallerMemory(safe: true);
        }

        /// <summary>
        ///     Replays the sample: for each ordinary tier, the first case of each (op, layout, operand-0 dtype), under every
        ///     <see cref="ForeignKind"/>.
        /// </summary>
        /// <returns>The pass's observations.</returns>
        private static ForeignSweepResult RunForeignSweep()
        {
            var corpusDir = Path.Combine(AppContext.BaseDirectory, "Fuzz", "corpus");
            var files = Directory.GetFiles(corpusDir, "*.jsonl")
                .Select(Path.GetFileName)
                .Where(f => !f.StartsWith("index_", StringComparison.Ordinal))   // the indexer tiers have their own schema
                .Where(f => !f.StartsWith("ma_", StringComparison.Ordinal))      // masked operands wrap ordinary ones
                .Where(f => !f.EndsWith(".host.jsonl", StringComparison.Ordinal))
                .OrderBy(f => f, StringComparer.Ordinal)
                .ToArray();

            var findings = new Dictionary<(string, ForeignKind, string), (long, string, string)>();
            var byKind = new Dictionary<ForeignKind, long>();
            var skipped = new Dictionary<ForeignKind, long>();
            long cases = 0;
            var kinds = (ForeignKind[])Enum.GetValues(typeof(ForeignKind));
            var seen = new HashSet<(string, string, string)>();

            void Record(string op, ForeignKind kind, string violation, string sampleId, string detail)
            {
                var key = (op, kind, violation);
                findings[key] = findings.TryGetValue(key, out var g) ? (g.Item1 + 1, g.Item2, g.Item3) : (1, sampleId, detail);
            }

            foreach (var file in files)
            {
                using var corpus = FuzzCorpus.Open(file);
                foreach (var c in corpus)
                {
                    if (c.Operands.Length == 0)
                        continue;
                    if (!seen.Add((c.Op, c.Layout ?? "?", c.Operands[0].Dtype)))
                        continue;

                    foreach (var kind in kinds)
                    {
                        if (!ReplayForeign(c, kind, Record))
                        {
                            skipped[kind] = skipped.GetValueOrDefault(kind) + 1;
                            continue;
                        }
                        cases++;
                        byKind[kind] = byKind.GetValueOrDefault(kind) + 1;
                    }
                }
            }

            return new ForeignSweepResult(cases, byKind, findings, skipped);
        }

        /// <summary>
        ///     Replays one case with its operands rebuilt over <paramref name="kind"/> memory and records every finding.
        /// </summary>
        /// <param name="c">The corpus case.</param>
        /// <param name="kind">The memory kind.</param>
        /// <param name="record">Records a finding: (op, kind, violation, sample id, detail).</param>
        /// <returns>False when an operand could not be rebuilt over this kind (the case is skipped for it).</returns>
        /// <exception cref="InvalidOperationException">The ARC probe or the pool scan cannot read an internal field.</exception>
        private static bool ReplayForeign(FuzzCorpus.Case c, ForeignKind kind, Action<string, ForeignKind, string, string, string> record)
        {
            var built = new List<ForeignOperand>(c.Operands.Length);
            try
            {
                foreach (var o in c.Operands)
                {
                    var f = ForeignOperand.TryBuild(o, kind);
                    if (f is null)
                        return false;
                    built.Add(f);
                }
            }
            catch
            {
                // A kind that cannot express this operand (an unsupported dtype for the buffer route): not a verdict.
                foreach (var f in built)
                    f.Abandon();
                return false;
            }

            var operands = built.Select(f => f.Array).ToArray();
            if (c.Alias && operands.Length == 1)
                operands = new[] { operands[0], operands[0] };
            var ledger = OwnershipLedger.OfPositional(operands);

            try
            {
                DisposeResult(Invoke(c, operands), operands);
            }
            catch
            {
                // Value and error verdicts belong to FuzzCorpusTests; what this gate owns is what the call did to the
                // memory, on the success and the error path alike.
            }

            foreach (var finding in ledger.Check(scanPool: true))
                if (finding.IsOverDisposal)
                    record(c.Op, kind, finding.Kind.ToString(), c.Id, finding.ToString());

            // The harness's own release: dispose the operands, then judge each kind's release accounting before the caller
            // memory is freed.
            foreach (var f in built)
            {
                string problem = f.Release();
                if (problem != null)
                    record(c.Op, kind, "ReleasedTwice", c.Id, problem);
            }

            var foreignRanges = built.Where(f => f.CallerRange.hi > f.CallerRange.lo).Select(f => f.CallerRange).ToList();
            var pooled = PoolScan.FirstInside(foreignRanges, built.SelectMany(f => f.CallerSizes));
            if (pooled != IntPtr.Zero)
                record(c.Op, kind, "ForeignMemoryPooled", c.Id,
                       $"after the operands were disposed, the buffer pool holds 0x{(ulong)pooled.ToInt64():X} inside memory the caller owns");

            foreach (var f in built)
                f.FreeCallerMemory(pooled == IntPtr.Zero);
            return true;
        }

        /// <summary>
        ///     One operand rebuilt over a <see cref="ForeignKind"/> of memory, with the bookkeeping the gate checks.
        /// </summary>
        private sealed class ForeignOperand
        {
            /// <summary>The operand view the call receives (the corpus case's shape, strides and offset).</summary>
            public NDArray Array { get; private set; }

            /// <summary>The byte range of memory the CALLER owns (unmanaged kinds; empty for managed and pooled memory).</summary>
            public (ulong lo, ulong hi) CallerRange { get; private set; }

            /// <summary>
            ///     The pool bucket sizes a buffer returned in the caller's place would have: the caller memory's byte length
            ///     and the operand view's (see <see cref="PoolScan"/>). Empty for the managed and pooled kinds.
            /// </summary>
            public IReadOnlyList<long> CallerSizes { get; private set; } = System.Array.Empty<long>();

            /// <summary>The caller's unmanaged memory, freed by <see cref="FreeCallerMemory"/> (zero for the other kinds).</summary>
            private IntPtr _callerMemory;

            /// <summary>How often the external release action ran (the <see cref="ForeignKind.ExternalRelease"/> kind).</summary>
            private int _releases;

            /// <summary>Whether the kind has a release action the gate counts.</summary>
            private bool _countsReleases;

            /// <summary>The non-owning wrap's generic factory (<c>new UnmanagedMemoryBlock&lt;T&gt;(T*, count)</c>).</summary>
            private static readonly MethodInfo WrapRawMethod =
                typeof(ForeignOperand).GetMethod(nameof(WrapRaw), BindingFlags.NonPublic | BindingFlags.Static)!;

            /// <summary>
            ///     Rebuilds a corpus operand over <paramref name="kind"/> memory.
            /// </summary>
            /// <param name="o">The corpus operand.</param>
            /// <param name="kind">The memory kind.</param>
            /// <returns>The operand, or null when the operand has no memory (an empty array is the same in every kind).</returns>
            /// <exception cref="NotSupportedException">The dtype is not one the kind's route takes.</exception>
            /// <exception cref="ArgumentOutOfRangeException"><paramref name="kind"/> is not a defined kind.</exception>
            public static unsafe ForeignOperand TryBuild(FuzzCorpus.Operand o, ForeignKind kind)
            {
                var tc = FuzzCorpus.DtypeToTC(o.Dtype);
                foreach (var d in o.Shape)
                    if (d == 0)
                        return null;

                byte[] bytes = FuzzCorpus.FromHex(o.Buffer);
                long count = o.BufferSize;
                int itemSize = tc.SizeOf();
                if (count <= 0 || bytes.Length < count * itemSize)
                    return null;

                var f = new ForeignOperand();
                NDArray root;
                switch (kind)
                {
                    case ForeignKind.Pooled:
                        root = np.empty(new Shape(count), tc);
                        fixed (byte* src = bytes)
                            Buffer.MemoryCopy(src, root.Address, count * itemSize, count * itemSize);
                        break;
                    case ForeignKind.PinnedArray:
                    {
                        var arr = System.Array.CreateInstance(tc.AsType(), count);
                        var h = GCHandle.Alloc(arr, GCHandleType.Pinned);
                        try
                        {
                            fixed (byte* src = bytes)
                                Buffer.MemoryCopy(src, (void*)h.AddrOfPinnedObject(), count * itemSize, count * itemSize);
                        }
                        finally
                        {
                            h.Free();
                        }
                        root = new NDArray(arr, new Shape(count));   // the wrap constructor: NumSharp pins the caller's array
                        break;
                    }
                    case ForeignKind.ByteBuffer:
                        root = np.ndarray(new Shape(count), tc, buffer: (byte[])bytes.Clone());
                        break;
                    case ForeignKind.FromBuffer:
                        root = np.frombuffer((byte[])bytes.Clone(), tc);
                        break;
                    case ForeignKind.RawPointer:
                    case ForeignKind.ExternalRelease:
                    {
                        long n = count * itemSize;
                        var p = (IntPtr)NativeMemory.AlignedAlloc((nuint)n, 64);
                        fixed (byte* src = bytes)
                            Buffer.MemoryCopy(src, (void*)p, n, n);
                        f._callerMemory = p;
                        f.CallerRange = ((ulong)p.ToInt64(), (ulong)p.ToInt64() + (ulong)n);
                        long viewBytes = itemSize;
                        foreach (var d in o.Shape)
                            viewBytes *= d;
                        f.CallerSizes = new[] { n, viewBytes };
                        IArraySlice slice;
                        if (kind == ForeignKind.RawPointer)
                        {
                            slice = (IArraySlice)WrapRawMethod.MakeGenericMethod(tc.AsType()).Invoke(null, new object[] { p, count })!;
                        }
                        else
                        {
                            // The release action is the caller's: it only counts. The memory is freed by the harness after the
                            // case, once the pool scan proved nothing NumSharp holds still points into it.
                            f._countsReleases = true;
                            slice = ArraySlice.WrapOwned(tc, (void*)p, count, () => System.Threading.Interlocked.Increment(ref f._releases));
                        }
                        root = new NDArray(new UnmanagedStorage(slice, Shape.Vector(count)));
                        break;
                    }
                    default:
                        throw new ArgumentOutOfRangeException(nameof(kind), kind, null);
                }

                // The operand is a VIEW of the root (the corpus layout), holding the only reference left once the root is
                // disposed — the same single-reference structure FuzzCorpus.Reconstruct builds.
                var view = new NDArray(root.Storage, new Shape(o.Shape, o.Strides, o.Offset, o.BufferSize));
                root.Dispose();
                if (o.Writeable == false)
                    view.setflags(write: false);
                f.Array = view;
                return f;
            }

            /// <summary>The non-owning wrap of caller memory.</summary>
            /// <typeparam name="T">The element type.</typeparam>
            /// <param name="p">The caller's memory.</param>
            /// <param name="count">Its element count.</param>
            /// <returns>A slice whose block never frees, pools or counts.</returns>
            private static unsafe IArraySlice WrapRaw<T>(IntPtr p, long count) where T : unmanaged
                => new ArraySlice<T>(new UnmanagedMemoryBlock<T>((T*)p, count));

            /// <summary>
            ///     Disposes the operand (the harness's last reference) and judges the release accounting.
            /// </summary>
            /// <returns>A problem description (the caller's release action ran more than once), or null.</returns>
            public string Release()
            {
                Array.Dispose();
                int releases = System.Threading.Volatile.Read(ref _releases);
                return _countsReleases && releases > 1
                    ? $"the caller's release action ran {releases} times (once is the contract: the second release frees memory the caller may already reuse)"
                    : null;
            }

            /// <summary>
            ///     Frees the caller's unmanaged memory (unmanaged kinds) after the case — only when nothing NumSharp holds
            ///     points into it; otherwise the memory is leaked on purpose (freeing a block the pool parks would corrupt the
            ///     next allocation of its size).
            /// </summary>
            /// <param name="safe">Whether the pool scan found nothing inside the caller's memory.</param>
            public unsafe void FreeCallerMemory(bool safe)
            {
                if (_callerMemory != IntPtr.Zero && safe)
                    NativeMemory.AlignedFree((void*)_callerMemory);
                _callerMemory = IntPtr.Zero;
            }

            /// <summary>Releases a partially built case's memory (a later operand could not be rebuilt).</summary>
            public void Abandon()
            {
                Array?.Dispose();
                FreeCallerMemory(PoolScan.FirstInside(new[] { CallerRange }, CallerSizes) == IntPtr.Zero);
            }
        }
    }
}
