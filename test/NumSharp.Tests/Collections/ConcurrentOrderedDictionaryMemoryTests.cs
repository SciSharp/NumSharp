#nullable enable

using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Collections;

namespace NumSharp.Tests.Collections
{
    /// <summary>
    ///     Memory-contract pins for <see cref="ConcurrentOrderedDictionary{TKey,TValue}" /> — the allocation and
    ///     retention behavior the type promises, measured with the thread-local allocation counter and
    ///     <see cref="WeakReference" /> probes so a regression (a boxed enumerator, a lost fast path, a slot that
    ///     stops being released) fails deterministically. The comparative numbers against <see cref="List{T}" />
    ///     and the framework dictionary live in <c>ConcurrentOrderedDictionary.TODO.md § Memory analysis</c>; these tests
    ///     pin only the collection's OWN contracts, which hold on any 64-bit runtime and in both Debug and Release.
    /// </summary>
    [TestClass]
    public class ConcurrentOrderedDictionaryMemoryTests
    {
        /// <summary>
        ///     Measures the managed bytes one invocation of <paramref name="op" /> allocates on this thread —
        ///     minimum over several rounds after warmup, so tier-0→tier-1 JIT transitions and one-time lazy
        ///     initializations cannot pollute the figure (the standing house pattern for allocation asserts).
        /// </summary>
        /// <param name="op">The operation to measure; created once by the caller so delegate allocation is outside the measured region.</param>
        /// <param name="rounds">How many measured rounds to take the minimum over.</param>
        /// <returns>The minimum bytes allocated by a single invocation.</returns>
        private static long MinAllocated(Action op, int rounds = 5)
        {
            for (int i = 0; i < 3; i++)
            {
                op(); // warmup: JIT, lazy statics, pool priming — none of it may count against the contract
            }

            long best = long.MaxValue;
            for (int r = 0; r < rounds; r++)
            {
                long before = GC.GetAllocatedBytesForCurrentThread();
                op();
                best = System.Math.Min(best, GC.GetAllocatedBytesForCurrentThread() - before);
            }

            return best;
        }

        /// <summary>Full-fence collection retried a few times before declaring an object collected, so background/concurrent GC modes cannot flake the negative asserts.</summary>
        /// <param name="wr">The probe to wait on.</param>
        /// <returns><see langword="true" /> once the target has been collected; <see langword="false" /> after the retries expire.</returns>
        private static bool WaitCollected(WeakReference wr)
        {
            for (int i = 0; i < 5; i++)
            {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                GC.Collect();
                if (!wr.IsAlive)
                {
                    return true;
                }
            }

            return false;
        }

        // ---------------------------------------------------------------- allocation contracts (thread-local counter)

        [TestMethod]
        public void Reads_AreAllocationFree_OnEveryPath()
        {
            var d = new ConcurrentOrderedDictionary<int, int>(4_096);
            for (int i = 0; i < 4_096; i++)
            {
                d.Add(i, i);
            }

            long sink = 0;
            Action reads = () =>
            {
                for (int i = 0; i < 1_000; i++)
                {
                    d.TryGetValue(i, out int v);
                    sink += v;
                    sink += d.IndexOf(i);
                    d.TryGetIndex(i, out int idx);
                    sink += idx;
                    d.TryGetAt(i, out int at);
                    sink += at;
                    sink += d[i];
                    sink += d.GetKeyAt(i);
                    if (d.ContainsKey(i))
                    {
                        sink++;
                    }

                    sink += d.Count;
                }
            };

            Assert.AreEqual(0, MinAllocated(reads), "a lock-free read path allocated — a fast path was lost");
            GC.KeepAlive(sink);
        }

        [TestMethod]
        public void Enumeration_And_SnapshotView_AreAllocationFree()
        {
            var d = new ConcurrentOrderedDictionary<int, int>(4_096);
            for (int i = 0; i < 4_096; i++)
            {
                d.Add(i, i);
            }

            long sink = 0;
            Action fullScans = () =>
            {
                foreach (int v in d) // struct enumerator: must never box
                {
                    sink += v;
                }

                var view = d.Snapshot(); // readonly struct + span: zero heap
                foreach (int v in view)
                {
                    sink += v;
                }

                ReadOnlySpan<int> span = view.AsSpan();
                for (int i = 0; i < span.Length; i++)
                {
                    sink += span[i];
                }
            };

            Assert.AreEqual(0, MinAllocated(fullScans), "enumeration or the snapshot view allocated — an enumerator got boxed or a snapshot started copying");
            GC.KeepAlive(sink);
        }

        [TestMethod]
        public void AtomicValueReplace_IsAllocationFree()
        {
            // The in-place ref-seam update: for an atomically-writable TValue, replacing an existing key's value
            // must allocate NOTHING (no node swap, no array clone) on any of the three replace entry points.
            var d = new ConcurrentOrderedDictionary<int, long>(1_024);
            for (int i = 0; i < 1_024; i++)
            {
                d.Add(i, i);
            }

            Action replaces = () =>
            {
                for (int i = 0; i < 500; i++)
                {
                    d.SetByKey(i, i * 2);
                    d.SetAt(i, i * 3);
                    d.TryUpdate(i, i * 4, i * 3);
                }
            };

            Assert.AreEqual(0, MinAllocated(replaces), "an atomic in-place value replace allocated — the ref-seam fast path regressed to node/array reallocation");
        }

        [TestMethod]
        public void PresizedAppend_AllocatesOnlyTheHashNode()
        {
            // Within capacity, an append is exactly one hash-node allocation (like the framework dictionary's
            // TryAdd): no per-add snapshot holder, no array growth. The bound is 72 B: the 40 B node, plus — in
            // Debug builds only — the vendored map's own verbatim `key is null` check, whose box IL executes for
            // value-type keys under unoptimized codegen (24 B; the vendored body stays byte-faithful to upstream,
            // so that one check is not rewritten — every NumSharp-owned path is guarded and allocation-free).
            var d = new ConcurrentOrderedDictionary<int, int>(200_000);
            int next = 0;
            Action adds = () =>
            {
                for (int i = 0; i < 1_000; i++)
                {
                    d.TryAdd(next++, 1);
                }
            };

            long perAdd = MinAllocated(adds) / 1_000;
            Assert.IsTrue(perAdd <= 72, $"a presized in-capacity append allocated {perAdd} B — more than one hash node (the per-add Store holder is back, or growth triggered)");
        }

        [TestMethod]
        public void TailPop_And_SwapBack_AllocateOnlyOneSnapshotHolder()
        {
            // The O(1) removals publish one small Store object each — never fresh arrays. 96 B bounds the 40 B
            // holder with slack; the contrast test below proves the interior path is orders of magnitude bigger.
            var d = new ConcurrentOrderedDictionary<int, int>(64_000);
            for (int i = 0; i < 64_000; i++)
            {
                d.Add(i, i);
            }

            int popAt = 63_999;
            Action pop = () => d.TryRemove(popAt--, out _);
            long popBytes = MinAllocated(pop, rounds: 200);
            Assert.IsTrue(popBytes <= 96, $"a tail pop allocated {popBytes} B — the O(1) shared-arrays path regressed to copying");

            int swapAt = 1_000;
            Action swap = () => d.TryRemoveSwapBack(swapAt++, out _);
            long swapBytes = MinAllocated(swap, rounds: 200);
            Assert.IsTrue(swapBytes <= 96, $"an in-place swap-back allocated {swapBytes} B — the O(1) path regressed to copying");
        }

        [TestMethod]
        public void InteriorRemove_PaysCapacitySizedArrays_TailPopDoesNot()
        {
            // The documented memory split between the removal families, pinned as a ratio so it is layout- and
            // platform-independent: one interior removal allocates two capacity-sized arrays (~8 B/slot here),
            // hundreds of times a tail pop's single 40 B holder.
            const int Capacity = 32_768;
            var d = new ConcurrentOrderedDictionary<int, int>(Capacity);
            for (int i = 0; i < Capacity; i++)
            {
                d.Add(i, i);
            }

            int interiorKey = 5_000;
            Action interior = () => d.TryRemove(interiorKey++, out _);
            long interiorBytes = MinAllocated(interior, rounds: 10);

            int tailKey = Capacity - 1;
            Action tail = () => d.TryRemove(tailKey--, out _);
            long tailBytes = MinAllocated(tail, rounds: 10);

            Assert.IsTrue(interiorBytes >= Capacity * 2L * sizeof(int), $"interior removal allocated only {interiorBytes} B — it stopped copying, which would corrupt live snapshots");
            Assert.IsTrue(tailBytes <= 96, $"tail pop allocated {tailBytes} B");
            Assert.IsTrue(interiorBytes > 100 * System.Math.Max(1, tailBytes), "the interior/tail memory split collapsed");
        }

        [TestMethod]
        public void NonAtomicReplace_AddRangeBatchesManyClonesIntoOne()
        {
            // decimal (non-atomic) replacement must clone the values array per call — O(capacity) bytes each.
            // The documented mitigation: an AddRange batch of replacements performs ONE copy-on-write for the
            // whole batch. Pinned as a ratio: 100 batched replaces must cost far less than 100 singles.
            const int Capacity = 8_192;
            var d = new ConcurrentOrderedDictionary<int, decimal>(Capacity);
            for (int i = 0; i < Capacity; i++)
            {
                d.Add(i, i);
            }

            Action singles = () =>
            {
                for (int i = 0; i < 100; i++)
                {
                    d.SetByKey(i, i + 0.5m);
                }
            };
            long singlesBytes = MinAllocated(singles);

            var batch = new KeyValuePair<int, decimal>[100];
            for (int i = 0; i < 100; i++)
            {
                batch[i] = new KeyValuePair<int, decimal>(i, i + 0.25m);
            }

            Action batched = () => d.AddRange(batch);
            long batchedBytes = MinAllocated(batched);

            Assert.IsTrue(singlesBytes >= 100L * Capacity * sizeof(decimal) / 2, $"100 single wide-value replaces allocated only {singlesBytes} B — the tear-free clone discipline changed");
            Assert.IsTrue(batchedBytes * 10 < singlesBytes, $"AddRange batching gave no memory win: batch {batchedBytes} B vs singles {singlesBytes} B");
        }

        // ---------------------------------------------------------------- retention contracts (WeakReference probes)

        /// <summary>Seeds two entries, appends a probe value at the tail and pops it — the O(1) shared-arrays removal that deliberately does not scrub the slot.</summary>
        /// <param name="d">The collection under probe.</param>
        /// <returns>A weak reference to the popped value (no strong reference survives this frame).</returns>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static WeakReference AddThenTailPop(ConcurrentOrderedDictionary<int, object> d)
        {
            var value = new object();
            d.Add(2, value);
            var wr = new WeakReference(value);
            Assert.IsTrue(d.TryRemove(2, out _)); // tail position → O(1) pop, arrays shared downward
            return wr;
        }

        [TestMethod]
        public void TailPop_RetainsTheValue_UntilTheNextAppendCopiesTheArrays()
        {
            var d = new ConcurrentOrderedDictionary<int, object>();
            d.Add(0, new object());
            d.Add(1, new object());

            WeakReference wr = AddThenTailPop(d);

            // Documented retention: the vacated slot is not scrubbed (an enumerator captured at the old count must
            // still read valid data), so the popped value stays reachable through the shared arrays.
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            Assert.IsTrue(wr.IsAlive, "the popped value was scrubbed — a captured enumerator would now read garbage");

            // The floor rule makes the next append copy-on-write WITHOUT carrying the vacated slot — that copy is
            // the release point.
            d.Add(3, new object());
            Assert.IsTrue(WaitCollected(wr), "the popped value survived the copy-on-write append — the vacated slot leaked into the fresh arrays");
        }

        /// <summary>Adds a probe value followed by another entry and removes the probe from the interior (fresh-array compaction excludes it).</summary>
        /// <param name="d">The collection under probe (must already hold at least one entry so the removal is interior).</param>
        /// <returns>A weak reference to the removed value.</returns>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static WeakReference AddThenInteriorRemove(ConcurrentOrderedDictionary<int, object> d)
        {
            var value = new object();
            d.Add(10, value);
            d.Add(11, new object()); // something after it, so removing 10 shifts (interior path)
            var wr = new WeakReference(value);
            Assert.IsTrue(d.TryRemove(10, out _));
            return wr;
        }

        [TestMethod]
        public void InteriorRemove_ReleasesTheRemovedValueImmediately()
        {
            var d = new ConcurrentOrderedDictionary<int, object>();
            d.Add(0, new object());
            WeakReference wr = AddThenInteriorRemove(d);
            Assert.IsTrue(WaitCollected(wr), "an interior-removed value stayed reachable — the compaction copied the removed slot or the old arrays are being retained");
        }

        /// <summary>Adds a probe value and clears the collection.</summary>
        /// <param name="d">The collection under probe.</param>
        /// <returns>A weak reference to the cleared-out value.</returns>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static WeakReference AddThenClear(ConcurrentOrderedDictionary<int, object> d)
        {
            var value = new object();
            d.Add(0, value);
            var wr = new WeakReference(value);
            d.Clear();
            return wr;
        }

        [TestMethod]
        public void Clear_ReleasesEverything()
        {
            var d = new ConcurrentOrderedDictionary<int, object>();
            WeakReference wr = AddThenClear(d);
            Assert.IsTrue(WaitCollected(wr), "Clear left a value reachable");
        }

        /// <summary>
        ///     Holds a boxed enumerator over a one-entry snapshot, removes the entry and appends another (pushing a
        ///     fresh-array store), asserts the enumerator still pins the removed value, then drops the enumerator
        ///     with this frame.
        /// </summary>
        /// <param name="d">The collection under probe.</param>
        /// <returns>A weak reference to the removed value, for the caller to verify release after this frame dies.</returns>
        /// <remarks>
        ///     Both the "pinned" check and the drop happen INSIDE this method on purpose: the enumerator reference
        ///     must never touch the caller's frame, where untracked tier-0/Debug stack slots would keep it alive
        ///     and turn the release assert into a false leak (observed doing exactly that in the analysis probes).
        /// </remarks>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static WeakReference HoldEnumeratorCheckPinnedThenDrop(ConcurrentOrderedDictionary<int, object> d)
        {
            var value = new object();
            d.Add(0, value);
            var wr = new WeakReference(value);

            IEnumerator<object> pin = ((IEnumerable<object>)d).GetEnumerator(); // boxed → references the captured value array
            Assert.IsTrue(d.TryRemove(0, out _));
            d.Add(1, new object()); // floor copy: from here only the enumerator references the old arrays

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            Assert.IsTrue(wr.IsAlive, "a held enumerator failed to pin its snapshot — its captured array was collected under it");

            GC.KeepAlive(pin); // the pin's last use; the reference dies with this frame
            return wr;
        }

        [TestMethod]
        public void HeldEnumerator_PinsItsSnapshotArrays_ReleasedWhenDropped()
        {
            var d = new ConcurrentOrderedDictionary<int, object>();
            WeakReference wr = HoldEnumeratorCheckPinnedThenDrop(d);
            Assert.IsTrue(WaitCollected(wr), "the snapshot arrays stayed reachable after the enumerator was dropped");
        }

        [TestMethod]
        public void AddRange_AfterTailRemoval_DoesNotCompoundBufferCapacity()
        {
            // Regression (surfaced by the randomized concurrency storm, then reduced to this deterministic
            // single-threaded pin): an AddRange copy-on-write forced ONLY by the floor rule — a tail removal
            // raised the floor, but the arrays still have room — must NOT double the capacity the way a genuine
            // out-of-room grow does. Before the fix it did, so every AddRange that followed a tail removal doubled
            // the buffer; with the floor re-raised each round the doublings COMPOUNDED (the store's arrays grew
            // 4K → 16K → … → hundreds of millions of slots) for a working set that never exceeded ~500 entries,
            // ending in OutOfMemory. It is pinned as a cumulative-allocation ceiling checked every round so a
            // reintroduction trips deterministically around round 14 — long before it can OOM the test host —
            // rather than as an internal-capacity assertion that would couple the gate to a private field name.
            const int Seed = 500;
            var d = new ConcurrentOrderedDictionary<int, long>();
            for (int k = 0; k < Seed; k++)
            {
                d.Add(k, k);
            }

            var batch = new KeyValuePair<int, long>[1];

            // Warm up a few churn rounds so JIT and one-time costs are outside the measured region, and so the
            // floor is already raised before the first measured AddRange.
            for (int w = 0; w < 8; w++)
            {
                int key = 100_000 + w;
                batch[0] = new KeyValuePair<int, long>(key, key);
                d.AddRange(batch);
                d.TryRemove(key, out _);
            }

            long before = GC.GetAllocatedBytesForCurrentThread();
            for (int round = 0; round < 4_000; round++)
            {
                // Append one fresh tail key: because the previous round's tail removal raised the floor, this
                // AddRange must copy-on-write even though the arrays have room — exactly the path that used to
                // double the capacity.
                int key = 1_000_000 + round;
                batch[0] = new KeyValuePair<int, long>(key, key);
                d.AddRange(batch);

                // Remove it through the O(1) tail path, re-raising the floor for the next round.
                Assert.IsTrue(d.TryRemove(key, out _));

                long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
                Assert.IsTrue(allocated < 128L * 1024 * 1024,
                    $"round {round}: AddRange-after-tail-removal has allocated {allocated / (1024 * 1024)} MB for a {d.Count}-element map — the floor-forced copy is compounding the buffer capacity (the doubling bug is back)");
            }

            // The working set never grew, and the map is still fully intact — capacity stayed bounded.
            Assert.AreEqual(Seed, d.Count);
            for (int k = 0; k < Seed; k++)
            {
                Assert.IsTrue(d.TryGetValue(k, out long v) && v == k, $"seeded key {k} was lost");
            }
        }
    }
}
