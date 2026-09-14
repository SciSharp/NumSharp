using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Collections;

namespace NumSharp.Tests.Collections
{
    /// <summary>
    ///     Functional pins for <see cref="ConcurrentOrderedDict{TKey,TValue}" />: the dual key/index paths, ordering,
    ///     duplicate semantics, the O(1) tail-removal + append-floor snapshot rule, the order-breaking swap-back
    ///     removal, batch upsert, the snapshot view, and the non-atomic-value (node-swap / array-clone) code paths.
    ///     The multi-threaded guarantees are pinned separately in <see cref="ConcurrentOrderedDictConcurrencyTests" />.
    /// </summary>
    [TestClass]
    public class ConcurrentOrderedDictTests
    {
        /// <summary>Cross-checks every invariant the two paths promise each other on a quiescent instance: positions round-trip (key→index→key), both value routes agree, and the snapshot surfaces (arrays, pairs, view) match.</summary>
        /// <typeparam name="TKey">The dictionary's key type.</typeparam>
        /// <typeparam name="TValue">The dictionary's value type.</typeparam>
        /// <param name="d">The instance to audit.</param>
        private static void AssertFullyConsistent<TKey, TValue>(ConcurrentOrderedDict<TKey, TValue> d)
            where TKey : notnull
        {
            int n = d.Count;
            TKey[] keys = d.Keys;
            TValue[] values = d.ToArray();
            Assert.AreEqual(n, keys.Length);
            Assert.AreEqual(n, values.Length);

            var view = d.Snapshot();
            Assert.AreEqual(n, view.Count);

            var pairs = d.Pairs.ToArray();
            Assert.AreEqual(n, pairs.Length);

            for (int i = 0; i < n; i++)
            {
                TKey k = d.GetKeyAt(i);
                Assert.AreEqual(keys[i], k, $"Keys[{i}] mismatch");
                Assert.AreEqual(i, d.IndexOf(k), $"IndexOf(GetKeyAt({i})) must round-trip");
                Assert.IsTrue(d.TryGetIndex(k, out int idx) && idx == i);
                Assert.IsTrue(d.ContainsKey(k));
                Assert.IsTrue(d.TryGetValue(k, out TValue byKey));
                Assert.IsTrue(d.TryGetAt(i, out TValue byIndex));
                Assert.AreEqual(byKey, byIndex, $"key path and list path disagree at {i}");
                Assert.AreEqual(values[i], byIndex);
                Assert.AreEqual(view[i], byIndex);
                Assert.AreEqual(view.GetKeyAt(i), k);
                Assert.AreEqual(pairs[i].Key, k);
                Assert.AreEqual(pairs[i].Value, byIndex);
            }
        }

        [TestMethod]
        public void Empty_HasNoEntries_AndReadsBehave()
        {
            var d = new ConcurrentOrderedDict<string, int>();
            Assert.AreEqual(0, d.Count);
            Assert.IsTrue(d.IsEmpty);
            Assert.IsFalse(d.ContainsKey("x"));
            Assert.IsFalse(d.TryGetValue("x", out _));
            Assert.IsFalse(d.TryGetAt(0, out _));
            Assert.AreEqual(-1, d.IndexOf("x"));
            Assert.AreEqual(0, d.ToArray().Length);
            Assert.AreEqual(0, d.Keys.Length);
            Assert.AreEqual(0, d.Snapshot().Count);
            Assert.IsFalse(d.GetEnumerator().MoveNext());
        }

        [TestMethod]
        public void Ctor_NegativeCapacity_Throws()
            => Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => _ = new ConcurrentOrderedDict<int, int>(-1));

        [TestMethod]
        public void Add_PreservesInsertionOrder_AndBothPathsAgree()
        {
            var d = new ConcurrentOrderedDict<string, int>();
            d.Add("a", 1);
            d.Add("b", 2);
            d.Add("c", 3);

            CollectionAssert.AreEqual(new[] { "a", "b", "c" }, d.Keys);
            CollectionAssert.AreEqual(new[] { 1, 2, 3 }, d.ToArray());
            Assert.AreEqual(2, d["b"]); // key path (string key: the TKey indexer binds)
            Assert.AreEqual(2, d[1]);   // list path
            AssertFullyConsistent(d);
        }

        [TestMethod]
        public void Add_DuplicateKey_Throws_AndTryAddReturnsFalse()
        {
            var d = new ConcurrentOrderedDict<string, int> { };
            Assert.IsTrue(d.TryAdd("k", 1));
            Assert.IsFalse(d.TryAdd("k", 2));
            Assert.ThrowsExactly<ArgumentException>(() => d.Add("k", 3));
            Assert.AreEqual(1, d.GetByKey("k")); // first value survives; failed adds change nothing
            Assert.AreEqual(1, d.Count);
        }

        [TestMethod]
        public void NullKey_Throws_OnEveryWriteEntryPoint()
        {
            var d = new ConcurrentOrderedDict<string, int>();
            Assert.ThrowsExactly<ArgumentNullException>(() => d.TryAdd(null!, 1));
            Assert.ThrowsExactly<ArgumentNullException>(() => d.SetByKey(null!, 1));
            Assert.ThrowsExactly<ArgumentNullException>(() => d.TryRemove(null!, out _));
            Assert.ThrowsExactly<ArgumentNullException>(() => d.TryRemoveSwapBack(null!, out _));
            Assert.ThrowsExactly<ArgumentNullException>(() => d.GetOrAdd(null!, 1));
            Assert.ThrowsExactly<ArgumentNullException>(() => d.AddOrUpdate(null!, 1, (k, v) => v));
            Assert.ThrowsExactly<ArgumentNullException>(() => d.AddRange(null!));
            Assert.ThrowsExactly<ArgumentNullException>(() => d.RemoveWhere(null!));
        }

        [TestMethod]
        public void IntKey_IndexerBindsToIndex_NotKey()
        {
            // The documented int-key footgun: d[x] is POSITION for TKey == int; key access goes through methods.
            var d = new ConcurrentOrderedDict<int, string>();
            d.Add(10, "ten");
            d.Add(20, "twenty");

            Assert.AreEqual("ten", d[0]);
            Assert.AreEqual("twenty", d[1]);
            Assert.AreEqual("twenty", d.GetByKey(20));
            Assert.AreEqual(1, d.IndexOf(20));
        }

        [TestMethod]
        public void SetByKey_ExistingKey_UpdatesValueInPlace_KeepingPosition()
        {
            var d = new ConcurrentOrderedDict<string, int>();
            d.Add("a", 1);
            d.Add("b", 2);
            d.SetByKey("a", 99);

            Assert.AreEqual(99, d.GetByKey("a"));
            Assert.AreEqual(0, d.IndexOf("a")); // position unchanged
            Assert.AreEqual(99, d[0]);          // list path sees the same value
            AssertFullyConsistent(d);
        }

        [TestMethod]
        public void SetByKey_NewKey_Appends()
        {
            var d = new ConcurrentOrderedDict<string, int>();
            d.SetByKey("a", 1);
            d.SetByKey("b", 2);
            CollectionAssert.AreEqual(new[] { "a", "b" }, d.Keys);
            AssertFullyConsistent(d);
        }

        [TestMethod]
        public void SetAt_ReplacesValue_KeepsKeyAndPosition()
        {
            var d = new ConcurrentOrderedDict<string, int>();
            d.Add("a", 1);
            d.Add("b", 2);

            d.SetAt(1, 42);
            Assert.AreEqual(42, d.GetByKey("b"));
            Assert.AreEqual("b", d.GetKeyAt(1));

            d[0] = 7; // the int indexer's setter is SetAt
            Assert.AreEqual(7, d.GetByKey("a"));

            Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => d.SetAt(2, 0));
            Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => d.SetAt(-1, 0));
            AssertFullyConsistent(d);
        }

        [TestMethod]
        public void IndexGetters_OutOfRange_Throw_TryFormsReturnFalse()
        {
            var d = new ConcurrentOrderedDict<string, int>();
            d.Add("a", 1);
            Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => _ = d[1]);
            Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => _ = d[-1]);
            Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => d.GetKeyAt(1));
            Assert.IsFalse(d.TryGetAt(1, out _));
            Assert.IsFalse(d.TryGetAt(-1, out _));
        }

        [TestMethod]
        public void GetByKey_Missing_Throws_KeyNotFound()
        {
            var d = new ConcurrentOrderedDict<string, int>();
            Assert.ThrowsExactly<KeyNotFoundException>(() => d.GetByKey("nope"));
        }

        [TestMethod]
        public void GetOrAdd_ReturnsExisting_OrAppends()
        {
            var d = new ConcurrentOrderedDict<string, int>();
            Assert.AreEqual(1, d.GetOrAdd("a", 1));
            Assert.AreEqual(1, d.GetOrAdd("a", 2));          // existing wins
            Assert.AreEqual(5, d.GetOrAdd("b", k => 5));     // factory form
            Assert.AreEqual(5, d.GetOrAdd("b", k => 6));
            CollectionAssert.AreEqual(new[] { "a", "b" }, d.Keys);
            AssertFullyConsistent(d);
        }

        [TestMethod]
        public void AddOrUpdate_AddsThenUpdates()
        {
            var d = new ConcurrentOrderedDict<string, int>();
            Assert.AreEqual(1, d.AddOrUpdate("k", 1, (k, v) => v + 10));
            Assert.AreEqual(11, d.AddOrUpdate("k", 1, (k, v) => v + 10));
            Assert.AreEqual(11, d.GetByKey("k"));
            Assert.AreEqual(1, d.Count);
        }

        [TestMethod]
        public void TryUpdate_SwapsOnlyOnComparisonMatch()
        {
            var d = new ConcurrentOrderedDict<string, int>();
            d.Add("k", 1);
            Assert.IsFalse(d.TryUpdate("k", 9, comparisonValue: 2)); // stale comparand
            Assert.AreEqual(1, d.GetByKey("k"));
            Assert.IsTrue(d.TryUpdate("k", 9, comparisonValue: 1));
            Assert.AreEqual(9, d.GetByKey("k"));
            Assert.IsFalse(d.TryUpdate("missing", 1, 1));
        }

        [TestMethod]
        public void TryRemove_Interior_ShiftsTail_AndRepairsIndices()
        {
            var d = new ConcurrentOrderedDict<string, int>();
            d.Add("a", 1);
            d.Add("b", 2);
            d.Add("c", 3);
            d.Add("d", 4);

            Assert.IsTrue(d.TryRemove("b", out int removed));
            Assert.AreEqual(2, removed);
            CollectionAssert.AreEqual(new[] { "a", "c", "d" }, d.Keys);
            CollectionAssert.AreEqual(new[] { 1, 3, 4 }, d.ToArray());
            Assert.AreEqual(1, d.IndexOf("c")); // shifted entries re-indexed
            Assert.AreEqual(2, d.IndexOf("d"));
            Assert.IsFalse(d.TryRemove("b", out _));
            AssertFullyConsistent(d);
        }

        [TestMethod]
        public void TryRemove_Tail_IsO1Path_AndStaysConsistent()
        {
            var d = new ConcurrentOrderedDict<int, int>();
            for (int i = 0; i < 100; i++)
                d.Add(i, i * 2);

            // Pop everything from the back through the O(1) tail path.
            for (int i = 99; i >= 0; i--)
            {
                Assert.IsTrue(d.TryRemove(i, out int v));
                Assert.AreEqual(i * 2, v);
                Assert.AreEqual(i, d.Count);
            }

            Assert.IsTrue(d.IsEmpty);
            AssertFullyConsistent(d);
        }

        [TestMethod]
        public void TailRemove_ThenAppend_DoesNotDisturbOlderEnumeratorSnapshot()
        {
            // THE append-floor pin: a tail removal shares its arrays with older snapshots, so the append that
            // reuses the vacated position must copy instead of overwriting the slot the old enumerator can read.
            var d = new ConcurrentOrderedDict<string, int>();
            d.Add("a", 1);
            d.Add("b", 2);
            d.Add("c", 3);

            var enumerator = d.GetEnumerator(); // snapshot at count 3

            Assert.IsTrue(d.TryRemove("c", out _)); // O(1) tail removal (arrays shared, floor raised)
            d.Add("d", 4);                          // would land on c's old slot — must copy-on-write instead

            var seen = new List<int>();
            while (enumerator.MoveNext())
                seen.Add(enumerator.Current);

            CollectionAssert.AreEqual(new[] { 1, 2, 3 }, seen, "the pre-removal snapshot must still see c's value, not d's");
            CollectionAssert.AreEqual(new[] { 1, 2, 4 }, d.ToArray(), "the live view sees the new state");
            AssertFullyConsistent(d);
        }

        [TestMethod]
        public void RemoveAt_RemovesByPosition()
        {
            var d = new ConcurrentOrderedDict<string, int>();
            d.Add("a", 1);
            d.Add("b", 2);
            d.Add("c", 3);

            d.RemoveAt(1);
            CollectionAssert.AreEqual(new[] { "a", "c" }, d.Keys);
            Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => d.RemoveAt(2));
            Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => d.RemoveAt(-1));
            AssertFullyConsistent(d);
        }

        [TestMethod]
        public void TryRemoveSwapBack_MovesLastIntoHole_BreakingOrderOnly()
        {
            var d = new ConcurrentOrderedDict<string, int>();
            d.Add("a", 1);
            d.Add("b", 2);
            d.Add("c", 3);
            d.Add("d", 4);

            Assert.IsTrue(d.TryRemoveSwapBack("b", out int removed));
            Assert.AreEqual(2, removed);
            Assert.AreEqual(3, d.Count);
            CollectionAssert.AreEqual(new[] { "a", "d", "c" }, d.Keys); // d moved into b's hole
            Assert.AreEqual(1, d.IndexOf("d"));                          // mover re-indexed
            Assert.IsFalse(d.ContainsKey("b"));

            // Removing the current last entry degenerates to the tail path.
            Assert.IsTrue(d.TryRemoveSwapBack("c", out _));
            CollectionAssert.AreEqual(new[] { "a", "d" }, d.Keys);

            Assert.IsFalse(d.TryRemoveSwapBack("zz", out _));
            AssertFullyConsistent(d);
        }

        [TestMethod]
        public void TryRemoveSwapBack_NonAtomicTypes_TakesCopyPath_SameSemantics()
        {
            // decimal (16 bytes) is not atomically writable, so the swap must copy the arrays — semantics identical.
            var d = new ConcurrentOrderedDict<decimal, decimal>();
            d.Add(1m, 10m);
            d.Add(2m, 20m);
            d.Add(3m, 30m);

            Assert.IsTrue(d.TryRemoveSwapBack(1m, out decimal removed));
            Assert.AreEqual(10m, removed);
            CollectionAssert.AreEqual(new[] { 3m, 2m }, d.Keys);
            Assert.AreEqual(0, d.IndexOf(3m));
            AssertFullyConsistent(d);
        }

        [TestMethod]
        public void RemoveWhere_CompactsInOnePass_AndReindexes()
        {
            var d = new ConcurrentOrderedDict<int, int>();
            for (int i = 0; i < 10; i++)
                d.Add(i, i);

            int removed = d.RemoveWhere((k, v) => (k & 1) == 0); // drop evens
            Assert.AreEqual(5, removed);
            CollectionAssert.AreEqual(new[] { 1, 3, 5, 7, 9 }, d.Keys);
            AssertFullyConsistent(d);

            Assert.AreEqual(0, d.RemoveWhere((k, v) => false)); // no-match fast path
            Assert.AreEqual(5, d.Count);

            Assert.AreEqual(5, d.RemoveWhere((k, v) => true));
            Assert.IsTrue(d.IsEmpty);
        }

        [TestMethod]
        public void RemoveWhere_ThrowingPredicate_LeavesBothPathsConsistent()
        {
            var d = new ConcurrentOrderedDict<int, int>();
            for (int i = 0; i < 6; i++)
                d.Add(i, i);

            // Predicate removes 0 and 2, then throws at 4: everything matched SO FAR must be removed
            // consistently, everything else (including the throwing entry) must survive on both paths.
            Assert.ThrowsExactly<InvalidOperationException>(() => d.RemoveWhere((k, v) =>
            {
                if (k == 4)
                    throw new InvalidOperationException("boom");
                return (k & 1) == 0;
            }));

            CollectionAssert.AreEqual(new[] { 1, 3, 4, 5 }, d.Keys);
            AssertFullyConsistent(d);
        }

        [TestMethod]
        public void Clear_EmptiesBothPaths_AndAcceptsNewEntries()
        {
            var d = new ConcurrentOrderedDict<string, int>();
            d.Add("a", 1);
            d.Clear();
            Assert.AreEqual(0, d.Count);
            Assert.IsFalse(d.ContainsKey("a"));

            d.Add("b", 2);
            CollectionAssert.AreEqual(new[] { "b" }, d.Keys);
            AssertFullyConsistent(d);
        }

        [TestMethod]
        public void AddRange_UpsertsWithPythonDictSemantics()
        {
            var d = new ConcurrentOrderedDict<string, int>();
            d.Add("x", 0);
            d.AddRange(new[]
            {
                new KeyValuePair<string, int>("a", 1),
                new KeyValuePair<string, int>("x", 99), // existing: value replaced, position kept (index 0)
                new KeyValuePair<string, int>("b", 2),
                new KeyValuePair<string, int>("a", 11), // dup within the batch: last value wins, first position
            });

            CollectionAssert.AreEqual(new[] { "x", "a", "b" }, d.Keys);
            CollectionAssert.AreEqual(new[] { 99, 11, 2 }, d.ToArray());
            AssertFullyConsistent(d);
        }

        [TestMethod]
        public void AddRange_NullKeyMidBatch_Throws_ButAppliedPrefixStaysConsistent()
        {
            var d = new ConcurrentOrderedDict<string, int>();
            var batch = new[]
            {
                new KeyValuePair<string, int>("a", 1),
                new KeyValuePair<string, int>(null!, 2),
                new KeyValuePair<string, int>("c", 3),
            };

            Assert.ThrowsExactly<ArgumentNullException>(() => d.AddRange(batch));

            // The pair before the bad one must be fully visible on BOTH paths (no key that resolves but
            // never enumerates), the pair after must not exist at all.
            CollectionAssert.AreEqual(new[] { "a" }, d.Keys);
            Assert.AreEqual(1, d.GetByKey("a"));
            Assert.IsFalse(d.ContainsKey("c"));
            AssertFullyConsistent(d);
        }

        [TestMethod]
        public void Ctor_FromPairs_MatchesAddRangeSemantics()
        {
            var d = new ConcurrentOrderedDict<string, int>(new[]
            {
                new KeyValuePair<string, int>("a", 1),
                new KeyValuePair<string, int>("b", 2),
                new KeyValuePair<string, int>("a", 3),
            });

            CollectionAssert.AreEqual(new[] { "a", "b" }, d.Keys);
            CollectionAssert.AreEqual(new[] { 3, 2 }, d.ToArray());
        }

        [TestMethod]
        public void Enumerator_IsASnapshot_UnaffectedByLaterAppends()
        {
            var d = new ConcurrentOrderedDict<int, int>();
            d.Add(0, 0);
            d.Add(1, 1);

            var e = d.GetEnumerator();
            d.Add(2, 2); // in-place append bumps the live count, not the captured one

            var seen = new List<int>();
            while (e.MoveNext())
                seen.Add(e.Current);
            CollectionAssert.AreEqual(new[] { 0, 1 }, seen);
            Assert.AreEqual(3, d.Count);
        }

        [TestMethod]
        public void Snapshot_View_IndexesAndSpans_AndIsFixedInShape()
        {
            var d = new ConcurrentOrderedDict<string, int>();
            d.Add("a", 1);
            d.Add("b", 2);

            var view = d.Snapshot();
            d.Add("c", 3); // not visible to the captured view

            Assert.AreEqual(2, view.Count);
            Assert.AreEqual(1, view[0]);
            Assert.AreEqual(2, view[1]);
            Assert.AreEqual("b", view.GetKeyAt(1));
            Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => _ = view[2]);
            Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => view.GetKeyAt(-1));

            Assert.IsTrue(view.AsSpan().SequenceEqual(new[] { 1, 2 }));
            Assert.IsTrue(view.KeysAsSpan().SequenceEqual(new[] { "a", "b" }));

            int sum = 0;
            foreach (int v in view)
                sum += v;
            Assert.AreEqual(3, sum);

            // A default view is a safe empty, never a null-reference trap.
            ConcurrentOrderedDict<string, int>.ValuesView empty = default;
            Assert.AreEqual(0, empty.Count);
            Assert.AreEqual(0, empty.AsSpan().Length);
            Assert.AreEqual(0, empty.KeysAsSpan().Length);
            Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => _ = empty[0]);
        }

        [TestMethod]
        public void Snapshot_View_SeesInPlaceAtomicValueUpdates()
        {
            // Documented live-value semantics: the view shares the array, so an atomic in-place replace shows.
            var d = new ConcurrentOrderedDict<string, int>();
            d.Add("a", 1);
            var view = d.Snapshot();
            d.SetByKey("a", 42);
            Assert.AreEqual(42, view[0]);
        }

        [TestMethod]
        public void CopyTo_ValidatesArguments_AndCopies()
        {
            var d = new ConcurrentOrderedDict<string, int>();
            d.Add("a", 1);
            d.Add("b", 2);

            var target = new int[4];
            d.CopyTo(target, 1);
            CollectionAssert.AreEqual(new[] { 0, 1, 2, 0 }, target);

            Assert.ThrowsExactly<ArgumentNullException>(() => d.CopyTo(null!, 0));
            Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => d.CopyTo(target, -1));
            Assert.ThrowsExactly<ArgumentException>(() => d.CopyTo(new int[1], 0));
        }

        [TestMethod]
        public void NonAtomicValues_Decimal_ReplaceAndRemove_StayCorrect()
        {
            // decimal exercises the tear-free node-swap + value-array-clone branches everywhere.
            var d = new ConcurrentOrderedDict<string, decimal>();
            d.Add("a", 1.5m);
            d.Add("b", 2.5m);

            d.SetByKey("a", 9.25m);
            Assert.AreEqual(9.25m, d.GetByKey("a"));
            Assert.AreEqual(9.25m, d[0]);

            d.SetAt(1, 7.75m);
            Assert.AreEqual(7.75m, d.GetByKey("b"));

            Assert.IsTrue(d.TryUpdate("b", 8m, 7.75m));
            Assert.AreEqual(8m, d.GetByKey("b"));

            Assert.IsTrue(d.TryRemove("a", out decimal removed));
            Assert.AreEqual(9.25m, removed);
            CollectionAssert.AreEqual(new[] { "b" }, d.Keys);
            AssertFullyConsistent(d);
        }

        [TestMethod]
        public void CustomComparer_DedupsAccordingly()
        {
            var d = new ConcurrentOrderedDict<string, int>(StringComparer.OrdinalIgnoreCase);
            d.Add("Key", 1);
            Assert.IsFalse(d.TryAdd("KEY", 2));
            Assert.AreEqual(1, d.GetByKey("key"));
            Assert.AreSame(StringComparer.OrdinalIgnoreCase, d.Comparer);
        }

        [TestMethod]
        public void Growth_FromEmptyAndFromCapacity_KeepsEverything()
        {
            foreach (var d in new[] { new ConcurrentOrderedDict<int, int>(), new ConcurrentOrderedDict<int, int>(3) })
            {
                const int n = 1000; // crosses many doubling boundaries
                for (int i = 0; i < n; i++)
                    d.Add(i, i * 3);

                Assert.AreEqual(n, d.Count);
                for (int i = 0; i < n; i++)
                {
                    Assert.AreEqual(i * 3, d.GetByKey(i));
                    Assert.AreEqual(i, d.IndexOf(i));
                }

                AssertFullyConsistent(d);
            }
        }

        [TestMethod]
        public void MixedChurn_AddRemoveReAdd_KeepsBothPathsAligned()
        {
            var d = new ConcurrentOrderedDict<int, int>();
            var reference = new List<KeyValuePair<int, int>>(); // an oracle: ordered upsert list
            var rng = new System.Random(1234);

            for (int op = 0; op < 5000; op++)
            {
                int k = rng.Next(200);
                switch (rng.Next(5))
                {
                    case 0 or 1: // upsert
                        d.SetByKey(k, op);
                        int at = reference.FindIndex(p => p.Key == k);
                        if (at < 0)
                            reference.Add(new KeyValuePair<int, int>(k, op));
                        else
                            reference[at] = new KeyValuePair<int, int>(k, op);
                        break;
                    case 2: // ordered remove
                        bool removed = d.TryRemove(k, out _);
                        int ri = reference.FindIndex(p => p.Key == k);
                        Assert.AreEqual(ri >= 0, removed);
                        if (ri >= 0)
                            reference.RemoveAt(ri);
                        break;
                    case 3: // add-if-absent
                        bool added = d.TryAdd(k, op);
                        Assert.AreEqual(reference.All(p => p.Key != k), added);
                        if (added)
                            reference.Add(new KeyValuePair<int, int>(k, op));
                        break;
                    case 4: // positional value overwrite
                        if (reference.Count > 0)
                        {
                            int idx = rng.Next(reference.Count);
                            d.SetAt(idx, op);
                            reference[idx] = new KeyValuePair<int, int>(reference[idx].Key, op);
                        }

                        break;
                }
            }

            CollectionAssert.AreEqual(reference.Select(p => p.Key).ToArray(), d.Keys);
            CollectionAssert.AreEqual(reference.Select(p => p.Value).ToArray(), d.ToArray());
            AssertFullyConsistent(d);
        }
    }
}
