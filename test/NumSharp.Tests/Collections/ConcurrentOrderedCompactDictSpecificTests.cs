#nullable enable

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Collections;

namespace NumSharp.Tests.Collections
{
    /// <summary>
    ///     Pins for what is specific to <see cref="ConcurrentOrderedCompactDict{TKey,TValue}" /> beyond the contract it
    ///     shares with its sibling (whose four gates run against it verbatim as the mirrored test classes): the
    ///     bit-tagged small-key path and the types excluded from it, hash-tag collisions, the dummy/rebuild cycle of
    ///     the open-addressed index, the whole-generation copy for wide values, the index capacity ceiling, and the
    ///     concurrency gun that proves the key path stays tear-free under a racing in-place swap-back — the property
    ///     the compact design had to earn.
    /// </summary>
    [TestClass]
    public class ConcurrentOrderedCompactDictSpecificTests
    {
        /// <summary>The value law the guns check on every hit: a value that belongs to another key is detectable.</summary>
        /// <param name="k">The key.</param>
        /// <returns>The only value ever stored under <paramref name="k" />.</returns>
        private static int F(int k) => k * 7 + 3;

        /// <summary>Quiescent audit: positions round-trip key↔index and both value paths agree.</summary>
        /// <typeparam name="TKey">The key type.</typeparam>
        /// <typeparam name="TValue">The value type.</typeparam>
        /// <param name="d">The instance to audit.</param>
        private static void AssertFullyConsistent<TKey, TValue>(ConcurrentOrderedCompactDict<TKey, TValue> d)
            where TKey : notnull
        {
            int n = d.Count;
            TKey[] keys = d.Keys;
            Assert.AreEqual(n, keys.Length);
            for (int i = 0; i < n; i++)
            {
                TKey k = d.GetKeyAt(i);
                Assert.AreEqual(keys[i], k);
                Assert.AreEqual(i, d.IndexOf(k), $"IndexOf(GetKeyAt({i})) must round-trip");
                Assert.IsTrue(d.ContainsKey(k));
                Assert.IsTrue(d.TryGetValue(k, out TValue byKey));
                Assert.IsTrue(d.TryGetAt(i, out TValue byIndex));
                Assert.AreEqual(byKey, byIndex, $"key path and list path disagree at {i}");
            }
        }

        /// <summary>An enum with a 4-byte underlying type — bit-tagged.</summary>
        private enum IntEnum
        {
            Zero = 0,
            Neg = -7,
            Big = int.MaxValue,
            Small = int.MinValue,
        }

        /// <summary>An enum with a 1-byte underlying type — bit-tagged.</summary>
        private enum ByteEnum : byte
        {
            A = 0,
            B = 1,
            Z = 255,
        }

        // ------------------------------------------------------------------------------ bit-tagged keys

        [TestMethod]
        public void SmallValueTypeKeys_AreBitTagged_AndRoundTripEveryExtreme()
        {
            // Each of these key types tags the index word with the key's own bits (a hit never touches the keys
            // array); the extremes and negatives must survive the sign-extension into the word's high half.
            RoundTrip(new[] { short.MinValue, (short)-1, (short)0, (short)1, short.MaxValue, (short)-32000, (short)12345 });
            RoundTrip(new[] { ushort.MinValue, (ushort)1, (ushort)0x8000, ushort.MaxValue });
            RoundTrip(new[] { sbyte.MinValue, (sbyte)-1, (sbyte)0, (sbyte)1, sbyte.MaxValue });
            RoundTrip(new[] { byte.MinValue, (byte)1, (byte)0x80, byte.MaxValue });
            RoundTrip(new[] { char.MinValue, 'a', 'Z', '耀', char.MaxValue });
            RoundTrip(new[] { false, true });
            RoundTrip(new[] { int.MinValue, -1, 0, 1, int.MaxValue, -123456, 654321 });
            RoundTrip(new[] { uint.MinValue, 1u, 0x80000000u, uint.MaxValue });
            RoundTrip(new[] { IntEnum.Zero, IntEnum.Neg, IntEnum.Big, IntEnum.Small });
            RoundTrip(new[] { ByteEnum.A, ByteEnum.B, ByteEnum.Z });
        }

        /// <summary>Adds every key with a distinct value, then audits lookups, positions, swap-back and ordered removal.</summary>
        /// <typeparam name="TKey">The key type under test.</typeparam>
        /// <param name="keys">Distinct keys.</param>
        private static void RoundTrip<TKey>(TKey[] keys)
            where TKey : notnull
        {
            var d = new ConcurrentOrderedCompactDict<TKey, int>();
            for (int i = 0; i < keys.Length; i++)
            {
                Assert.IsTrue(d.TryAdd(keys[i], i * 11), $"{typeof(TKey).Name}: {keys[i]} must add");
                Assert.IsFalse(d.TryAdd(keys[i], -1), $"{typeof(TKey).Name}: {keys[i]} must dedup");
            }

            Assert.AreEqual(keys.Length, d.Count);
            for (int i = 0; i < keys.Length; i++)
            {
                Assert.IsTrue(d.TryGetValue(keys[i], out int v) && v == i * 11, $"{typeof(TKey).Name}: {keys[i]} lookup");
                Assert.AreEqual(i, d.IndexOf(keys[i]));
            }

            AssertFullyConsistent(d);

            // Swap-back the first key: the last moves into slot 0, everyone else stays put.
            Assert.IsTrue(d.TryRemoveSwapBack(keys[0], out int removed) && removed == 0);
            Assert.IsFalse(d.ContainsKey(keys[0]));
            Assert.AreEqual(0, d.IndexOf(keys[^1]));
            AssertFullyConsistent(d);

            // Ordered removal of the rest, one by one, keeping consistency at every step.
            while (d.Count > 0)
            {
                TKey k = d.GetKeyAt(d.Count / 2);
                Assert.IsTrue(d.TryRemove(k, out _));
                Assert.IsFalse(d.ContainsKey(k));
                AssertFullyConsistent(d);
            }
        }

        [TestMethod]
        public void FloatKeys_AreNotBitTagged_SoSignedZeroAndNaNFollowTheComparer()
        {
            // -0.0f == 0.0f and NaN.Equals(NaN) under the default comparer — bit-equality would get both wrong, so
            // float keys must hash instead of embedding their bits.
            var d = new ConcurrentOrderedCompactDict<float, string>();
            Assert.IsTrue(d.TryAdd(0.0f, "zero"));
            Assert.IsFalse(d.TryAdd(-0.0f, "negative zero"), "-0.0f must be the same key as 0.0f");
            Assert.IsTrue(d.TryGetValue(-0.0f, out string? z) && z == "zero");

            Assert.IsTrue(d.TryAdd(float.NaN, "nan"));
            Assert.IsFalse(d.TryAdd(float.NaN, "nan again"), "NaN must equal itself as a key");
            Assert.IsTrue(d.TryGetValue(float.NaN, out string? n) && n == "nan");
            Assert.AreEqual(2, d.Count);
            AssertFullyConsistent(d);
        }

        /// <summary>A comparer that identifies a number with its negation, so bit-equality and key-equality disagree.</summary>
        private sealed class AbsComparer : IEqualityComparer<int>
        {
            /// <inheritdoc />
            public bool Equals(int x, int y) => System.Math.Abs(x) == System.Math.Abs(y);

            /// <inheritdoc />
            public int GetHashCode(int obj) => System.Math.Abs(obj);
        }

        [TestMethod]
        public void CustomComparer_DisablesTheBitTag_AndDrivesEqualityAndHashing()
        {
            var d = new ConcurrentOrderedCompactDict<int, string>(new AbsComparer());
            Assert.IsTrue(d.TryAdd(5, "five"));
            Assert.IsFalse(d.TryAdd(-5, "minus five"), "the comparer says -5 is 5");
            Assert.IsTrue(d.ContainsKey(-5));
            Assert.IsTrue(d.TryGetValue(-5, out string? v) && v == "five");
            Assert.AreEqual(0, d.IndexOf(-5));
            d.SetByKey(-5, "still five");
            Assert.AreEqual("still five", d.GetByKey(5));
            Assert.IsTrue(d.TryRemove(-5, out _));
            Assert.IsTrue(d.IsEmpty);
            Assert.AreSame(typeof(AbsComparer), d.Comparer.GetType());
        }

        // ------------------------------------------------------------------------------ hash-tagged keys and collisions

        [TestMethod]
        public void LongKeys_WithEqualHashes_AreDisambiguatedByTheKeyCompare()
        {
            // long.GetHashCode folds the halves: 1 and 1<<32 collide on the tag, so the key compare must decide.
            var d = new ConcurrentOrderedCompactDict<long, int>();
            long a = 1L, b = 1L << 32, c = (1L << 32) | 1L;
            Assert.AreEqual(a.GetHashCode(), b.GetHashCode(), "the test needs a genuine tag collision");
            Assert.IsTrue(d.TryAdd(a, 1) && d.TryAdd(b, 2) && d.TryAdd(c, 3));
            Assert.IsTrue(d.TryGetValue(a, out int va) && va == 1);
            Assert.IsTrue(d.TryGetValue(b, out int vb) && vb == 2);
            Assert.IsTrue(d.TryGetValue(c, out int vc) && vc == 3);
            Assert.IsTrue(d.TryRemove(a, out _));
            Assert.IsFalse(d.ContainsKey(a));
            Assert.IsTrue(d.TryGetValue(b, out vb) && vb == 2, "removing a collider must not hide its neighbour");
            Assert.IsTrue(d.TryRemoveSwapBack(b, out _));
            Assert.IsTrue(d.TryGetValue(c, out vc) && vc == 3);
            AssertFullyConsistent(d);
        }

        /// <summary>A comparer whose hash is constant: every key lands on one probe run, the worst case for open addressing.</summary>
        private sealed class ConstantHashComparer : IEqualityComparer<string>
        {
            /// <inheritdoc />
            public bool Equals(string? x, string? y) => string.Equals(x, y, StringComparison.Ordinal);

            /// <inheritdoc />
            public int GetHashCode(string obj) => 42;
        }

        [TestMethod]
        public void ConstantHash_EveryKeyOnOneProbeRun_StaysCorrectThroughRemovalsAndRebuilds()
        {
            var d = new ConcurrentOrderedCompactDict<string, int>(new ConstantHashComparer());
            const int N = 300;
            for (int i = 0; i < N; i++)
            {
                d.Add("k" + i, i);
            }

            for (int i = 0; i < N; i++)
            {
                Assert.IsTrue(d.TryGetValue("k" + i, out int v) && v == i);
            }

            Assert.IsFalse(d.ContainsKey("absent"), "a miss must terminate at the run's end even with 300 colliders");

            // Swap-back every even key (dummies pile up on the run), then ordered-remove every remaining key ≥ 100.
            for (int i = 0; i < N; i += 2)
            {
                Assert.IsTrue(d.TryRemoveSwapBack("k" + i, out int removed) && removed == i);
            }

            Assert.AreEqual(N / 2, d.Count);
            for (int i = 1; i < N; i += 2)
            {
                Assert.IsTrue(d.TryGetValue("k" + i, out int v) && v == i, $"k{i} lost after the swap-backs");
            }

            // The 150 survivors are the odd keys; 100 of them carry values ≥ 100.
            Assert.AreEqual(100, d.RemoveWhere((k, v) => v >= 100));
            Assert.AreEqual(50, d.Count);
            for (int i = 1; i < 100; i += 2)
            {
                Assert.IsTrue(d.TryGetValue("k" + i, out int v) && v == i);
            }

            Assert.IsFalse(d.ContainsKey("k101"));

            // Re-adding after the dummies forces the load-triggered rebuild on the same run.
            for (int i = 0; i < N; i++)
            {
                d.SetByKey("k" + i, -i);
            }

            Assert.AreEqual(N, d.Count);
            Assert.IsFalse(d.ContainsKey("absent"));
            AssertFullyConsistent(d);
        }

        // ------------------------------------------------------------------------------ dummies, floors, rebuilds

        [TestMethod]
        public void PopPushChurn_AndSwapBackChurn_KeepProbesTerminating_AndStateConsistent()
        {
            // Every pop dummies a word and every push after it copies (the floor rule); every swap-back dummies a
            // word in place. Dummies must never make a miss probe spin and must be cleared by the rebuilds.
            var d = new ConcurrentOrderedCompactDict<int, int>(16);
            for (int i = 0; i < 12; i++)
            {
                d.Add(i, F(i));
            }

            for (int cycle = 0; cycle < 5_000; cycle++)
            {
                int k = 1_000 + cycle;
                Assert.IsTrue(d.TryAdd(k, F(k)));
                Assert.IsTrue(d.TryRemove(k, out int v) && v == F(k)); // tail pop
                Assert.IsFalse(d.ContainsKey(k), "a popped key must miss");
                Assert.IsFalse(d.ContainsKey(-1 - cycle), "an absent key must miss (the probe must terminate)");
            }

            Assert.AreEqual(12, d.Count);
            AssertFullyConsistent(d);

            var rng = new System.Random(3);
            var live = new List<int>(Enumerable.Range(0, 12));
            int next = 50_000;
            for (int cycle = 0; cycle < 20_000; cycle++)
            {
                int victimAt = rng.Next(live.Count);
                int victim = live[victimAt];
                Assert.IsTrue(d.TryRemoveSwapBack(victim, out int v) && v == F(victim));
                live[victimAt] = live[^1];
                live.RemoveAt(live.Count - 1);

                int k = next++;
                Assert.IsTrue(d.TryAdd(k, F(k)));
                live.Add(k);

                if ((cycle & 1023) == 0)
                {
                    foreach (int p in live)
                    {
                        Assert.IsTrue(d.TryGetValue(p, out int pv) && pv == F(p));
                    }

                    Assert.IsFalse(d.ContainsKey(-1));
                    AssertFullyConsistent(d);
                }
            }

            Assert.AreEqual(12, d.Count);
            CollectionAssert.AreEquivalent(live, d.Keys);
            AssertFullyConsistent(d);
        }

        [TestMethod]
        public void InteriorRemovals_ManyInARow_TriggerTheCleanRebuild_AndStayConsistent()
        {
            var d = new ConcurrentOrderedCompactDict<int, int>();
            for (int i = 0; i < 500; i++)
            {
                d.Add(i, F(i));
            }

            // Remove from the front until fewer live entries than dummies would remain — the copy switches from the
            // renumbering pass to a clean rebuild; either way the result must be exact.
            for (int i = 0; i < 400; i++)
            {
                Assert.IsTrue(d.TryRemove(i, out int v) && v == F(i));
                Assert.IsFalse(d.ContainsKey(i));
            }

            Assert.AreEqual(100, d.Count);
            for (int i = 400; i < 500; i++)
            {
                Assert.IsTrue(d.TryGetValue(i, out int v) && v == F(i));
                Assert.AreEqual(i - 400, d.IndexOf(i));
            }

            AssertFullyConsistent(d);

            for (int i = 0; i < 400; i++)
            {
                d.Add(i, F(i)); // re-adds append at the tail
            }

            Assert.AreEqual(500, d.Count);
            AssertFullyConsistent(d);
        }

        // ------------------------------------------------------------------------------ wide types

        [TestMethod]
        public void GuidKeys_SwapBackTakesTheCopyPath_WithIdenticalSemantics()
        {
            var d = new ConcurrentOrderedCompactDict<Guid, int>();
            var keys = Enumerable.Range(0, 6).Select(_ => Guid.NewGuid()).ToArray();
            for (int i = 0; i < keys.Length; i++)
            {
                d.Add(keys[i], i);
            }

            var view = d.Snapshot();
            Assert.IsTrue(d.TryRemoveSwapBack(keys[1], out int removed) && removed == 1);
            CollectionAssert.AreEqual(new[] { keys[0], keys[5], keys[2], keys[3], keys[4] }, d.Keys);
            Assert.AreEqual(1, d.IndexOf(keys[5]));
            Assert.AreEqual(6, view.Count);
            Assert.AreEqual(keys[1], view.GetKeyAt(1)); // the copy path leaves the captured snapshot pristine
            Assert.AreEqual(1, view[1]);
            AssertFullyConsistent(d);
        }

        [TestMethod]
        public void DecimalValues_ReplaceCopiesTheWholeGeneration_SoACapturedViewKeepsTheOldValue()
        {
            var d = new ConcurrentOrderedCompactDict<int, decimal>();
            for (int i = 0; i < 8; i++)
            {
                d.Add(i, i);
            }

            var view = d.Snapshot();
            d.SetByKey(3, 3.5m);
            d.SetAt(5, 5.5m);
            Assert.AreEqual(3m, view[3], "a wide value is never rewritten in place: the old generation keeps the old value");
            Assert.AreEqual(5m, view[5]);
            Assert.AreEqual(3.5m, d.GetByKey(3));
            Assert.AreEqual(5.5m, d[5]);
            Assert.IsTrue(d.TryRemoveSwapBack(0, out _));
            Assert.AreEqual(0m, view[0]);
            AssertFullyConsistent(d);

            // Contrast: an atomic value IS visible through a captured view (the documented live-value semantics).
            var a = new ConcurrentOrderedCompactDict<int, long>();
            a.Add(1, 10);
            var av = a.Snapshot();
            a.SetByKey(1, 11);
            Assert.AreEqual(11L, av[0]);
        }

        // ------------------------------------------------------------------------------ limits

        [TestMethod]
        public void Capacity_BeyondTheIndexCeiling_ThrowsBeforeAllocating()
            => Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => _ = new ConcurrentOrderedCompactDict<int, int>(800_000_000));

        // ------------------------------------------------------------------------------ the concurrency gun

        /// <summary>Gun size: enough threads to contend hard even on small CI boxes, capped so the box is not drowned.</summary>
        private static int GunThreads => System.Math.Max(4, System.Math.Min(Environment.ProcessorCount, 8));

        /// <summary>Arms <paramref name="threads" /> shooters on one gate, fires them together, joins, and rethrows every failure.</summary>
        /// <param name="threads">Shooter count.</param>
        /// <param name="body">Per-thread workload; receives the 0-based thread id.</param>
        /// <exception cref="AggregateException">One or more shooters threw.</exception>
        private static void FireGun(int threads, Action<int> body)
        {
            using var go = new ManualResetEventSlim(false);
            using var ready = new CountdownEvent(threads);
            var errors = new ConcurrentQueue<Exception>();
            var shooters = new Thread[threads];
            for (int t = 0; t < threads; t++)
            {
                int id = t;
                shooters[t] = new Thread(() =>
                {
                    ready.Signal();
                    go.Wait();
                    try
                    {
                        body(id);
                    }
                    catch (Exception ex)
                    {
                        errors.Enqueue(ex);
                    }
                })
                {
                    IsBackground = true,
                    Name = $"cocd-specific-gun-{t}",
                };
                shooters[t].Start();
            }

            ready.Wait();
            go.Set();
            foreach (Thread s in shooters)
            {
                Assert.IsTrue(s.Join(TimeSpan.FromMinutes(2)), $"{s.Name} did not finish — deadlock or livelock");
            }

            if (!errors.IsEmpty)
            {
                throw new AggregateException(errors);
            }
        }

        /// <summary>
        ///     The swap-back gun: one writer swap-back-removes/re-adds 16 hot keys and appends pinned keys that the
        ///     next swap-back moves; the other threads read those keys as fast as they can. On every hit the value must
        ///     obey the value law (the key path never pairs a key with another key's value), a published pinned key
        ///     must never be absent (a moved key is always resolvable), and a list-path scan must never see a value
        ///     outside the law (no torn slots).
        /// </summary>
        /// <typeparam name="TKey">The key type (drives the bit-tag vs. hash-tag read path).</typeparam>
        /// <param name="d">The empty instance under fire.</param>
        /// <param name="toKey">Maps the gun's int key space onto <typeparamref name="TKey" />.</param>
        /// <param name="milliseconds">How long to fire.</param>
        private static void SwapBackGun<TKey>(ConcurrentOrderedCompactDict<TKey, int> d, Func<int, TKey> toKey, int milliseconds)
            where TKey : notnull
        {
            for (int k = 0; k < 16; k++)
            {
                d.Add(toKey(k), F(k));
            }

            int pinnedCounter = 1_000_000;
            int movedCandidate = 0;
            long wrongPair = 0, absent = 0, torn = 0, hits = 0;
            var stopAt = DateTime.UtcNow.AddMilliseconds(milliseconds);
            int threads = GunThreads;

            FireGun(threads, id =>
            {
                var rng = new System.Random(101 + id);
                if (id == 0)
                {
                    // The writer.
                    long ops = 0;
                    while (DateTime.UtcNow < stopAt)
                    {
                        int k = rng.Next(16);
                        if (!d.TryRemoveSwapBack(toKey(k), out _))
                        {
                            d.TryAdd(toKey(k), F(k));
                        }

                        if ((++ops & 63) == 0)
                        {
                            int p = ++pinnedCounter;
                            d.TryAdd(toKey(p), F(p));
                            Volatile.Write(ref movedCandidate, p); // never removed: readers may assert it from now on
                        }
                    }

                    return;
                }

                long myHits = 0, myWrong = 0, myAbsent = 0, myTorn = 0, i = 0;
                while (DateTime.UtcNow < stopAt)
                {
                    int k = rng.Next(16);
                    if (d.TryGetValue(toKey(k), out int v))
                    {
                        myHits++;
                        if (v != F(k))
                        {
                            myWrong++;
                        }
                    }

                    int p = Volatile.Read(ref movedCandidate);
                    if (p != 0)
                    {
                        if (!d.TryGetValue(toKey(p), out int pv))
                        {
                            myAbsent++;
                        }
                        else if (pv != F(p))
                        {
                            myWrong++;
                        }

                        if (d.IndexOf(toKey(p)) < 0)
                        {
                            myAbsent++;
                        }
                    }

                    if ((++i & 2047) == 0)
                    {
                        foreach (int sv in d.Snapshot())
                        {
                            if ((sv - 3) % 7 != 0)
                            {
                                myTorn++;
                            }
                        }
                    }
                }

                Interlocked.Add(ref hits, myHits);
                Interlocked.Add(ref wrongPair, myWrong);
                Interlocked.Add(ref absent, myAbsent);
                Interlocked.Add(ref torn, myTorn);
            });

            Assert.IsTrue(hits > 1_000, $"the gun did not fire ({hits} hits)");
            Assert.AreEqual(0L, wrongPair, "the key path returned another key's value under a racing swap-back");
            Assert.AreEqual(0L, absent, "a moved (pinned) key was transiently unresolvable");
            Assert.AreEqual(0L, torn, "a list-path scan saw a value outside the value law");
            AssertFullyConsistent(d);
        }

        [TestMethod]
        public void Gun_SwapBackRacing_BitTaggedIntKeys_KeyPathNeverTears()
            => SwapBackGun(new ConcurrentOrderedCompactDict<int, int>(), k => k, milliseconds: 1_500);

        [TestMethod]
        public void Gun_SwapBackRacing_HashTaggedStringKeys_KeyPathNeverTears()
            => SwapBackGun(new ConcurrentOrderedCompactDict<string, int>(), k => "key-" + k, milliseconds: 1_500);

        [TestMethod]
        public void Gun_SwapBackRacing_LongKeysWithLongValues_KeyPathNeverTears()
        {
            // 8-byte key and value: still in-place on 64-bit (both atomic), so the validated read is what protects it.
            var d = new ConcurrentOrderedCompactDict<long, int>();
            SwapBackGun(d, k => (long)k << 20, milliseconds: 1_000);
        }
    }
}
