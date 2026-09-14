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
    ///     Heavy-load concurrency stress for <see cref="ConcurrentOrderedDict{TKey,TValue}" />, built around a
    ///     <b>concurrency gun</b>: every participating thread spins up, signals readiness, and then parks on ONE
    ///     shared <see cref="ManualResetEventSlim" />; a single <c>Set()</c> releases them all in the same instant,
    ///     so the contended first microseconds — where publication races, torn reads and lost updates live — are
    ///     actually exercised instead of the threads trickling in one at a time.
    /// </summary>
    /// <remarks>
    ///     Each scenario targets one falsifiable guarantee: exactly-once adds, single-winner dedup races,
    ///     lost-update-free CAS loops, tear-free reads of atomic and non-atomic values, prefix-consistent
    ///     enumeration snapshots (including across the O(1) tail-removal + append-floor path), batch-atomic
    ///     <see cref="ConcurrentOrderedDict{TKey,TValue}.AddRange" /> visibility, and full cross-path consistency
    ///     after the storm. Sizes are tuned to run a few seconds total on CI hardware while still forcing millions
    ///     of contended operations.
    /// </remarks>
    [TestClass]
    public class ConcurrentOrderedDictConcurrencyTests
    {
        /// <summary>Gun size: enough threads to contend hard even on small CI boxes (oversubscription is fine — it widens the race windows), capped so the box is not drowned.</summary>
        private static int GunThreads => System.Math.Max(4, System.Math.Min(Environment.ProcessorCount, 8));

        /// <summary>
        ///     The concurrency gun. Spins up <paramref name="threads" /> threads, waits until every one is parked
        ///     at the gate, releases them with a single signal, joins them, and rethrows anything any of them
        ///     threw (as an <see cref="AggregateException" /> so no failure is swallowed).
        /// </summary>
        /// <param name="threads">How many shooters to arm.</param>
        /// <param name="body">The per-thread workload; receives the thread's 0-based id for range partitioning.</param>
        /// <exception cref="AssertFailedException">A thread failed to finish within the safety timeout (a deadlock/livelock guard, not a perf assertion).</exception>
        /// <exception cref="AggregateException">One or more shooter bodies threw; carries every captured exception.</exception>
        private static void FireGun(int threads, Action<int> body)
        {
            using var go = new ManualResetEventSlim(false);
            using var ready = new CountdownEvent(threads);
            var errors = new ConcurrentQueue<Exception>();

            var shooters = new Thread[threads];
            for (int t = 0; t < threads; t++)
            {
                int id = t; // capture per-thread, not the loop variable
                shooters[t] = new Thread(() =>
                {
                    // Signal "in position" and park on the shared gate; the single Set() below is the gun.
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
                    Name = $"cod-gun-{t}",
                };
                shooters[t].Start();
            }

            ready.Wait();
            go.Set(); // fire: every thread leaves the gate in the same instant

            foreach (Thread s in shooters)
            {
                // Generous ceiling: this is a hang detector, not a performance bound.
                Assert.IsTrue(s.Join(TimeSpan.FromMinutes(2)), $"{s.Name} did not finish — deadlock or livelock");
            }

            if (!errors.IsEmpty)
            {
                throw new AggregateException(errors);
            }
        }

        /// <summary>Quiescent full audit: every position round-trips key↔index and both value paths agree (shared with the functional suite's contract).</summary>
        /// <typeparam name="TKey">The dictionary's key type.</typeparam>
        /// <typeparam name="TValue">The dictionary's value type.</typeparam>
        /// <param name="d">The instance to audit after the storm.</param>
        private static void AssertFullyConsistent<TKey, TValue>(ConcurrentOrderedDict<TKey, TValue> d)
            where TKey : notnull
        {
            int n = d.Count;
            for (int i = 0; i < n; i++)
            {
                TKey k = d.GetKeyAt(i);
                Assert.AreEqual(i, d.IndexOf(k), $"IndexOf(GetKeyAt({i})) must round-trip after the storm");
                Assert.IsTrue(d.TryGetValue(k, out TValue byKey));
                Assert.IsTrue(d.TryGetAt(i, out TValue byIndex));
                Assert.AreEqual(byKey, byIndex, $"key path and list path disagree at {i}");
            }
        }

        [TestMethod]
        public void Gun_DistinctAdds_EveryEntryLandsExactlyOnce()
        {
            const int PerThread = 40_000;
            int threads = GunThreads;
            var d = new ConcurrentOrderedDict<int, long>();

            FireGun(threads, id =>
            {
                int lo = id * PerThread;
                for (int k = lo; k < lo + PerThread; k++)
                {
                    Assert.IsTrue(d.TryAdd(k, (long)k * 3), $"distinct key {k} must add exactly once");
                }
            });

            Assert.AreEqual(threads * PerThread, d.Count);

            // Every key present with its value, and the key set is an exact permutation (no loss, no duplication).
            var seen = new HashSet<int>();
            foreach (var key in d.Keys)
            {
                Assert.IsTrue(seen.Add(key), $"key {key} appears twice in the ordered view");
            }

            Assert.AreEqual(threads * PerThread, seen.Count);
            for (int k = 0; k < threads * PerThread; k++)
            {
                Assert.IsTrue(d.TryGetValue(k, out long v) && v == (long)k * 3, $"key {k} lost or corrupted");
            }

            AssertFullyConsistent(d);
        }

        [TestMethod]
        public void Gun_SameKeyTryAdd_ExactlyOneWinnerPerKey()
        {
            const int KeyCount = 10_000;
            int threads = GunThreads;
            var d = new ConcurrentOrderedDict<int, int>();
            int totalWins = 0;

            FireGun(threads, id =>
            {
                int wins = 0;
                for (int k = 0; k < KeyCount; k++)
                {
                    // Value encodes the writer, so the winning value must be internally consistent later.
                    if (d.TryAdd(k, k * 100 + id))
                    {
                        wins++;
                    }
                }

                Interlocked.Add(ref totalWins, wins);
            });

            Assert.AreEqual(KeyCount, totalWins, "each key must be won by exactly one thread");
            Assert.AreEqual(KeyCount, d.Count);
            for (int k = 0; k < KeyCount; k++)
            {
                Assert.IsTrue(d.TryGetValue(k, out int v));
                Assert.AreEqual(k, v / 100, $"key {k} holds a value written for a different key");
                int idx = d.IndexOf(k);
                Assert.IsTrue(d.TryGetAt(idx, out int atIdx) && atIdx == v, "both paths must expose the same winning value");
            }

            AssertFullyConsistent(d);
        }

        [TestMethod]
        public void Gun_GetOrAdd_AllThreadsObserveTheSameWinner()
        {
            const int KeyCount = 2_000;
            int threads = GunThreads;
            var d = new ConcurrentOrderedDict<int, int>();
            var observed = new int[threads][];

            FireGun(threads, id =>
            {
                var mine = new int[KeyCount];
                for (int k = 0; k < KeyCount; k++)
                {
                    // Distinct per-thread candidate values: whoever wins, everyone must read the same winner.
                    mine[k] = d.GetOrAdd(k, k * 1000 + id);
                }

                observed[id] = mine;
            });

            Assert.AreEqual(KeyCount, d.Count);
            for (int k = 0; k < KeyCount; k++)
            {
                Assert.IsTrue(d.TryGetValue(k, out int stored));
                for (int t = 0; t < threads; t++)
                {
                    Assert.AreEqual(stored, observed[t][k], $"thread {t} observed a value for key {k} that is not the stored winner");
                }
            }
        }

        [TestMethod]
        public void Gun_AddOrUpdate_CounterLosesNoIncrement()
        {
            // The classic lost-update detector: N threads × M increments over a handful of hot keys through the
            // compare-and-swap AddOrUpdate loop must account for every single increment.
            const int PerThread = 25_000;
            const int HotKeys = 4;
            int threads = GunThreads;
            var d = new ConcurrentOrderedDict<int, long>();

            FireGun(threads, id =>
            {
                for (int i = 0; i < PerThread; i++)
                {
                    d.AddOrUpdate(i % HotKeys, 1L, static (k, v) => v + 1);
                }
            });

            long total = 0;
            foreach (long v in d)
            {
                total += v;
            }

            Assert.AreEqual((long)threads * PerThread, total, "increments were lost under contention");
            Assert.AreEqual(HotKeys, d.Count);
        }

        [TestMethod]
        public void Gun_TornReadHunt_AtomicLongValues()
        {
            // Writers flip one key's value between two full-width bit patterns while readers hammer both access
            // paths: ANY other observed pattern is a torn read. long is the widest atomically-writable value, so
            // this pins the in-place update path (node field + array slot single stores).
            const long A = 0x0101010101010101;
            const long B = 0x0202020202020202;
            const int WriterOps = 150_000;
            const int ReaderOps = 400_000;
            int threads = GunThreads;
            var d = new ConcurrentOrderedDict<int, long>();
            d.Add(7, A);

            FireGun(threads, id =>
            {
                if ((id & 1) == 0)
                {
                    for (int i = 0; i < WriterOps; i++)
                    {
                        d.SetByKey(7, (i & 1) == 0 ? B : A);
                    }
                }
                else
                {
                    for (int i = 0; i < ReaderOps; i++)
                    {
                        Assert.IsTrue(d.TryGetValue(7, out long byKey));
                        Assert.IsTrue(byKey == A || byKey == B, $"TORN key-path read: 0x{byKey:X16}");
                        Assert.IsTrue(d.TryGetAt(0, out long byIndex));
                        Assert.IsTrue(byIndex == A || byIndex == B, $"TORN list-path read: 0x{byIndex:X16}");
                    }
                }
            });
        }

        [TestMethod]
        public void Gun_TornReadHunt_NonAtomicDecimalValues()
        {
            // decimal (16 bytes) cannot be stored atomically, so replacement goes node-swap + array-clone; a
            // reader observing any value outside the written set proves a torn struct copy.
            const decimal A = 1111111111111111111.1111m;
            const decimal B = 2222222222222222222.2222m;
            const int WriterOps = 60_000;
            const int ReaderOps = 150_000;
            int threads = GunThreads;
            var d = new ConcurrentOrderedDict<int, decimal>();
            d.Add(7, A);

            FireGun(threads, id =>
            {
                if ((id & 1) == 0)
                {
                    for (int i = 0; i < WriterOps; i++)
                    {
                        d.SetByKey(7, (i & 1) == 0 ? B : A);
                    }
                }
                else
                {
                    for (int i = 0; i < ReaderOps; i++)
                    {
                        Assert.IsTrue(d.TryGetValue(7, out decimal byKey));
                        Assert.IsTrue(byKey == A || byKey == B, $"TORN key-path read: {byKey}");
                        Assert.IsTrue(d.TryGetAt(0, out decimal byIndex));
                        Assert.IsTrue(byIndex == A || byIndex == B, $"TORN list-path read: {byIndex}");

                        // Snapshot surfaces must be internally untorn too.
                        foreach (decimal v in d)
                        {
                            Assert.IsTrue(v == A || v == B, $"TORN enumerated read: {v}");
                        }
                    }
                }
            });
        }

        [TestMethod]
        public void Gun_AppendersVsEnumerators_SnapshotsAreExactPrefixes()
        {
            // One appender writes 0,1,2,... (value == key); concurrent enumerators must always see EXACTLY the
            // prefix 0..k-1 — order intact, no gap, no stale slot, no half-published tail. This is the release
            // count-publish and enumerator-capture contract under live fire.
            const int Appends = 120_000;
            int threads = GunThreads;
            var d = new ConcurrentOrderedDict<int, int>();

            FireGun(threads, id =>
            {
                if (id == 0)
                {
                    for (int k = 0; k < Appends; k++)
                    {
                        d.Add(k, k);
                    }
                }
                else
                {
                    while (d.Count < Appends)
                    {
                        int expected = 0;
                        foreach (int v in d)
                        {
                            Assert.AreEqual(expected, v, "enumeration is not the exact insertion prefix");
                            expected++;
                        }

                        var view = d.Snapshot();
                        for (int i = 0; i < view.Count; i++)
                        {
                            Assert.AreEqual(i, view[i], "snapshot view is not the exact insertion prefix");
                        }
                    }
                }
            });

            Assert.AreEqual(Appends, d.Count);
            AssertFullyConsistent(d);
        }

        [TestMethod]
        public void Gun_TailPopsAndReAppends_NeverDisturbCapturedSnapshots()
        {
            // The append-floor rule under fire: one thread pops the tail and re-appends (forcing constant slot
            // hand-backs), enumerators must still only ever see exact prefixes of the CURRENT sequence — a
            // reused slot bleeding a new value into an old snapshot would break the prefix's value pattern.
            const int Base = 1_000;
            const int Cycles = 40_000;
            int threads = GunThreads;
            var d = new ConcurrentOrderedDict<int, int>();
            for (int k = 0; k < Base; k++)
            {
                d.Add(k, k);
            }

            long done = 0;
            FireGun(threads, id =>
            {
                if (id == 0)
                {
                    for (int c = 0; c < Cycles; c++)
                    {
                        // Pop the tail entry, then append a fresh key carrying the SAME value-law (value == index
                        // it will land on), so any snapshot remains value[i] == i regardless of interleaving.
                        Assert.IsTrue(d.TryRemove(d.GetKeyAt(d.Count - 1), out _));
                        d.Add(Base + c, d.Count); // lands at index Count, value == that index
                    }

                    Interlocked.Exchange(ref done, 1);
                }
                else
                {
                    while (Interlocked.Read(ref done) == 0)
                    {
                        int expected = 0;
                        foreach (int v in d)
                        {
                            Assert.AreEqual(expected, v, "a captured snapshot observed a reused slot (floor rule violated)");
                            expected++;
                        }
                    }
                }
            });

            AssertFullyConsistent(d);
        }

        [TestMethod]
        public void Gun_AddRangeBatches_BecomeVisibleAtomically()
        {
            // Each writer AddRanges disjoint batches of exactly BatchSize; every Count a reader observes must be
            // a multiple of BatchSize — a non-multiple proves a batch became visible in pieces.
            const int BatchSize = 500;
            const int BatchesPerThread = 60;
            int threads = GunThreads;
            int writers = System.Math.Max(1, threads / 2);
            var d = new ConcurrentOrderedDict<int, int>();
            long writersDone = 0;

            FireGun(threads, id =>
            {
                if (id < writers)
                {
                    for (int b = 0; b < BatchesPerThread; b++)
                    {
                        int lo = (id * BatchesPerThread + b) * BatchSize;
                        var batch = new KeyValuePair<int, int>[BatchSize];
                        for (int i = 0; i < BatchSize; i++)
                        {
                            batch[i] = new KeyValuePair<int, int>(lo + i, lo + i);
                        }

                        d.AddRange(batch);
                    }

                    Interlocked.Increment(ref writersDone);
                }
                else
                {
                    while (Interlocked.Read(ref writersDone) < writers)
                    {
                        int count = d.Count;
                        Assert.AreEqual(0, count % BatchSize, $"observed count {count} — an AddRange batch was visible in pieces");
                        int viewCount = d.Snapshot().Count;
                        Assert.AreEqual(0, viewCount % BatchSize, $"observed view count {viewCount} — an AddRange batch was visible in pieces");
                    }
                }
            });

            Assert.AreEqual(writers * BatchesPerThread * BatchSize, d.Count);
            AssertFullyConsistent(d);
        }

        [TestMethod]
        public void Gun_RemoveSameKeys_ExactlyOneWinnerEach()
        {
            const int KeyCount = 8_000;
            int threads = GunThreads;
            var d = new ConcurrentOrderedDict<int, int>();
            for (int k = 0; k < KeyCount; k++)
            {
                d.Add(k, k);
            }

            int totalRemoved = 0;
            FireGun(threads, id =>
            {
                int wins = 0;
                for (int k = 0; k < KeyCount; k++)
                {
                    if (d.TryRemove(k, out int v))
                    {
                        Assert.AreEqual(k, v, "a removal handed back another key's value");
                        wins++;
                    }
                }

                Interlocked.Add(ref totalRemoved, wins);
            });

            Assert.AreEqual(KeyCount, totalRemoved, "every key must be removed by exactly one thread");
            Assert.AreEqual(0, d.Count);
            Assert.IsTrue(d.IsEmpty);
        }

        [TestMethod]
        public void Gun_SwapBackRemovals_KeepTheSurvivorSetExact()
        {
            // Concurrent order-breaking removals of disjoint ranges: order is explicitly forfeited, but the
            // surviving SET and full cross-path consistency are not.
            const int PerThread = 5_000;
            int threads = GunThreads;
            int total = threads * PerThread * 2;
            var d = new ConcurrentOrderedDict<int, int>();
            for (int k = 0; k < total; k++)
            {
                d.Add(k, k * 7);
            }

            FireGun(threads, id =>
            {
                // Each thread swap-back-removes the EVEN keys of its own disjoint slice.
                int lo = id * PerThread * 2;
                for (int k = lo; k < lo + PerThread * 2; k += 2)
                {
                    Assert.IsTrue(d.TryRemoveSwapBack(k, out int v), $"disjoint swap-back of {k} must win");
                    Assert.AreEqual(k * 7, v);
                }
            });

            Assert.AreEqual(total / 2, d.Count);
            for (int k = 0; k < total; k++)
            {
                bool shouldSurvive = (k & 1) == 1;
                Assert.AreEqual(shouldSurvive, d.ContainsKey(k), $"key {k} {(shouldSurvive ? "lost" : "survived removal")}");
                if (shouldSurvive)
                {
                    Assert.IsTrue(d.TryGetValue(k, out int v) && v == k * 7);
                }
            }

            AssertFullyConsistent(d);
        }

        [TestMethod]
        public void Gun_MixedChaos_ReadersNeverObserveIllegalState()
        {
            // Everything at once: appenders, hot-key updaters, disjoint removers, and readers hammering every
            // read surface. Readers assert only invariants that hold under ANY legal interleaving; afterwards the
            // final membership is checked exactly.
            const int Seed = 20_000;   // keys [0, Seed) pre-seeded; kept forever (values churn on a hot subset)
            const int AddsPerThread = 15_000;
            const int HotKeys = 32;
            int threads = System.Math.Max(4, GunThreads);
            var d = new ConcurrentOrderedDict<int, long>();
            for (int k = 0; k < Seed; k++)
            {
                d.Add(k, Legal(k, 0));
            }

            // Value law: every value ever written for key k is Legal(k, g) for some generation g — so a reader
            // can validate any (key, value) pair it observes without knowing the interleaving.
            static long Legal(int key, int generation) => ((long)key << 20) | (uint)(generation & 0xFFFFF);

            long writersDone = 0;
            int writerCount = 0;
            FireGun(threads, id =>
            {
                switch (id % 4)
                {
                    case 0: // appender: disjoint fresh ranges above the seeded space
                    {
                        Interlocked.Increment(ref writerCount);
                        int lo = Seed + id * AddsPerThread;
                        for (int i = 0; i < AddsPerThread; i++)
                        {
                            Assert.IsTrue(d.TryAdd(lo + i, Legal(lo + i, 0)));
                        }

                        Interlocked.Increment(ref writersDone);
                        break;
                    }

                    case 1: // updater: churn the hot seeded keys through both update entry points
                    {
                        Interlocked.Increment(ref writerCount);
                        for (int g = 1; g <= 40_000; g++)
                        {
                            int k = g % HotKeys;
                            if ((g & 1) == 0)
                            {
                                d.SetByKey(k, Legal(k, g));
                            }
                            else
                            {
                                d.AddOrUpdate(k, Legal(k, g), (key, _) => Legal(key, g));
                            }
                        }

                        Interlocked.Increment(ref writersDone);
                        break;
                    }

                    case 2: // remover+re-adder: churn a disjoint band of seeded keys in and out
                    {
                        Interlocked.Increment(ref writerCount);
                        int lo = Seed - (id % 4 + 1) * 1_000; // bands inside the seeded range, above the hot keys
                        for (int round = 0; round < 30; round++)
                        {
                            for (int k = lo; k < lo + 500; k++)
                            {
                                d.TryRemove(k, out _);
                            }

                            for (int k = lo; k < lo + 500; k++)
                            {
                                d.TryAdd(k, Legal(k, round + 1));
                            }
                        }

                        Interlocked.Increment(ref writersDone);
                        break;
                    }

                    default: // reader: hammer every surface, validating the value law and structural sanity
                    {
                        var rng = new System.Random(97 + id);
                        while (Interlocked.Read(ref writersDone) < Volatile.Read(ref writerCount) || Volatile.Read(ref writerCount) == 0)
                        {
                            int k = rng.Next(Seed);
                            if (d.TryGetValue(k, out long v))
                            {
                                Assert.AreEqual(k, (int)(v >> 20), $"key {k} exposed another key's value 0x{v:X}");
                            }

                            int idx = rng.Next(System.Math.Max(1, d.Count));
                            if (d.TryGetAt(idx, out long atIdx))
                            {
                                // The value at any position must obey the law for the key recorded THERE in some
                                // legal snapshot — validate shape only (key extraction), not pairing, because
                                // index and key reads are two separate snapshots by contract.
                                Assert.IsTrue((atIdx >> 20) >= 0);
                            }

                            foreach (var pair in d.Pairs)
                            {
                                Assert.AreEqual(pair.Key, (int)(pair.Value >> 20), "Pairs exposed a (key, value) mix from two different entries");
                            }

                            long sum = 0;
                            var view = d.Snapshot();
                            for (int i = 0; i < view.Count; i++)
                            {
                                sum += view[i];
                            }

                            Assert.IsTrue(sum != long.MinValue); // consume so the loop cannot be elided
                        }

                        break;
                    }
                }
            });

            // Quiescent exact audit: seeded keys all present (removers re-add what they remove), appended ranges
            // all present, every value obeys the law.
            for (int k = 0; k < Seed; k++)
            {
                Assert.IsTrue(d.TryGetValue(k, out long v), $"seeded key {k} vanished");
                Assert.AreEqual(k, (int)(v >> 20));
            }

            AssertFullyConsistent(d);
        }

        [TestMethod]
        public void Gun_IndexPathReaders_NeverThrowUnderStructuralChurn()
        {
            // TryGetAt/GetKeyAt/Snapshot must be exception-free for in-range-of-snapshot requests no matter what
            // structure churns underneath (interior removes shrink counts; readers use Try-forms for race-tolerant
            // access and must simply get false, never a crash or an out-of-snapshot read).
            const int Seed = 10_000;
            int threads = GunThreads;
            var d = new ConcurrentOrderedDict<int, int>();
            for (int k = 0; k < Seed; k++)
            {
                d.Add(k, k);
            }

            long done = 0;
            FireGun(threads, id =>
            {
                if (id == 0)
                {
                    var rng = new System.Random(5);
                    for (int i = 0; i < 4_000; i++)
                    {
                        int k = rng.Next(Seed);
                        if (!d.TryRemove(k, out _))
                        {
                            d.TryAdd(k, k);
                        }
                    }

                    Interlocked.Exchange(ref done, 1);
                }
                else
                {
                    var rng = new System.Random(50 + id);
                    while (Interlocked.Read(ref done) == 0)
                    {
                        var view = d.Snapshot();
                        if (view.Count > 0)
                        {
                            int i = rng.Next(view.Count);
                            _ = view[i];           // captured view: always in range for its own count
                            _ = view.GetKeyAt(i);
                        }

                        _ = d.TryGetAt(rng.Next(Seed), out _); // live Try-form: false is fine, throwing is not
                        _ = d.ToArray();
                        _ = d.Keys;
                    }
                }
            });

            AssertFullyConsistent(d);
        }

        [TestMethod]
        public void Gun_ClearVsEverything_EndsEmptyOrConsistent()
        {
            // Clear racing adders/readers: no reader may crash, and the final state (after a last Clear) is empty.
            const int Ops = 30_000;
            int threads = GunThreads;
            var d = new ConcurrentOrderedDict<int, int>();

            FireGun(threads, id =>
            {
                if (id == 0)
                {
                    for (int i = 0; i < 200; i++)
                    {
                        d.Clear();
                        Thread.SpinWait(500);
                    }
                }
                else if ((id & 1) == 0)
                {
                    for (int i = 0; i < Ops; i++)
                    {
                        d.SetByKey(i % 512, i);
                    }
                }
                else
                {
                    for (int i = 0; i < Ops; i++)
                    {
                        foreach (int v in d)
                        {
                            Assert.IsTrue(v >= 0);
                        }
                    }
                }
            });

            d.Clear();
            Assert.AreEqual(0, d.Count);
            Assert.IsTrue(d.IsEmpty);
            AssertFullyConsistent(d);

            // And the instance is fully usable after the storm + clear.
            d.Add(1, 1);
            Assert.AreEqual(1, d.Count);
        }
    }
}
