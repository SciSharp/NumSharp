using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Collections
{
    // Inside the namespace declaration ON PURPOSE. From .NET 9 the BCL ships
    // System.Collections.Generic.OrderedDictionary<TKey,TValue>, so a file-level `using NumSharp.Collections;` beside
    // `using System.Collections.Generic;` makes every `OrderedDictionary<,>` reference ambiguous (CS0104) on net10.0.
    // A using directive attached to the namespace declaration is consulted before the file-level ones, so the simple
    // name resolves to NumSharp's type here without qualifying every use.
    using NumSharp.Collections;

    /// <summary>
    ///     Known-failing reproductions for <see cref="OrderedDictionary{TKey,TValue}" />, the lean lock-free ordered
    ///     dictionary proposed as <c>System.Collections.Concurrent.ConcurrentOrderedDictionary</c>
    ///     (docs/proposals/ConcurrentOrderedDictionary.md). Every test asserts the <b>correct</b> outcome the type's own
    ///     documentation and the proposal's concurrency specification promise, and fails today.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Each test names the violated clause and the root cause, and is marked <see cref="OpenBugsAttribute" /> so CI
    ///         excludes it until the bug is fixed; remove the attribute together with the fix. The races are made as
    ///         deterministic as the type allows: where the type calls user code (the key comparer) between two steps of a
    ///         racy sequence, a comparer hook injects the competing writer at exactly that point, so the interleaving is
    ///         forced rather than hoped for. Where no user code runs inside the window (the enumerator construction, the
    ///         store/flag handshake of the lock-free replace), a bounded storm is used instead; on the current code those
    ///         fail within milliseconds on any multi-core machine, and after the fix they cannot fail at all.
    ///     </para>
    ///     <para>
    ///         Verified against the shipped code on x64 (i9-13900K, Release): the enumerator storm reproduces ~160K
    ///         IndexOutOfRangeException per second, the lock-free replace loses ~3% of its writes under copy churn, and
    ///         both drop to zero with the fixes named on each test.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class OrderedDictionaryOpenBugsTests
    {
        /// <summary>Reader threads per storm: enough to contend on small CI boxes, capped so the host is not drowned (the storms stop at the first anomaly anyway).</summary>
        private static int StormReaders => System.Math.Max(2, System.Math.Min(Environment.ProcessorCount - 1, 4));

        /// <summary>
        ///     Runs every body on its own thread behind one start gate (the house "concurrency gun" shape: all threads are
        ///     parked, then released by a single signal) until <paramref name="budget" /> elapses or
        ///     <paramref name="anomalyFound" /> reports a violation, then joins them and rethrows whatever they threw.
        /// </summary>
        /// <param name="budget">The longest the storm may run. A fixed implementation runs the full budget; a broken one stops early at the first anomaly.</param>
        /// <param name="anomalyFound">Polled by the loop condition each body receives, so the first recorded violation stops every thread promptly.</param>
        /// <param name="bodies">The per-thread workloads; each loops while the <see cref="Func{Boolean}" /> it is handed returns <see langword="true" />.</param>
        /// <exception cref="AssertFailedException">A thread did not finish within the hang ceiling (a deadlock/livelock guard, not a performance bound).</exception>
        /// <exception cref="AggregateException">One or more bodies threw an exception they did not handle; carries every captured exception.</exception>
        private static void RunStorm(TimeSpan budget, Func<bool> anomalyFound, params Action<Func<bool>>[] bodies)
        {
            using var go = new ManualResetEventSlim(false);
            using var ready = new CountdownEvent(bodies.Length);
            var errors = new ConcurrentQueue<Exception>();
            var clock = new Stopwatch();

            // The loop condition every body polls. The clock starts only when the gate opens, so thread start-up cost
            // never eats the budget; the anomaly check makes a broken implementation stop at its first violation.
            bool KeepGoing() => clock.Elapsed < budget && !anomalyFound();

            var threads = new Thread[bodies.Length];
            for (int t = 0; t < bodies.Length; t++)
            {
                Action<Func<bool>> body = bodies[t]; // capture per-thread, not the loop variable
                threads[t] = new Thread(() =>
                {
                    ready.Signal();
                    go.Wait();
                    try
                    {
                        body(KeepGoing);
                    }
                    catch (Exception ex)
                    {
                        errors.Enqueue(ex);
                    }
                })
                {
                    IsBackground = true,
                    Name = $"odict-storm-{t}",
                };
                threads[t].Start();
            }

            ready.Wait();
            clock.Start();
            go.Set();

            foreach (Thread thread in threads)
            {
                Assert.IsTrue(thread.Join(TimeSpan.FromMinutes(2)), $"{thread.Name} did not finish — deadlock or livelock");
            }

            if (!errors.IsEmpty)
            {
                throw new AggregateException(errors);
            }
        }

        // ------------------------------------------------------------------------------------------------------------
        // 1. Snapshot enumeration tears across generations.
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     <b>Contract:</b> "Enumeration captures the value array and count at the start" (type remarks) and "snapshot
        ///     enumeration never throws ... and reflects a consistent prefix" (proposal §7.1). <b>Bug:</b>
        ///     <c>GetEnumerator()</c> is <c>new Enumerator(_t._values, Volatile.Read(ref _t._count))</c> — it reads the
        ///     volatile generation field TWICE, so a growth published between the reads pairs the OLD (shorter) value
        ///     array with the NEW (larger) count: the enumerator then runs past the array (IndexOutOfRangeException) or
        ///     yields slots the captured generation never wrote (phantom <c>default</c> values). <b>Fix:</b> read the
        ///     generation once into a local and take both fields from it, as <c>ToArray</c>/<c>AsValuesSpan</c> already do.
        /// </summary>
        /// <remarks>Every value ever stored is <c>i + 1 ≥ 1</c>, so a yielded <c>0</c> can only be an unwritten slot.</remarks>
        [TestMethod]
        [OpenBugs]
        public void Enumerate_WhileAWriterRegrowsTheTable_NeverThrowsAndNeverYieldsAnUnwrittenSlot()
        {
            var d = new OrderedDictionary<int, int>();
            long outOfRange = 0, phantoms = 0, enumerations = 0, writerRounds = 0;

            var bodies = new List<Action<Func<bool>>>
            {
                // Writer: every round clears and refills 64 keys, publishing a new generation at each growth (4→8→16→...).
                go =>
                {
                    while (go())
                    {
                        d.Clear();
                        for (int i = 0; i < 64; i++)
                        {
                            d.TryAdd(i, i + 1);
                        }

                        Interlocked.Increment(ref writerRounds);
                    }
                },
            };

            for (int r = 0; r < StormReaders; r++)
            {
                bodies.Add(go =>
                {
                    while (go())
                    {
                        try
                        {
                            foreach (int v in d)
                            {
                                if (v == 0)
                                {
                                    Interlocked.Increment(ref phantoms);
                                }
                            }

                            Interlocked.Increment(ref enumerations);
                        }
                        catch (IndexOutOfRangeException)
                        {
                            Interlocked.Increment(ref outOfRange);
                        }
                    }
                });
            }

            RunStorm(TimeSpan.FromSeconds(2), () => Interlocked.Read(ref outOfRange) + Interlocked.Read(ref phantoms) > 0, bodies.ToArray());

            Assert.AreEqual(0L, outOfRange + phantoms,
                $"a foreach over OrderedDictionary must enumerate ONE generation's (values, count) snapshot; observed " +
                $"{outOfRange} IndexOutOfRangeException and {phantoms} phantom default value(s) over {enumerations} completed " +
                $"enumeration(s) / {writerRounds} writer round(s) — GetEnumerator() reads the volatile generation field twice");
        }

        // ------------------------------------------------------------------------------------------------------------
        // 2. A wide-value replace aliases the index across generations.
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>The value stored for <paramref name="key" /> in the wide-value tests: never zero, so <c>default(decimal)</c> is distinguishable from every legal value.</summary>
        /// <param name="key">The key.</param>
        /// <returns>The key's value under the test's value law.</returns>
        private static decimal WideValue(int key) => key * 1.5m + 0.25m;

        /// <summary>
        ///     <b>Contract:</b> "A lock-free reader always sees valid, untorn keys and values" (type remarks) / "no torn or
        ///     garbage reads" (proposal §7.1). <b>Bug:</b> replacing the value of an existing key when
        ///     <typeparamref name="TValue" /> is NOT atomically writable (<c>decimal</c>, <c>Guid</c>, any struct wider
        ///     than a machine word, <c>long</c> on 32-bit) publishes a new generation that clones ONLY the value array and
        ///     SHARES the old generation's index and key arrays. The next in-place append then writes a new index word
        ///     and key slot into those shared arrays, so a reader still holding the OLD generation resolves the new key
        ///     through the shared index and returns its own, never-written <c>values[slot]</c> — <c>default</c> for a
        ///     present key. This is exactly the rule ConcurrentOrderedDictionary.COMPACT.md §4.7 forbids ("a non-atomic
        ///     replace copies the whole generation"), which the compact sibling follows. <b>Fix:</b> copy the whole
        ///     generation (index + keys + values) on a wide-value replace.
        /// </summary>
        /// <remarks>
        ///     Deterministic: <c>TryGetValue</c> captures its generation and only THEN calls the comparer's
        ///     <c>GetHashCode</c>, so a comparer hook runs the competing writer (wide replace + append, on another thread)
        ///     inside that window. The correct outcome is either "not found" (the reader's snapshot predates the append) or
        ///     the real value; the shipped code returns <c>(true, default)</c>. The window is not an artefact of the hook:
        ///     with a trivial comparer the race reproduced ~47K times per second on the same machine.
        /// </remarks>
        [TestMethod]
        [OpenBugs]
        public void WideValueReplaceThenAppend_ReaderHoldingTheOlderGeneration_NeverReadsAnUnwrittenValue()
        {
            const int NewKey = 100;
            var comparer = new ReaderHookComparer();
            var d = new OrderedDictionary<int, decimal>(64, comparer);
            for (int k = 0; k < 8; k++)
            {
                d.TryAdd(k, WideValue(k));
            }

            comparer.ArmOnce(NewKey, Environment.CurrentManagedThreadId, () =>
            {
                // Runs on the reader thread INSIDE TryGetValue(NewKey), after it captured its generation. A different
                // thread performs the two writes, so they are ordinary (non-reentrant) mutations.
                var writer = new Thread(() =>
                {
                    d.SetByKey(0, 999.5m);            // wide value → new generation sharing the old index/keys arrays
                    d.TryAdd(NewKey, WideValue(NewKey)); // in-place append → new word + key in those SHARED arrays
                })
                {
                    IsBackground = true,
                    Name = "odict-wide-writer",
                };
                writer.Start();
                Assert.IsTrue(writer.Join(TimeSpan.FromSeconds(30)), "the writer thread must finish while the reader waits");
            });

            bool found = d.TryGetValue(NewKey, out decimal value);

            Assert.IsTrue(comparer.Fired, "harness: the hook must have run inside TryGetValue");
            Assert.IsTrue(!found || value == WideValue(NewKey),
                $"TryGetValue({NewKey}) returned found={found}, value={value}; the correct outcome is not-found (the reader's " +
                $"generation predates the append) or {WideValue(NewKey)}. A present key read as default(decimal) means the " +
                "reader resolved the key through the index its generation SHARES with the newer one, then read its own stale " +
                "value array (values-only copy-on-write on a wide-value replace)");
        }

        // ------------------------------------------------------------------------------------------------------------
        // 3. The lock-free replace loses updates to a copying resize (store-buffering race).
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     <b>Contract:</b> "no update is ever lost" (type remarks) / "No lost updates ... including the hard race of a
        ///     lock-free value update running concurrently with a resize that copies the backing arrays" and "per key,
        ///     operations are linearizable (a value update on key k is immediately visible to a later read of k)"
        ///     (proposal §7.1/§7.2). <b>Bug:</b> the lock-free <c>SetByKey</c> does a plain value store and then
        ///     volatile-reads <c>_resizing</c>/<c>_t</c>, while a resize does a volatile write of <c>_resizing = true</c> and
        ///     then copies the value array. That is the store-buffering (Dekker) litmus shape: release/acquire ordering
        ///     does not forbid "each side's load misses the other side's store" on x86-TSO or ARM64, so the replacer can
        ///     see <c>_resizing == false</c> while the resize's copy reads the OLD value; the resize then publishes a
        ///     generation without the completed write. <b>Fix:</b> a full fence (<c>Interlocked.MemoryBarrier()</c>)
        ///     between the value store and the flag/generation re-check, AND after every <c>_resizing = true</c> before
        ///     the copy. Measured: fences on both sides → 0 lost writes, +3.3 ns per replace (4.9 → 8.2 ns), still ~5×
        ///     faster than a locked replace.
        /// </summary>
        /// <remarks>
        ///     Each replacer is the ONLY writer of its key, so after its own <c>SetByKey(k, i)</c> returns, every read of
        ///     <c>k</c> it makes must return <c>i</c> until its next write; an older value is a lost update. (A storm with
        ///     several writers per key cannot detect this — last-writer-wins hides a lost intermediate write.) The
        ///     churn thread forces a copying resize every iteration (an order-preserving removal republishes the whole
        ///     generation). No user code runs inside the window, so this stays a bounded storm; it reproduced at ~3% of
        ///     all replaces on the shipped code.
        /// </remarks>
        [TestMethod]
        [OpenBugs]
        public void LockFreeReplace_RacingCopyingResizes_NeverLosesACompletedWrite()
        {
            const int Replacers = 4, ChurnKey = 1000;
            var d = new OrderedDictionary<int, long>(64);
            for (int k = 0; k < Replacers; k++)
            {
                d.TryAdd(k, 0);
            }

            d.TryAdd(ChurnKey, 0);

            long lost = 0, sets = 0, resizes = 0;
            string firstLoss = null;
            var bodies = new List<Action<Func<bool>>>();
            for (int r = 0; r < Replacers; r++)
            {
                int key = r; // this thread is the sole writer of `key`
                bodies.Add(go =>
                {
                    long written = 0;
                    while (go())
                    {
                        written++;
                        d.SetByKey(key, written);
                        Interlocked.Increment(ref sets);

                        // Read-your-own-write, repeatedly: the resize that swallows the store may publish slightly after
                        // SetByKey returned, so a single read right away would usually still see the value.
                        for (int probe = 0; probe < 32; probe++)
                        {
                            d.TryGetValue(key, out long seen);
                            if (seen != written)
                            {
                                Interlocked.Increment(ref lost);
                                Interlocked.CompareExchange(ref firstLoss, $"key {key}: SetByKey({written}) returned, then a read saw {seen}", null);
                                break;
                            }
                        }
                    }
                });
            }

            bodies.Add(go =>
            {
                long c = 0;
                while (go())
                {
                    d.TryRemove(ChurnKey, out _); // copies the value array into a fresh generation (sets _resizing)
                    d.TryAdd(ChurnKey, c++);
                    Interlocked.Increment(ref resizes);
                }
            });

            RunStorm(TimeSpan.FromSeconds(3), () => Interlocked.Read(ref lost) > 0, bodies.ToArray());

            Assert.AreEqual(0L, lost,
                $"a completed lock-free SetByKey must never be lost; {lost} loss(es) over {sets} replace(s) racing {resizes} " +
                $"copying resize(s) — first: {firstLoss}. The store/flag handshake needs a full fence on BOTH sides " +
                "(release/acquire cannot forbid store-buffering)");
        }

        // ------------------------------------------------------------------------------------------------------------
        // 4. Null keys are accepted.
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     <b>Contract:</b> <c>where TKey : notnull</c>, the proposed BCL surface, the three sibling ordered maps
        ///     (<see cref="ConcurrentOrderedDictionary{TKey,TValue}" />, <see cref="ConcurrentOrderedCompactDictionary{TKey,TValue}" />)
        ///     and <c>Dictionary</c>/<c>ConcurrentDictionary</c> all reject a null key with
        ///     <see cref="ArgumentNullException" />. <b>Bug:</b> <see cref="OrderedDictionary{TKey,TValue}" /> never checks,
        ///     and <c>EqualityComparer&lt;string&gt;.Default.GetHashCode(null)</c> is 0 rather than a throw, so a null
        ///     reference key is silently stored, found and enumerated like any other key. <b>Fix:</b> the siblings'
        ///     <c>NullCheck</c> (with its <c>typeof(TKey).IsValueType</c> guard) on every key entry point.
        /// </summary>
        /// <remarks>Each entry point is checked on a fresh instance, so one accepted null cannot change what a later check observes; the failure lists every entry point that accepted null.</remarks>
        [TestMethod]
        [OpenBugs]
        public void NullKey_IsRejectedOnEveryKeyEntryPoint_LikeTheSiblingsAndTheBcl()
        {
            var entryPoints = new (string Name, Action<OrderedDictionary<string, int>> Call)[]
            {
                ("TryAdd", d => d.TryAdd(null!, 1)),
                ("SetByKey", d => d.SetByKey(null!, 1)),
                ("this[key] set", d => d[null!] = 1),
                ("GetOrAdd", d => d.GetOrAdd(null!, 1)),
                ("AddRange", d => d.AddRange(new[] { new KeyValuePair<string, int>(null!, 1) })),
                ("TryGetValue", d => d.TryGetValue(null!, out _)),
                ("GetByKey", d => d.GetByKey(null!)),
                ("this[key] get", d => _ = d[null!]),
                ("ContainsKey", d => d.ContainsKey(null!)),
                ("IndexOf", d => d.IndexOf(null!)),
                ("TryRemove", d => d.TryRemove(null!, out _)),
            };

            var accepted = new List<string>();
            foreach (var (name, call) in entryPoints)
            {
                var d = new OrderedDictionary<string, int>();
                d.TryAdd("present", 7);
                try
                {
                    call(d);
                    accepted.Add($"{name} (no exception)");
                }
                catch (ArgumentNullException)
                {
                    // the correct outcome
                }
                catch (Exception ex)
                {
                    accepted.Add($"{name} ({ex.GetType().Name} instead of ArgumentNullException)");
                }
            }

            Assert.AreEqual(0, accepted.Count, "every key entry point must throw ArgumentNullException for a null key; these did not: " + string.Join(", ", accepted));
        }

        // ------------------------------------------------------------------------------------------------------------
        // 5. A huge capacity hangs the constructor.
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     <b>Contract:</b> a capacity the index cannot address must be refused, as the compact sibling does
        ///     (<see cref="ArgumentOutOfRangeException" /> above ~715 million entries). <b>Bug:</b> the constructor sizes its
        ///     index with <c>while (len &lt; cap * 100 / 70 + 1) len &lt;&lt;= 1;</c> on an <c>int</c>; for a capacity of
        ///     ~752 million or more the target exceeds 2^30, the shift wraps to <c>int.MinValue</c> and then to 0, and
        ///     <c>0 &lt;&lt; 1 == 0</c> never terminates — the constructor spins forever instead of throwing (and
        ///     <c>AppendGrow</c>'s <c>newLen &lt;&lt;= 1</c> has the same overflow). <b>Fix:</b> size in <c>long</c> and
        ///     throw above the addressable maximum (the index needs a power-of-two <c>int[]</c> ≤ <c>Array.MaxLength</c>).
        /// </summary>
        /// <remarks>
        ///     The constructor runs on a background thread so a hang fails the test instead of hanging the run. There is no
        ///     way to stop a spinning managed thread, so on the buggy code it keeps one core busy until the test host exits;
        ///     it runs at the lowest priority to keep that from disturbing the rest of the run. No memory is allocated
        ///     before the spin (the index-length loop precedes every array allocation).
        /// </remarks>
        [TestMethod]
        [OpenBugs]
        public void HugeCapacity_IsRefusedWithArgumentOutOfRange_InsteadOfSpinningForever()
        {
            Exception outcome = null;
            var ctor = new Thread(() =>
            {
                try
                {
                    _ = new OrderedDictionary<byte, byte>(800_000_000);
                }
                catch (Exception ex)
                {
                    outcome = ex;
                }
            })
            {
                IsBackground = true,
                Priority = ThreadPriority.Lowest,
                Name = "odict-huge-capacity",
            };
            ctor.Start();

            bool finished = ctor.Join(TimeSpan.FromSeconds(10));

            Assert.IsTrue(finished, "new OrderedDictionary<byte,byte>(800_000_000) did not return within 10 s: the power-of-two index-length loop overflowed int (len → int.MinValue → 0) and spins forever");
            Assert.IsInstanceOfType(outcome, typeof(ArgumentOutOfRangeException), $"an unaddressable capacity must throw ArgumentOutOfRangeException; got {outcome?.GetType().Name ?? "no exception"}");
        }

        // ------------------------------------------------------------------------------------------------------------
        // 6. A throw during a resize leaves the lock-free replace permanently locked.
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     Reports whether <c>SetByKey</c> on an existing key completes while ANOTHER thread holds the write lock — the
        ///     observable definition of "the atomically-writable value replace is lock-free".
        /// </summary>
        /// <param name="d">The instance under test (<c>int</c> values are atomically writable, so the lock-free path applies).</param>
        /// <param name="key">An existing key to replace.</param>
        /// <returns><see langword="true" /> if the replace finished while the lock was held; <see langword="false" /> if it waited for the lock.</returns>
        /// <exception cref="AssertFailedException">The lock holder never reached its blocking point, or a helper thread failed to finish.</exception>
        /// <remarks>
        ///     The lock is held by an <c>AddRange</c> whose source enumerator blocks inside its first <c>MoveNext</c> — the
        ///     type runs that enumerator under its write lock by design. The holder is always released before returning, so
        ///     a non-lock-free replace is observed as "did not finish in time", never as a hang.
        /// </remarks>
        private static bool ReplaceCompletesWhileAnotherThreadHoldsTheWriteLock(OrderedDictionary<int, int> d, int key)
        {
            using var inside = new ManualResetEventSlim(false);
            using var release = new ManualResetEventSlim(false);
            using var replaced = new ManualResetEventSlim(false);

            // Runs inside AddRange's locked foreach: announce we hold the lock, then keep holding it until released.
            IEnumerable<KeyValuePair<int, int>> BlockingSource()
            {
                inside.Set();
                release.Wait();
                yield break;
            }

            var holder = new Thread(() => d.AddRange(BlockingSource())) { IsBackground = true, Name = "odict-lock-holder" };
            holder.Start();
            Assert.IsTrue(inside.Wait(TimeSpan.FromSeconds(30)), "harness: the lock holder never entered AddRange's source");

            var replacer = new Thread(() =>
            {
                d.SetByKey(key, 42);
                replaced.Set();
            })
            {
                IsBackground = true,
                Name = "odict-replacer",
            };
            replacer.Start();

            // Lock-free: done in microseconds. Locked: parked on Monitor.Enter until the holder is released below.
            bool completedWhileLocked = replaced.Wait(TimeSpan.FromSeconds(2));

            release.Set();
            Assert.IsTrue(holder.Join(TimeSpan.FromSeconds(30)), "harness: the lock holder did not finish");
            Assert.IsTrue(replacer.Join(TimeSpan.FromSeconds(30)), "harness: the replacer did not finish after the lock was released");
            return completedWhileLocked;
        }

        /// <summary>
        ///     <b>Contract:</b> "An atomically-writable value replace is lock-free" (type remarks, proposal §7.2).
        ///     <b>Bug:</b> every generation-replacing resize sets <c>_resizing = true</c>, then allocates and rebuilds the
        ///     index — which calls the user's comparer (<c>GetHashCode</c> for every surviving key) — and only then clears
        ///     the flag, with no <c>try/finally</c>. A comparer (or allocation) exception inside that window leaves the
        ///     collection itself consistent (the new generation is never published) but <c>_resizing</c> stuck at
        ///     <see langword="true" /> forever, so every later lock-free replace fails its re-check and silently takes the
        ///     write lock. <b>Fix:</b> clear the flag in a <c>finally</c> (or set it only after the fallible work, right
        ///     before the copy that must be guarded).
        /// </summary>
        [TestMethod]
        [OpenBugs]
        public void ComparerThrowDuringAResize_DoesNotLeaveTheReplacePermanentlyLocked()
        {
            var comparer = new PoisonComparer();
            var d = new OrderedDictionary<int, int>(comparer);
            for (int k = 0; k < 10; k++)
            {
                d.TryAdd(k, k * 10);
            }

            Assert.IsTrue(ReplaceCompletesWhileAnotherThreadHoldsTheWriteLock(d, 5), "harness: a healthy instance must replace lock-free");

            // Key 7 is hashed only by the index rebuild of TryRemove(3) (TryRemove hashes key 3 itself for its probe).
            comparer.Arm(7);
            var thrown = Assert.ThrowsExactly<InvalidOperationException>(() => d.TryRemove(3, out _));
            comparer.Disarm();
            StringAssert.Contains(thrown.Message, "poisoned");

            // The failed removal never published, so the collection is intact...
            Assert.AreEqual(10, d.Count);
            Assert.IsTrue(d.ContainsKey(3));

            // ...and the value replace must still be lock-free.
            Assert.IsTrue(ReplaceCompletesWhileAnotherThreadHoldsTheWriteLock(d, 5),
                "after a comparer threw mid-resize, SetByKey on an existing key waited for the write lock: _resizing was left " +
                "stuck at true, so the lock-free replace silently degraded to a locked one for the life of the instance");
        }

        // ------------------------------------------------------------------------------------------------------------
        // Comparers used to force interleavings.
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     A key comparer with a one-shot hook: when the armed thread hashes the armed key, the hook runs synchronously
        ///     inside <c>GetHashCode</c>. <c>OrderedDictionary.TryGetValue</c> captures its generation BEFORE hashing, so
        ///     the hook executes exactly in the window between "reader chose its snapshot" and "reader probes it".
        /// </summary>
        private sealed class ReaderHookComparer : IEqualityComparer<int>
        {
            /// <summary>The key whose hashing fires the hook (<see cref="int.MinValue" /> = disarmed).</summary>
            private int _key = int.MinValue;

            /// <summary>The managed thread id allowed to fire the hook, so the writer's own hashing of the same key never re-enters it.</summary>
            private int _threadId;

            /// <summary>The pending hook; cleared before it runs so it fires exactly once.</summary>
            private Action _hook;

            /// <summary>Gets whether the hook has run.</summary>
            public bool Fired { get; private set; }

            /// <summary>Arms a one-shot hook for <paramref name="key" /> hashed on thread <paramref name="threadId" />.</summary>
            /// <param name="key">The key whose hashing fires the hook.</param>
            /// <param name="threadId">The only thread on which it fires.</param>
            /// <param name="hook">The action to run inside <c>GetHashCode</c>; it may block (the caller holds no lock of the dictionary).</param>
            public void ArmOnce(int key, int threadId, Action hook)
            {
                _key = key;
                _threadId = threadId;
                _hook = hook;
            }

            /// <summary>Plain integer equality.</summary>
            /// <param name="x">The first key.</param>
            /// <param name="y">The second key.</param>
            /// <returns>Whether the keys are equal.</returns>
            public bool Equals(int x, int y) => x == y;

            /// <summary>A distinct hash per key; fires the armed hook first when the armed thread hashes the armed key.</summary>
            /// <param name="key">The key to hash.</param>
            /// <returns>The key's hash.</returns>
            public int GetHashCode(int key)
            {
                if (key == _key && Environment.CurrentManagedThreadId == _threadId && _hook is { } hook)
                {
                    _hook = null;
                    Fired = true;
                    hook();
                }

                return key * 31;
            }
        }

        /// <summary>A key comparer that throws <see cref="InvalidOperationException" /> when hashing one armed "poison" key — a comparer failure injected at a chosen point of a mutation.</summary>
        private sealed class PoisonComparer : IEqualityComparer<int>
        {
            /// <summary>The key whose hashing throws (<see cref="int.MinValue" /> = disarmed).</summary>
            private int _poison = int.MinValue;

            /// <summary>Makes hashing <paramref name="key" /> throw until <see cref="Disarm" />.</summary>
            /// <param name="key">The poison key.</param>
            public void Arm(int key) => Volatile.Write(ref _poison, key);

            /// <summary>Restores normal hashing for every key.</summary>
            public void Disarm() => Volatile.Write(ref _poison, int.MinValue);

            /// <summary>Plain integer equality.</summary>
            /// <param name="x">The first key.</param>
            /// <param name="y">The second key.</param>
            /// <returns>Whether the keys are equal.</returns>
            public bool Equals(int x, int y) => x == y;

            /// <summary>The key itself as its hash; throws for the armed poison key.</summary>
            /// <param name="key">The key to hash.</param>
            /// <returns>The key's hash.</returns>
            /// <exception cref="InvalidOperationException"><paramref name="key" /> is the armed poison key.</exception>
            public int GetHashCode(int key)
            {
                if (key == Volatile.Read(ref _poison))
                {
                    throw new InvalidOperationException($"poisoned comparer: hashing key {key}");
                }

                return key;
            }
        }
    }
}
