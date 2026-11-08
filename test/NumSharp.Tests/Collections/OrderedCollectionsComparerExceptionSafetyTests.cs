using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Collections
{
    // Inside the namespace declaration so `OrderedDictionary<,>` binds to NumSharp's type rather than being ambiguous
    // with System.Collections.Generic.OrderedDictionary<,> (.NET 9+) — see OrderedDictionaryContractTests.
    using NumSharp.Collections;

    /// <summary>
    ///     Exception safety of the ordered concurrent maps when the <b>key comparer</b> throws in the middle of a
    ///     structural mutation — including the collections' own <see cref="LockRecursionException" />, which their
    ///     documented "evil comparer" defence raises from inside the comparer when it tries to write back.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         The contract under test is the one both shipped types document: user code the collection runs under its
    ///         write lock (an AddRange source, a RemoveWhere predicate, <b>a key comparer</b>) that writes back is
    ///         "refused with LockRecursionException instead of being allowed to corrupt", and "a throw ... always lands the
    ///         collection in a consistent 'applied everything up to the failure' state — never with a key that resolves but
    ///         does not enumerate" (ConcurrentOrderedDictionary remarks, ConcurrentOrderedDictionary.TODO.md "Thread-safety
    ///         contract").
    ///     </para>
    ///     <para>
    ///         <b>ConcurrentOrderedDictionary broke it until these tests pinned it</b> (committed known-failing as
    ///         <c>[OpenBugs]</c> in fe756f9e, then fixed). Its order-preserving interior removal, both branches of its
    ///         swap-back removal (in place for word-sized keys/values, copying for wide ones) and its RemoveWhere all
    ///         mutated the key map FIRST and then re-indexed the shifted/moved keys through the map — a comparer call per
    ///         key — BEFORE publishing the new list-path store. A comparer that threw during that re-index (here: the
    ///         refused write-back) aborted the operation half-applied: a removed key no longer resolved yet still
    ///         enumerated (or a moved key enumerated twice), and already-re-indexed keys reported positions the published
    ///         store did not have — the refusal meant to prevent corruption caused it. The fix is the discipline the type
    ///         already applied to allocations: every comparer-calling lookup (the key-map node of every entry the operation
    ///         will unlink or re-index, resolved through the vendored map's node handles) happens BEFORE the first
    ///         mutation, and the mutation then goes through those handles without the comparer — so a refused write-back
    ///         now aborts the operation with nothing changed.
    ///     </para>
    ///     <para>
    ///         <b>ConcurrentOrderedCompactDictionary and OrderedDictionary always kept the contract</b> and are pinned here
    ///         as controls: the compact type re-indexes from the tags stored in its index words (no comparer after the
    ///         probe) and probes every key it will touch before mutating; OrderedDictionary rebuilds a private generation and
    ///         publishes it only at the end, so a throw leaves the published generation untouched (its separate stuck-flag
    ///         defect is guarded in OrderedDictionaryContractTests).
    ///     </para>
    ///     <para>
    ///         Every scenario is deterministic and single-threaded: keys 0..9 with a value law (<c>10 × key</c>), and a
    ///         comparer whose <c>GetHashCode</c> for one "poison" key attempts a write-back into the same collection. The
    ///         poison key is chosen so that it is hashed ONLY during the phase under test (never by the operation's own
    ///         probe of the key it removes).
    ///     </para>
    /// </remarks>
    [TestClass]
    public class OrderedCollectionsComparerExceptionSafetyTests
    {
        /// <summary>The number of seeded keys (0..9); also the key universe the audit checks for keys that resolve without enumerating.</summary>
        private const int Keys = 10;

        /// <summary>The value law for word-sized values: every key maps to <c>10 × key</c>, so a key paired with another key's value is detectable.</summary>
        private static readonly Func<int, int> IntLaw = k => k * 10;

        /// <summary>The value law for wide values (<see cref="decimal" /> is not atomically writable, which routes the swap-back to its copying branch).</summary>
        private static readonly Func<int, decimal> WideLaw = k => k * 10m + 0.5m;

        // ------------------------------------------------------------------------------------------------------------
        // ConcurrentOrderedDictionary — regression guards (these four failed before the node-handle fix).
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     <c>TryRemove(3)</c> of an interior key: key 7 is poisoned, and it is hashed only while resolving the entries
        ///     the shift moves. Correct: consistent paths (either the removal fully applied or not at all). Before the fix
        ///     the map dropped key 3 first and re-indexed keys 4..9 one by one through the comparer, so the refused
        ///     write-back at key 7 left key 3 enumerating at position 3 without resolving, and keys 4..6 reporting
        ///     positions 3..5 the published store did not have; now every shifted entry's node is resolved before the
        ///     first mutation, so the refusal aborts the removal with nothing changed.
        /// </summary>
        [TestMethod]
        public void ConcurrentOrderedDictionary_InteriorRemoval_RefusedWriteBackMidReindex_LeavesBothPathsConsistent()
        {
            var (d, comparer) = SeedConcurrentOrderedDictionary(IntLaw);

            Exception thrown = RunPoisoned(comparer, poisonKey: 7, () => d.TryRemove(3, out _));

            Assert.IsInstanceOfType(thrown, typeof(LockRecursionException), "harness: the write-back must be refused mid-reindex");
            AssertConsistent(Of(d), IntLaw, "after TryRemove(3) threw while re-indexing key 7");
        }

        /// <summary>
        ///     <c>TryRemoveSwapBack(3)</c>, in-place branch (<c>int</c> keys and values), with the moved key 9 poisoned.
        ///     Correct: consistent paths. Before the fix the map dropped key 3 and the last entry (key 9) was written into
        ///     slot 3 of the LIVE store before key 9's recorded position was rewritten through the comparer — where the
        ///     refused write-back fired — leaving the live store at count 10 with key 9 at positions 3 AND 9 (enumerated
        ///     twice) and key 3 gone; now the moved key's node is resolved before any mutation, so nothing changes.
        /// </summary>
        [TestMethod]
        public void ConcurrentOrderedDictionary_SwapBackRemoval_RefusedWriteBackMidReindex_LeavesBothPathsConsistent()
        {
            var (d, comparer) = SeedConcurrentOrderedDictionary(IntLaw);

            Exception thrown = RunPoisoned(comparer, poisonKey: 9, () => d.TryRemoveSwapBack(3, out _));

            Assert.IsInstanceOfType(thrown, typeof(LockRecursionException), "harness: the write-back must be refused mid-reindex");
            AssertConsistent(Of(d), IntLaw, "after TryRemoveSwapBack(3) threw while re-pointing moved key 9");
        }

        /// <summary>
        ///     <c>TryRemoveSwapBack(3)</c>, copying branch (<see cref="decimal" /> values are not atomically writable), with
        ///     the moved key 9 poisoned. Correct: consistent paths. Before the fix the moved entry was placed into fresh
        ///     arrays, but the map still dropped key 3 BEFORE key 9 was re-pointed through the comparer, and the fresh
        ///     store was published only after that — so the refused write-back left the old store published, key 3
        ///     enumerating at position 3 while no longer resolving; now the moved key's node is resolved first.
        /// </summary>
        [TestMethod]
        public void ConcurrentOrderedDictionary_WideSwapBackRemoval_RefusedWriteBackMidReindex_LeavesBothPathsConsistent()
        {
            var (d, comparer) = SeedConcurrentOrderedDictionary(WideLaw);

            Exception thrown = RunPoisoned(comparer, poisonKey: 9, () => d.TryRemoveSwapBack(3, out _));

            Assert.IsInstanceOfType(thrown, typeof(LockRecursionException), "harness: the write-back must be refused mid-reindex");
            AssertConsistent(Of(d), WideLaw, "after the copying TryRemoveSwapBack(3) threw while re-pointing moved key 9");
        }

        /// <summary>
        ///     <c>RemoveWhere(even)</c> with survivor 7 poisoned. Correct: consistent paths. Before the fix keys 0, 2, 4, 6
        ///     were dropped from the map one by one and survivors 1, 3, 5 re-indexed through the comparer; the refused
        ///     write-back fired while re-indexing survivor 7, and again inside the <c>finally</c> meant to publish the
        ///     consistent prefix — so the publish never happened, leaving 0, 2, 4, 6 enumerating without resolving and 1,
        ///     3, 5 reporting positions 0..2 the published store did not have. Now the node of every entry the pass may
        ///     touch is resolved before the first removal (the refusal aborts the pass with nothing removed), and the
        ///     <c>finally</c> no longer calls the comparer at all.
        /// </summary>
        [TestMethod]
        public void ConcurrentOrderedDictionary_RemoveWhere_RefusedWriteBackMidReindex_LeavesBothPathsConsistent()
        {
            var (d, comparer) = SeedConcurrentOrderedDictionary(IntLaw);

            Exception thrown = RunPoisoned(comparer, poisonKey: 7, () => d.RemoveWhere((k, _) => k % 2 == 0));

            Assert.IsInstanceOfType(thrown, typeof(LockRecursionException), "harness: the write-back must be refused mid-reindex");
            AssertConsistent(Of(d), IntLaw, "after RemoveWhere(even) threw while re-indexing survivor 7");
        }

        // ------------------------------------------------------------------------------------------------------------
        // ConcurrentOrderedCompactDictionary — the same attacks, contract kept (CI regression guards).
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>The compact type's interior removal renumbers index words by their stored tags — no comparer runs after its probe, so the poison is never hashed and the removal completes consistently.</summary>
        [TestMethod]
        public void ConcurrentOrderedCompactDictionary_InteriorRemoval_NeverConsultsTheComparerAfterItsProbe()
        {
            var (d, comparer) = SeedConcurrentOrderedCompactDictionary(IntLaw);

            Exception thrown = RunPoisoned(comparer, poisonKey: 7, () => d.TryRemove(3, out _));

            Assert.IsNull(thrown, $"the renumbering removal should not hash other keys; it threw {thrown?.GetType().Name}");
            Assert.AreEqual(Keys - 1, d.Count);
            Assert.IsFalse(d.ContainsKey(3));
            AssertConsistent(Of(d), IntLaw, "after TryRemove(3) with key 7 poisoned");
        }

        /// <summary>The compact type probes BOTH keys of a swap-back (the removed and the moved one) before any mutation, so the refused write-back while hashing the moved key aborts the in-place swap-back with nothing changed.</summary>
        [TestMethod]
        public void ConcurrentOrderedCompactDictionary_SwapBackRemoval_RefusedWriteBack_ChangesNothing()
        {
            var (d, comparer) = SeedConcurrentOrderedCompactDictionary(IntLaw);

            Exception thrown = RunPoisoned(comparer, poisonKey: 9, () => d.TryRemoveSwapBack(3, out _));

            Assert.IsInstanceOfType(thrown, typeof(LockRecursionException), "hashing the moved key must hit the refused write-back");
            Assert.AreEqual(Keys, d.Count, "the refusal happened before any mutation, so nothing may have changed");
            Assert.AreEqual(3, d.IndexOf(3));
            AssertConsistent(Of(d), IntLaw, "after TryRemoveSwapBack(3) was refused while probing moved key 9");
        }

        /// <summary>The same guarantee on the compact type's copying swap-back branch (wide values): both keys are probed in the live generation before the private copy is even made, so the refusal changes nothing.</summary>
        [TestMethod]
        public void ConcurrentOrderedCompactDictionary_WideSwapBackRemoval_RefusedWriteBack_ChangesNothing()
        {
            var (d, comparer) = SeedConcurrentOrderedCompactDictionary(WideLaw);

            Exception thrown = RunPoisoned(comparer, poisonKey: 9, () => d.TryRemoveSwapBack(3, out _));

            Assert.IsInstanceOfType(thrown, typeof(LockRecursionException), "hashing the moved key must hit the refused write-back");
            Assert.AreEqual(Keys, d.Count, "the refusal happened before any mutation, so nothing may have changed");
            Assert.AreEqual(3, d.IndexOf(3));
            AssertConsistent(Of(d), WideLaw, "after the copying TryRemoveSwapBack(3) was refused while probing moved key 9");
        }

        /// <summary>The compact type's RemoveWhere runs only the predicate as user code and repairs its index from stored tags, so a poisoned comparer is never reached and the pass completes consistently.</summary>
        [TestMethod]
        public void ConcurrentOrderedCompactDictionary_RemoveWhere_NeverConsultsTheComparer()
        {
            var (d, comparer) = SeedConcurrentOrderedCompactDictionary(IntLaw);

            Exception thrown = RunPoisoned(comparer, poisonKey: 7, () => d.RemoveWhere((k, _) => k % 2 == 0));

            Assert.IsNull(thrown, $"RemoveWhere should not hash keys; it threw {thrown?.GetType().Name}");
            Assert.AreEqual(Keys / 2, d.Count);
            AssertConsistent(Of(d), IntLaw, "after RemoveWhere(even) with key 7 poisoned");
        }

        // ------------------------------------------------------------------------------------------------------------
        // OrderedDictionary — consistent (its stuck-flag defect is covered elsewhere).
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>OrderedDictionary rebuilds a PRIVATE generation (re-hashing every survivor through the comparer) and publishes only at the end, so a write-back refused mid-rebuild leaves the published generation exactly as it was.</summary>
        [TestMethod]
        public void OrderedDictionary_InteriorRemoval_RefusedWriteBackMidRebuild_LeavesThePublishedGenerationIntact()
        {
            var comparer = new PoisonComparer();
            var d = new OrderedDictionary<int, int>(comparer);
            for (int k = 0; k < Keys; k++)
            {
                d.TryAdd(k, IntLaw(k));
            }

            comparer.Attack = () => d.TryAdd(-1, IntLaw(-1));

            Exception thrown = RunPoisoned(comparer, poisonKey: 7, () => d.TryRemove(3, out _));

            Assert.IsInstanceOfType(thrown, typeof(LockRecursionException), "harness: the write-back must be refused mid-rebuild");
            Assert.AreEqual(Keys, d.Count, "the rebuilt generation was never published, so the removal must not be visible");
            AssertConsistent(Of(d), IntLaw, "after TryRemove(3) threw while re-hashing key 7");
        }

        // ------------------------------------------------------------------------------------------------------------
        // Harness.
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>Seeds keys 0..9 into a ConcurrentOrderedDictionary whose comparer's attack is a write-back into the same instance.</summary>
        /// <typeparam name="TValue">The value type; word-sized types take the in-place swap-back branch, wide ones the copying branch.</typeparam>
        /// <param name="law">The value stored for each key.</param>
        /// <returns>The instance and its (disarmed) comparer.</returns>
        private static (ConcurrentOrderedDictionary<int, TValue> Dict, PoisonComparer Comparer) SeedConcurrentOrderedDictionary<TValue>(Func<int, TValue> law)
        {
            var comparer = new PoisonComparer();
            var d = new ConcurrentOrderedDictionary<int, TValue>(comparer);
            for (int k = 0; k < Keys; k++)
            {
                d.TryAdd(k, law(k));
            }

            // The documented attack: a comparer writing back into the collection it serves. Under the write lock the type
            // refuses it with LockRecursionException, thrown from inside GetHashCode.
            comparer.Attack = () => d.TryAdd(-1, law(-1));
            return (d, comparer);
        }

        /// <summary>Seeds keys 0..9 into a ConcurrentOrderedCompactDictionary whose comparer's attack is a write-back into the same instance.</summary>
        /// <typeparam name="TValue">The value type; word-sized types take the in-place swap-back branch, wide ones the copying branch.</typeparam>
        /// <param name="law">The value stored for each key.</param>
        /// <returns>The instance and its (disarmed) comparer.</returns>
        private static (ConcurrentOrderedCompactDictionary<int, TValue> Dict, PoisonComparer Comparer) SeedConcurrentOrderedCompactDictionary<TValue>(Func<int, TValue> law)
        {
            var comparer = new PoisonComparer();
            var d = new ConcurrentOrderedCompactDictionary<int, TValue>(comparer);
            for (int k = 0; k < Keys; k++)
            {
                d.TryAdd(k, law(k));
            }

            comparer.Attack = () => d.TryAdd(-1, law(-1));
            return (d, comparer);
        }

        /// <summary>Arms <paramref name="comparer" /> on <paramref name="poisonKey" />, runs <paramref name="operation" />, always disarms, and returns what the operation threw.</summary>
        /// <param name="comparer">The comparer the collection under test was built with.</param>
        /// <param name="poisonKey">The key whose hashing runs the attack.</param>
        /// <param name="operation">The mutation to interrupt.</param>
        /// <returns>The exception the operation threw, or <see langword="null" /> if it completed.</returns>
        private static Exception RunPoisoned(PoisonComparer comparer, int poisonKey, Action operation)
        {
            comparer.Arm(poisonKey);
            try
            {
                operation();
                return null;
            }
            catch (Exception ex)
            {
                return ex;
            }
            finally
            {
                // Disarm before auditing: the audit itself hashes every key.
                comparer.Disarm();
            }
        }

        /// <summary>A <c>TryGetValue</c> shape the three collection types all satisfy, so one audit covers them.</summary>
        /// <typeparam name="TValue">The value type.</typeparam>
        /// <param name="key">The key to look up.</param>
        /// <param name="value">The value found, or default.</param>
        /// <returns>Whether the key resolved.</returns>
        private delegate bool TryGet<TValue>(int key, out TValue value);

        /// <summary>The read surface the audit needs, bound to one instance of any of the three ordered map types.</summary>
        /// <typeparam name="TValue">The value type.</typeparam>
        /// <param name="Name">The type name for failure messages.</param>
        /// <param name="Count">The live count.</param>
        /// <param name="KeysInOrder">The keys in enumeration order.</param>
        /// <param name="ValuesInOrder">The values in enumeration order.</param>
        /// <param name="IndexOf">Key → position.</param>
        /// <param name="TryGetValue">Key → value.</param>
        /// <param name="ContainsKey">Key membership.</param>
        private sealed record Surface<TValue>(string Name, Func<int> Count, Func<int[]> KeysInOrder, Func<TValue[]> ValuesInOrder, Func<int, int> IndexOf, TryGet<TValue> TryGetValue, Func<int, bool> ContainsKey);

        /// <summary>Binds the audit surface to a ConcurrentOrderedDictionary.</summary>
        /// <typeparam name="TValue">The value type.</typeparam>
        /// <param name="d">The instance.</param>
        /// <returns>Its surface.</returns>
        private static Surface<TValue> Of<TValue>(ConcurrentOrderedDictionary<int, TValue> d) => new("ConcurrentOrderedDictionary", () => d.Count, () => d.Keys, d.ToArray, d.IndexOf, d.TryGetValue, d.ContainsKey);

        /// <summary>Binds the audit surface to a ConcurrentOrderedCompactDictionary.</summary>
        /// <typeparam name="TValue">The value type.</typeparam>
        /// <param name="d">The instance.</param>
        /// <returns>Its surface.</returns>
        private static Surface<TValue> Of<TValue>(ConcurrentOrderedCompactDictionary<int, TValue> d) => new("ConcurrentOrderedCompactDictionary", () => d.Count, () => d.Keys, d.ToArray, d.IndexOf, d.TryGetValue, d.ContainsKey);

        /// <summary>Binds the audit surface to an OrderedDictionary.</summary>
        /// <typeparam name="TValue">The value type.</typeparam>
        /// <param name="d">The instance.</param>
        /// <returns>Its surface.</returns>
        private static Surface<TValue> Of<TValue>(OrderedDictionary<int, TValue> d) => new("OrderedDictionary", () => d.Count, d.KeysToArray, d.ToArray, d.IndexOf, d.TryGetValue, d.ContainsKey);

        /// <summary>
        ///     Asserts the two access paths describe the same set of entries: the list path (count, ordered keys/values) and
        ///     the key path (membership, position, value) agree for every seeded key, no key enumerates twice, and the value
        ///     law holds, so no key is paired with another key's value.
        /// </summary>
        /// <typeparam name="TValue">The value type.</typeparam>
        /// <param name="s">The surface of the instance to audit (must be quiescent: no concurrent writers).</param>
        /// <param name="law">The value every key must carry.</param>
        /// <param name="context">What just happened, for the failure message.</param>
        /// <exception cref="AssertFailedException">Any disagreement; the message lists up to eight of them.</exception>
        private static void AssertConsistent<TValue>(Surface<TValue> s, Func<int, TValue> law, string context)
        {
            var eq = EqualityComparer<TValue>.Default;
            var problems = new List<string>();
            int count = s.Count();
            int[] keys = s.KeysInOrder();
            TValue[] values = s.ValuesInOrder();
            if (keys.Length != count || values.Length != count)
            {
                problems.Add($"Count={count} but {keys.Length} key(s) / {values.Length} value(s) enumerate");
            }

            var enumerated = new HashSet<int>();
            for (int i = 0; i < System.Math.Min(keys.Length, values.Length); i++)
            {
                int k = keys[i];
                if (!enumerated.Add(k))
                {
                    problems.Add($"key {k} enumerates twice (again at position {i})");
                }

                if (!eq.Equals(values[i], law(k)))
                {
                    problems.Add($"position {i}: key {k} is paired with value {values[i]}, expected {law(k)}");
                }

                if (!s.ContainsKey(k))
                {
                    problems.Add($"key {k} enumerates at position {i} but ContainsKey({k}) is false");
                }

                int at = s.IndexOf(k);
                if (at != i)
                {
                    problems.Add($"IndexOf({k}) = {at}, but the key enumerates at position {i}");
                }

                if (!s.TryGetValue(k, out TValue byKey) || !eq.Equals(byKey, values[i]))
                {
                    problems.Add($"TryGetValue({k}) disagrees with the value {values[i]} enumerated at position {i}");
                }
            }

            for (int k = 0; k < Keys; k++)
            {
                if (!enumerated.Contains(k) && (s.ContainsKey(k) || s.IndexOf(k) != -1 || s.TryGetValue(k, out _)))
                {
                    problems.Add($"key {k} resolves on the key path but never enumerates");
                }
            }

            Assert.AreEqual(0, problems.Count, $"{s.Name} {context}: the key and list paths disagree — " + string.Join("; ", problems.Take(8)));
        }

        /// <summary>
        ///     A key comparer that runs an attack — typically a write-back into the collection it serves — when hashing one
        ///     armed "poison" key, and throws <see cref="InvalidOperationException" /> if the attack itself did not throw.
        ///     Hashing is the key itself, so poison selection is exact.
        /// </summary>
        private sealed class PoisonComparer : IEqualityComparer<int>
        {
            /// <summary>The key whose hashing runs the attack (<see cref="int.MinValue" /> = disarmed).</summary>
            private int _poison = int.MinValue;

            /// <summary>Gets or sets the action run when the poison key is hashed; under the collection's write lock a write-back is refused with <see cref="LockRecursionException" />, which then escapes <c>GetHashCode</c>.</summary>
            public Action Attack { get; set; }

            /// <summary>Makes hashing <paramref name="key" /> run the attack until <see cref="Disarm" />.</summary>
            /// <param name="key">The poison key.</param>
            public void Arm(int key) => Volatile.Write(ref _poison, key);

            /// <summary>Restores normal hashing for every key.</summary>
            public void Disarm() => Volatile.Write(ref _poison, int.MinValue);

            /// <summary>Plain integer equality.</summary>
            /// <param name="x">The first key.</param>
            /// <param name="y">The second key.</param>
            /// <returns>Whether the keys are equal.</returns>
            public bool Equals(int x, int y) => x == y;

            /// <summary>The key itself as its hash; for the armed poison key, runs the attack and then throws.</summary>
            /// <param name="key">The key to hash.</param>
            /// <returns>The key's hash.</returns>
            /// <exception cref="LockRecursionException">The attack's write-back was refused because the caller holds the collection's write lock.</exception>
            /// <exception cref="InvalidOperationException">The poison key was hashed and the attack did not throw on its own.</exception>
            public int GetHashCode(int key)
            {
                if (key == Volatile.Read(ref _poison))
                {
                    Attack?.Invoke();

                    // An attack that was not refused (e.g. hashed outside the lock) still must not let the hash succeed:
                    // the scenario is "the comparer fails here", whichever way it fails.
                    throw new InvalidOperationException($"poisoned comparer: hashing key {key}");
                }

                return key;
            }
        }
    }
}
