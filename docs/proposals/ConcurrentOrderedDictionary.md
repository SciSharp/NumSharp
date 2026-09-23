# API Proposal: `System.Collections.Concurrent.ConcurrentOrderedDictionary<TKey, TValue>`

**Status:** Draft proposal · **Area:** System.Collections.Concurrent · **Type:** New type
**Reference implementation:** `NumSharp.Collections.OrderedDictionary<TKey, TValue>` (measured + concurrency-verified; see [§8](#8-reference-implementation--evidence))

> **Naming note.** The reference implementation is NumSharp's lock-free, value-once
> `NumSharp.Collections.OrderedDictionary<TKey, TValue>`. Every *unqualified* `OrderedDictionary<TKey, TValue>` /
> `OrderedDictionary<,>` in this document means the BCL's `System.Collections.Generic.OrderedDictionary<TKey, TValue>`
> (.NET 9), and every `ConcurrentOrderedDictionary` means the type proposed here. NumSharp also ships a node-based
> sibling named `NumSharp.Collections.ConcurrentOrderedDictionary<TKey, TValue>`; despite the shared name it is **not**
> the reference implementation.

---

## 1. Summary

A **thread-safe, insertion-ordered, index-addressable** dictionary with **lock-free reads**. It is the concurrent
counterpart of `System.Collections.Generic.OrderedDictionary<TKey, TValue>` (added in .NET 9, not thread-safe) and the
ordered counterpart of `System.Collections.Concurrent.ConcurrentDictionary<TKey, TValue>` (thread-safe, unordered). No
existing BCL type is all three of thread-safe, ordered, and index-addressable at once.

## 2. Background and motivation

The framework has three near-neighbors, none of which occupies this cell:

| Type | Thread-safe | Insertion-ordered | Index-addressable (`this[int]` / `IndexOf`) |
|---|:---:|:---:|:---:|
| `Dictionary<TKey, TValue>` | ❌ | ❌ (no contract) | ❌ |
| `OrderedDictionary<TKey, TValue>` (.NET 9) | ❌ | ✅ | ✅ |
| `ConcurrentDictionary<TKey, TValue>` | ✅ | ❌ | ❌ |
| **`ConcurrentOrderedDictionary<TKey, TValue>` (proposed)** | ✅ | ✅ | ✅ (read) |

Today, code that needs "a dictionary that keeps insertion order **and** is safe under concurrent readers/writers" has
only bad options:

- **Lock a `Dictionary`/`OrderedDictionary` manually** — correct but serializes *reads* too (no read scaling), and
  every call site must remember the lock. This is the pattern this type is meant to replace.
- **`ConcurrentDictionary`** — scales, but has **no order** and **no positional access**; enumeration order is
  unspecified and changes across runs.
- **Roll your own** — the lock-free-read + ordered + generational-resize machinery is genuinely hard to get right
  (the reference implementation's value below is a concurrency gun that catches exactly these bugs).

**Representative scenarios**

- Concurrent caches and registries that must **enumerate/replay in insertion order** (event logs keyed by id,
  ordered LRU-style stores, deterministic pipeline stages).
- Producer/consumer maps where readers scan the values as an **ordered, index-addressable list** while writers
  update entries by key — a `List<T>`-shaped read view over a keyed, concurrent store.
- Observable/UI-backing collections that need a stable, index-addressable order under background mutation.

## 3. Goals and non-goals

**Goals**
- Lock-free `O(1)` reads (`TryGetValue`, `ContainsKey`, `IndexOf`, positional get, `Count`) that scale with cores.
- Deterministic **insertion order** for enumeration, `ToArray`, positional access, and the `Keys`/`Values` views.
- The full set of **atomic combinators** (`TryAdd`, `TryUpdate`, `TryRemove`, `GetOrAdd`, `AddOrUpdate`) so callers
  never need an external lock for check-then-act.
- Snapshot enumeration that never throws `InvalidOperationException` under concurrent mutation (the
  `ConcurrentDictionary` contract).
- Compact footprint (value stored **once**; ~3 machine words per entry).

**Non-goals**
- **Arbitrary positional insertion / positional removal** (`Insert(index, …)`, `RemoveAt(index)` at a chosen slot).
  Insertion order is append-defined; removal is by key. Positional *mutation* under concurrency + ordering is both
  semantically fraught (indices shift under other removes) and `O(n)`; it is intentionally out of scope.
- **Lock-free writers.** Structural writes are serialized (see [§7.2](#72-thread-safety-contract)); this is the price
  of a global insertion order, and matches how `ConcurrentDictionary` itself uses locks for writes.
- **`O(1)` order-preserving interior removal.** Order-preserving removal is `O(n)` (like `List<T>.RemoveAt`); see
  [§7.4](#74-complexity) and [§9](#9-risks).

## 4. API proposal

```csharp
namespace System.Collections.Concurrent;

/// <summary>
/// A thread-safe dictionary that preserves insertion order and supports positional (index) reads.
/// Reads are lock-free; structural writes are serialized; enumeration is a point-in-time snapshot.
/// </summary>
public class ConcurrentOrderedDictionary<TKey, TValue> :
    IDictionary<TKey, TValue>,
    IReadOnlyDictionary<TKey, TValue>,
    IReadOnlyList<KeyValuePair<TKey, TValue>>,   // positional read, in insertion order
    ICollection<KeyValuePair<TKey, TValue>>
    where TKey : notnull
{
    // ---- construction ----
    public ConcurrentOrderedDictionary();
    public ConcurrentOrderedDictionary(int capacity);
    public ConcurrentOrderedDictionary(IEqualityComparer<TKey>? comparer);
    public ConcurrentOrderedDictionary(int capacity, IEqualityComparer<TKey>? comparer);
    public ConcurrentOrderedDictionary(IEnumerable<KeyValuePair<TKey, TValue>> collection,
                                       IEqualityComparer<TKey>? comparer = null);

    // ---- state ----
    public IEqualityComparer<TKey> Comparer { get; }
    public int Count { get; }
    public bool IsEmpty { get; }

    // ---- keyed access (lock-free reads; set is lock-free for atomically-writable values) ----
    public TValue this[TKey key] { get; set; }               // NB: with TKey==int, the compiler binds d[i] to this[int]
    public bool ContainsKey(TKey key);
    public bool TryGetValue(TKey key, out TValue value);

    // ---- positional access (read-only, insertion order, lock-free) ----
    public KeyValuePair<TKey, TValue> this[int index] { get; }
    public int IndexOf(TKey key);                            // -1 if absent
    public TKey GetKeyAt(int index);
    public TValue GetValueAt(int index);

    // ---- atomic combinators (never require an external lock) ----
    public bool TryAdd(TKey key, TValue value);
    public bool TryUpdate(TKey key, TValue newValue, TValue comparisonValue);
    public bool TryRemove(TKey key, out TValue value);
    public bool TryRemove(KeyValuePair<TKey, TValue> item);
    public TValue GetOrAdd(TKey key, TValue value);
    public TValue GetOrAdd(TKey key, Func<TKey, TValue> valueFactory);
    public TValue GetOrAdd<TArg>(TKey key, Func<TKey, TArg, TValue> valueFactory, TArg factoryArgument);
    public TValue AddOrUpdate(TKey key, TValue addValue, Func<TKey, TValue, TValue> updateValueFactory);
    public TValue AddOrUpdate(TKey key, Func<TKey, TValue> addValueFactory,
                              Func<TKey, TValue, TValue> updateValueFactory);
    public TValue AddOrUpdate<TArg>(TKey key, Func<TKey, TArg, TValue> addValueFactory,
                                    Func<TKey, TValue, TArg, TValue> updateValueFactory, TArg factoryArgument);
    public void Clear();

    // ---- bulk / snapshot views (each is a point-in-time snapshot) ----
    public KeyValuePair<TKey, TValue>[] ToArray();
    public IReadOnlyList<TKey> Keys { get; }                 // ordered
    public IReadOnlyList<TValue> Values { get; }             // ordered
    public Enumerator GetEnumerator();                       // insertion order

    public struct Enumerator : IEnumerator<KeyValuePair<TKey, TValue>>
    {
        public KeyValuePair<TKey, TValue> Current { get; }
        public bool MoveNext();
        public void Reset();
        public void Dispose();
    }

    // Explicit interface members (IDictionary/ICollection/IReadOnly*) map onto the above; e.g.
    //   IDictionary<TKey,TValue>.Add            -> TryAdd, throwing ArgumentException on a duplicate key
    //   ICollection<KeyValuePair<>>.IsReadOnly  -> false
    //   IReadOnlyList<KeyValuePair<>>.this[int]  -> positional read
    // (elided here for brevity)
}
```

### 4.1 Notes on the surface

- **`TryUpdate(key, newValue, comparisonValue)`** is the compare-and-swap update from `ConcurrentDictionary`; it is
  the atomic building block for lock-free read-modify-write on a value and is required for correctness under
  concurrency (a plain `this[key] = f(this[key])` is a torn read-modify-write).
- **Positional API is read-only.** `this[int]`, `GetKeyAt`, `GetValueAt`, `IndexOf` give the `List<T>`-style view;
  there is deliberately no `Insert(index,…)`/`SetAt(index,…)`/`RemoveAt(index)`. See [§3](#3-goals-and-non-goals).
- **`Keys`/`Values`** return ordered, snapshotting `IReadOnlyList<>` views (not live, not the `ICollection` the
  `IDictionary` interface member returns — that stays an explicit implementation).
- **The `int`-key footgun** (`d[i]` binds to `this[int]` when `TKey==int`) is inherited from `OrderedDictionary<,>`
  and is documented identically: use `TryGetValue`/`GetValueAt` explicitly with `int` keys.

## 5. API usage

```csharp
var d = new ConcurrentOrderedDictionary<string, int>();

// atomic mutators — no external lock, ever
d.TryAdd("a", 1);
int v = d.GetOrAdd("b", static k => Compute(k));
d.AddOrUpdate("a", 1, static (k, old) => old + 1);      // "a" -> 2
d.TryUpdate("a", 3, comparisonValue: 2);                 // CAS: succeeds only if "a" is still 2

// ordered, index-addressable reads (lock-free)
foreach (var kv in d) Console.WriteLine(kv);             // insertion order: a, b
KeyValuePair<string,int> first = d[0];                   // positional
int idx = d.IndexOf("b");                                // 1

// a List<T>-shaped read view over the values, safe to scan while others mutate
IReadOnlyList<int> values = d.Values;                    // snapshot, ordered
```

## 6. Semantics and design

### 6.1 Ordering

Entries enumerate, and are positionally addressed, in **insertion order**. `TryRemove` preserves order (subsequent
entries shift down one position). `Clear` resets to empty. Re-adding a previously-removed key places it at the end
(a new insertion). Replacing a value (`this[key] = …`, `TryUpdate`, `AddOrUpdate` on an existing key) does **not**
change its position.

### 6.2 Layout (why it is compact and read-fast)

An open-addressed index of 32-bit slot references maps each key to a dense insertion-order slot; two contiguous arrays
hold keys and values at that slot. A value is stored **once** (≈ 3 machine words per entry — index word + key + value),
versus a node-per-entry design. Positional reads and enumeration scan the contiguous value array directly (`List<T>`
memory profile). All shape lives in a single immutable-in-shape *generation* published through one volatile field, so a
reader takes a consistent snapshot in a single acquiring read.

## 7. Concurrency specification

This is the load-bearing part of the contract.

### 7.1 Guarantees

- **No torn or garbage reads.** A reader never observes a partially-written entry or a torn value. The per-slot
  reference is published with release / read with acquire; a value update to an atomically-writable `TValue` is a
  single atomic store; a value update to a non-atomically-writable `TValue` (a wide struct, or `long` on a 32-bit
  runtime) publishes a fresh value array instead of storing in place, so it cannot tear either.
- **No lost updates.** A completed write is always durably reflected, *including* the hard race of a lock-free value
  update running concurrently with a resize that copies the backing arrays: a resize-in-progress flag held across the
  whole copy+publish, combined with a generation re-check after the store, forces the update to retry under the lock if
  it could have landed on an abandoned generation.
- **No corruption.** The structure is never left internally inconsistent under any interleaving.
- **Snapshot enumeration.** `GetEnumerator`, `ToArray`, `Keys`, `Values` capture a point in time; they never throw
  `InvalidOperationException` due to concurrent mutation and reflect a consistent prefix/subset — the
  `ConcurrentDictionary` enumeration contract.

### 7.2 Thread-safety contract

- **Reads are lock-free and wait-free-ish** (`TryGetValue`, `ContainsKey`, `IndexOf`, positional gets, `Count`,
  starting an enumeration): they take no lock and scale with cores.
- **Value replacement of an existing key** (`this[key] = …`, `TryUpdate`, `AddOrUpdate`-update) is **lock-free** when
  `TValue` is atomically writable; otherwise it takes the write lock.
- **Structural writes** (`TryAdd`, `TryRemove`, `Clear`, add-path of `GetOrAdd`/`AddOrUpdate`, resize) are
  **serialized by a single lock** — one at a time. This is deliberate: a global insertion order requires a
  serialization point. (`ConcurrentDictionary` likewise locks for writes.)
- **Consistency model.** Per key, operations are linearizable (a value update on key *k* is immediately visible to a
  later read of *k*). *Across* keys/paths, views are snapshot-consistent, not instantaneously linked: `Count`,
  enumeration, and `IndexOf` are each a moment-in-time value, exactly as with `ConcurrentDictionary` — do not assume
  two *separate* calls (`IndexOf(k)` then `this[i]`) line up while writers run.
- **Memory model.** All cross-thread publication uses `Volatile`/release-acquire, so the guarantees hold on weak
  memory models (Arm64), not only x86/x64.

### 7.3 What is *not* a fault (expected concurrent behavior)

- A read concurrent with a write on the same key returns the value from just-before or just-after it (both valid);
  last-writer-wins for two concurrent value updates.
- A key mid-insertion may not yet be visible to a concurrent reader (the insert races the read) — but a half-inserted
  entry is never observed.
- `this[int]`/`GetValueAt` racing a removal that shrinks `Count` may throw `ArgumentOutOfRangeException`; `this[key]`
  racing a removal may throw `KeyNotFoundException`. These are defined outcomes; use a snapshot (`ToArray`, `Values`)
  for hot ordered scans.
- A user callback (a `valueFactory`/`updateValueFactory`, an `IEnumerable` source, the comparer) that this type runs
  *inside its write lock* must not re-enter and write back; doing so is refused with `LockRecursionException` rather
  than deadlocking or corrupting. Factory callbacks are otherwise invoked outside the lock, matching
  `ConcurrentDictionary`.

### 7.4 Complexity

| Operation | Complexity | Synchronization |
|---|---|---|
| `TryGetValue`, `ContainsKey`, `IndexOf`, `this[key]` get | `O(1)` average | lock-free |
| `this[int]`, `GetKeyAt`, `GetValueAt`, `Count` | `O(1)` | lock-free |
| `this[key]` set / `TryUpdate` / `AddOrUpdate` (existing, atomic `TValue`) | `O(1)` | lock-free |
| `TryAdd`, `GetOrAdd` (add), grow | `O(1)` amortized | serialized (lock) |
| **`TryRemove` (order-preserving)** | **`O(n)`** | serialized (lock) |
| `Clear` | `O(1)` | serialized (lock) |
| `GetEnumerator`/`ToArray`/`Keys`/`Values` | `O(1)` to obtain, `O(n)` to consume | snapshot |

Memory: ~3 machine words per entry (value stored once).

## 8. Reference implementation & evidence

A complete implementation exists as `NumSharp.Collections.OrderedDictionary<TKey, TValue>`, with the following measured
results (N = 500,000; single process; best-of-9; x64):

**vs the closest existing type, `System.Collections.Generic.OrderedDictionary<TKey, TValue>` (.NET 9, single-threaded):** reads at parity
(key-get 0.93×, `IndexOf` 0.96×, positional 1.06×), enumeration **1.53×** (contiguous span vs the boxed enumerator),
build 0.25× and replace 0.84× — the modest single-threaded overhead of the synchronization that buys thread-safety.
The differentiator is not single-thread speed; it is that `OrderedDictionary<,>` **is not thread-safe at all**, whereas
this type provides lock-free concurrent reads a manually-locked `OrderedDictionary` cannot (a lock serializes reads).

**vs `ConcurrentDictionary` (the thread-safe baseline it would sit beside):** faster on every measured op — build
2.3–4.6×, key-get 2.9×, replace 23×, enumeration 41× — and **2.8× the concurrent-read throughput** (≈914M vs ≈325M
lookups/sec across 4 threads), in addition to providing ordering and positional access it lacks entirely.

**vs `Dictionary`/`List` (single-threaded specialists):** wins reads/enumeration vs `Dictionary` (key-get 1.6×,
enum 1.6×); loses build to both and positional to `List` (raw array, no hashing) — expected, and not the comparison
class (neither is thread-safe or, for `List`, keyed).

**Concurrency verification.** A randomized concurrency gun (the same class of test the reference codebase uses for its
other concurrent collections) exercised **all** public methods under 20 threads: 8 writers on disjoint key ranges with
a per-key linearizable model checked on **every** operation, 6 lock-free value-replacers on shared keys racing constant
resizes, 6 readers over every read method with value-encoding tear detection, and 800+ quiescent full-consistency
barriers, plus a `Clear`/bulk-add chaos phase. Result across two runs: **~280M operations, 0 tears, 0 lost updates, 0
corruption, 0 crashes.** A focused concurrent-get-plus-write test on the *same* keys under resize churn: **195.7M
concurrent reads + 6.0M concurrent writes, 0 faulty reads, 0 lost writes.**

## 9. Alternative designs

1. **Name / namespace.** `ConcurrentOrderedDictionary<TKey, TValue>` in `System.Collections.Concurrent` is the
   convention-correct choice (mirrors `ConcurrentDictionary`, pairs with `OrderedDictionary<,>`). Alternatives
   considered: `OrderedConcurrentDictionary` (word order), or an `ordered:` option on `ConcurrentDictionary`
   (rejected — ordering changes the type's iteration contract and cost model, not a flag).
2. **Inline-entry layout** (key+value+hash in one node array, like `Dictionary`) would win single-thread build and
   key-get but doubles the enumeration/positional memory walk and loses the `List<T>` value-scan profile and the
   compact footprint. The value-once layout is chosen to make the ordered/positional read path first-class.
3. **Lock-free writers** (a fully non-blocking ordered append) was prototyped and rejected: a correct multi-producer
   *ordered* append with de-duplication hits the dead-slot/tombstone problem, which compromises the dense-order
   invariant and slows reads. Serializing writers (as `ConcurrentDictionary` does) is simpler and correct.
4. **Positional mutation** (`Insert`/`RemoveAt`/`SetAt` by index) to fully match `OrderedDictionary<,>` — deferred as
   an [open question](#10-open-questions); `O(n)` and racy under concurrency.

## 10. Open questions

- Should positional **mutation** (`Insert(index,…)`, `RemoveAt(index)`, `SetAt(index,…)`) be included for parity with
  `OrderedDictionary<TKey, TValue>`, given its `O(n)` cost and the index-shift hazard under concurrency? (Proposal:
  no, initially.)
- Should an explicitly **order-breaking `O(1)` removal** be offered (swap-the-last-entry-into-the-hole) for
  remove-heavy workloads that do not need order preserved after a delete?
- `Keys`/`Values` as **snapshotting** `IReadOnlyList<>` vs **live** views — snapshot is proposed for a clean
  concurrent contract; a lazy live view is possible but its ordering under concurrent structural change is subtle.
- `IProducerConsumerCollection<T>` and the non-generic `IDictionary`/`ICollection` surface — include for full parity
  with `ConcurrentDictionary`, or keep the generic surface only?

## 11. Risks

- **`O(n)` order-preserving removal.** Interior deletes are `List<T>.RemoveAt`-class; remove-heavy ordered workloads
  are a poor fit. Documented, and the [open question](#10-open-questions) on swap-back removal mitigates it opt-in.
- **Serialized writers.** Write-concurrency does not scale (reads do). Acceptable and standard for an ordered
  concurrent structure, but worth stating plainly.
- **API surface size.** The type combines the `ConcurrentDictionary` combinators with the `OrderedDictionary`
  positional surface; care is needed to keep the explicit-interface mapping and the `int`-key indexer story clear.
- **Two `this[...]` indexers** (`int` and `TKey`) create the documented `int`-key ambiguity, inherited from
  `OrderedDictionary<,>`.
```
