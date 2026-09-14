# `ConcurrentOrderedDict<TKey,TValue>` — performance comparison & gaps

A thread-safe, insertion-ordered, index-addressable map: a **key path** (dedup + O(1) lookup by
`TKey`) and a **list path** (O(1) access by `int` index; enumeration/`ToArray`/`Values` yield the
values in insertion order). It is built on the vendored `ConcurrentDictionary` clone in this folder.

This note tracks the **two speed targets** and what closed the gaps. **Status: every closable gap is
closed** — the 2026-09-14 rewrite landed the inline-value node payload, the mutable-count store, the
clone ref seam, and the batch/view/O(1)-removal APIs; the sections below record each gap's fix and
the residual, structural costs that remain by design.

* **Target A — `List<TValue>`** for the *list-path read* operations (enumerate, `ToArray`, index get). ✅ met
* **Target B — `ConcurrentDictionary<TKey,TValue>`** for the *key-path* operations (get, add, update, remove). ✅ met (get/update/remove at parity; unsized sequential add ~1.2× with presized/batched forms at or beyond parity)

> ⚠️ These are different baselines for different operations. `List<T>` is not thread-safe and has no
> keys; `ConcurrentDictionary` has no order and no index. `ConcurrentOrderedDict` does *all* of it at
> once, so some overhead over each baseline is structural, not incidental — the sections below mark
> which gaps were closed and which costs are inherent.

---

## Measured comparison (before → after the 2026-09-14 rewrite)

Release, best-of-11, 1M `int` entries, single-threaded micro-benchmarks on an i9-13900K (treat the
**ratios**, not the absolute ms, as the signal; `SysCD` absolute times swing ±60% run-to-run with
allocator/GC regime, ratios within one run are stable). Reproduce with the BDN suite below.

| Operation | vs baseline | before | after | Verdict |
|---|---|---|---|---|
| `foreach` / enumerate | `List<T>` | ~1.0× | **1.02×** (snapshot-view foreach **0.87×** — beats List) | ✅ parity |
| `ToArray()` | `List<T>` | ~1.0× | **0.99×** | ✅ parity |
| index get `this[int]` | `List<T>` | ~1.36× | **~1.2×** live / **0.99×** via `Snapshot()` view / **0.97×** via `AsSpan()` | ✅ closed (A1) |
| key get `TryGetValue` | `ConcurrentDictionary` | ~1.4–3× | **0.98–1.06×** (run-to-run band of the clone itself) | ✅ closed (B1) |
| `IndexOf` (key→index) | CD `TryGetValue` | ~1.5× | **0.96×** | ✅ closed |
| sequential add (unsized) | `ConcurrentDictionary` | ~2.4–2.6× | **1.17–1.22×** | ✅ closed to the structural floor (B2) |
| sequential add (presized) | unsized CD | ~1.5× | **0.43–0.71×** (faster than unsized CD; ≈ presized CD + lock) | ✅ |
| `AddRange` (batch) | unsized CD build | — | **1.03×** | ✅ new API |
| replace existing `SetByKey` | CD indexer | ~1.2× | **1.02–1.07×** | ✅ closed |
| remove last (pop-back) | `ConcurrentDictionary` | **O(n) per pop** (quadratic drains) | **O(1) per pop**; 1M build+drain **faster than CD** (106 vs 117 ms) | ✅ closed (B3 tail) |
| remove any, order-breaking `TryRemoveSwapBack` | `ConcurrentDictionary` | — | **O(1)**; 20K build+drain 1.30× CD composite | ✅ new API |
| bulk remove `RemoveWhere` | CD remove loop | — | **0.86×** (one compaction pass beats CD's per-key removes) | ✅ new API |
| remove interior (order-preserving) | `ConcurrentDictionary` | O(n) | O(n) — array shift + tail re-index | 🔴 inherent (see B3) |

**Reads are lock-free** (key lookup, index, `Count`, `ToArray`, `Keys`, `Values`, enumeration, the
snapshot view). `Count`/`ToArray`/`Keys`/`Values` remain **finer-grained than `ConcurrentDictionary`**,
which takes *all* stripe locks for those; we serve them from the published snapshot with no lock.

---

## Design in one paragraph (post-rewrite)

The key path is `ConcurrentDictionary<TKey, ValueIndex>` (the clone, `concurrencyLevel: 1` since all
writes are externally serialized) where **`ValueIndex { TValue _value; int _index }` lives inline in
the hash node** — a key lookup is exactly one node visit, the same memory walk as the plain concurrent
dictionary. The list path is a **`Store { TKey[] _keys; TValue[] _values; readonly int _floor; int _count }`**
published through one `volatile` field, with three rules that keep every read lock-free and every
enumerator a true snapshot **without allocating a holder per append**: (1) `_count` only ever grows on
a given `Store` — appends write slots then release-publish the count; every shrinking transition swaps
the `Store` object; (2) slots a reader can see are never structurally rewritten — `_floor` records the
old high-water mark when an O(1) tail removal shares arrays downward, and an append below it
copies-on-write; (3) values are rewritten in place only when `TValue` is single-store atomic
(otherwise node-swap + array-clone, so nothing tears). Single-field in-place node writes (the atomic
value, the always-atomic index) go through the clone's `GetValueRefOrNullRef` ref seam under the one
global write lock.

---

## Gap A1 — index get was ~1.36× `List<T>` — ✅ CLOSED

**Cause.** `this[int]` does one `volatile` read of `_store` **per call** plus a bounds check.

**Fix shipped:** `Snapshot()` → `ValuesView` (readonly struct: `Count`, `this[int]`, `GetKeyAt`,
`AsSpan()`, `KeysAsSpan()`, span `GetEnumerator`). Captures the volatile state once; a tight loop over
the view measures **0.99×** `List<T>` and the span form **0.97×**.

- [x] Snapshot accessor (`Snapshot()` / `ValuesView`).
- [x] Documented: the view is point-in-time in *shape*; in-place atomic value replacements remain
      visible through it (the same live-value semantics enumeration always had).

The single-call `this[int]` keeps the volatile read (~1.2×) by design — it cannot drop it without
losing lock-free consistency; the view is the hot-loop escape hatch.

---

## Gap B1 — key get was ~1.4–3× `ConcurrentDictionary` — ✅ CLOSED

**Cause.** The old design stored a separate heap `Entry` per key and dereferenced `entry._value` —
one extra dependent pointer-chase (a second cache miss) per lookup.

**Fix shipped — the TODO's "Option 2", made to work:** the map value is now the struct
`ValueIndex { TValue, int }` stored **inline in the node** (measured: a struct node payload costs the
same as an `int` payload — 0.99× BCL), plus a tiny seam on the clone —
`internal ref TValue GetValueRefOrNullRef(TKey key)` in `ConcurrentDictionary.NumSharp.cs` — so the
two problems that made Option 2 "likely not a win" disappeared:

* **Updates don't reallocate nodes:** under the write lock, an atomically-writable value and the
  (always-atomic `int`) index are stored **in place through the ref**, one field at a time. Readers
  copy the struct with plain loads and can only ever see each field old-or-new (single aligned stores
  ≤ machine word), never a hybrid of one field — and every lock-free consumer reads exactly one field.
* **Removal shifts don't reallocate nodes:** the re-index writes `vi._index` in place (atomic int),
  same cost as the old mutable-`Entry` design, zero allocation, for any `TValue`.

Non-atomic `TValue` (e.g. `decimal`) keeps the clone's own tear-free discipline: replacement publishes
a fresh node (map indexer) + a cloned value array.

Attribution runs (kept for posterity): BCL 18.9 ms ≙ clone 18.7 ms ≙ clone-with-8B-struct 18.7 ms;
the last 25% was the *wrapper* (an out-`bool` stack round-trip + un-inlined call) — fixed by the
null-ref-sentinel seam (`Unsafe.IsNullRef`, a register compare) + `AggressiveInlining`. Final:
**0.98–1.06×**, the clone's own run-to-run band.

- [x] Option 2 (struct in node) via the ref seam — shipped.
- [x] ~~Option 1 (two maps)~~ — unnecessary; strictly worse (2× memory, 2× add work).
- [x] ~~Option 3 (accept 3×)~~ — superseded.

---

## Gap B2 — sequential add was ~2.4–2.6× `ConcurrentDictionary` — ✅ CLOSED to the structural floor

**Cause (was three allocations + three bucket walks).** Per add: an `Entry` alloc, a `Store` alloc, a
node alloc, plus contains-check walks before the insert walk.

**Fixes shipped, all three TODO items:**

- [x] **`AddRange(IEnumerable<KeyValuePair<…>>)`** — one lock for the whole batch, known-count presize
      (`TryGetNonEnumeratedCount`), one publication; newly appended entries become visible to the list
      path **atomically as one batch**. Upsert semantics identical to the pairs constructor (which now
      delegates to it). Measured **1.03×** an unsized CD build.
- [x] **Per-add `Store` allocation dropped** — not via the risky two-field seqlock, but by making
      `Store._count` mutable-and-monotonic: an in-place append writes the slots then release-publishes
      the count (`Volatile.Write`); a reader acquire-reads the count once and the slots below it are
      guaranteed fully written. Shrinking transitions still swap the `Store`, so an enumerator's
      captured `(array, count)` can never go backwards or tear.
- [x] **`Entry` allocation dropped entirely** — folded into the node as `ValueIndex` (Gap B1's fix).
      An in-capacity add now allocates exactly what `ConcurrentDictionary.TryAdd` allocates: one node.
- [x] **One bucket walk per add** — `TryAppendUnderLock` lets the map's `TryAdd` itself answer
      "already exists?" (the framework's own approach); the pre-checks that cost a full extra walk on
      the common absent path are gone (kept only where they guard O(n) work, i.e. removals).
- [x] **Pre-size** via `new ConcurrentOrderedDict(capacity)` — with growth gone, measured **faster
      than an unsized CD build** (0.43–0.71×) and ≈ presized CD + the lock.

**Residual (structural):** unsized sequential add is **~1.17–1.22×** CD. That is the floor for this
design: the same node alloc + hash walk as CD, plus the global `Monitor` (CD's stripe lock is the only
lock it takes), two array slot writes, the release count-publish, and doubling-growth of two arrays.
The lock cannot be striped (a global insertion order is a whole-structure invariant), and the arrays
are the entire list path. Use presize or `AddRange` where build speed matters.

---

## Gap B3 — ordered remove is O(n) (vs `ConcurrentDictionary` O(1)) — ✅ amortized/opted-out, interior case inherent

**Cause / inherent.** Preserving insertion order + contiguous indices means an interior remove shifts
the tail and re-indexes it. There is no way to keep both contiguous O(1) index *and* O(1) ordered
interior remove; tombstoning would forfeit Target A.

**Fixes shipped, all three TODO items:**

- [x] **O(1) tail removal** (new): removing the *last* entry publishes a lower-count `Store` over the
      **same arrays** — no copy, no shift, no re-index. The new `_floor` field freezes the vacated
      slot against in-place append reuse, so enumerators captured before the pop keep reading valid
      data (pinned by test: pop-then-push does not disturb an older enumerator's snapshot). Stack-like
      usage (append/pop at the back) is now O(1) per op — a 1M build+drain measures **faster than
      `ConcurrentDictionary`** (106 vs 117 ms).
- [x] **`TryRemoveSwapBack(key, out value)`** — opt-in, clearly-named order-breaking O(1) removal:
      moves the last entry into the hole (in place when both `TKey` and `TValue` are single-store
      atomic — the documented relaxation lets a concurrent enumerator see the moved entry twice; for
      wide keys/values it degrades to an O(n) memcpy that stays snapshot-pure, still re-index-free).
- [x] **`RemoveWhere(match)`** — one compaction pass for any number of removals, one publication,
      predicate constraints documented (runs under the write lock; exception-safe: a throwing
      predicate still lands the collection in the consistent "removed everything matched so far"
      state). Measured **0.86×** of CD's equivalent per-key remove loop.
- [x] **Cheaper re-index** — the shift's re-index is now an in-place atomic `int` store per shifted
      key through the ref seam (no allocation; the hash walk per shifted key remains — carrying a
      node array in the `Store` to avoid it would tax every append and was not worth it once the tail
      and bulk paths above removed the common quadratic drains).

**Residual (inherent):** a single order-preserving interior removal remains O(n) — array shift plus
tail re-index — exactly like `List<T>.RemoveAt`.

---

## Locking (write path) — global lock vs striped

- **Reads:** lock-free — same as `ConcurrentDictionary` on single keys, **finer** on
  `Count`/`ToArray`/`Keys`/`Values` (lock-free here; the framework type takes all stripe locks).
- **Writes:** a **single global lock**, coarser than CD's striped per-bucket locks. **Inherent**: the
  dominant write (append) is a whole-structure op (one global order, one shared array pair) and
  cannot be striped. The lock is held only for the mutation — user `valueFactory` /
  `updateValueFactory` delegates run **outside** it (framework contract), `AddOrUpdate` applies
  updates by compare-and-swap.
- The map is constructed with `concurrencyLevel: 1` on purpose: its stripe locks can never be
  contended (all map writes happen under our lock), so extra stripes were pure footprint.
- [ ] ~~`ReaderWriterLockSlim`~~ — evaluated and **rejected**: it does not speed the append path, adds
  uncontended overhead over `Monitor`, and would make the type `IDisposable`. The concurrency-gun
  suite shows the `Monitor` path scales fine for the update-churn scenarios it was meant for.

---

## Vendored `ConcurrentDictionary` clone — deviations from the BCL

The clone in `ConcurrentDictionary.cs` is faithful to dotnet/runtime's source; the concurrency core
(striped locks, lock-free reads, `GrowTable`) is verbatim. Deviations:

- **`partial`** (one keyword on the class declaration) so NumSharp-only members live in
  **`ConcurrentDictionary.NumSharp.cs`** without touching the vendored body. That file currently holds
  exactly one member: `internal ref TValue GetValueRefOrNullRef(TKey key)` — `TryGetValue`'s bucket
  walk returning `ref node._value` (null-ref sentinel when absent). Its three caller obligations
  (lifetime inside one serialized critical section; single-field ≤word-size writes only; caller-side
  write serialization) are documented on the member. A future re-sync against upstream is a plain file
  replace + re-adding the `partial` keyword.
- `AlternateLookup<TAlternateKey>` / `IAlternateEqualityComparer` dropped (net9+; uses
  `allows ref struct`, which does not compile on net8.0).
- The string-key `NonRandomizedStringEqualityComparer` hash-flooding hardening (CoreLib-internal) is
  absent; the provided/default comparer is used directly. Consequence: default-comparer **string**
  keys hash through randomized Marvin always — same as CD *after* it detects flooding; CD's
  pre-flooding fast string path can be a bit quicker. `int`/value-type keys are identical.
- The `[DebuggerTypeProxy]` debug view dropped.

`SR` / `ThrowHelper` / `HashHelpers` are vendored in `ConcurrentDictionaryInternals.cs` in the same
`NumSharp.Collections.Concurrent` namespace so the clone body keeps its original references verbatim.

---

## Gates (benchmarks + tests)

- **BDN suite** (fair, grouped, baselined): `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Collections/ConcurrentOrderedDictBenchmarks.cs`
  — Build / KeyGet / Enumerate / ToArray / IndexGet / Replace / Remove groups at N = 1K / 100K / 1M,
  each with its `List<T>` or framework-CD baseline **plus the vendored clone rows** so any gap is
  attributable (ordered-dict layer vs clone-vs-BCL drift). Run:
  `dotnet run -c Release -- --filter "*ConcurrentOrderedDict*"` (menu option 15).
- **Functional pins:** `test/NumSharp.Tests/Collections/ConcurrentOrderedDictTests.cs` — ordering,
  dup semantics, the int-key indexer footgun, tail-pop + append-floor snapshot purity, swap-back,
  RemoveWhere (incl. throwing-predicate consistency), AddRange (incl. mid-batch-throw consistency),
  views, non-atomic `decimal` paths, and a 5K-op randomized churn against a `List`-based oracle.
- **Concurrency gun:** `test/NumSharp.Tests/Collections/ConcurrentOrderedDictConcurrencyTests.cs` —
  every scenario arms N threads that park on ONE `ManualResetEventSlim` and fire simultaneously:
  exactly-once distinct adds, single-winner same-key adds/removes, `GetOrAdd` single-value,
  `AddOrUpdate` lost-increment counter, torn-read hunts (atomic `long` bit patterns AND non-atomic
  `decimal`), exact-prefix enumeration under appenders, the floor rule under pop/re-append fire,
  `AddRange` batch-atomicity (observed counts are batch multiples), swap-back survivor sets, index
  readers under structural churn, `Clear` vs everything, and a mixed-chaos soak with a value-law
  readers verify on every observation.

## How to measure (reproduce the table)

The `dotnet run` recipe below still works for quick spot checks (`dotnet run -c Release -` — Release
is mandatory, Debug taints NumSharp.Core ~2×); prefer the BDN suite for anything you intend to quote.

```csharp
#:project <path>/src/NumSharp.Core/NumSharp.Core.csproj
using System.Diagnostics; using NumSharp.Collections;
const int N = 1_000_000, Reps = 11;
double Ms(Action a){ a(); a(); long b=long.MaxValue; for(int r=0;r<Reps;r++){ var sw=Stopwatch.StartNew(); a(); sw.Stop(); b=Math.Min(b,sw.ElapsedTicks);} return b*1000.0/Stopwatch.Frequency; }
var cod=new ConcurrentOrderedDict<int,int>(N); for(int i=0;i<N;i++) cod.Add(i,i);
Console.WriteLine($"view scan: {Ms(()=>{ long s=0; foreach(var v in cod.Snapshot()) s+=v; GC.KeepAlive(s); }):F3} ms");
```

---

## Checklist — all done

1. [x] **`AddRange`** — batch-atomic bulk load (closed most of Gap B2 for bulk; 1.03× CD).
2. [x] **`Snapshot()` / `ValuesView`** — `List`-parity index/scan (closed Gap A1).
3. [x] **Per-add `Store` allocation dropped** — monotonic mutable count + release publish (measured, kept).
4. [x] **`RemoveWhere` / `TryRemoveSwapBack` / O(1) tail removal + `_floor`** — Gap B3 amortized/opted out.
5. [x] **Value-inline map** (Gap B1, via the clone ref seam) — key get at CD parity; add allocates like CD.
6. [x] ~~`ReaderWriterLockSlim`~~ — rejected (see Locking).
