# `ConcurrentOrderedDictionary<TKey,TValue>` — performance comparison & gaps

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
> keys; `ConcurrentDictionary` has no order and no index. `ConcurrentOrderedDictionary` does *all* of it at
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
global write lock — or, inside the removals, through node handles (`FindNode`/`NodeHandle.Value`/
`TryRemoveNode`, same partial) resolved before the first mutation, so a throwing key comparer can
never abort a removal half-applied (see "Comparer exception safety" below).

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
`public ref TValue GetValueRefOrNullRef(TKey key)` in `ConcurrentDictionary.RefAccessors.cs` — so the
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
- [x] **Pre-size** via `new ConcurrentOrderedDictionary(capacity)` — with growth gone, measured **faster
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
- [x] **Re-index made exception-safe (2026-09-23)** — the re-index used to run AFTER the key was
      removed from the map, through the comparer, so a comparer throw stranded the removal half-applied.
      It now resolves every shifted node before the first mutation when the lookups can run user code
      (a custom comparer, or a key type with its own `Equals`/`GetHashCode`): an 8-bytes-per-shifted-entry
      handle array plus a second pass over the nodes, measured ~1.7× the one-pass re-index cache-resident
      and ~2.5× at 100K entries. For framework key types under the default comparer the lookups cannot
      throw or write back, so they keep the one-pass re-index (now 5–7 % faster than before — the removed
      key is unlinked by node identity instead of a second hash walk). Details and numbers:
      "Comparer exception safety" below.

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

## Thread-safety contract (audited 2026-09-14)

The honest answer to "are we absolutely thread-safe": **yes, for the stated contract** — every
guarantee below is either structurally enforced, exception-enforced, or a *named* relaxation. There
is no undefined behavior left for well-formed use; the one formerly-UB corner (user callbacks writing
back under the lock) is now detected and refused.

**Guaranteed, always (any thread mix, any interleaving):**

* **No torn reads, ever.** Keys and values are only ever written where a single aligned ≤word store
  is atomic; wide (`decimal`/`Guid`/large-struct) keys and values are never written in place — node
  swap on the key path, array clone / fresh arrays on the list path (hunted by the `long`-bit-pattern
  and `decimal` torn-read guns).
* **Per-location coherence:** no read path ever observes a single key's value "going back in time"
  (single-writer monotonic-register gun, all three read paths).
* **Publication order (happens-before):** a reader that observes `Count == c` can read every slot
  below `c` fully written, and every entry visible on the list path already resolves on the key path
  — the skew between the two paths is *directional*, key path first (release map-insert →
  release count-publish, acquire on the reader side; pinned by the append-only happens-before gun).
* **Snapshot integrity:** an enumerator/`Snapshot()` sees a frozen (array, count) pair: exact count,
  no duplicates, no phantoms, no structural rewrites of visible slots — across appends, interior
  removals, tail removals + re-appends (the `_floor` rule), `Clear`, and `AddRange`.
* **Pair integrity per snapshot** under every order-preserving operation: `Pairs`/the view never
  combine a key with a value that was not stored under it (swap-back excepted, below).
* **Exactly-once effects:** racing `TryAdd`/`TryRemove`/`TryRemoveSwapBack` on the same key have one
  winner; `TryUpdate` is a true CAS (per-key `final value == success count`, exactly);
  `GetOrAdd` returns one winner to every caller and only ever a value a factory actually produced;
  `AddOrUpdate` loses no increments; `AddRange` batches appear on the list path atomically.
* **Exception-tightness:** every fallible allocation happens before the key map is touched, every
  key-comparer call of a mutation (user code — including this type's own `LockRecursionException`
  refusal of a comparer that writes back) happens before its first mutation, and the batch operations
  publish in `finally` — a throw (source enumerator, predicate, comparer, OOM) lands in a consistent
  "applied everything up to the failure" state, never a key that resolves but does not enumerate.
  *Corrected 2026-09-23:* the comparer clause was FALSE until then — interior removal, both swap-back
  branches and `RemoveWhere` re-indexed through the comparer after mutating the map (pinned by
  `OrderedCollectionsComparerExceptionSafetyTests`, fixed with node handles; see below). *Known
  out-of-memory-only gap (open):* the vendored map grows its table AFTER inserting a key
  (`TryAddInternal` → `GrowTable`), and `AddRange` allocates its published `Store` in its `finally`, so
  an out-of-memory raised by either can leave the key just added resolving without enumerating.
  Closing it means growing the map BEFORE the insert (a partial member that pre-grows when the next
  insert would exceed the budget) and allocating `AddRange`'s holder when it goes fresh — both on the
  hot add path, so they want their own measurement.

**Named relaxations (documented on the members, asserted as-relaxed by the tests):**

1. **Cross-path skew** while a write is in flight (key path leads; `IndexOf` may be momentarily
   stale under concurrent removals). Directional, bounded by one operation.
2. **Cross-path value-clock:** an atomic in-place value replace hits the hash node a hair before the
   array slot — each path is monotonic on its own, alternating paths is not (register gun pins both
   facts).
3. **`TryRemoveSwapBack` only:** a concurrent enumerator over the same arrays can see the moved
   entry twice / the removed entry not at all, and a reader combining that ONE slot's key+value
   during the swap window can transiently pair them mixed. Individual reads still never tear; the
   key path is unaffected; the order-preserving removals keep full purity — that split is the whole
   reason both removal families exist.

**Enforced (turned from silent corruption into `LockRecursionException`):** user code that the
collection itself runs *inside* the write lock — an `AddRange` source enumerator, a `RemoveWhere`
predicate, or a key comparer — attempting any write back into the collection. `Monitor` is
reentrant, so before the guard this would have interleaved two half-applied mutations; now every
public mutator checks `Monitor.IsEntered(_writeLock)` before locking (the whole mutating surface is
swept by a per-mutator attack test). `GetOrAdd`/`AddOrUpdate` factories run outside the lock and may
call back freely; lock-free reads are legal from anywhere, including inside predicates. Measured
cost of the guard (within-process A/B, 20M iterations): **2.84 ns per mutation** — ~2–3% of a real
add and invisible under host noise on the sweep benchmarks; reads pay nothing (no guard on any read
path). That is the deliberate price of converting silent corruption into a deterministic exception.

---

## Comparer exception safety (fixed 2026-09-23)

**The defect.** Interior removal, both `TryRemoveSwapBack` branches and `RemoveWhere` removed the key from
the map FIRST and then re-indexed the shifted/moved keys through the map — one comparer call per key —
before publishing the new `Store`. A comparer that threw in that window (the documented evil-comparer
defence included: its refused write-back raises `LockRecursionException` from inside `GetHashCode`)
aborted the removal half-applied, permanently: the removed key enumerated without resolving, a moved key
enumerated twice (in-place swap-back), and `RemoveWhere`'s `finally` re-ran the comparer and skipped the
publish altogether. Pinned by the four `ConcurrentOrderedDictionary_*` tests in
`OrderedCollectionsComparerExceptionSafetyTests` (committed `[OpenBugs]` in fe756f9e; now CI).

**The fix.** Every removal resolves the key-map node of every entry it will unlink or re-index BEFORE its
first mutation, then mutates through those node handles (`NodeHandle.Value` for the index write,
`TryRemoveNode` to unlink by identity) — no comparer call can follow the first mutation, so a comparer
throw aborts the operation with nothing changed. `RemoveWhere` additionally pre-allocates the `Store` it
publishes from its `finally`, and `ReplaceExistingUnderLock` (non-atomic `TValue`) now allocates its
value-array clone and `Store` BEFORE it swaps the node (an out-of-memory in the clone used to leave the
key path ahead of the list path).

**Two paths, by who can run inside a lookup.** Resolving everything first costs a transient handle array
(8 B per shifted entry) and a second pass over the nodes — measured per shifted key (10K-entry map,
one P-core): one-pass lookup+write 1.18–1.25 ns; resolve-first 2.0 ns cache-resident, 3.1 ns at 100K
(the second pass is memory-bound). That hazard only exists when a lookup can run USER code, so
`_lookupsRunNoUserCode` (a framework key type — primitive, enum, `string`, `decimal` — under the default
comparer) keeps the one-pass re-index for the common case; every custom comparer and every key type
with its own `Equals`/`GetHashCode` takes the resolve-first path. Measured (A/B of the pre-fix and fixed
sources embedded in one process, pinned P-core, 150 ms warm-up, best-of, median over repeated processes):

| `<int,int>` operation | pre-fix | fixed | fixed / pre-fix |
|---|---:|---:|---:|
| interior `TryRemove` at position 0 + re-add, N=1K — default comparer (one-pass) | 1.70 µs | 1.48 µs | **0.87** |
| — N=10K | 15.8 µs | 14.0 µs | **0.89** |
| — N=100K | 223 µs | 218 µs | **0.98** |
| same, custom comparer (resolve-first), N=1K | 3.33 µs | 5.10 µs | **1.53** |
| — N=10K | 32.0 µs | 51.2 µs | **1.62** |
| — N=100K | 339 µs | 807 µs | **2.37** |
| `RemoveWhere(even)`, N=100K — default comparer | 908 µs | 871 µs | **0.95** |
| — custom comparer | 1,006 µs | 1,345 µs | **1.33** |
| `TryRemoveSwapBack` drain, N=100K — default comparer | 32.7 ns/op | 31.0 ns/op | **0.95** |
| — custom comparer | 34.9 ns/op | 33.3 ns/op | **0.97** |
| `TryRemoveSwapBack` drain `<int,decimal>` (copying branch), N=2K | 757 ns/op | 762 ns/op | **1.00** |
| tail-pop `TryRemove` drain, N=100K | 31.2 ns/op | 30.5 ns/op | **0.99** |
| `SetByKey` existing `<int,decimal>` (B11's allocate-first reorder), N=1K | 452 ns | 450 ns | **0.99** |
| *A/A harness noise (pre-fix vs pre-fix, the interior rows)* | | | *0.99–1.01* |

The swap-back and tail paths got slightly faster on both comparer classes: the removed key is unlinked by
node identity (`TryRemoveNode`) instead of a second hash walk, and swap-back needs one lookup fewer.

One placement trap found on the way (recorded on `ReindexOnePassUnderLock`): the trusted one-pass loop,
byte-identical in the JIT's tier-1 output to the pre-fix loop, ran ~0.4 ns/key slower while inlined in
the now-larger `RemoveCoreUnderLock`; as a method of its own it runs at (or slightly above) the pre-fix
speed. Check the disassembly (`DOTNET_JitDisasm`) before believing an algorithmic explanation for a
per-key gap between two identical loops.

---

## Memory analysis (measured 2026-09-14; x64, Release, N=1M footprint / thread-local alloc counter)

### Steady-state footprint (bytes/entry, presized, full-GC deltas)

| Shape | `List<T>` (values only) | `ConcurrentDictionary` | `ConcurrentOrderedDictionary` | COD premium vs CD |
|---|---:|---:|---:|---:|
| `<int,int>` | 4.0 | 49.3 | **57.3** | +16% |
| `<int,long>` | 8 | 49.3 | **69.3** | +41% |
| `<int,decimal>` | 16 | 57.3 | **85.3** | +49% |
| `<string,string>` (structure only, strings shared) | 8 | 57.0 | **81.0** | +42% |

Decomposition for `<int,int>`: hash node 40 B (identical to CD's — the inline `ValueIndex` rides node
padding when `TValue` ≤ 4 B) + buckets ~9 B + the two contiguous arrays 8 B. **The arrays ARE the
List-parity read path and the +8 B is their entire cost**; for 8-byte-plus values the node grows a
further 8 B (`ValueIndex` alignment), which is where the bigger premiums come from. Unsized
`<int,int>` measured 59.6 B/entry (N=1M sits near a power of two; worst-case doubling slack is the
usual up-to-2× on the array portion, like `List<T>`). The old `Entry`-object design would have added
a fourth component (~24 B/entry + a second pointer chase) — folding it into the node removed it.

### Allocation churn per operation (thread-local counter, min-of-rounds)

| Operation | `List<T>` | `ConcurrentDictionary` | `ConcurrentOrderedDictionary` |
|---|---:|---:|---:|
| read / `IndexOf` / `this[int]` / `Count` | 0 | 0 | **0** |
| full enumeration | 0 | **56 B** (iterator class) | **0** (struct); `Pairs` = one iterator object |
| `Snapshot()` + span scan | — | — | **0** |
| replace existing, atomic `TValue` | 4 (List set: 0) | 0 | **0** (ref-seam in-place) |
| replace existing, `decimal` (non-atomic) | 0 | 48 B (node swap) | **16 B × capacity** (values-array clone; 1.6 MB at 100K) |
| — same, batched via `AddRange` (1000 dups) | — | — | **~4 KB/op amortized** (ONE copy-on-write per batch — ~400× less) |
| add, fully presized (amortized) | 4 | 48.7 | **56.7** (node 40 + the ctor's buckets/arrays amortized; steady-state in-capacity add = node only, pinned ≤64 B by test) |
| add, unsized build (amortized, incl. growth churn) | ~8 | 121.4 (GrowTable RE-CREATES every node) | 143.3 (that same node churn + array doubling) |
| `ToArray` / `Keys` | N×elem | N×elem (+ all stripe locks) | N×elem |
| remove: tail pop | — | 40 B | **40 B** (one Store holder) |
| remove: swap-back (in-place) | — | — | **40 B** |
| remove: interior, order-preserving | **0** (in-place shift) | ~0 | **2 × capacity-sized arrays** (800 KB at 100K int); **+8 B × shifted entries** (a transient node-handle array) when the key comparer can run user code — never for framework key types under the default comparer |
| `RemoveWhere` (any match count) | — | — | **the same 2 arrays once** (800,152 B measured at 100K — k interior removes cost k×, one bulk pass costs 1×); **+8 B × entries from the first match on** when the comparer can run user code |

The interior-removal column is the concurrency price stated in memory terms: `List<T>` shifts in
place because it has no lock-free readers to protect; COD must give readers immutable snapshots, so
an order-preserving interior removal buys that safety with two fresh arrays. **LOH note:** the
arrays cross the large-object threshold at ~21K `int` entries (85 KB), so interior-removal churn at
large N is gen2/LOH churn — use `RemoveWhere` (one pass for any count), `TryRemoveSwapBack` (40 B),
or tail pops; and batch wide-value (`decimal`-class) replacements through `AddRange`, or prefer
atomic-width/reference values for update-heavy workloads.

### Retention semantics (verified with WeakReference probes; pinned by `ConcurrentOrderedDictionaryMemoryTests`)

* A **tail-popped** value stays reachable through the shared arrays until the next append's
  copy-on-write (the floor rule), a compaction, or `Clear` — the slot is deliberately not scrubbed
  because an enumerator captured at the old count must still read valid data. Released exactly at
  the first append after the pop (verified).
* An **interior-removed** value is released immediately (fresh arrays exclude it; the old snapshot
  becomes garbage once unreferenced). `Clear` releases everything.
* A **held enumerator or `ValuesView` pins BOTH captured arrays** (2 × capacity × element size) for
  its lifetime — bound snapshot lifetimes at large N. The framework dictionary's enumerator, by
  contrast, pins only the node chain it walks; that is the flip side of COD's contiguous snapshots.
* Probe-writing trap (cost a false "leak" during this analysis): under tier-0/Debug codegen,
  untracked caller-frame stack temps keep a dropped enumerator reachable — retention probes must
  hold AND drop the reference inside one `NoInlining` helper frame.

### Where the comparative measurements live

`benchmark`'s BDN suite reports the `Allocated` column for every composite scenario
(MemoryDiagnoser is on in the shared config); the per-entry/footprint script used for the tables
above is reproducible via the recipe in "How to measure". The pairs constructor now pre-sizes the
hash map as well as the arrays from a countable source (`TryGetNonEnumeratedCount`), because map
growth re-creates every node it holds — fully pre-sized builds allocate exactly the live bytes.

---

## Vendored `ConcurrentDictionary` clone — deviations from the BCL

The clone in `ConcurrentDictionary.cs` is faithful to dotnet/runtime's source; the concurrency core
(striped locks, lock-free reads, `GrowTable`) is verbatim. Deviations:

- **`partial`** (one keyword on the class declaration) so NumSharp-only members live in
  **`ConcurrentDictionary.RefAccessors.cs`** without touching the vendored body. That file holds
  `public ref TValue GetValueRefOrNullRef(TKey key)` — `TryGetValue`'s bucket walk returning
  `ref node._value` (null-ref sentinel when absent) — and, since 2026-09-23, the INTERNAL node-handle trio:
  `NodeHandle` (a typed wrapper around one live node; `Value` is a ref to its value field),
  `FindNode(key)` (the same walk, returning the node) and `TryRemoveNode(handle)` (unlink by identity:
  the bucket comes from the node's stored hash code, the chain is compared by reference — no comparer
  call, `TryRemoveInternal`'s bookkeeping). The seam's three caller obligations (lifetime inside one
  serialized critical section; single-field ≤word-size writes only; caller-side write serialization)
  are documented on the member and apply to handles too. A future re-sync against upstream is a plain
  file replace + re-adding the `partial` keyword, then re-checking the internals the partial reads
  (`Node`, `Tables`, `_tables`, `GetBucket`, `GetBucketAndLock`, `GetHashCode(comparer, key)`).
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

- **BDN suite** (fair, grouped, baselined): `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Collections/ConcurrentOrderedDictionaryBenchmarks.cs`
  — Build / KeyGet / Enumerate / ToArray / IndexGet / Replace / Remove groups at N = 1K / 100K / 1M,
  each with its `List<T>` or framework-CD baseline **plus the vendored clone rows** so any gap is
  attributable (ordered-dict layer vs clone-vs-BCL drift). Run:
  `dotnet run -c Release -- --filter "*ConcurrentOrderedDictionary*"` (menu option 15).
- **Comparer exception safety:** `test/NumSharp.Tests/Collections/OrderedCollectionsComparerExceptionSafetyTests.cs`
  — deterministic, single-threaded: a comparer that attempts a write-back (refused with
  `LockRecursionException`) while hashing one chosen key, fired inside interior removal, both swap-back
  branches and `RemoveWhere`; then a full audit that the key and list paths still agree. The same
  attacks run against `ConcurrentOrderedCompactDictionary` and `OrderedDictionary` as controls.
- **Functional pins:** `test/NumSharp.Tests/Collections/ConcurrentOrderedDictionaryTests.cs` — ordering,
  dup semantics, the int-key indexer footgun, tail-pop + append-floor snapshot purity, swap-back,
  RemoveWhere (incl. throwing-predicate consistency), AddRange (incl. mid-batch-throw consistency),
  views, non-atomic `decimal` paths, and a 5K-op randomized churn against a `List`-based oracle.
- **Concurrency gun:** `test/NumSharp.Tests/Collections/ConcurrentOrderedDictionaryConcurrencyTests.cs` —
  every scenario arms N threads that park on ONE `ManualResetEventSlim` and fire simultaneously:
  exactly-once distinct adds, single-winner same-key adds/removes, `GetOrAdd` single-value,
  `AddOrUpdate` lost-increment counter, torn-read hunts (atomic `long` bit patterns AND non-atomic
  `decimal`), exact-prefix enumeration under appenders, the floor rule under pop/re-append fire,
  `AddRange` batch-atomicity (observed counts are batch multiples), swap-back survivor sets, index
  readers under structural churn, `Clear` vs everything, and a mixed-chaos soak with a value-law
  readers verify on every observation.
- **Advanced/adversarial tier:** `test/NumSharp.Tests/Collections/ConcurrentOrderedDictionaryAdvancedConcurrencyTests.cs`
  — each scenario targets one clause of the thread-safety contract above: `Barrier`-phased storms
  with rotating roles and a FULL quiescent audit at every post-phase point (oversubscribed threads
  so the scheduler preempts inside locks), the single-writer monotonic register (per-path
  no-time-travel on all three read surfaces), the append-only happens-before probe (list visibility
  ⇒ key visibility; `Count` ⇒ readable correct slots, boundary slot probed every pass), exact
  `TryUpdate` CAS accounting (final value == wins, per key), `GetOrAdd` stampede provenance (stored
  value must decode to a factory that really ran), pair-integrity under ordered churn (swap-back-free
  by design — the contract split), enumerator no-duplicate/no-phantom under interior churn,
  two-instances-of-one-closed-type isolation (the shared `Store.Empty` static under `Clear` storms),
  the full per-mutator reentrancy attack sweep (all 15 mutators refused with
  `LockRecursionException` from inside a predicate), the evil-comparer write-back attack, and a
  partitioned all-ops chaos (AddRange + racing swap-backs with single-winner accounting + interior
  churn + tail stack cycles + `RemoveWhere` cycles + full-surface readers) with an exactly computable
  final oracle.

## How to measure (reproduce the table)

The `dotnet run` recipe below still works for quick spot checks (`dotnet run -c Release -` — Release
is mandatory, Debug taints NumSharp.Core ~2×); prefer the BDN suite for anything you intend to quote.

```csharp
#:project <path>/src/NumSharp.Core/NumSharp.Core.csproj
using System.Diagnostics; using NumSharp.Collections;
const int N = 1_000_000, Reps = 11;
double Ms(Action a){ a(); a(); long b=long.MaxValue; for(int r=0;r<Reps;r++){ var sw=Stopwatch.StartNew(); a(); sw.Stop(); b=Math.Min(b,sw.ElapsedTicks);} return b*1000.0/Stopwatch.Frequency; }
var cod=new ConcurrentOrderedDictionary<int,int>(N); for(int i=0;i<N;i++) cod.Add(i,i);
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

---

## The compact sibling — `ConcurrentOrderedCompactDictionary` (discovery 2026-09-14, SHIPPED the same day)

`ConcurrentOrderedDictionary.COMPACT.md` (beside this file) is the discovery ledger; the numbers reproduce with
`benchmark/collections/probes/compact_ordered_dict_probe.cs`. In one paragraph: the Store CAN be absorbed
into a compact insertion-ordered hash table (slot == index, one copy of each key/value, no per-entry node)
**without adding copy-on-write to any operation that is not COW today** — append/tail-pop/atomic-replace/
swap-back stay in place, interior removal/non-atomic replace/growth stay COW. What goes away is the 40 B
hash node and the vendored clone: `<int,int>` 57.3 → 16–21 B/entry (below the plain CD's 49.3), builds
3.4–4.3× faster, drains ~3× faster, front interior removals 5× faster (a streaming renumber replaces the
per-key hash-walk re-index). Costs: near-tail interior removal ~3× slower (the index is copied too) and the
layout decides the DRAM-scale key-hit cost. **Recommended layout: open addressing** (CPython's `dk_indices`
shape with the key embedded in an 8-byte index word) — at parity/faster on every read path at every size,
and the only layout whose in-place `TryRemoveSwapBack` keeps the key path tear-free (a chained compact
layout lets a reader of the removed key pair its old key with the moved value; the open-addressed reader
re-validates the single index word after the value load, ABA-free because dummied positions are never
reused within a generation). Implementation plan and the port traps are in §7–§8 of that document.

**Shipped as `ConcurrentOrderedCompactDictionary<TKey,TValue>`** (`ConcurrentOrderedCompactDictionary.cs`, same public
surface and contract as this type): open-addressed index of 8-byte words `(tag<<32)|(slot+1)` (the tag is the
key's own bits for ≤4-byte primitive/enum keys under the default comparer, else the comparer's hash), dense
`keys[]`/`values[]`, generation holders with the same count/floor rules plus a dummy count, the validated read
(acquire value load + word re-read), a full-fence dummy store before the in-place swap-back overwrite,
write-atomic guards (wide keys/values take the copy path), and whole-generation copies for wide-value replaces
(no cross-generation aliasing). Gates: the four suites of this type mirrored verbatim onto it +
`ConcurrentOrderedCompactDictionarySpecificTests` (bit-tag types, float/custom-comparer exclusions, tag collisions,
constant-hash worst case, dummy/rebuild churn, wide types, the swap-back gun) — 85 tests green on net8.0 and
net10.0, Debug and Release; the probe's `-- gun` with the `cocd` mode: 0 failures over 1.9 G reads. BDN rows
`COCD.*` sit beside the `COD.*` rows in every group.
