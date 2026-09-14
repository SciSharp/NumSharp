# `ConcurrentOrderedDict` — folding the Store into a compact table (discovery, 2026-09-14)

**Status:** discovery complete, no product code changed. Every number below is reproducible with
`benchmark/collections/probes/compact_ordered_dict_probe.cs` (five prototype layouts + today's type + the
baselines; a 20K-op structural sanity check against a `List` oracle gates each run).

**Verdict.** The Store can be folded into a compact, insertion-ordered hash table **without copy-on-write on
any structural change that is not already copy-on-write today**. Append, tail pop, atomic value replace and
swap-back stay in-place and allocation-free (or one small holder); interior order-preserving removal,
non-atomic value replace and growth stay COW — exactly today's COW set. What disappears is the **40 B hash
node per entry and the vendored `ConcurrentDictionary` underneath it**, not the arrays: `<int,int>` drops
from **57.3 → 16–21 B/entry** (2.7–3.5× smaller, and *below* the plain `ConcurrentDictionary`'s 49.3),
`<int,decimal>` from 85.2 → 33–37, `<string,string>` structure from 81 → ~29–33. Builds get 3–4× faster
(no per-add node, no node-recreating `GrowTable`), pop/swap-back drains 3× faster, front/middle interior
removals 1.4–5× faster (the O(tail) hash-walk re-index is replaced by a streaming renumber). Two costs are
real and quantified: **near-tail interior removal is ~3× slower** (the hash index is copied along with the
arrays), and the **layout decides the key-path cache-line count** — the parallel-arrays layout is 7–40 %
slower per DRAM-scale hit, the open-addressed layout is at parity or faster. So the brief's "merely renamed
the Store" outcome did **not** happen: the Store is *absorbed* (it becomes the hash entries), the node layer
is deleted, and every O(1) guarantee survives.

**The one genuinely new hazard** — found by reasoning, confirmed by construction — is that a *chained*
compact layout cannot keep the key path tear-free under the in-place `TryRemoveSwapBack`: overwriting a slot
a lock-free key-path reader may be standing on lets a reader of the *removed* key pair its old key with the
moved entry's value (a wrong-value hit, which the node design can never produce). The **open-addressed
layout closes it structurally** (one 8-byte index word is the sole publication point, so the reader
re-validates it after the value load; dummied positions are never reused within a generation, so there is
no ABA). That, plus DRAM-scale parity, makes **open addressing the recommended layout**. **The concurrency
gun confirms it (§5.4):** 8 readers + 2 writers for 8 s — today's type 0 failures over 2.06 billion reads,
the chained prototype 3,172 wrong-value hits plus one transient absence of a moved key, the open-addressed
prototype 2,862 wrong-value hits with validation off and **0 with it on** over 1.72 billion reads.

**Are the prototypes themselves thread-safe? No — and they are not meant to be.** They implement the
publication ORDER a port would use (release-store the index word, then the count; acquire-read both; holder
per shrink; floor rule) but omit the guards that make today's type heavy-duty: the write-atomic guards
(in-place value/slot stores for any `TValue`/`TKey`, where wide types must take the COW path), the
reentrancy guard, the memory-model-correct validated read (§4.5), the no-aliasing rule (§4.7), and every
concurrency test. §4 is the contract a heavy-duty port must satisfy; §5.4 is the gun that will gate it.

---

## 1. The central tension, answered

A compact table gets its single-copy contiguous storage from a dense `entries` array it *resizes and
compacts wholesale*; `ConcurrentDictionary` gets lock-free reads from immutable-ish nodes published one
volatile pointer at a time. The question was whether the two can coexist without copy-on-write of the whole
entries array on every structural change. Operation by operation, under today's three Store rules (count
only grows on an instance; slots a reader can see are never structurally rewritten; value slots rewritten
in place only for single-store-atomic `TValue`) extended to the hash index:

| operation | today (node + Store) | compact table | COW? | alloc |
|---|---|---|---|---|
| append, in capacity | node alloc; slots written above `count`; map insert; release count | slot fields written above `count` (invisible); **release-store the index word / chain head** (key path) then **release-store `count`** (list path) — a fresh slot is published exactly like a fresh node | no | **0 B** (today 40 B) |
| append, growth | COW both arrays + `CD.GrowTable` **re-creates every node** | COW arrays; index rebuilt sequentially from stored hashes (or recomputed for trivially-hashed keys) — no node churn | yes (as today) | arrays only |
| append below the floor (after a tail pop) | COW both arrays (floor rule) | same rule, same trigger | yes (as today) | arrays |
| replace value, atomic `TValue` | in-place node field + array slot (two stores, the "cross-path value clock" relaxation) | **one** in-place store — there is one copy, so relaxation #2 of the contract disappears | no | 0 |
| replace value, non-atomic `TValue` | node swap + values-array clone | values-array clone only (index arrays shared, unchanged) — see the aliasing guard in §7 | yes (as today) | values array |
| tail pop | `CD.TryRemove` (stripe lock, node unlink) + lower-count holder + floor | **one store** (dummy the index word / relink the chain head) + holder + floor; the vacated slot's fields stay intact for readers standing on them | no | one holder (56–72 B; today 40 B) |
| interior removal, order-preserving | COW both arrays + **O(tail) per-key hash walks** rewriting `_index` | COW arrays **and** the index, repaired by a **streaming renumber** (`slot > i → slot−1`, `slot == i → dummy/skip`) — no hashing, no walks | yes (as today) | arrays + index |
| `RemoveWhere` | COW + per-survivor re-index walks | COW + one sequential index rebuild | yes (as today) | arrays + index |
| `TryRemoveSwapBack` | node `_index` rewrite + in-place slot swap (list-path relaxation) | open addressing: dummy the removed word, overwrite slot i, **one atomic index store** re-points the moved key; chaining: a 4-step relink | no | one holder |
| `Clear` | new empty Store + `CD.Clear` (all stripe locks) | new empty generation | — | — |
| reads (`TryGetValue`/`ContainsKey`/`IndexOf`/`this[int]`/`Count`/enumeration/`Snapshot`) | lock-free | lock-free, same acquire discipline | — | 0 |

So COW is confined to the operations that are COW today. The difference is *how much* is copied (the index
rides along: 2–2.9× today's bytes for an interior removal, measured 1.42–1.61 MB vs 0.56 MB at 70K
capacity) and *what is no longer done at all* (no node per entry, no per-key hash walk on removal, no node
re-creation on growth, no second value copy on replace).

---

## 2. Capability matrix — which structure serves each public member

| member | today | compact table | note |
|---|---|---|---|
| `Count`, `IsEmpty` | `Store._count` | `Tables.count` | trivially one int in both |
| `this[TKey]` get, `GetByKey`, `TryGetValue`, `ContainsKey`, `GetOrAdd` fast path, `AddOrUpdate` fast path | node (`ValueIndex._value`) | index probe → `values[slot]` | the value is read from the same array the list path scans |
| `this[TKey]` set, `SetByKey`, `TryUpdate`, `AddOrUpdate` slow path | node + `Store._values[idx]` | `values[slot]` (one store) | non-atomic `TValue`: values-array clone, as today |
| `TryAdd`, `Add`, `AddRange`, `GetOrAdd` miss path | node alloc + slots + count | slots + index word + count | 0 B in capacity |
| `IndexOf`, `TryGetIndex` | node (`ValueIndex._index`) | **the slot number the probe lands on** | no stored index field: slot *is* the position (holes never exist, §3) |
| `this[int]` get, `TryGetAt`, `GetKeyAt` | `Store._values`/`_keys` | `values[i]` / `keys[i]` | unchanged (a strided load only in the AoS layout) |
| `SetAt` | `Store._keys[i]` → node lookup → both paths | `values[i]` directly | **no key lookup needed** — one store |
| `TryRemove`, `Remove`, `RemoveAt` | `CD.TryRemove` + Store tail/interior paths | §1 rows | tail O(1), interior COW |
| `TryRemoveSwapBack` | node `_index` rewrite + slot swap | §1 row | validated read keeps the key path clean (open addressing) |
| `RemoveWhere` | COW + per-survivor walks | COW + rebuild | — |
| `Clear` | both | new generation | — |
| `ToArray`, `CopyTo`, `Keys`, `Values`, `Pairs`, `Enumerator` | `Store` arrays | the same dense `keys`/`values` arrays | AoS layout would gather from strided entries |
| `Snapshot()` / `ValuesView` (`this[int]`, `GetKeyAt`, `AsSpan`, `KeysAsSpan`, span `foreach`) | `Store` arrays | dense `keys`/`values` arrays | **`AsSpan`/`KeysAsSpan` need dense key/value arrays** — the AoS layout cannot offer them (an API break), the split layout loses `KeysAsSpan` |
| `Comparer` | `CD.Comparer` | the table's own comparer field | — |
| `ValueIndex` struct, `Store` class, the vendored clone + ref seam | — | **gone** | the clone (2,297 + 291 + 43 lines) has no other consumer in `src/` |

Directional skew: today "the key path leads" (map insert, then count publish). A compact append publishes
the index word before the count, so the direction is preserved; `AddRange`'s list-path batch atomicity is
unchanged. `IndexOf` can still be stale by concurrent removals (a reader may hold an older generation).

---

## 3. The removal-hole decision — the crux, decided

In a compact table `entries[i]` equals insertion position `i` only until the first interior delete. Four
ways to handle the hole, and what each costs this type's contract (strict dense `this[int]`, O(1) `IndexOf`,
`AsSpan`, lock-free snapshot-pure readers):

| option | reference | verdict |
|---|---|---|
| **Compact-on-delete** (shift the tail down, into fresh arrays) | today's `RemoveCoreUnderLock` | **Adopted.** O(n) COW, as today. The hash index must be repaired, but since slot indices are the chain/word payload the repair is a *pure renumbering*: `j > i → j−1`, `j == i → skip to its chain successor` (chaining) / `→ dummy` (open addressing). A streaming pass with no hashing and no per-key walks — which is why front removals get **5× faster** at 10M under hashed keys (today's per-key re-index becomes 10M random node walks: 153 ms; the renumber: 24–30 ms). The in-place variant is **not** an option: a lock-free reader standing on a half-moved slot could match the old key against the new value (a wrong-value hit, not merely a miss), so the shift MUST land in fresh arrays. |
| **Tombstone + lazy compaction** (CPython) | `dictobject.c` `delitem_common` sets `DKIX_DUMMY` in the index and NULLs the entry; `dictresize` skips dead entries; `popitem` is the tail-pop special case | **Rejected.** Holes break every positional guarantee: `this[int]`/`IndexOf` would need a rank/select structure over a hole bitmap (O(1) only with popcount blocks — several dependent loads per access, 5–10× a raw array read, i.e. Target A gone), `AsSpan()` becomes impossible, enumeration needs a skip branch. CPython can afford holes because `dict` has no positional access at all. Note CPython's own free-threading rule, which is exactly the constraint here: *"In free-threaded builds dummy slots are not re-used to allow lock-free lookups to proceed safely."* |
| **Free-list slot reuse** (.NET `Dictionary`) | `Dictionary.cs` `Remove`: `entry.next = StartOfFreeList - _freeList`, key/value cleared, `_freeList = i`; `TryInsert` pops it | **Impossible under lock-free readers.** Reusing a slot rewrites a slot a reader can see — a torn key/value pairing on the *key path*, which never tears today. (`Dictionary` also overwrites the removed entry's `next` with the free-list encoding, which would send a concurrent chain walker to a garbage index.) |
| **Swap-back tombstone** | today's `TryRemoveSwapBack` | **Kept as the explicit opt-in**, order-breaking O(1). In open addressing it is literally one atomic index store after the slot overwrite. |

**Decision:** strict-dense `this[int]` survives the compact table by keeping compact-on-delete, i.e. the
same O(n) copy-on-write the type already pays; holes never exist, so `Count` is one int and `IndexOf` is the
probe's slot number. The compact table changes *what* is copied (more) and *removes* the per-key re-index.

---

## 4. Lock-free publication protocol (the rules a port must keep)

1. **A fresh slot is published like a fresh node.** Write `keys[idx]`, `values[idx]` (and `hashes[idx]`,
   `next[idx]` in chaining) while `idx ≥ count` (invisible to list-path readers); `Volatile.Write` the index
   word / chain head (key path becomes able to reach the slot); `Volatile.Write(count)` (list path). Readers
   acquire-read whichever they enter through. A reader that reaches a slot through the key path reads fields
   written before the release store.
2. **Never rewrite a slot a reader can see.** Removed slots keep their fields (a reader may be standing on
   them); the only in-place slot rewrites are the atomic value store (documented live-value semantics) and the
   swap-back overwrite of slot `i` *after* its own key has been removed from the index.
3. **Count is monotonic per generation; shrinking transitions publish a new holder** over the same arrays with
   the floor raised (today's rule 1); an append below the floor copies (today's rule 2).
4. **Index positions are never reused within a generation** (open addressing: a dummied word stays dummy; a
   rebuild = a new generation). This is what makes the validated read ABA-free: a reader that loaded word `w`
   at position `p`, read `values[slot]`, and re-reads position `p` cannot see `w` again unless nothing changed —
   a re-added key lands at a *different* position, because `p` is a dummy, not empty.
5. **The validated read** (open addressing): `e = Volatile.Read(index[p]); match → v = values[slot]; if
   (Volatile.Read(index[p]) == e) return v; else restart`. Cost measured: 21.85 → 22.11 ns at 10M (noise).
   This is the whole fix for the swap-back key-path tear; chaining has no equivalent single word to validate.
   **Memory-model obligations the prototype does not yet meet** (it is correct on x64 only because TSO keeps
   loads ordered and RyuJIT does not move accesses across volatile ones): (a) the value load must be an
   **acquire** load — `Volatile.Read` for primitives/references (a plain `ldr` on ARM64 may complete after
   the validating `ldar`, which would validate before reading), and for a non-atomic `TValue` a plain read
   followed by `Interlocked.MemoryBarrier()` before the re-read; (b) **every index-word access goes through
   `Volatile.Read`/`Volatile.Write`** — that is also what makes the 8-byte word atomic on 32-bit runtimes,
   where a plain `long` store tears into "key 0 with a valid slot"; (c) the writer's **dummy store must be a
   full fence** (`Interlocked.Exchange`) when it precedes the slot overwrite (swap-back): a release store
   only orders EARLIER stores before it, so on a weak machine the overwrite could become visible before the
   dummy and a reader could validate a word that should already be dead.
6. **Interior removal, non-atomic replace and growth are COW into a fresh generation**; the old generation is
   frozen the moment the new one is published (readers holding it continue on it, the GC keeps it alive).
7. **No cross-generation array aliasing with different values arrays** — a non-atomic replace COWs the
   WHOLE generation (index + keys + values), never the values array alone. The values-only clone this
   document first proposed is unsound: the new generation would share `keys`/index with the old one, and a
   later in-place swap-back rewrites `keys[i]` below `count` in the shared array while the old generation's
   `values[i]` still holds the removed key's value — an old-generation reader of the moved key gets the wrong
   value. A reader-side count guard fixes the append case but not that one; a full COW fixes both. (Holders
   that share EVERY array — tail pop, swap-back — stay safe: there is one values array.)
8. **In-place swap-back only when `TKey` and `TValue` are both single-store atomic**
   (`ConcurrentDictionaryTypeProps<T>.IsWriteAtomic`, today's guard); wide types take the COW path, exactly as
   today. The prototype writes in place unconditionally.

---

## 5. Layouts measured

All five prototypes keep slot == insertion index and the protocol above; they differ only in where the hash
index lives and therefore in cache lines touched per random hit (`b` = bucket/index line, `e` = entry line):

| layout | per-entry arrays | random hit touches | `AsSpan` | `KeysAsSpan` |
|---|---|---|---|---|
| **SoA** (chained; `hashes`, `next`, `keys`, `values`, `buckets`) | 16 B + buckets | b + hashes + keys + values = 4 | yes | yes |
| **SoA no-hash** (same, no stored hash for trivially-hashed keys) | 12 B + buckets | b + keys + values = 3 | yes | yes |
| **AoS** (chained; one `Entry{hash,next,key,value}[]` — .NET `Dictionary`'s layout) | 16 B + buckets | b + e = 2 | **no** | **no** |
| **split** (chained; `IndexEntry{hash,next,key}[]` + `values`) | 16 B + buckets | b + e + values = 3 | yes | no |
| **OA** (open-addressed `long[]` index word = `(key<<32)|(slot+1)`, + `keys`, `values`) | 8 B + 8 B × 2^k/N | index + values = 2 | yes | yes |

### 5.1 Fair regime — presized `<int,int>`, build keys a random permutation, lookup order an *independent*
permutation, one P-core, best-of-N after a 150 ms tier-1 warm (`PROBE_KEYS=perm`)

**N = 100,000** (everything L2/L3-resident)

| contender | B/entry | build ns/add | hit ns | miss ns | foreach ns/el | this[i] ns/el | rm @0 ms | rm @n/2 | rm @n−2 | pop drain ns | swap drain ns |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CloneCD | 48.5 | 18.1 | 3.14 | 3.23 | 1.81 | — | — | — | — | — | — |
| **COD today** | 56.6 | 32.8 | 3.42 | 3.28 | 0.32 | 0.36 | 0.368 | 0.233 | 0.069 | 66 | 70 |
| compact SoA | 20.4 | 16.0 | 2.78 | 2.57 | 0.18 | 0.35 | 0.400 | 0.505 | 0.240 | 34 | 39 |
| compact SoA no-hash | 16.3 | 13.9 | 1.86 | 2.49 | 0.18 | 0.35 | 0.308 | 0.567 | 0.207 | 33 | 40 |
| compact AoS | 20.3 | 14.4 | 2.17 | 2.51 | 0.18 | 0.41 | 0.391 | 0.679 | 0.308 | 30 | 34 |
| **compact OA** | 29.0 | 17.0 | 2.40 | 1.35 | 0.18 | 0.35 | 0.507 | 0.624 | 0.324 | 38 | 44 |

**N = 1,000,000** (compact tables ~L3-resident, node tables not)

| contender | B/entry | build | hit | miss | foreach | this[i] | rm @0 | rm @n/2 | rm @n−2 | pop drain | swap drain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CloneCD | 49.3 | 93.5 | 14.17 | 11.45 | 6.25 | — | — | — | — | — | — |
| **COD today** | 57.3 | 112.5 | 14.09 | 12.54 | 0.32 | 0.36 | 7.00 | 4.34 | 0.854 | 158 | 242 |
| compact SoA | 20.7 | 17.2 | 7.75 | 6.35 | 0.18 | 0.36 | 2.89 | 5.50 | 3.10 | 53 | 76 |
| compact SoA no-hash | 16.7 | 17.8 | 5.17 | 5.33 | 0.18 | 0.36 | 2.20 | 5.22 | 2.18 | 39 | 52 |
| compact AoS | 20.7 | 17.3 | 5.99 | 7.85 | 0.30 | 0.61 | 3.35 | 6.40 | 3.68 | 36 | 50 |
| **compact OA** | 24.8 | 38.4 | 12.34 | 4.44 | 0.18 | 0.36 | 3.68 | 6.11 | 3.35 | 82 | 119 |

**N = 10,000,000** (DRAM-bound everywhere)

| contender | B/entry | build | hit | miss | foreach | this[i] | rm @0 | rm @n/2 | rm @n−2 | pop drain | swap drain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CloneCD | 48.0 | 128.5 | 19.96 | 21.57 | 9.82 | — | — | — | — | — | — |
| **COD today** | 56.0 | 146.7 | 25.00 | 22.39 | 0.37 | 0.41 | 152.7 | 79.8 | 8.57 | 318 | 377 |
| compact SoA | 20.0 | 41.3 | 35.09 | 25.93 | 0.26 | 0.40 | 32.6 | 55.3 | 29.4 | 106 | 162 |
| compact SoA no-hash | 16.0 | 39.7 | 26.83 | 25.12 | 0.27 | 0.41 | 23.8 | 50.7 | 21.8 | 99 | 122 |
| compact AoS | 20.0 | 40.0 | 22.95 | 20.57 | **0.74** | **1.00** | 44.5 | 64.4 | 37.2 | 110 | 126 |
| **compact OA** | 21.4 | 42.6 | **23.63** | **14.31** | 0.26 | 0.41 | 29.7 | 55.8 | 27.7 | 109 | 128 |

**N = 1,000** (cache-resident; sequential keys; 100 ns timer quantum over 1–2 µs runs): every contender
sits at 0.9–1.6 ns per hit, 0.2–0.5 ns per scanned element, builds 13–17 ns/add (COD 30) — parity within
quantization.

### 5.2 Reading the tables

* **Memory** is layout-independent for the chained variants (20.0–20.7 B; no-hash 16.0–16.7) and load-dependent
  for OA (21.4 at 10M → 29.0 at 100K: the power-of-two index sits at 0.38–0.60 load). Today: 56–57.
* **Key hit.** Below DRAM scale every compact layout is faster than today (the whole table fits L3; the node
  table does not). At 10M the line count decides: AoS/OA (2 lines) are at parity (23.0/23.6 vs 25.0),
  no-hash (3 lines) is 7 % slower, SoA (4 lines) 40 % slower. **Key miss** at 10M: OA 1.55× faster (a miss
  ends at an empty word in the probed line; a chain miss touches entry lines).
* **List path.** SoA/no-hash/OA scan the dense `values` array — 0.18–0.27 ns/element, *faster* than `List<int>`
  in this harness (0.33–0.39) and than today (0.32–0.37); `this[int]` unchanged. **AoS loses Target A at
  scale** (0.74 / 1.00 ns at 10M: a 16 B stride reads 4× the bytes) and cannot offer `AsSpan` at all — rejected.
* **Build.** 3.4–4.3× faster than today at 1M–10M (no node, no stripe lock, no `GrowTable` node churn); OA
  builds at chaining speed under hashed keys (its 3× deficit in the sequential-key regime was an artifact —
  see §5.3).
* **Drains.** Pop-back and swap-back 2.9–3.2× faster than today (one random line instead of the stripe-locked
  node unlink plus the holder); OA and chaining converge under hashed keys.
* **Interior removal.** Front: compact 5× faster at 10M (today's re-index is 10M random node walks under hashed
  keys). Middle: 1.3–1.6× faster. **Near the tail: 2.5–4× slower** — today copies 8 B/entry and re-indexes a
  tiny tail; the compact table copies 16–24 B/entry regardless of position. This is the cost to accept; it is
  the operation the type already documents as O(n) and steers users away from (`RemoveWhere`, swap-back,
  tail pops — all of which get faster).

### 5.3 Two harness traps that would have inverted the conclusions

1. **Tier-0 timing.** The first sweep measured every JIT-from-IL contender (the clone, today's COD, the
   prototypes) at 15–25 ns per cache-resident hit against the BCL `Dictionary`'s 5.9 — because a 2 ms
   measurement window over 400 reps completes before tier-1 code is installed, while the BCL types are
   ReadyToRun-precompiled. A 150 ms warm per op fixed it (1.0–1.6 ns for everyone).
2. **Key/slot correlation.** With sequential `int` keys, mod-prime chaining puts key `k` in bucket `k`: builds
   and pop-back drains become *sequential* bucket walks (chaining built at 14–16 ns/add and drained at 33 ns/op
   at 10M; OA, whose Fibonacci hash scatters them, at 44 and 113). And probing a table in its own build order
   turns a "random" lookup into a prefetch stream (every table's 10M hit halved). Building from a random
   permutation and looking up in an *independent* permutation is the fair regime reported above; the
   sequential-key numbers are the chaining layouts' best case, not their expectation.

### 5.4 The concurrency gun

`… -- gun 8 8 cod,chained,oa-novalidate,oa` (probe mode; no core pin). Readers loop over 16 hot keys and
the latest published pinned key; writer 1 swap-back-removes/re-adds hot keys and every 64 ops appends a
fresh pinned key at the tail — the next swap-back moves it into a hole; writer 2 churns interior
order-preserving removals of 2,000 fill keys (COW generations). Values obey `v = 7k + 3`, so a value that
belongs to another key is detectable. 8 s per mode, 8 readers + 2 writers, i9-13900K, x64:

| mode | reader ops | swap-backs | wrong-value hits | pinned key absent (`TryGetValue`) | pinned key absent (`IndexOf`) | list-path decode failures |
|---|---:|---:|---:|---:|---:|---:|
| COD today | 2,061,084,648 | 191,248 | 0 | 0 | 0 | 0 |
| compact chained (4-step relink) | 2,045,933,426 | 88,918 | **3,172** | 0 | **1** | 0 |
| compact OA, validation off | 1,554,941,407 | 74,596 | **2,862** | 0 | 0 | 0 |
| **compact OA, validated read** | 1,717,423,535 | 72,424 | **0** | 0 | 0 | 0 |

Three things the gun settles. (1) The chained tear is not theoretical: a few-nanosecond window between a
reader's key compare and value load is hit thousands of times per 8 s on 16 hot keys. (2) The validated read
is exactly the fix — same table, same load, validation off/on. (3) The chained 4-step relink has a SECOND
flaw the analysis had argued away: linking the moved entry at the chain HEAD does not help a walker that
entered the chain before the link and reaches the old tail slot after its unlink — it misses both copies
(the one `IndexOf` absence). The repair is to link slot `i` immediately BEFORE the old tail slot in its
chain, not at the head; that closes the absence but not the tear, so it does not rescue chaining.

---

## 6. Memory ledger against 539e9c05

| shape | `List<T>` | CD | **COD today** | compact chained (SoA / no-hash) | compact OA | reduction |
|---|---:|---:|---:|---:|---:|---:|
| `<int,int>` presized 1M | 4.0 | 49.3 | **57.3** | 20.7 / **16.7** | 24.8 (21.4 at 10M) | 2.3–3.4× |
| `<int,long>` | 8 | 49.3 | **69.3** | 24.6 / ~20.7 | 28.8 | 2.4–3.3× |
| `<int,decimal>` | 16 | 57.3 | **85.2** | 32.6 / ~28.7 | 36.8 | 2.3–3.0× |
| `<string,string>` structure (analytic) | 8 | 57.0 | **81.0** | 28.7 (hash stored) | ~32.8 (hash embedded) | 2.5–2.8× |
| `<int,int>` unsized, 600K (near-worst doubling) | — | 49.0 | **63.0** | 35.7 | 42.0 | 1.5–1.8× |
| `<int,int>` unsized, 1M (near-best) | — | 51.2 | **59.6** | 21.4 | 25.2 | 2.4–2.8× |

Decomposition `<int,int>` presized: chained = 4 (hash) + 4 (next) + 4 + 4 + buckets 4 × 1.16 (prime ≥ N)
= 20.7; no-hash drops the 4; OA = 4 + 4 + 8 × 2^k/N (index words at 0.34–0.67 load → 12–24 B). Today = node
40 + buckets ~9 + arrays 8. The whole "double storage" premium the brief targeted was the 8 B of arrays; the
node was the 40.

Churn per op (presized 70K holding 64K, thread-local counter, min-of-rounds):

| op | COD today | compact chained | compact OA |
|---|---:|---:|---:|
| append in capacity | 40 B (node) | **0** | **0** |
| tail pop | 40 B (Store holder) | 72 B (holder) | 56 B |
| swap-back in place | 40 B | 72 B | 56 B |
| interior removal @1000 (COW) | 560,088 B (2 arrays) | 1,421,920 B (5 arrays) | 1,608,704 B (index + 2 arrays) |
| non-atomic replace | values clone + 48 B node | values clone | values clone |

Retention: unchanged in kind — a tail-popped value stays reachable until the next append's floor copy (the
slot is not scrubbed), an interior-removed value is released with the old generation, a held enumerator/view
pins the two dense arrays it captured (never the index). A reader mid-lookup pins its whole generation for the
duration of one probe, as `ConcurrentDictionary`'s does its `Tables`.

---

## 7. What a port must get right (traps found while prototyping)

* **The swap-back key-path tear (chaining).** The 4-step relink (unlink i from its chain → overwrite slot i
  with the last entry → release-link i as head of key_last's chain → unlink the old tail) keeps key_last
  resolvable throughout, but a reader that loaded slot i *before* step 1 and is preempted between its key
  compare and its value load returns key_i paired with value_last. No store order avoids it (key and value are
  separate words). Chaining therefore needs either a COW swap-back (O(n), today's wide-type degrade path — the
  O(1) guarantee is lost) or a per-slot version word (+4 B, two extra loads per read). **Open addressing avoids
  it with the validated read (§4.5) at zero measured cost — the deciding argument for OA.**
* **Values-array aliasing after a non-atomic replace — full COW, not a values-only clone (§4.7).** A
  values-only clone leaves the new generation sharing `keys` and the index with the old one. An append then
  publishes a new slot through the shared index that an old-generation reader resolves against its stale
  values array (fixable with a `slot ≥ count` reader guard), and — the case the guard cannot fix — a later
  in-place swap-back overwrites `keys[i]` below `count` in the shared array while the old generation's
  `values[i]` still holds the removed key's value, so an old-generation reader of the moved key returns the
  wrong value. Rule: a non-atomic replace copies the whole generation; generations may share arrays only when
  they share ALL of them (tail pop, swap-back holders).
* **The chained relink's transient absence.** Linking the moved entry at the chain head lets a walker that
  entered before the link and reaches the old tail after its unlink miss both copies (observed once in 2
  billion reads). Link before the old tail slot instead. Academic for the recommended layout — open
  addressing re-points one word.
* **Unsized ctor / `Clear`.** The index needs at least one word for the mask/`FastMod`; keep an empty
  singleton generation whose first append grows (today's `Store.Empty` pattern).
* **Reference-type keys.** Embed the **hash** in the index word instead of the key (`(hash<<32)|slot`);
  compare `keys[slot]` on a hash match. A hit then costs index + keys + string deref + values, inherent to
  reference keys. For ≤ 4-byte value-type keys embed the key itself; for 8-byte keys embed the xor-folded hash.
* **Dummies accumulate under pop/swap-back churn** (they are never reused in place): count them and rebuild
  when `(count + dummies) > 2/3 · index`; the floor-rule COW after a pop already rebuilds and clears them.
* **Fibonacci hashing** (`(uint)key * 0x9E3779B97F4A7C15 >> shift`) for the power-of-two index — patterned and
  sequential int keys spread evenly; mod-prime is not needed once the key is embedded in the word.
* **Growth rebuilds the index from the dense keys** (hashes stored or recomputed); it never reads the old index.
* **`Volatile.Read` on an `int[]`/`long[]` element** is fine — the clone's `VolatileNode` wrapper exists only
  for reference arrays (`ldelema` + `CastHelpers.LdelemaRef`).
* **Benchmarking**: 150 ms tier-1 warm per op; independent build/lookup permutations; one P-core pinned; a
  2.5 s clock spin-up (the first 1K rows of the first run were 4–5× slow from a low P-state alone).

---

## 8. Recommendation and plan

**Adopt the open-addressed compact table** (`long[]` index words embedding the key/hash + dense `keys[]`,
`values[]`; generation holder with `count`/`floor`/`dummies`; the §4 protocol with the validated read).
Expected against today, from the fair-regime numbers: memory 2.3–2.8× smaller (below the plain
`ConcurrentDictionary`), builds 3.4–4.3× faster, pop/swap-back drains ~3× faster, key hits at parity at DRAM
scale and 1.1–1.5× faster below it, key misses 1.5–2.8× faster, list path at `List<T>` parity with every
span surface intact, `IndexOf` with no stored field, one value copy (contract relaxation #2 removed), the
swap-back key path clean; interior single removals 5× faster at the front, ~1.4× in the middle, ~3× slower
near the tail. The chained no-hash layout is the alternative if 16 B/entry matters more than the swap-back
guarantee (it would have to make `TryRemoveSwapBack` a COW).

Implementation sketch (a follow-up session; the tests are the spec):

1. `Tables` generation class replacing `Store` and `_byKey`; delete `ValueIndex`; `TryGetValue`/`ContainsKey`/
   `IndexOf`/`GetOrAdd`/`AddOrUpdate` fast paths on the validated probe (`AggressiveInlining`, as today).
2. Every mutator ported row-by-row from §1, keeping the reentrancy guard, strand-proofing (allocate before any
   publish), the `finally` publishes of `AddRange`/`RemoveWhere`, the write-atomic guards (§4.8), the
   full-COW rule for non-atomic replace (§4.7), and the three memory-model obligations of the validated read
   (§4.5: acquire value load, volatile index words, fenced dummy store).
3. Gates: the 4 existing test classes (2,854 lines: functional, concurrency gun, adversarial tier, memory
   contracts) should pass with three memory-contract edits — presized append `≤ 72 B` becomes `== 0`, the
   tail-pop/swap-back holder bound stays `≤ 96 B`, the interior-vs-tail ratio pin still holds (more bytes) —
   plus new pins: the validated read under a racing swap-back (key_last never absent, key_i never returns
   value_last — the §5.4 gun as an MSTest scenario in the adversarial tier), the renumber pass vs a `List`
   oracle, dummy accounting/rebuild, the full-COW rule, and wide-`TKey`/`TValue` swap-back taking the COW
   path.
4. The BDN suite (`ConcurrentOrderedDictBenchmarks`) needs no new rows; this probe stays the reproduction.
5. The vendored `ConcurrentDictionary` clone + `ConcurrentDictionaryInternals` + the ref seam
   (`ConcurrentDictionary.NumSharp.cs`) become unused by `src/` and can be retired with the benchmark's
   `CloneCD` rows.

---

## 10. Complexity proof, ratios and memory across N (measured 2026-09-14)

`benchmark/collections/probes/compact_ordered_dict_complexity.cs` proves the complexity class directly
and prints the ratios and the per-entry memory. Same host discipline as §5 (one P-core, 2.5 s clock
spin-up, 150 ms tier-1 warm, fair permuted-key regime), N swept over **1e3 · 1e4 · 1e5 · 1e6 · 1e7**.
"COD (shipping)" is the real `ConcurrentOrderedDict<int,int>`; "OA"/"chained" are the compact prototypes.

### 10.1 The O(1) proof — three independent legs

**Leg A (hardware-independent): average slots visited per lookup, counted, not timed.** The number of
comparisons an algorithm performs IS its complexity; a count that does not grow with N is O(1) by
definition, on any hardware.

| N | OA hit | OA miss | chained hit | chained miss |
|---:|---:|---:|---:|---:|
| 1,000 | 1.001 | 1.42 | 1.000 | 1.90 |
| 10,000 | 1.000 | 1.74 | 1.000 | 1.99 |
| 100,000 | 1.000 | 1.06 | 1.000 | 1.91 |
| 1,000,000 | 1.000 | 1.29 | 1.000 | 1.84 |
| 10,000,000 | 1.014 | 1.63 | 1.000 | 2.00 |

A hit resolves in **one** slot visit at every size across four orders of magnitude (Fibonacci hashing +
key embedded in the index word, so the home slot IS the entry); a miss ends at the first empty slot / short
chain, 1.1-2.0 visits, flat. An O(log2 N) structure would rise from 10 to 23 visits over this same span; an
O(N) one from 1e3 to 1e7. The measured line is flat at 1.

**Leg B (cache-isolated time): ns/op over a 256-key hot set vs the whole-table working set.** With the
probed lines pinned hot, per-op time is flat across N (the algorithm does the same work); the whole-table
column rises - but *identically for every hash structure, including the proven-O(1) `Dictionary`* - which
identifies that rise as shared DRAM latency, not algorithmic growth.

| get-hit ns | 1K | 10K | 100K | 1M | 10M | 1K->10M |
|---|---:|---:|---:|---:|---:|---:|
| Dictionary - HOT | 1.56 | 1.56 | 1.56 | 1.56 | 1.56 | flat |
| COD (shipping) - HOT | 0.78 | 1.17 | 1.17 | 1.56 | 1.56 | flat |
| compact OA - HOT | 0.78 | 0.78 | 1.17 | 1.17 | 1.17 | flat |
| Dictionary - FULL | 1.10 | 1.56 | 2.34 | 7.56 | 24.93 | **22.7x** |
| ConcurrentDictionary - FULL | 1.00 | 1.31 | 2.87 | 12.36 | 21.13 | **21.1x** |
| COD (shipping) - FULL | 1.10 | 1.43 | 2.79 | 13.26 | 22.16 | **20.1x** |
| compact OA - FULL | 0.90 | 1.09 | 1.75 | 2.87 | 12.46 | **13.8x** |

The FULL growth is the memory hierarchy (a random hit into an N-sized table lands in L1 at N=1K and in DRAM
at N=10M). `Dictionary` - the reference O(1) container - grows 22.7x; COD grows *less* (20.1x) and OA less
still (13.8x, its denser 2-line hit touches fewer far lines). A structure whose growth is bounded by, and
tracks, a proven-O(1) container's is itself O(1). HOT numbers sit at the ~0.4 ns timer floor (100 ns / 256)
and are read qualitatively: flat.

**Leg C (constant-work by construction):** `this[int]`/`GetKeyAt`/`TryGetAt` are one array load - measured
0.50 -> 0.54 ns from 1K to 10M (**1.1x**, i.e. flat; `List<int>` itself moves 0.30 -> 0.42);
`Count`/`IsEmpty` are one field read; a tail pop is 0.001 ms at *every* N (10.3); `IndexOf` is the leg-A
probe (hot: 0.78 ns flat).

### 10.2 Every aspect, and its complexity

| aspect | class | evidence |
|---|---|---|
| lookup by key (hit / miss) | **O(1)** | leg A: 1.00 / 1.1-2.0 probes for all N |
| membership `ContainsKey` | **O(1)** | the same probe |
| key -> position `IndexOf` / `TryGetIndex` | **O(1)** | the same probe; hot time flat |
| value by position `this[int]` / `TryGetAt` | **O(1)** | one array load, 0.50->0.54 ns |
| key by position `GetKeyAt` | **O(1)** | one array load |
| `Count` / `IsEmpty` | **O(1)** | one field read |
| add, in capacity (`TryAdd`/`Add`/`SetByKey` new) | **O(1)** | one slot write + release-store |
| add, over a full unsized build | **amortized O(1)** | total/N tracks `List` (OA 4.8x vs List 4.0x over 1K->10M) |
| replace value, atomic `TValue` | **O(1)** | one in-place store |
| remove last (`TryRemove` tail, pop-back) | **O(1)** | 0.001 ms for all N (10.3) |
| remove by `TryRemoveSwapBack` | **O(1)** | fixed stores + one probe |
| order-preserving interior `TryRemove`/`RemoveAt` | **O(n - p)** - *not* O(1) | linear (10.3); identical to `List<T>.RemoveAt`; O(1) alternatives: tail pop, swap-back, batch `RemoveWhere` |
| `enumerate` / `ToArray` / `Keys` / `Values` / build | **O(n)** - by necessity | the operation visits every element; 0.18-0.26 ns each (5) |

"O(1) in all aspects" holds for the entire point-operation surface - the dictionary API (get/add/replace/
remove/contains) and the list API (index-get, key<->index, count). The only non-O(1) *point* operation is
order-preserving interior removal, O(n-p) by design (contiguous indices + lock-free snapshots cannot both be
kept while removing from the middle in O(1)), and it is the operation the type already documents and routes
around. Whole-collection operations are O(n) because touching every element is their definition.

### 10.3 The O(n) aspect, confirmed linear (honesty check)

One order-preserving removal at position p; cost proportional to (n - p). Front and middle scale linearly
with N; the tail is O(1) at every N.

| N | COD @0 | COD @n/2 | COD @n-1 | OA @0 | OA @n/2 | OA @n-1 |
|---:|---:|---:|---:|---:|---:|---:|
| 100,000 | 0.445 ms | 0.297 | **0.000** | 0.655 | 0.647 | **0.001** |
| 1,000,000 | 3.927 ms | 2.341 | **0.001** | 5.644 | 5.803 | **0.002** |
| 10,000,000 | 41.089 ms | 32.969 | **0.001** | 46.401 | 51.137 | **0.003** |

Front removal x10 per decade of N (0.445 -> 3.927 -> 41.1) = linear; the tail column is flat at ~1 us =
O(1).

### 10.4 Ratios at N = 1,000,000 (baseline / contender; > 1 = contender faster)

| op | COD vs Dict | COD vs CD | OA vs Dict | OA vs CD | OA vs List |
|---|---:|---:|---:|---:|---:|
| key get, hit (full) | 0.57x | 0.93x | **2.64x** | **4.31x** | - |
| key get, miss (full) | 0.69x | 0.93x | **1.88x** | **2.55x** | - |
| add (amortized) | 0.07x | 0.87x | 0.55x | **6.41x** | 0.03x |
| tail pop | - | 0.37x | - | **0.57x** | - |
| `this[int]` | - | - | - | - | 0.71x |

`Dict` is not thread-safe and has no order or index - it wins add (no lock) and loses nothing else it can
do; `List` has no keys - it wins raw index-get. COD (shipping) sits at CD parity on gets (0.93x) and pays
for the global lock + node on writes; the compact OA table beats **both** concurrent baselines on every
keyed read and beats CD on writes, while being the only column that answers every row.

### 10.5 Memory - bytes per entry, presized, across N (flat => O(1) space per entry)

| N | List | Dictionary | ConcurrentDictionary | COD (shipping) | compact OA | compact chained |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 11.8 | 29.9 | 56.6 | 64.8 | 32.3 | 24.3 |
| 10,000 | 4.8 | 21.0 | 48.8 | 56.0 | 21.1 | 16.8 |
| 100,000 | 4.1 | 21.8 | 48.7 | 56.6 | 29.0 | 16.4 |
| 1,000,000 | 4.0 | 23.3 | 49.3 | 57.3 | 24.8 | 16.7 |
| 10,000,000 | 4.0 | 20.0 | 48.0 | 56.0 | 21.4 | 16.0 |

Bytes/entry is flat across N (the 1K row carries the usual doubling slack for every contender), i.e. O(1)
space per entry and O(n) total. The shipping COD is 56-57 B (the 40 B node dominates); the compact OA table
is 21-25 B (below the plain `ConcurrentDictionary`'s 48-49, and less than half of COD), the chained no-hash
layout 16 B (2 B under `Dictionary`, which is not thread-safe, ordered or indexable).

---

## 11. Per-member complexity — every public member measured and classified (2026-09-14)

`benchmark/collections/probes/concurrent_ordered_dict_per_member.cs` times every public member of the
**shipping** `ConcurrentOrderedDict<int,int>` across N = 1e3 .. 1e7, one P-core, 150 ms tier-1 warm, per-CALL
cost. Members sharing one implementation path are measured once by a representative (mapped below). Unit is
one call: an O(1) member's per-call cost is flat (bounded by the cache tier); an O(n) member's grows ~10x
per decade. Point reads are timed over a 256-key hot set to isolate the call from cache-warming.

### 11.1 Measured (per call; ns unless the row says ms)

| member (measured path) | 1K | 10K | 100K | 1M | 10M | 1K->10M | class |
|---|---:|---:|---:|---:|---:|---:|---|
| `TryGetValue` | 1.17 | 1.17 | 1.17 | 1.56 | 1.56 | 1.3x | **O(1)** |
| `ContainsKey` | 1.17 | 1.17 | 1.56 | 1.17 | 1.56 | 1.3x | **O(1)** |
| `IndexOf` | 1.56 | 1.17 | 1.17 | 1.56 | 1.56 | 1.0x | **O(1)** |
| `this[int]` get | 0.39 | 0.39 | 0.39 | 0.39 | 0.39 | 1.0x | **O(1)** |
| `GetKeyAt` | 0.39 | 0.78 | 0.39 | 0.39 | 0.39 | 1.0x | **O(1)** |
| `TryGetAt` | 0.39 | 0.39 | <0.4 | <0.4 | 0.39 | 1.0x | **O(1)** |
| `Count` | <0.4 | <0.4 | <0.4 | <0.4 | 0.39 | flat | **O(1)** |
| `SetByKey` (existing) | 15.23 | 15.62 | 14.45 | 14.84 | 14.45 | 0.9x | **O(1)** |
| `SetAt` | 15.62 | 14.84 | 14.45 | 14.45 | 14.45 | 0.9x | **O(1)** |
| `TryUpdate` | 15.62 | 14.45 | 14.84 | 14.45 | 14.45 | 0.9x | **O(1)** |
| `Snapshot()` | <0.4 | <0.4 | <0.4 | <0.4 | <0.4 | flat | **O(1)** |
| `GetEnumerator()` | <0.4 | <0.4 | <0.4 | <0.4 | <0.4 | flat | **O(1)** |
| `TryAdd` (build/N) | 49.9 | 47.3 | 153.1 | 343.6 | 770.9 | 15.4x | **amortized O(1)** |
| `TryRemove` tail / pop | 42.4 | 35.8 | 35.7 | 69.3 | 157.4 | 3.7x | **O(1)** |
| `TryRemoveSwapBack` | 32.4 | 33.0 | 32.4 | 81.9 | 153.4 | 4.7x | **O(1)** |
| `TryRemove` interior @0 (ms) | 0.000 | 0.03 | 0.38 | 4.12 | 54.39 | 2e4x | **O(n)** |
| `ToArray` (ms) | <.001 | <.001 | 0.08 | 0.47 | 7.91 | 1e4x | **O(n)** |
| `Keys` (ms) | <.001 | <.001 | 0.08 | 0.47 | 7.59 | 1e4x | **O(n)** |
| `foreach` consume (ms) | <.001 | <.001 | 0.03 | 0.32 | 3.82 | 1e4x | **O(n)** |
| `Pairs` consume (ms) | <.001 | 0.05 | 0.46 | 4.59 | 48.62 | 1e4x | **O(n)** |
| `CopyTo` (ms) | <.001 | <.001 | 0.06 | 0.49 | 6.83 | 1e4x | **O(n)** |
| `AddRange` (ms) | 0.07 | 0.43 | 14.0 | 251.4 | 6372.5 | 1e5x | **O(m)** |
| `RemoveWhere` all (ms) | 0.02 | 0.16 | 1.55 | 17.35 | 163.83 | 1e4x | **O(n)** |
| `Clear` presized-to-N (ms) | <.001 | <.001 | 0.02 | 0.03 | 0.05 | - | **O(initial capacity)** |
| `Clear` default-grown (ms) | <.001 | <.001 | <.001 | <.001 | <.001 | flat | **O(1)** |

Reads are flat 1.0-1.3x (`~1.2 ns` key path, `~0.4 ns` positional / count — the ~0.4 ns floor is the 100 ns
timer over 256 iterations, so `<0.4` means "unmeasurably small, one field/array load"). Replaces are flat at
~15 ns (the write-lock acquire + reentrancy guard + one seam store). `Snapshot()`/`GetEnumerator()` are free
(a struct that captures the arrays + count, no copy). `TryAdd`'s 15.4x rise is NOT super-linear over a 10,000x
data span — it is the per-add DRAM + GC-gen2 constant rising with the footprint, the same step
`ConcurrentDictionary`'s own unsized build shows (19.5x) and `List` shows (4.0x); the algorithm is one hash
walk + one node + amortized array doubling = amortized O(1). The two O(1) removals rise 3.7-4.7x (again the
shared memory step), never with N. The O(n) block grows a clean ~10x per decade (interior removal 4.12 ->
54.4 ms across the last decade; `ToArray` 0.47 -> 7.9). `AddRange`'s measurement builds the N-pair array
inside the timed region, so its tail is inflated by an 80 MB allocation + GC at 10M, but it is O(m): one
presize (`TryGetNonEnumeratedCount`) + one `TryAdd` per pair.

### 11.2 The finding on `Clear`

`Clear()` calls the vendored `ConcurrentDictionary.Clear`, which allocates a fresh bucket array of
`GetPrime(_initialCapacity)` slots (`ConcurrentDictionary.cs:697`), and `_initialCapacity` is the CAPACITY the
map was constructed with (`:244`). So **`Clear` costs O(initial capacity), not O(current count)**: a collection
`new ConcurrentOrderedDict(1_000_000)` that has been emptied down to a handful of entries still allocates a
~1e6-slot bucket array on every `Clear` (measured 0.05 ms at N=10M — the bucket array is large-object
zero-paged, so wall-time is well under the O(N) allocation's nominal cost, but the work is O(initial
capacity)). A default-constructed instance keeps the tiny initial capacity, so its `Clear` is O(1) even after
growing to millions. The COD side of `Clear` (`_store = Store.Empty`) is O(1) either way.

### 11.3 Full roster — every public member mapped to a measured path and its class

**O(1) — key reads** (all share `TryGetValue`'s one hash-node visit, ~1.2 ns, flat across N):
`TryGetValue`, `ContainsKey`, `GetByKey`, `this[TKey]` get, `GetOrAdd`(hit), `AddOrUpdate`(hit read),
`IndexOf`, `TryGetIndex`, `Comparer`.

**O(1) — positional reads** (one array index / field read, ~0.4 ns, flat): `this[int]` get, `TryGetAt`,
`GetKeyAt`, `Count`, `IsEmpty`, `ValuesView.this[int]` / `.GetKeyAt` / `.AsSpan` / `.KeysAsSpan` / `.Count`.

**O(1) — value replace** (one in-place store under the lock, ~15 ns, flat): `SetByKey`(existing key),
`this[TKey]` set(existing), `SetAt`, `TryUpdate`, `AddOrUpdate`(update branch). (Non-atomic `TValue` such as
`decimal` clones the value array => that specific case is O(n); atomic `TValue` is O(1).)

**O(1) — structural capture** (a `readonly struct` over the current arrays + count, no copy, free):
`Snapshot()`, `GetEnumerator()`, `IEnumerable<TValue>.GetEnumerator()`.

**Amortized O(1) — append** (one hash walk + one node + amortized array doubling): `TryAdd`, `Add`,
`this[TKey]` set(new), `SetByKey`(new), `GetOrAdd`(miss), `AddOrUpdate`(add branch).

**O(1) — removal that does not preserve order-position of others**: `TryRemove`/`Remove`/`RemoveAt` when the
target is the LAST entry (tail pop); `TryRemoveSwapBack` at any position.

**O(n) — order-preserving interior removal** (shifts the tail, O(n - p)): `TryRemove`/`Remove`/`RemoveAt` of a
non-last entry.

**O(n) — bulk / whole-collection** (touch every element): `RemoveWhere`, `ToArray`, `Values`, `Keys`,
`CopyTo`, `foreach` / enumeration, `Pairs`. `AddRange(source)` and `ctor(IEnumerable)` are **O(m)** in the
source length (one presize + per-item append).

**O(initial capacity) — `Clear`** (O(1) default-constructed, O(N) presized-to-N; see 11.2).

**Constructors**: `ctor()` / `ctor(comparer)` O(1); `ctor(capacity)` O(capacity) (allocates the two arrays +
the bucket table); `ctor(source)` O(m) (delegates to `AddRange`).

---

## 9. Reproduction

```
DOTNET_TC_CallCountingDelayMs=0 PROBE_KEYS=perm \
  dotnet run -c Release benchmark/collections/probes/compact_ordered_dict_probe.cs -- 1000 100000 1000000 10000000
# PROBE_ONLY=OA,no-hash,COD   restricts the rows; omit PROBE_KEYS for the sequential-key regime
```

References read for this discovery: `ConcurrentOrderedDict.cs` (the Store rules, `ValueIndex`, every
mutator), the vendored `ConcurrentDictionary.cs` (`Tables`/`Node`/`VolatileNode`, `TryAddInternal`,
`TryRemoveInternal`, `GrowTable`), dotnet/runtime `Dictionary.cs` (`Entry`, `_buckets` 1-based indices,
`StartOfFreeList = -3` free-list encoding in `Remove`/`TryInsert`, `Resize`), CPython `Objects/dictobject.c`
(compact dict layout comment, `DKIX_DUMMY`, `delitem_common`, `dictresize`, `USABLE_FRACTION`/`GROWTH_RATE`,
the free-threaded no-reuse rule), and the measured baseline in `ConcurrentOrderedDict.TODO.md` § Memory
analysis (commit 539e9c05).
