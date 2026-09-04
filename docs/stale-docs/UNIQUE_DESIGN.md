# `np.unique` family — design & measured perf decisions

Covers `np.unique` (bare + `return_index`/`return_inverse`/`return_counts`), the instance
`NDArray.unique`, and the Array-API siblings `np.unique_values` / `np.unique_counts` /
`np.unique_inverse` / `np.unique_all`.

Source files: `Manipulation/np.unique.cs`, `Manipulation/np.unique_values.cs`,
`Manipulation/NDArray.unique.cs` (+ `.Kwargs.cs` sort path, `.Hash.cs` integer hash path,
`.Radix.cs` radix sort helpers). The radix core is the shared axis-sort kernel
`Backends/Default/Sorting/RadixSort.cs`.

---

## The engine selection, mirroring NumPy — but bailing to radix where NumPy thrashes

NumPy 2.4.2 itself picks between **hash** and **sort** per operation and dtype:

* `np.unique(int)` and `np.unique_values(int)` route through `_unique_hash` (`unique.cpp`, a
  `std::unordered_set`). This is **cardinality-scaled** and catastrophic at high cardinality —
  a 2 M-element `int32` with ~2 M distinct values takes **≈1540 ms** (`int64` ≈1430 ms), because
  every insert into the multi-megabyte table is a cache miss and the table rehashes ~20 times.
* `np.unique(float)` and float `unique_values` are **not** in NumPy's hash map, so they take the
  sort path (SIMD `vqsort` from `x86-simd-sort`, ≈60 ms for 2 M `float64`).
* `unique_counts` / `unique_inverse` / `unique_all` (any dtype) take the sort path (they need the
  sorted structure and/or an argsort), which is why they are slow across the board in NumPy
  (200–1400 ms — mergesort argsort + cumsum + nonzero + diff).

NumSharp's selection is the same shape, **plus a bail-out that catches the cardinality NumPy's hash
can't**:

| Path | Dtypes | Entry points |
|------|--------|--------------|
| **Integer hash + radix bail-out** | bool/byte/sbyte/int16/uint16/int32/uint32/int64/uint64/char | `np.unique(int)` values-only, `unique_values`/`unique_counts` (int), `np.unique(int, return_counts)` counts-only |
| **Radix sort / argsort** | + single/double | `np.unique(float)`, all `return_index`/`inverse`/`counts` perm paths, `unique_inverse`/`unique_all` (every dtype) |
| **Scalar BCL sort** | Half, Complex, Decimal | their `uniqueFlatSorted*` paths (no monotonic radix key) |

Complex additionally has a dedicated hash path (`UniqueHashSortedComplex`) for `unique_values`/
`unique_counts` — NumPy hashes complex too.

---

## What changed (2026-08-22): the sort was a comparison sort

The flat pipeline (`_unique1d` port) sorts a key buffer (+ perm for the index/inverse/counts) then
masks/dedups. The sort was **`System.Array.Sort`** — the .NET BCL introsort, a *comparison* sort
(~n·log n comparisons, ~50 % branch-mispredict on random data). That made **every** sort-path cell
lose to NumPy:

* `np.unique` int low/med and **all** floats: **0.14–0.77×**.
* `unique_values`/`unique_counts` **all** floats: **0.12–0.61×**.
* `unique_counts` int **high** cardinality (the hash-with-counts table thrashing): **0.25–0.30×**.

Three moves fixed it, all reusing machinery that already shipped:

### 1. Radix replaces the BCL sort (`NDArray.unique.Radix.cs`)

`RadixSortValues<T>` / `RadixArgSortKeysPerm<T>` route the sort through the **same LSD
`RadixSort`** the axis-sort driver runs: O(n·nbytes), comparison-free, with a single-pass histogram
and a **trivial-pass skip** that makes it magnitude-adaptive (small integers do 1 pass, not 4). The
per-dtype **monotonic key transforms** are the same ones `AxisSort` uses (`ToKey32/64`, `FromKey32/64`
— JIT-const `typeof(T)` dispatch, exactly one branch per instantiation, `Unsafe.As` reinterpret, the
`HashKey<T>` pattern), so ascending key order == NumPy value order and `-0.0` sorts before `+0.0`
exactly as `Array.Sort`'s float comparer. Two additive `RadixSort.ArgSortU32/U64` overloads expose
the co-sorted **key** buffer alongside the sorted index column, so the perm path recovers the sorted
values through the key's inverse transform with **no gather**.

**Bit-parity is by construction.** Everything around the sort is unchanged — NaN partitions to the
tail with original bits preserved, mask via IEEE `Equals`, first-occurrence via min-perm-within-run,
inverse reshape, counts. Radix produces the same total order as the BCL comparer for every non-NaN
value, so the mask/dedup output is identical. (Radix is stable, so within-run order is
ascending-index — first occurrence — which the min-scan also recovers.)

### 2. Integer values-only / counts-only → the hash (`uniqueValuesOnly<T>`, `uniqueFlatSorted<T>`)

`np.unique(int)` values-only and `np.unique(int, return_counts)` now route to the **same hash path**
as `unique_values`/`unique_counts` (they were on the sort path). For integers `equal_nan` is
irrelevant, so the result is identical. The hash beats any sort at low/medium cardinality (a
100-distinct 2 M array is a 100-entry table in L1).

### 3. High-cardinality radix bail-out in the hash (`UniqueHashSortedInt<T>`)

The one case the hash loses is high cardinality — past ~L2 the open-addressing table is a cache miss
per insert (this is why NumPy's own hash thrashes). Rather than guess cardinality up front (a cheap
sample provably can't distinguish 100k-distinct-in-1M from all-unique), the loop **observes growth**:
once `uniqueCount ≥ 2¹⁷ (131 072)` — table past L2 — it abandons the hash and radix-sorts the
already-extracted `data` buffer, then dedup-emits (`RadixDedupSortedInt`). Low/medium cardinality
never reaches the threshold (the array's own distinct count is the ceiling), so it keeps the hash;
the wasted inserts before a bail are bounded by the threshold. The radix result is the identical
sorted set.

The bail-out is the lever that turns int-high from a *loss inherited from NumPy's own algorithm* into
a rout: `int32` 2 M-distinct `np.unique` goes **NumPy 1540 ms → NumSharp 21 ms (74×)**.

### 4. Counts-only skips the perm argsort (`EmitValuesAndCountsFloat`, generic/float branches)

`return_counts` with neither index nor inverse does **not** need a permutation — counts are
run-lengths of the sorted values, derived from the mask alone. The float paths now sort **values
only** (no perm argsort, no perm allocation) and emit values+counts directly; the integer path
routes to the hash. This is what flipped float `unique_counts` from 0.12–0.70× to 1.2–2.6×.

---

## Why floats keep the sort (no float hash)

Measured and rejected earlier (see `NDArray.unique.Hash.cs` header): hashing floats helps duplicated
input but regresses ~2× on all-unique input (insert N + scalar-sort N), and a cheap cardinality
sample cannot distinguish the two. NumPy makes the same call (floats are not in its hash map). So the
float paths radix-sort, which is already a large win over the old BCL sort and — with the bail-out
absent (floats never hash) — has no thrash to catch. `float32` at the very smallest cardinalities sits
at ~parity with NumPy's SIMD `vqsort`; closing that last gap is a **shared sort-core** change (porting
a SIMD quicksort), not a property of `unique`.

## Why Half / Complex / Decimal keep the scalar BCL path

No monotonic radix key: Half needs a float compare (no `Vector<Half>`), Complex is lexicographic,
Decimal is 16-byte value equality (and `1.0m != 1.00m` by bits, so no bitwise hash). These are the
same three dtypes `AxisSort` routes to its scalar kernel. They are rare in `unique` and correct as-is.

---

## Measured results (NPY/NS, >1 = NumSharp faster; 2 M elements, best-of-9 × min-of-3-runs, Release)

Ratio = `NumPy_ms / NumSharp_ms`. `low` ≈ 100 distinct, `med` ≈ 50 000, `high` ≈ mostly-unique.

| op | before (worst→best) | after (worst→best) | headline |
|----|--------------------|--------------------|----------|
| `np.unique` | 0.14× – 14× | **0.98× – 74×** | int32-high 14→74×; all floats now ≥0.98× |
| `unique_values` | 0.18× – 16× | **1.10× – 69×** | float32-low 0.18→1.10×; int32-high 8→69× |
| `unique_counts` | 0.12× – 14× | **1.22× – 15×** | int-high 0.25→3.25×; all floats 0.12→≥1.2× |
| `unique_inverse` | 1.14× – 7.9× | **4.67× – 14×** | radix argsort widens every cell |
| `unique_all` | 1.00× – 14× | **3.27× – 38×** | int-low 14→38×; float32-high 1.9→8.6× |

**81 / 90 cells ≥ 1.5× NumPy; 0 degradations vs the pre-change NumSharp baseline; the single
sub-1.0 cell is `np.unique float32-low` at 0.98× (parity, 16.6 vs 16.3 ms), bounded by NumPy's SIMD
sort.** The high-cardinality bail-out is pinned by `NDArray.unique.Test.cs ::
Unique_HighCardinality_RadixBailout` (the oracle's small corpus arrays never reach the 131 072
threshold). Bit-exactness is gated by the oracle `FuzzMatrix` (sort/multioutput tiers) + 132 unique
unit tests.

## Follow-on (out of scope here)

`np.sort` numeric is ~0.5× NumPy (our radix vs NumPy's SIMD `vqsort`). Porting a SIMD radix or a
vectorized quicksort into the shared sort core would lift `np.sort`, `np.argsort`, the float `unique`
tail, and `partition`/`searchsorted` together — a bigger, separate lever.
