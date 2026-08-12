# `np.unique` family — design & performance

NumSharp's uniqueness family: `np.unique` and the NumPy 2.x Array-API set functions
`np.unique_values` / `unique_counts` / `unique_inverse` / `unique_all`. Target is 1-to-1
behavioural parity with **NumPy 2.4.2**. This document is the map of *why* the code is shaped the
way it is — every non-obvious decision here is backed by a measurement, and the measurements are the
point (SIMD and hashing each help in exactly one regime and hurt in another).

---

## 1. API surface

| Function | Returns | Semantics |
|---|---|---|
| `np.unique(ar, axis=null, equal_nan=true, sorted=true)` | `NDArray` | bare-return form (no `return_*` flag) |
| `np.unique(ar, return_index, return_inverse=false, return_counts=false, axis=null, equal_nan=true, sorted=true)` | `NDArray[]` `[values, index?, inverse?, counts?]` | full form |
| `np.unique_values(x)` | `NDArray` | `≙ unique(x, equal_nan=false, sorted=false)` |
| `np.unique_counts(x)` | `UniqueCountsResult{Values, Counts}` | `≙ unique(x, return_counts=true, equal_nan=false)` |
| `np.unique_inverse(x)` | `UniqueInverseResult{Values, InverseIndices}` | `≙ unique(x, return_inverse=true, equal_nan=false)` |
| `np.unique_all(x)` | `UniqueAllResult{Values, Indices, InverseIndices, Counts}` | all four outputs, `equal_nan=false` |

The four Array-API functions are thin wrappers over the `unique` machinery with **`equal_nan=false`**
(each NaN is a distinct value) and **`axis=null`** (flattened). NumPy returns namedtuples; C# stands
in with named-field `readonly struct`s (implicit `NDArray[]` conversion + `Deconstruct` + indexer —
the `np.meshgrid` house shape). `values` preserves the input dtype; `index`/`inverse`/`counts` are
int64 (NumPy `intp`).

Files: `Manipulation/np.unique.cs`, `np.unique_values.cs` (the four wrappers + result structs),
`NDArray.unique.cs` (instance surface + dispatch), `NDArray.unique.Kwargs.cs` (flat sort path + axis
path), `NDArray.unique.Hash.cs` (int/complex hash fast path).

---

## 2. Three execution paths

```
np.unique / unique_values / unique_counts / unique_inverse / unique_all
        │
        ├── axis != null ──────────────► AXIS PATH        (NDArray.unique.Kwargs.cs :: uniqueAxisKwargs)
        │                                slab sort + dedup over sub-arrays
        │
        └── axis == null (flatten) ──┬── unique_values / unique_counts, int-family or complex
                                     │        ► HASH PATH  (NDArray.unique.Hash.cs)
                                     │
                                     └── everything else (index/inverse needed, or float/decimal)
                                              ► SORT PATH  (NDArray.unique.Kwargs.cs :: uniqueFlatKwargs)
```

### 2a. SORT path (`uniqueFlatKwargs`) — the general flat algorithm

Port of NumPy's `_unique1d`: extract keys (+ perm when `return_index`/`return_inverse`), sort,
build the adjacent-differs mask, emit values/index/inverse/counts. NaN-capable dtypes partition NaN
to the tail (`equal_nan=false` → each NaN distinct; `equal_nan=true` → one representative),
stabilising the NaN tail by original index to match NumPy's mergesort. `inverse` is reshaped to the
input's shape (NumPy 2.0), including the 0-d `()` case.

Cost profile (measured, `long` keys): **sort is 87–90 % of the pipeline**; extract 2–3 %, mask 4–6 %,
emit 4–6 %. `Array.Sort` (scalar introsort) is the bottleneck; NumPy uses SIMD `vqsort`
(`x86-simd-sort`) which is 5–10× faster on the same sort — the single reason float `unique_values`
loses to NumPy.

### 2b. HASH path (`NDArray.unique.Hash.cs`) — int-family + complex, values/counts only

Open-addressing (linear-probe) dedup with load-factor-0.5 growth, then a sort of the *unique set*
(typically ≪ n). Beats the sort path at every cardinality because it sorts U values, not n. Used
only by `unique_values`/`unique_counts` (they need no perm) and only for the dtypes **NumPy itself
hashes** — the integer family and complex — see §4.

- **Integer family** (bool/byte/sbyte/int16/uint16/int32/uint32/int64/uint64/char):
  `UniqueHashSortedInt<T>`. Bit-reinterpret is a bijection with the value, so `HashKey` + `.Equals`
  dedup is exact; `Array.Sort` on the small unique set gives NumPy's ascending order.
- **Complex**: `UniqueHashSortedComplex`. Any-NaN complex is partitioned out (each distinct,
  appended after the sorted finite set); `-0.0` is normalized to `+0.0` **per component for hashing
  only**; the **first-occurrence original bits are stored** (NumPy keeps the first element's signed
  zero) and equality is `Complex ==` (IEEE per component: `-0.0 == +0.0`). The finite set is
  lex-sorted (real, then imaginary).

### 2c. AXIS path (`uniqueAxisKwargs`) — uniqueness over sub-arrays

`moveaxis(axis → 0)`, then sort the sub-array indices with a lexicographic **slab comparator** and
dedup adjacent equal slabs. Each slab is compared as a NumPy *structured (void) scalar* — see §3.

---

## 3. Parity semantics (all probed against 2.4.2)

- **`equal_nan`** — flat path: `false` keeps every NaN distinct, `true` collapses to one. **Axis
  path IGNORES `equal_nan`** (the consolidated void dtype fails `_unique1d`'s `"cfmM"` guard): every
  NaN sub-array is distinct (`nan != nan`), and a signed-zero sub-array collapses (`-0.0 == +0.0`,
  representative following input order). Slab equality therefore uses IEEE `==`, and slab ordering
  IEEE `<`/`>` (not `.Equals`/`CompareTo`, which mis-handle both NaN and `-0.0`), with a stable
  tie-break by original index reproducing NumPy's mergesort argsort.
- **`sorted`** (NumPy 2.3) — accepted for API parity but a **no-op**: NumSharp always returns sorted
  output. NumPy's `sorted=false` hash order (int/complex, values-only) is `std::unordered_set`
  iteration order — platform/compiler-specific, not reproducible in C#. Spec-compliant (order
  unspecified) and consistent with the `unique_*` family.
- **`inverse_indices`** — reshaped to the input's shape so `take(values, inverse)` reconstructs it,
  including a 0-d input → `()` scalar (the flat path returns `(1,)` there; the wrappers restore it).
- **axis out of range** → `AxisError` ("axis {a} is out of bounds for array of dimension {n}",
  NumPy-verbatim, reporting the original axis).
- **dtype coverage** — all 15 NumSharp dtypes. Char and Decimal have no NumPy analog: verified by the
  reconstruct invariant. Decimal is deliberately kept on the sort path (`1.0m != 1.00m` by bits, so a
  bit-hash would not dedup them; the sort path's value-equality does).

### Documented divergences (`[Misaligned]`)

1. **`unique_values` order** (integer/complex only): NumPy returns `std::unordered_set` hash order;
   NumSharp returns sorted (identical as a set). **Float dtypes are NOT in NumPy's hash map**, so
   NumPy sorts them too → NumSharp's float `unique_values` is bit-exact, not divergent.
2. **float16 + axis + NaN**: NumPy 2.4.2's void sort mis-orders float16 NaN to the FRONT
   (inconsistent with its own 1-D float16 sort and its float32/64 void sorts, all NaN-last).
   NumSharp keeps the consistent NaN-to-end ordering; set and counts identical, only NaN position
   differs.

---

## 4. Performance — the measured decisions

All ratios **NPY/NS** (>1 = NumSharp faster), Release, best-of-rounds, warm.

### 4a. Why the axis slab comparator is SCALAR, not SIMD or IL

The dominant cost of axis-unique is the **O(n·log n) comparison count**, and each lexicographic
compare **early-exits at the first differing element** (~1 element in real data). A POC raced
hand-written SIMD comparators (an upper bound on IL+SIMD, which JITs to the same machine code)
against the scalar path inside a real `Array.Sort`:

| regime | pure SIMD | hybrid (scalar-probe → SIMD) |
|---|---|---|
| typical (differ early) | **1.2–1.64× slower** | **1.2× slower** |
| adversarial (long common prefix) | 2.4–3× faster | 2.5–3.5× faster |

SIMD *hurts* the common case (vector-load + `MoveMask` + `tzcnt` overhead vs a compare that exits at
element 0), and even the hybrid loses because a larger hot comparator body inlines/optimises worse.
It only wins the pathological long-common-prefix case — where NumSharp's scalar path is **already
10–24× faster than NumPy** (NumPy's `axis=` sort is the generic scalar `VOID_compare`, not SIMD). The
scalar generic comparator (`CompareSlabsT<T>`, JIT-monomorphised per value type) is correct.

**Axis perf (NPY/NS):** 1.6–24× across the (n, slabSize) spectrum; the lead grows with slab size
(2.6× → 24×) because the scan early-exits while NumPy's per-void-element cost climbs.

### 4b. Why floats use the SORT path (and int+complex use HASH)

This mirrors **NumPy's own algorithm selection**: NumPy hashes int + complex (both in its
`_unique_hash` map) and sorts floats with SIMD `vqsort`.

- **Int**: our hash beats NumPy's hash — `unique_values` **1.3–5.4×**, `unique_counts` **1.0–1.5×**
  (≤1 % cardinality; high-cardinality counts is memory-latency-bound).
- **Complex**: our hash crushes NumPy's scalar lexicographic complex sort —
  `unique_values`/`unique_counts` **3.0–6.7×** (except the memory-bound high-cardinality counts cell,
  ~1.1×).
- **Float**: measured a hash extension end-to-end and **rejected it**. It helps duplicated data
  (2.5–7× over our sort path) but **regresses ~2× on all-unique input** (insert N + scalar-sort N vs
  a single sort), and a cheap cardinality sample provably cannot distinguish "100K-unique-in-1M"
  (hash helps) from "all-unique" (hash hurts) — both look unique in any sub-cardinality sample. NumPy
  sidesteps this with SIMD `vqsort`, which NumSharp's core lacks. So floats keep the sort path; their
  `unique_values`/`unique_counts` cells (~0.1–0.3× vs NumPy) are a shared-sort-core limitation, not a
  property of these functions.
- **`unique_all`** (all dtypes, sort path): **2.3–7.6×** — the "give me everything" call, where NumPy
  pays for a mergesort argsort + cumsum + nonzero + diff.

### 4c. The hash finalizer must fully avalanche (splitmix64)

A single Fibonacci multiply (`k * φ⁻¹`) then a low-bits table index **catastrophically clusters**
float bit patterns: integer-valued doubles (`1.0, 2.0, 4.0 …`) carry their discriminating bits high
and have zero low bits, so they all landed in bucket 0 — **~500 probes/element** (vs 1.3 for small
integers). `HashKey`/`HashKeyComplex` use the **splitmix64 finalizer**, which avalanches every input
bit into the low bits: doubles drop to 1.4 probes/element and the integer path is unchanged (and
slightly faster). This matters for any structured integer-like key, not just floats.

---

## 5. Gates

- **Differential fuzz corpus** (`test/oracle/gen_oracle.py` → `unique` tier in `sort.jsonl`): the
  flat `unique` values-only path, bit-exact vs NumPy 2.4.2.
- **Unit tests** (all from probed NumPy output):
  `Manipulation/np.unique_values.Test.cs` (Array-API family + result structs + the `unique_values`
  order `[Misaligned]`), `np.unique.AxisEdgeCases.Test.cs` (axis NaN-distinct / signed-zero collapse /
  `equal_nan` no-op / `sorted=` / `AxisError` / float16 `[Misaligned]`), `np.unique.EdgeCases.Test.cs`,
  `NDArray.unique.Test.cs`.
- **In-process differential matrices** (author-run, not committed): a 102-case flat matrix (13 NumPy
  dtypes × layouts × NaN/inf/-0.0/empty/0-d), a 450-case axis matrix (dtypes × axes × return-flags ×
  layouts × NaN/-0.0/complex/empty; 448 exact + 2 excused float16), and a 16-case float/complex hash
  battery (growth path, NaN-heavy, signed zero, subnormals, complex NaN variants).

---

## 6. File map

| Concern | File |
|---|---|
| `np.*` surface + result structs | `Manipulation/np.unique.cs`, `np.unique_values.cs` |
| Instance surface + path dispatch | `Manipulation/NDArray.unique.cs` |
| Flat SORT path + AXIS path | `Manipulation/NDArray.unique.Kwargs.cs` |
| HASH fast path (int + complex) + splitmix | `Manipulation/NDArray.unique.Hash.cs` |
| Fuzz gate | `test/NumSharp.UnitTest/Fuzz/` (`unique` in `sort.jsonl`) |
| Unit gates | `test/NumSharp.UnitTest/Manipulation/np.unique*.Test.cs`, `NDArray.unique.Test.cs` |
