# Handover: full combinatorial advanced indexing (NumPy `mapping.c` parity)

**Branch:** `nditer`  **Date:** 2026-06-27  **Status:** IN PROGRESS — Phases A–E done; random sweep **697 → 0 divergences** across every measurable window (all 5 buckets fixed, committed, CI-pinned). The ONLY open item is **R3**, a pre-existing flaky teardown heap-corruption (memory safety) that blocks un-marking the full-corpus `Index_Random` gate but does NOT affect correctness or CI. See "Execution status" below.

This is the successor to [`advanced-index-axis-placement.md`](./advanced-index-axis-placement.md)
(which resolved the *two-advanced-indices-with-a-slice* case via `TryBuildMultiAdvancedGrid`).
A differential fuzz built this session proves that the per-shape **patchwork of `Try*`
handlers does not generalise**: ~660–700 distinct divergences remain across exotic
*mixed advanced-index* combinations, plus a flaky **heap-corruption crash** in that path.
Closing the gap means porting NumPy's **single unified two-stage algorithm**
(`prepare_index` + `MapIterNew`), replacing the heuristic stack. This document is the
how-to.

> Project principle (from `.claude/CLAUDE.md`): *"Match NumPy Implementation Patterns —
> don't just match behaviour, match NumPy's implementation structure."* This work is the
> canonical application of that rule to indexing.

---

## 0. TL;DR

1. **Lock the gate first** (Phase A): promote the scratchpad differential harness into the
   committed oracle pipeline (`test/oracle/` + a `[FuzzMatrix]` replay test). Until the
   harness is committed it cannot defend the fix.
2. **Hunt the memory-safety crash** (Phase B, priority-0): a still-unidentified OOB write in
   the mixed-advanced path corrupts the heap (flaky `AccessViolation`). It must be found and
   fixed (or subsumed by the rewrite) before anything else ships.
3. **Port `prepare_index`** (Phase C): one classifier pass replacing `NormalizeIndexInputs`
   + the scattered validation. Kills the "accepts-what-NumPy-rejects" (374 cases) and most
   "NumSharp-throws-on-valid" (200 cases) divergences.
4. **Port `MapIterNew` + `_get_transpose`** (Phase D): one unified broadcast / axis-placement
   / gather-scatter, replacing `TryFetchSliceWithSingleAdvanced`, `TryBuildMultiAdvancedGrid`,
   `TryFetchLeadingMaskWithBasic`, `TryBuild0dBoolWithBasic` and the `_NDArrayFound` loop.
   Kills the shape/value divergences (85 cases) and the rest.
5. **Gate:** the curated sweep (2369 cases) stays 0/0 **and** the random sweep reaches 0/0
   (currently ~660–700 fails), full CI green on net8.0 + net10.0, `FuzzMatrix` green.

---

## 0b. Execution status (2026-06-27)

Executed against this plan. Differential random sweep (seed 20240626, 10000 cases): **697 → 0
divergences** across every measurable window ([0,5000) + [5000,6700) + [6700,7000) + [7000,10000)
= 10000 cases, all 0/0). Curated + dtype gate stays **0/0**; full CI suite **11003 pass / 0 fail on
net8.0 AND net10.0** (the +23 are the new `Indexing.CombinatorialParity` gate). Commits
`d20fb793` … `7e968f5e` on `nditer`. The full single-process [5000,10000) run still SEGFAULTS at a
flaky teardown OOB (R3) — windows are summed; correctness is bit-exact where they complete.

**Done:**
- **Phase A** (`d20fb793`) — oracle committed: `test/oracle/gen_index_oracle.py`, three corpora
  (`index_curated` 2265, `index_dtype` 104, `index_random_20240626` 10000), `Fuzz/IndexOracleTests.cs`
  with `[FuzzMatrix]` `Index_Curated`/`Index_Dtype` (CI gates) + `[OpenBugs]` `Index_Random`.
- **Phase B** (`468f7419`, `373978d8`) — block-copy bounds guards at every gather/scatter copy site;
  opt-in page-heap (`NUMSHARP_GUARD_PAGES`, `OsVirtualMemory.AllocGuarded`). **The major flaky
  corruptor is FIXED**: `SetData(int[])` didn't wrap negative coordinates (getter's `GetData` does),
  so `b[(object)-1] = v` wrote at `buffer[-1]` — an OOB heap write found by amplifying each divergent
  case in a loop until `V6[-1]=scalar` crashed.
- **Phase C** (`0be160ea`) — `Selection/NDArray.Indexing.PrepareIndex.cs`: faithful `prepare_index`
  port (classify + validate: too-many-indices, bool-array dim, integer/array value bounds,
  advanced-block broadcast-together, single-ellipsis, invalid-type). Wired as the multi-index gate.
- **Phase D** (`32cb060b`, `bea57936`, `36161bb8`, `65315bfe`, `9c2e16b2`) — slice neg-step
  out-of-range start fix; `largestOffset` neg-stride bound fix; grid handles a single (incl. n-d)
  advanced index and subsumes the removed `np.take` path; 0-d bool joins the advanced block in the
  grid (`MixKind.ZeroBool`); ellipsis no longer counts a 0-d bool as axis-consuming; **the grid is
  now the single advanced path for ALL `HAS_FANCY` tuples** incl. pure-advanced (fancy+int,
  fancy+fancy with no slice — which the old `_NDArrayFound` broadcast path mis-placed for a
  multi-dim fancy + int, e.g. `AT[…,arr(4,1),-1]` → `(4,1)`).
- **Phase E (partial)** (`7c72c006`, `56c9e7c6`, `aed44b31`, `6a3aa93c`) — empty index array (any
  dtype) is an empty integer fancy; empty value into a non-empty whole-array region raises ValueError;
  setter pure-basic slice path was missing a `return` (fell through to fancy — fixed many
  `A[...]=scalar` / `A[:,:]=scalar` / `A[None]=v` cases); non-subshaped fancy assignment validates the
  value broadcasts to the selection.

### Remaining work — R3 only (the 5 divergence buckets are DONE)

The 61 random divergences were 5 buckets; ALL are fixed, committed, and CI-pinned in
`Indexing.CombinatorialParity.MatrixTests.cs`. Each was gated behind the curated/dtype gate (stayed
0/0) and the random window it targeted, plus the full CI suite (net8.0 + net10.0):

| Bucket | What | Commit |
|---|---|---|
| **R1** | value must broadcast to an EMPTY / scalar selection on assignment, else `ValueError` (0-d-bool-False branch, scalar-element target = "setting an array element with a sequence", bool-mask zero-select) | `aea9fc78` |
| **B2** | a 0-d (scalar) array rejects any axis-consuming single index → "too many indices" (the single-index path bypasses `PrepareIndex`) | `fe80982b` |
| **B3** | empty advanced indices gather an empty result: skip fancy-array bounds when the block is empty (`FinishPrepare`), and a zero-length bool mask axis matches ANY axis size (`IsPartialShapeMatch`) | `880bc3df` |
| **B4** | basic assignment into an EMPTY slice selection is a no-op, not a `NpyIter.Copy` crash (`UnmanagedStorage.SetData`) | `b0a3048f` |
| **R2** | non-consecutive 0-d-bool/int advanced block moves to FRONT (`TryBuild0dBoolWithBasic` bails → grid; grid carries the pure-0-d-bool block dims via `outShape`) **+ a core `Shape.Broadcast` hash collision** (`(1,1)`≡`(0,1)` → broadcast_to wrongly no-op'd to an empty target) | `d6f30629` |
| — | (found while writing the regression tests) a MULTI-DIM empty bool mask was routed as a single empty fancy → wrong rank; now stays a mask | `7e968f5e` |

The order executed was R1 → B2 → B3 → B4 → R2 (the actual buckets, re-derived from the live
`diverge.txt`, differed slightly from this doc's original R1/R2 split). The two highlights worth
keeping in mind: **B3's `ScanFancyBounds` skip** mirrors NumPy bounds-checking advanced values only at
gather time (an empty broadcast gathers nothing), and **R2 uncovered a general broadcasting bug** —
`Shape.Broadcast` short-circuited on a shape-hash match without confirming the dims, and the hash
collides a 0-length axis with a size-1 axis.

**R3 — A flaky, layout-dependent teardown heap-corruption (memory safety; STILL OPEN).**
- *Symptom:* the main loop now COMPLETES (prints the result), but the full 10000-case run SEGFAULTS at
  process teardown (GC finalization walks a heap corrupted earlier in the run). `[0,5000)` and
  `[5000,5000)` windows complete — only the full accumulation tips it.
- *Symptom (re-measured this session, all divergences now fixed so this is the SOLE blocker):* every
  window completes 0/0 in isolation, but the single-process `[5000,10000)` run SEGFAULTs (exit 127,
  silent AccessViolation). The crash point is **wildly flaky** — observed at 6285 / 6432 / 6456 / 6514
  / 6756 / 9879 across runs; even the tightest reliable window `[6200,6400)` only crashes ~1/3 of runs
  under GC-per-case. The corruptor is an OOB write whose 1-past slot is usually harmless and only
  crashes once a GC/alloc lands a live object there.
- *Narrowed this session (supersedes the prior "pinned managed" guess):*
  1. **It is a SPECIFIC corpus shape, not allocation volume.** A 30 000-iteration stress loop of pure
     fancy + grid gathers with forced GC is clean; and `[6200,6400)` crashes at the same place
     regardless of how many cases preceded it (286 vs 1320). But no single case is the deterministic
     corruptor — every sub-window of a crashing window runs clean, so it is cumulative + threshold.
  2. **It is NOT a pooled native-buffer overrun.** `OsVirtualMemory.AllocGuarded` is already
     end-aligned (byte `[len]` is the guard page, so a 1-past write faults instantly), yet
     `NUMSHARP_GUARD_PAGES=1` runs clean — so the overrun targets an allocation guard mode does NOT
     reroute: a **pinned managed `FromArray` array** or a **direct-VirtualAlloc large zeroed buffer**
     (`UnmanagedMemoryBlock.AllocateZeroed` bypasses the pool ≥128 KiB).
  3. `DOTNET_GCStress=0xC` and guard-pages both perturb GC timing enough to MASK it (inconclusive,
     not a fix).
- *Plan for the next session:* a **per-case red-zone on `FromArray`** (and on the direct-VirtualAlloc
  zeroed path) — over-allocate by a guard span, sentinel it, keep `Count` at the usable length, and
  verify all live blocks' sentinels on a per-case hook via a weak-ref registry (gate behind a new env
  flag, revert after). Or capture the AV with WinDbg / `dotnet-dump` (procdump is installed; managed
  heap-corruption analysis needs `dotnet-dump`, which is NOT installed — `dotnet tool install -g
  dotnet-dump`). The reliable repro is `replay_index_jsonl.cs index_random_20240626.jsonl 6200 200 gc`
  (run it several times; ~1/3 crash).
- *Until fixed:* `Index_Random` stays `[OpenBugs]` — its full-corpus single-process run can crash. The
  fixes are pinned independently by the `Indexing.CombinatorialParity` `[FuzzMatrix]` gate, so the
  parity is protected in CI regardless.

**R4 — Final cleanup + close the gate (the test-pin + doc parts are DONE; dead-code delete + un-mark remain).**
- ✅ DONE: the 5 buckets are committed and pinned in `Indexing.CombinatorialParity.MatrixTests.cs`
  (`7e968f5e`); this doc's status is updated.
- TODO (low risk, deferred): the getter's `TryFetchSliceWithSingleAdvanced` is unreferenced — delete
  it. The `_NDArrayFound` advanced loops (getter + setter) are now reachable only for cases the grid
  declined; with the grid owning ALL `HAS_FANCY` confirm they are dead for advanced and delete (keep
  `Storage.GetView` for pure-basic). Not done yet because the R3 crash makes the full-corpus
  regression net harder to trust; do it once R3 is closed, behind the curated + windowed sweep.
- TODO (BLOCKED on R3): remove `[OpenBugs]` from `Index_Random` (`Fuzz/IndexOracleTests.cs`) so the
  full corpus becomes a CI gate, re-run under `NUMSHARP_GUARD_PAGES=1` (DOD §10), and flip Status to
  DONE. Blocked purely by the R3 teardown crash, not by any divergence.

### Diagnostic tooling (scratchpad — recreate from `Fuzz/IndexOracleTests.cs`)

The scratchpad harnesses are not committed; recreate `replay_index_jsonl.cs` by lifting the
base-recipe + token reconstruction from `test/NumSharp.UnitTest/Fuzz/IndexOracleTests.cs` (the C# half
is identical). It reads the committed corpus `index_random_20240626.jsonl` (resolve an absolute path
or it looks under `test/.../Fuzz/corpus/`):
- **`replay_index_jsonl.cs <file> [SKIP] [LIMIT] [gc]`** — primary measurement; prints categorized
  counts (agree-OK/ERR, ns-threw-on-valid, ns-accepted-invalid, shape/val diff) and writes flushed
  `trace.txt`/`diverge.txt` (survive a crash). Each `diverge.txt` line starts with the case `id` +
  `[category]`. The 4th arg `gc` adds `GC.Collect()` per case → trips closer to the R3 corruptor.
- **Mini-corpus:** `python` one-liner that filters the JSONL by a set of ids from `diverge.txt`; pass
  the absolute mini-file path as `<file>`. This is how each bucket was isolated to ≤13 cases.
- **R3 repro:** `replay_index_jsonl.cs index_random_20240626.jsonl 6200 200 gc` (~1/3 crash). A pure
  stress loop (30k gathers + GC) is CLEAN — so reproduce only through the corpus shapes.
- Run under `NUMSHARP_GUARD_PAGES=1` for end-aligned page-heap (catches POOLED native overruns only —
  it does NOT catch R3, which confirms R3 is a pinned-managed / direct-VirtualAlloc overrun).
- **Gotcha:** clear `%LOCALAPPDATA%\Temp\dotnet\runfile` after every `NumSharp.Core` rebuild — the
  file-based-app cache serves a stale Core (a green run can hide an unbuilt fix). Also note each
  `dotnet run <file>.cs` keys its runfile cache on the FILE; a `--no-build` after switching scripts or
  clearing the cache fails with "cannot find the file" — let it build once.

---

## 1. What is already done (this session — do not regress)

13 indexing fixes landed on `nditer`, each NumPy-verified and full-CI-gated
(`b03e40b7` … `998c1d23`). The **curated** differential sweep — every common index form ×
14 memory layouts × 13 dtypes × {get,set} = **2369 cases — is 0 divergences** (bit-exact
shape, values, and which inputs raise). Highlights relevant here:

| Commit | Fix | Keep-working invariant |
|---|---|---|
| `59f1445c` | per-axis integer OOB in coordinate access | `a[i,j]` validates each axis |
| `f85789bc` | fancy negative-OOB (was OOB read) | `a[[-7]]` raises, not garbage |
| `56b3d5c6` | broadcast value into ≥2-D fancy subspace (set) | `a[[0,1]] = -1` fills rows |
| `34b40af9` | single fancy on non-contiguous view | `a.T[[0,2]]` gathers |
| `80d07662` | multi-fancy broadcast **by shape** not size | `b[ia, ib2d]` → `(2,2,4)` |
| `a58bffa4` | 0-d bool + **basic** keeps source axes | `a[True,:]` → `(1,3,4)` |
| `9a2513c7` | over-index with **slices** raises | `a[:,:,:]` → `IndexError` |
| `21a28047` | broadcast/validate on assignment | `a[0]=[1,2]` → `ValueError` |
| `afd2ca22` | `SetData(NDArray,long[])` broadcasts+validates | `a[(object)0]=v` fills row |
| `998c1d23` | over-index with **advanced** raises (mem-safety guard) | `a[arr,arr,arr]` raises |

These are pinned in `test/NumSharp.UnitTest/Selection/Indexing.{Basic,Fancy,Edge,LayoutValue}Parity.MatrixTests.cs`
(now zero active `[OpenBugs]`) and `NDArray.Indexing.Test.cs`.

---

## 2. The gap, precisely

Random fuzzing (~10k pathological indices, seeded) over **mixed advanced-index
combinations** yields ~660–700 divergences. Categorised from the captured `diverge.txt`:

### By failure mode
| # | NumPy | NumSharp | Meaning |
|--:|---|---|---|
| 295 | raises `IndexError` | **returns** a shape | NumSharp accepts a malformed advanced combo (e.g. bool-array length ≠ axis, too-many after newaxis) |
| 79 | raises `ValueError` | **returns** a shape | advanced indices that **cannot broadcast together** are accepted |
| 85 | OK (shape) | OK but **wrong shape/values** | advanced-block axis placement / slice interleave is wrong |
| 72 | OK | throws `IndexOutOfRangeException` | valid combo rejected |
| 33 | OK | throws `ArgumentException` | "" |
| 32 | OK | throws `ArgumentOutOfRangeException` | "" |
| 30 | OK | throws `IncorrectShapeException` | "" |
| 21 | OK | throws `IndexError` | "" |
| 12 | OK | throws `OverflowException` | "" |
| — | raises | **hard crash** (`AccessViolation`, flaky) | OOB write → heap corruption |

### By index-form feature (every divergence is a *combination*)
| feature | count |
|---|--:|
| bool-array **+** fancy-array | 150 |
| fancy **+** slice (multi-dim fancy) | 136 |
| 0-d-bool **+** fancy | 121 |
| multi-fancy (≥2 integer arrays, exotic shapes) | 56 |
| fancy **+** newaxis / ellipsis | 29 |
| 0-d-bool combos | 24 |
| other mixes (empty arrays, bool+slice+fancy, …) | ~140 |

**Worked examples** (`A = arange(12).reshape(3,4)`, all probed vs NumPy 2.4.2):

```
A[arr([2,-3],(2,1)), b0(True)]        NumPy (2,1,4) [8,9,10,11,0,1,2,3]   NS (2,1) [8,0]      # 0d-bool+fancy
A[arr([2,-3,-1,0],(2,2)), slice(-1,_,2)]  NumPy (2,2,1) [11,3,11,3]       NS (2,2) [11,3,11,3]# multidim fancy+slice (missing axis)
A[b0(True), arr([2,2],(2,))]          NumPy (2,4) [8..11,8..11]           NS (2,) [2,2]       # 0d-bool+fancy
A[slice(4,-1,-2), arr([4,1],(2,))]    NumPy IndexError                    NS (0,2)            # accepts invalid
A[arr([0,0,0,2],(2,2)), slice()]      NumPy (2,2,4)                       NS IncorrectShape   # rejects valid
A[barr([F],(1,)), new, barr([F,F,F,F],(4,))] NumPy IndexError            NS (0,1)            # bool+newaxis+bool
```

**The common root cause:** every one of these *falls through* the `Try*` fast-path stack to
the `_NDArrayFound` broadcast loop, which (a) does not validate like NumPy, and (b) models
only "all advanced indices broadcast together" — it has **no general axis-placement** for
advanced-blocks interleaved with slices/newaxis/0-d-bools, and **no general broadcast-or-raise**
across heterogeneous advanced shapes. Patching more shapes into the stack is a losing game;
the categories overlap and interact.

---

## 3. Why the current architecture can't reach parity

Today `FetchIndices(object[])` (getter, `Selection/NDArray.Indexing.Selection.Getter.cs`) and
`SetIndices(object[])` (setter) are a **stack of pattern-matched fast paths**, each handling
one tuple *shape* and bailing otherwise:

```
NormalizeIndexInputs           # tuple spread + sequence/mask coercion
indicesLen==1 switch           # NDArray / int / ulong / bool / int[] / long[] / string …
TryBuild0dBoolWithBasic        # 0-d bool + ONLY basic            (bails if any advanced present)
TryFetchLeadingMaskWithBasic   # leading k-D mask + ONLY basic
TryFetchSliceWithSingleAdvanced# EXACTLY ONE advanced + basic     (bails if advCount!=1 or k-D mask)
TryBuildMultiAdvancedGrid      # ≥2 advanced + at least one slice/newaxis (bails if no basic, over-index, …)
_NDArrayFound loop             # everything else → broadcast all advanced together → FetchIndices<T>
all-ints / all-bools / slices  # → Storage.GetView
```

Each handler encodes a *slice* of NumPy's algorithm. The combinations the fuzz hits
(0-d-bool **and** fancy; bool-array **and** fancy; multi-dim fancy **and** slice; empty in the
mix) match **none** of the fast paths and fall to `_NDArrayFound`, which is not the full
algorithm. You cannot enumerate the combinations — NumPy's correctness comes from **one**
uniform pass, not N special cases. **The fix is to replace the stack with NumPy's two stages.**

---

## 4. The target: NumPy's two-stage model

Authoritative source (cloned in repo): `src/numpy/numpy/_core/src/multiarray/mapping.c`.

### 4a. Stage 1 — `prepare_index(self, index, ...)`  (`mapping.c:772`)
Walks the index tuple **once** and produces:
- a normalized array of typed index ops (`npy_index_info indices[]`), and
- an `index_type` bitmask, and
- `index_num`, `ndim` (output basic dims), `out_fancy_ndim`.

Per item it sets exactly one classification and validates as it goes:

| Item | flag set (`mapping.c`) | notes |
|---|---|---|
| `Ellipsis` | `HAS_ELLIPSIS` (:319) | at most one; expands to fill remaining axes |
| `np.newaxis`/`None` | `HAS_NEWAXIS` (:335) | adds an output axis, consumes no source axis |
| `slice` | `HAS_SLICE` (:348) | own output axis |
| python int | `HAS_INTEGER` (:380) | 0-d advanced (part of the fancy block) |
| 0-d bool | `HAS_BOOL` / `HAS_0D_BOOL` | → a length-1 (True) / length-0 (False) fancy index, **consumes no axis** |
| bool array (k-D) | `HAS_FANCY|HAS_BOOL` (:510) | replaced by its `nonzero()` → **k** integer arrays, each consuming an axis; **must match `arr.shape[i]`** for each of its k axes or raise `IndexError` (`mapping.c:1150` "must have the same number of dimensions") |
| integer array | `HAS_FANCY` (:559/609) | one advanced index |
| 0-d integer array | `HAS_INTEGER|HAS_SCALAR_ARRAY` (:594) | acts like an int but is "advanced" for placement |

Validation done here (all currently scattered/missing in NumSharp):
- **too many indices** (`mapping.c:168,303,551,667`) — axis-consuming count > ndim.
- **bool array dimensionality** must match the array's corresponding axes.
- integer / 0-d-array OOB is checked later in MapIter, but bounds are normalized here.

> **NumSharp parallel today:** `NormalizeIndexInputs` (tuple spread, sequence/mask coercion)
> + the `_NDArrayFound` `np.nonzero` expansion + ad-hoc `IndexError` throws. **Replace all of
> it with a single `PrepareIndex` that returns a typed op list + `IndexType` flags.**

### 4b. Routing on `index_type`  (`mapping.c` `MapIterNew` / `array_subscript`)
- Pure basic (`HAS_SLICE|HAS_NEWAXIS|HAS_ELLIPSIS|HAS_INTEGER`, **no** `HAS_FANCY`) → a **view**
  via the existing `Storage.GetView(Slice[])` (already correct — keep it).
- Any `HAS_FANCY` → the advanced path: **MapIter** (Stage 2).
- `HAS_BOOL` with `mask.ndim == arr.ndim` and nothing else → fast 1-D boolean (already correct,
  `TensorEngine.BooleanMask`).

### 4c. Stage 2 — `PyArray_MapIterNew` + `_get_transpose`
1. **Bools → nonzero** already happened in Stage 1, so now we have only integer advanced
   indices + basic (slice / newaxis / int).
2. **Broadcast ALL advanced indices together** → `fancy_dims` (`mit->fancy_dims`, the shared
   advanced block `bshape`). They share output axes — *not* an outer product with each other.
   Heterogeneous shapes must broadcast or raise `ValueError "shape mismatch"`.
3. **Each slice / newaxis keeps its own output axis** (outer product with the block).
4. **Consec / axis placement** (`mapping.c:62 _get_transpose`, `MapIterNew` ~`:2486–2513`):
   - advanced indices **consecutive** in the tuple → the `bshape` axes stay **in place**
     (where the block sits among the output axes);
   - a slice/newaxis **separates** the advanced indices → the `bshape` axes move to the
     **front**; slice axes follow in order.
   - `_get_transpose(fancy_ndim, consec, ndim, …)` builds the permutation. **For a get**:
     gather advanced-first, then transpose the block to `consec`. **For a set**: the inverse
     transpose on the value before scatter.
5. **Gather/scatter:** build one integer index array **per source axis** (advanced axes ←
   broadcast block; slice axes ← `GetIndicesFromSlice` reshaped to their own dim; int axes ←
   length-1), `broadcast_arrays` to the final grid, then a **single** `FetchIndices<T>` (get) /
   `SetIndices<T>` (set). These N-array primitives already exist and are correct.

> **NumSharp parallel today:** `TryBuildMultiAdvancedGrid` already implements (2)–(5) for the
> *≥2-advanced-with-a-slice* sub-case (see `advanced-index-axis-placement.md`). **Generalise it
> into the single MapIter entry** for *all* `HAS_FANCY` tuples, and delete the narrower
> `Try*` handlers (their cases become inputs to the same algorithm).

### 4d. Assignment specifics  (`mapping.c` `array_assign_subscript` ~`:3365`)
- Same index resolution; then the value is **broadcast to the advanced/grid result shape**
  (with the inverse `consec` transpose), and scattered.
- Value-shape mismatch → `ValueError "shape mismatch: value array of shape %S could not be
  broadcast to indexing result of shape %S"` (`mapping.c:3333`). NumSharp already emits the
  no-space tuple text in `UnmanagedStorage.SetData` and the fancy setter — reuse it.
- Overlap: NumPy resolves source/dest overlap (`COPY_IF_OVERLAP`). NumSharp's `SetIndices`
  copies offsets first, so simple overlap is safe; **add an explicit overlap copy** if the
  value aliases the destination through a view (probe `a[idx] = a[idx2]`).

---

## 5. Implementation plan (phased, each phase independently gateable)

### Phase A — lock the differential gate (do first, ~½ day)
The proof harness currently lives only in scratchpad. Make it a committed, CI-replayable gate
following the existing oracle pattern (`.claude/CLAUDE.md` → "Differential-Fuzz Pipeline",
`test/oracle/`, `test/NumSharp.UnitTest/Fuzz/`):
1. Move `scratchpad/gen_index_oracle.py` → `test/oracle/gen_index_oracle.py`. It already emits
   a portable **token** corpus (see §7) for getter+setter across 14 base recipes + a dtype
   sweep, with a seeded random-fuzz layer. Have it write `Fuzz/corpus/index_*.jsonl`.
2. Port `scratchpad/replay_index_oracle.cs` into a `[FuzzMatrix]` test (`Fuzz/IndexOracleTests.cs`):
   reconstruct base+index from tokens (the C# half already exists in the scratchpad file),
   bit-compare shape/values/raise. Reuse `BitDiff`/`FuzzCorpus` helpers.
3. Commit the corpus so CI replays it with **no Python at test time** (matches the existing
   `FuzzMatrix` gate). Keep the random layer's seed in the corpus filename for reproducibility.
4. **Definition of done for the whole project = this gate at 0 divergences on both the curated
   and the random corpora.**

### Phase B — kill the memory-safety crash (priority-0, ~1 day)
A flaky `AccessViolation` (heap corruption) survives even after the `998c1d23` over-index guard,
so there is a **second OOB write** in the mixed-advanced path. It is not the over-index case
(those now throw cleanly). Hunt:
1. Re-run the random sweep with the trace writer (scratchpad `replay_index_oracle.cs` already
   writes `trace.txt` flushed per case and `diverge.txt`). Corruption is **delayed** — the
   crash point ≠ the corruptor.
2. Make it deterministic: run with `DOTNET_gcServer=0 DOTNET_gcConcurrent=0` and
   `DOTNET_GCHeapHardLimit`, or enable .NET's `GCStress` / Windows AppVerifier page-heap on the
   test host so the OOB faults **at the write**, not later. That turns the flaky delayed crash
   into a deterministic stack at the offending `Buffer.MemoryCopy` / pointer store.
3. Suspects, in order: the fancy **scatter** (`SetIndicesND` / the `FetchIndicesNDNonLinear`
   odometer rewrite `34b40af9`) writing `dstAddr[baseDst+j]` when `retShape`/`subShape` are
   miscomputed for a **multi-dim fancy + slice** result; and `PrepareIndexGetters` reading
   `srcShape[i]`/`strides[i]` when `ndsCount` momentarily exceeds `ndim` in a fall-through combo.
4. The Phase C/D rewrite likely **subsumes** this (correct shapes ⇒ no OOB), but ship a guard
   fix immediately regardless — memory safety must not wait for the full port.

### Phase C — `PrepareIndex` (classifier + validator, ~1–2 days)
New file `Selection/NDArray.Indexing.PrepareIndex.cs`:
- `internal readonly struct IndexOp { IndexKind Kind; NDArray Array; Slice Slice; long IntVal; }`
  with `enum IndexKind { Ellipsis, NewAxis, Slice, Integer, FancyArray, BoolArray, ZeroDBool, ScalarArray }`.
- `internal static (IndexOp[] ops, IndexType flags, int srcAxesConsumed) PrepareIndex(Shape shape, object[] raw)`:
  expand ellipsis, classify every item (port the `mapping.c:319–667` switch), expand each
  bool array to its `nonzero()` integer arrays (k of them), validate too-many-indices and
  bool-array dimensionality, normalise scalars. **No gather yet** — pure analysis.
- Replace `NormalizeIndexInputs` callers with `PrepareIndex`. This alone removes most of the
  374 "accepts-invalid" and 200 "rejects-valid" divergences and gives uniform error messages.
- Gate: random sweep "accepts-invalid"/"rejects-valid" buckets → 0; curated still 0.

### Phase D — unified `MapIter` gather/scatter (~2–3 days)
- `internal NDArray MapIterGet(IndexOp[] ops, IndexType flags)` and
  `MapIterSet(IndexOp[] ops, IndexType flags, NDArray value)`.
- If pure basic → `Storage.GetView` (unchanged). Else run §4c: broadcast advanced block,
  compute consec + the `_get_transpose` permutation, build one index array per source axis,
  `broadcast_arrays`, single `FetchIndices`/`SetIndices`, apply the (inverse for set) transpose.
- **Delete** `TryFetchSliceWithSingleAdvanced`, `TrySetSliceWithSingleAdvanced`,
  `TryBuildMultiAdvancedGrid`, `TryFetchLeadingMaskWithBasic`, `TryBuild0dBoolWithBasic`, and
  the `_NDArrayFound` loop — their cases are now inputs to `MapIterGet/Set`. Keep
  `GetIndicesFromSlice`, `FetchIndices<T>`/`SetIndices<T>`, `BooleanMask(Set)` (the 1-D mask
  fast path), and `Storage.GetView`.
- Port `_get_transpose` faithfully (`mapping.c:62`); it is the subtlety that the patchwork
  never had. Unit-test it directly on the truth table in §2 + `advanced-index-axis-placement.md` §2.
- Gate: random sweep shape/value bucket → 0.

### Phase E — edge cases + setter overlap (~1 day)
- Empty advanced indices in combos (`arr([],(0,))`, empty bool) → empty result with correct
  trailing/slice axes (several fuzz divergences are here).
- 0-d bool **mixed with fancy** (currently `TryBuild0dBoolWithBasic` bails) — falls out
  naturally once 0-d bool is a length-1/0 fancy op in `PrepareIndex`.
- `HAS_SCALAR_ARRAY` (0-d integer array) placement vs python int.
- Assignment `COPY_IF_OVERLAP` (§4d) — add the overlap copy; probe `a[idx] = a[idx2]`.
- Gate: full random + curated 0/0, full CI net8.0 + net10.0, `FuzzMatrix`.

---

## 6. Code map — keep vs replace

| Symbol | File | Disposition |
|---|---|---|
| `FetchIndices(object[])` / `SetIndices(object[])` | `Selection/NDArray.Indexing.Selection.{Getter,Setter}.cs` | **Rewrite** to `PrepareIndex` → `MapIterGet/Set` |
| `NormalizeIndexInputs`, `SequenceToIndexArray` | Getter.cs | **Fold into** `PrepareIndex` (tuple spread + sequence coercion stay) |
| `TryFetchSliceWithSingleAdvanced`, `TrySetSliceWithSingleAdvanced` | Getter/Setter | **Delete** (subsumed) |
| `TryBuildMultiAdvancedGrid` | Getter.cs | **Promote/merge** into `MapIterGet/Set` (it is 70 % of the algorithm) |
| `TryFetchLeadingMaskWithBasic`, `TryBuildLeadingMaskBasicIndex` | Getter.cs | **Delete** (mask→nonzero in `PrepareIndex`) |
| `TryBuild0dBoolWithBasic` | Getter.cs | **Delete** (0-d bool becomes a fancy op) |
| `_NDArrayFound` loop | Getter/Setter | **Delete** |
| `GetIndicesFromSlice` | Selection.cs | **Keep** |
| `PrepareIndexGetters`, `NormalizeIndexArray` | Selection.cs | **Keep** (with the Phase-B bounds audit) |
| `FetchIndices<T>`, `FetchIndicesND`, `FetchIndicesNDNonLinear`, `SetIndices<T>`, `SetIndicesND` | Getter/Setter | **Keep** — the N-array gather/scatter kernels |
| `this[NDArray<bool>]`, `TensorEngine.BooleanMask(Set)` | Indexing.Masking.cs | **Keep** — full/prefix 1-D mask fast path |
| `Storage.GetView(Slice[])` (+ over-index check `9a2513c7`) | `UnmanagedStorage.Slicing.cs` | **Keep** — pure-basic path |

---

## 7. The differential harness (the gate)

Scratchpad now (move to `test/oracle/` in Phase A):
- `gen_index_oracle.py` — NumPy 2.4.2 oracle. Emits `index_corpus.json`:
  `{cases:[{op,base,tokens,value?,np:{ok,shape,vals}|{ok:false,err}}], dtype_cases:[…]}`.
  Curated matrix (basic/fancy/bool/0-d-bool/mixed × 14 bases) **+** seeded `random_fuzz(seed,
  7000, 3000)`. Current: 12265 cases (5748 ok / 6517 raise) + 104 dtype cases.
- `replay_index_oracle.cs` — `#:project NumSharp.Core` file-based app. Rebuilds the identical
  base by recipe name, the index by token, runs get/set, bit-compares to the recorded NumPy
  result. Writes `trace.txt` (per-case, flushed) and `diverge.txt`. Accepts `-- SKIP LIMIT`
  for bisection.

**Token encoding** (portable, both sides interpret identically):
`["int",n] ["slice",start,stop,step] ["new"] ["ell"] ["arr",flat,shape] ["barr",flatbool,shape]
["b0",bool] ["a0",n]`; value: `["scalar",n] | ["arr",flat,shape]`. Base recipes (mirrored in
both): `S,V0,V1,V6,A,AT,ARS,ACS,ANR,ANC,ASO,ABC,B,BT,E03` (arange-filled; views via the same
slice/transpose/broadcast ops). All data int64 ⇒ values compared exactly as int64; the dtype
sweep re-encodes per dtype (complex → re/im pairs, bool → 0/1, half → double).

**Run (current, from scratchpad):**
```bash
python gen_index_oracle.py                       # -> index_corpus.json
dotnet run replay_index_oracle.cs                # prints INDEX CASES x/y + categorised diffs
# bisection / crash hunt:
dotnet run replay_index_oracle.cs -- 6000 500    # process cases [6000,6500)
```
**Gotcha:** the file-based runfile cache (`%LOCALAPPDATA%\Temp\dotnet\runfile`) can serve a
stale `NumSharp.Core`. Clear it after a Core rebuild; trust only `dotnet test` / clean builds.

---

## 8. Memory-safety crash — what is known

- Manifests as exit 127 / no output / `AccessViolation` on the **random** sweep, **flaky**
  (some full runs finish: one printed `pass=11546 fail=719`).
- Delayed: `trace.txt` tail named `ANC[empty-bool]` and `V6[0,arr,arr]` on different runs, but
  **both are catchable in isolation** ⇒ the corruptor is an earlier OOB **write**, the crash is
  a later allocation hitting the corrupted heap.
- Therefore: deterministic page-heap / GCStress (Phase B step 2) is the way to catch it at the
  write. Do this before trusting any "it's fixed" — a flaky crash can hide behind a green run.

---

## 9. NumPy reference index (file:line)

`src/numpy/numpy/_core/src/multiarray/mapping.c`:
- `prepare_index` — **772**; classification switch **319–667**; `HAS_*` flags **319,335,348,380,510,559,594,609**.
- too-many-indices errors — **168, 303, 551, 667**.
- bool-array dim check — **1150**.
- `_get_transpose` (consec permutation) — **62**.
- `MapIterNew` consec computation — search `consec` ~**2486–2513**.
- get dispatch — **1520**; subscript — **1790/1874**; assign + "shape mismatch … could not be
  broadcast to indexing result of shape" — **3333**, assign dispatch **3365**.

NumPy docs: <https://numpy.org/doc/stable/user/basics.indexing.html> (Advanced indexing →
"Combining advanced and basic indexing" is the axis-placement rule in prose).

---

## 10. Definition of done / risks

**DOD**
- Committed index oracle (curated **and** random) replays **0 divergences**, both frameworks.
- `FuzzMatrix` + full CI (`TestCategory!=OpenBugs&!=HighMemory&!=LargeMemoryTest`) green on
  net8.0 + net10.0.
- The 2369-case curated matrix and all `Indexing.*ParityMatrixTests` stay green (no regression).
- No `AccessViolation` under page-heap/GCStress over the full random corpus.
- Setter parity incl. value-broadcast-to-grid and overlap.

**Risks / gotchas**
- This touches the hottest core path; gate **every** phase behind the full sweep before landing.
- `_get_transpose` is the subtle bit — unit-test the permutation directly, not just end-to-end.
- Keep the pure-basic `GetView` path (it returns **views**; advanced returns **copies** — the
  copy/view contract is tested in `LayoutValueParity` C1–C4, don't break it).
- 0-d-bool merge semantics: multiple 0-d bools broadcast into ONE length-(∏ truth) axis
  (`A[True,True]`→`(1,3,4)`, `A[True,False]`→`(0,3,4)`); preserve when 0-d bool becomes a fancy op.
- `uint64`/`ulong` scalar and `int8` fancy dtype quirks are already fixed (`9384f7f8`,
  `391cc415`) — keep `NormalizeIndexArray`'s dtype widening.

---

## 11. First-day checklist

1. Read this doc + `advanced-index-axis-placement.md` + `mapping.c:62`, `:772`.
2. Phase A: commit the oracle (`test/oracle/gen_index_oracle.py`, corpus, `Fuzz/IndexOracleTests.cs`).
   Confirm it reproduces ~660–700 random divergences and 0 curated. **This is the gate.**
3. Phase B: page-heap the random sweep, capture the OOB write stack, fix it (memory safety).
4. Phase C: `PrepareIndex`; re-run sweep, watch the accepts-invalid / rejects-valid buckets drop.
5. Phase D: `MapIterGet/Set` + `_get_transpose`; delete the `Try*` stack; sweep → 0.
6. Phase E: edges + overlap; full CI both frameworks; remove this doc's OPEN status.
