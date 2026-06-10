# NpyIter — Gap Analysis vs NumPy 2.4.2 & Performance Roadmap

**Date:** 2026-06-10 · **Branch:** `nditer` · **Machine:** i9-13900K (AVX2) · **NumPy:** 2.4.2
**Method:** source audit of `Backends/Iterators/*` against `src/numpy/numpy/_core/src/{multiarray/nditer_*, umath/*}` + measured variation grid (`benchmark/poc/variation_probe.{cs,py}`, Release, back-to-back). All numbers below are Release (`dotnet run -c Release`) — see handover §4.12 for why Debug numbers are invalid.

**Goal restated:** NpyIter becomes the **global iteration core** — every `np.*` loop (element-wise, reduction, selection, copy/cast) is one per-chunk kernel driven by the iterator, with the iterator owning layout/broadcast/cast/mask/overlap/parallel concerns once, centrally.

---

## 1. Executive summary

The element-wise `np.*` surface is **already at or beyond NumPy on nearly every measured layout** — contiguous, broadcast (row/col/scalar), negative-stride, F-order, transposed, mixed-dtype, 5-D, `where`, f32 axis sums, `mean`. Three holes remain, all precisely characterized:

1. **Correctness: overlapping operands silently corrupt** (NumPy: `COPY_IF_OVERLAP`; us: nothing — proven by probe).
2. **Performance: genuinely-strided shapes run scalar** inside the Tier-3B shell — 3.0–7.1× behind NumPy (S1/S2/S3), while this session's POC kernels prove 1.3–1.9× *ahead* of NumPy is reachable on the same shapes (hardware gather).
3. **Performance: small-N dispatch** — 1.34 µs vs NumPy 0.42 µs per call at N=1K (3.2×). The raw iterator costs 0.40 µs; the other ~0.9 µs is `np.*` glue: result allocation, `MultiNew` argument arrays, routing-ladder overhead, and **no `out=` anywhere in the API**.

Plus the architectural inversion: NpyIter today is the **4th choice** in the dispatch ladder (trivial-bypass → fused-1D → buffered → NpyIter) and the typed `Execute*` layer hands the whole array to legacy Direct kernels (iterator as setup helper). Fusion (`ExecuteExpression`, proven 2.8–5.4× faster than NumPy) has **zero production callers**.

---

## 2. Measured state — variation grid (production `np.*`, end-to-end, both sides allocate)

| probe | shape / op (f32 unless noted) | NumSharp | NumPy | ratio | verdict |
|---|---|--:|--:|--:|---|
| P1 | contig binary `a+b` 4M | 3.65 ms | 3.52 ms | 1.04 | parity |
| P2 | row broadcast `(2k,2k)+(1,2k)` | 3.25 ms | 3.23 ms | 1.01 | parity |
| P3 | col broadcast `(2k,2k)+(2k,1)` | 2.94 ms | 3.07 ms | 0.96 | faster |
| P4 | scalar broadcast `a+5` | 2.84 ms | 2.97 ms | 0.96 | faster |
| P5 | neg-stride unary `sqrt(a[::-1])` | 2.83 ms | 2.83 ms | 1.00 | parity |
| P6 | neg-stride binary `a[::-1]+b[::-1]` | 3.85 ms | 3.47 ms | 1.11 | ≈parity |
| P7 | F-order binary `aF+bF` | 3.05 ms | 3.36 ms | 0.91 | faster |
| P8 | transposed binary `a.T+b.T` | 3.05 ms | 3.29 ms | 0.93 | faster |
| P9 | mixed dtype `i32+f64` 4M | 5.83 ms | 7.26 ms | 0.80 | faster |
| P10 | astype strided `a[::2]→f64` | 5.92 ms | 6.15 ms | 0.96 | parity |
| P11 | `where(cond,x,y)` contig 4M | 3.18 ms | 3.82 ms | 0.83 | faster |
| P12 | `sum(axis=0)` f32 (2k,2k) | 0.265 ms | 0.309 ms | 0.86 | faster |
| P13 | `sum(axis=1)` f32 (2k,2k) | 0.211 ms | 0.780 ms | 0.27 | 3.7× faster |
| P14 | 5-D contig unary sqrt | 2.76 ms | 2.91 ms | 0.95 | faster |
| **P15** | **small-N binary 1K** | **1.34 µs** | **0.42 µs** | **3.2×** | **GAP** |
| P16 | `mean` f32 contig 4M | 0.251 ms | 0.712 ms | 0.35 | 2.8× faster |
| **S1** | **strided binary `a[::2]+b[::2]` 1M** | **1264 µs** | 416 µs | **3.0×** | **GAP** (POC kernel: 319 µs) |
| **S2** | **strided 2-D `sqrt(a[::2,::2])` 1M** | **2654 µs** | 374 µs | **7.1×** | **GAP** (POC kernel: 206 µs) |
| **S3** | **strided `sum(a[::2])` 1M** | **371 µs** | 205 µs | **1.8×** | **GAP** (POC kernel: 109 µs) |

Notes:
- P8/P7 are fast because transposes of C-contig are F-contig → KEEPORDER coalesces them to one contiguous run. The *genuinely* strided shapes are S1/S2/S3 — exactly where the Tier-3B shell falls to `EmitScalarStridedLoop` (S2 ≈ 12 cycles/element).
- P13/P16: the previously-reported "f32 axis-tier gap 4.6×" is **refuted under Release** — another Debug-tainted number (handover §4.12). f32 axis reductions are *faster* than NumPy.
- **Correctness probe (overlap, write-ahead direction):** `add(a[:-1], a[:-1], out=a[1:])` → NumSharp `[1,2,4,6,8,16,32,64]` (cascade through clobbered reads at the vector boundary), NumPy `[1,2,4,6,8,10,12,14]`. **Silent corruption.**
- POC reference (same iterator, hand-written per-chunk hw-gather kernels, `benchmark/poc/POC_RESULTS.md`): every aspect at-or-faster than NumPy, fusion 2.8–5.4× faster.

---

## 3. Capability matrix — our NpyIter vs NumPy's nditer

From source audit (`NpyIter*.cs` vs `nditer_constr.c`/`nditer_api.c`/`ufunc_object.c`).

### Implemented and real
| capability | state |
|---|---|
| Multi-operand broadcast construction, `MultiNew`/`AdvancedNew` | ✅ incl. `op_axes`, `iterShape`, `ALLOCATE` (null operand + dtype), `COMMON_DTYPE` |
| Order resolution C/F/A/K + negative-stride flipping (KEEPORDER) | ✅ `npyiter_flip_negative_strides` equivalent |
| Axis coalescing | ✅ `NpyIterCoalescing` (size-1 absorption + stride-compat merge) |
| EXTERNAL_LOOP / ONEITERATION / per-element iternext | ✅ (EXLOOP over-advance fixed in Phase 0) |
| Buffering + cast (`NpyIterBufferManager`, 8192 default = NumPy's `NPY_BUFSIZE`) | ✅ via execution bridge; **latent Advance bug (b)** |
| GROWINNER (buffered) | ✅ `CalculateGrowInnerSize` |
| REDUCE operands, `IsFirstVisit`, `BufferedReduce` double loop | ✅ |
| Casting matrix (`CanCast` safe/same_kind) | ✅ 338/338 NumPy-identical |
| API surface: `GetInnerFixedStrideArray`, `RemoveAxis`, `RemoveMultiIndex`, `EnableExternalLoop`, `Copy`, `GotoIterIndex`, `ResetToIterIndexRange`, multi-index delegates | ✅ exists |
| WRITEMASKED/ARRAYMASK pairing validation, `MaskOp` tracking | ✅ construction only |

### Declared but hollow (the capability gaps)
| capability | state | NumPy behavior |
|---|---|---|
| **COPY_IF_OVERLAP / overlap detection** | ❌ **nothing anywhere** (proven corruption) | ufuncs always pass it; `mem_overlap.c` bounds-check + Diophantine solver |
| **WRITEMASKED execution** | ❌ masks ignored at execute/copy-back | masked transfer functions skip masked-off elements |
| **VIRTUAL operands** | ❌ debug-print only | buffer-only operand, no backing array |
| **RANGED iteration** | ⚠️ machinery exists (`ResetToIterIndexRange`), **no construction flag handling, no consumer** | `NpyIter_ResetToIterIndexRange` — basis for threaded ufuncs in downstream libs |
| **DELAY_BUFALLOC** | ❌ flag ignored — buffers allocate at construction | ufunc default; buffers allocate on first `Reset` |
| **REUSE_REDUCE_LOOPS** | ❌ debug-print only | reduce double-loop setup reused across buffer fills |
| NumSharp-extension flags `CONTIGUOUS`(set)/`GATHER_ELIGIBLE`/`EARLY_EXIT`/`PARALLEL_SAFE` | ❌ `GATHER_ELIGIBLE`/`EARLY_EXIT`/`PARALLEL_SAFE` never set or read | n/a (our own additions — wire or delete) |

### Architectural deltas vs NumPy's ufunc layer
| concern | NumPy | NumSharp today |
|---|---|---|
| ufunc iterator config | `EXTERNAL_LOOP \| BUFFERED \| GROWINNER \| DELAY_BUFALLOC \| COPY_IF_OVERLAP \| ZEROSIZE_OK \| REFS_OK` | `EXTERNAL_LOOP` only |
| mixed-dtype element-wise | buffered cast → **same-type SIMD** inner loop | per-element convert in scalar body (no SIMD) — yet still 0.80× (P9); buffering would extend the win |
| inner-loop selection | per-call `get_loop(fixed_strides)` → contig/strided/scalar specialization | kernel cache keyed by dtype+`DetectExecutionPath` (coarser but cached — fine) |
| trivial small-N path | `check_for_trivial_loop` bypasses iterator, same inner-loop fn either way | trivial bypass exists but runs a *different kernel family* (Direct whole-array); glue costs 3.2× at 1K |
| reductions | `PyUFunc_ReduceWrapper` → nditer (`op_axes`, REDUCE_OK, buffered) | axis reductions bypass NpyIter entirely (Direct-only; `NpyAxisIter` scalar for var/std/cum*) |
| `out=` / `where=` kwargs | every ufunc | **absent from the entire np.\* API** |
| iteration as THE core | one driver | NpyIter is 4th in the dispatch ladder; typed `Execute*` hands whole arrays to legacy Direct kernels |

---

## 4. Variation-coverage map (/np-function grid → status)

**A. Single-array layouts (25):** C-contig ✅ · F-contig ✅ · strided 1-D unary ✅ (pre-iterator fused rescue) · **strided 1-D binary ❌ S1** · **≥2-D strided ❌ S2** · transposed ✅ (F-contig coalesce) · negative-stride ✅ · simple slice ✅ · **sliced+composed ≥2-D ❌ (S2 class)** · broadcast ✅ · scalar-broadcast ✅ · partial broadcast ✅ · 0-d/1-element ⚠️ (small-N tax) · empty ✅ · NewAxis/singleton ✅ (coalesced) · 5+D ✅ (P14) · stride>bufferSize ⚠️ untested edge · reshape view/copy ✅ · fancy/bool-mask results ✅ · read-only broadcast ✅ · non-owning ✅ · aligned ✅.

**B. Pairwise paths (6):** SimdFull ✅ · SimdScalarLeft/Right ✅ · SimdChunk ✅ (inner-contig runtime dispatch) · **General ❌ scalar (S1/S2)** · mixed dtypes ⚠️ (correct, fast-ish, but scalar body — no SIMD).

**C. Per-operand (8):** **aliased/overlapping ❌❌ silent corruption** · **in-place `out=` ❌ no API** · REDUCE operand ✅ · **WRITEMASKED ❌ exec missing** · **VIRTUAL ❌ stub** · buffered/cast ⚠️ (works via bridge; Advance bug latent) · read-only ✅.

**D. Iteration flags (8):** coalesced ✅ · IDENTPERM/NEGPERM ✅ · EXLOOP ✅ · **RANGED ⚠️ unused machinery** · GROWINNER ✅ buffered-only · **GATHER_ELIGIBLE ❌ unwired** · EARLY_EXIT ⚠️ (ExecuteReducing early-exits; flag itself dead) · **PARALLEL_SAFE ❌ unwired**.

**E. Composite (4):** src-broadcast+dst-contig ✅ · src-contig+dst-strided ⚠️ (store is scalar lane-by-lane) · buffer-required ✅ (bridge) · **REUSE_REDUCE_LOOPS ❌ stub**.

---

## 5. Proposed changes — prioritized waves

Each wave is independently shippable, gated by `variation_probe.{cs,py}` + the full suite (9477).

### Wave 1 — Correctness (blockers for "global core" status)
| # | change | files | evidence/gate |
|---|---|---|---|
| 1.1 | **Overlap detection + COPY_IF_OVERLAP.** Port NumPy's `mem_overlap.c` bounds-interval check (conservative first; exact Diophantine later). Honor the flag in `MultiNew`; pass it from the production binary/unary/compare routes like NumPy's ufunc layer does; add `OVERLAP_ASSUME_ELEMENTWISE` short-circuit (the common `out=input` identity case must NOT copy). | new `NpyIterOverlap.cs`; `NpyIter.cs` construction | overlap probe in `variation_probe.cs` returns NumPy-identical; overlap test matrix (forward/backward/partial/self) |
| 1.2 | **Buffered-cast Advance fix (bug b).** Advance buffered operands by `BufStrides`, not `Strides×ElementSizes`. Prereq for Wave 4. | `NpyIter.State.cs:730`, `NpyIter.cs:1511/1521` | new multi-fill (>8192) buffered-cast test |
| 1.3 | **WRITEMASKED/ARRAYMASK execution + VIRTUAL operands.** Masked copy-back in `NpyIterBufferManager`, masked write contract in `ForEach`/Tier-3B; VIRTUAL = buffer-only operand. Unlocks `np.place`/`copyto(where=)`/`np.where` migration onto one driver. | `NpyIterBufferManager.cs`, `NpyIter.Execution.cs` | masked-write parity tests vs NumPy |
| 1.4 | Hygiene: CA2014 stackalloc-in-loop (`NpyIterCoalescing.cs:244`); audit coalescing size-1-axis stride rule; delete-or-wire the dead extension flags. | — | suite |

### Wave 2 — Small-N dispatch (P15: 1.34 → ≤0.5 µs target)
| # | change | rationale |
|---|---|---|
| 2.1 | **`out=` (and `where=`) parameters on np.* ufuncs** (`np.add(a, b, out: r)` + operators keep allocating). Kills the result allocation on hot paths, required for NumPy API parity, and is the idiomatic answer to the .NET allocation tax (POC finding: fresh 4 MB ≈ +0.3–0.4 ms in page faults). Depends on 1.1 for safety when `out` aliases an input. | API parity + biggest single small-N lever |
| 2.2 | **Zero-alloc `MultiNew`**: `ReadOnlySpan<NDArray>`/`Span<NpyIterPerOpFlags>` overloads; eliminate the per-call `new[]{...}` at every call site; cached delegate instances for `ForEach` kernels (no method-group→delegate alloc per call). | each np.* call today allocates 2–3 helper arrays + 1 delegate |
| 2.3 | **Phase 1 trivial constructor** (`TryNewTrivial`: NOp ≤ 3, all-contig, C-order, no cast → fill minimal state, skip coalescing/order resolution) + optional iterator-state pooling keyed (NOp, NDim). Raw construct+exec is already 0.40 µs; this targets the remaining glue. | NumPy's `check_for_trivial_loop` applied to construction |
| 2.4 | **Small-buffer reuse in UnmanagedStorage** (pool or eager-free-on-Dispose so freed pages stay warm). | the allocator finding affects every benchmark and real workload |

### Wave 3 — Strided roofline (S1/S2/S3: 3–13× headroom; kernels already proven)
| # | change | target |
|---|---|---|
| 3.1 | **Phase 2a: `EmitFusedStridedSimdLoop`** at the `lblScalarStrided` site (`DirectILKernelGenerator.InnerLoop.cs:304`): **AVX2 hardware-gather body** for f32/f64/i32/i64 (index vector hoisted, scale=1 byte offsets, guard `Avx2.IsSupported && |7·stride| ≤ int.MaxValue`), insert-gather fallback otherwise. Transcribe `PocKernels.AddF32/SqrtF32/SumF32`; unrolls: binary 2×, unary 4×, reduce 4 accumulators. | S1 1264→~330 µs, S2 2654→~210 µs (beats NumPy) |
| 3.2 | Wire `GATHER_ELIGIBLE` at construction; include in the Tier-3B kernel cache key so gather/insert/scalar bodies are selected per layout class. | — |
| 3.3 | **Strided store** (dest-strided): NumPy's `npyv_storen` analog (scalar lane stores from the vector) so CONTIG→NCONTIG and NCONTIG→NCONTIG vectorize too (composite class E2). | `a[::2] = sqrt(b[::2])` shapes |
| 3.4 | Strided **reduce** body via hw-gather + 4 accumulators in `ExecuteReduction`'s strided kernel. | S3 371→~110 µs |

### Wave 4 — NumPy-default iterator config for ufuncs
| # | change | rationale |
|---|---|---|
| 4.1 | Implement **DELAY_BUFALLOC** (defer buffer allocation to first Reset/use). | prereq: buffered-by-default must not tax small-N |
| 4.2 | Switch the production element-wise route to NumPy's config: **`EXTERNAL_LOOP \| BUFFERED \| GROWINNER \| DELAY_BUFALLOC \| COPY_IF_OVERLAP`** (after 1.1/1.2). Mixed-dtype then runs buffered-cast → same-type SIMD inner loop instead of the per-element-convert scalar body. | P9 0.80× → est. 0.5–0.6×; covers unaligned/exotic layouts uniformly |

### Wave 5 — Reductions through the core
| # | change | rationale |
|---|---|---|
| 5.1 | Route **var/std/cumsum/cumprod/all/any axis paths** (today scalar `NpyAxisIter`) through `NpyIterRef` + Tier-3B kernels (`op_axes` + REDUCE_OK). Keep the 2b widening kernels as inner loops for sum/prod/min/max. | the remaining scalar axis paths; one driver |
| 5.2 | Implement **REUSE_REDUCE_LOOPS** (cache the reduce double-loop schedule across buffer fills). | NumPy-parity for buffered reductions |
| 5.3 | Retire `NpyAxisIter` once 5.1 lands. | dead machinery |

### Wave 6 — Exceed NumPy (architecture dividends)
| # | change | evidence |
|---|---|---|
| 6.1 | **Expose fusion**: `np.evaluate(expr)` or lazy expression operators → `ExecuteExpression`. Make `ExecuteExpression` set/assert EXTERNAL_LOOP internally (the 38× foot-gun). Fix mixed-dtype tree promotion to NumPy `result_type` semantics; add reduction nodes (`sum(a*b)` one-pass). | measured 2.8–5.4× faster than NumPy (POC F/G), zero production callers today |
| 6.2 | **Parallel ForEach** over the outer dimension: `RANGED` + `Copy()` per worker, gated on `PARALLEL_SAFE` (no overlap → needs 1.1; no REDUCE, or REDUCE with per-worker accumulators), `IterSize ≥ threshold`, honoring the existing `np.multithreading` toggle. NumPy's nditer has the API but NumPy itself never threads it — **pure exceed**, est. 2–6× on multi-MB ops (this box: 8 P-cores). | `ResetToIterIndexRange`/`GotoIterIndex`/`Copy` already exist |
| 6.3 | Continue **Phase 3 migration** per family (binary → comparison → unary → scan → copy → Modf → Where/Place via 1.3) so the Direct partials retire and every op inherits Waves 1–5 automatically. | handover §8 |

### Expected end-state vs the /np-function bar ("≥1.5× NumPy across variations")
Already ≥1.5×: P13/P16-class reductions, fusion (when exposed), strided shapes after Wave 3, parallel large-N after 6.2. At parity (1.0–1.1×, DRAM-bound — physics, not code): P1–P8 class. The bar is unreachable only where memory bandwidth is the roofline NumPy also sits on; everywhere else the waves above target it.

---

## 6. Verification harness

```bash
# the gate for every wave (run both, compare ratios):
dotnet run -c Release - < benchmark/poc/variation_probe.cs
python benchmark/poc/variation_probe.py

# POC reference numbers (iterator + proven kernels):
dotnet run -c Release - < benchmark/poc/npyiter_parity_poc.cs
python benchmark/poc/npyiter_parity_poc.py

# suite gate
cd test/NumSharp.UnitTest && dotnet test --no-build -f net10.0 --filter "TestCategory!=OpenBugs&TestCategory!=HighMemory"
```

The overlap probe inside `variation_probe.cs` must print NumPy's `[1 2 4 6 8 10 12 14]` once Wave 1.1 lands — today it prints the corrupted cascade.
