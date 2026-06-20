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

Plus the architectural inversion: NpyIter today is the **4th choice** in the dispatch ladder (trivial-bypass → fused-1D → buffered → NpyIter) and the typed `Execute*` layer hands the whole array to legacy Direct kernels (iterator as setup helper). ~~Fusion (`ExecuteExpression`, proven 2.8–5.4× faster than NumPy) has **zero production callers**.~~ **Fusion shipped (Wave 6.1): `np.evaluate` — 3.2–6.1× faster than NumPy on 4M chains, NumPy result_type per node.**

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
| WRITEMASKED/ARRAYMASK pairing validation, `MaskOp` tracking | ✅ full (Wave 1.3: + masked copy-back execution, NumPy error-text parity) |

### Declared but hollow (the capability gaps)
| capability | state | NumPy behavior |
|---|---|---|
| **COPY_IF_OVERLAP / overlap detection** | ✅ **implemented (Wave 1.1, 2026-06-10)** — `NpyMemOverlap.cs` full solver port + FORCECOPY/write-back; production element-wise routes pass the flag | ufuncs always pass it; `mem_overlap.c` bounds-check + Diophantine solver |
| **WRITEMASKED execution** | ✅ **implemented (Wave 1.3, 2026-06-10)** — masked copy-back in `FlushBufferWindow` (mask-run decomposition keeps memcpy/SIMD-cast per TRUE stretch); enforcement ONLY at buffered copy-back, NumPy-verified (unbuffered/BUFNEVER ops write directly — kernel contract) | masked transfer functions skip masked-off elements |
| **VIRTUAL operands** | ✅ **implemented (Wave 1.3)** — NumPy 2.x allocate-equivalent semantics (verified: `npyiter_allocate_arrays` allocates for EVERY null op; the NEP-12 buffer-only design never landed; requested dtype DISCARDED → common dtype of real operands; ARRAYMASK default bool; `virtual+allocate` → ALLOCATE wins) | NumPy 2.4.2: allocate-equivalent, dtype request discarded |
| **RANGED iteration** | ⚠️ machinery exists (`ResetToIterIndexRange`), **no construction flag handling, no consumer** | `NpyIter_ResetToIterIndexRange` — basis for threaded ufuncs in downstream libs |
| **DELAY_BUFALLOC** | ✅ **implemented (Wave 4)** — construction defers; `Reset()`/`EnsureBuffersReady()` materialize (NumPy errors without reset; NumSharp auto-ensures at the execution entry points) | ufunc default; buffers allocate on first `Reset` |
| **REUSE_REDUCE_LOOPS** | ❌ debug-print only | reduce double-loop setup reused across buffer fills |
| **COMMON_DTYPE promotion** | ⚠️ **NEP50 divergence found (Wave 6.1, unfixed)** — `NpyIterCasting.PromoteTypes` says "float always wins over int", so i4+f4→f4 / i8+f4→f4 / i2+f2→f2; NumPy result_type promotes the int to its tier float first (i4+f4→**f8**, i2+f2→**f4**). Affects COMMON_DTYPE iterators and VIRTUAL common-dtype resolution only (the engine's binary routes use the correct `np._FindCommonType` tables; `np.evaluate` uses its own pinned `NpyExprTypeRules.PromoteStrong`). Fix when a COMMON_DTYPE consumer lands. | `PyArray_ResultType` (NEP50) |
| NumSharp-extension flags `CONTIGUOUS`/`GATHER_ELIGIBLE`/`PARALLEL_SAFE` | ✅ **resolved (Wave 1.4, 2026-06-10)** — `PARALLEL_SAFE` wired (construction sets it for no-REDUCE + ≤1 WRITE operand with COPY_IF_OVERLAP-resolved overlap; exposed as `IsParallelSafe` for Wave 6.2); `EARLY_EXIT` deleted (early exit is a kernel property — `SupportsEarlyExit`/`ShouldExit` — not iterator state) | n/a (our own additions) |

### Architectural deltas vs NumPy's ufunc layer
| concern | NumPy | NumSharp today |
|---|---|---|
| ufunc iterator config | `EXTERNAL_LOOP \| BUFFERED \| GROWINNER \| DELAY_BUFALLOC \| COPY_IF_OVERLAP \| ZEROSIZE_OK \| REFS_OK` | `EXTERNAL_LOOP` only |
| mixed-dtype element-wise | buffered cast → **same-type SIMD** inner loop | **resolved (Wave 4)**: both architectures available; A/B says fused per-element convert WINS for cheap ops (binary/compare keep it, 0.79× vs NumPy), buffered-cast wins for promoting unary math (sqrt/exp-class, 0.81× vs NumPy) |
| inner-loop selection | per-call `get_loop(fixed_strides)` → contig/strided/scalar specialization | kernel cache keyed by dtype+`DetectExecutionPath` (coarser but cached — fine) |
| trivial small-N path | `check_for_trivial_loop` bypasses iterator, same inner-loop fn either way | trivial bypass exists but runs a *different kernel family* (Direct whole-array); glue costs 3.2× at 1K |
| reductions | `PyUFunc_ReduceWrapper` → nditer (`op_axes`, REDUCE_OK, buffered) | axis reductions bypass NpyIter entirely (Direct-only; `NpyAxisIter` scalar for var/std/cum*) |
| `out=` / `where=` kwargs | every ufunc | ✅ **shipped (Wave 2.1)** on the elementwise core (8 binary + 10 unary) + `np.evaluate(…, out:)` (Wave 6.1) |
| iteration as THE core | one driver | NpyIter is 4th in the dispatch ladder; typed `Execute*` hands whole arrays to legacy Direct kernels |

---

## 4. Variation-coverage map (/np-function grid → status)

**A. Single-array layouts (25):** C-contig ✅ · F-contig ✅ · strided 1-D unary ✅ (pre-iterator fused rescue) · strided 1-D binary ✅ **fixed (Wave 3 hw-gather, kernel 318 µs)** · ≥2-D strided ✅ **fixed (Wave 3: 941 vs NumPy 1282 e2e)** · transposed ✅ (F-contig coalesce) · negative-stride ✅ · simple slice ✅ · sliced+composed ≥2-D ✅ **fixed (Wave 3)** · broadcast ✅ · scalar-broadcast ✅ · partial broadcast ✅ · 0-d/1-element ⚠️ (small-N tax) · empty ✅ · NewAxis/singleton ✅ (coalesced) · 5+D ✅ (P14) · stride>bufferSize ⚠️ untested edge · reshape view/copy ✅ · fancy/bool-mask results ✅ · read-only broadcast ✅ · non-owning ✅ · aligned ✅.

**B. Pairwise paths (6):** SimdFull ✅ · SimdScalarLeft/Right ✅ · SimdChunk ✅ (inner-contig runtime dispatch) · General ✅ **hw-gather for 32/64-bit dtypes (Wave 3); scalar fallback otherwise** · mixed dtypes ⚠️ (correct, fast-ish, but scalar body — no SIMD).

**C. Per-operand (8):** aliased/overlapping ✅ **fixed (Wave 1.1: COPY_IF_OVERLAP + write-back)** · in-place `out=` ✅ **shipped (Wave 2.1 ufunc core; Wave 6.1 np.evaluate)** · REDUCE operand ✅ (Wave 1.4 adds NumPy's stretched-write validation: REDUCE_OK + readable required, REDUCE flags set) · WRITEMASKED ✅ **exec implemented (Wave 1.3: masked flush, full validation parity; buffered-REDUCE write-back refuses loudly → Wave 5)** · VIRTUAL ✅ **implemented (Wave 1.3: allocate-equivalent, NumPy 2.x verified)** · buffered/cast ✅ (Wave 4 windowed machinery; Advance bug (b) fixed Wave 1.2, remaining family sites fixed Wave 1.4) · read-only ✅.

**D. Iteration flags (8):** coalesced ✅ (Wave 1.4: NumPy-strict size-1 rule on the fill_axisdata stride-0 invariant) · IDENTPERM/NEGPERM ✅ · EXLOOP ✅ · **RANGED ⚠️ unused machinery** · GROWINNER ✅ buffered-only · GATHER_ELIGIBLE ✅ **wired (Wave 3; informational — kernels self-dispatch)** · EARLY_EXIT ✅ **flag deleted (Wave 1.4** — early exit lives at the kernel level: `SupportsEarlyExit`/`ShouldExit`**)** · PARALLEL_SAFE ✅ **wired (Wave 1.4:** construction sets it for no-REDUCE + ≤1 COPY_IF_OVERLAP-resolved WRITE; `IsParallelSafe` ready for 6.2**)**.

**E. Composite (4):** src-broadcast+dst-contig ✅ · src-contig+dst-strided ✅ **(Wave 3 scatter store: vectorized compute, per-lane stores)** · buffer-required ✅ (bridge) · **REUSE_REDUCE_LOOPS ❌ stub**.

---

## 5. Proposed changes — prioritized waves

Each wave is independently shippable, gated by `variation_probe.{cs,py}` + the full suite (9477).

### Wave 1 — Correctness (blockers for "global core" status)
| # | change | files | evidence/gate |
|---|---|---|---|
| 1.1 | ✅ **DONE (2026-06-10).** **Overlap detection + COPY_IF_OVERLAP** — full port of NumPy's `mem_overlap.c` (`NpyMemOverlap.cs`: extent fast path + GCD-pruned bounded-Diophantine DFS with Int128 intermediates, maxWork semantics 0/-1/N), FORCECOPY + write-back-on-Dispose in `NpyIter.cs` (nditer_constr.c:3083-3311 parity incl. the `OVERLAP_ASSUME_ELEMENTWISE` exact-alias short-circuit + internal-overlap check), flags wired into the production binary/unary/compare routes. Solver validated 14/14 against `np.shares_memory`/`np.may_share_memory`; behaviors B1–B7 NumPy-identical; 13 tests in `NpyIterOverlapTests.cs`; suite 9490 green; no perf regression (small-N 1.37 µs, fresh outputs cost one extent check). **Bonus bug found & guarded:** the Layer-2 typed helpers (`ExecuteBinary`/`ExecuteUnary`/`ExecuteComparison`/`ExecuteScan`) bridge to whole-array kernels that IGNORE output strides — a strided write operand was silently written contiguously; now throws `InvalidOperationException` (proper fix = Phase 3 per-chunk migration; Tier-3B route was always correct). Note: write-back resolves at `Dispose` (NumPy WRITEBACKIFCOPY semantics) — consume results after the `using` scope. |
| 1.2 | ✅ **DONE (2026-06-10, with Wave 4).** **Buffered-cast Advance fix (bug b)** — all five array-traversal sites (`Advance`, `GotoIterIndex`, `ExternalLoopNext`, `GotoIndex`, `GotoMultiIndex`) now multiply source-element strides by **`SrcElementSizes`** (the buffer-dtype `ElementSizes` made every reposition 2× under an int32→float64 cast). Gated by the multi-fill (>8192) tests in `NpyIterBufferedWindowTests.cs`. |
| 1.3 | ✅ **DONE (2026-06-10).** **WRITEMASKED/ARRAYMASK execution + VIRTUAL operands.** Probing NumPy 2.4.2 first settled the architecture: **masking is enforced in exactly ONE place — the buffered copy-back** (`npyiter_copy_from_buffers`, nditer_api.c:2001-2026); unbuffered WRITEMASKED operands write the array directly (kernel contract), and NumPy 2.x BUFNEVER means `'buffered'` + contiguous same-dtype operands ALSO skip enforcement — so no driver/Tier-3B changes were needed at all, just the flush. Shipped: **(a) masked copy-back** — `FlushBufferWindow` dispatches WRITEMASKED ops to `CopyWindowFromBufferMasked` (same run-decomposed window walk; mask cursor rides the flat window counter; mask read from the mask's buffer when buffered else from its window-start array ptr — the BUFNEVER switch, nditer_api.c:2009-2014; per run, `CopyRunFromBufferMasked` decomposes into TRUE stretches handed to the unmasked run copier, NumPy's `_strided_masked_wrapper` structure, so memcpy/SIMD-cast kernels survive dense masks; stride-0 broadcast mask gates whole runs). **(b) full validation parity, NumPy texts verbatim** — WRITEMASKED-requires-WRITE; VIRTUAL-requires-READWRITE (incl. NumPy's doubled "be"); null-op-requires-ALLOCATE∥VIRTUAL; VIRTUAL-requires-null; ARRAYMASK-can't-reduce (standard + op_axes paths); the WRITEMASKED∧REDUCE mask-broadcast check moved to a **deferred post-stride loop** (NumPy defers identically, tail of `npyiter_allocate_arrays` c:3351-3370 — the old inline call only covered op_axes; the standard broadcast path was unchecked); `"Only bool and uint8 masks are supported."` at buffer allocation when a WRITEMASKED op actually buffers (mask array AND buffer dtype ∈ {bool,uint8}; unbuffered non-bool masks construct fine, NumPy parity). **(c) VIRTUAL** — NumPy 2.x reality (source + probes): allocate-equivalent (real backing array for every null op; NEP-12 never landed; `NPY_OP_ITFLAG_VIRTUAL`'s only consumer is DebugPrint), requested dtype DISCARDED → common dtype over real operands, ARRAYMASK virtual defaults bool, `virtual+allocate` → ALLOCATE wins (dtype honored); flag mapped through `TranslateOpFlags`. **(d) buffered-REDUCE + WRITEMASKED**: construction succeeds (NumPy accepts the aligned-mask pattern) but the legacy reduce write-back (`CopyReduceBuffersToArrays`) refuses loudly instead of silently writing unmasked slots — masked reduce flush lands with Wave 5 (`[Misaligned]`-tagged test pins this). **Perf**: the expanded per-op validation cost +11.9 ns/ctor on the small-N path (targeted `ctor_probe.cs` interleaved A/B vs 7c8e0588) — eliminated by an OR-sweep fast-path gate (plain constructions skip the full loop) + single-mask-test virtual detection: final 181.1 vs 181.2 ns/ctor (dead even); S1/S2/S3 at baseline. Unlocks `np.place`/`copyto(where=)`/`np.where` migration onto one driver (6.3). Tests: `NpyIterWriteMaskedExecutionTests.cs` (24: enforcement matrix incl. NOT-enforced unbuffered/BUFNEVER parity, cast/readwrite-increment/writeonly-keeps-originals, strided-mask-buffered, broadcast-mask-2D, multi-window 20005, uint8-nonzero-true, error-text parity ×8, VIRTUAL matrix ×8). Suite 9554 (was 9530). | `NpyIter.cs`, `NpyIterBufferManager.cs`, `NpyIterCasting.cs` | probes + `ctor_probe.cs` + suite |
| 1.4 | ✅ **DONE (2026-06-10).** The hygiene wave turned up three real bugs, all reproduced against NumPy 2.4.2 and fixed by adopting NumPy's structure: **(a) the size-1-axis stride audit found the root divergence** — NumPy's `fill_axisdata` forces stride 0 for every operand on any size-1 iterator axis and on broadcast-stretched dims (nditer_constr.c:1594-1615); NumSharp copied raw operand strides, so its relaxed coalesce condition + NumPy's merge rule ("take stride1 when stride0==0") could keep the WRONG stride — `RemoveMultiIndex` on a (1,4) view with element strides (1,2) iterated [0,1,2,3] instead of [0,2,4,6]. Fixed at the root: stride normalization at fill (both standard-broadcast and op_axes paths), coalesce condition reverted to NumPy's strict form ((shape==1 && stride==0) ∥ formula); contiguous size-1 shapes ((2,4,1)/(1,4)/(4,1)) still fully coalesce. **(b) the op_axes fill used the raw array stride for an operand size-1 axis stretched to a larger iter dim → out-of-bounds reads** (op_axes=[[0],[0]] over [(3,), (1,)] read garbage); now stride 0 + `ApplyOpAxes` applies the same reduce-validation as op_axis=-1 entries. **(c) bug-(b) family sites #6/#7: `FlipNegativeStrides` and `GetAxisStrideArray` multiplied source element strides by the BUFFER dtype size** — K-order + BUFFERED int32→f64 over negative strides landed the base pointer 2× too far (garbage); now `SrcElementSizes` (production unary was safe — it forces C/F order, so the flip never ran there). Also shipped: **NumPy's write-broadcast validation** (stretched WRITE dim without REDUCE_OK → "output operand requires a reduction…" per NumPy W1; write-only reduce → W3 error; with REDUCE_OK → REDUCE flags set and accumulation works, W7); CA2014 stackalloc hoisted out of the reorder insertion-sort; dead `CalculateGrowInnerSize` (latent bug: `expectedStride` not reset per operand — op 2+ checked against op 1's accumulated product) + `PrepareBuffers` + `FinalizeBuffers` deleted (zero callers since Wave 4); `EARLY_EXIT` flag deleted, `PARALLEL_SAFE` wired (free at construction — reuses COPY_IF_OVERLAP's work) + `IsParallelSafe`. **Review addendum (same day):** the post-commit review found that the normalization made `GetMinStride` return 0 for size-1 axes, and (i) the DESCENDING K-order reorder sank them innermost (inner loop of 1, linearity lost) — fixed: key-0 axes sort OUTERMOST in descending mode (sequence-neutral; NumPy-outcome-equivalent); (ii) the forced-C/F non-coalesced branch never sorts at all, so a trailing size-1 axis sat innermost — a PRE-EXISTING pathology ((N,1)-strided ufuncs ran N one-element inner loops, 23–30 ms for 4M f32 on both trees) rooted in NumSharp gating coalescing on all-contiguous where NumPy coalesces UNCONDITIONALLY after order resolution — fixed via `RemoveUnitAxes` (absorbs size-1 axes on the non-index-tracked non-coalesced branch, exactly what NumPy's strict trivial branch does): **(4M,1) strided f32: add 23.4→4.2 ms (5.5×), sqrt 20.2→4.1 ms (4.9×) — interleaved best-of-3 vs 33058b83**. Tests: `NpyIterSizeOneStrideTests.cs` (17: 3 bug repros, coalesce sanity, W1/W3/W7 parity, axis-stride byte semantics, unit-axis absorption + multi-index preservation + (N,1) e2e, 4 PARALLEL_SAFE). Suite 9530 (was 9513); interleaved A/B vs the pre-wave tree perf-neutral on the variation grid (P15 1.309 vs 1.334 µs), (N,1)-class shapes 4.9–5.5× faster. | `NpyIter.cs`, `NpyIterCoalescing.cs`, `NpyIterFlags.cs`, `NpyIterBufferManager.cs` | suite + probe |

### Wave 2 — Small-N dispatch (P15: 1.34 → ≤0.5 µs target)
| # | change | rationale |
|---|---|---|
| 2.1 | ✅ **DONE (2026-06-10, commit 5962a5e1).** **`out=`/`where=` on the elementwise np.* core** — binary (add/subtract/multiply/divide/true_divide/mod/power/floor_divide) + unary (sqrt/exp/log/sin/cos/tan/abs/absolute/negative/square). All semantics probed vs NumPy 2.4.2 and pinned verbatim (19 tests): out joins the broadcast but never stretches (inputs broadcast UP to a bigger out); loop dtype from INPUTS, out validated same_kind ("Cannot cast ufunc 'add' output from dtype('float64') to dtype('int32')…" incl. ufunc names remainder/absolute/negative); reference identity; where must be bool ('safe' cast text), broadcasts AND joins the output shape, false slots keep prior out; aliasing safe via Wave-1.1 COPY_IF_OVERLAP. Engine: `DefaultEngine.UfuncOut.cs` Into-paths share kernels+cache keys with the existing Tier-3B routes; dtype-mismatched out = CAST operand through the Wave-4 windowed flush; where rides as trailing ARRAYMASK with WRITEMASKED out (NumPy op[nop]=wheremask) and `ForEach` got the masked inner loop (mask-TRUE run decomposition around the unmasked kernel — SIMD survives dense masks). TensorEngine: 16 signatures gained trailing `@out`/`where` (house ReduceAdd/clip pattern). **Bonus bug fixed:** `ResolveInnerLoopCount` read `Shape[-1]` on 0-d EXLOOP iterators (AV; unreachable before — the scalar×scalar bypass kept 0-d out of ForEach until out= routed it). **np.add(a,b,out) e2e: 446 ns vs 834 ns allocating** — the idiomatic answer to the allocation tax. | API parity + biggest single small-N lever |
| 2.2 | ⚠️ **PARTIAL (2026-06-10).** Call-invariant per-operand FLAG arrays hoisted to static readonly at all 7 iterator call sites (binary/unary routes + 4 ufunc-out configs). The operand `new[]{lhs,rhs,result}` arrays stay per-call BY DESIGN: the iterator stores the reference (`_operands`) and the overlap machinery can construct nested iterators (np.copyto inside MaterializeForcedCopies) on the same thread — thread-static reuse would alias live iterators. Full span overloads remain open (needs an `_operands` ownership strategy). | each np.* call today allocates 2–3 helper arrays + 1 delegate |
| 2.3 | **Phase 1 trivial constructor** (`TryNewTrivial`: NOp ≤ 3, all-contig, C-order, no cast → fill minimal state, skip coalescing/order resolution) + optional iterator-state pooling keyed (NOp, NDim). Construction measured 177 ns (ctor_probe) — this targets the iterator-routed small-N classes (mixed-dtype 1K ≈ 2.4 µs) and the out= route (446 ns). **NOTE (Wave-2 profiling): the dominant P15 lever is NOT construction — it is the finalizer lifecycle**: np.add(1K) ≈ 834 ns of which result allocation is ≈ 804 ns, and of THAT ≈ 500 ns is the two finalizable objects per result (~NDArray + Disposer registration, finalizer-queue churn, extra-gen survival). `~NDArray` cannot be removed: ArcLifecycleTests pin captured-slice + dropped-NDArray reclamation (the Disposer stays reachable through the captured slice, so only the NDArray finalizer's Release frees). Reaching P15 ≤ 0.5 µs needs a finalizer-model design decision (object pooling / conditional registration), not ctor work. | NumPy's `check_for_trivial_loop` applied to construction |
| 2.4 | ✅ **DONE (2026-06-10).** **`SizeBucketedBufferPool` window opened to 1 B–64 MiB** (was 4 KiB–1 MiB — the 1K-f32 small-N result, 4000 B, missed the floor by 96 bytes and every 4M-element output, 16–32 MiB, missed the cap) with per-bucket cap 2 at ≥ 1 MiB (bounds resident memory; the tcache pattern needs only the hot output shape warm). **In-place toggle-verified ~2× on every allocation-heavy row: P1 contig add 4M 3.37→1.74 ms, S2 strided sqrt 790→432–538 µs, P9 mixed 6.4→3.4–4.2 ms, P2–P4 broadcast ≈ 1.48 ms** — the warm-page reuse eliminates demand-zero page faults during the kernel's write pass (the POC allocator-tax finding). Plus: **GC pressure moved pool-side tracking the buffer's LIVE state** (Add on Take, Remove on Return — NOT residency: keeping pressure registered for idle pooled buffers told the GC ~100–200 MB was live and drove constant gen2 collections, measured 30–50% degradation everywhere) and **`GC.SuppressFinalize(Disposer)` once the buffer is freed** (the finalizer would be a guaranteed no-op; saves finalizer-queue churn on every released buffer; UMB dispose 121→70 ns). | the allocator finding affects every benchmark and real workload |

### Wave 3 — Strided roofline (S1/S2/S3: 3–13× headroom; kernels already proven)
| # | change | target |
|---|---|---|
| 3.1 | **Phase 2a: `EmitFusedStridedSimdLoop`** at the `lblScalarStrided` site (`DirectILKernelGenerator.InnerLoop.cs:304`): **AVX2 hardware-gather body** for f32/f64/i32/i64 (index vector hoisted, scale=1 byte offsets, guard `Avx2.IsSupported && |7·stride| ≤ int.MaxValue`), insert-gather fallback otherwise. Transcribe `PocKernels.AddF32/SqrtF32/SumF32`; unrolls: binary 2×, unary 4×, reduce 4 accumulators. | ✅ **DONE (2026-06-10).** Kernel: strided add 318 µs (POC 334), strided 2-D sqrt 201 µs (POC 203). Production e2e same-run vs NumPy: S1 1080 vs 1033 (parity), S2 941 vs 1282 (1.36× faster); preallocated-out iterator route 354 µs — the residual e2e delta is the Wave-2.4 allocator tax. Implementation: gather dispatcher + `EmitSimdGatherLoop` in the Tier-3B shell (per-input AVX2 vgather, byte-offset index vector hoisted, runtime stride guard, stride-0/negative valid), reusing the caller's vectorBody; 4× unroll + remainder + scalar tail; i32/u32/f32/i64/u64/f64 at V256+AVX2, scalar fallback otherwise. |
| 3.2 | ✅ **DONE.** `GATHER_ELIGIBLE` computed in `UpdateContiguityFlags`. Kernel-key selection unnecessary (the Tier-3B kernel is layout-polymorphic; runtime dispatch ≈ 2 compares/chunk). **Bonus bug found & fixed: the NumSharp extension flags ALIASED the shifted NumPy flags** (`CONTIGUOUS==GROWINNER`, `GATHER_ELIGIBLE==ONEITERATION`, `EARLY_EXIT==DELAYBUF`, `PARALLEL_SAFE==REDUCE`) — setting GATHER_ELIGIBLE made `ForEach` run ONE inner loop and silently skip all remaining rows. Renumbered to free bits 3–6; pinned by `NpyIterFlags_ExtensionFlags_DoNotAliasNumPyFlags`. | — |
| 3.3 | **Strided store** (dest-strided): NumPy's `npyv_storen` analog (scalar lane stores from the vector) so CONTIG→NCONTIG and NCONTIG→NCONTIG vectorize too (composite class E2). | ✅ **DONE.** Scatter-store gather variant (per-lane GetElement + scalar store — NumPy's `npyv_storen` shape): NCONTIG→NCONTIG vectorizes the compute; all 6 dtypes exercised through the real iterator. |
| 3.4 | Strided **reduce** body via hw-gather + 4 accumulators in `ExecuteReduction`'s strided kernel. | ✅ **DONE.** Gather section in `EmitReductionStridedLoop` ndim==1: 4 vector accumulators → tree-merge → horizontal → scalar-tail continuation; shares identity/combine helpers with the contiguous SIMD path (identical NaN semantics). Sum/Min/Max/Prod. **S3 371→117 µs vs NumPy 205–259 (1.7–2.2× faster).** |

### Wave 4 — NumPy-default iterator config for ufuncs ✅ DONE (2026-06-10)
| # | change | outcome |
|---|---|---|
| 4.1 | ✅ **DELAY_BUFALLOC implemented** (defer buffer allocation + first fill to `Reset`/first execution; NumPy raises without reset, NumSharp auto-ensures). | construction of buffered iterators is allocation-free |
| 4.2 | ✅ **Windowed buffered iteration implemented end-to-end** — the real deliverable turned out to be bigger than the flag: buffered NON-REDUCE iteration had **no iternext at all** (construction did one eager fill; >8192 elements silently processed only the first window). Now: `BufferedNext` (EXLOOP window-jump: flush→jump→refill, NumPy `npyiter_buffered_iternext`) + `BufferedElementNext` (per-element protocol without EXLOOP, NumPy nditer-templ specialization), row-aligned `ComputeTransferSize` (NumPy-observed 4000/4000/2000 fills), per-operand fills via the SIMD IL cast kernels, NumPy buffering criterion (*buffer only when REQUIRED*: cast/CONTIG/non-linear — linear strided operands keep true strides through `BufStrides`), flush-on-Dispose/Reset/single-fill. **Production verdict (A/B-measured, i9-13900K Release): NumPy's buffered-cast→SIMD architecture LOSES to our fused per-element-convert IL for cheap ops** (add contig 2M: 2.20 vs 1.49 ms; add strided: 3.18 vs 2.98; div: 1.72 vs 1.61) because the extra L2 round-trip outweighs the SIMD gain — NumPy buffers because AOT C loops cannot fuse casts; runtime IL can. **Promoting unary math DOES win buffered** (sqrt 2M: 1.65 vs 2.25 ms = 1.36×) → `np.sqrt/exp/log(int)`-class routes through the NumPy config (`EXTERNAL_LOOP\|BUFFERED\|GROWINNER\|DELAY_BUFALLOC\|COPY_IF_OVERLAP`, op_dtypes=output, unsafe casting); binary/compare keep the (faster) fused bodies. End-to-end vs NumPy: mixed add 0.79×, mixed mul 0.88×, **sqrt(i32) 0.81×**, strided mixed 1.00×. Also fixed en-route: `NpyIterRef.GetDataPtr`'s legacy partial-window recomputation (read stale data on the final window), `IsSingleInnerLoop`'s EXLOOP shortcut dropping windows 2+, `Copy()` missing the new window fields. Tests: `NpyIterBufferedWindowTests.cs` (12, all multi-fill); suite 9513. |

### Wave 5 — Reductions through the core
| # | change | rationale |
|---|---|---|
| 5.1 | Route **var/std/cumsum/cumprod/all/any axis paths** (today scalar `NpyAxisIter`) through `NpyIterRef` + Tier-3B kernels (`op_axes` + REDUCE_OK). Keep the 2b widening kernels as inner loops for sum/prod/min/max. | the remaining scalar axis paths; one driver |
| 5.2 | Implement **REUSE_REDUCE_LOOPS** (cache the reduce double-loop schedule across buffer fills). | NumPy-parity for buffered reductions |
| 5.3 | Retire `NpyAxisIter` once 5.1 lands. | dead machinery |

### Wave 6 — Exceed NumPy (architecture dividends)
| # | change | evidence |
|---|---|---|
| 6.1 | ✅ **DONE (2026-06-10).** **Fusion exposed as `np.evaluate(expr[, operands][, out])`** — ArrayNode leaves + implicit conversions (`(NpyExpr)a * b + 2`; exact-match NDArray/scalar operator overloads required: through implicit conversions alone C# binds literals to NDArray's object-operators and silently strong-types weak NEP50 scalars), reference-deduplicated binding ((a-b)/(a+b) = 3 iterator operands), `EXTERNAL_LOOP\|COPY_IF_OVERLAP` construction, out= with same_kind validation + windowed cast flush + overlap-safe aliasing. **Mixed-dtype trees now follow NumPy result_type PER NODE** (`NpyExpr.Typing.cs`: NEP50 strong-strong incl. the int/float tier crossing i4+f4→f8; weak python-scalar literals — i4+2→i4, f2+2.5→f2, bool+2→i64, exact "Python integer 300 out of bounds for uint8" OverflowError; true_divide ints→f64; arctan2 tier floats i1→f16/i4→f64; power/remainder/floor_divide bool→i8 + the literal negative-int-exponent ValueError; unary float tiers bool/i8→f16, i16→f32, i32+→f64; bool add=OR/multiply=AND; boolean negative/subtract + invert/bitwise-float TypeErrors verbatim) — pinned by the `(i4*i4)+f8` int32-wraparound test (1410065408.5, NOT 1e10+0.5); legacy `Compile()` emission contract preserved (every node at OutputType when no typing table). **Reduction roots** `NpyExpr.Sum/Prod/Min/Max/Mean` compile a one-pass raw kernel (4-acc unroll, aux accumulator slot, carry-in across chunks): `sum(a*b)` never materializes `a*b`; NumPy reduce dtypes (int→i64/uint→u64/floats preserved; mean ints→f64); f16/f32 sums accumulate in f64 and cast back (documented divergence from NumPy's pairwise — usually MORE accurate; f2 70000-ones → inf identically); min/max NaN-propagate; empty: sum=0, prod=1, mean=NaN at result dtype, min/max raise "zero-size array to reduction operation minimum which has no identity". `ExecuteExpression` now **throws without EXTERNAL_LOOP** unless single-chunk (the measured ~40× foot-gun). | **Measured (Release, best-of-9, 4M f64, same box, NumPy 2.4.2): a*b+c 4.13 vs 13.23 ms (3.2×), (a-b)/(a+b) 3.24 vs 19.65 (6.1×), sum(a*b) 2.45 vs 8.72 (3.6×), i4*2+f8 2.94 vs 10.21 (3.5×), f32 sum 1.47 vs 4.31 (2.9×) — and 1.2–4× over NumSharp's own unfused chains. Gates: `benchmark/fusion/evaluate_bench.{cs,py}`; variation grid + ctor probe (183 ns) neutral; suite 9596 (23 new, `NpyEvaluateTests.cs`).** |
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

# canonical NpyIter benchmark (every aspect x cache tiers -> one sheet;
# results + findings ledger: benchmark/npyiter/README.md + npyiter_results.md):
python benchmark/npyiter/npyiter_sheet.py

# suite gate
cd test/NumSharp.UnitTest && dotnet test --no-build -f net10.0 --filter "TestCategory!=OpenBugs&TestCategory!=HighMemory"
```

The overlap probe inside `variation_probe.cs` must print NumPy's `[1 2 4 6 8 10 12 14]` once Wave 1.1 lands — today it prints the corrupted cascade.
