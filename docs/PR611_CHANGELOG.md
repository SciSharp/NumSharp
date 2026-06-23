# PR #611 — Changelog

**[Major Rewrite] NumPy nditer port, NpyExpr DSL with 3-tier custom-op API, C/F/A/K memory layout support, stride-native matmul**

**455 commits · 806 files · +234,348 / −19,179** (vs `master`, after #612)

---

## TL;DR

- **`NpyIter` — full port of NumPy 2.4.2's `nditer`** (~12.5K lines): all iteration orders (C/F/A/K), all indexing modes, buffered casting, buffered-reduce double-loop, masking, memory-overlap protection (`COPY_IF_OVERLAP`), windowed buffering (`DELAY_BUFALLOC`), unlimited operands and dimensions. 566+ byte-for-byte NumPy parity scenarios.
- **`NpyExpr` DSL + three-tier custom-op API** — write your own ufuncs: raw IL (Tier 3A), element-wise scalar/SIMD (Tier 3B), or composable expression trees with operator overloads (Tier 3C). Exposed as the public **`np.evaluate`**, which runs fused expressions **3.2–6.1× faster than NumPy** (which can't fuse), with per-node NumPy `result_type` typing and fused reductions.
- **`out=` / `where=` / `dtype=` ufunc kwargs across the elementwise API** — the kwargs on every NumPy ufunc, spanning the binary, unary-math, comparison, predicate, and bitwise families with exact NumPy broadcast/cast/error-text semantics. Plus `np.bitwise_and/or/xor` and `np.positive` at the `np.*` surface.
- **NumPy-parity benchmark: geomean 1.00× at 10M elements** across ~409 ops (166 faster / 171 close / 36 slower) — measured by a new official BenchmarkDotNet-vs-NumPy suite committed with the report.
- **36 new `np.*` APIs** — `sort`, `pad` (11 modes), `tile`, `median`/`percentile`/`quantile` (all 13 interpolation methods) + their `nan*` variants, `average`, `ptp`, `take`/`put`/`place`, `extract`/`compress`, `diagonal`/`trace`, `argwhere`/`flatnonzero`, `unravel_index`/`ravel_multi_index`/`indices`, `delete`/`insert`/`append`, `diff`/`ediff1d`, `asfortranarray`/`ascontiguousarray`, `np.multithreading`.
- **C/F/A/K order support wired through the whole API** — `Shape` understands F-contiguity, `OrderResolver` resolves NumPy order modes, ~68 layout bugs fixed across 9 fix groups.
- **Stride-native matmul/dot** — BLIS-style GEBP GEMM absorbs arbitrary strides for all dtypes (kills a ~100× penalty on transposed inputs); fused 1-D dot is 3.5–9× faster with zero GC; opt-in multithreaded dot ~2× faster than NumPy's default on 1M vectors.
- **Sorting, casts & Complex finished** — `np.sort`/`np.argsort` on a radix line-kernel (closes a Missing Function); a SIMD strided-cast campaign that killed the cast cliffs (15×8×15 `astype` matrix: 716 → ~391 lagging cells, 852 → 1,177 winning cells vs NumPy); `np.zeros` via `calloc`/demand-zero (O(1), was ~1000× slower); the six Complex transcendentals (`sinh`…`arctan`); and bit-exact pairwise summation for `sum`/`mean`.
- **Deterministic memory management** — atomic reference counting + `IDisposable` on `NDArray`, plus a tcache-style buffer pool (1 B – 64 MiB window).
- **Differential fuzzing infrastructure** — 37,445 bit-exact NumPy-comparison cases across 24 corpus tiers, a seeded random fuzzer with shrinker, a CI FuzzMatrix gate, and a nightly soak workflow.
- **Legacy iterator stack deleted outright** — `MultiIterator`, the Regen-generated cast templates, *and* `NDIterator` itself (interface + class + `AsIterator` extensions) are all gone; every code path now iterates through `NpyIter` / `NpyFlatIterator` / `GetAtIndex`.
- **Test suite: 9,990 passed / 0 failed** on net8.0 + net10.0 (+2,600 new test methods), plus the 37,445-case fuzz corpus replayed by the FuzzMatrix gate.

---

## 1. NpyIter — full NumPy `nditer` port

From-scratch C# port of NumPy 2.4.2's iterator machinery under `src/NumSharp.Core/Backends/Iterators/` (~12,557 lines), promoted to **public API** with NDArray overloads.

| Capability | Detail |
|---|---|
| Iteration orders | C, F, A, K — incl. NEGPERM negative-stride handling, axis reordering + coalescing to full 1-D collapse |
| Indexing modes | `MULTI_INDEX`, `C_INDEX`, `F_INDEX`, `RANGE` (parallel chunking), `GotoIndex` / `GotoMultiIndex` / `GotoIterIndex` |
| Buffering | Buffered casting with all 5 casting rules, **windowed buffered iteration**, `DELAY_BUFALLOC`, buffered-reduce double-loop (incl. `bufferSize < coreSize`) |
| Reductions | `op_axes` with `-1` reduction axes, `REDUCE_OK`, `IsFirstVisit`, `REUSE_REDUCE_LOOPS` slab accumulation |
| Overlap safety | **`COPY_IF_OVERLAP`** via a port of NumPy's `mem_overlap` solver (`NpyMemOverlap.cs`) — overlapping in/out operands no longer silently corrupt |
| Masking | `WRITEMASKED` + `ARRAYMASK` **executed** — the buffered window flush writes back only mask-nonzero elements; `VIRTUAL` operands (null op slots) construct with NumPy 2.x semantics |
| Operands / dims | **Unlimited operands** (NumPy caps at `NPY_MAXARGS=64`) and **unlimited dimensions** (NumPy caps at `NPY_MAXDIMS=64`) via dynamic allocation |
| APIs | `Copy`, `GetIterView`, `RemoveAxis`, `RemoveMultiIndex`, `ResetBasePointers`, `IterRange`, `DebugPrint`, fixed/axis stride queries, `GetValue<T>`/`SetValue<T>`, … |
| Casting parity | `NpyIterCasting.CanCast` matches NumPy's `safe`/`same_kind` lattice exactly |

Validated by a dedicated battletest harness: **566 scenarios** replayed against NumPy 2.4.2 byte-for-byte, a permanent variation-probe harness, and `tools/iterator_parity`. Dozens of parity bugs found and fixed against NumPy ground truth: negative-stride flipping, `NO_BROADCAST` enforcement, `F_INDEX` coalescing, buffered-reduction stride inversion, K-order on broadcast inputs, EXLOOP `iternext`, buffered-cast `Advance`, ranged `Reset()` desync, buffer free-list corruption, the size-1 stride-0 invariant (a `(1,4)` view with nonzero stride corrupted `RemoveMultiIndex`), `op_axes` out-of-bounds reads on stretched size-1 axes, write-broadcast validation, `PARALLEL_SAFE` wiring, and unit-axis absorption — each reproduced against NumPy first, then fixed by adopting NumPy's constructor structure.

### Execution at NumPy speed

`NpyIter` isn't just correct — it is now the production execution engine: `DefaultEngine`'s binary, unary, and comparison ops (same- and mixed-dtype) route through the NpyIter Tier-3B shell, and it measures **at-or-faster than NumPy on every probed aspect** (Release, i9-13900K, NumPy 2.4.2):

| Aspect (float32) | NumSharp | NumPy | Ratio |
|---|---|---|---|
| contig sqrt 10M | 2.98 ms | 3.24 ms | 0.92× |
| contig add 10M | 3.91 ms | 4.09 ms | 0.96× |
| strided add 1M | 319 µs | 416 µs | 0.77× |
| strided sqrt 1M | 206 µs | 374 µs | 0.55× |
| strided sum 1M | 109 µs | 205 µs | 0.53× |
| **fused** `a*b+c` 10M | 4.77 ms | 13.38 ms | **0.36×** |
| **fused** `(a-b)/(a+b)` 10M | 4.12 ms | 22.33 ms | **0.18×** |

Key mechanisms: an O(1) **trivial-loop bypass** that skips iterator construction for contiguous operands, identity-broadcast fast paths, **AVX2 hardware-gather** (`vgatherdps`) strided SIMD in the Tier-3B shell (NumPy uses scalar loops for strided binary/reduce — its floors are beatable), and strided-reduction kernels (2-D strided sqrt 1.36× faster than NumPy, strided sum 2.2× faster).

## 2. NpyExpr DSL + three-tier custom-op API

User-extensible kernel layer on top of `NpyIter` — the public answer to "how do I write my own ufunc":

- **Tier 3A — `ExecuteRawIL`**: emit raw IL against the NumPy ufunc signature `void(void** dataptrs, long* strides, long count, void* aux)`.
- **Tier 3B — `ExecuteElementWise`**: provide scalar + vector IL; the shell supplies a 4×-unrolled SIMD loop, remainder vector, scalar tail, and strided fallback.
- **Tier 3C — `ExecuteExpression`**: compose `NpyExpr` trees with C# operators (`(a - b) / (a + b)`), 50+ node types (arithmetic, trig, exp/log, rounding, predicates, comparisons, `Min/Max/Clamp/Where`), plus **`Call()`** to splice any delegate/`MethodInfo` into a fused kernel. Compiled once, cached by structural key, ~5 ns dispatch.

This is what powers the fusion wins — one pass, no temporaries — and it is exposed publicly as **`np.evaluate(expr[, operands][, out])`**:

- **Per-node NumPy `result_type` typing** — every node resolves to its NumPy 2.4.2 dtype, so mixed trees wrap correctly: `(i4*i4)+f8` wraps the multiply in int32 (→ `1410065408`) before promoting. Strong-strong NEP50 (incl. int/float tier crossing), weak python-scalar literals (`i4+2 → i4`, `i4/2 → f8`) with NumPy's exact `OverflowError`, and special resolvers (`true_divide`, `arctan2`, negative-integer-literal `power` → `ValueError`, bool `add`=OR/`multiply`=AND).
- **Fused reductions** — `NpyExpr.Sum/Prod/Min/Max/Mean` compile a one-pass inner loop; `sum(a*b)` reads `a` and `b` once and never materializes the product. NumPy reduction dtypes (int→i64, uint→u64, mean→f64).
- **`out=` joins via the ufunc rules** (same_kind validation, reference identity, overlap-safe aliasing through `COPY_IF_OVERLAP`); an `EXTERNAL_LOOP` guard prevents the silent `count==1` slow path.
- **Measured** (Release, 4M f64, NumPy 2.4.2): `a*b+c` **3.2×**, `(a-b)/(a+b)` **6.1×**, `sum(a*b)` **3.6×**, `sum f32` 2.9×, `i4*2+f8` 3.5× faster. Permanent gate in `benchmark/fusion/evaluate_bench.{cs,py}`.

## 3. Legacy iterator stack retired

- `MultiIterator` **deleted**; all callers migrated to `NpyIter.Copy` / multi-operand execution.
- The Regen template `NDIterator.template.cs` + 16 generated `NDIterator.Cast.*` partials **deleted** (−3,870 LOC in one commit).
- `NDIterator` (interface + `NDIterator<T>` + `AsIterator` extensions) **deleted entirely** — `[Obsolete]` tombstones that threw at runtime after the migration and were referenced by nothing live. Production iteration runs through `NpyIter`/`NpyIterRef` (kernels), `GetAtIndex` (flat reads), and `NpyFlatIterator` (`np.broadcast(...).iters`).
- `~400` per-dtype `NPTypeCode` switch sites replaced by a generic `NpFunc` dispatch utility.

## 4. C/F/A/K memory-layout support

- `Shape` now tracks **F-contiguity** with NumPy-convention contiguity computation; new `OrderResolver` resolves `C`/`F`/`A`/`K` for every API with an `order` parameter.
- Order support wired through: `copy`, `array`, `asarray`, `asanyarray`, `*_like`, `astype`, `flatten`, `ravel`, `reshape`, `eye`, `concatenate`, `cumsum`, `argsort`, `tile`, plus **post-hoc F-contig preservation across the IL-kernel dispatchers**.
- New: `np.asfortranarray`, `np.ascontiguousarray`.
- `np.where` selects C/F output layout the way NumPy does; `ravel('F')` of an F-contig source returns a **view** (was a 3,000× copy).
- ~68 layout bugs fixed across 9 TDD fix groups, backed by ~3,300 lines of new order tests (Sections 41–51: reductions/keepdims, matmul/dot/outer/convolve, broadcasting-from-F, manipulation, file I/O `fortran_order`, Decimal scalar path, fancy-write isolation, …).

## 5. New & completed `np.*` APIs

**New functions (36):**

| Area | APIs |
|---|---|
| Fused / ufunc | `np.evaluate` (fused expressions — see §2), `np.bitwise_and`, `np.bitwise_or`, `np.bitwise_xor`, `np.positive` |
| Sorting | `np.sort` (+ `ndarray.sort`; `np.argsort` reimplemented) — radix line-kernel on NpyIter, stable, NaN-last, all axes / orders (`IterAllButAxis` drive mirroring NumPy's `_new_sortlike`) |
| Manipulation | `np.pad` (all 11 NumPy modes + callable), `np.tile`, `np.delete`, `np.insert`, `np.append` |
| Indexing/selection | `np.take`, `np.put`, `np.place`, `np.extract`, `np.compress`, `np.argwhere`, `np.flatnonzero`, `np.diagonal`, `np.trace`, `np.unravel_index`, `np.ravel_multi_index`, `np.indices` |
| Statistics | `np.median`, `np.percentile`, `np.quantile` (**all 13 interpolation methods**, tuple axis, `out=`, `keepdims`, QuickSelect engine), `np.average` (`weights`, `returned`, tuple-axis; fused kernel 1.3–1.6× faster than NumPy at 1M), `np.ptp`, `np.nanmedian`, `np.nanpercentile`, `np.nanquantile` |
| Math | `np.diff`, `np.ediff1d` |
| Creation | `np.asfortranarray`, `np.ascontiguousarray` |
| Runtime | `np.multithreading(enabled, max_threads)` — opt-in threaded kernels |

**Rebuilt to full NumPy 2.x parity:**

- `np.clip` — `min=`/`max=` keyword aliases, default-None bounds, NumPy 2.x dtype promotion, `out=` validation.
- `np.unique` — 5 missing kwargs, sort+mask algorithm (up to 43× faster), NaN partitioning, `n > Array.MaxLength` fallback.
- `np.searchsorted` — `side=`, `sorter=`, multidim validation; IL binary-search kernels 5–25× faster (beats NumPy on 20/22 benchmarks).
- `np.copyto` — `casting=`, `where=` masked copies at NumPy speed (was 7–72× slower).
- `np.asarray` — `copy=`, `like=`, `device=`, dtype-as-string. `np.concatenate` — full parity + C/F fast paths. `np.all`/`np.any` — tuple-axis, `out=`, `where=`. `np.expand_dims` — tuple axis. `np.repeat` — `axis=` parameter. `np.power` — integer-power semantics, negative-exponent `ValueError`, crash fix.
- `np.broadcast` — N-operand form (`0..64`, then unlimited — NumPy parity, was 2-operand only), live index cursor, lazy `.iters`, `.numiter`.
- Engine completeness: bool/char `max`/`min`, Complex quantile, `IsInf` implemented (was a stub); the six **Complex transcendentals** `sinh`/`cosh`/`tanh`/`arcsin`/`arccos`/`arctan` implemented (hybrid BCL + C99 edge fix-ups, NumPy 2.4.2 parity — were `NotSupportedException`).
- **Full 15-dtype coverage pushed through the hot paths** — the SByte/Half/Complex dtypes introduced in #612 now work across every kernel family this PR touches (reductions, indexing, trace, casts, quantile, …).

**`out=` / `where=` / `dtype=` ufunc kwargs (NumPy parity):**

The kwargs present on every NumPy ufunc now span the elementwise core — binary (`add`, `subtract`, `multiply`, `divide`, `true_divide`, `mod`, `power`, `floor_divide`), unary-math (`sqrt`, `exp`, `log`, `sin`, `cos`, `tan`, `abs`/`absolute`, `negative`, `square`), the six comparisons, predicates (`isnan`/`isfinite`/`isinf`), bitwise, `invert`, `arctan2` — each as **one NumPy-shaped overload**, every rule pinned against NumPy 2.4.2:

- `out` joins the broadcast but **never stretches** (mismatched/stretchable `out` raise NumPy's exact texts, trailing space included); loop dtype resolved from inputs (NEP50), `out` only needs a same_kind cast; the provided instance is returned (reference identity).
- `where` must be exactly `bool` (mask cast under 'safe'); it broadcasts over operands **and** participates in output shape; mask-false slots keep prior `out` contents.
- `out` **aliasing an input is well-defined** via `COPY_IF_OVERLAP` — `add(x[:-1], x[:-1], out=x[1:])` matches NumPy exactly.
- `dtype=` **computes in the loop dtype** (`subtract(300, 5, dtype=i16) = 295`), with the bool `add`→OR / `multiply`→AND remap keyed off the **final** loop dtype so `add(True, True, dtype=i32) = 2`.

## 6. Linear algebra

- **Stride-native GEMM for all 12 numeric dtypes** — BLIS-style GEBP with stride-aware packers; the 8×16 `Vector256` FMA micro-kernel reads packed panels, so transposed/sliced inputs cost nothing extra. Eliminates the ~100× fallback penalty (`np.dot(x.T, grad)`: 240 ms → ~1 ms) and the boxing `GetValue` fallback chain.
- **Full `matmul` gufunc semantics** — batched stacking, 1-D promotion/squeeze rules, validated by a dedicated differential matrix (816 cases).
- **Fused single-pass 1-D dot** — 3.5–9× faster, **zero GC** (was up to 446 gen-0 collections per call at 100K).
- **`np.multithreading`** — opt-in parallel 1-D dot: 1M float dot 172 → 60 µs, ~2× faster than NumPy's default build. Off by default; bitwise-identical summation order when off.

## 7. Performance (beyond NpyIter and linalg)

| Op | Improvement |
|---|---|
| Axis reductions, narrow ints | **Widening SIMD** (int16→int32 accum etc.): `sum(int16, axis=1)` 1058 ms → 2.7 ms (**389×**, now faster than NumPy); int32/uint32 2.3–4.6×; also fixes a uint32 axis-sum **corruption** bug |
| `mean` (axis) | **217×** (Phase-0 bug surgery); `var`/`std` 21×; `count_nonzero` 20× |
| `np.nonzero` | IL SIMD kernel closes an **8–241×** gap to NumPy |
| `np.where` | IL kernels for scalar-broadcast & non-contiguous (1.2–2× NumPy on broadcast conditions) |
| Strided 1-D unary | Fused strided-SIMD kernel: 0.55 ns/elem flat — beats NumPy at every size; strided `sqrt` reached parity via gather→tile→SIMD buffering |
| Strided flat reductions | Incremental-advance path: strided sum 8.3× faster (11.8× behind NumPy → 1.4×) |
| Comparisons | **PDEP**-based packed mask→bool store; broadcast/strided compares routed via NpyIter |
| Axis-0 reductions | Column-tiled accumulation (breaks the output RAW dependency); 8× pairwise unrolled flat reductions |
| Allocation | tcache-style **size-bucketed buffer pool** with a **1 B – 64 MiB** window (covers both the small-N ufunc result and 4M+ outputs that previously paid a fresh `VirtualAlloc` + demand-zero faults); ≥1 MiB buckets capped at 2 buffers; pool-side GC memory pressure tracking live state; `GC.SuppressFinalize` on free; `using`/ARC adopted across `concatenate`, `allclose`, `convolve`, `tile`, `eye`, masking, shuffle, … |
| Casts (SIMD campaign) | Strided/gathered SIMD kernels across the full 15×8×15 `astype` matrix — `cvtt` float→int, Giesen f16↔ widen/narrow, complex deinterleave, sub-word VPSHUFB shuffles, fused VPGATHER whole-array kernels, single-pass KEEPORDER same-type copy. Cliffs eliminated: **716 → ~391 lagging cells, 852 → 1,177 winning cells** vs NumPy |
| `np.zeros` | `calloc` / Windows `VirtualAlloc` demand-zero — O(1) regardless of size (10M f64: 14.3 ms → ~0.01 ms, was ~1000× slower) |
| Broadcast-reduce | Stride-0 axes folded algebraically in the flat-reduction chokepoint (no O(D×N) materialize) — `sum(broadcast_to(...))` now **~534–700× faster**, beats NumPy, bit-exact |
| `sum`/`mean` (float) | Bit-exact NumPy **pairwise summation** ported onto the per-chunk reduce path — matches `np.add.reduce` bit-for-bit (unblocks float32) |
| `np.any`/`np.all` (bool/char) | Reinterpret to byte/ushort → existing integer SIMD path (was a 5–12× scalar cliff); fixes a latent AVX2 32-lane mask-overflow correctness bug |
| Complex/Half/Decimal reductions | NpyIter chunked `ForEach` axis reductions — Decimal 5–13×, Half mean 1.6–3.7×, Complex mean 15–45×→parity; float16 negate ~10× via sign-bit flip |
| Casts (`float→int32`) | NumPy-faithful SIMD `cvtt`, strided/reversed/gathered variants |
| `np.split` family | O(1) sub-shape derivation, direct views — 1.5–4× faster than NumPy |
| Where/copyto/searchsorted/unique | see §5 |

## 8. Official benchmark suite + honest methodology

- New cross-platform `run_benchmark.py` entry point: BenchmarkDotNet Full rigor (50 iters, InProcessEmit) × all suites × {1K, 100K, 10M} vs NumPy 2.x — **1,813 C# measurements, 1,111 matched op×dtype×size comparisons**, structural op-name join, tracked markdown report + per-suite artifacts + history snapshots. Coverage spans **all 15 dtypes** (SByte/Half/Complex suites added).
- **Headline:** geomean NumSharp÷NumPy = **1.00× at N=10M** (166 ops faster / 171 close / 36 slower) — parity across the whole op surface at memory-bound sizes; ~1.9× at 1K where per-call dispatch dominates (tracked as the next focus).
- Found and neutralized a **benchmark-invalidating tooling bug**: `dotnet run` file-based apps compile the project reference in Debug (optimizations off) even with `Configuration=Release` properties — hand loops measured ~2× slow while DynamicMethod IL was immune. Benchmarks now assert `IsJITOptimizerDisabled == false` and refuse to mislead; the rule is documented.
- **Canonical NpyIter benchmark** — a section-addressable harness covering 33 op families × {scalar/1K/100K/1M/10M}, integrated into `run_benchmark.py`, plus a **post-release CI workflow** (`.github/workflows/benchmark.yml`) that auto-commits report cards to master.
- **Frontier findings — found, then fixed.** Adversarial probes flagged real losses; the headline ones are now closed: `np.sum` over a `broadcast_to` view (was **54× slower**) folds stride-0 axes algebraically and runs **~534–700× faster** than NumPy, bit-exact; scalar `np.any`/`np.all` on bool/char (was **5–12× slower**) reinterpret onto the integer SIMD path; `np.zeros` (was ~1000× slower) is calloc-backed. Remaining tracked items: small-N (~1K) per-call dispatch overhead and a few iterator edge cases pinned as `[OpenBugs]`/skipped repros. A win surfaced too: hand-rolled 8-band parallel iteration **4.7×**.

## 9. Differential fuzzing vs NumPy (new infrastructure)

- **37,445 bit-exact corpus cases** across 24 JSONL tiers generated from real NumPy 2.4.2 outputs: casts (full 15×15 matrix), binary arith (NEP50), div/mod/power, comparisons, unary (incl. float16 inputs + all narrow ints), reductions, NaN-aware reductions, cumulative, statistics, logic/extrema, bitwise+shift, where/place, manipulation, matmul, modf multi-output, sorting/searching, parameter sweeps, SIMD-tail boundaries (900 cases around vector-width edges), operand aliasing, and error-parity (exception-for-exception).
- **Seeded random fuzzer** with an element-wise shrinker for minimal repros; **metamorphic invariant tier** (11 algebraic properties).
- **CI integration:** FuzzMatrix gate wired into the build workflow + a new nightly **fuzz-soak** workflow (`.github/workflows/fuzz-soak.yml`).
- Findings inventoried in `docs/FUZZ_FINDINGS.md`; every fixed class re-armed as a permanent regression gate. The error-parity tier alone surfaced 1 critical crash; the op tiers surfaced 17+ distinct bug classes that are now fixed (see §10).

## 10. Correctness — NumPy-parity bug fixes

**Semantics (behavioral changes, may affect callers):**

- `floor_divide` / `mod`: NumPy-exact floored semantics and divide-by-zero results.
- Comparisons: `<=` / `>=` now return `False` for NaN (IEEE/NumPy).
- Flat `min`/`max` propagate NaN.
- `np.negative(uint)` wraps modulo 2ⁿ instead of throwing; `bool - bool` and `-bool`/`np.negative(bool)` now **throw** (NumPy behavior).
- Transcendental ufuncs use NEP50 width-based float promotion.
- `np.power`: negative integer exponent raises `ValueError`; exact integer-power semantics.
- Cast semantics aligned with NumPy across all dtype pairs (IL kernels + `ConvertValue`); `complex→bool` no longer drops the imaginary part; `float→int` SIMD uses truncation (`cvtt`) like NumPy.
- Broadcasting keeps rank when a 1-D `[1]` meets a lower-rank operand; quantile-family dtype & bool handling; Complex `np.where`.
- Integer `reciprocal(0)` is per-width exact: `int32`/`int64` → `MinValue`, `uint64` → 2⁶³, but `0` for int8/int16/uint8/uint16/uint32 (was MinValue/0 across the board); `bool` → int8.
- `clip`/`maximum`/`minimum`: float16 signed-zero scalar tail, NaN propagation through the SIMD kernel, and correct F-contiguous/strided element pairing.
- `float16` axis sum accumulates in `float32` (NumPy parity); Complex flat `min`/`max` return the NaN-bearing element verbatim; Complex unary math ported from NumPy's own C99 algorithms.

**Crashes & corruption:**

- Overlapping-operand corruption eliminated iterator-wide (`COPY_IF_OVERLAP`, §1).
- Masked iteration: a buffered `WRITEMASKED` write landed garbage in exactly the slots NumPy preserves (silent corruption of the elements the caller asked to protect) — now writes back only mask-nonzero elements.
- uint32 axis-sum produced wrong values past 8 distinct columns (widening-SIMD rewrite).
- `np.pad`: 5 correctness/crash bugs (battle-tested against NumPy 2.4.2); linear_ramp preserved Complex dtype.
- `UnmanagedStorage`/`ArraySlice`: `CopyTo` direction + bounds bugs; `CloneData` partial-buffer bug; scalar offset lost on `Clone`; buffered `NpyIter.Clone` shared buffers; `DTypeSize` reported `Marshal.SizeOf` instead of in-memory stride; `NPTypeCode.Char.SizeOf` returned 1 (real: 2); stale Decimal priority.
- `TensorEngine` now propagates through `Cast`/`Transpose`/`copy`/`reshape`/`ravel` (custom engines were silently dropped).
- `take` with `out=` enforces NumPy's safe-cast direction; `put`/`place` non-contiguous writeback fixes; `argsort` on non-C-contiguous input.
- NpyIter `ForEach`/`ExecuteGeneric`/`ExecuteReducing` read past the end without `EXTERNAL_LOOP`.
- `np.exp2` float32-output IL kernel was malformed (`InvalidProgramException`); `np.power` with a Half exponent threw `InvalidCastException`; a narrowing `dtype=` on a complex float-ufunc segfaulted — all fixed.
- Complex `nansum` axis reduction read uninitialized memory for `ndim ≥ 3`; the AVX2 32-lane `any()` mask overflow (byte/sbyte) returned wrong results; net8.0 complex `abs` and axis `min`/`max` NaN propagation corrected.

## 11. Memory management — ARC + `IDisposable`

- `NDArray` now implements **`IDisposable`** backed by **atomic reference counting** on the unmanaged block: CAS-driven `TryAddRef`/`Release`, idempotent `Dispose`, finalizer safety net, immortal non-owning wraps. Views keep parents alive; parent disposal never invalidates live views.
- Hammered by a 15-case lifecycle suite incl. 32-thread × 1,000-op concurrency races and 50-way parallel dispose — zero corruption.
- Deterministic release means hot loops no longer wait on the finalizer queue; combined with the buffer pool this removes most steady-state GC pressure (`dot` at 100K: 446 collections → 0).

## 12. `Char8` primitive

New 1-byte character type (`NumSharp.Char8`) — the NumPy `S1`/Python `bytes(1)` equivalent — with conversions, operators, span helpers, and **100% Python `bytes` API parity** validated against a Python oracle. Vendored .NET ASCII/Latin-1 reference sources under `src/dotnet/` document the upstream implementations it mirrors.

## 13. Examples — trainable MNIST MLP

New `examples/NeuralNetwork.NumSharp`: a 2-layer MLP with a naive implementation and a **fused** one (single-`NpyIter` bias+ReLU fusion, fused softmax-cross-entropy backward, Adam optimizer). Originally needed a "copy transposed views before `np.dot`" workaround (31× training speedup at the time); the stride-native GEMM (§6) made the workaround unnecessary. Converges to >99% test accuracy in the bundled demo.

## 14. Kernel architecture & hygiene

- `ILKernelGenerator` split into **`DirectILKernelGenerator`** (legacy whole-array kernels, 51 partials under `Kernels/Direct/`) and **`ILKernelGenerator`** (NpyIter-driven per-chunk kernels — the target model matching NumPy's `PyUFuncGenericFunction`); migration path documented per kernel family.
- All `Vector128/256/512` and `Math`/`MathF` reflection centralized in `VectorMethodCache` / `ScalarMethodCache`; IL-emitted typed-field copier replaces the `UnmanagedStorage.Alias` switch.
- 24 dead kernel methods poisoned with `[Obsolete(error: true)]` pending deletion; dead axis-reduction SIMD paths removed.

## 15. Documentation

- **NpyIter/NDIter book**: `docs/website-src/docs/NDIter.md` (7-technique quick reference, decision tree, memory model, gotchas) + `ndarray.md`.
- **DocFX website — Benchmarks vs NumPy**: `benchmarks.md` (head-to-head evidence companion to the IL-generation page), `benchmark-iterator.md`, `benchmark-matrix.md`, driven by the auto-committed report artifacts.
- **Engineering ledgers**: `PERF_LEDGER.md` (every optimization with before/after), `NPYITER_GAPS_AND_ROADMAP.md` (gap analysis vs NumPy 2.4.2 + prioritized roadmap), `MIGRATE_NPYITER.md`, IL-kernel playbook, fuzz findings/coverage.
- **Branch quality audit** findings are pinned as `test/NumSharp.UnitTest/AuditV2/AuditV2_*.cs` — every Tier-1 finding fixed or reproduced as an `[OpenBugs]` test.

## 16. Tests & CI

- **+2,600 test methods**; suite now **9,990 passed / 0 failed** on net8.0 + net10.0. Zero regressions maintained commit-by-commit.
- New suites: `np.evaluate` (per-node wraparound, dtype matrices, weak scalars + overflow, fused-vs-unfused, `out=` identity/cast/aliasing, fused reductions), `out=`/`where=`/`dtype=` parity suites (broadcast/cast/error-text pins), WRITEMASKED/VIRTUAL parity; NpyIter battletests (566 scenarios), order-support sections 41–51, ARC lifecycle, clone regression, np.pad/average/median/percentile/ptp/diff battle tests, IL-kernel battle tests, behavioral audit harness.
- CI: fuzz gate in `build-and-release.yml`, nightly `fuzz-soak.yml`, **new post-release `benchmark.yml`** (auto-commits NumPy-comparison report cards to master).
- **Known gaps stay visible**: the still-unimplemented NumPy functions are `flip`/`fliplr`/`flipud`/`rot90`, `diag`, `gradient`, and `round` (`np.sort` is now done); small-N (~1K) per-call dispatch overhead is the headline performance focus (`docs/NPYITER_GAPS_AND_ROADMAP.md`); a few iterator edge cases remain pinned as `[OpenBugs]`/skipped repros. Every open issue found by the audits/fuzzers/benches is checked in as a failing-by-design test rather than ignored.

---

## Breaking changes

| Change | Impact | Migration |
|---|---|---|
| `bool - bool`, `-bool`, `np.negative(bool)` now throw | Matches NumPy | Use `^` / cast to int first |
| NaN `<=` / `>=` returns `False` | Matches IEEE & NumPy | Use `np.isnan` explicitly |
| `floor_divide`/`mod` divide-by-zero & floored results | Matches NumPy | — |
| `np.negative(uint)` wraps instead of throwing | Matches NumPy | — |
| `np.power(int, negative int)` raises `ValueError` | Matches NumPy | Use float exponents |
| Cast edge cases (overflow/NaN/complex→bool/float→int truncation) | Matches NumPy | — |
| Transcendental ufuncs: NEP50 width-based promotion | Return dtype may change | — |
| `np.clip`/quantile-family dtype promotion | Return dtype may change | — |
| Broadcast views are read-only; broadcasting keeps rank for 1-D `[1]` | Matches NumPy | `.copy()` to write |
| `MultiIterator` **and** `NDIterator` (+ `NDIterator<T>`, `AsIterator`) removed | Public types removed (threw at runtime anyway) | Use `NpyIter` / `NpyIter.Copy` / `NpyFlatIterator` |
| NpyIter: `MaxOperands=8` and 64-dim limits removed | None (loosening) | — |
| `np.copyto` unwriteable-destination error type corrected | Exception type change | — |

---

*Everything above was validated against NumPy 2.4.2 ground truth — by 37k differential corpus cases, 566 iterator parity scenarios, and per-feature battle tests run on actual NumPy output.*
